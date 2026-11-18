//! ## Response caching functionality for the Responses API.

use anyhow::Result;
use indexmap::IndexMap;
use std::sync::LazyLock;
use std::sync::{Arc, RwLock};

use crate::openai::Message;
use crate::responses_types::ResponseResource;

/// Trait for caching responses
pub trait ResponseCache: Send + Sync {
    /// Store a response object with the given ID
    fn store_response(&self, id: String, response: ResponseResource) -> Result<()>;

    /// Retrieve a response object by ID
    fn get_response(&self, id: &str) -> Result<Option<ResponseResource>>;

    /// Delete a response object by ID
    fn delete_response(&self, id: &str) -> Result<bool>;

    /// Store conversation history for a response
    fn store_conversation_history(&self, id: String, messages: Vec<Message>) -> Result<()>;

    /// Retrieve conversation history for a response
    fn get_conversation_history(&self, id: &str) -> Result<Option<Vec<Message>>>;
}

/// `store` defaults to true, so without a cap every request on a long-running server is kept forever.
const MAX_CACHED_RESPONSES: usize = 4096;

/// In-memory implementation of ResponseCache that evicts the oldest entries past its capacity
pub struct InMemoryResponseCache {
    responses: Arc<RwLock<IndexMap<String, ResponseResource>>>,
    conversation_histories: Arc<RwLock<IndexMap<String, Vec<Message>>>>,
    capacity: usize,
}

impl InMemoryResponseCache {
    /// Create a new in-memory cache
    pub fn new() -> Self {
        Self::with_capacity(MAX_CACHED_RESPONSES)
    }

    /// Create a new in-memory cache holding at most `capacity` responses and histories each
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            responses: Arc::new(RwLock::new(IndexMap::new())),
            conversation_histories: Arc::new(RwLock::new(IndexMap::new())),
            capacity,
        }
    }
}

// Re-inserting an id moves it to the back, so an entry's age is measured from its last store.
fn insert_bounded<V>(map: &mut IndexMap<String, V>, id: String, value: V, capacity: usize) {
    map.shift_remove(&id);
    map.insert(id, value);
    while map.len() > capacity {
        map.shift_remove_index(0);
    }
}

impl Default for InMemoryResponseCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseCache for InMemoryResponseCache {
    fn store_response(&self, id: String, response: ResponseResource) -> Result<()> {
        let mut responses = self.responses.write().unwrap();
        insert_bounded(&mut responses, id, response, self.capacity);
        Ok(())
    }

    fn get_response(&self, id: &str) -> Result<Option<ResponseResource>> {
        let responses = self.responses.read().unwrap();
        Ok(responses.get(id).cloned())
    }

    fn delete_response(&self, id: &str) -> Result<bool> {
        // IMPORTANT: Lock ordering must be maintained to prevent deadlocks.
        // Order: responses -> conversation_histories
        // All methods that acquire multiple locks must follow this order.
        //
        // We acquire all locks before any modifications to ensure atomicity.
        // The locks are released in reverse order when dropped at end of scope.
        let mut responses = self.responses.write().unwrap();
        let mut histories = self.conversation_histories.write().unwrap();

        let response_removed = responses.shift_remove(id).is_some();
        let history_removed = histories.shift_remove(id).is_some();

        Ok(response_removed || history_removed)
    }

    fn store_conversation_history(&self, id: String, messages: Vec<Message>) -> Result<()> {
        let mut histories = self.conversation_histories.write().unwrap();
        insert_bounded(&mut histories, id, messages, self.capacity);
        Ok(())
    }

    fn get_conversation_history(&self, id: &str) -> Result<Option<Vec<Message>>> {
        let histories = self.conversation_histories.read().unwrap();
        Ok(histories.get(id).cloned())
    }
}

/// Global response cache instance
pub static RESPONSE_CACHE: LazyLock<Arc<dyn ResponseCache>> =
    LazyLock::new(|| Arc::new(InMemoryResponseCache::new()));

/// Helper function to get the global cache instance
pub fn get_response_cache() -> Arc<dyn ResponseCache> {
    RESPONSE_CACHE.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses_types::{ItemStatus, OutputContent, OutputItem, ResponseStatus};

    #[test]
    fn test_in_memory_cache() {
        let cache = InMemoryResponseCache::new();

        // Create a test response
        let response =
            ResponseResource::new("test-id".to_string(), "test-model".to_string(), 1234567890)
                .with_status(ResponseStatus::Completed)
                .with_output(vec![OutputItem::message(
                    "msg-1".to_string(),
                    vec![OutputContent::text("Hello".to_string())],
                    ItemStatus::Completed,
                )]);

        // Store and retrieve
        cache
            .store_response("test-id".to_string(), response.clone())
            .unwrap();
        let retrieved = cache.get_response("test-id").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test-id");

        // Delete
        let deleted = cache.delete_response("test-id").unwrap();
        assert!(deleted);
        let retrieved = cache.get_response("test-id").unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_conversation_history() {
        let cache = InMemoryResponseCache::new();

        let messages = vec![Message {
            audio: None,
            content: Some(crate::openai::MessageContent::from_text(
                "Hello".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        cache
            .store_conversation_history("test-id".to_string(), messages.clone())
            .unwrap();

        let retrieved = cache.get_conversation_history("test-id").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), 1);
    }

    fn response(id: &str) -> ResponseResource {
        ResponseResource::new(id.to_string(), "test-model".to_string(), 0)
    }

    fn store_all(cache: &InMemoryResponseCache, ids: &[&str]) {
        for id in ids {
            cache.store_response(id.to_string(), response(id)).unwrap();
        }
    }

    fn history(text: &str) -> Vec<Message> {
        vec![Message {
            audio: None,
            content: Some(crate::openai::MessageContent::from_text(text.to_string())),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }]
    }

    #[test]
    fn oldest_entries_are_evicted_past_capacity() {
        let cache = InMemoryResponseCache::with_capacity(2);
        for id in ["a", "b", "c"] {
            cache.store_response(id.to_string(), response(id)).unwrap();
            cache
                .store_conversation_history(id.to_string(), history(id))
                .unwrap();
        }

        assert!(cache.get_response("a").unwrap().is_none());
        assert!(cache.get_conversation_history("a").unwrap().is_none());
        for id in ["b", "c"] {
            assert!(cache.get_response(id).unwrap().is_some());
            assert!(cache.get_conversation_history(id).unwrap().is_some());
        }
    }

    #[test]
    fn restoring_an_entry_protects_it_from_eviction() {
        let cache = InMemoryResponseCache::with_capacity(2);
        store_all(&cache, &["a", "b", "a", "c"]);

        assert!(cache.get_response("a").unwrap().is_some());
        assert!(cache.get_response("b").unwrap().is_none());
        assert!(cache.get_response("c").unwrap().is_some());
    }

    #[test]
    fn deleting_frees_a_slot_without_disturbing_the_others() {
        let cache = InMemoryResponseCache::with_capacity(2);
        store_all(&cache, &["a", "b"]);
        assert!(cache.delete_response("a").unwrap());
        store_all(&cache, &["c"]);

        assert!(cache.get_response("b").unwrap().is_some());
        assert!(cache.get_response("c").unwrap().is_some());
    }
}
