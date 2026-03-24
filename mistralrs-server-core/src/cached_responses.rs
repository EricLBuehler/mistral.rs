//! ## Response caching functionality for the Responses API.

use anyhow::Result;
use std::collections::HashMap;
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

/// In-memory implementation of ResponseCache
pub struct InMemoryResponseCache {
    responses: Arc<RwLock<HashMap<String, ResponseResource>>>,
    conversation_histories: Arc<RwLock<HashMap<String, Vec<Message>>>>,
}

impl InMemoryResponseCache {
    /// Create a new in-memory cache
    pub fn new() -> Self {
        Self {
            responses: Arc::new(RwLock::new(HashMap::new())),
            conversation_histories: Arc::new(RwLock::new(HashMap::new())),
        }
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
        responses.insert(id, response);
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

        let response_removed = responses.remove(id).is_some();
        let history_removed = histories.remove(id).is_some();

        Ok(response_removed || history_removed)
    }

    fn store_conversation_history(&self, id: String, messages: Vec<Message>) -> Result<()> {
        let mut histories = self.conversation_histories.write().unwrap();
        histories.insert(id, messages);
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
            content: Some(crate::openai::MessageContent::from_text(
                "Hello".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        cache
            .store_conversation_history("test-id".to_string(), messages.clone())
            .unwrap();

        let retrieved = cache.get_conversation_history("test-id").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), 1);
    }
}
