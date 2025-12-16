//! ## Response caching functionality for the Responses API.

use anyhow::Result;
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::RwLock;

use crate::openai::{Message, ResponsesChunk, ResponsesObject};

/// Trait for caching responses
pub trait ResponseCache: Send + Sync {
    /// Store a response object with the given ID
    fn store_response(&self, id: String, response: ResponsesObject) -> Result<()>;

    /// Retrieve a response object by ID
    fn get_response(&self, id: &str) -> Result<Option<ResponsesObject>>;

    /// Delete a response object by ID
    fn delete_response(&self, id: &str) -> Result<bool>;

    /// Store streaming chunks for a response
    fn store_chunks(&self, id: String, chunks: Vec<ResponsesChunk>) -> Result<()>;

    /// Retrieve streaming chunks for a response
    fn get_chunks(&self, id: &str) -> Result<Option<Vec<ResponsesChunk>>>;

    /// Store conversation history for a response
    fn store_conversation_history(&self, id: String, messages: Vec<Message>) -> Result<()>;

    /// Retrieve conversation history for a response
    fn get_conversation_history(&self, id: &str) -> Result<Option<Vec<Message>>>;

    /// Store an active request ID mapping (API ID -> Engine ID)
    fn add_active_request(&self, api_id: String, engine_id: usize, model_id: Option<String>) -> Result<()>;
    /// Remove an active request ID mapping
    fn remove_active_request(&self, api_id: &str) -> Result<()>;
    /// Get the engine ID for an active request
    fn get_active_request_id(&self, api_id: &str) -> Result<Option<(usize, Option<String>)>>;

    /// Retrieve all response objects
    fn get_all_responses(&self) -> Result<Vec<ResponsesObject>>;
}

#[derive(Default, Clone)]
struct CacheEntry {
    response: Option<ResponsesObject>,
    chunks: Option<Vec<ResponsesChunk>>,
    history: Option<Vec<Message>>,
}

struct CacheInner {
    store: LruCache<String, CacheEntry>,
    active_requests: HashMap<String, (usize, Option<String>)>,
}

impl CacheInner {
    fn new(capacity: usize) -> Self {
        Self {
            store: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
            active_requests: HashMap::new(),
        }
    }
}

/// In-memory implementation of ResponseCache with LRU eviction
pub struct InMemoryResponseCache {
    inner: RwLock<CacheInner>,
}

impl InMemoryResponseCache {
    /// Create a new in-memory cache with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: RwLock::new(CacheInner::new(capacity)),
        }
    }
}

impl Default for InMemoryResponseCache {
    fn default() -> Self {
        Self::new(1000) // Default capacity
    }
}

impl ResponseCache for InMemoryResponseCache {
    fn store_response(&self, id: String, response: ResponsesObject) -> Result<()> {
        let mut inner = self.inner.write().unwrap();
        // Get or insert entry
        let entry = inner.store.get_or_insert_mut(id, CacheEntry::default);
        entry.response = Some(response);
        Ok(())
    }

    fn get_response(&self, id: &str) -> Result<Option<ResponsesObject>> {
        let mut inner = self.inner.write().unwrap();
        // Use get() to update LRU position
        Ok(inner.store.get(id).and_then(|e| e.response.clone()))
    }

    fn delete_response(&self, id: &str) -> Result<bool> {
        let mut inner = self.inner.write().unwrap();
        let removed_store = inner.store.pop(id).is_some();
        let removed_active = inner.active_requests.remove(id).is_some();
        Ok(removed_store || removed_active)
    }

    fn store_chunks(&self, id: String, chunks: Vec<ResponsesChunk>) -> Result<()> {
        let mut inner = self.inner.write().unwrap();
        let entry = inner.store.get_or_insert_mut(id, CacheEntry::default);
        entry.chunks = Some(chunks);
        Ok(())
    }

    fn get_chunks(&self, id: &str) -> Result<Option<Vec<ResponsesChunk>>> {
        let mut inner = self.inner.write().unwrap();
        Ok(inner.store.get(id).and_then(|e| e.chunks.clone()))
    }

    fn store_conversation_history(&self, id: String, messages: Vec<Message>) -> Result<()> {
        let mut inner = self.inner.write().unwrap();
        let entry = inner.store.get_or_insert_mut(id, CacheEntry::default);
        entry.history = Some(messages);
        Ok(())
    }

    fn get_conversation_history(&self, id: &str) -> Result<Option<Vec<Message>>> {
        let mut inner = self.inner.write().unwrap();
        Ok(inner.store.get(id).and_then(|e| e.history.clone()))
    }

    fn add_active_request(
        &self,
        api_id: String,
        engine_id: usize,
        model_id: Option<String>,
    ) -> Result<()> {
        let mut inner = self.inner.write().unwrap();
        inner.active_requests.insert(api_id, (engine_id, model_id));
        Ok(())
    }

    fn remove_active_request(&self, api_id: &str) -> Result<()> {
        let mut inner = self.inner.write().unwrap();
        inner.active_requests.remove(api_id);
        Ok(())
    }

    fn get_active_request_id(&self, api_id: &str) -> Result<Option<(usize, Option<String>)>> {
        let inner = self.inner.read().unwrap();
        Ok(inner.active_requests.get(api_id).cloned())
    }

    fn get_all_responses(&self) -> Result<Vec<ResponsesObject>> {
        let inner = self.inner.read().unwrap();
        // This does NOT update LRU position as we are just iterating
        Ok(inner
            .store
            .iter()
            .filter_map(|(_, entry)| entry.response.clone())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_cache_lru() {
        let cache = InMemoryResponseCache::new(2); // Small capacity for testing

        let response1 = ResponsesObject {
            id: "1".to_string(),
            object: "response",
            created_at: 1.0,
            model: "m".to_string(),
            status: "c".to_string(),
            output: vec![],
            output_text: None,
            usage: None,
            error: None,
            metadata: None,
            instructions: None,
            incomplete_details: None,
        };
        let mut response2 = response1.clone();
        response2.id = "2".to_string();
        let mut response3 = response1.clone();
        response3.id = "3".to_string();

        cache.store_response("1".to_string(), response1).unwrap();
        cache.store_response("2".to_string(), response2).unwrap();

        // Access 1 to make it most recently used
        assert!(cache.get_response("1").unwrap().is_some());

        // Add 3, which should evict 2 (LRU)
        cache.store_response("3".to_string(), response3).unwrap();

        assert!(cache.get_response("1").unwrap().is_some());
        assert!(cache.get_response("3").unwrap().is_some());
        assert!(cache.get_response("2").unwrap().is_none());
    }
}