//! ## Response caching functionality for the Responses API.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::{Arc, RwLock};

use crate::openai::{ResponsesChunk, ResponsesObject};

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
}

/// In-memory implementation of ResponseCache
pub struct InMemoryResponseCache {
    responses: Arc<RwLock<HashMap<String, ResponsesObject>>>,
    chunks: Arc<RwLock<HashMap<String, Vec<ResponsesChunk>>>>,
}

impl InMemoryResponseCache {
    /// Create a new in-memory cache
    pub fn new() -> Self {
        Self {
            responses: Arc::new(RwLock::new(HashMap::new())),
            chunks: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryResponseCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseCache for InMemoryResponseCache {
    fn store_response(&self, id: String, response: ResponsesObject) -> Result<()> {
        let mut responses = self.responses.write().unwrap();
        responses.insert(id, response);
        Ok(())
    }

    fn get_response(&self, id: &str) -> Result<Option<ResponsesObject>> {
        let responses = self.responses.read().unwrap();
        Ok(responses.get(id).cloned())
    }

    fn delete_response(&self, id: &str) -> Result<bool> {
        let mut responses = self.responses.write().unwrap();
        let mut chunks = self.chunks.write().unwrap();

        let response_removed = responses.remove(id).is_some();
        let chunks_removed = chunks.remove(id).is_some();

        Ok(response_removed || chunks_removed)
    }

    fn store_chunks(&self, id: String, chunks: Vec<ResponsesChunk>) -> Result<()> {
        let mut chunk_storage = self.chunks.write().unwrap();
        chunk_storage.insert(id, chunks);
        Ok(())
    }

    fn get_chunks(&self, id: &str) -> Result<Option<Vec<ResponsesChunk>>> {
        let chunks = self.chunks.read().unwrap();
        Ok(chunks.get(id).cloned())
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

    #[test]
    fn test_in_memory_cache() {
        let cache = InMemoryResponseCache::new();

        // Create a test response
        let response = ResponsesObject {
            id: "test-id".to_string(),
            object: "response",
            created_at: 1234567890.0,
            model: "test-model".to_string(),
            status: "completed".to_string(),
            output: vec![],
            output_text: None,
            usage: None,
            error: None,
            metadata: None,
            instructions: None,
            incomplete_details: None,
        };

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
}
