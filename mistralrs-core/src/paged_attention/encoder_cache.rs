//! Encoder output cache for multimodal models.
//!
//! Caches vision/audio encoder outputs so that when a prefix cache hit occurs,
//! the encoder doesn't need to re-process media that was already encoded.
//!
//! This is a placeholder for the full encoder cache. Per-model integration
//! (checking the cache before running the encoder, storing outputs after) will
//! be added incrementally for each vision/audio model.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use candle_core::Tensor;

/// A cached encoder output for a single multimodal input.
struct CachedEncoderOutput {
    /// The encoder output tensor (e.g., vision features after projection).
    output: Tensor,
    /// Set of sequence IDs currently referencing this cached output.
    active_users: HashSet<usize>,
}

/// Manages cached encoder outputs for multimodal models.
///
/// When a sequence has a prefix cache hit, its images/audio may already have
/// been encoded by a previous request. The `EncoderCacheManager` stores these
/// encoder outputs keyed by the content hash of the media data.
///
/// Usage pattern (to be wired into each vision model):
/// 1. Before running the vision encoder, check `get(content_hash)`.
/// 2. On cache hit: use the cached tensor, call `add_user(hash, seq_id)`.
/// 3. On cache miss: run the encoder, call `insert(hash, tensor, seq_id)`.
/// 4. When a sequence completes, call `remove_user(hash, seq_id)`.
/// 5. Entries with no active users are candidates for LRU eviction.
pub struct EncoderCacheManager {
    /// Map from content hash to cached encoder output.
    cache: HashMap<String, CachedEncoderOutput>,
    /// Maximum number of cached entries.
    max_entries: usize,
}

impl EncoderCacheManager {
    /// Create a new encoder cache with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_entries,
        }
    }

    /// Look up a cached encoder output by content hash.
    pub fn get(&self, content_hash: &str) -> Option<&Tensor> {
        self.cache.get(content_hash).map(|entry| &entry.output)
    }

    /// Insert a new encoder output into the cache.
    pub fn insert(&mut self, content_hash: String, output: Tensor, seq_id: usize) {
        if self.cache.len() >= self.max_entries {
            self.evict_lru();
        }
        let entry = self
            .cache
            .entry(content_hash)
            .or_insert_with(|| CachedEncoderOutput {
                output,
                active_users: HashSet::new(),
            });
        entry.active_users.insert(seq_id);
    }

    /// Add a sequence as a user of an existing cached entry.
    pub fn add_user(&mut self, content_hash: &str, seq_id: usize) {
        if let Some(entry) = self.cache.get_mut(content_hash) {
            entry.active_users.insert(seq_id);
        }
    }

    /// Remove a sequence from a cached entry's user set.
    pub fn remove_user(&mut self, content_hash: &str, seq_id: usize) {
        if let Some(entry) = self.cache.get_mut(content_hash) {
            entry.active_users.remove(&seq_id);
        }
    }

    /// Remove all references for a sequence across all cached entries.
    pub fn free_sequence(&mut self, seq_id: usize) {
        for entry in self.cache.values_mut() {
            entry.active_users.remove(&seq_id);
        }
    }

    /// Evict entries with no active users to make room for new entries.
    fn evict_lru(&mut self) {
        // Remove the first entry with no active users
        let key_to_remove = self
            .cache
            .iter()
            .find(|(_, entry)| entry.active_users.is_empty())
            .map(|(k, _)| k.clone());

        if let Some(key) = key_to_remove {
            self.cache.remove(&key);
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}
