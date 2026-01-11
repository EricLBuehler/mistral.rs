//! Prefix caching for PagedAttention.
//!
//! This module implements automatic prefix caching inspired by vLLM's approach.
//! The key idea is to hash KV cache blocks by their token content and reuse
//! blocks across requests that share common prefixes (e.g., system prompts).
//!
//! Each block is identified by a hash of:
//! - The hash of all previous blocks (parent hash)
//! - The tokens contained in the current block
//!
//! This creates a chain of hashes that uniquely identifies each block's position
//! and content in a sequence.

use std::{
    collections::{HashMap, VecDeque},
    hash::{DefaultHasher, Hash, Hasher},
    time::Instant,
};

use super::{BlockRef, LogicalTokenBlock};

/// A hash that uniquely identifies a KV cache block by its content and position.
/// The hash incorporates:
/// - The parent block's hash (or 0 for the first block)
/// - The tokens in this block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(u64);

impl BlockHash {
    /// Compute the hash for a block given its parent hash and tokens.
    pub fn new(parent_hash: Option<BlockHash>, tokens: &[usize]) -> Self {
        let mut hasher = DefaultHasher::new();
        // Include parent hash in the chain
        if let Some(parent) = parent_hash {
            parent.0.hash(&mut hasher);
        } else {
            0u64.hash(&mut hasher);
        }
        // Hash the tokens
        tokens.hash(&mut hasher);
        BlockHash(hasher.finish())
    }

    /// Get the raw hash value.
    #[allow(dead_code)]
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// Metadata for a cached block in the prefix cache.
struct CachedBlockEntry {
    /// The block reference - holds the block alive via Arc refcount.
    block_ref: BlockRef,
    /// Number of tokens in this block (may be less than block_size for partial blocks).
    num_tokens: usize,
    /// Last access time for LRU eviction.
    last_access: Instant,
    /// Number of active sequences using this cached block.
    /// When this reaches 0, the block becomes eligible for eviction.
    active_users: usize,
}

/// The prefix cache maintains a global hash table mapping block hashes to physical blocks.
/// This enables automatic KV cache reuse across requests with shared prefixes.
///
/// The cache holds BlockRef instances, which automatically manage refcounts:
/// - When a block is cached, the cache holds a reference (keeps block alive)
/// - When a block is evicted, dropping the BlockRef returns it to the pool
/// - No manual refcount management needed!
pub struct PrefixCacher {
    /// Map from block hash to cached block entry.
    cache: HashMap<BlockHash, CachedBlockEntry>,
    /// LRU queue of block hashes with active_users == 0, ordered by last access time.
    /// Front = least recently used (evict first).
    lru_queue: VecDeque<BlockHash>,
    /// Whether prefix caching is enabled.
    enabled: bool,
    /// Statistics
    hits: usize,
    misses: usize,
}

impl PrefixCacher {
    /// Create a new prefix cache.
    pub fn new(enabled: bool) -> Self {
        Self {
            cache: HashMap::new(),
            lru_queue: VecDeque::new(),
            enabled,
            hits: 0,
            misses: 0,
        }
    }

    /// Check if prefix caching is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set whether prefix caching is enabled.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Compute block hashes for a sequence of logical token blocks.
    /// Returns a vector of hashes, one per block.
    pub fn compute_block_hashes(&self, logical_blocks: &[LogicalTokenBlock]) -> Vec<BlockHash> {
        let mut hashes = Vec::with_capacity(logical_blocks.len());
        let mut parent_hash = None;

        for block in logical_blocks {
            // Only hash full blocks for caching (partial blocks are still being filled)
            let tokens = if block.is_full() {
                block.toks()
            } else {
                // For partial blocks, use what we have
                &block.toks()[..block.num_tokens()]
            };
            let hash = BlockHash::new(parent_hash, tokens);
            hashes.push(hash);
            parent_hash = Some(hash);
        }

        hashes
    }

    /// Try to find cached blocks for a sequence of logical token blocks.
    /// Returns:
    /// - Vector of (block_index, BlockRef) for cache hits (cloned refs)
    /// - The number of blocks that were cache hits (starting from index 0)
    ///
    /// Only matches contiguous blocks starting from the beginning.
    /// The returned BlockRefs are clones - they share ownership with the cache.
    pub fn match_prefix(
        &mut self,
        logical_blocks: &[LogicalTokenBlock],
    ) -> (Vec<(usize, BlockRef)>, usize) {
        if !self.enabled || logical_blocks.is_empty() {
            return (Vec::new(), 0);
        }

        let hashes = self.compute_block_hashes(logical_blocks);
        let mut matched_blocks = Vec::new();
        let mut num_matched = 0;
        let now = Instant::now();

        for (idx, (hash, logical_block)) in hashes.iter().zip(logical_blocks.iter()).enumerate() {
            // Only cache full blocks
            if !logical_block.is_full() {
                break;
            }

            if let Some(entry) = self.cache.get_mut(hash) {
                // Cache hit! Verify token count matches.
                if entry.num_tokens == logical_block.num_tokens() {
                    // Update access time and active users count
                    entry.last_access = now;
                    entry.active_users += 1;

                    // Remove from LRU queue if it was there (active_users was 0)
                    if entry.active_users == 1 {
                        self.lru_queue.retain(|h| h != hash);
                    }

                    // Clone the BlockRef - this automatically increments the Arc refcount
                    matched_blocks.push((idx, entry.block_ref.clone()));
                    num_matched = idx + 1;
                    self.hits += 1;
                } else {
                    // Token count mismatch - hash collision, stop matching
                    self.misses += 1;
                    break;
                }
            } else {
                // Cache miss
                self.misses += 1;
                break;
            }
        }

        (matched_blocks, num_matched)
    }

    /// Insert blocks into the prefix cache.
    /// This is called after a sequence completes prefill to cache its blocks.
    ///
    /// `logical_blocks`: The logical token blocks
    /// `block_refs`: The corresponding block references (must be same length)
    /// `start_idx`: Index to start caching from (skip already-cached blocks)
    pub fn insert_blocks(
        &mut self,
        logical_blocks: &[LogicalTokenBlock],
        block_refs: &[BlockRef],
        start_idx: usize,
    ) {
        if !self.enabled {
            return;
        }

        assert_eq!(
            logical_blocks.len(),
            block_refs.len(),
            "logical and physical block counts must match"
        );

        let hashes = self.compute_block_hashes(logical_blocks);
        let now = Instant::now();

        for idx in start_idx..logical_blocks.len() {
            let logical_block = &logical_blocks[idx];

            // Only cache full blocks
            if !logical_block.is_full() {
                continue;
            }

            let hash = hashes[idx];

            // Don't overwrite existing entries
            if self.cache.contains_key(&hash) {
                continue;
            }

            // Clone the BlockRef - this keeps the block alive in the cache
            let block_ref = block_refs[idx].clone();

            self.cache.insert(
                hash,
                CachedBlockEntry {
                    block_ref,
                    num_tokens: logical_block.num_tokens(),
                    last_access: now,
                    active_users: 0, // Not actively used by any sequence
                },
            );

            // Add to LRU queue since active_users is 0
            self.lru_queue.push_back(hash);
        }
    }

    /// Decrement active users count for cached blocks when a sequence finishes or is preempted.
    /// Blocks with active_users == 0 become eligible for eviction.
    pub fn release_blocks(&mut self, logical_blocks: &[LogicalTokenBlock]) {
        if !self.enabled || logical_blocks.is_empty() {
            return;
        }

        let hashes = self.compute_block_hashes(logical_blocks);

        for (hash, logical_block) in hashes.iter().zip(logical_blocks.iter()) {
            if !logical_block.is_full() {
                continue;
            }

            if let Some(entry) = self.cache.get_mut(hash) {
                if entry.active_users > 0 {
                    entry.active_users -= 1;
                    if entry.active_users == 0 {
                        // Add to LRU queue
                        self.lru_queue.push_back(*hash);
                    }
                }
            }
        }
    }

    /// Evict blocks from the cache to free up memory.
    /// Dropping the BlockRefs will automatically return blocks to the pool.
    ///
    /// Uses LRU eviction policy:
    /// 1. Only evict blocks with active_users == 0
    /// 2. Evict least recently used blocks first
    pub fn evict_blocks(&mut self, num_blocks_needed: usize) {
        if !self.enabled {
            return;
        }

        let mut evicted_count = 0;

        while evicted_count < num_blocks_needed && !self.lru_queue.is_empty() {
            if let Some(hash) = self.lru_queue.pop_front() {
                if let Some(entry) = self.cache.remove(&hash) {
                    // Only evict if active_users is still 0
                    if entry.active_users == 0 {
                        // Dropping entry.block_ref here automatically returns block to pool
                        evicted_count += 1;
                    } else {
                        // Put it back, it's being used again
                        self.cache.insert(hash, entry);
                    }
                }
            }
        }
    }

    /// Get the number of blocks currently in the cache.
    pub fn num_cached_blocks(&self) -> usize {
        self.cache.len()
    }

    /// Get the number of evictable blocks (active_users == 0).
    pub fn num_evictable_blocks(&self) -> usize {
        self.cache
            .values()
            .filter(|entry| entry.active_users == 0)
            .count()
    }

    /// Get cache statistics (hits, misses).
    pub fn stats(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }

    /// Get hit rate as a percentage.
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Clear the entire cache.
    /// Dropping all BlockRefs will return their blocks to the pool.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.lru_queue.clear();
        self.cache.clear();
        // All BlockRefs are dropped here, returning blocks to pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_hash_consistency() {
        let tokens1 = vec![1, 2, 3, 4];
        let tokens2 = vec![1, 2, 3, 4];
        let tokens3 = vec![1, 2, 3, 5];

        let hash1 = BlockHash::new(None, &tokens1);
        let hash2 = BlockHash::new(None, &tokens2);
        let hash3 = BlockHash::new(None, &tokens3);

        // Same tokens should produce same hash
        assert_eq!(hash1, hash2);
        // Different tokens should produce different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_block_hash_chain() {
        let tokens1 = vec![1, 2, 3, 4];
        let tokens2 = vec![5, 6, 7, 8];

        let hash1 = BlockHash::new(None, &tokens1);
        let hash2_with_parent = BlockHash::new(Some(hash1), &tokens2);
        let hash2_without_parent = BlockHash::new(None, &tokens2);

        // Same tokens but different parent should produce different hash
        assert_ne!(hash2_with_parent, hash2_without_parent);
    }
}
