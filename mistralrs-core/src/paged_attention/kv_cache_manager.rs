//! KV Cache Manager for paged attention.
//!
//! This is a port of vLLM's v1 `KVCacheManager` + `FullAttentionManager`.
//! It manages block allocation, prefix cache lookups, and per-request block tracking.
//!
//! The manager owns a `BlockPool` and provides high-level operations:
//! - `get_computed_blocks`: Find the longest prefix cache hit for a request.
//! - `allocate_slots`: Allocate blocks for new tokens.
//! - `free`: Free blocks when a request completes or is preempted.
//! - `cache_blocks`: Cache newly-full blocks after computation.

use std::collections::HashMap;

use super::block_hash::BlockHash;
use super::block_pool::BlockPool;

/// Result of `get_computed_blocks`: cached block IDs and how many tokens they cover.
#[derive(Debug)]
pub struct ComputedBlocks {
    /// Block IDs from the prefix cache, one per block position.
    /// For models with sliding window layers, early blocks may be `null_block_id`
    /// (placeholder for blocks outside the attention window).
    pub block_ids: Vec<usize>,
    /// Number of tokens covered by the cached blocks.
    /// Always a multiple of `block_size`.
    pub num_computed_tokens: usize,
}

/// Per-request block allocation state.
struct RequestBlocks {
    /// Block IDs allocated for this request, in sequence order.
    block_ids: Vec<usize>,
    /// Number of blocks that are already cached (skip during `cache_blocks`).
    num_cached_blocks: usize,
}

/// KV Cache Manager — manages block allocation and prefix caching.
///
/// Each instance handles one "type" of KV cache layer (e.g., full attention).
/// For models with alternating sliding window layers (Gemma2, GPT-OSS),
/// separate instances manage the full-attention and sliding-window block tables,
/// sharing the same underlying `BlockPool`.
pub struct KVCacheManager {
    block_pool: BlockPool,
    block_size: usize,
    enable_caching: bool,
    /// KV cache group IDs used for prefix cache lookups.
    /// Most models have a single group `[0]`. Models with multiple attention
    /// types (e.g., full + sliding window) use different group IDs per manager.
    kv_cache_group_ids: Vec<u32>,
    /// Per-request block tracking.
    req_to_blocks: HashMap<usize, RequestBlocks>,
}

impl KVCacheManager {
    /// Create a new KV cache manager.
    ///
    /// - `num_gpu_blocks`: Total number of physical GPU blocks.
    /// - `block_size`: Tokens per block.
    /// - `enable_caching`: Whether prefix caching is enabled.
    /// - `kv_cache_group_ids`: Group IDs for prefix cache lookups.
    pub fn new(
        num_gpu_blocks: usize,
        block_size: usize,
        enable_caching: bool,
        kv_cache_group_ids: Vec<u32>,
    ) -> Self {
        Self {
            block_pool: BlockPool::new(num_gpu_blocks, enable_caching, block_size),
            block_size,
            enable_caching,
            kv_cache_group_ids,
            req_to_blocks: HashMap::new(),
        }
    }

    /// Get a reference to the block pool.
    pub fn block_pool(&self) -> &BlockPool {
        &self.block_pool
    }

    /// Get a mutable reference to the block pool.
    pub fn block_pool_mut(&mut self) -> &mut BlockPool {
        &mut self.block_pool
    }

    /// Get the null block ID (placeholder for skipped/unused slots).
    pub fn null_block_id(&self) -> usize {
        self.block_pool.null_block_id()
    }

    /// Get the block size (tokens per block).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get KV cache usage as a fraction [0.0, 1.0].
    pub fn usage(&self) -> f64 {
        self.block_pool.usage()
    }

    /// Get the number of free blocks available.
    pub fn num_free_blocks(&self) -> usize {
        self.block_pool.num_free_blocks()
    }

    /// Whether prefix caching is enabled.
    pub fn caching_enabled(&self) -> bool {
        self.enable_caching
    }

    /// Find the longest prefix cache hit for a request.
    ///
    /// Walks the request's block hashes and checks for cached blocks.
    /// Returns cached block IDs and the number of computed tokens.
    ///
    /// **Important**: When all tokens hit the cache, we must recompute the
    /// last block to produce logits. So `max_length` should be
    /// `num_tokens - 1` (the caller is responsible for this).
    pub fn get_computed_blocks(
        &self,
        block_hashes: &[BlockHash],
        num_tokens: usize,
    ) -> ComputedBlocks {
        if !self.enable_caching || block_hashes.is_empty() {
            return ComputedBlocks {
                block_ids: Vec::new(),
                num_computed_tokens: 0,
            };
        }

        // Max cache hit: at most num_tokens - 1 (need to recompute last token for logits)
        let max_cache_hit_length = num_tokens.saturating_sub(1);
        let max_num_blocks = max_cache_hit_length / self.block_size;

        let mut cached_block_ids = Vec::new();

        for (i, &block_hash) in block_hashes.iter().enumerate() {
            if i >= max_num_blocks {
                break;
            }

            // Look up this block hash across all group IDs
            if let Some(ids) = self
                .block_pool
                .get_cached_block(block_hash, &self.kv_cache_group_ids)
            {
                // For simplicity, take the first group's block.
                // Multi-group support would need to return all group block IDs
                // to construct separate block tables per group.
                debug_assert_eq!(
                    ids.len(),
                    1,
                    "Multi-group prefix cache lookup not yet implemented: found {} groups",
                    ids.len()
                );
                cached_block_ids.push(ids[0]);
            } else {
                // Chain is broken — no further blocks can match
                break;
            }
        }

        let num_computed_tokens = cached_block_ids.len() * self.block_size;

        ComputedBlocks {
            block_ids: cached_block_ids,
            num_computed_tokens,
        }
    }

    /// Allocate blocks for a request.
    ///
    /// This handles both new requests (with optional prefix cache hits) and
    /// running requests that need additional blocks.
    ///
    /// - `request_id`: The sequence ID.
    /// - `num_tokens`: Total number of tokens that need a slot (including
    ///   cached and new tokens).
    /// - `computed_blocks`: Block IDs from prefix cache (from `get_computed_blocks`).
    ///   Empty if no cache hit or if the request is already running.
    ///
    /// Returns `Some(new_block_ids)` on success, `None` if not enough free blocks.
    pub fn allocate_slots(
        &mut self,
        request_id: usize,
        num_tokens: usize,
        computed_blocks: &[usize],
    ) -> Option<Vec<usize>> {
        let num_required_blocks = num_tokens.div_ceil(self.block_size);

        if let Some(req) = self.req_to_blocks.get(&request_id) {
            // Running request — just need to allocate additional blocks
            let num_existing = req.block_ids.len();
            let num_new_blocks = num_required_blocks.saturating_sub(num_existing);

            if num_new_blocks == 0 {
                return Some(Vec::new());
            }

            let new_block_ids = self.block_pool.get_new_blocks(num_new_blocks)?;
            self.req_to_blocks
                .get_mut(&request_id)
                .unwrap()
                .block_ids
                .extend_from_slice(&new_block_ids);
            return Some(new_block_ids);
        }

        // New request — incorporate computed blocks + allocate new ones
        let num_computed = computed_blocks.len();
        let num_new_blocks = num_required_blocks.saturating_sub(num_computed);

        // Count evictable blocks among computed blocks (blocks with ref_cnt == 0
        // that are in the free list — touching them will remove them from the
        // free list, so we need to account for this in the capacity check).
        let num_evictable = if self.enable_caching {
            computed_blocks
                .iter()
                .filter(|&&id| self.block_pool.block_ref_cnt(id) == 0)
                .count()
        } else {
            0
        };

        let total_needed = num_new_blocks + num_evictable;
        if total_needed > self.block_pool.num_free_blocks() {
            return None;
        }

        // Touch the computed blocks (increment ref_cnt, remove from free list)
        if !computed_blocks.is_empty() && self.enable_caching {
            self.block_pool.touch(computed_blocks);
        }

        // Allocate new blocks
        let new_block_ids = if num_new_blocks > 0 {
            self.block_pool
                .get_new_blocks(num_new_blocks)
                .expect("Should have enough blocks after capacity check")
        } else {
            Vec::new()
        };

        // Build the full block list: computed + new
        let mut all_block_ids = Vec::with_capacity(num_required_blocks);
        all_block_ids.extend_from_slice(computed_blocks);
        all_block_ids.extend_from_slice(&new_block_ids);

        self.req_to_blocks.insert(
            request_id,
            RequestBlocks {
                block_ids: all_block_ids,
                num_cached_blocks: num_computed,
            },
        );

        Some(new_block_ids)
    }

    /// Free all blocks for a request.
    ///
    /// Blocks are freed in reverse order so that tail blocks (most specific)
    /// are evicted first when the free list is used for LRU eviction.
    pub fn free(&mut self, request_id: usize) {
        if let Some(req) = self.req_to_blocks.remove(&request_id) {
            // Free in reverse order for LRU eviction priority
            let reversed: Vec<usize> = req.block_ids.into_iter().rev().collect();
            self.block_pool.free_blocks(&reversed);
        }
    }

    /// Cache newly-full blocks after tokens are computed.
    ///
    /// Called after each step (prefill or decode) to register full blocks
    /// in the prefix cache hash map so future requests can reuse them.
    ///
    /// - `request_id`: The sequence ID.
    /// - `block_hashes`: The block hashes for the request's token sequence.
    /// - `num_computed_tokens`: Total number of tokens computed so far
    ///   (including cached tokens from prefix hits).
    pub fn cache_blocks(
        &mut self,
        request_id: usize,
        block_hashes: &[BlockHash],
        num_computed_tokens: usize,
    ) {
        if !self.enable_caching {
            return;
        }

        let req = match self.req_to_blocks.get_mut(&request_id) {
            Some(r) => r,
            None => return,
        };

        let num_full_blocks = num_computed_tokens / self.block_size;
        if req.num_cached_blocks >= num_full_blocks {
            return;
        }

        // Cache each full block for each group ID
        for &group_id in &self.kv_cache_group_ids {
            self.block_pool.cache_full_blocks(
                &req.block_ids,
                block_hashes,
                req.num_cached_blocks,
                num_full_blocks,
                group_id,
            );
        }

        req.num_cached_blocks = num_full_blocks;
    }

    /// Get the block IDs allocated for a request.
    pub fn get_block_ids(&self, request_id: usize) -> Option<&[usize]> {
        self.req_to_blocks
            .get(&request_id)
            .map(|r| r.block_ids.as_slice())
    }

    /// Get the number of blocks allocated for a request.
    pub fn num_blocks_for_request(&self, request_id: usize) -> usize {
        self.req_to_blocks
            .get(&request_id)
            .map(|r| r.block_ids.len())
            .unwrap_or(0)
    }

    /// Check if a request has allocated blocks.
    pub fn has_request(&self, request_id: usize) -> bool {
        self.req_to_blocks.contains_key(&request_id)
    }

    /// Get the number of cached blocks for a request.
    pub fn num_cached_blocks(&self, request_id: usize) -> usize {
        self.req_to_blocks
            .get(&request_id)
            .map(|r| r.num_cached_blocks)
            .unwrap_or(0)
    }

    /// Reset the prefix cache. Only succeeds if all blocks are free.
    pub fn reset_prefix_cache(&mut self) -> bool {
        self.block_pool.reset_prefix_cache()
    }

    /// Get the slot mapping for a request's tokens.
    ///
    /// Maps each token position to its physical slot in the KV cache:
    /// `slot = block_id * block_size + offset_within_block`
    ///
    /// - `start_token`: First token position to map (e.g., skip cached tokens).
    /// - `num_tokens`: Number of tokens to map.
    ///
    /// Returns a vector of slot indices, or `None` if the request doesn't exist.
    pub fn get_slot_mapping(
        &self,
        request_id: usize,
        start_token: usize,
        num_tokens: usize,
    ) -> Option<Vec<i64>> {
        let req = self.req_to_blocks.get(&request_id)?;
        let mut slots = Vec::with_capacity(num_tokens);

        for token_pos in start_token..start_token + num_tokens {
            let block_idx = token_pos / self.block_size;
            let offset = token_pos % self.block_size;

            if block_idx < req.block_ids.len() {
                let block_id = req.block_ids[block_idx];
                slots.push((block_id * self.block_size + offset) as i64);
            } else {
                // Should not happen if blocks are correctly allocated
                slots.push(super::_PAD_SLOT_ID);
            }
        }

        Some(slots)
    }

    /// Build the block table for a request (for the paged attention kernel).
    ///
    /// Returns the block IDs in sequence order, padded to `max_blocks` with 0.
    pub fn get_block_table(&self, request_id: usize, max_blocks: usize) -> Option<Vec<i32>> {
        let req = self.req_to_blocks.get(&request_id)?;
        let mut table = Vec::with_capacity(max_blocks);

        #[allow(clippy::cast_possible_truncation)]
        for &block_id in &req.block_ids {
            table.push(block_id as i32);
        }

        // Pad with zeros
        table.resize(max_blocks, 0);
        Some(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::block_hash::compute_block_hashes;

    #[test]
    fn test_basic_allocation() {
        let mut mgr = KVCacheManager::new(16, 4, false, vec![0]);

        // Allocate for a request with 10 tokens (needs 3 blocks: ceil(10/4))
        let new_blocks = mgr.allocate_slots(1, 10, &[]).unwrap();
        assert_eq!(new_blocks.len(), 3);
        assert_eq!(mgr.num_blocks_for_request(1), 3);
    }

    #[test]
    fn test_running_request_extends() {
        let mut mgr = KVCacheManager::new(16, 4, false, vec![0]);

        // Initial allocation: 8 tokens = 2 blocks
        mgr.allocate_slots(1, 8, &[]).unwrap();
        assert_eq!(mgr.num_blocks_for_request(1), 2);

        // Request grows to 12 tokens = 3 blocks, needs 1 more
        let new_blocks = mgr.allocate_slots(1, 12, &[]).unwrap();
        assert_eq!(new_blocks.len(), 1);
        assert_eq!(mgr.num_blocks_for_request(1), 3);
    }

    #[test]
    fn test_allocation_fails_when_full() {
        let mut mgr = KVCacheManager::new(4, 4, false, vec![0]);
        // 4 blocks total, 1 null = 3 free

        mgr.allocate_slots(1, 12, &[]).unwrap(); // takes all 3
        assert!(mgr.allocate_slots(2, 4, &[]).is_none());
    }

    #[test]
    fn test_free_returns_blocks() {
        let mut mgr = KVCacheManager::new(8, 4, false, vec![0]);

        mgr.allocate_slots(1, 12, &[]).unwrap();
        assert_eq!(mgr.num_free_blocks(), 4); // 8-1null-3alloc = 4

        mgr.free(1);
        assert_eq!(mgr.num_free_blocks(), 7); // 8-1null = 7
        assert!(!mgr.has_request(1));
    }

    #[test]
    fn test_prefix_cache_hit() {
        let mut mgr = KVCacheManager::new(16, 4, true, vec![0]);

        // Request 1: tokens [1,2,3,4,5,6,7,8] = 2 blocks
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);

        mgr.allocate_slots(1, 8, &[]).unwrap();
        mgr.cache_blocks(1, &hashes, 8);

        // Free request 1 (blocks stay in cache)
        mgr.free(1);

        // Request 2: same prefix -> should get cache hit
        let computed = mgr.get_computed_blocks(&hashes, 12);
        assert_eq!(computed.num_computed_tokens, 8);
        assert_eq!(computed.block_ids.len(), 2);

        // Allocate with cached blocks
        let new_blocks = mgr.allocate_slots(2, 12, &computed.block_ids).unwrap();
        assert_eq!(new_blocks.len(), 1); // only 1 new block needed
        assert_eq!(mgr.num_blocks_for_request(2), 3); // 2 cached + 1 new
    }

    #[test]
    fn test_prefix_cache_partial_hit() {
        let mut mgr = KVCacheManager::new(16, 4, true, vec![0]);

        // Cache 2 blocks
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);
        mgr.allocate_slots(1, 8, &[]).unwrap();
        mgr.cache_blocks(1, &hashes, 8);
        mgr.free(1);

        // Request with 3 blocks: first 2 should hit cache, 3rd is new
        let tokens_ext: Vec<u32> = (1..=12).collect();
        let hashes_ext = compute_block_hashes(&tokens_ext, 4, &[], &[]);
        let computed = mgr.get_computed_blocks(&hashes_ext, 12);
        assert_eq!(computed.num_computed_tokens, 8);
    }

    #[test]
    fn test_cache_blocks_incremental() {
        let mut mgr = KVCacheManager::new(16, 4, true, vec![0]);

        let tokens: Vec<u32> = (1..=16).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);

        mgr.allocate_slots(1, 16, &[]).unwrap();

        // Cache first 2 blocks
        mgr.cache_blocks(1, &hashes, 8);
        assert_eq!(mgr.num_cached_blocks(1), 2);

        // Cache all 4 blocks
        mgr.cache_blocks(1, &hashes, 16);
        assert_eq!(mgr.num_cached_blocks(1), 4);
    }

    #[test]
    fn test_slot_mapping() {
        let mut mgr = KVCacheManager::new(16, 4, false, vec![0]);

        mgr.allocate_slots(1, 8, &[]).unwrap();
        let block_ids = mgr.get_block_ids(1).unwrap().to_vec();

        // Map tokens 0..8
        let slots = mgr.get_slot_mapping(1, 0, 8).unwrap();
        assert_eq!(slots.len(), 8);

        // First 4 tokens should be in block_ids[0]
        for (i, slot) in slots.iter().enumerate().take(4) {
            assert_eq!(*slot, (block_ids[0] * 4 + i) as i64);
        }
        // Next 4 in block_ids[1]
        for (i, slot) in slots[4..].iter().enumerate().take(4) {
            assert_eq!(*slot, (block_ids[1] * 4 + i) as i64);
        }
    }

    #[test]
    fn test_slot_mapping_skip_cached() {
        let mut mgr = KVCacheManager::new(16, 4, true, vec![0]);

        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);
        mgr.allocate_slots(1, 8, &[]).unwrap();
        mgr.cache_blocks(1, &hashes, 8);
        mgr.free(1);

        // New request reuses cache
        let computed = mgr.get_computed_blocks(&hashes, 12);
        mgr.allocate_slots(2, 12, &computed.block_ids).unwrap();

        // Slot mapping for only new tokens (starting from token 8)
        let slots = mgr.get_slot_mapping(2, 8, 4).unwrap();
        assert_eq!(slots.len(), 4);
    }

    #[test]
    fn test_block_table() {
        let mut mgr = KVCacheManager::new(16, 4, false, vec![0]);
        mgr.allocate_slots(1, 8, &[]).unwrap();

        let table = mgr.get_block_table(1, 5).unwrap();
        assert_eq!(table.len(), 5);
        // Last entries should be 0 (padding)
        assert_eq!(table[2], 0);
        assert_eq!(table[3], 0);
        assert_eq!(table[4], 0);
    }

    #[test]
    fn test_get_computed_blocks_caps_at_prompt_minus_one() {
        let mut mgr = KVCacheManager::new(16, 4, true, vec![0]);

        // Cache exactly 2 blocks (8 tokens)
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);
        mgr.allocate_slots(1, 8, &[]).unwrap();
        mgr.cache_blocks(1, &hashes, 8);
        mgr.free(1);

        // If num_tokens == 8, max_cache_hit = 7, which is less than 8 (2 blocks)
        // So we can only use 1 block (4 tokens)
        let computed = mgr.get_computed_blocks(&hashes, 8);
        assert_eq!(computed.num_computed_tokens, 4);
        assert_eq!(computed.block_ids.len(), 1);
    }

    #[test]
    fn test_reset_prefix_cache() {
        let mut mgr = KVCacheManager::new(8, 4, true, vec![0]);

        let tokens: Vec<u32> = (1..=4).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);
        mgr.allocate_slots(1, 4, &[]).unwrap();
        mgr.cache_blocks(1, &hashes, 4);

        // Can't reset while blocks are in use
        assert!(!mgr.reset_prefix_cache());

        mgr.free(1);
        assert!(mgr.reset_prefix_cache());

        // Cache should be empty now
        let computed = mgr.get_computed_blocks(&hashes, 8);
        assert_eq!(computed.num_computed_tokens, 0);
    }
}
