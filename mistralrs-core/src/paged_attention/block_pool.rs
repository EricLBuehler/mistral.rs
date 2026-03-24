//! Block pool for paged attention KV cache management.
//!
//! This module implements a flat block pool with a doubly-linked free list,
//! following vLLM's v1 `BlockPool` + `FreeKVCacheBlockQueue` design.
//!
//! Key properties:
//! - O(1) allocation (pop from free list head — LRU eviction order)
//! - O(1) free (append to free list tail)
//! - O(1) remove from middle of free list (for cache hits)
//! - Freed blocks retain their hash for potential future reuse
//! - Blocks are only truly evicted (hash cleared) when reallocated

use std::collections::HashMap;

use super::block_hash::{BlockHash, BlockHashWithGroupId};

/// Sentinel value for "no link" in the doubly-linked free list.
const NO_LINK: usize = usize::MAX;

/// Metadata for a single KV cache block.
///
/// Blocks are stored in a flat `Vec` indexed by `block_id`. The free list
/// is implemented as a doubly-linked list threaded through `prev_free`/`next_free`
/// fields, avoiding any heap allocation for list nodes.
#[derive(Debug)]
pub struct KVCacheBlock {
    /// Block ID, ranging from 0 to num_gpu_blocks - 1.
    #[allow(dead_code)]
    pub block_id: usize,
    /// Reference count. 0 means the block is free (in the free list or eviction candidate).
    pub ref_cnt: u32,
    /// The content-based hash of this block, set when the block is full and cached.
    /// Retained even when freed (ref_cnt drops to 0) so the block can be reused
    /// if a future request has the same prefix. Cleared only when the block is
    /// evicted (reallocated to a different request).
    pub block_hash: Option<BlockHashWithGroupId>,
    /// Previous block in the free list (NO_LINK if not in free list or at head).
    prev_free: usize,
    /// Next block in the free list (NO_LINK if not in free list or at tail).
    next_free: usize,
    /// Whether this is the null block (placeholder, never freed or cached).
    pub is_null: bool,
}

impl KVCacheBlock {
    fn new(block_id: usize) -> Self {
        Self {
            block_id,
            ref_cnt: 0,
            block_hash: None,
            prev_free: NO_LINK,
            next_free: NO_LINK,
            is_null: false,
        }
    }

    /// Check if this block is currently in the free list.
    #[allow(dead_code)]
    fn is_in_free_list(&self) -> bool {
        self.prev_free != NO_LINK || self.next_free != NO_LINK
    }

    /// Reset the hash when the block is evicted (reallocated).
    fn reset_hash(&mut self) {
        self.block_hash = None;
    }
}

/// Doubly-linked list of free blocks, ordered for LRU eviction.
///
/// Uses fake head and tail sentinels (stored as indices into the blocks array)
/// to avoid branching. Supports O(1) `popleft`, `remove`, and `append`.
///
/// Eviction order:
/// - Front (head) = least recently used -> evict first
/// - Back (tail) = most recently freed -> evict last
struct FreeKVCacheBlockQueue {
    /// Number of free blocks currently in the queue.
    num_free_blocks: usize,
    /// Index of the fake head sentinel in the blocks array.
    fake_head: usize,
    /// Index of the fake tail sentinel in the blocks array.
    fake_tail: usize,
}

impl FreeKVCacheBlockQueue {
    /// Initialize the free list with all blocks linked in order.
    ///
    /// `blocks` must already contain the fake head at `fake_head` and fake tail at `fake_tail`.
    /// All blocks in `block_ids` are linked between them.
    fn new(
        blocks: &mut [KVCacheBlock],
        block_ids: &[usize],
        fake_head: usize,
        fake_tail: usize,
    ) -> Self {
        let n = block_ids.len();

        // Link consecutive blocks
        for i in 0..n {
            let id = block_ids[i];
            blocks[id].prev_free = if i > 0 { block_ids[i - 1] } else { fake_head };
            blocks[id].next_free = if i + 1 < n {
                block_ids[i + 1]
            } else {
                fake_tail
            };
        }

        // Connect fake head and tail
        if n > 0 {
            blocks[fake_head].next_free = block_ids[0];
            blocks[fake_tail].prev_free = block_ids[n - 1];
        } else {
            blocks[fake_head].next_free = fake_tail;
            blocks[fake_tail].prev_free = fake_head;
        }

        Self {
            num_free_blocks: n,
            fake_head,
            fake_tail,
        }
    }

    /// Pop the first (least recently used) block from the free list.
    fn popleft(&mut self, blocks: &mut [KVCacheBlock]) -> Option<usize> {
        let first_id = blocks[self.fake_head].next_free;
        if first_id == self.fake_tail {
            return None; // Empty
        }

        let next_id = blocks[first_id].next_free;

        // Unlink first_id
        blocks[self.fake_head].next_free = next_id;
        blocks[next_id].prev_free = self.fake_head;
        blocks[first_id].prev_free = NO_LINK;
        blocks[first_id].next_free = NO_LINK;

        self.num_free_blocks -= 1;
        Some(first_id)
    }

    /// Remove a specific block from the free list (O(1) since doubly-linked).
    fn remove(&mut self, blocks: &mut [KVCacheBlock], block_id: usize) {
        let prev_id = blocks[block_id].prev_free;
        let next_id = blocks[block_id].next_free;

        debug_assert!(
            prev_id != NO_LINK && next_id != NO_LINK,
            "remove() called on block {} not in free list",
            block_id
        );

        // Unlink
        blocks[prev_id].next_free = next_id;
        blocks[next_id].prev_free = prev_id;
        blocks[block_id].prev_free = NO_LINK;
        blocks[block_id].next_free = NO_LINK;

        self.num_free_blocks -= 1;
    }

    /// Append a block to the tail of the free list (most recently freed).
    fn append(&mut self, blocks: &mut [KVCacheBlock], block_id: usize) {
        let last_id = blocks[self.fake_tail].prev_free;

        blocks[last_id].next_free = block_id;
        blocks[block_id].prev_free = last_id;
        blocks[block_id].next_free = self.fake_tail;
        blocks[self.fake_tail].prev_free = block_id;

        self.num_free_blocks += 1;
    }
}

/// Map from block hash (with group ID) to cached block(s).
///
/// Most hash keys map to a single block. When hash collisions or duplicate
/// caching occur, the value becomes a HashMap of block_id -> block_id.
/// This follows vLLM's `BlockHashToBlockMap` optimization to avoid allocating
/// a HashMap for the common single-block case.
pub struct BlockHashToBlockMap {
    cache: HashMap<BlockHashWithGroupId, CachedBlocks>,
}

enum CachedBlocks {
    Single(usize),
    Multiple(HashMap<usize, usize>), // block_id -> block_id
}

impl BlockHashToBlockMap {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get any cached block ID for the given hash, or None.
    fn get_one(&self, key: &BlockHashWithGroupId) -> Option<usize> {
        match self.cache.get(key)? {
            CachedBlocks::Single(id) => Some(*id),
            CachedBlocks::Multiple(map) => map.values().next().copied(),
        }
    }

    /// Insert a block into the cache.
    fn insert(&mut self, key: BlockHashWithGroupId, block_id: usize) {
        match self.cache.get_mut(&key) {
            None => {
                self.cache.insert(key, CachedBlocks::Single(block_id));
            }
            Some(CachedBlocks::Single(existing_id)) => {
                let existing = *existing_id;
                let mut map = HashMap::new();
                map.insert(existing, existing);
                map.insert(block_id, block_id);
                self.cache.insert(key, CachedBlocks::Multiple(map));
            }
            Some(CachedBlocks::Multiple(map)) => {
                map.insert(block_id, block_id);
            }
        }
    }

    /// Remove a specific block_id from the cache entry for the given hash.
    /// Returns the block_id if found and removed.
    fn pop(&mut self, key: &BlockHashWithGroupId, block_id: usize) -> Option<usize> {
        let entry = self.cache.remove(key)?;
        match entry {
            CachedBlocks::Single(id) => {
                if id == block_id {
                    Some(id)
                } else {
                    // Put it back — this block_id doesn't match
                    self.cache.insert(*key, CachedBlocks::Single(id));
                    None
                }
            }
            CachedBlocks::Multiple(mut map) => {
                let result = map.remove(&block_id);
                if map.len() == 1 {
                    // Compact back to Single variant
                    let single_id = *map.values().next().unwrap();
                    self.cache.insert(*key, CachedBlocks::Single(single_id));
                } else if !map.is_empty() {
                    self.cache.insert(*key, CachedBlocks::Multiple(map));
                }
                result
            }
        }
    }

    fn len(&self) -> usize {
        self.cache.len()
    }

    fn clear(&mut self) {
        self.cache.clear();
    }
}

/// The block pool manages all physical KV cache blocks.
///
/// It provides allocation, freeing, and prefix cache operations.
/// Freed blocks retain their hash for potential reuse — they are only truly
/// evicted (hash cleared) when they are reallocated to a new request.
pub struct BlockPool {
    /// All blocks, indexed by block_id. Includes 2 extra sentinel blocks at the end.
    blocks: Vec<KVCacheBlock>,
    /// The free block queue (doubly-linked list).
    free_queue: FreeKVCacheBlockQueue,
    /// Hash-to-block map for prefix cache lookups.
    cached_block_hash_to_block: BlockHashToBlockMap,
    /// Whether prefix caching is enabled.
    enable_caching: bool,
    /// Total number of real GPU blocks (excludes sentinels and null block).
    num_gpu_blocks: usize,
    /// The null block ID (block 0). Used as a placeholder, never freed.
    null_block_id: usize,
    /// The block size (number of tokens per block) for hash computation.
    hash_block_size: usize,
}

impl BlockPool {
    /// Create a new block pool.
    ///
    /// `num_gpu_blocks`: Number of physical GPU blocks.
    /// `enable_caching`: Whether to enable prefix caching.
    /// `hash_block_size`: Block size used for hash computation.
    pub fn new(num_gpu_blocks: usize, enable_caching: bool, hash_block_size: usize) -> Self {
        assert!(num_gpu_blocks > 0, "Must have at least 1 GPU block");

        // Allocate blocks: [0..num_gpu_blocks) are real blocks,
        // [num_gpu_blocks] = fake head sentinel, [num_gpu_blocks+1] = fake tail sentinel
        let fake_head = num_gpu_blocks;
        let fake_tail = num_gpu_blocks + 1;
        let total = num_gpu_blocks + 2;

        let mut blocks: Vec<KVCacheBlock> = (0..total).map(KVCacheBlock::new).collect();

        // All real block IDs (0..num_gpu_blocks) go into the free list initially
        let all_ids: Vec<usize> = (0..num_gpu_blocks).collect();
        let free_queue = FreeKVCacheBlockQueue::new(&mut blocks, &all_ids, fake_head, fake_tail);

        let mut pool = Self {
            blocks,
            free_queue,
            cached_block_hash_to_block: BlockHashToBlockMap::new(),
            enable_caching,
            num_gpu_blocks,
            null_block_id: 0, // Will be set below
            hash_block_size,
        };

        // Pop the first block as the null block (placeholder, never freed)
        let null_id = pool
            .free_queue
            .popleft(&mut pool.blocks)
            .expect("Pool should have blocks");
        pool.blocks[null_id].is_null = true;
        pool.null_block_id = null_id;

        pool
    }

    /// Get the null block ID (placeholder for skipped/unused slots).
    pub fn null_block_id(&self) -> usize {
        self.null_block_id
    }

    /// Get the number of free blocks available for allocation.
    pub fn num_free_blocks(&self) -> usize {
        self.free_queue.num_free_blocks
    }

    /// Get total number of GPU blocks (excluding sentinels, including null block).
    pub fn num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }

    /// Get KV cache usage as a fraction [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    pub fn usage(&self) -> f64 {
        let total = self.num_gpu_blocks - 1; // Exclude null block
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.num_free_blocks() as f64 / total as f64)
    }

    /// Look up cached blocks for a given hash across the specified group IDs.
    ///
    /// Returns `Some(vec_of_block_ids)` if ALL groups have a cached block for
    /// this hash, or `None` if any group misses.
    pub fn get_cached_block(
        &self,
        block_hash: BlockHash,
        kv_cache_group_ids: &[u32],
    ) -> Option<Vec<usize>> {
        let mut cached_ids = Vec::with_capacity(kv_cache_group_ids.len());
        for &group_id in kv_cache_group_ids {
            let key = BlockHashWithGroupId {
                block_hash,
                group_id,
            };
            match self.cached_block_hash_to_block.get_one(&key) {
                Some(id) => cached_ids.push(id),
                None => return None,
            }
        }
        Some(cached_ids)
    }

    /// Touch blocks — increment ref_cnt and remove from free list if ref_cnt was 0.
    ///
    /// Called when cached blocks are reused by a new request (prefix cache hit).
    pub fn touch(&mut self, block_ids: &[usize]) {
        for &block_id in block_ids {
            let block = &mut self.blocks[block_id];
            if block.ref_cnt == 0 && !block.is_null {
                // Remove from free list since it's being actively used again
                self.free_queue.remove(&mut self.blocks, block_id);
            } else {
                // For blocks already removed, just update ref_cnt below
            }
            self.blocks[block_id].ref_cnt += 1;
        }
    }

    /// Free blocks — decrement ref_cnt, append to free list tail if it hits 0.
    ///
    /// Blocks retain their hash when freed! They stay in the cache for potential
    /// future reuse. They are only evicted (hash cleared) when reallocated.
    ///
    /// `ordered_block_ids`: blocks ordered by eviction priority (first = evict first).
    /// For a request being freed, pass blocks in REVERSE order so the tail of the
    /// sequence (most specific) is evicted first.
    pub fn free_blocks(&mut self, ordered_block_ids: &[usize]) {
        // First pass: decrement ref_cnt
        for &block_id in ordered_block_ids {
            debug_assert!(
                self.blocks[block_id].ref_cnt > 0,
                "Block {block_id} ref_cnt underflow: attempting to free block with ref_cnt=0"
            );
            self.blocks[block_id].ref_cnt = self.blocks[block_id].ref_cnt.saturating_sub(1);
        }

        // Second pass: add newly-free blocks to the free list
        for &block_id in ordered_block_ids {
            if self.blocks[block_id].ref_cnt == 0 && !self.blocks[block_id].is_null {
                self.free_queue.append(&mut self.blocks, block_id);
            }
        }
    }

    /// Allocate `num_blocks` new blocks from the free pool.
    ///
    /// Blocks are popped from the head of the free list (LRU eviction).
    /// If a popped block has a cached hash, it is evicted from the cache.
    ///
    /// Returns `None` if not enough free blocks are available.
    pub fn get_new_blocks(&mut self, num_blocks: usize) -> Option<Vec<usize>> {
        if num_blocks > self.free_queue.num_free_blocks {
            return None;
        }

        let mut result = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let block_id = self
                .free_queue
                .popleft(&mut self.blocks)
                .expect("Should have enough free blocks");

            // Evict from cache if this block was cached
            if self.enable_caching {
                self.maybe_evict_cached_block(block_id);
            }

            debug_assert_eq!(self.blocks[block_id].ref_cnt, 0);
            self.blocks[block_id].ref_cnt = 1;
            result.push(block_id);
        }

        Some(result)
    }

    /// Cache full blocks by assigning hashes and inserting into the hash map.
    ///
    /// Called after prefill completes to make newly-full blocks available for
    /// prefix cache reuse by future requests.
    ///
    /// `block_ids`: all block IDs for the request.
    /// `block_hashes`: computed hashes for the request's blocks.
    /// `num_cached_blocks`: number of blocks already cached (skip these).
    /// `num_full_blocks`: total number of full blocks to cache.
    /// `kv_cache_group_id`: the KV cache group this block belongs to.
    pub fn cache_full_blocks(
        &mut self,
        block_ids: &[usize],
        block_hashes: &[BlockHash],
        num_cached_blocks: usize,
        num_full_blocks: usize,
        kv_cache_group_id: u32,
    ) {
        if !self.enable_caching || num_cached_blocks >= num_full_blocks {
            return;
        }

        assert!(
            block_hashes.len() >= num_full_blocks,
            "Not enough block hashes ({}) for {} full blocks",
            block_hashes.len(),
            num_full_blocks
        );

        for idx in num_cached_blocks..num_full_blocks {
            let block_id = block_ids[idx];
            let block = &mut self.blocks[block_id];

            // Skip null blocks and already-cached blocks
            if block.is_null || block.block_hash.is_some() {
                continue;
            }

            let hash_with_group = BlockHashWithGroupId {
                block_hash: block_hashes[idx],
                group_id: kv_cache_group_id,
            };

            block.block_hash = Some(hash_with_group);
            self.cached_block_hash_to_block
                .insert(hash_with_group, block_id);
        }
    }

    /// Evict a cached block's hash from the cache map and reset its hash.
    fn maybe_evict_cached_block(&mut self, block_id: usize) {
        let block_hash = self.blocks[block_id].block_hash;
        if let Some(hash) = block_hash {
            self.cached_block_hash_to_block.pop(&hash, block_id);
            self.blocks[block_id].reset_hash();
        }
    }

    /// Reset the entire prefix cache. Only succeeds if all blocks are free.
    pub fn reset_prefix_cache(&mut self) -> bool {
        let num_used = self.num_gpu_blocks - self.num_free_blocks();
        if num_used != 1 {
            // Only the null block should be "used"
            return false;
        }

        self.cached_block_hash_to_block.clear();
        for block in &mut self.blocks {
            block.reset_hash();
        }

        true
    }

    /// Get the number of cached blocks in the hash map.
    pub fn num_cached_blocks(&self) -> usize {
        self.cached_block_hash_to_block.len()
    }

    /// Get the block size used for hash computation.
    pub fn hash_block_size(&self) -> usize {
        self.hash_block_size
    }

    /// Check whether caching is enabled.
    pub fn caching_enabled(&self) -> bool {
        self.enable_caching
    }

    /// Get the ref_cnt for a block (for debugging/testing).
    pub fn block_ref_cnt(&self, block_id: usize) -> u32 {
        self.blocks[block_id].ref_cnt
    }

    /// Get the block hash for a block (for debugging/testing).
    pub fn block_hash(&self, block_id: usize) -> Option<BlockHashWithGroupId> {
        self.blocks[block_id].block_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::block_hash::hash_block_tokens;

    #[test]
    fn test_basic_allocation() {
        let mut pool = BlockPool::new(4, false, 16);
        // 4 blocks total, 1 is null -> 3 free
        assert_eq!(pool.num_free_blocks(), 3);

        let blocks = pool.get_new_blocks(2).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(pool.num_free_blocks(), 1);

        // Check ref_cnt
        for &id in &blocks {
            assert_eq!(pool.block_ref_cnt(id), 1);
        }
    }

    #[test]
    fn test_free_returns_to_pool() {
        let mut pool = BlockPool::new(4, false, 16);
        let blocks = pool.get_new_blocks(3).unwrap();
        assert_eq!(pool.num_free_blocks(), 0);

        pool.free_blocks(&blocks);
        assert_eq!(pool.num_free_blocks(), 3);

        // All blocks should have ref_cnt 0
        for &id in &blocks {
            assert_eq!(pool.block_ref_cnt(id), 0);
        }
    }

    #[test]
    fn test_allocation_fails_when_exhausted() {
        let mut pool = BlockPool::new(2, false, 16);
        // 2 blocks, 1 null -> 1 free
        assert_eq!(pool.num_free_blocks(), 1);

        let _b = pool.get_new_blocks(1).unwrap();
        assert_eq!(pool.num_free_blocks(), 0);

        assert!(pool.get_new_blocks(1).is_none());
    }

    #[test]
    fn test_prefix_cache_basic() {
        let mut pool = BlockPool::new(8, true, 4);

        // Allocate 3 blocks for a request
        let block_ids = pool.get_new_blocks(3).unwrap();

        // Compute some hashes
        let h0 = hash_block_tokens(None, &[1, 2, 3, 4], None);
        let h1 = hash_block_tokens(Some(h0), &[5, 6, 7, 8], None);
        let h2 = hash_block_tokens(Some(h1), &[9, 10, 11, 12], None);
        let hashes = vec![h0, h1, h2];

        // Cache the blocks
        pool.cache_full_blocks(&block_ids, &hashes, 0, 3, 0);

        // Verify they're cached
        assert_eq!(pool.num_cached_blocks(), 3);

        // Look up by hash
        let cached = pool.get_cached_block(h0, &[0]);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap()[0], block_ids[0]);
    }

    #[test]
    fn test_prefix_cache_reuse_after_free() {
        let mut pool = BlockPool::new(8, true, 4);

        let block_ids = pool.get_new_blocks(2).unwrap();
        let h0 = hash_block_tokens(None, &[1, 2, 3, 4], None);
        let h1 = hash_block_tokens(Some(h0), &[5, 6, 7, 8], None);
        let hashes = vec![h0, h1];

        pool.cache_full_blocks(&block_ids, &hashes, 0, 2, 0);

        // Free the blocks (they should retain their hash)
        pool.free_blocks(&block_ids);
        assert_eq!(pool.num_free_blocks(), 7); // back in free list (7 = 8 - 1 null)

        // The blocks should still be findable by hash
        let cached = pool.get_cached_block(h0, &[0]);
        assert!(cached.is_some());

        // Touch to reuse (simulates a new request with same prefix)
        let cached_ids = cached.unwrap();
        pool.touch(&cached_ids);
        assert_eq!(pool.block_ref_cnt(cached_ids[0]), 1);
        // Block should no longer be in free list
        assert_eq!(pool.num_free_blocks(), 6);
    }

    #[test]
    fn test_eviction_on_reallocation() {
        let mut pool = BlockPool::new(4, true, 4);
        // 3 free blocks (1 null)

        // Fill all blocks
        let block_ids = pool.get_new_blocks(3).unwrap();
        let h0 = hash_block_tokens(None, &[1, 2, 3, 4], None);
        pool.cache_full_blocks(&block_ids, &[h0, h0, h0], 0, 1, 0);

        // Free all
        pool.free_blocks(&block_ids);

        // Now allocate again — should evict the cached block
        let new_ids = pool.get_new_blocks(3).unwrap();
        assert_eq!(new_ids.len(), 3);

        // The evicted block should no longer be in cache
        // (one of the blocks had hash h0, now it's been evicted)
    }

    #[test]
    fn test_touch_ref_cnt_management() {
        let mut pool = BlockPool::new(8, true, 4);
        let block_ids = pool.get_new_blocks(1).unwrap();
        assert_eq!(pool.block_ref_cnt(block_ids[0]), 1);

        // Touch increments ref_cnt
        pool.touch(&block_ids);
        assert_eq!(pool.block_ref_cnt(block_ids[0]), 2);

        // Free once — ref_cnt should be 1, not in free list
        pool.free_blocks(&block_ids);
        assert_eq!(pool.block_ref_cnt(block_ids[0]), 1);

        // Free again — ref_cnt 0, added to free list
        pool.free_blocks(&block_ids);
        assert_eq!(pool.block_ref_cnt(block_ids[0]), 0);
    }

    #[test]
    fn test_null_block_never_freed() {
        let mut pool = BlockPool::new(4, false, 16);
        let null_id = pool.null_block_id();

        // Null block should not be in free list
        assert!(pool.blocks[null_id].is_null);

        // Even if we try to free it, it won't go to the free list
        pool.blocks[null_id].ref_cnt = 1;
        pool.free_blocks(&[null_id]);
        // ref_cnt decremented but not added to free list
        assert_eq!(pool.block_ref_cnt(null_id), 0);
    }

    #[test]
    fn test_usage() {
        let mut pool = BlockPool::new(4, false, 16);
        // 3 usable blocks (4 total - 1 null)
        assert!(pool.usage() < 0.01); // ~0

        let _b = pool.get_new_blocks(3).unwrap();
        assert!((pool.usage() - 1.0).abs() < 0.01); // ~1.0
    }

    #[test]
    fn test_get_cached_block_multiple_groups() {
        let mut pool = BlockPool::new(8, true, 4);

        // Allocate blocks for two groups
        let ids_g0 = pool.get_new_blocks(1).unwrap();
        let ids_g1 = pool.get_new_blocks(1).unwrap();

        let h0 = hash_block_tokens(None, &[1, 2, 3, 4], None);

        // Cache with different group IDs
        pool.cache_full_blocks(&ids_g0, &[h0], 0, 1, 0);
        pool.cache_full_blocks(&ids_g1, &[h0], 0, 1, 1);

        // Should find both groups
        let cached = pool.get_cached_block(h0, &[0, 1]);
        assert!(cached.is_some());
        let cached = cached.unwrap();
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0], ids_g0[0]);
        assert_eq!(cached[1], ids_g1[0]);

        // Should fail if one group is missing
        let cached = pool.get_cached_block(h0, &[0, 2]);
        assert!(cached.is_none());
    }

    #[test]
    fn test_reset_prefix_cache() {
        let mut pool = BlockPool::new(4, true, 4);
        let ids = pool.get_new_blocks(2).unwrap();
        let h0 = hash_block_tokens(None, &[1, 2, 3, 4], None);
        pool.cache_full_blocks(&ids, &[h0, h0], 0, 1, 0);

        // Can't reset while blocks are in use
        assert!(!pool.reset_prefix_cache());

        // Free blocks
        pool.free_blocks(&ids);

        // Now reset should succeed
        assert!(pool.reset_prefix_cache());
        assert_eq!(pool.num_cached_blocks(), 0);
    }
}
