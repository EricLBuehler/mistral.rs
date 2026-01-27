use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Mutex},
};

use super::block_engine_sequence::BlockEngineSequence;
use super::prefix_cacher::PrefixCacher;

#[derive(Debug, Clone)]
pub struct LogicalTokenBlock {
    tokens: Vec<usize>,
    block_size: usize,
    num_tokens: usize,
}

impl LogicalTokenBlock {
    pub fn new(block_size: usize) -> Self {
        Self {
            tokens: [0].repeat(block_size),
            block_size,
            num_tokens: 0,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn is_full(&self) -> bool {
        self.num_tokens == self.block_size
    }

    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    pub fn append_token_id(&mut self, token: usize) {
        assert!(!self.is_full());
        self.tokens[self.num_tokens] = token;
        self.num_tokens += 1;
    }

    pub fn pop_token(&mut self) {
        assert_ne!(self.num_tokens, 0);
        self.tokens.pop();
        self.num_tokens -= 1;
    }

    pub fn toks(&self) -> &[usize] {
        &self.tokens
    }
}

impl Hash for LogicalTokenBlock {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tokens.hash(state);
    }
}

// ============================================================================
// Automatic Reference-Counted Block System
// ============================================================================

/// Inner state for a single physical block.
/// When this is dropped (last Arc reference gone), the block returns to the free list.
struct BlockInner {
    block_id: usize,
    pool: Arc<BlockPoolInner>,
}

impl Drop for BlockInner {
    fn drop(&mut self) {
        // Automatically return block to free list when last reference is dropped
        self.pool
            .free_list
            .lock()
            .expect("BlockPool free_list lock poisoned")
            .push(self.block_id);
    }
}

/// Shared pool state accessed by all BlockRef instances.
struct BlockPoolInner {
    free_list: Mutex<Vec<usize>>,
    #[allow(dead_code)]
    block_size: usize,
    #[allow(dead_code)]
    num_blocks: usize,
}

/// A reference-counted handle to a physical block.
///
/// This works exactly like `Arc`:
/// - Clone to share (automatic refcount increment)
/// - Drop when done (automatic refcount decrement)
/// - When the last reference is dropped, the block automatically returns to the free pool
///
/// This design makes refcount bugs impossible - there's no manual increment/decrement.
#[derive(Clone)]
pub struct BlockRef(Arc<BlockInner>);

impl std::fmt::Debug for BlockRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockRef")
            .field("block_id", &self.0.block_id)
            .field("ref_count", &Arc::strong_count(&self.0))
            .finish()
    }
}

impl BlockRef {
    /// Get the underlying block ID (index into GPU memory).
    #[inline]
    pub fn block_id(&self) -> usize {
        self.0.block_id
    }

    /// Check if this block is shared (more than one reference exists).
    /// Used for copy-on-write decisions.
    #[inline]
    pub fn is_shared(&self) -> bool {
        Arc::strong_count(&self.0) > 1
    }

    /// Get the current reference count (for debugging).
    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

impl PartialEq for BlockRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.block_id == other.0.block_id
    }
}

impl Eq for BlockRef {}

impl Hash for BlockRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.block_id.hash(state);
    }
}

/// The block pool that manages physical block allocation.
///
/// Blocks are automatically returned to the pool when all references are dropped.
pub struct BlockPool {
    inner: Arc<BlockPoolInner>,
}

impl BlockPool {
    /// Create a new block pool with the given number of blocks.
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        let free_list: Vec<usize> = (0..num_blocks).collect();
        Self {
            inner: Arc::new(BlockPoolInner {
                free_list: Mutex::new(free_list),
                block_size,
                num_blocks,
            }),
        }
    }

    /// Allocate a block from the pool. Returns None if no blocks are available.
    pub fn allocate(&self) -> Option<BlockRef> {
        let block_id = self
            .inner
            .free_list
            .lock()
            .expect("BlockPool free_list lock poisoned")
            .pop()?;

        Some(BlockRef(Arc::new(BlockInner {
            block_id,
            pool: Arc::clone(&self.inner),
        })))
    }

    /// Get the number of free blocks.
    #[inline]
    pub fn num_free(&self) -> usize {
        self.inner
            .free_list
            .lock()
            .expect("BlockPool free_list lock poisoned")
            .len()
    }

    /// Get the block size.
    #[inline]
    #[allow(dead_code)]
    pub fn block_size(&self) -> usize {
        self.inner.block_size
    }

    /// Get the total number of blocks in the pool.
    #[inline]
    #[allow(dead_code)]
    pub fn num_blocks(&self) -> usize {
        self.inner.num_blocks
    }
}

/// Block table type: maps sequence ID to list of block references.
pub type BlockTable = Vec<BlockRef>;
pub type BlockTables = HashMap<usize, BlockTable>;

#[derive(Debug)]
pub enum AllocStatus {
    Ok,
    Later { waitlisted_count: usize },
    Impossible,
}

type SeqID = usize;

/// A BlockEngine maps each Sequence (identified by its SeqID), to physical token blocks.
/// The physical token blocks may not match the logical token blocks because during
/// scheduling, physical blocks are allocated to accommodate the new tokens generated.
/// These new tokens will be added to the logical token block for each sequence.
pub struct BlockEngine {
    num_gpu_blocks: usize,
    block_size: usize,
    /// The block pool for allocation. Blocks are automatically returned when refs are dropped.
    pool: BlockPool,
    /// Maps sequence ID to its block table.
    pub block_tables: HashMap<SeqID, BlockTable>,
    /// Prefix cache for reusing KV cache blocks across requests with shared prefixes.
    prefix_cacher: PrefixCacher,
    /// Track number of cached blocks used per sequence (for releasing cache entries).
    cached_blocks_per_seq: HashMap<SeqID, usize>,
}

impl BlockEngine {
    #[must_use]
    pub fn new(block_size: usize, num_gpu_blocks: usize, prefix_caching_enabled: bool) -> Self {
        Self {
            num_gpu_blocks,
            block_size,
            pool: BlockPool::new(block_size, num_gpu_blocks),
            block_tables: HashMap::new(),
            prefix_cacher: PrefixCacher::new(prefix_caching_enabled),
            cached_blocks_per_seq: HashMap::new(),
        }
    }

    /// Check if prefix caching is enabled.
    pub fn prefix_caching_enabled(&self) -> bool {
        self.prefix_cacher.is_enabled()
    }

    /// Set whether prefix caching is enabled.
    pub fn set_prefix_caching_enabled(&mut self, enabled: bool) {
        self.prefix_cacher.set_enabled(enabled);
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn can_allocate(&mut self, seq: &mut impl BlockEngineSequence) -> AllocStatus {
        let logical_blocks = seq.logical_token_blocks();
        let num_required_blocks = logical_blocks.len();

        // Check how many blocks we can get from prefix cache
        let num_cached = if self.prefix_cacher.is_enabled() {
            let (_, num_matched) = self.prefix_cacher.match_prefix(logical_blocks);
            num_matched
        } else {
            0
        };

        // We only need to allocate blocks that aren't in the cache
        let num_new_blocks_needed = num_required_blocks.saturating_sub(num_cached);
        let num_free_gpu_blocks = self.pool.num_free();

        // Also count evictable blocks from prefix cache as potentially available
        let num_evictable = self.prefix_cacher.num_evictable_blocks();
        let total_available = num_free_gpu_blocks + num_evictable;

        if self.num_gpu_blocks < num_required_blocks {
            AllocStatus::Impossible
        } else if total_available < num_new_blocks_needed {
            AllocStatus::Later {
                waitlisted_count: seq.increment_waitlist_count(),
            }
        } else {
            AllocStatus::Ok
        }
    }

    pub fn allocate(&mut self, seq: &mut impl BlockEngineSequence) {
        let num_blocks_needed = seq.logical_token_blocks().len();
        let seq_id = seq.get_id();
        let block_size = seq.block_size();

        // If there are prefill physical blocks, use those here.
        if let Some(physical_blocks_prefill) = seq.take_physical_blocks_prefill() {
            let mut block_table = physical_blocks_prefill;
            let n_extra_blocks = num_blocks_needed - block_table.len();
            for _ in 0..n_extra_blocks {
                block_table.push(self.allocate_block_with_eviction());
            }
            self.block_tables.insert(seq_id, block_table);
            self.cached_blocks_per_seq.insert(seq_id, 0);
            seq.set_prefix_cache_len(0);
            return;
        }

        // Re-borrow logical_blocks after the mutable borrow above is done
        let logical_blocks = seq.logical_token_blocks();

        // Try to get blocks from prefix cache
        let (cached_blocks, num_cached) = if self.prefix_cacher.is_enabled() {
            self.prefix_cacher.match_prefix(logical_blocks)
        } else {
            (Vec::new(), 0)
        };

        let mut block_table = Vec::with_capacity(num_blocks_needed);

        // Use cached blocks for the prefix (they're already cloned by match_prefix)
        for (idx, block_ref) in cached_blocks {
            // Extend block_table to the right size
            while block_table.len() < idx {
                block_table.push(self.allocate_block_with_eviction());
            }
            block_table.push(block_ref);
        }

        // Allocate new blocks for the rest
        for _ in block_table.len()..num_blocks_needed {
            block_table.push(self.allocate_block_with_eviction());
        }

        self.cached_blocks_per_seq.insert(seq_id, num_cached);
        self.block_tables.insert(seq_id, block_table);

        // Calculate number of cached tokens (full blocks only)
        let cached_tokens = num_cached * block_size;
        seq.set_prefix_cache_len(cached_tokens);
    }

    /// Check if the last allocate() call resulted in a prefix cache hit.
    /// Returns the number of blocks that were reused from cache.
    pub fn last_allocate_had_cache_hit(&self, seq_id: usize) -> usize {
        self.cached_blocks_per_seq
            .get(&seq_id)
            .copied()
            .unwrap_or(0)
    }

    /// Allocate a block, evicting from prefix cache if necessary.
    fn allocate_block_with_eviction(&mut self) -> BlockRef {
        // Try to allocate from free pool first
        if let Some(block_ref) = self.pool.allocate() {
            return block_ref;
        }

        // Need to evict from prefix cache
        // Evicting drops the cache's reference, which may return blocks to the pool
        self.prefix_cacher.evict_blocks(1);

        // Now allocate
        self.pool
            .allocate()
            .expect("Should have free blocks after eviction")
    }

    pub fn can_append_token_to_seq(&self, seq: &impl BlockEngineSequence) -> bool {
        let free_blocks = self.pool.num_free();
        let evictable = self.prefix_cacher.num_evictable_blocks();
        seq.blocks_to_add_new_tok() <= free_blocks + evictable
    }

    /// Free a sequence's blocks and optionally cache them for prefix reuse.
    /// If `logical_blocks` is provided and prefix caching is enabled, full blocks
    /// will be added to the prefix cache for potential reuse by future requests.
    pub fn free_sequence_with_caching(
        &mut self,
        id: usize,
        logical_blocks: Option<&[LogicalTokenBlock]>,
    ) {
        if let Some(block_table) = self.block_tables.remove(&id) {
            let num_cached = self.cached_blocks_per_seq.remove(&id).unwrap_or(0);

            // Cache blocks for prefix reuse (skip already-cached blocks)
            if let Some(logical_blocks) = logical_blocks {
                if self.prefix_cacher.is_enabled() && block_table.len() == logical_blocks.len() {
                    // Insert new blocks into cache (starting after cached blocks)
                    // The cache will clone the BlockRefs, keeping them alive
                    self.prefix_cacher
                        .insert_blocks(logical_blocks, &block_table, num_cached);
                }
            }

            // Release cached blocks' active user counts in the prefix cacher
            if num_cached > 0 {
                if let Some(logical_blocks) = logical_blocks {
                    let cached_logical = &logical_blocks[..num_cached.min(logical_blocks.len())];
                    self.prefix_cacher.release_blocks(cached_logical);
                }
            }

            // The block_table is dropped here, which automatically decrements refcounts.
            // Blocks that are only held by the cache will stay alive.
            // Blocks that are not cached will return to the free pool.
        }
    }

    pub fn free_sequence(&mut self, id: usize) {
        // Free without caching (for aborted sequences or when we don't have logical blocks)
        if let Some(_block_table) = self.block_tables.remove(&id) {
            self.cached_blocks_per_seq.remove(&id);
            // block_table is dropped here, automatically returning blocks to pool
        }
    }

    /// Free a sequence's blocks during preemption.
    /// This properly releases prefix cache active user counts.
    pub fn free_sequence_for_preemption(
        &mut self,
        id: usize,
        logical_blocks: &[LogicalTokenBlock],
    ) {
        if let Some(_block_table) = self.block_tables.remove(&id) {
            let num_cached = self.cached_blocks_per_seq.remove(&id).unwrap_or(0);

            // Release cached blocks' active user counts in the prefix cache
            if num_cached > 0 && self.prefix_cacher.is_enabled() {
                let cached_logical = &logical_blocks[..num_cached.min(logical_blocks.len())];
                self.prefix_cacher.release_blocks(cached_logical);
            }

            // block_table is dropped here, automatically returning blocks to pool
        }
    }

    /// Returns the COW mapping (src, dst).
    /// COW is performed if there are multiple references to the last physical block.
    pub fn append_token_slot_to_seq(
        &mut self,
        sequence: &impl BlockEngineSequence,
    ) -> Option<(usize, usize)> {
        let seq_id = sequence.get_id();
        let logical_len = sequence.logical_token_blocks().len();
        let table_len = match self.block_tables.get(&seq_id) {
            Some(table) => table.len(),
            None => return None,
        };

        // Keep the physical table aligned with the logical block structure to avoid
        // over-allocation (and metadata underflow) when prompts land exactly on
        // block boundaries.
        if table_len > logical_len {
            self.block_tables
                .get_mut(&seq_id)
                .expect("table existence checked above")
                .truncate(logical_len);
        } else if logical_len > table_len {
            let missing = logical_len - table_len;
            let mut new_blocks = Vec::with_capacity(missing);
            for _ in 0..missing {
                new_blocks.push(self.allocate_block_with_eviction());
            }
            self.block_tables
                .get_mut(&seq_id)
                .expect("table existence checked above")
                .extend(new_blocks);
        }

        let blocks_to_add = sequence.blocks_to_add_new_tok();

        match blocks_to_add {
            1 => {
                // Allocate first, then push to table
                let new_block = self.allocate_block_with_eviction();
                self.block_tables.get_mut(&seq_id).unwrap().push(new_block);
                None
            }
            0 => {
                // Get the last block
                let table = self.block_tables.get(&seq_id).unwrap();
                let last_block = table.last().unwrap();

                if !last_block.is_shared() {
                    None
                } else {
                    // We would be writing into shared, so COW.
                    let old_number = last_block.block_id();

                    // Allocate new block
                    let new_block = self.allocate_block_with_eviction();
                    let new_number = new_block.block_id();

                    // Replace in table (old block ref is dropped, decrementing its refcount)
                    let table = self.block_tables.get_mut(&seq_id).unwrap();
                    *table.last_mut().unwrap() = new_block;

                    Some((old_number, new_number))
                }
            }
            _ => {
                unreachable!()
            }
        }
    }

    /// Get prefix cache statistics (hits, misses).
    pub fn prefix_cache_stats(&self) -> (usize, usize) {
        self.prefix_cacher.stats()
    }

    /// Get prefix cache hit rate as a percentage.
    pub fn prefix_cache_hit_rate(&self) -> f64 {
        self.prefix_cacher.hit_rate()
    }

    /// Get number of blocks in prefix cache.
    pub fn prefix_cache_size(&self) -> usize {
        self.prefix_cacher.num_cached_blocks()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_auto_return_to_pool() {
        let pool = BlockPool::new(16, 4);
        assert_eq!(pool.num_free(), 4);

        let block = pool.allocate().unwrap();
        assert_eq!(pool.num_free(), 3);

        drop(block);
        assert_eq!(pool.num_free(), 4); // Block returned automatically
    }

    #[test]
    fn test_block_shared_detection() {
        let pool = BlockPool::new(16, 4);
        let block = pool.allocate().unwrap();

        assert!(!block.is_shared()); // Only one reference

        let clone = block.clone();
        assert!(block.is_shared()); // Now shared
        assert!(clone.is_shared());

        drop(clone);
        assert!(!block.is_shared()); // Back to single reference
    }

    #[test]
    fn test_block_only_freed_when_last_ref_dropped() {
        let pool = BlockPool::new(16, 4);
        let block = pool.allocate().unwrap();
        let clone1 = block.clone();
        let clone2 = block.clone();

        assert_eq!(pool.num_free(), 3);
        assert_eq!(block.ref_count(), 3);

        drop(block);
        assert_eq!(pool.num_free(), 3); // Still held by clones

        drop(clone1);
        assert_eq!(pool.num_free(), 3); // Still held by clone2

        drop(clone2);
        assert_eq!(pool.num_free(), 4); // Now freed
    }

    #[test]
    fn test_pool_exhaustion_and_recovery() {
        let pool = BlockPool::new(16, 2);

        let b1 = pool.allocate().unwrap();
        let b2 = pool.allocate().unwrap();
        assert!(pool.allocate().is_none()); // Pool exhausted

        drop(b1);
        let b3 = pool.allocate(); // Can allocate again
        assert!(b3.is_some());
        assert_eq!(pool.num_free(), 0); // All allocated

        drop(b2);
        drop(b3);
        assert_eq!(pool.num_free(), 2); // All returned
    }

    #[test]
    fn test_block_ids_are_reused() {
        let pool = BlockPool::new(16, 2);

        let b1 = pool.allocate().unwrap();
        let id1 = b1.block_id();
        drop(b1);

        let b2 = pool.allocate().unwrap();
        assert_eq!(b2.block_id(), id1); // Same ID reused (LIFO)
    }
}
