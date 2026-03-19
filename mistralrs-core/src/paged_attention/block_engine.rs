use std::{
    collections::HashMap,
    hash::Hash,
    marker::PhantomData,
    ops::Deref,
    sync::Arc,
};
use parking_lot::{Mutex, MutexGuard};

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

#[derive(Hash, PartialEq, Eq)]
pub struct _PhysicalTokenBlock {
    pub block_id: usize,
    block_size: usize,
    refcount: usize,
    is_gpu: bool,
}

impl _PhysicalTokenBlock {
    pub fn refcount(&self) -> usize {
        self.refcount
    }
    pub fn increment_refcount(&mut self) {
        self.refcount += 1;
    }
    pub fn decrement_refcount(&mut self) {
        assert!(self.refcount >= 1);
        self.refcount -= 1;
    }
}

pub struct PhysicalTokenBlock(pub Mutex<_PhysicalTokenBlock>);

impl std::fmt::Debug for PhysicalTokenBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.0.lock();
        f
            .debug_struct("PhysicalTokenBlock")
            .field("block_id", &inner.block_id)
            .field("block_size", &inner.block_size)
            .field("refcount", &inner.refcount)
            .field("is_gpu", &inner.is_gpu)
            .finish()
    }
}

impl PhysicalTokenBlock {
    pub fn deref_mut(&self) -> MutexGuard<'_, _PhysicalTokenBlock> {
        loop {
            if let Some(v) = self.0.try_lock() {
                return v;
            }
        }
    }
}

impl PartialEq for PhysicalTokenBlock {
    fn eq(&self, other: &Self) -> bool {
        *self.deref_mut() == *other.deref_mut()
    }
}

impl Hash for PhysicalTokenBlock {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.deref_mut().hash(state)
    }
}

impl Eq for PhysicalTokenBlock {}

type BlockTable = Vec<Arc<PhysicalTokenBlock>>;
struct GPUAllocator;

struct GPUAllocatorWrapper(usize);
impl Deref for GPUAllocatorWrapper {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct Allocator<T> {
    free_blocks: BlockTable,
    _ghost: PhantomData<T>,
}

impl<T> Allocator<T> {
    fn allocate(&mut self) -> Arc<PhysicalTokenBlock> {
        let block = self.free_blocks.pop().unwrap();
        block.deref_mut().refcount = 1;
        block
    }

    fn free_block(&mut self, block: Arc<PhysicalTokenBlock>) {
        if block.deref_mut().refcount == 0 {
            panic!(
                "PhysicalTokenBlock with id {} experienced a double free!",
                block.deref_mut().block_id
            );
        }
        block.deref_mut().refcount -= 1;
        if block.deref_mut().refcount == 0 {
            self.free_blocks.push(block);
        }
    }
}

impl Allocator<GPUAllocator> {
    fn new(block_size: usize, num_blocks: usize) -> Self {
        let mut free_blocks = Vec::new();
        for id in 0..num_blocks {
            free_blocks.push(Arc::new(PhysicalTokenBlock(Mutex::new(
                _PhysicalTokenBlock {
                    block_id: id,
                    block_size,
                    refcount: 0,
                    is_gpu: true,
                },
            ))))
        }
        Allocator {
            free_blocks,
            _ghost: PhantomData,
        }
    }

    fn get_num_free_blocks(&self) -> GPUAllocatorWrapper {
        GPUAllocatorWrapper(self.free_blocks.len())
    }
}

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
    gpu_allocator: Allocator<GPUAllocator>,
    pub block_tables: HashMap<SeqID, BlockTable>,
    /// Prefix cache for reusing KV cache blocks across requests with shared prefixes.
    prefix_cacher: PrefixCacher,
    /// Track number of cached blocks used per sequence (for freeing).
    cached_blocks_per_seq: HashMap<SeqID, usize>,
}

pub type BlockTables = HashMap<usize, BlockTable>;

impl BlockEngine {
    #[must_use]
    pub fn new(block_size: usize, num_gpu_blocks: usize, prefix_caching_enabled: bool) -> Self {
        Self {
            num_gpu_blocks,
            block_size,
            gpu_allocator: Allocator::<GPUAllocator>::new(block_size, num_gpu_blocks),
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
        let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();

        // Also count evictable blocks from prefix cache as potentially available
        let num_evictable = self.prefix_cacher.num_evictable_blocks();
        let total_available = *num_free_gpu_blocks + num_evictable;

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
            let mut block_table = physical_blocks_prefill.clone();
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

        // Use cached blocks for the prefix
        for (idx, physical_block) in cached_blocks {
            // Extend block_table to the right size
            while block_table.len() < idx {
                block_table.push(self.allocate_block_with_eviction());
            }
            // The cached block already has its refcount incremented by match_prefix
            block_table.push(physical_block);
        }

        // Allocate new blocks for the rest
        for _ in block_table.len()..num_blocks_needed {
            block_table.push(self.allocate_block_with_eviction());
        }

        self.cached_blocks_per_seq.insert(seq_id, num_cached);
        self.block_tables.insert(seq_id, block_table);

        // Calculate number of cached tokens (full blocks only)
        // num_cached is the number of full blocks that were cache hits
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
    fn allocate_block_with_eviction(&mut self) -> Arc<PhysicalTokenBlock> {
        // Try to allocate from free pool first
        if *self.gpu_allocator.get_num_free_blocks() > 0 {
            return self.gpu_allocator.allocate();
        }

        // Need to evict from prefix cache
        let evicted = self.prefix_cacher.evict_blocks(1);
        for block in evicted {
            // Decrement refcount and return to free pool
            block.deref_mut().decrement_refcount();
            if block.deref_mut().refcount == 0 {
                self.gpu_allocator.free_blocks.push(block);
            }
        }

        // Now allocate
        self.gpu_allocator.allocate()
    }

    pub fn can_append_token_to_seq(&self, seq: &impl BlockEngineSequence) -> bool {
        let free_blocks = self.gpu_allocator.get_num_free_blocks();
        let evictable = self.prefix_cacher.num_evictable_blocks();
        // Physical blocks = logical blocks
        seq.blocks_to_add_new_tok() <= *free_blocks + evictable
    }

    /// Free a sequence's blocks and optionally cache them for prefix reuse.
    /// If `logical_blocks` is provided and prefix caching is enabled, full blocks
    /// will be added to the prefix cache for potential reuse by future requests.
    pub fn free_sequence_with_caching(
        &mut self,
        id: usize,
        logical_blocks: Option<&[LogicalTokenBlock]>,
    ) {
        // Handle double free if run out of tokens
        if let Some(block_table) = self.block_tables.remove(&id) {
            let num_cached = self.cached_blocks_per_seq.remove(&id).unwrap_or(0);

            // Cache blocks for prefix reuse (skip already-cached blocks)
            if let Some(logical_blocks) = logical_blocks {
                if self.prefix_cacher.is_enabled() && block_table.len() == logical_blocks.len() {
                    // Insert new blocks into cache (starting after cached blocks)
                    self.prefix_cacher
                        .insert_blocks(logical_blocks, &block_table, num_cached);
                }
            }

            // Release cached blocks' reference counts
            if num_cached > 0 {
                if let Some(logical_blocks) = logical_blocks {
                    let cached_logical = &logical_blocks[..num_cached.min(logical_blocks.len())];
                    self.prefix_cacher.release_blocks(cached_logical);
                }
            }

            // Free non-cached blocks (or all if not caching)
            for (idx, block) in block_table.iter().enumerate() {
                // Skip blocks that were from cache and are now in cache
                // (they have refcount > 1 due to cache holding a reference)
                if idx < num_cached && self.prefix_cacher.is_enabled() {
                    // This was a cached block - just decrement our reference
                    self.gpu_allocator.free_block(block.clone());
                } else if self.prefix_cacher.is_enabled() && logical_blocks.is_some() {
                    // This block is being added to cache, so cache holds the ref
                    // We don't free it to the allocator
                } else {
                    self.gpu_allocator.free_block(block.clone());
                }
            }
        }
    }

    pub fn free_sequence(&mut self, id: usize) {
        // Free without caching (for aborted sequences or when we don't have logical blocks)
        if let Some(block_table) = self.block_tables.remove(&id) {
            self.cached_blocks_per_seq.remove(&id);
            // Free all blocks
            for block in block_table.iter() {
                self.gpu_allocator.free_block(block.clone());
            }
        }
    }

    /// Free a sequence's blocks during preemption.
    /// This properly releases prefix cache refs so cached blocks can be evicted.
    pub fn free_sequence_for_preemption(
        &mut self,
        id: usize,
        logical_blocks: &[LogicalTokenBlock],
    ) {
        if let Some(block_table) = self.block_tables.remove(&id) {
            let num_cached = self.cached_blocks_per_seq.remove(&id).unwrap_or(0);

            // Release cached blocks' reference counts in the prefix cache
            if num_cached > 0 && self.prefix_cacher.is_enabled() {
                let cached_logical = &logical_blocks[..num_cached.min(logical_blocks.len())];
                self.prefix_cacher.release_blocks(cached_logical);
            }

            // Free all blocks
            for block in block_table.iter() {
                self.gpu_allocator.free_block(block.clone());
            }
        }
    }

    // Returns the COW mapping (src, dst).
    // COW is performed if there are multiple references to the last physical block.
    pub fn append_token_slot_to_seq(
        &mut self,
        sequence: &impl BlockEngineSequence,
    ) -> Option<(usize, usize)> {
        let seq_id = sequence.get_id();
        let blocks_to_add = sequence.blocks_to_add_new_tok();

        // Check if table exists
        if !self.block_tables.contains_key(&seq_id) {
            return None;
        }

        match blocks_to_add {
            1 => {
                // Allocate first, then push to table
                let new_block = self.allocate_block_with_eviction();
                self.block_tables.get_mut(&seq_id).unwrap().push(new_block);
                None
            }
            0 => {
                // Get the last block's info first
                let table = self.block_tables.get(&seq_id).unwrap();
                let last_block = table.last().unwrap();
                let is_gpu = last_block.deref_mut().is_gpu;
                let refcount = last_block.deref_mut().refcount;

                assert!(is_gpu);

                if refcount == 1 {
                    None
                } else {
                    // We would be writing into shared, so COW.
                    let old_block = last_block.clone();
                    let old_number = old_block.deref_mut().block_id;

                    // Now allocate and mutate
                    let new_block = self.allocate_block_with_eviction();
                    let new_number = new_block.deref_mut().block_id;

                    // Free old block
                    self.gpu_allocator.free_block(old_block);

                    // Replace in table
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
