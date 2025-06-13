use std::{
    collections::{hash_map::Entry, HashMap},
    hash::Hash,
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, Mutex, MutexGuard},
};

use super::block_engine_sequence::BlockEngineSequence;

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
        match self.0.lock() {
            Ok(inner) => f
                .debug_struct("PhysicalTokenBlock")
                .field("block_id", &inner.block_id)
                .field("block_size", &inner.block_size)
                .field("refcount", &inner.refcount)
                .field("is_gpu", &inner.is_gpu)
                .finish(),
            Err(_) => write!(f, "PhysicalTokenBlock(<locked>)"),
        }
    }
}

impl PhysicalTokenBlock {
    pub fn deref_mut(&self) -> MutexGuard<'_, _PhysicalTokenBlock> {
        loop {
            if let Ok(v) = self.0.try_lock() {
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
struct CPUAllocator;

struct GPUAllocatorWrapper(usize);
// struct CPUAllocatorWrapper(usize);

impl Deref for GPUAllocatorWrapper {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// impl Deref for CPUAllocatorWrapper {
//     type Target = usize;

//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

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

impl Allocator<CPUAllocator> {
    fn new(block_size: usize, num_blocks: usize) -> Self {
        let mut free_blocks = Vec::new();
        for id in 0..num_blocks {
            free_blocks.push(Arc::new(PhysicalTokenBlock(Mutex::new(
                _PhysicalTokenBlock {
                    block_id: id,
                    block_size,
                    refcount: 0,
                    is_gpu: false,
                },
            ))))
        }
        Allocator {
            free_blocks,
            _ghost: PhantomData,
        }
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
    cpu_allocator: Allocator<CPUAllocator>,
    pub block_tables: HashMap<SeqID, BlockTable>,
}

pub type BlockTables = HashMap<usize, BlockTable>;

impl BlockEngine {
    #[must_use]
    pub fn new(block_size: usize, num_gpu_blocks: usize, num_cpu_blocks: usize) -> Self {
        Self {
            num_gpu_blocks,
            block_size,
            gpu_allocator: Allocator::<GPUAllocator>::new(block_size, num_gpu_blocks),
            cpu_allocator: Allocator::<CPUAllocator>::new(block_size, num_cpu_blocks),
            block_tables: HashMap::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn can_allocate(&self, seq: &mut impl BlockEngineSequence) -> AllocStatus {
        let num_required_blocks = seq.logical_token_blocks().len();
        let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();

        if self.num_gpu_blocks < num_required_blocks {
            AllocStatus::Impossible
        } else if *num_free_gpu_blocks < num_required_blocks {
            AllocStatus::Later {
                waitlisted_count: seq.increment_waitlist_count(),
            }
        } else {
            AllocStatus::Ok
        }
    }

    pub fn allocate(&mut self, seq: &mut impl BlockEngineSequence) {
        // If there are prefill physical blocks, use those here.
        if let Some(physical_blocks_prefill) = seq.take_physical_blocks_prefill() {
            let mut block_table = physical_blocks_prefill.clone();
            let n_extra_blocks = seq.logical_token_blocks().len() - block_table.len();
            for _ in 0..n_extra_blocks {
                block_table.push(self.gpu_allocator.allocate());
            }
            self.block_tables.insert(seq.get_id(), block_table.clone());
        } else {
            let mut block_table = Vec::new();
            for _logcical_idx in 0..seq.logical_token_blocks().len() {
                block_table.push(self.gpu_allocator.allocate());
            }
            self.block_tables.insert(seq.get_id(), block_table.clone());
        }
    }

    pub fn can_append_token_to_seq(&self, seq: &impl BlockEngineSequence) -> bool {
        let free_blocks = self.gpu_allocator.get_num_free_blocks();
        // Physical blocks = logical blocks
        seq.blocks_to_add_new_tok() <= *free_blocks
    }

    pub fn free_sequence(&mut self, id: usize) {
        // Handle double free if run out of tokens
        if let Some(block_table) = self.block_tables.get(&id) {
            // Free from block table
            for block in block_table {
                if block.deref_mut().is_gpu {
                    self.gpu_allocator.free_block(block.clone())
                } else {
                    self.cpu_allocator.free_block(block.clone())
                }
            }

            self.block_tables.remove(&id);
        }
    }

    #[allow(dead_code)]
    pub fn can_swap_out_seq(&self, seq: &impl BlockEngineSequence) -> bool {
        let blocks_required: usize = self
            .block_tables
            .iter()
            .filter(|(id, _)| seq.get_id() == **id)
            .map(|(_, table)| table.len())
            .sum();
        blocks_required <= self.cpu_allocator.free_blocks.len()
    }

    /// Update the block table so that the sequence does no longer reserve any GPU
    /// physical blocks, and only has CPU physical blocks.
    #[allow(dead_code)]
    pub fn swap_out(&mut self, seq: &impl BlockEngineSequence) -> HashMap<usize, usize> {
        // GPU block to a CPU block
        let mut new_mapping = HashMap::new();
        let seq_id = seq.get_id();

        let mut new_block_table = Vec::new();
        let block_table = self.block_tables.get(&seq_id).unwrap();

        for gpu_block in block_table {
            let cpu_block =
                if let Entry::Vacant(e) = new_mapping.entry(gpu_block.deref_mut().block_id) {
                    // Create a new block
                    let cpu_block = self.cpu_allocator.allocate();
                    e.insert(cpu_block.clone());
                    cpu_block
                } else {
                    // Reuse a block
                    let cpu_block = new_mapping
                        .get(&gpu_block.deref_mut().block_id)
                        .unwrap()
                        .clone();
                    cpu_block.deref_mut().refcount += 1;
                    cpu_block
                };
            new_block_table.push(cpu_block);
            self.gpu_allocator.free_block(gpu_block.clone());
        }
        self.block_tables.insert(seq_id, new_block_table);

        new_mapping
            .iter()
            .map(|(k, v)| (*k, v.deref_mut().block_id))
            .collect::<HashMap<_, _>>()
    }

    // Returns the COW mapping (src, dst).
    // COW is performed if there are multiple references to the last physical block.
    pub fn append_token_slot_to_seq(
        &mut self,
        sequence: &impl BlockEngineSequence,
    ) -> Option<(usize, usize)> {
        let table = self.block_tables.get_mut(&sequence.get_id())?;

        match sequence.blocks_to_add_new_tok() {
            1 => {
                table.push(self.gpu_allocator.allocate());
                None
            }
            0 => {
                let last_block = table.last_mut().unwrap();
                assert!(last_block.deref_mut().is_gpu);
                if last_block.deref_mut().refcount == 1 {
                    None
                } else {
                    // We would be writing into shared, so COW.
                    let new_block = self.gpu_allocator.allocate();
                    self.gpu_allocator.free_block(last_block.clone());
                    let old_number = last_block.deref_mut().block_id;
                    let new_number = new_block.deref_mut().block_id;
                    *last_block = new_block;
                    Some((old_number, new_number))
                }
            }
            _ => {
                unreachable!()
            }
        }
    }

    pub fn can_swap_in_seq(&self, seq: &impl BlockEngineSequence) -> bool {
        let blocks_required: usize = self
            .block_tables
            .iter()
            .filter(|(id, _)| seq.get_id() == **id)
            .map(|(_, table)| table.len())
            .sum();
        blocks_required <= self.gpu_allocator.free_blocks.len()
    }

    /// Update the block table so that the sequence does no longer reserve any CPU
    /// physical blocks, and only has GPU physical blocks.
    pub fn swap_in(&mut self, seq: &impl BlockEngineSequence) -> HashMap<usize, usize> {
        // CPU block to a GPU block
        let mut new_mapping = HashMap::new();
        let seq_id = seq.get_id();

        let mut new_block_table = Vec::new();
        let block_table = self.block_tables.get(&seq_id).unwrap();

        for cpu_block in block_table {
            let gpu_block =
                if let Entry::Vacant(e) = new_mapping.entry(cpu_block.deref_mut().block_id) {
                    // Create a new block
                    let gpu_block = self.cpu_allocator.allocate();
                    e.insert(gpu_block.clone());
                    gpu_block
                } else {
                    // Reuse a block
                    let gpu_block = new_mapping
                        .get(&cpu_block.deref_mut().block_id)
                        .unwrap()
                        .clone();
                    gpu_block.deref_mut().refcount += 1;
                    gpu_block
                };
            new_block_table.push(gpu_block);
            self.gpu_allocator.free_block(cpu_block.clone());
        }
        self.block_tables.insert(seq_id, new_block_table);

        new_mapping
            .iter()
            .map(|(k, v)| (*k, v.deref_mut().block_id))
            .collect::<HashMap<_, _>>()
    }
}
