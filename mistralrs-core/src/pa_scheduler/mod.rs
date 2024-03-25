//! The PAScheduler uses a BlockEngine to schedule and automatically batch sequences. The
//! primary method `schedule` returns the batched sequences as inputs, as well as the
//! operations to be executed on the cache by the CacheEngine.

mod backend;
/// The higher-level manager of the blocks allocated. Operations performed by the block engine do
/// not directly change memory.
pub mod block_engine;
/// This is the lower-level manager of the cache. It manages swapping and copying the blocks and
/// actually allocates the KV cache for the CPU and GPU. It is used by the Engine to execute
/// operations issued by the PAScheduler.
pub mod cache_engine;
pub mod sequence;

type CPUBlockFrom = usize;
type GPUBlockFrom = usize;
type CPUBlockTo = usize;
type GPUBlockTo = usize;
type SrcBlockFrom = usize;
type DstBlocksTo = Vec<usize>;

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use self::{
    block_engine::{AllocStatus, BlockEngine},
    cache_engine::CacheConfig,
    sequence::{PASequenceGroup, PASequenceStatus},
};

pub trait ConfigLike {
    fn get_num_kv_heads(&self) -> usize;
    fn get_hidden_size(&self) -> usize;
    fn get_num_hidden_layers(&self) -> usize;
    fn get_num_attention_heads(&self) -> usize;
    fn get_vocab_size(&self) -> usize;
    fn get_sliding_window(&self) -> Option<usize>;
    fn get_head_size(&self) -> usize {
        self.get_hidden_size() / self.get_num_attention_heads()
    }
}

pub struct PASchedulerOutput {
    pub scheduled: Arc<VecDeque<Arc<PASequenceGroup>>>,
    pub blocks_to_swap_in: HashMap<CPUBlockFrom, GPUBlockTo>,
    pub blocks_to_swap_out: HashMap<GPUBlockFrom, CPUBlockTo>,
    pub blocks_to_copy: HashMap<SrcBlockFrom, DstBlocksTo>,
    pub ignored_seq_groups: Arc<VecDeque<Arc<PASequenceGroup>>>,
}

pub struct PASchedulerConfig {
    pub max_num_seqs: usize,
}

pub struct PAScheduler {
    waiting: VecDeque<Arc<PASequenceGroup>>,
    running: VecDeque<Arc<PASequenceGroup>>,
    swapped_out: VecDeque<Arc<PASequenceGroup>>,
    config: PASchedulerConfig,
    pub block_engine: BlockEngine,
}

impl PAScheduler {
    pub fn new(config: PASchedulerConfig, cache_config: &CacheConfig) -> Self {
        assert!(cache_config.fully_init);
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            swapped_out: VecDeque::new(),
            config,
            block_engine: BlockEngine::new(
                cache_config.block_size,
                cache_config.num_gpu_blocks.unwrap(),
                cache_config.num_cpu_blocks.unwrap(),
            ),
        }
    }

    pub fn add_sequence(&mut self, seq_group: PASequenceGroup) {
        self.waiting.push_back(Arc::new(seq_group));
    }

    pub fn schedule(&mut self) -> PASchedulerOutput {
        // If there are no swapped seqs (they have higher priority), add seqs that are in the
        // waiting queue to the running queue.
        if self.swapped_out.is_empty() {
            let mut scheduled = VecDeque::new();
            let mut ignored_seq_groups = VecDeque::new();
            while !self.waiting.is_empty() {
                let seq_group = self.waiting.front().unwrap().clone();

                // If adding this seq means we will have too many, stop as no more could be added.
                if self.config.max_num_seqs
                    == self
                        .running
                        .iter()
                        .map(|group| group.get_seqs().len())
                        .sum::<usize>()
                        + 1
                {
                    break;
                }

                // If we cannot allocate either now or in the future, either do not continue or remove the sequence.
                let can_allocate = self.block_engine.can_allocate(&seq_group);
                match can_allocate {
                    AllocStatus::Later => break, //If we can only allocate later, do not bother iterating over the rest.
                    AllocStatus::Impossible => {
                        eprintln!("Input prompt with length of {} tokens is too long and exceeds capacity of block engine.",
                            seq_group.get_prompt_len()
                        );
                        seq_group.set_status(PASequenceStatus::FinishedIgnored);
                        ignored_seq_groups.push_back(self.waiting.pop_front().unwrap());
                    }
                    _ => {}
                }

                seq_group.set_status(PASequenceStatus::Running);
                self._allocate(&seq_group);

                let seq_group = self.waiting.pop_front().unwrap();
                self.running.push_back(seq_group.clone());
                scheduled.push_back(seq_group);
            }

            // If we did schedule, or we ignored sequences.
            if !scheduled.is_empty() || !ignored_seq_groups.is_empty() {
                return PASchedulerOutput {
                    scheduled: Arc::new(scheduled),
                    blocks_to_swap_in: HashMap::new(),
                    blocks_to_copy: HashMap::new(),
                    blocks_to_swap_out: HashMap::new(),
                    ignored_seq_groups: Arc::new(ignored_seq_groups),
                };
            }
        }

        let mut blocks_to_swap_out = HashMap::new();
        let mut blocks_to_swap_in = HashMap::new();
        let mut blocks_to_copy = HashMap::new();

        // Reserve token slots for the running sequence groups, preempting the lowest (earliest) first.
        // Preempt lowest priority sequences that are in the running queue, forming a
        // new running queue that has the actually running sequences. Remember the preempted
        // sequences, which will be put into the waiting or swapped out state depending on
        // the preemption method (recompute or swap, respectively).

        // Sorts by creation time, in descending order so that earliest are latest (first come first serve).
        self.sort_running_by_priority_fcfs();

        let mut running = VecDeque::new();
        let mut preempted = VecDeque::new();
        while !self.running.is_empty() {
            let seq_group = self.running.pop_front().unwrap();
            let mut finished_with_break = false;
            while !self.block_engine.can_append_token_to_seq(&seq_group) {
                // If we cannot, now we need to preempt some seqs
                if !self.running.is_empty() {
                    // There is something to preempt.
                    let seq_to_preempt = self.running.pop_back().unwrap();
                    self._preempt(seq_to_preempt.clone(), &mut blocks_to_swap_out);
                    preempted.push_back(seq_to_preempt);
                } else {
                    // Nothing to preempt, preempt ourselves. Also, do not bother looking at anything else.
                    self._preempt(seq_group.clone(), &mut blocks_to_swap_out);
                    preempted.push_back(seq_group.clone());
                    finished_with_break = true;
                    break;
                }
            }
            if !finished_with_break {
                // If we need to, append physical blocks for a new token. We do not need to if there is enough space.
                // If we just got preempted, there is no reason to allocate
                self._append_token_slot_to_seq_group(&seq_group, &mut blocks_to_copy);
                running.push_back(seq_group);
            }
        }
        self.running = running;

        // Try to swap in the swapped out sequences and add these to the
        // running state if possible.

        // Sorts by creation time, in descending order so that earliest are latest (first come first serve).
        self.sort_swapped_out_by_priority_fcfs();

        if preempted.is_empty() {
            while !self.swapped_out.is_empty() {
                let seq_group = self.swapped_out.front().unwrap();

                // If the GPU cannot handle the group being swapped in, stop
                if !self.block_engine.can_swap_in_seq_group(seq_group) {
                    break;
                }

                let seq_group = self.swapped_out.pop_front().unwrap();
                // Swap in the blocks
                let to_swap_in = self.block_engine.swap_in(&seq_group);
                blocks_to_swap_in.extend(to_swap_in);
                // Reserve a new slot
                self._append_token_slot_to_seq_group(&seq_group, &mut blocks_to_copy);
                self.running.push_back(seq_group);
            }
        }

        PASchedulerOutput {
            scheduled: self.running.clone().into(),
            blocks_to_swap_in,
            blocks_to_copy,
            blocks_to_swap_out,
            ignored_seq_groups: Arc::new(VecDeque::new()),
        }
    }

    pub fn has_unfinished_sequences(&self) -> bool {
        !self.running.is_empty()
    }

    pub fn free_finished_sequence_groups(&mut self) {
        let mut to_free = Vec::new();
        let clone = self.running.clone();
        self.running = clone
            .iter()
            .filter(|group| {
                if group.is_finished() {
                    to_free.push((*group).clone());
                    false
                } else {
                    true
                }
            })
            .cloned()
            .collect::<VecDeque<_>>();
        for group in to_free {
            self._free(&group);
        }
    }
}

impl PAScheduler {
    fn remove_seq_group(&mut self, seq_group: &PASequenceGroup) {
        // Remove it if it is in waiting
        if let Some(idx) = self
            .waiting
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            self.waiting.remove(idx);
        };
        // Remove it if it is in running
        if let Some(idx) = self
            .running
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            self.running.remove(idx);
        };
        // Remove it if it is in swapped out
        if let Some(idx) = self
            .swapped_out
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            self.swapped_out.remove(idx);
        };
    }
    fn _append_token_slot_to_seq_group(
        &mut self,
        seq_group: &PASequenceGroup,
        blocks_to_copy: &mut HashMap<usize, Vec<usize>>,
    ) {
        for seq in seq_group.get_seqs().values_mut() {
            let op = self.block_engine.append_token_slot_to_seq(seq);
            if let Some((src_block, dst_block)) = op {
                if let std::collections::hash_map::Entry::Vacant(e) =
                    blocks_to_copy.entry(src_block)
                {
                    e.insert(vec![dst_block]);
                } else {
                    blocks_to_copy.get_mut(&src_block).unwrap().push(dst_block);
                }
            }
        }
    }

    fn _abort_seq_group(&mut self, seq_group: &PASequenceGroup) {
        self.remove_seq_group(seq_group);
        seq_group.set_status(PASequenceStatus::FinishedAborted);
        self._free(seq_group);
    }

    /// Preempt either by recomputation (for single sequence), or by swapping (for multiple).
    fn _preempt(
        &mut self,
        seq_group: Arc<PASequenceGroup>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
    ) {
        match seq_group.get_seqs().len() {
            1 => self._preempt_by_recompute(seq_group),
            _ => self._preempt_by_swap(seq_group, blocks_to_swap_out),
        }
    }

    fn _preempt_by_recompute(&mut self, seq_group: Arc<PASequenceGroup>) {
        seq_group.set_status(PASequenceStatus::Waiting);
        self._free(&seq_group);
        self.waiting.push_front(seq_group);
    }

    fn _preempt_by_swap(
        &mut self,
        seq_group: Arc<PASequenceGroup>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
    ) {
        if !self.block_engine.can_swap_out_seq_group(&seq_group) {
            // If we cannot swap it out, abort the sequence group.
            self._abort_seq_group(&seq_group);
            return;
        }
        let new_to_swap = self.block_engine.swap_out(&seq_group);
        blocks_to_swap_out.extend(new_to_swap);
        seq_group.set_status(PASequenceStatus::Swapped);

        self.swapped_out.push_back(seq_group);
    }

    fn _allocate(&mut self, seq_group: &PASequenceGroup) {
        self.block_engine.allocate(seq_group)
    }

    fn _free(&mut self, seq_group: &PASequenceGroup) {
        for seq in seq_group.get_seqs().values() {
            self.block_engine.free_sequence(seq);
        }
    }

    fn sort_running_by_priority_fcfs(&mut self) {
        self.running
            .make_contiguous()
            .sort_by_key(|seq_group| seq_group.arrival_time());
        self.running.make_contiguous().reverse();
    }

    fn sort_swapped_out_by_priority_fcfs(&mut self) {
        self.swapped_out
            .make_contiguous()
            .sort_by_key(|seq_group| seq_group.arrival_time());
        self.swapped_out.make_contiguous().reverse();
    }
}
