//! The Scheduler uses a BlockEngine to schedule and automatically batch sequences. The
//! primary method `schedule` returns the batched sequences as inputs, as well as the
//! operations to be executed on the cache by the CacheEngine.

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

use tracing::warn;

use crate::{
    paged_attention::BlockEngine,
    sequence::{Sequence, SequenceState},
};

use super::{block_engine::AllocStatus, BlockEngineSequence, CacheConfig};

pub struct SchedulerOutput {
    pub scheduled: Vec<Arc<Sequence>>,
    pub blocks_to_swap_in: HashMap<CPUBlockFrom, GPUBlockTo>,
    pub blocks_to_swap_out: HashMap<GPUBlockFrom, CPUBlockTo>,
    pub blocks_to_copy: HashMap<SrcBlockFrom, DstBlocksTo>,
}

pub struct SchedulerConfig {
    pub max_num_seqs: usize,
}

pub struct Scheduler {
    waiting: VecDeque<Arc<Sequence>>,
    running: VecDeque<Arc<Sequence>>,
    swapped_out: VecDeque<Arc<Sequence>>,
    config: SchedulerConfig,
    pub block_engine: BlockEngine,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, cache_config: &CacheConfig) -> Self {
        assert!(cache_config.fully_init);
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            swapped_out: VecDeque::new(),
            config,
            block_engine: BlockEngine::new(
                cache_config.block_size,
                cache_config.num_gpu_blocks,
                cache_config.num_cpu_blocks,
            ),
        }
    }

    pub fn add_sequence(&mut self, seq: Sequence) {
        self.waiting.push_back(Arc::new(seq));
    }

    pub fn schedule(&mut self) -> SchedulerOutput {
        // If there are no swapped seqs (they have higher priority), add seqs that are in the
        // waiting queue to the running queue.
        if self.swapped_out.is_empty() {
            let mut scheduled = VecDeque::new();
            let mut did_ignore = false;
            while !self.waiting.is_empty() {
                let seq = self.waiting.front().unwrap().clone();

                // If adding this seq means we will have too many, stop as no more could be added.
                if self.config.max_num_seqs == self.running.iter().count() + 1 {
                    break;
                }

                // If we cannot allocate either now or in the future, either do not continue or remove the sequence.
                let can_allocate = self.block_engine.can_allocate(&*seq);
                match can_allocate {
                    AllocStatus::Later => break, // If we can only allocate later, do not bother iterating over the rest.
                    AllocStatus::Impossible => {
                        warn!(
                            "Input prompt with length of {} tokens is too long and exceeds capacity of block engine. Sequence will be ignored.",
                            seq.prompt_tokens()
                        );
                        seq.set_state(SequenceState::FinishedIgnored);
                        did_ignore = true;
                    }
                    _ => {}
                }

                seq.set_state(SequenceState::RunningPrompt);
                self._allocate(&seq);

                let seq = self.waiting.pop_front().unwrap();
                scheduled.push_back(seq);
            }

            // If we did schedule, or we ignored sequences.
            if !scheduled.is_empty() || did_ignore {
                return SchedulerOutput {
                    scheduled: scheduled.into(),
                    blocks_to_swap_in: HashMap::new(),
                    blocks_to_copy: HashMap::new(),
                    blocks_to_swap_out: HashMap::new(),
                };
            }

            // Add them to the running otherwise
            self.running.extend(scheduled);
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
        let mut did_preempt = false;
        while !self.running.is_empty() {
            let seq = self.running.pop_front().unwrap();
            let mut finished_with_break = false;
            while !self.block_engine.can_append_token_to_seq(&*seq) {
                // If we cannot, now we need to preempt some seqs
                if !self.running.is_empty() {
                    // There is something to preempt.
                    let seq_to_preempt = self.running.pop_back().unwrap();
                    self._preempt(seq_to_preempt, &mut blocks_to_swap_out);
                    did_preempt = true;
                } else {
                    // Nothing to preempt, preempt ourselves. Also, do not bother looking at anything else.
                    self._preempt(seq.clone(), &mut blocks_to_swap_out);
                    did_preempt = true;
                    finished_with_break = true;
                    break;
                }
            }
            if !finished_with_break {
                // If we need to, append physical blocks for a new token. We do not need to if there is enough space.
                // If we just got preempted, there is no reason to allocate
                self._append_token_slot_to_seq(&seq, &mut blocks_to_copy);
                running.push_back(seq);
            }
        }
        self.running = running;

        // Try to swap in the swapped out sequences and add these to the
        // running state if possible.

        // Sorts by creation time, in descending order so that earliest are latest (first come first serve).
        self.sort_swapped_out_by_priority_fcfs();

        if !did_preempt {
            while !self.swapped_out.is_empty() {
                let seq = self.swapped_out.front().unwrap();

                // If the GPU cannot handle the group being swapped in, stop
                if !self.block_engine.can_swap_in_seq(&**seq) {
                    break;
                }

                let seq = self.swapped_out.pop_front().unwrap();
                // Swap in the blocks
                let to_swap_in = self.block_engine.swap_in(&*seq);
                blocks_to_swap_in.extend(to_swap_in);
                // Reserve a new slot
                self._append_token_slot_to_seq(&seq, &mut blocks_to_copy);
                self.running.push_back(seq);
            }
        }

        SchedulerOutput {
            scheduled: self.running.clone().into(), // Clone should be cheap.
            blocks_to_swap_in,
            blocks_to_copy,
            blocks_to_swap_out,
        }
    }

    pub fn has_unfinished_sequences(&self) -> bool {
        !self.running.is_empty() || !self.waiting.is_empty()
    }

    pub fn free_finished_sequence_groups(&mut self) {
        let mut to_free_ids = Vec::new();
        self.running.retain(|seq| {
            if seq.is_finished_paged_attn() {
                to_free_ids.push(seq.get_id());
                false
            } else {
                true
            }
        });
        for id in to_free_ids {
            self._free(id);
        }
    }
}

impl Scheduler {
    fn remove_seq(&mut self, seq: &Sequence) {
        // Remove it if it is in waiting
        if let Some(idx) = self
            .waiting
            .iter()
            .position(|other| other.get_id() == seq.get_id())
        {
            self.waiting.remove(idx);
        };
        // Remove it if it is in running
        if let Some(idx) = self
            .running
            .iter()
            .position(|other| other.get_id() == seq.get_id())
        {
            self.running.remove(idx);
        };
        // Remove it if it is in swapped out
        if let Some(idx) = self
            .swapped_out
            .iter()
            .position(|other| other.get_id() == seq.get_id())
        {
            self.swapped_out.remove(idx);
        };
    }
    fn _append_token_slot_to_seq(
        &mut self,
        seq: &Sequence,
        blocks_to_copy: &mut HashMap<usize, Vec<usize>>,
    ) {
        let op = self.block_engine.append_token_slot_to_seq(seq);
        if let Some((src_block, dst_block)) = op {
            if let std::collections::hash_map::Entry::Vacant(e) = blocks_to_copy.entry(src_block) {
                e.insert(vec![dst_block]);
            } else {
                blocks_to_copy.get_mut(&src_block).unwrap().push(dst_block);
            }
        }
    }

    fn _abort_seq(&mut self, seq: &Sequence) {
        self.remove_seq(seq);
        seq.set_state(SequenceState::FinishedAborted);
        self._free(seq.get_id());
    }

    /// Preempt either by recomputation (for single sequence), or by swapping (for multiple).
    fn _preempt(&mut self, seq: Arc<Sequence>, blocks_to_swap_out: &mut HashMap<usize, usize>) {
        self._preempt_by_recompute(seq)
    }

    fn _preempt_by_recompute(&mut self, seq: Arc<Sequence>) {
        seq.set_state(SequenceState::Waiting);
        self._free(seq.get_id());
        self.waiting.push_front(seq);
    }

    fn _preempt_by_swap(
        &mut self,
        seq: Arc<Sequence>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
    ) {
        if !self.block_engine.can_swap_out_seq(&*seq) {
            // If we cannot swap it out, abort the sequence group.
            self._abort_seq(&seq);
            return;
        }
        let new_to_swap = self.block_engine.swap_out(&*seq);
        blocks_to_swap_out.extend(new_to_swap);
        seq.set_state(SequenceState::Swapped);

        self.swapped_out.push_back(seq);
    }

    fn _allocate(&mut self, seq: &Sequence) {
        self.block_engine.allocate(seq)
    }

    fn _free(&mut self, seq_id: usize) {
        self.block_engine.free_sequence(seq_id);
    }

    fn sort_running_by_priority_fcfs(&mut self) {
        self.running
            .make_contiguous()
            .sort_by_key(|seq| seq.timestamp());
        self.running.make_contiguous().reverse();
    }

    fn sort_swapped_out_by_priority_fcfs(&mut self) {
        self.swapped_out
            .make_contiguous()
            .sort_by_key(|seq| seq.timestamp());
        self.swapped_out.make_contiguous().reverse();
    }
}
