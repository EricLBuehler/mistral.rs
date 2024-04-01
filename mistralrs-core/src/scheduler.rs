use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    rc::Rc,
};

use crate::{
    deref_refcell,
    pa::block_engine::{AllocStatus, BlockEngine},
    sequence::{SequenceGroup, SequenceState},
};

type CPUBlockFrom = usize;
type GPUBlockFrom = usize;
type CPUBlockTo = usize;
type GPUBlockTo = usize;
type SrcBlockFrom = usize;
type DstBlocksTo = Vec<usize>;

pub struct SchedulerOutput {
    pub scheduled: Box<[Rc<RefCell<SequenceGroup>>]>,
    pub blocks_to_swap_in: HashMap<CPUBlockFrom, GPUBlockTo>,
    pub blocks_to_swap_out: HashMap<GPUBlockFrom, CPUBlockTo>,
    pub blocks_to_copy: HashMap<SrcBlockFrom, DstBlocksTo>,
    pub ignored_seq_groups: Box<[Rc<RefCell<SequenceGroup>>]>,
}

pub struct Scheduler {
    waiting: VecDeque<Rc<RefCell<SequenceGroup>>>,
    running: VecDeque<Rc<RefCell<SequenceGroup>>>,
    swapped_out: VecDeque<Rc<RefCell<SequenceGroup>>>,
    pub block_engine: BlockEngine,
    pub max_num_seqs: usize,
}

impl Scheduler {
    pub fn new(
        max_num_seqs: usize,
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> Self {
        Self {
            running: VecDeque::new(),
            waiting: VecDeque::new(),
            swapped_out: VecDeque::new(),
            block_engine: BlockEngine::new(block_size, num_gpu_blocks, num_cpu_blocks),
            max_num_seqs,
        }
    }

    pub fn add_seq(&mut self, seq: Rc<RefCell<SequenceGroup>>) {
        self.waiting.push_back(seq)
    }

    /// Schedule all sequences based on their state and the available space.
    pub fn schedule(&mut self) -> SchedulerOutput {
        // If there are no swapped seqs (they have higher priority), add seqs that are in the
        // waiting queue to the running queue.
        if self.swapped_out.is_empty() {
            let mut scheduled = VecDeque::new();
            let mut ignored_seq_groups = VecDeque::new();
            while !self.waiting.is_empty() {
                let seq_group = self.waiting.front().unwrap().clone();

                // If adding this seq means we will have too many, stop as no more could be added.
                if self.max_num_seqs
                    == self
                        .running
                        .iter()
                        .map(|group| deref_refcell!(group).get_seqs().len())
                        .sum::<usize>()
                        + 1
                {
                    break;
                }

                // If we cannot allocate either now or in the future, either do not continue or remove the sequence.
                let can_allocate = self.block_engine.can_allocate(&*deref_refcell!(seq_group));
                match can_allocate {
                    AllocStatus::Later => break, //If we can only allocate later, do not bother iterating over the rest.
                    AllocStatus::Impossible => {
                        eprintln!("Input prompt with length of {} tokens is too long and exceeds capacity of block engine.",
                            deref_refcell!(seq_group).get_prompt_len()
                        );
                        deref_refcell!(seq_group).set_state(SequenceState::DoneIgnored);
                        ignored_seq_groups.push_back(self.waiting.pop_front().unwrap());
                    }
                    _ => {}
                }

                deref_refcell!(seq_group).set_state(SequenceState::Running);
                self._allocate(&*deref_refcell!(seq_group));

                let seq_group = self.waiting.pop_front().unwrap();
                self.running.push_back(seq_group.clone());
                scheduled.push_back(seq_group);
            }

            // If we did schedule, or we ignored sequences.
            if !scheduled.is_empty() || !ignored_seq_groups.is_empty() {
                return SchedulerOutput {
                    scheduled: scheduled.iter().cloned().collect::<Vec<_>>().into(),
                    blocks_to_swap_in: HashMap::new(),
                    blocks_to_copy: HashMap::new(),
                    blocks_to_swap_out: HashMap::new(),
                    ignored_seq_groups: ignored_seq_groups
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .into(),
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
            while !self
                .block_engine
                .can_append_token_to_seq(&*deref_refcell!(seq_group))
            {
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
                self._append_token_slot_to_seq_group(
                    &*deref_refcell!(seq_group),
                    &mut blocks_to_copy,
                );
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
                if !self
                    .block_engine
                    .can_swap_in_seq_group(&*deref_refcell!(seq_group))
                {
                    break;
                }

                let seq_group = self.swapped_out.pop_front().unwrap();
                // Swap in the blocks
                let to_swap_in = self.block_engine.swap_in(&*deref_refcell!(seq_group));
                blocks_to_swap_in.extend(to_swap_in);
                // Reserve a new slot
                self._append_token_slot_to_seq_group(
                    &*deref_refcell!(seq_group),
                    &mut blocks_to_copy,
                );
                self.running.push_back(seq_group);
            }
        }

        SchedulerOutput {
            scheduled: self.running.iter().cloned().collect::<Vec<_>>().into(),
            blocks_to_swap_in,
            blocks_to_copy,
            blocks_to_swap_out,
            ignored_seq_groups: Vec::new().into(),
        }
    }

    pub fn free_finished_sequence_groups(&mut self) {
        let mut to_free = Vec::new();
        let clone = self.running.clone();
        self.running = clone
            .iter()
            .filter(|group| {
                if deref_refcell!(group).is_finished() {
                    to_free.push((*group).clone());
                    false
                } else {
                    true
                }
            })
            .cloned()
            .collect::<VecDeque<_>>();
        for group in to_free {
            self._free(group);
        }
    }
}

impl Scheduler {
    fn remove_seq_group(&mut self, seq_group: &SequenceGroup) {
        // Remove it if it is in waiting
        if let Some(idx) = self
            .waiting
            .iter()
            .position(|grp| deref_refcell!(grp).id() == seq_group.id())
        {
            self.waiting.remove(idx);
        };
        // Remove it if it is in running
        if let Some(idx) = self
            .running
            .iter()
            .position(|grp| deref_refcell!(grp).id() == seq_group.id())
        {
            self.running.remove(idx);
        };
        // Remove it if it is in swapped out
        if let Some(idx) = self
            .swapped_out
            .iter()
            .position(|grp| deref_refcell!(grp).id() == seq_group.id())
        {
            self.swapped_out.remove(idx);
        };
    }

    fn _append_token_slot_to_seq_group(
        &mut self,
        seq_group: &SequenceGroup,
        blocks_to_copy: &mut HashMap<usize, Vec<usize>>,
    ) {
        for seq in seq_group.get_seqs().values() {
            let op = self
                .block_engine
                .append_token_slot_to_seq(&*deref_refcell!(seq));
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

    fn _abort_seq_group(&mut self, seq_group: Rc<RefCell<SequenceGroup>>) {
        self.remove_seq_group(&*deref_refcell!(seq_group));
        deref_refcell!(seq_group).set_state(SequenceState::DoneAborted);
        self._free(seq_group);
    }

    /// Preempt either by recomputation (for single sequence), or by swapping (for multiple).
    fn _preempt(
        &mut self,
        seq_group: Rc<RefCell<SequenceGroup>>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
    ) {
        let len = deref_refcell!(seq_group).get_seqs().len();
        match len {
            1 => self._preempt_by_recompute(seq_group),
            _ => self._preempt_by_swap(seq_group, blocks_to_swap_out),
        }
    }

    fn _preempt_by_recompute(&mut self, seq_group: Rc<RefCell<SequenceGroup>>) {
        deref_refcell!(seq_group).set_state(SequenceState::Waiting);
        self._free(seq_group.clone());
        self.waiting.push_front(seq_group);
    }

    fn _preempt_by_swap(
        &mut self,
        seq_group: Rc<RefCell<SequenceGroup>>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
    ) {
        if !self
            .block_engine
            .can_swap_out_seq_group(&*deref_refcell!(seq_group))
        {
            // If we cannot swap it out, abort the sequence group.
            self._abort_seq_group(seq_group);
            return;
        }
        let new_to_swap = self.block_engine.swap_out(&*deref_refcell!(seq_group));
        blocks_to_swap_out.extend(new_to_swap);
        deref_refcell!(seq_group).set_state(SequenceState::Swapped);

        self.swapped_out.push_back(seq_group);
    }

    fn _allocate(&mut self, seq_group: &SequenceGroup) {
        self.block_engine.allocate(seq_group)
    }

    fn _free(&mut self, seq_group: Rc<RefCell<SequenceGroup>>) {
        for seq in deref_refcell!(seq_group).get_seqs().values() {
            self.block_engine.free_sequence(&*deref_refcell!(seq));
        }
    }

    fn sort_running_by_priority_fcfs(&mut self) {
        self.running
            .make_contiguous()
            .sort_by_key(|seq_group| deref_refcell!(seq_group).timestamp());
        self.running.make_contiguous().reverse();
    }

    fn sort_swapped_out_by_priority_fcfs(&mut self) {
        self.swapped_out
            .make_contiguous()
            .sort_by_key(|seq_group| deref_refcell!(seq_group).timestamp());
        self.swapped_out.make_contiguous().reverse();
    }
}
