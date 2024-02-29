use std::{
    cell::RefCell,
    collections::{vec_deque::Iter, HashMap, VecDeque},
    rc::Rc,
};

use crate::{
    deref_mut_refcell, deref_refcell,
    sequence::{Sequence, SequenceState},
};
use range_checked::UsizeBounded;

pub trait FcfsBacker {
    fn new() -> Self;
    fn next(&mut self) -> Option<Rc<RefCell<Sequence>>>;
    fn add(&mut self, item: Rc<RefCell<Sequence>>);
    fn iter(&self) -> impl Iterator<Item = &Rc<RefCell<Sequence>>>;
    fn sort_ascending_ids(&mut self);
}

impl FcfsBacker for VecDeque<Rc<RefCell<Sequence>>> {
    fn new() -> Self {
        Self::new()
    }
    fn add(&mut self, item: Rc<RefCell<Sequence>>) {
        self.push_back(item)
    }
    fn next(&mut self) -> Option<Rc<RefCell<Sequence>>> {
        self.pop_front()
    }
    fn iter(&self) -> Iter<'_, Rc<RefCell<Sequence>>> {
        self.iter()
    }
    fn sort_ascending_ids(&mut self) {
        let slice = self.make_contiguous();
        slice.sort_by_key(|seq| *deref_refcell!(seq).id());
    }
}

pub struct SchedulerOutput {
    pub completion: Box<[Rc<RefCell<Sequence>>]>,
    pub prompt: Box<[Rc<RefCell<Sequence>>]>,
}

pub enum SchedulerMethod {
    Fixed(UsizeBounded<1, { usize::MAX }, false>),
}

pub struct Scheduler<Backer: FcfsBacker> {
    waiting: Backer,
    running: Vec<Rc<RefCell<Sequence>>>,
    method: SchedulerMethod,
}

impl<Backer: FcfsBacker> Scheduler<Backer> {
    pub fn new(method: SchedulerMethod) -> Self {
        Self {
            running: Vec::new(),
            waiting: Backer::new(),
            method,
        }
    }

    pub fn add_seq(&mut self, seq: Sequence) {
        self.waiting.add(Rc::new(RefCell::new(seq)))
    }

    /// Schedule all sequences based on their state and the available space.
    pub fn schedule(&mut self) -> SchedulerOutput {
        // Filter out all done sequences
        let running = self.running.clone();
        let mut running = running
            .iter()
            .filter(|seq| deref_refcell!(seq).is_running())
            .cloned()
            .collect::<Vec<_>>();

        // Sort the waiting seqs
        self.waiting.sort_ascending_ids();

        // If the waiting sequence will fit, add it. Keep track of its id.
        let mut waiting_to_remove = Vec::new();
        for seq in self.waiting.iter() {
            if self.sequence_fits(&running, &*deref_refcell!(seq)) {
                waiting_to_remove.push(*deref_refcell!(seq).id());
                if deref_refcell!(seq).is_waiting() {
                    deref_mut_refcell!(seq).set_state(SequenceState::RunningPrompt);
                }
                running.push(seq.clone());
            }
        }

        // Remove sequences moved from waiting -> running.
        let mut waiting = Backer::new();
        for seq in self.waiting.iter() {
            if !waiting_to_remove.contains(deref_refcell!(seq).id()) {
                waiting.add(seq.clone());
            }
        }

        // Now, get the sequences with the smallest sequence lengths, and allow them to catch up.
        let mut seq_buckets: HashMap<usize, Vec<Rc<RefCell<Sequence>>>> = HashMap::new();
        let mut min_len = usize::MAX;
        for seq in &running {
            let len = deref_refcell!(seq).len();
            if len < min_len {
                min_len = len;
            }
            match seq_buckets.get_mut(&len) {
                Some(bucket) => bucket.push(seq.clone()),
                None => {
                    seq_buckets.insert(len, vec![seq.clone()]);
                }
            }
        }
        let (running, waiting) = if seq_buckets.len() <= 1 {
            // Full steam ahead or have everything
            (running, waiting)
        } else {
            // Set the min seqs to be the running ones, and the rest to be waiting (but their states are not changed!)
            // Allow the min seqs to catch up.
            let min_seqs = seq_buckets.remove(&min_len).unwrap();
            for (_, seqs) in seq_buckets {
                for seq in seqs {
                    waiting.add(seq);
                }
            }
            // Know min_seqs.len < running.len() <= max
            (min_seqs, waiting)
        };

        let mut completion = Vec::new();
        let mut prompt = Vec::new();
        for seq in &running {
            if deref_refcell!(seq).is_completion() {
                completion.push(seq.clone());
            } else {
                prompt.push(seq.clone());
            }
        }

        self.waiting = waiting;
        self.running = running;

        SchedulerOutput {
            completion: completion.into(),
            prompt: prompt.into(),
        }
    }

    fn sequence_fits(&self, running: &[Rc<RefCell<Sequence>>], _seq: &Sequence) -> bool {
        match &self.method {
            SchedulerMethod::Fixed(n) => (running.len() + 1) <= **n,
        }
    }
}
