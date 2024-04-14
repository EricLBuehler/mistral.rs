use std::collections::{
    vec_deque::{Iter, IterMut},
    VecDeque,
};

use crate::sequence::{Sequence, SequenceState};
use range_checked::UsizeBounded;

pub trait FcfsBacker: Default {
    fn new() -> Self;
    fn next(&mut self) -> Option<Sequence>;
    fn add(&mut self, item: Sequence);
    fn into_iter(self) -> impl Iterator<Item = Sequence>;
    fn iter(&self) -> impl Iterator<Item = &Sequence>;
    fn mut_iter(&mut self) -> impl Iterator<Item = &mut Sequence>;
    fn sort_ascending_ids(&mut self);
}

impl FcfsBacker for VecDeque<Sequence> {
    fn new() -> Self {
        Self::new()
    }
    fn add(&mut self, item: Sequence) {
        self.push_back(item)
    }
    fn next(&mut self) -> Option<Sequence> {
        self.pop_front()
    }
    fn iter(&self) -> Iter<'_, Sequence> {
        self.iter()
    }
    fn mut_iter(&mut self) -> IterMut<'_, Sequence> {
        self.iter_mut()
    }
    fn into_iter(self) -> impl Iterator<Item = Sequence> {
        <Self as IntoIterator>::into_iter(self)
    }
    fn sort_ascending_ids(&mut self) {
        let slice = self.make_contiguous();
        slice.sort_by_key(|seq| *seq.id());
    }
}

pub struct SchedulerOutput<'a> {
    pub completion: Box<[&'a mut Sequence]>,
    pub prompt: Box<[&'a mut Sequence]>,
}

pub enum SchedulerMethod {
    Fixed(UsizeBounded<1, { usize::MAX }, false>),
}

pub struct Scheduler<Backer: FcfsBacker> {
    waiting: Backer,
    running: Vec<Sequence>,
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
        if seq.is_running() {
            // prefill case
            self.running.push(seq);
        } else {
            self.waiting.add(seq);
        }
    }

    /// Schedule all sequences based on their state and the available space.
    pub fn schedule(&mut self) -> SchedulerOutput {
        // Filter out all done sequences
        let running = std::mem::take(&mut self.running);
        let mut waiting = std::mem::take(&mut self.waiting);
        let mut running = running
            .into_iter()
            .filter(|seq| seq.is_running())
            .collect::<Vec<_>>();

        match (waiting.iter().count(), running.len()) {
            (0, 0) => {
                self.running = running;
                return SchedulerOutput {
                    prompt: vec![].into(),
                    completion: vec![].into(),
                };
            }
            (_, 0) => {
                for seq in waiting.into_iter() {
                    seq.set_state(SequenceState::RunningPrompt);
                    self.running.push(seq);
                }
                self.waiting = Backer::new();
                return SchedulerOutput {
                    prompt: self.running.iter_mut().collect::<Vec<_>>().into(),
                    completion: vec![].into(),
                };
            }
            (0, _) => {
                self.running = running;
                return SchedulerOutput {
                    prompt: vec![].into(),
                    completion: self.running.iter_mut().collect::<Vec<_>>().into(),
                };
            }
            _ => {}
        }

        // Sort the waiting seqs
        waiting.sort_ascending_ids();

        // If the waiting sequence will fit, add it. Otherwise remove it
        let mut new_waiting = Backer::new();
        for seq in waiting.into_iter() {
            if self.sequence_fits(&running, &seq) {
                if seq.is_waiting() {
                    seq.set_state(SequenceState::RunningPrompt);
                }
                running.push(seq);
            } else {
                new_waiting.add(seq);
            }
        }

        self.waiting = new_waiting;
        self.running = running;

        let mut completion = Vec::new();
        let mut prompt = Vec::new();
        for seq in &mut self.running {
            if seq.is_completion() {
                completion.push(seq);
            } else {
                prompt.push(seq);
            }
        }

        SchedulerOutput {
            completion: completion.into(),
            prompt: prompt.into(),
        }
    }

    fn sequence_fits(&self, running: &[Sequence], _seq: &Sequence) -> bool {
        match &self.method {
            SchedulerMethod::Fixed(n) => (running.len() + 1) <= **n,
        }
    }
}
