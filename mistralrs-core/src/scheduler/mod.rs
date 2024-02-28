use std::{
    collections::{vec_deque::Iter, VecDeque},
    iter::zip,
};

use crate::request::Sequence;

pub trait FcfsBacker {
    fn new() -> Self;
    fn next(&mut self) -> Option<Sequence>;
    fn add(&mut self, item: Sequence);
    fn iter(&self) -> impl Iterator<Item = &Sequence>;
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
}

pub struct SchedulerOutput {
    pub seqs: Vec<Sequence>,
}

pub struct Scheduler<Backer: FcfsBacker> {
    waiting: Backer,
    running: Vec<Sequence>,
}

impl<Backer: FcfsBacker> Scheduler<Backer> {
    pub fn new() -> Self {
        Self {
            running: Vec::new(),
            waiting: Backer::new(),
        }
    }

    pub fn add_seq(&mut self, seq: Sequence) {
        self.waiting.add(seq)
    }

    /// Schedule all sequences based on their state and the available space.
    pub fn schedule(&mut self) -> SchedulerOutput {
        // Filter out all done sequences
        let running = self.running.clone();
        let mut running = running
            .iter()
            .filter(|seq| seq.is_running())
            .map(|seq| seq.clone())
            .collect::<Vec<_>>();

        // If the waiting sequence will fit, add it. Keep track of its id.
        let mut waiting_to_remove = Vec::new();
        for seq in self.waiting.iter() {
            if self.sequence_fits(&running, seq) {
                waiting_to_remove.push(seq.id());
                running.push(seq.clone());
            }
        }

        // Remove sequences moved from waiting -> running.
        let mut waiting = Backer::new();
        for (id, seq) in zip(waiting_to_remove, self.waiting.iter()) {
            if seq.id() != id {
                waiting.add(seq.clone());
            }
        }

        self.waiting = waiting;
        self.running = running.clone();

        SchedulerOutput { seqs: running }
    }

    fn sequence_fits(&self, running: &[Sequence], seq: &Sequence) -> bool {
        todo!()
    }
}
