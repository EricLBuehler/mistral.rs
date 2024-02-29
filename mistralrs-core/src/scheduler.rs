use std::{
    cell::RefCell,
    collections::{vec_deque::Iter, VecDeque},
    iter::zip,
    rc::Rc,
};

use crate::{
    deref_mut_refcell, deref_refcell,
    sequence::{Sequence, SequenceState},
};

pub trait FcfsBacker {
    fn new() -> Self;
    fn next(&mut self) -> Option<Rc<RefCell<Sequence>>>;
    fn add(&mut self, item: Rc<RefCell<Sequence>>);
    fn iter(&self) -> impl Iterator<Item = &Rc<RefCell<Sequence>>>;
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
}

pub struct SchedulerOutput {
    pub completion: Box<[Rc<RefCell<Sequence>>]>,
    pub prompt: Box<[Rc<RefCell<Sequence>>]>,
}

pub enum SchedulerMethod {
    Fixed(usize),
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

        // If the waiting sequence will fit, add it. Keep track of its id.
        let mut waiting_to_remove = Vec::new();
        for seq in self.waiting.iter() {
            if self.sequence_fits(&running, &*deref_refcell!(seq)) {
                waiting_to_remove.push(*deref_refcell!(seq).id());
                deref_mut_refcell!(seq).set_state(SequenceState::RunningCompletion);
                running.push(seq.clone());
            }
        }

        // Remove sequences moved from waiting -> running.
        let mut waiting = Backer::new();
        for (id, seq) in zip(waiting_to_remove, self.waiting.iter()) {
            if *deref_refcell!(seq).id() != id {
                waiting.add(seq.clone());
            }
        }

        self.waiting = waiting;
        self.running = running.clone();

        let mut completion = Vec::new();
        let mut prompt = Vec::new();
        for seq in running {
            if deref_refcell!(seq).is_completion() {
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

    fn sequence_fits(&self, running: &[Rc<RefCell<Sequence>>], _seq: &Sequence) -> bool {
        match self.method {
            SchedulerMethod::Fixed(n) => running.len() + 1 < n,
        }
    }
}
