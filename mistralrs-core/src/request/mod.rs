use crate::{models::Cache, response::Response};
use std::{cell::Cell, sync::mpsc::Sender};

pub struct Request {
    pub prompt: String,
    pub response: Sender<Response>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum SequenceState {
    Done,
    Running,
    Waiting,
}

#[derive(Clone)]
pub struct Sequence {
    tokens: Vec<u32>,
    id: usize,
    state: Cell<SequenceState>,
    gen_idx: usize,
    cache: Cache,
}

impl Sequence {
    pub fn new_waiting(tokens: Vec<u32>, id: usize, layers: usize) -> Self {
        Self {
            tokens,
            id,
            state: Cell::new(SequenceState::Waiting),
            gen_idx: 0,
            cache: Cache::new(layers),
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn id(&self) -> &usize {
        &self.id
    }

    pub fn is_running(&self) -> bool {
        self.state.get() == SequenceState::Running
    }

    pub fn get_toks(&self) -> &[u32] {
        &self.tokens
    }

    pub fn gen_idx(&mut self) -> &mut usize {
        &mut self.gen_idx
    }

    pub fn cache(&self) -> &Cache {
        &self.cache
    }
}
