#![allow(dead_code)]

pub struct SpeculativeGraphPlan {
    proposal_len: usize,
    max_batch_size: Option<usize>,
}

impl SpeculativeGraphPlan {
    pub fn new(proposal_len: usize, max_batch_size: Option<usize>) -> Self {
        Self { proposal_len, max_batch_size }
    }
}

#[path = "autotuner_green.rs"]
mod autotuner;
