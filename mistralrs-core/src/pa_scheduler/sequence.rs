use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use candle_sampling::logits_processor::Logprobs;

use super::block_engine::LogicalTokenBlock;

#[derive(Clone)]
pub enum PASequenceStatus {
    FinishedIgnored,
    Waiting,
    Running,
    Swapped,
    FinishedAborted,
    Finished(String),
}

/// A PASequence holds information about the data it contains (the tokens), and the logical token blocks
/// to which it is mapped.
pub struct PASequence {
    seq_id: usize,
    logical_token_blocks: Vec<LogicalTokenBlock>,
    block_size: usize,
    status: PASequenceStatus,
}

impl PASequence {
    pub fn new(prompt_token_ids: Vec<u32>, seq_id: usize, block_size: usize) -> Self {
        let mut this = Self {
            seq_id,
            logical_token_blocks: Vec::new(),
            block_size,
            status: PASequenceStatus::Waiting,
        };
        this.append_tokens_to_blocks(prompt_token_ids);
        this
    }

    pub fn add_token(&mut self, token: u32) {
        self.append_token_to_blocks(token);
    }

    pub fn blocks_to_add_new_tok(&self) -> usize {
        let last = self.logical_token_blocks.last();
        if !last.is_some_and(|last| last.is_full()) {
            // If we have space
            0
        } else {
            1
        }
    }

    pub fn get_logical_token_blocks(&self) -> usize {
        self.logical_token_blocks.len()
    }

    pub fn get_id(&self) -> usize {
        self.seq_id
    }

    pub fn is_finished(&self) -> bool {
        matches!(
            self.status,
            PASequenceStatus::FinishedAborted
                | PASequenceStatus::FinishedIgnored
                | PASequenceStatus::Finished(_)
        )
    }

    fn append_tokens_to_blocks(&mut self, tokens: Vec<u32>) {
        for tok in tokens {
            self.append_token_to_blocks(tok);
        }
    }

    fn append_token_to_blocks(&mut self, token: u32) {
        let last = self.logical_token_blocks.last_mut();
        if !last.as_ref().is_some_and(|last| last.is_full()) {
            // If we have space
            let last = last.unwrap();
            last.append_token_id(token);
        } else {
            self.logical_token_blocks
                .push(LogicalTokenBlock::new(self.block_size));
            self.logical_token_blocks
                .last_mut()
                .unwrap()
                .append_token_id(token);
        }
    }
}

type SeqID = usize;

/// A PASequenceGroup holds the `n` (see SamplingParams) sequences generated from a single prompt.
/// A PASequenceGroup contains only sequences with the same prompt. They will always be scheduled together.
pub struct PASequenceGroup {
    seqs: HashMap<SeqID, PASequence>,
    arrival_time: u64,
    group_id: usize,
    request_id: String,
    created: u64,
}

impl PASequenceGroup {
    pub fn new(
        seqs: Vec<PASequence>,
        arrival_time: u64,
        group_id: usize,
        request_id: String,
        created: u64,
    ) -> Self {
        let mut seq_map = HashMap::new();
        for seq in seqs.into_iter() {
            seq_map.insert(seq.get_id(), seq);
        }
        Self {
            seqs: seq_map,
            arrival_time,
            group_id,
            request_id,
            created,
        }
    }

    pub fn set_status(&mut self, status: PASequenceStatus) {
        for seq in self.seqs.values_mut() {
            seq.status = status.clone();
        }
    }

    /// Blocks to add one new token to each sequence
    pub fn total_blocks_to_add_new_tok(&self) -> usize {
        self.seqs
            .values()
            .map(|seq| seq.blocks_to_add_new_tok())
            .sum()
    }

    pub fn get_prompt_len(&self) -> usize {
        self.seqs.len()
    }

    pub fn get_total_logical_token_blocks(&self) -> usize {
        self.seqs
            .values()
            .map(|seq| seq.get_logical_token_blocks())
            .sum()
    }

    pub fn get_seqs(&self) -> &HashMap<SeqID, PASequence> {
        &self.seqs
    }

    pub fn arrival_time(&self) -> u64 {
        self.arrival_time
    }

    pub fn get_id(&self) -> &usize {
        &self.group_id
    }

    pub fn is_finished(&self) -> bool {
        self.seqs.iter().all(|(_, x)| x.is_finished())
    }

    pub fn get_request_id(&self) -> &String {
        &self.request_id
    }

    pub fn get_created_time(&self) -> u64 {
        self.created
    }
}
