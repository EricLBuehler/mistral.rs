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

pub struct PASequenceData {
    prompt_token_ids: Vec<usize>,
    output_token_ids: Vec<Logprobs>,
    cumulative_logprob: f32,
    status: PASequenceStatus,
}

impl PASequenceData {
    pub fn new(prompt_token_ids: Vec<usize>) -> Self {
        Self {
            prompt_token_ids,
            output_token_ids: Vec::new(),
            cumulative_logprob: 0.,
            status: PASequenceStatus::Waiting,
        }
    }

    pub fn append_token_id(&mut self, logprobs: Logprobs) {
        self.cumulative_logprob += logprobs.logprob;
        self.output_token_ids.push(logprobs);
    }

    pub fn set_status(&mut self, status: PASequenceStatus) {
        self.status = status;
    }

    fn get_cumulative_logprob(&self) -> f32 {
        self.cumulative_logprob
    }
}

/// A PASequence holds information about the data it contains (the tokens), and the logical token blocks
/// to which it is mapped.
pub struct _PASequence {
    data: Mutex<PASequenceData>,
    seq_id: usize,
    logical_token_blocks: Vec<LogicalTokenBlock>,
    block_size: usize,
}

impl _PASequence {
    pub fn new(prompt_token_ids: Vec<usize>, seq_id: usize, block_size: usize) -> Self {
        let mut this = Self {
            data: Mutex::new(PASequenceData::new(prompt_token_ids.clone())),
            seq_id,
            logical_token_blocks: Vec::new(),
            block_size,
        };
        this.append_tokens_to_blocks(prompt_token_ids);
        this
    }

    pub fn add_token(&mut self, logprobs: Logprobs) {
        self.append_token_to_blocks(logprobs.token);
        self.deref_mut().append_token_id(logprobs);
    }

    pub fn blocks_to_add_new_tok(&mut self) -> usize {
        let last = self.logical_token_blocks.last_mut();
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

    pub fn is_prompt(&self) -> bool {
        self.deref().output_token_ids.is_empty()
    }

    pub fn get_prompt_len(&self) -> usize {
        self.deref().prompt_token_ids.len()
    }

    pub fn get_len(&self) -> usize {
        self.deref().prompt_token_ids.len() + self.deref().output_token_ids.len()
    }

    pub fn get_token_ids(&self) -> Vec<usize> {
        let mut res = self.deref().prompt_token_ids.clone();
        res.extend(
            self.deref()
                .output_token_ids
                .iter()
                .map(|logprobs| logprobs.token)
                .clone(),
        );
        res
    }

    pub fn get_last_token_id(&self) -> usize {
        if self.deref().output_token_ids.is_empty() {
            *self.deref().prompt_token_ids.last().unwrap()
        } else {
            self.deref().output_token_ids.last().unwrap().token
        }
    }

    pub fn is_finished(&self) -> bool {
        matches!(
            self.deref().status,
            PASequenceStatus::FinishedAborted
                | PASequenceStatus::FinishedIgnored
                | PASequenceStatus::Finished(_)
        )
    }

    pub fn get_cumulative_logprob(&self) -> f32 {
        self.deref().get_cumulative_logprob()
    }

    pub fn set_finish_reason(&mut self, finish_reason: String) {
        self.deref()
            .set_status(PASequenceStatus::Finished(finish_reason.clone()));
    }

    pub fn get_finish_reason(&self) -> String {
        match &self.deref().status {
            PASequenceStatus::Finished(state) => state.clone(),
            PASequenceStatus::FinishedAborted => "abort".to_string(),
            PASequenceStatus::FinishedIgnored => "length".to_string(),
            _ => {
                unreachable!("No finish reason.")
            }
        }
    }

    #[must_use]
    /// Clones the internal logprobs.
    pub fn get_output_tokens(&self) -> Vec<Logprobs> {
        self.deref().output_token_ids.clone() // TODO(EricLBuehler): Better way to do this?
    }

    fn append_tokens_to_blocks(&mut self, tokens: Vec<usize>) {
        for tok in tokens {
            self.append_token_to_blocks(tok);
        }
    }

    fn append_token_to_blocks(&mut self, token: usize) {
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

impl _PASequence {
    pub fn deref(&self) -> MutexGuard<'_, PASequenceData> {
        loop {
            if let Ok(res) = self.data.try_lock() {
                return res;
            }
        }
    }

    pub fn deref_mut(&self) -> MutexGuard<'_, PASequenceData> {
        loop {
            if let Ok(res) = self.data.try_lock() {
                return res;
            }
        }
    }
}

pub struct PASequence(pub Mutex<_PASequence>);

impl PASequence {
    pub fn deref_mut(&self) -> MutexGuard<'_, _PASequence> {
        loop {
            if let Ok(v) = self.0.try_lock() {
                return v;
            }
        }
    }
}

type SeqID = usize;

/// A PASequenceGroup holds the `n` (see SamplingParams) sequences generated from a single prompt.
/// A PASequenceGroup contains only sequences with the same prompt. They will always be scheduled together.
pub struct PASequenceGroup {
    seqs: HashMap<SeqID, Arc<PASequence>>,
    arrival_time: u64,
    group_id: usize,
    request_id: String,
    created: u64,
}

impl PASequenceGroup {
    pub fn new(
        seqs: &[Arc<PASequence>],
        arrival_time: u64,
        group_id: usize,
        request_id: String,
        created: u64,
    ) -> Self {
        let mut seq_map = HashMap::new();
        for seq in seqs {
            seq_map.insert(seq.deref_mut().get_id(), seq.clone());
        }
        Self {
            seqs: seq_map,
            arrival_time,
            group_id,
            request_id,
            created,
        }
    }

    pub fn set_status(&self, status: PASequenceStatus) {
        for seq in self.seqs.values() {
            seq.deref_mut().deref().set_status(status.clone());
        }
    }

    /// Blocks to add one new token to each sequence
    pub fn total_blocks_to_add_new_tok(&self) -> usize {
        self.seqs
            .values()
            .map(|seq| seq.deref_mut().blocks_to_add_new_tok())
            .sum()
    }

    pub fn get_prompt_len(&self) -> usize {
        self.seqs.len()
    }

    pub fn get_total_logical_token_blocks(&self) -> usize {
        self.seqs
            .values()
            .map(|seq| seq.deref_mut().get_logical_token_blocks())
            .sum()
    }

    pub fn get_seqs(&self) -> &HashMap<SeqID, Arc<PASequence>> {
        &self.seqs
    }

    pub fn arrival_time(&self) -> u64 {
        self.arrival_time
    }

    pub fn get_id(&self) -> &usize {
        &self.group_id
    }

    pub fn is_finished(&self) -> bool {
        self.seqs.iter().all(|(_, x)| x.deref_mut().is_finished())
    }

    pub fn get_request_id(&self) -> &String {
        &self.request_id
    }

    pub fn get_created_time(&self) -> u64 {
        self.created
    }
}
