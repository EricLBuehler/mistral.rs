use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::{Rc, Weak},
    sync::mpsc::Sender,
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::Tensor;
use candle_sampling::logits_processor::{LogitsProcessor, Logprobs};

use crate::{
    deref_mut_refcell, deref_refcell,
    pa::block_engine::LogicalTokenBlock,
    response::{Choice, Response},
    ChatCompletionUsage,
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StopReason {
    Eos,
    StopTok(u32),
    Length(usize),
    ModelLength(usize),
}

#[derive(Clone, Copy, PartialEq)]
pub enum SequenceState {
    Done(StopReason),
    Running,
    Waiting,
    Error,
    DoneIgnored,
    Swapped,
    DoneAborted,
}

pub struct Sequence {
    // Metadata, const
    id: usize,
    prompt_len: usize,
    max_len: Option<usize>,
    timestamp: u128,
    logits_processor: LogitsProcessor,
    stop_tokens: Vec<u32>,
    return_logprobs: bool,
    responder: Sender<Response>,
    group: Weak<RefCell<SequenceGroup>>,

    // Cache
    scaling_cache: Option<Tensor>,
    cache: Vec<Option<(Tensor, Tensor)>>,
    xlora_cache: Option<Vec<Option<(Tensor, Tensor)>>>,

    // Mutables
    tokens: Vec<u32>,
    logprobs: Vec<Logprobs>,

    // GPU things
    pub prompt_tok_per_sec: f32,
    pub prompt_timestamp: Option<u128>,
    pub total_sampling_time: u128,
    state: Cell<SequenceState>,

    // PA things
    logical_token_blocks: Vec<LogicalTokenBlock>,
    block_size: usize,
}

impl Sequence {
    #[allow(clippy::too_many_arguments)]
    pub fn new_waiting(
        tokens: Vec<u32>,
        id: usize,
        timestamp: u128,
        layers: usize,
        responder: Sender<Response>,
        logits_processor: LogitsProcessor,
        stop_tokens: Vec<u32>,
        max_len: Option<usize>,
        return_logprobs: bool,
        is_xlora: bool,
        group: Weak<RefCell<SequenceGroup>>,
        block_size: usize,
    ) -> Self {
        let prompt_len = tokens.len();
        let mut this = Self {
            tokens: tokens.clone(),
            logprobs: Vec::new(),
            prompt_len,
            id,
            timestamp,
            state: Cell::new(SequenceState::Waiting),
            cache: vec![None; layers],
            xlora_cache: if is_xlora {
                Some(vec![None; layers])
            } else {
                None
            },
            responder,
            logits_processor,
            stop_tokens,
            max_len,
            return_logprobs,
            prompt_tok_per_sec: 0.,
            prompt_timestamp: None,
            scaling_cache: None,
            total_sampling_time: 0,
            group,
            logical_token_blocks: Vec::new(),
            block_size,
        };
        this.append_tokens_to_blocks(tokens);
        this
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

    pub fn is_waiting(&self) -> bool {
        self.state.get() == SequenceState::Waiting
    }

    pub fn get_toks(&self) -> &[u32] {
        &self.tokens
    }

    pub fn cache(&mut self) -> &mut Vec<Option<(Tensor, Tensor)>> {
        &mut self.cache
    }

    pub fn xlora_cache(&mut self) -> &mut Vec<Option<(Tensor, Tensor)>> {
        self.xlora_cache.as_mut().unwrap()
    }

    pub fn scaling_cache(&mut self) -> &mut Option<Tensor> {
        &mut self.scaling_cache
    }

    pub fn is_xlora(&self) -> bool {
        self.xlora_cache.is_some()
    }

    pub fn logits_processor(&mut self) -> &mut LogitsProcessor {
        &mut self.logits_processor
    }

    pub fn add_token(&mut self, tok: Logprobs) {
        self.append_token_to_blocks(tok.token);
        self.tokens.push(tok.token);
        self.logprobs.push(tok);
    }

    pub fn responder(&self) -> Sender<Response> {
        self.responder.clone()
    }

    pub fn set_state(&self, state: SequenceState) {
        self.state.set(state);
    }

    pub fn is_done(&self, tok: u32, eos_tok: u32, max_model_len: usize) -> Option<StopReason> {
        if tok == eos_tok {
            Some(StopReason::Eos)
        } else if self.stop_tokens.contains(&tok) {
            Some(StopReason::StopTok(tok))
        } else if self.max_len.is_some()
            && self.tokens.len().saturating_sub(self.prompt_len) == self.max_len.unwrap()
        {
            // add_token was already called
            Some(StopReason::Length(self.max_len.unwrap()))
        } else if self.state.get() == SequenceState::DoneIgnored {
            // add_token was already called
            Some(StopReason::Length(self.len()))
        } else if self.tokens.len().saturating_sub(self.prompt_len) == max_model_len {
            Some(StopReason::ModelLength(max_model_len))
        } else {
            None
        }
    }

    pub fn logprobs(&self) -> &[Logprobs] {
        &self.logprobs
    }

    pub fn return_logprobs(&self) -> bool {
        self.return_logprobs
    }

    pub fn prompt_tokens(&self) -> usize {
        self.prompt_len
    }

    pub fn timestamp(&self) -> u128 {
        self.timestamp
    }

    pub fn prompt_timestamp(&self) -> Option<u128> {
        self.prompt_timestamp
    }

    pub fn add_choice_to_group(&self, choice: Choice) {
        let group = self.group.upgrade().unwrap();
        deref_mut_refcell!(group).done_count += 1;
        deref_mut_refcell!(group).choices.push(choice);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time travel has occurred!")
            .as_millis();

        deref_mut_refcell!(group).total_comple_time += now - self.prompt_timestamp.unwrap();
        deref_mut_refcell!(group).total_prompt_time +=
            self.prompt_timestamp.unwrap() - self.timestamp;
        deref_mut_refcell!(group).total_time += now - self.timestamp;

        deref_mut_refcell!(group).total_prompt_toks += self.prompt_len;
        deref_mut_refcell!(group).total_toks += self.len();

        deref_mut_refcell!(group).total_sampling_time += self.total_sampling_time;
    }

    pub fn get_next_choice_index(&self) -> usize {
        let group = self.group.upgrade().unwrap();
        let choices = &deref_refcell!(group).choices;
        choices.len()
    }

    pub fn get_group(&self) -> Rc<RefCell<SequenceGroup>> {
        self.group.upgrade().unwrap()
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

    pub fn is_finished(&self) -> bool {
        matches!(
            self.state.get(),
            SequenceState::Done(_) | SequenceState::DoneAborted | SequenceState::DoneIgnored
        )
    }

    pub fn is_prompt(&self) -> bool {
        self.len() == self.prompt_len
    }
}

pub struct SequenceGroup {
    done_count: usize,
    n_choices: usize,
    pub total_prompt_toks: usize,
    pub total_toks: usize,
    pub total_prompt_time: u128,
    pub total_time: u128,
    pub total_comple_time: u128,
    pub total_sampling_time: u128,
    choices: Vec<Choice>,
    seqs: HashMap<usize, Rc<RefCell<Sequence>>>,
    id: usize,
}

impl SequenceGroup {
    pub fn new(n_choices: usize, id: usize) -> Self {
        Self {
            done_count: 0,
            choices: Vec::new(),
            n_choices,
            total_prompt_toks: 0,
            total_toks: 0,
            total_prompt_time: 0,
            total_time: 0,
            total_comple_time: 0,
            total_sampling_time: 0,
            seqs: HashMap::new(),
            id,
        }
    }

    pub fn is_done(&self) -> bool {
        self.done_count == self.n_choices
    }

    pub fn get_choices(&self) -> &[Choice] {
        &self.choices
    }

    pub fn get_usage(&self) -> ChatCompletionUsage {
        #[allow(clippy::cast_precision_loss)]
        ChatCompletionUsage {
            completion_tokens: self.total_toks - self.total_prompt_toks,
            prompt_tokens: self.total_prompt_toks,
            total_tokens: self.total_toks,
            avg_tok_per_sec: (self.total_toks as f32 / self.total_time as f32) * 1000.,
            avg_prompt_tok_per_sec: (self.total_prompt_toks as f32 / self.total_prompt_time as f32)
                * 1000.,
            avg_compl_tok_per_sec: ((self.total_toks - self.total_prompt_toks) as f32
                / self.total_comple_time as f32)
                * 1000.,
            avg_sample_tok_per_sec: (self.total_toks as f32 / self.total_sampling_time as f32)
                * 1000.,
        }
    }

    pub fn add_seq(&mut self, seq: Rc<RefCell<Sequence>>) {
        let id = *deref_refcell!(seq).id();
        self.seqs.insert(id, seq);
    }

    pub fn get_seqs(&self) -> &HashMap<usize, Rc<RefCell<Sequence>>> {
        &self.seqs
    }

    /// Blocks to add one new token to each sequence
    pub fn total_blocks_to_add_new_tok(&self) -> usize {
        self.seqs
            .values()
            .map(|seq| deref_refcell!(seq).blocks_to_add_new_tok())
            .sum()
    }

    pub fn get_total_logical_token_blocks(&self) -> usize {
        self.seqs
            .values()
            .map(|seq| deref_refcell!(seq).get_logical_token_blocks())
            .sum()
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn timestamp(&self) -> u128 {
        deref_refcell!(self.seqs.values().nth(0).unwrap()).timestamp
    }

    pub fn set_state(&self, status: SequenceState) {
        for seq in self.seqs.values() {
            deref_mut_refcell!(seq).set_state(status.clone());
        }
    }

    pub fn get_prompt_len(&self) -> usize {
        deref_refcell!(self.seqs.iter().nth(0).unwrap().1).prompt_len
    }

    pub fn is_finished(&self) -> bool {
        self.seqs
            .iter()
            .all(|(_, x)| deref_refcell!(x).is_finished())
    }
}
