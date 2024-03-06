use std::{cell::Cell, sync::mpsc::Sender};

use candle_core::Tensor;
use candle_sampling::logits_processor::{LogitsProcessor, Logprobs};

use crate::response::Response;

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
    RunningPrompt,
    RunningCompletion,
    Waiting,
    Error,
}

pub struct Sequence {
    tokens: Vec<u32>,
    logprobs: Vec<Logprobs>,
    prompt_len: usize,
    id: usize,
    timestamp: u64,
    state: Cell<SequenceState>,
    cache: Vec<Option<(Tensor, Tensor)>>,
    xlora_cache: Option<Vec<Option<(Tensor, Tensor)>>>,
    responder: Sender<Response>,
    logits_processor: LogitsProcessor,
    stop_tokens: Vec<u32>,
    max_len: Option<usize>,
    return_logprobs: bool,
}

impl Drop for Sequence {
    fn drop(&mut self) {
        println!("Dropping {}", self.id);
        dbg!(&self.cache);
        self.cache.clear();
    }
}

impl Sequence {
    #[allow(clippy::too_many_arguments)]
    pub fn new_waiting(
        tokens: Vec<u32>,
        id: usize,
        timestamp: u64,
        layers: usize,
        responder: Sender<Response>,
        logits_processor: LogitsProcessor,
        stop_tokens: Vec<u32>,
        max_len: Option<usize>,
        return_logprobs: bool,
        is_xlora: bool,
    ) -> Self {
        let prompt_len = tokens.len();
        Self {
            tokens,
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
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn id(&self) -> &usize {
        &self.id
    }

    pub fn is_running(&self) -> bool {
        self.state.get() == SequenceState::RunningCompletion
            || self.state.get() == SequenceState::RunningPrompt
    }

    pub fn is_completion(&self) -> bool {
        self.state.get() == SequenceState::RunningCompletion
    }

    pub fn is_prompt(&self) -> bool {
        self.state.get() == SequenceState::RunningPrompt
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

    pub fn is_xlora(&self) -> bool {
        self.xlora_cache.is_some()
    }

    pub fn logits_processor(&mut self) -> &mut LogitsProcessor {
        &mut self.logits_processor
    }

    pub fn add_token(&mut self, tok: Logprobs) {
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

    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }
}
