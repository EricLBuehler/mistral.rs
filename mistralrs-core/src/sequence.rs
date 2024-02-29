use std::{cell::Cell, sync::mpsc::Sender};

use candle_sampling::logits_processor::LogitsProcessor;

use crate::{models::Cache, response::Response};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StopReason {
    Eos,
    StopTok(u32),
    Length(usize),
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
    prompt_len: usize,
    id: usize,
    state: Cell<SequenceState>,
    gen_idx: usize,
    cache: Cache,
    responder: Sender<Response>,
    logits_processor: LogitsProcessor,
    stop_tokens: Vec<u32>,
    max_len: usize,
}

impl Sequence {
    pub fn new_waiting(
        tokens: Vec<u32>,
        id: usize,
        layers: usize,
        responder: Sender<Response>,
        logits_processor: LogitsProcessor,
        stop_tokens: Vec<u32>,
        max_len: Option<usize>,
    ) -> Self {
        let prompt_len = tokens.len();
        Self {
            tokens,
            prompt_len,
            id,
            state: Cell::new(SequenceState::Waiting),
            gen_idx: 0,
            cache: Cache::new(layers),
            responder,
            logits_processor,
            stop_tokens,
            max_len: max_len.unwrap_or(64),
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

    pub fn gen_idx(&mut self) -> &mut usize {
        &mut self.gen_idx
    }

    pub fn cache(&self) -> &Cache {
        &self.cache
    }

    pub fn logits_processor(&mut self) -> &mut LogitsProcessor {
        &mut self.logits_processor
    }

    pub fn add_token(&mut self, tok: u32) {
        self.tokens.push(tok);
    }

    pub fn responder(&self) -> Sender<Response> {
        self.responder.clone()
    }

    pub fn set_state(&self, state: SequenceState) {
        self.state.set(state);
    }

    pub fn is_done(&self, tok: u32, eos_tok: u32) -> Option<StopReason> {
        if tok == eos_tok {
            Some(StopReason::Eos)
        } else if self.stop_tokens.contains(&tok) {
            Some(StopReason::StopTok(tok))
        } else if self.tokens.len().saturating_sub(self.prompt_len) == self.max_len {
            // add_token was already called
            Some(StopReason::Length(self.max_len))
        } else {
            None
        }
    }
}
