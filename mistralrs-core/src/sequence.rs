use std::{
    cell::{Cell, Ref, RefCell},
    rc::Rc,
    sync::mpsc::Sender,
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::{Device, IntDType, Result, Tensor};
use candle_sampling::logits_processor::{LogitsProcessor, Logprobs};

use crate::{
    deref_mut_refcell, deref_refcell,
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
    RunningPrompt,
    RunningCompletion,
    Waiting,
    Error,
}

pub struct Sequence {
    // Metadata, const
    id: usize,
    prompt_len: usize,
    max_len: Option<usize>,
    timestamp: u128,
    logits_processor: LogitsProcessor,
    stop_tokens: Vec<Tensor>, // scalars
    return_logprobs: bool,
    responder: Sender<Response>,

    // Cache
    scaling_cache: Option<Tensor>,
    cache: Vec<Option<(Tensor, Tensor)>>,
    xlora_cache: Option<Vec<Option<(Tensor, Tensor)>>>,

    // Mutables
    tokens: Tensor,
    logprobs: Vec<Logprobs>,
    position: Tensor, // scalar
    position_usize: usize,

    // GPU things
    pub prompt_tok_per_sec: f32,
    pub prompt_timestamp: Option<u128>,
    group: Rc<RefCell<SequenceGroup>>,
    pub total_sampling_time: u128,
    state: Cell<SequenceState>,
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
        group: Rc<RefCell<SequenceGroup>>,
        device: &Device,
    ) -> Result<Self> {
        let prompt_len = tokens.len();
        Ok(Self {
            tokens: Tensor::new(tokens, device)?,
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
            stop_tokens: stop_tokens
                .iter()
                .map(|t| Tensor::new(vec![*t], device).unwrap())
                .collect::<Vec<_>>(),
            max_len,
            return_logprobs,
            prompt_tok_per_sec: 0.,
            prompt_timestamp: None,
            group,
            scaling_cache: None,
            total_sampling_time: 0,
            position: Tensor::new(0i64, device).unwrap(),
            position_usize: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.dims1().unwrap()
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

    pub fn get_toks(&self) -> &Tensor {
        &self.tokens
    }

    pub fn get_position_scalar(&mut self) -> &mut Tensor {
        &mut self.position
    }

    pub fn get_position_usize(&mut self) -> &mut usize {
        &mut self.position_usize
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

    pub fn add_token(&mut self, tok: Logprobs) -> Result<()> {
        self.tokens = Tensor::cat(&[&self.tokens, &tok.token.unsqueeze(0)?], 0)?;
        self.logprobs.push(tok);
        Ok(())
    }

    pub fn responder(&self) -> Sender<Response> {
        self.responder.clone()
    }

    pub fn set_state(&self, state: SequenceState) {
        self.state.set(state);
    }

    pub fn is_done(
        &self,
        tok: Tensor,
        eos_tok: Tensor,
        max_model_len: usize,
    ) -> Option<StopReason> {
        // TODO(EricLBuehler): Is there a way to avoid this copy?
        /*if tok
            .eq(&eos_tok)
            .unwrap()
            .to_scalar::<u8>()
            .unwrap()
            .is_true()
        {
            Some(StopReason::Eos)
        } else if self.stop_tokens.iter().any(|stop_t| {
            stop_t
                .eq(&tok)
                .unwrap()
                .to_scalar::<u8>()
                .unwrap()
                .is_true()
        }) {
            Some(StopReason::StopTok(tok.to_scalar::<u32>().unwrap()))
        } else */if self.max_len.is_some()
            && self.len().saturating_sub(self.prompt_len) == self.max_len.unwrap()
        {
            // add_token was already called
            Some(StopReason::Length(self.max_len.unwrap()))
        } else if self.len().saturating_sub(self.prompt_len) == max_model_len {
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
        deref_mut_refcell!(self.group).done_count += 1;
        deref_mut_refcell!(self.group).choices.push(choice);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time travel has occurred!")
            .as_millis();

        deref_mut_refcell!(self.group).total_comple_time += now - self.prompt_timestamp.unwrap();
        deref_mut_refcell!(self.group).total_prompt_time +=
            self.prompt_timestamp.unwrap() - self.timestamp;
        deref_mut_refcell!(self.group).total_time += now - self.timestamp;

        deref_mut_refcell!(self.group).total_prompt_toks += self.prompt_len;
        deref_mut_refcell!(self.group).total_toks += self.len();
        deref_mut_refcell!(self.group).total_sampling_time += self.total_sampling_time;
    }

    pub fn get_group(&self) -> Ref<'_, SequenceGroup> {
        deref_refcell!(self.group)
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
}

impl SequenceGroup {
    pub fn new(n_choices: usize) -> Self {
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
}
