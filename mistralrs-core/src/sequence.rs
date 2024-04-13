use std::{
    cell::{Cell, RefCell, RefMut},
    rc::Rc,
    sync::mpsc::Sender,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    aici::{cfg::CfgParser, recognizer::StackRecognizer, rx::RecRx},
    response::CompletionChoice,
    CompletionResponse,
};
use crate::{
    get_mut_group,
    models::LayerCaches,
    response::{ChatCompletionChunkResponse, Choice, ChunkChoice, Response, SYSTEM_FINGERPRINT},
    sampler::{Logprobs, Sampler},
    ChatCompletionResponse, Usage,
};
use candle_core::Tensor;
use regex_automata::util::primitives::StateID;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StopReason {
    Eos,
    StopTok(u32),
    Length(usize),
    ModelLength(usize),
    StopString(/* Index in the list of stop strings */ usize),
}

impl ToString for StopReason {
    fn to_string(&self) -> String {
        match self {
            StopReason::Eos => "stop".to_string(),
            StopReason::Length(_) | StopReason::ModelLength(_) => "length".to_string(),
            StopReason::StopTok(_) | StopReason::StopString(_) => "stop".to_string(),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum SequenceState {
    Done(StopReason),
    RunningPrompt,
    RunningCompletion,
    Waiting,
    Error,
    RunningPrefillPrompt,
}

#[derive(Clone)]
pub enum SequenceRecognizer {
    Regex(Box<StackRecognizer<StateID, RecRx>>),
    Cfg(Box<CfgParser>),
    None,
}
pub struct Sequence {
    // Metadata, const
    id: usize,
    prompt_len: usize,
    max_len: Option<usize>,
    timestamp: u128,
    sampler: Sampler,
    stop_tokens: Vec<u32>,
    stop_strings: Vec<String>,
    return_logprobs: bool,
    responder: Sender<Response>,
    response_index: usize,
    creation_time: u64,
    prefill_prompt_toks: Option<Vec<u32>>,
    pub suffix: Option<String>,
    pub prefix: Option<String>,

    // Cache
    scaling_cache: Option<Tensor>,
    cache: LayerCaches,
    xlora_cache: Option<LayerCaches>,

    // Mutables
    tokens: Vec<u32>,
    logprobs: Vec<Logprobs>,
    cumulative_logprob: f32,
    completion_bytes: Vec<u8>,
    stream_idx: usize,
    pub recognizer: SequenceRecognizer,

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
        sampler: Sampler,
        stop_tokens: Vec<u32>,
        stop_strings: Vec<String>,
        max_len: Option<usize>,
        return_logprobs: bool,
        is_xlora: bool,
        group: Rc<RefCell<SequenceGroup>>,
        response_index: usize,
        creation_time: u64,
        recognizer: SequenceRecognizer,
        suffix: Option<String>,
        prefix: Option<String>,
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
            sampler,
            stop_tokens,
            stop_strings,
            max_len,
            return_logprobs,
            prompt_tok_per_sec: 0.,
            prompt_timestamp: None,
            group,
            scaling_cache: None,
            total_sampling_time: 0,
            response_index,
            creation_time,
            recognizer,
            prefill_prompt_toks: None,
            suffix,
            prefix,
            cumulative_logprob: 0.,
            completion_bytes: Vec::new(),
            stream_idx: 0,
        }
    }

    pub fn prefill(mut self, cache: LayerCaches) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time travel has occurred!")
            .as_millis();
        self.prompt_timestamp = Some(now);

        self.cache = cache;
        self.set_state(SequenceState::RunningCompletion);
        self
    }

    pub fn prefill_subset(mut self, cache: LayerCaches, toks: Vec<u32>) -> Self {
        self.cache = cache;
        self.prefill_prompt_toks = Some(toks);
        self.set_state(SequenceState::RunningPrefillPrompt);
        self
    }

    pub fn len(&self) -> usize {
        if let Some(toks) = &self.prefill_prompt_toks {
            return toks.len();
        }
        self.tokens.len()
    }

    pub fn id(&self) -> &usize {
        &self.id
    }

    pub fn is_running(&self) -> bool {
        self.state.get() == SequenceState::RunningCompletion
            || self.state.get() == SequenceState::RunningPrompt
            || self.state.get() == SequenceState::RunningPrefillPrompt
    }

    pub fn is_completion(&self) -> bool {
        self.state.get() == SequenceState::RunningCompletion
    }

    pub fn is_prompt(&self) -> bool {
        self.state.get() == SequenceState::RunningPrompt
            || self.state.get() == SequenceState::RunningPrefillPrompt
    }

    pub fn is_waiting(&self) -> bool {
        self.state.get() == SequenceState::Waiting
    }

    pub fn get_toks(&self) -> &[u32] {
        if let Some(toks) = &self.prefill_prompt_toks {
            return toks;
        }
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

    pub fn sampler(&mut self) -> &mut Sampler {
        &mut self.sampler
    }

    pub fn add_token(
        &mut self,
        tok: Logprobs,
        completion_bytes: Vec<u8>,
        is_done: &Option<StopReason>,
    ) {
        let stopped_by_token = matches!(
            is_done,
            Some(StopReason::Eos) | Some(StopReason::StopTok(_))
        );
        if !stopped_by_token {
            // Completion bytes is used to check for stop strings, and as the streaming buffer.
            // We don't need to add stop tokens to the completion bytes to check for stop strings.
            // And by not adding it here, we can avoid having to delete these tokens from the streaming output.
            self.completion_bytes.extend_from_slice(&completion_bytes);
        }
        self.cumulative_logprob += tok.logprob;
        self.tokens.push(tok.token);
        self.logprobs.push(tok);
        self.prefill_prompt_toks = None;
    }

    pub fn responder(&self) -> Sender<Response> {
        self.responder.clone()
    }

    pub fn creation_time(&self) -> u64 {
        self.creation_time
    }

    pub fn set_state(&self, state: SequenceState) {
        if matches!(state, SequenceState::Error) {
            get_mut_group!(self).n_choices -= 1;
        }
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
            if !self.stop_strings.is_empty() {
                for (idx, s) in self.stop_strings.iter().enumerate() {
                    if galil_seiferas::gs_find(&self.completion_bytes, s.as_bytes()).is_some() {
                        return Some(StopReason::StopString(idx));
                    }
                }
            }
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

    pub fn stop_strings(&self) -> &[String] {
        &self.stop_strings
    }

    /// Returns the delta between the last two decoded sequences
    pub fn get_delta(
        &mut self,
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let is_first = self.stream_idx == 0;
        let slice = self.completion_bytes[self.stream_idx..].to_vec();
        let new_decoded = unsafe { String::from_utf8_unchecked(slice) };
        // Check if the sequence ends with valid utf8, if not skip it as it probably is a multi token sequence
        if new_decoded.ends_with('�') {
            return Ok(None);
        }
        self.stream_idx = self.completion_bytes.len();

        // The first token usually starts with a space. We don't want to add that to the delta.
        // Since we're using the completion_bytes, we need to take care of that ourselves.
        // Had we used HF's Tokenizer, it would have taken care of that for us.
        if is_first {
            return Ok(Some(new_decoded.trim_start().to_string()));
        }
        Ok(Some(new_decoded))
    }

    pub fn timestamp(&self) -> u128 {
        self.timestamp
    }

    pub fn prompt_timestamp(&self) -> Option<u128> {
        self.prompt_timestamp
    }

    fn update_time_info(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time travel has occurred!")
            .as_millis();

        if let Some(ts) = self.prompt_timestamp {
            get_mut_group!(self).total_completion_time += now - ts;
            get_mut_group!(self).total_prompt_time += ts - self.timestamp;
        }

        get_mut_group!(self).total_time += now - self.timestamp;

        get_mut_group!(self).total_prompt_toks += self.prompt_len;
        get_mut_group!(self).total_toks += self.len();

        get_mut_group!(self).total_sampling_time += self.total_sampling_time;
    }

    pub fn add_choice_to_group(&self, choice: Choice) {
        get_mut_group!(self).choices.push(choice);
        self.update_time_info();
    }

    pub fn add_completion_choice_to_group(&self, choice: CompletionChoice) {
        get_mut_group!(self)
            .completion_choices
            .push((self.cumulative_logprob, choice));
        self.update_time_info();
    }

    pub fn get_response_index(&self) -> usize {
        self.response_index
    }

    pub fn get_mut_group(&self) -> RefMut<'_, SequenceGroup> {
        get_mut_group!(self)
    }

    pub fn add_streaming_chunk_choice_to_group(&self, chunk: ChunkChoice) {
        get_mut_group!(self).streaming_chunks.push(chunk);
    }
}

pub struct SequenceGroup {
    n_choices: usize, // The target number of choices to return. Can be decreased if an error is thrown.
    best_of: Option<usize>, // Top n seqs based on cumulative logprobs.
    pub total_prompt_toks: usize,
    pub total_toks: usize,
    pub total_prompt_time: u128,
    pub total_time: u128,
    pub total_completion_time: u128,
    pub total_sampling_time: u128,
    choices: Vec<Choice>,
    completion_choices: Vec<(f32, CompletionChoice)>,
    pub streaming_chunks: Vec<ChunkChoice>,
    pub is_streaming: bool,
    pub is_chat: bool,
}

impl SequenceGroup {
    pub fn new(
        n_choices: usize,
        is_streaming: bool,
        is_chat: bool,
        best_of: Option<usize>,
    ) -> Self {
        Self {
            choices: Vec::new(),
            completion_choices: Vec::new(),
            n_choices,
            total_prompt_toks: 0,
            total_toks: 0,
            total_prompt_time: 0,
            total_time: 0,
            total_completion_time: 0,
            total_sampling_time: 0,
            streaming_chunks: Vec::new(),
            is_streaming,
            is_chat,
            best_of,
        }
    }

    /// This does not apply best_of.
    pub fn get_choices(&self) -> &[Choice] {
        &self.choices
    }

    /// This applies the best_of.
    pub fn get_completion_choices(&self) -> Vec<CompletionChoice> {
        let mut choices = self.completion_choices.clone();
        // Sort by descending logprobs
        choices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        choices
            .into_iter()
            .take(self.best_of.unwrap())
            .map(|(_, x)| x)
            .collect::<Vec<_>>()
    }

    pub fn get_usage(&self) -> Usage {
        #[allow(clippy::cast_precision_loss)]
        Usage {
            completion_tokens: self.total_toks - self.total_prompt_toks,
            prompt_tokens: self.total_prompt_toks,
            total_tokens: self.total_toks,
            avg_tok_per_sec: (self.total_toks as f32 / self.total_time as f32) * 1000.,
            avg_prompt_tok_per_sec: (self.total_prompt_toks as f32 / self.total_prompt_time as f32)
                * 1000.,
            avg_compl_tok_per_sec: ((self.total_toks - self.total_prompt_toks) as f32
                / self.total_completion_time as f32)
                * 1000.,
            avg_sample_tok_per_sec: (self.total_toks as f32 / self.total_sampling_time as f32)
                * 1000.,
            total_time_sec: self.total_time as f32 / 1000.,
            total_completion_time_sec: self.total_completion_time as f32 / 1000.,
            total_prompt_time_sec: self.total_prompt_time as f32 / 1000.,
            total_sampling_time_sec: self.total_sampling_time as f32 / 1000.,
        }
    }

    pub fn maybe_send_done_response(
        &self,
        response: ChatCompletionResponse,
        sender: Sender<Response>,
    ) {
        if self.choices.len() == self.n_choices {
            // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
            sender.send(Response::Done(response)).unwrap();
        }
    }

    pub fn maybe_send_streaming_response(&mut self, seq: &Sequence, model: String) {
        if self.streaming_chunks.len() == self.n_choices && self.is_streaming {
            let mut swap_streaming_chunks = vec![];

            std::mem::swap(&mut swap_streaming_chunks, &mut self.streaming_chunks);

            seq.responder()
                .send(Response::Chunk(ChatCompletionChunkResponse {
                    id: seq.id.to_string(),
                    choices: swap_streaming_chunks,
                    created: seq.timestamp,
                    model: model.clone(),
                    system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                    object: "chat.completion.chunk".to_string(),
                }))
                .unwrap();
        }
    }

    pub fn maybe_send_completion_done_response(
        &self,
        response: CompletionResponse,
        sender: Sender<Response>,
    ) {
        if self.completion_choices.len() == self.n_choices {
            // NOTE(EricLBuehler): Unwrap reasoning: The receiver should really be there, otherwise it is their fault.
            sender.send(Response::CompletionDone(response)).unwrap();
        }
    }
}
