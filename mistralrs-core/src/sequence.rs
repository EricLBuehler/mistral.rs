use crate::{
    get_mut_group,
    pipeline::{text_models_inputs_processor::PagedAttentionMeta, LayerCaches},
    response::{ChatCompletionChunkResponse, Choice, ChunkChoice, Response, SYSTEM_FINGERPRINT},
    sampler::{Logprobs, Sampler},
    ChatCompletionResponse, Usage,
};
use crate::{
    paged_attention::{BlockEngineSequence, LogicalTokenBlock},
    pipeline::{DiffusionGenerationParams, KvCache},
    response::CompletionChoice,
    tools::ToolCallingMatcher,
    CompletionChunkChoice, CompletionChunkResponse, CompletionResponse, ImageChoice,
    ImageGenerationResponse, ImageGenerationResponseFormat,
};
use candle_core::Tensor;
use std::{
    fmt::Display,
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{
    mpsc::{error::SendError, Sender},
    Mutex, MutexGuard,
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StopReason {
    Eos,
    StopTok(u32),
    Length(usize),
    ModelLength(usize),
    StopString {
        stop_string_idx: usize,
        completion_bytes_pos: usize,
    },
    Canceled,
    GeneratedImage,
}

impl Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::Eos => write!(f, "stop"),
            StopReason::Length(_) | StopReason::ModelLength(_) => write!(f, "length"),
            StopReason::StopTok(_) | StopReason::StopString { .. } => write!(f, "stop"),
            StopReason::Canceled => write!(f, "canceled"),
            StopReason::GeneratedImage => write!(f, "generated-image"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SequenceState {
    Done(StopReason),
    RunningPrompt,
    RunningCompletion,
    Waiting,
    Error,
    RunningPrefillPrompt,
    // For PagedAttention:
    FinishedAborted,
    FinishedIgnored,
    Swapped,
}

pub enum SequenceRecognizer {
    Llguidance(Box<llguidance::Matcher>),
    None,
}

enum SequenceCustomMetadata {
    PagedAttention {
        logical_token_blocks: Vec<LogicalTokenBlock>,
        block_size: usize,
    },
    None,
}

macro_rules! blocks_to_add_new_tok {
    ($logical_token_blocks:expr) => {{
        let last = $logical_token_blocks.last();
        if !last.is_some_and(|last| last.is_full() || last.is_empty()) {
            // If we have space
            0
        } else {
            1
        }
    }};
}

impl SequenceCustomMetadata {
    fn append_token_to_blocks(&mut self, tok: usize) {
        match self {
            Self::PagedAttention {
                logical_token_blocks,
                block_size,
            } => {
                let last = logical_token_blocks.last_mut();
                match last {
                    Some(last) => {
                        last.append_token_id(tok);
                    }
                    None => {
                        logical_token_blocks.push(LogicalTokenBlock::new(*block_size));
                        logical_token_blocks
                            .last_mut()
                            .unwrap()
                            .append_token_id(tok);
                    }
                }
                if logical_token_blocks.last().as_ref().unwrap().is_full() {
                    logical_token_blocks.push(LogicalTokenBlock::new(*block_size));
                }
            }
            Self::None => (),
        }
    }

    fn pop_token_from_blocks(&mut self) {
        match self {
            Self::PagedAttention {
                logical_token_blocks,
                block_size: _,
            } => {
                let last = logical_token_blocks.last_mut().unwrap();
                last.pop_token();
            }
            Self::None => (),
        }
    }

    fn append_tokens_to_blocks(&mut self, toks: Vec<usize>) {
        for tok in toks {
            self.append_token_to_blocks(tok);
        }
    }

    fn remove_tokens_from_blocks(&mut self, n: usize) {
        for _ in 0..n {
            self.pop_token_from_blocks();
        }
    }
}

#[derive(Clone, Copy)]
pub enum SeqStepType {
    PromptAndDecode,
    OneShot,
}

pub struct Sequence {
    // Metadata, const
    id: usize,
    prompt_len: usize,
    max_len: Option<usize>,
    timestamp: u128,
    sampler: Arc<Sampler>,
    stop_tokens: Vec<u32>,
    stop_strings: Vec<String>,
    return_logprobs: bool,
    responder: Sender<Response>,
    response_index: usize,
    creation_time: u64,
    prompt: String,
    sequence_stepping_type: SeqStepType,
    pub(crate) return_raw_logits: bool,
    token_offset: usize,
    eos_tokens: Vec<u32>,

    // Image generation
    image_gen_response_format: Option<ImageGenerationResponseFormat>,
    diffusion_params: Option<DiffusionGenerationParams>,

    // Completion requests
    suffix: Option<String>,
    prefix: Option<String>,

    // Speculative
    is_tmp: bool,

    // Prefix caching
    prefill_prompt_toks: Option<Vec<u32>>,

    // Cache
    normal_cache: Vec<Option<KvCache>>,
    normal_draft_cache: Vec<Option<KvCache>>,
    scaling_cache: Option<Tensor>,
    cache: LayerCaches,
    draft_cache: LayerCaches,
    xlora_cache: Option<LayerCaches>,

    // Preallocated KV cache (k,v)
    seq_preallocated_cache: Option<(Tensor, Tensor)>,

    // Mutables
    tokens: Vec<u32>,
    logprobs: Vec<Logprobs>,
    cumulative_logprob: f32,
    last_logprob: f32,
    last_completion_bytes_len: usize,
    last_is_done: Option<StopReason>,
    completion_bytes: Vec<u8>,
    stream_idx: usize,
    pub recognizer: SequenceRecognizer,
    scheduling_urgency: usize, // The number of passes since scheduling
    input_images: Option<Vec<image::DynamicImage>>,
    pub cached_pixel_values: Option<Tensor>,
    pub cached_img_thw: Option<Tensor>,
    pub cached_vid_thw: Option<Tensor>,
    pub has_changed_prompt: bool,

    // GPU things
    pub prompt_tok_per_sec: f32,
    pub prompt_timestamp: Option<u128>,
    pub total_prompt_time: Option<u128>,
    group: Arc<Mutex<SequenceGroup>>,
    state: RwLock<SequenceState>,

    // Custom backend metadata
    custom_metadata: SequenceCustomMetadata,

    // Tool calls
    pub tools: Option<Arc<ToolCallingMatcher>>,
}

impl BlockEngineSequence for Sequence {
    fn blocks_to_add_new_tok(&self) -> usize {
        match &self.custom_metadata {
            SequenceCustomMetadata::PagedAttention {
                logical_token_blocks,
                block_size: _,
            } => {
                blocks_to_add_new_tok!(logical_token_blocks)
            }
            SequenceCustomMetadata::None => unreachable!(),
        }
    }

    fn get_id(&self) -> usize {
        self.id
    }

    fn get_logical_token_blocks(&self) -> usize {
        match &self.custom_metadata {
            SequenceCustomMetadata::PagedAttention {
                logical_token_blocks,
                block_size: _,
            } => logical_token_blocks.len(),
            SequenceCustomMetadata::None => unreachable!(),
        }
    }
}

impl Sequence {
    #[allow(clippy::too_many_arguments)]
    pub fn new_waiting(
        tokens: Vec<u32>,
        prompt: String,
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
        group: Arc<Mutex<SequenceGroup>>,
        response_index: usize,
        creation_time: u64,
        recognizer: SequenceRecognizer,
        suffix: Option<String>,
        prefix: Option<String>,
        input_images: Option<Vec<image::DynamicImage>>,
        // Paged attention
        block_size: Option<usize>,
        //
        tools: Option<Arc<ToolCallingMatcher>>,
        image_gen_response_format: Option<ImageGenerationResponseFormat>,
        sequence_stepping_type: SeqStepType,
        diffusion_params: Option<DiffusionGenerationParams>,
        // Preallocated KV cache (k,v)
        seq_preallocated_cache: Option<(Tensor, Tensor)>,
        //
        return_raw_logits: bool,
        eos_tokens: Vec<u32>,
    ) -> Self {
        let prompt_len = tokens.len();
        let mut custom_metadata = if let Some(block_size) = block_size {
            SequenceCustomMetadata::PagedAttention {
                logical_token_blocks: Vec::new(),
                block_size,
            }
        } else {
            SequenceCustomMetadata::None
        };
        custom_metadata
            .append_tokens_to_blocks(tokens.iter().map(|x| *x as usize).collect::<Vec<_>>());
        Self {
            tokens,
            prompt,
            logprobs: Vec::new(),
            prompt_len,
            id,
            timestamp,
            state: RwLock::new(SequenceState::Waiting),
            normal_cache: vec![None; layers],
            normal_draft_cache: vec![None; layers],
            cache: vec![None; layers],
            draft_cache: vec![None; layers],
            xlora_cache: if is_xlora {
                Some(vec![None; layers])
            } else {
                None
            },
            seq_preallocated_cache,
            responder,
            sampler: sampler.into(),
            stop_tokens,
            stop_strings,
            max_len,
            return_logprobs,
            prompt_tok_per_sec: 0.,
            prompt_timestamp: None,
            group,
            scaling_cache: None,
            response_index,
            creation_time,
            recognizer,
            prefill_prompt_toks: None,
            suffix,
            prefix,
            cumulative_logprob: 0.,
            completion_bytes: Vec::new(),
            stream_idx: 0,
            last_completion_bytes_len: 0,
            last_logprob: 0.0,
            last_is_done: None,
            is_tmp: false,
            scheduling_urgency: 0,
            input_images,
            custom_metadata,
            tools,
            image_gen_response_format,
            sequence_stepping_type,
            diffusion_params,
            cached_pixel_values: None,
            cached_img_thw: None,
            cached_vid_thw: None,
            return_raw_logits,
            token_offset: 0,
            eos_tokens,
            has_changed_prompt: false,
            total_prompt_time: None,
        }
    }

    pub fn add_urgency(mut self) -> Self {
        self.scheduling_urgency += 1;
        self
    }

    pub fn reset_urgency(mut self) -> Self {
        self.scheduling_urgency = 0;
        self
    }

    /// Simple metric: (scheduling urgency) + log2(length)
    /// Takes into account: urgency (scales linear) and length (scales logarithmic)
    /// Scaling urgency is the number of scheduling passes where we have not been scheduled.
    pub fn compute_priority(&self) -> f64 {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        (self.scheduling_urgency as f64) + (self.len() as f64).log2()
    }

    pub fn prefill(
        mut self,
        cache: LayerCaches,
        xlora_cache: Option<LayerCaches>,
        toks: Vec<u32>,
    ) -> Self {
        self.cache = cache;
        self.xlora_cache = xlora_cache;
        self.prefill_prompt_toks = Some(toks);
        self.set_state(SequenceState::RunningPrefillPrompt);
        self
    }

    pub fn prefill_v2(
        mut self,
        cache: Vec<Option<KvCache>>,
        toks: Vec<u32>,
        offset: usize,
    ) -> Self {
        self.normal_cache = cache;
        self.prefill_prompt_toks = Some(toks);
        self.set_state(SequenceState::RunningPrefillPrompt);
        self.token_offset = offset;
        self
    }

    /// This is the number of tokens. If the KV cache is Some, then it will use that.
    pub fn len(&self) -> usize {
        if let Some(toks) = &self.prefill_prompt_toks {
            return toks.len();
        }
        if self.is_tmp {
            return self.tokens.len();
        }
        // Use xlora cache first because of non granular
        if self.xlora_cache.as_ref().is_some_and(|c| c[0].is_some()) {
            self.xlora_cache.as_ref().unwrap()[0]
                .as_ref()
                .unwrap()
                .0
                .dims()[2]
                + 1
        } else if let Some((_, x)) = &self.cache[0] {
            x.dims()[2] + 1
        } else {
            self.tokens.len()
        }
    }

    pub fn id(&self) -> &usize {
        &self.id
    }

    pub fn is_running(&self) -> bool {
        matches!(
            *self.state.read().unwrap(),
            SequenceState::RunningCompletion | SequenceState::RunningPrompt // | SequenceState::RunningPrefillPrompt
        )
    }

    pub fn is_completion(&self) -> bool {
        matches!(
            *self.state.read().unwrap(),
            SequenceState::RunningCompletion
        )
    }

    pub fn is_prompt(&self) -> bool {
        matches!(
            *self.state.read().unwrap(),
            SequenceState::RunningPrompt | SequenceState::RunningPrefillPrompt
        )
    }

    pub fn is_waiting(&self) -> bool {
        matches!(*self.state.read().unwrap(), SequenceState::Waiting)
    }

    pub fn is_finished_paged_attn(&self) -> bool {
        matches!(
            *self.state.read().unwrap(),
            SequenceState::FinishedAborted
                | SequenceState::FinishedIgnored
                | SequenceState::Done(_)
        )
    }

    pub fn get_toks(&self) -> &[u32] {
        if let Some(toks) = &self.prefill_prompt_toks {
            return toks;
        }
        &self.tokens
    }

    pub fn get_initial_prompt(&self) -> &str {
        &self.prompt
    }

    pub fn set_initial_prompt(&mut self, new: String) {
        self.prompt = new;
    }

    pub fn token_offset(&self) -> usize {
        self.token_offset
    }

    /// This will also set prompt_len
    pub(crate) fn set_toks_and_reallocate(
        &mut self,
        toks: Vec<u32>,
        paged_attn_metadata: Option<&mut PagedAttentionMeta<'_>>,
    ) {
        self.tokens.clone_from(&toks);
        self.prompt_len = self.tokens.len();
        // Handle possible block engine
        match &mut self.custom_metadata {
            SequenceCustomMetadata::PagedAttention {
                logical_token_blocks,
                block_size: _,
            } => {
                logical_token_blocks.clear();
            }
            SequenceCustomMetadata::None => (),
        }
        self.custom_metadata
            .append_tokens_to_blocks(toks.iter().map(|x| *x as usize).collect::<Vec<_>>());

        if let Some(metadata) = paged_attn_metadata {
            // Free and then reallocate as appropriate
            metadata.block_engine.free_sequence(*self.id());
            metadata.block_engine.allocate(self);
        }
    }

    pub fn completion_bytes(&self) -> &[u8] {
        &self.completion_bytes
    }

    pub fn preallocated_cache(&self) -> Option<&(Tensor, Tensor)> {
        self.seq_preallocated_cache.as_ref()
    }

    pub fn normal_cache(&mut self) -> &mut Vec<Option<KvCache>> {
        &mut self.normal_cache
    }

    pub fn normal_draft_cache(&mut self) -> &mut Vec<Option<KvCache>> {
        &mut self.normal_draft_cache
    }

    pub fn cache(&mut self) -> &mut Vec<Option<(Tensor, Tensor)>> {
        &mut self.cache
    }

    pub fn draft_cache(&mut self) -> &mut Vec<Option<(Tensor, Tensor)>> {
        &mut self.draft_cache
    }

    pub fn xlora_cache(&mut self) -> &mut Vec<Option<(Tensor, Tensor)>> {
        self.xlora_cache.as_mut().expect("No X-LoRA cache.")
    }

    pub fn scaling_cache(&mut self) -> &mut Option<Tensor> {
        &mut self.scaling_cache
    }

    pub fn is_xlora(&self) -> bool {
        self.xlora_cache.is_some()
    }

    pub fn sampler(&mut self) -> Arc<Sampler> {
        self.sampler.clone()
    }

    /// Add a some prefill tokens. Only meant for internal speculative decoding usage.
    pub fn set_prefill_toks(&mut self, toks: Vec<u32>) {
        self.prefill_prompt_toks = Some(toks)
    }

    /// Remove the prefill tokens.
    pub fn reset_prefill_toks(&mut self) {
        self.prefill_prompt_toks = None
    }

    /// Internal api to add one raw token.
    pub(crate) fn add_tmp_tok(&mut self, tok: u32) {
        self.is_tmp = true;
        self.tokens.push(tok);
        // Handle possible block engine
        self.custom_metadata.append_token_to_blocks(tok as usize);
    }

    /// Internal api to remove n raw tokens.
    pub(crate) fn remove_tmp_tok(&mut self, n: usize) {
        self.is_tmp = false;
        self.tokens.truncate(self.tokens.len() - n);
        // Handle possible block engine
        self.custom_metadata.remove_tokens_from_blocks(n);
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
            // Completion bytes is used to check for stop strings, and as the response buffer.
            // We don't need to add stop tokens to the completion bytes to check for stop strings.
            // And by not adding it here, we can avoid having to delete these tokens from the output.
            self.completion_bytes.extend_from_slice(&completion_bytes);
            self.last_completion_bytes_len = completion_bytes.len();
        }
        self.last_logprob = tok.logprob;
        self.last_is_done = *is_done;

        self.custom_metadata
            .append_token_to_blocks(tok.token as usize);

        self.cumulative_logprob += tok.logprob;
        self.tokens.push(tok.token);
        self.logprobs.push(tok);
        self.reset_prefill_toks();
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
        *self.state.write().unwrap() = state;
    }

    pub fn getstate(&self) -> SequenceState {
        *self.state.read().unwrap()
    }

    pub fn is_done(
        &self,
        tok: u32,
        eos_tok: Option<&[u32]>,
        max_model_len: usize,
    ) -> Option<StopReason> {
        let is_eos = match eos_tok {
            Some(eos_tok) => eos_tok.iter().any(|t| *t == tok),
            None => false,
        };
        if is_eos {
            Some(StopReason::Eos)
        } else if matches!(
            &*self.state.read().unwrap(),
            SequenceState::Done(StopReason::Canceled)
        ) {
            Some(StopReason::Canceled)
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
                    if let Some(pos) = galil_seiferas::gs_find(&self.completion_bytes, s.as_bytes())
                    {
                        return Some(StopReason::StopString {
                            stop_string_idx: idx,
                            completion_bytes_pos: pos,
                        });
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
        let new_decoded = self.peek_delta();
        if matches!(new_decoded, Ok(Some(_))) {
            self.stream_idx = self.completion_bytes.len();
        }
        new_decoded
    }

    /// Peeks at the delta between the last two decoded sequences, but does not advance the stream index.
    pub fn peek_delta(&self) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let is_first = self.stream_idx == 0;
        let new_decoded = String::from_utf8_lossy(&self.completion_bytes[self.stream_idx..]);
        // Check if the sequence ends with valid utf8, if not skip it as it probably is a multi token sequence
        if new_decoded.ends_with('�') {
            return Ok(None);
        }

        // The first token usually starts with a space. We don't want to add that to the delta.
        // Since we're using the completion_bytes, we need to take care of that ourselves.
        // Had we used HF's Tokenizer, it would have taken care of that for us.
        if is_first {
            return Ok(Some(new_decoded.trim_start().to_string()));
        }
        Ok(Some(new_decoded.to_string()))
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
            get_mut_group!(self).total_completion_time = now - ts;
            get_mut_group!(self).total_prompt_time = self.total_prompt_time.unwrap();
        }

        get_mut_group!(self).total_time = now - self.timestamp;

        get_mut_group!(self).total_prompt_toks = self.prompt_len;
        get_mut_group!(self).total_toks = self.len();
    }

    pub fn add_image_choice_to_group(&self, choice: ImageChoice) {
        get_mut_group!(self).image_choices.push(choice);
        self.update_time_info();
    }

    pub fn add_choice_to_group(&self, choice: Choice) {
        get_mut_group!(self).choices.push(choice);
        self.update_time_info();
    }

    pub fn add_raw_choice_to_group(&self, logit_chunks: Vec<Tensor>) {
        get_mut_group!(self)
            .raw_choices
            .push((logit_chunks, self.tokens.clone()));
        self.update_time_info();
    }

    pub fn add_completion_choice_to_group(&self, mut choice: CompletionChoice) {
        choice.text = format!(
            "{}{}{}",
            self.prefix.as_deref().unwrap_or(""),
            choice.text,
            self.suffix.as_deref().unwrap_or("")
        );
        get_mut_group!(self)
            .completion_choices
            .push((self.cumulative_logprob, choice));
        self.update_time_info();
    }

    pub fn get_response_index(&self) -> usize {
        self.response_index
    }

    pub fn get_mut_group(&self) -> MutexGuard<'_, SequenceGroup> {
        get_mut_group!(self)
    }

    pub fn add_streaming_chunk_choice_to_group(&self, chunk: ChunkChoice) {
        get_mut_group!(self).chat_streaming_chunks.push(chunk);
        self.update_time_info();
    }

    pub fn add_streaming_completion_chunk_choice_to_group(&self, chunk: CompletionChunkChoice) {
        get_mut_group!(self).completion_streaming_chunks.push(chunk);
        self.update_time_info();
    }

    pub fn take_images(&mut self) -> Option<Vec<image::DynamicImage>> {
        // So that we don't keep having an image after the actual prompt
        if self.has_changed_prompt {
            // Actual prompt
            self.input_images.take()
        } else {
            // Dummy inputs processing
            self.input_images.clone()
        }
    }

    pub fn clone_images(&mut self) -> Option<Vec<image::DynamicImage>> {
        self.input_images.clone()
    }

    pub fn images(&self) -> Option<&[image::DynamicImage]> {
        self.input_images.as_deref()
    }

    pub fn has_images(&self) -> bool {
        self.input_images
            .as_ref()
            .is_some_and(|images| !images.is_empty())
    }

    pub fn image_gen_response_format(&self) -> Option<ImageGenerationResponseFormat> {
        self.image_gen_response_format
    }

    pub fn sequence_stepping_type(&self) -> &SeqStepType {
        &self.sequence_stepping_type
    }

    pub fn get_diffusion_diffusion_params(&self) -> Option<DiffusionGenerationParams> {
        self.diffusion_params.clone()
    }

    pub fn eos_tokens(&self) -> &[u32] {
        &self.eos_tokens
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
    choices: Vec<Choice>,
    image_choices: Vec<ImageChoice>,
    raw_choices: Vec<(Vec<Tensor>, Vec<u32>)>,
    completion_choices: Vec<(f32, CompletionChoice)>,
    pub chat_streaming_chunks: Vec<ChunkChoice>,
    pub completion_streaming_chunks: Vec<CompletionChunkChoice>,
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
            image_choices: Vec::new(),
            raw_choices: Vec::new(),
            completion_choices: Vec::new(),
            n_choices,
            total_prompt_toks: 0,
            total_toks: 0,
            total_prompt_time: 0,
            total_time: 0,
            total_completion_time: 0,
            chat_streaming_chunks: Vec::new(),
            completion_streaming_chunks: Vec::new(),
            is_streaming,
            is_chat,
            best_of,
        }
    }

    pub fn get_choices(&self) -> &[Choice] {
        &self.choices
    }

    /// This may apply the best_of.
    pub fn get_completion_choices(&self) -> Vec<CompletionChoice> {
        if let Some(best_of) = self.best_of {
            let mut choices = self.completion_choices.clone();
            // Sort by descending logprobs
            choices.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("No ordering."));
            choices
                .into_iter()
                .take(best_of)
                .map(|(_, x)| x)
                .collect::<Vec<_>>()
        } else {
            self.completion_choices
                .clone()
                .into_iter()
                .map(|(_, x)| x)
                .collect::<Vec<_>>()
        }
    }

    pub fn get_image_choices(&self) -> &[ImageChoice] {
        &self.image_choices
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
            total_time_sec: self.total_time as f32 / 1000.,
            total_completion_time_sec: self.total_completion_time as f32 / 1000.,
            total_prompt_time_sec: self.total_prompt_time as f32 / 1000.,
        }
    }

    pub async fn maybe_send_chat_done_response(
        &self,
        response: ChatCompletionResponse,
        sender: Sender<Response>,
    ) -> Result<(), SendError<Response>> {
        if self.choices.len() == self.n_choices {
            sender.send(Response::Done(response)).await?;
        }

        Ok(())
    }

    pub async fn maybe_send_raw_done_response(
        &self,
        sender: Sender<Response>,
    ) -> Result<(), SendError<Response>> {
        if self.raw_choices.len() == self.n_choices {
            assert_eq!(self.raw_choices.len(), 1);
            let (logits_chunks, tokens) = self.raw_choices[0].clone();
            sender
                .send(Response::Raw {
                    logits_chunks,
                    tokens,
                })
                .await?;
        }

        Ok(())
    }

    pub async fn maybe_send_image_gen_response(
        &self,
        response: ImageGenerationResponse,
        sender: Sender<Response>,
    ) -> Result<(), SendError<Response>> {
        if self.image_choices.len() == self.n_choices {
            sender.send(Response::ImageGeneration(response)).await?;
        }

        Ok(())
    }

    pub async fn maybe_send_streaming_response(
        &mut self,
        seq: &Sequence,
        model: String,
        usage_opt: Option<Usage>,
    ) -> Result<(), Box<SendError<Response>>> {
        if self.chat_streaming_chunks.len() == self.n_choices && self.is_streaming {
            let mut swap_streaming_chunks = vec![];

            std::mem::swap(&mut swap_streaming_chunks, &mut self.chat_streaming_chunks);

            seq.responder()
                .send(Response::Chunk(ChatCompletionChunkResponse {
                    id: seq.id.to_string(),
                    choices: swap_streaming_chunks,
                    created: seq.timestamp,
                    model: model.clone(),
                    system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                    object: "chat.completion.chunk".to_string(),
                    usage: usage_opt,
                }))
                .await?;
        } else if self.completion_streaming_chunks.len() == self.n_choices && self.is_streaming {
            let mut swap_streaming_chunks = vec![];

            std::mem::swap(
                &mut swap_streaming_chunks,
                &mut self.completion_streaming_chunks,
            );

            seq.responder()
                .send(Response::CompletionChunk(CompletionChunkResponse {
                    id: seq.id.to_string(),
                    choices: swap_streaming_chunks,
                    created: seq.timestamp,
                    model: model.clone(),
                    system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                    object: "text_completion".to_string(),
                }))
                .await?;
        }
        Ok(())
    }

    pub async fn maybe_send_completion_done_response(
        &self,
        response: CompletionResponse,
        sender: Sender<Response>,
    ) -> Result<(), Box<SendError<Response>>> {
        if self.completion_choices.len() == self.n_choices {
            sender.send(Response::CompletionDone(response)).await?;
        }
        Ok(())
    }
}
