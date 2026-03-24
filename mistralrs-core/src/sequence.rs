use crate::{
    get_mut_arcmutex, get_mut_group,
    paged_attention::block_hash::MultiModalFeature,
    pipeline::{text_models_inputs_processor::PagedAttentionMeta, LayerCaches},
    reasoning_parsers::{ReasoningMode, ReasoningParser},
    response::{ChatCompletionChunkResponse, Choice, ChunkChoice, Response, SYSTEM_FINGERPRINT},
    sampler::{Logprobs, Sampler},
    AudioInput, ChatCompletionResponse, Usage,
};
use crate::{
    pipeline::{DiffusionGenerationParams, KvCache},
    response::CompletionChoice,
    tools::ToolCallingMatcher,
    CompletionChunkChoice, CompletionChunkResponse, CompletionResponse, ImageChoice,
    ImageGenerationResponse, ImageGenerationResponseFormat,
};
use candle_core::Tensor;
use std::{
    fmt::Display,
    hash::{DefaultHasher, Hash, Hasher},
    path::PathBuf,
    sync::{Arc, RwLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
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
    GeneratedSpeech,
    ToolCalls,
}

impl Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::Eos => write!(f, "stop"),
            StopReason::Length(_) | StopReason::ModelLength(_) => write!(f, "length"),
            StopReason::StopTok(_) | StopReason::StopString { .. } => write!(f, "stop"),
            StopReason::Canceled => write!(f, "canceled"),
            StopReason::GeneratedImage => write!(f, "generated_image"),
            StopReason::GeneratedSpeech => write!(f, "generated_speech"),
            StopReason::ToolCalls => write!(f, "tool_calls"),
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

#[derive(Clone, Copy)]
pub enum SeqStepType {
    PromptAndDecode,
    OneShot,
}

pub struct SequenceImages {
    images: Vec<image::DynamicImage>,
    hashes: Vec<u64>,
}

#[derive(Clone)]
pub struct SequenceAudios {
    audios: Vec<AudioInput>,
    hashes: Vec<u64>,
}

impl SequenceAudios {
    fn new(input_audios: Vec<AudioInput>) -> Self {
        let hashes = input_audios.iter().map(|a| {
            let mut hasher = DefaultHasher::new();
            for s in &a.samples {
                s.to_bits().hash(&mut hasher);
            }
            a.sample_rate.hash(&mut hasher);
            hasher.finish()
        });
        Self {
            hashes: hashes.collect(),
            audios: input_audios,
        }
    }

    fn clone_audios(&self) -> Vec<AudioInput> {
        self.audios.clone()
    }

    fn audios(&self) -> &[AudioInput] {
        &self.audios
    }

    fn audios_mut(&mut self) -> &mut Vec<AudioInput> {
        &mut self.audios
    }

    fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    fn keep_num_audios(&mut self, audios_to_keep: usize) {
        if self.audios.len() > audios_to_keep {
            let start = self.audios.len() - audios_to_keep;
            self.audios = self.audios[start..].to_vec();
            // Do not do this because we need all the hashes later in the prefix cacher.
            // self.hashes = self.hashes[start..].to_vec();
        }
    }
}

impl SequenceImages {
    fn new(input_images: Vec<image::DynamicImage>) -> Self {
        let hashes = input_images.iter().map(|x| {
            let mut hasher = DefaultHasher::new();
            x.as_bytes().hash(&mut hasher);
            hasher.finish()
        });
        Self {
            hashes: hashes.collect(),
            images: input_images,
        }
    }

    fn clone_images(&self) -> Vec<image::DynamicImage> {
        self.images.clone()
    }

    fn images(&self) -> &[image::DynamicImage] {
        &self.images
    }

    fn images_mut(&mut self) -> &mut Vec<image::DynamicImage> {
        &mut self.images
    }

    fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    fn keep_num_images(&mut self, images_to_keep: usize) {
        if self.images.len() > images_to_keep {
            let start = self.images.len() - images_to_keep;
            self.images = self.images[start..].to_vec();
            // Do not do this because we need all the hashes later in the prefix cacher.
            // self.hashes = self.hashes[start..].to_vec();
        }
    }
}

// Holds all multimodal (vision/diffusion) data for a Sequence.
pub struct MultimodalData {
    pub input_images: Option<SequenceImages>,
    pub input_audios: Option<SequenceAudios>,
    pub cached_pixel_values: Option<Tensor>,
    pub cached_img_thw: Option<Tensor>,
    pub cached_vid_thw: Option<Tensor>,
    /// Complete image grid THW covering ALL images in the sequence (including prefix-cached ones).
    /// Used by Qwen VL models for MRoPE position computation in `get_rope_index`.
    /// Unlike `cached_img_thw`, this is never cleared by `keep_num_images`.
    pub rope_img_grid_thw: Option<Tensor>,
    /// Complete video grid THW covering ALL videos in the sequence (including prefix-cached ones).
    pub rope_vid_grid_thw: Option<Tensor>,
    pub has_changed_prompt: bool,
    pub image_gen_response_format: Option<ImageGenerationResponseFormat>,
    pub diffusion_params: Option<DiffusionGenerationParams>,
    pub image_gen_save_file: Option<PathBuf>,
    /// Per-item multimodal feature positions for prefix caching block hashing.
    /// Each entry records which token range a multimodal item (image/audio) occupies,
    /// so that only blocks overlapping with that item include its content hash.
    /// Set once during the first `process_inputs()` call and never modified thereafter.
    mm_features: Vec<MultiModalFeature>,
}

impl MultimodalData {
    pub fn new(
        input_images: Option<Vec<image::DynamicImage>>,
        input_audios: Option<Vec<AudioInput>>,
        image_gen_response_format: Option<ImageGenerationResponseFormat>,
        diffusion_params: Option<DiffusionGenerationParams>,
        image_gen_save_file: Option<PathBuf>,
    ) -> Self {
        MultimodalData {
            input_images: input_images.map(SequenceImages::new),
            input_audios: input_audios.map(SequenceAudios::new),
            cached_pixel_values: None,
            cached_img_thw: None,
            cached_vid_thw: None,
            rope_img_grid_thw: None,
            rope_vid_grid_thw: None,
            has_changed_prompt: false,
            image_gen_response_format,
            diffusion_params,
            image_gen_save_file,
            mm_features: Vec::new(),
        }
    }

    pub fn take_images(&mut self) -> Option<Vec<image::DynamicImage>> {
        if self.has_changed_prompt {
            if let Some(input_images) = self.input_images.as_mut() {
                let mut images = Vec::new();
                std::mem::swap(&mut images, input_images.images_mut());
                Some(images)
            } else {
                None
            }
        } else {
            self.input_images.as_ref().map(|imgs| imgs.clone_images())
        }
    }

    pub fn clone_images(&self) -> Option<Vec<image::DynamicImage>> {
        self.input_images.as_ref().map(|imgs| imgs.clone_images())
    }

    pub fn images(&self) -> Option<&[image::DynamicImage]> {
        self.input_images.as_ref().map(|imgs| imgs.images())
    }

    pub fn image_hashes(&self) -> Option<&[u64]> {
        self.input_images.as_ref().map(|imgs| imgs.hashes())
    }

    pub fn has_images(&self) -> bool {
        self.input_images
            .as_ref()
            .is_some_and(|imgs| !imgs.images().is_empty())
    }

    pub fn take_audios(&mut self) -> Option<Vec<AudioInput>> {
        if self.has_changed_prompt {
            if let Some(input_audios) = self.input_audios.as_mut() {
                let mut audios = Vec::new();
                std::mem::swap(&mut audios, input_audios.audios_mut());
                Some(audios)
            } else {
                None
            }
        } else {
            self.input_audios.as_ref().map(|imgs| imgs.clone_audios())
        }
    }

    pub fn clone_audios(&self) -> Option<Vec<AudioInput>> {
        self.input_audios.as_ref().map(|a| a.clone_audios())
    }

    pub fn audios(&self) -> Option<&[AudioInput]> {
        self.input_audios.as_ref().map(|a| a.audios())
    }

    pub fn audio_hashes(&self) -> Option<&[u64]> {
        self.input_audios.as_ref().map(|a| a.hashes())
    }

    pub fn has_audios(&self) -> bool {
        self.input_audios
            .as_ref()
            .is_some_and(|a| !a.audios().is_empty())
    }

    pub fn keep_num_audios(&mut self, audios_to_keep: usize) {
        if let Some(auds) = self.input_audios.as_mut() {
            auds.keep_num_audios(audios_to_keep)
        }
    }

    pub fn keep_num_images(&mut self, images_to_keep: usize) {
        if let Some(imgs) = self.input_images.as_mut() {
            imgs.keep_num_images(images_to_keep);
        }
        // Invalidate preprocessed pixel value cache — the trimmed image set
        // no longer matches the cached tensor dimensions (used by Qwen VL models).
        self.cached_pixel_values = None;
        self.cached_img_thw = None;
        self.cached_vid_thw = None;
    }

    pub fn image_gen_response_format(&self) -> Option<ImageGenerationResponseFormat> {
        self.image_gen_response_format
    }

    pub fn image_gen_save_file(&self) -> Option<&PathBuf> {
        self.image_gen_save_file.as_ref()
    }

    pub fn diffusion_params(&self) -> Option<DiffusionGenerationParams> {
        self.diffusion_params.clone()
    }

    /// Per-item multimodal feature positions for prefix caching block hashing.
    pub fn mm_features(&self) -> &[MultiModalFeature] {
        &self.mm_features
    }

    /// Set per-item multimodal feature positions. Should be called once during the
    /// first `process_inputs()` call when all images/audios are available.
    pub fn set_mm_features(&mut self, features: Vec<MultiModalFeature>) {
        self.mm_features = features;
    }
}

/// Scan a token sequence for contiguous runs of a placeholder token ID.
/// Returns `(offset, length)` pairs for each run, in order of appearance.
///
/// Used by vision model input processors to find where each image's placeholder
/// tokens are in the expanded token sequence, so that `MultiModalFeature` entries
/// can be built for position-aware prefix cache block hashing.
pub fn find_image_placeholder_ranges(tokens: &[u32], placeholder_id: u32) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == placeholder_id {
            let start = i;
            while i < tokens.len() && tokens[i] == placeholder_id {
                i += 1;
            }
            ranges.push((start, i - start));
        } else {
            i += 1;
        }
    }
    ranges
}

/// Scan a token sequence for ranges delimited by start and end token IDs (inclusive).
/// Returns `(offset, length)` pairs for each range found.
///
/// Useful for models like Llama4 that wrap each image in `<|image_start|>...<|image_end|>`.
pub fn find_image_delimited_ranges(
    tokens: &[u32],
    start_id: u32,
    end_id: u32,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == start_id {
            let start = i;
            // Find matching end token
            while i < tokens.len() && tokens[i] != end_id {
                i += 1;
            }
            if i < tokens.len() {
                // Include the end token
                ranges.push((start, i - start + 1));
            }
        }
        i += 1;
    }
    ranges
}

/// Build `MultiModalFeature` entries from placeholder token ranges and image hashes.
///
/// Pairs each contiguous run of placeholder tokens (found by `find_image_placeholder_ranges`)
/// with the corresponding image content hash. If there are more images than placeholder ranges
/// (or vice versa), only the overlapping pairs are included.
pub fn build_mm_features_from_ranges(
    ranges: &[(usize, usize)],
    hashes: &[u64],
    kind: &str,
) -> Vec<MultiModalFeature> {
    ranges
        .iter()
        .zip(hashes.iter())
        .map(|(&(offset, length), hash)| MultiModalFeature {
            identifier: format!("{kind}:{hash}"),
            offset,
            length,
        })
        .collect()
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

    // Multimodal data (images, diffusion settings, pixel caches)
    pub multimodal: MultimodalData,

    // Completion requests
    suffix: Option<String>,
    prefix: Option<String>,

    // Speculative
    is_tmp: bool,

    // Prefix caching
    prefill_prompt_toks: Option<Vec<u32>>,
    /// Number of tokens at the start of the prompt that are cached (KV already computed).
    /// These tokens should be skipped during prefill.
    prefix_cache_len: usize,

    // Cache
    normal_cache: Vec<Option<KvCache>>,
    normal_draft_cache: Vec<Option<KvCache>>,
    scaling_cache: Option<Tensor>,
    cache: LayerCaches,
    draft_cache: LayerCaches,
    xlora_cache: Option<LayerCaches>,
    /// For hybrid models: index into the recurrent state pool
    recurrent_state_idx: Option<usize>,

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

    // GPU things
    pub prompt_tok_per_sec: f32,
    pub prompt_timestamp: Option<u128>,
    pub total_prompt_time: Option<u128>,
    pub step_start_instant: Option<Instant>,
    group: Arc<Mutex<SequenceGroup>>,
    state: RwLock<SequenceState>,

    // Tool calls
    pub tools: Option<Arc<ToolCallingMatcher>>,

    // Unified reasoning parser (think tags, channel tags, or Harmony)
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
    reasoning_mode: Option<ReasoningMode>,
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
        input_audios: Option<Vec<AudioInput>>,
        // Paged attention
        block_size: Option<usize>,
        //
        tools: Option<Arc<ToolCallingMatcher>>,
        image_gen_response_format: Option<ImageGenerationResponseFormat>,
        sequence_stepping_type: SeqStepType,
        diffusion_params: Option<DiffusionGenerationParams>,
        image_gen_save_file: Option<PathBuf>,
        // Preallocated KV cache (k,v)
        seq_preallocated_cache: Option<(Tensor, Tensor)>,
        //
        return_raw_logits: bool,
        eos_tokens: Vec<u32>,
    ) -> Self {
        let prompt_len = tokens.len();
        let _ = block_size; // Block management handled by KVCacheManager
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
            recurrent_state_idx: None,
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
            prefix_cache_len: 0,
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
            // Multimodal data
            multimodal: MultimodalData::new(
                input_images,
                input_audios,
                image_gen_response_format,
                diffusion_params,
                image_gen_save_file,
            ),
            tools,
            sequence_stepping_type,
            return_raw_logits,
            token_offset: 0,
            eos_tokens,
            total_prompt_time: None,
            step_start_instant: None,
            reasoning_parser: None,
            reasoning_mode: None,
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

    pub fn prefill_v2_normal(
        mut self,
        cache: Vec<Option<KvCache>>,
        toks: Vec<u32>,
        offset: usize,
    ) -> Self {
        self.normal_cache = cache;
        self.prefill_prompt_toks = Some(toks);
        self.set_state(SequenceState::RunningPrefillPrompt);
        self.token_offset = offset;
        self.prefix_cache_len = offset;
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

    /// Get the number of prefix tokens that are cached (KV already computed).
    /// These tokens should be skipped during prefill.
    pub fn prefix_cache_len(&self) -> usize {
        self.prefix_cache_len
    }

    /// Set the number of prefix tokens that are cached.
    pub fn set_prefix_cache_len(&mut self, len: usize) {
        self.prefix_cache_len = len;
    }

    /// Override the maximum generation length.
    /// If a max_len was already set, keeps the minimum of old and new values.
    pub fn set_max_len(&mut self, max_len: usize) {
        self.max_len = Some(
            self.max_len
                .map_or(max_len, |existing| existing.min(max_len)),
        );
    }

    /// This will also set prompt_len
    pub(crate) fn set_toks_and_reallocate(
        &mut self,
        toks: Vec<u32>,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) {
        self.tokens.clone_from(&toks);
        self.prompt_len = self.tokens.len();

        if let Some(metadata) = paged_attn_metadata {
            // Free and then reallocate with the new token count
            let seq_id = *self.id();
            let num_tokens = self.tokens.len();
            let mut kv_mgr = get_mut_arcmutex!(metadata.kv_cache_manager);
            kv_mgr.free(seq_id);
            if kv_mgr.allocate_slots(seq_id, num_tokens, &[]).is_none() {
                tracing::warn!(
                    "Failed to reallocate KV cache slots for sequence {seq_id} ({num_tokens} tokens)"
                );
            }
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

    pub fn recurrent_state_idx(&self) -> Option<usize> {
        self.recurrent_state_idx
    }

    pub fn set_recurrent_state_idx(&mut self, idx: Option<usize>) {
        self.recurrent_state_idx = idx;
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
    }

    /// Internal api to remove n raw tokens.
    pub(crate) fn remove_tmp_tok(&mut self, n: usize) {
        self.is_tmp = false;
        self.tokens.truncate(self.tokens.len() - n);
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

        // Process token through reasoning parser if enabled
        if let Some(ref mut parser) = self.reasoning_parser {
            if self.reasoning_mode == Some(ReasoningMode::Harmony) {
                parser.process_token(tok.token);
            }
            if !stopped_by_token {
                parser.process_bytes(&completion_bytes);
            }
        }

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
            let mut group = get_mut_group!(self);
            group.n_choices = group.n_choices.saturating_sub(1);
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
            Some(eos_tok) => eos_tok.contains(&tok),
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
            && self.tokens.len().saturating_sub(self.prompt_len) + 1 >= self.max_len.unwrap()
        {
            // add_token will be called after this check
            Some(StopReason::Length(self.max_len.unwrap()))
        } else if self.tokens.len().saturating_sub(self.prompt_len) >= max_model_len {
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

    /// Set the step start instant for accurate prompt timing measurement.
    /// Call this right before step() is called.
    pub fn set_step_start_instant(&mut self) {
        self.step_start_instant = Some(Instant::now());
    }

    pub(crate) fn update_time_info(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time travel has occurred!")
            .as_millis();

        // Prefer the recorded prompt time so it doesn't grow during decode steps.
        // Fall back to the in-flight Instant timing only while the prompt step is running.
        let prompt_time_ms = if let Some(pt) = self.total_prompt_time {
            pt
        } else if let Some(start) = self.step_start_instant {
            start.elapsed().as_millis()
        } else {
            0
        };

        if let Some(ts) = self.prompt_timestamp {
            get_mut_group!(self).total_completion_time = now - ts;
            get_mut_group!(self).total_prompt_time = prompt_time_ms;
        }

        get_mut_group!(self).total_time = now - self.timestamp;

        get_mut_group!(self).total_prompt_toks = self.prompt_len;
        get_mut_group!(self).total_toks = self.len();
    }

    pub fn add_image_choice_to_group(&self, choice: ImageChoice) {
        get_mut_group!(self).image_choices.push(choice);
    }

    pub fn add_speech_pcm_to_group(&self, pcm: Arc<Vec<f32>>, rate: usize, channels: usize) {
        get_mut_group!(self).speech_pcms.push((pcm, rate, channels));
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

    pub fn add_embedding_choice_to_group(&self, embedding: Vec<f32>) {
        get_mut_group!(self).embedding_choices.push(embedding);
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
        self.multimodal.take_images()
    }

    pub fn clone_images(&self) -> Option<Vec<image::DynamicImage>> {
        self.multimodal.clone_images()
    }

    pub fn images(&self) -> Option<&[image::DynamicImage]> {
        self.multimodal.images()
    }

    pub fn image_hashes(&self) -> Option<&[u64]> {
        self.multimodal.image_hashes()
    }

    pub fn has_images(&self) -> bool {
        self.multimodal.has_images()
    }

    pub fn take_audios(&mut self) -> Option<Vec<AudioInput>> {
        self.multimodal.take_audios()
    }

    pub fn clone_audios(&self) -> Option<Vec<AudioInput>> {
        self.multimodal.clone_audios()
    }

    pub fn audios(&self) -> Option<&[AudioInput]> {
        self.multimodal.audios()
    }

    pub fn audio_hashes(&self) -> Option<&[u64]> {
        self.multimodal.audio_hashes()
    }

    pub fn has_audios(&self) -> bool {
        self.multimodal.has_audios()
    }

    /// Keep these last n audios
    pub fn keep_num_audios(&mut self, audios_to_keep: usize) {
        self.multimodal.keep_num_audios(audios_to_keep)
    }

    /// Keep these last n images
    pub fn keep_num_images(&mut self, images_to_keep: usize) {
        self.multimodal.keep_num_images(images_to_keep)
    }

    pub fn image_gen_response_format(&self) -> Option<ImageGenerationResponseFormat> {
        self.multimodal.image_gen_response_format()
    }

    pub fn image_gen_save_file(&self) -> Option<&PathBuf> {
        self.multimodal.image_gen_save_file()
    }

    /// Per-item multimodal feature positions for prefix caching block hashing.
    pub fn mm_features(&self) -> &[MultiModalFeature] {
        self.multimodal.mm_features()
    }

    /// Set per-item multimodal feature positions. Should be called once during the
    /// first `process_inputs()` call when all images/audios are available.
    pub fn set_mm_features(&mut self, features: Vec<MultiModalFeature>) {
        self.multimodal.set_mm_features(features);
    }

    /// Count the number of multimodal items whose placeholder tokens fall entirely
    /// within the prefix cache. Used by vision model inputs_processors to trim
    /// pixel_values so they match only the non-cached image placeholder positions.
    pub fn count_prefix_cached_mm_items(&self) -> usize {
        let prefix_len = self.prefix_cache_len();
        if prefix_len == 0 {
            return 0;
        }
        self.mm_features()
            .iter()
            .filter(|f| f.offset + f.length <= prefix_len)
            .count()
    }

    /// Count the number of multimodal items of a specific kind whose placeholder
    /// tokens fall entirely within the prefix cache. `kind` should match the
    /// prefix used in `build_mm_features_from_ranges`, e.g. `"img"` or `"audio"`.
    pub fn count_prefix_cached_mm_items_by_kind(&self, kind: &str) -> usize {
        let prefix_len = self.prefix_cache_len();
        if prefix_len == 0 {
            return 0;
        }
        let identifier_prefix = format!("{kind}:");
        self.mm_features()
            .iter()
            .filter(|f| {
                f.offset + f.length <= prefix_len
                    && f.identifier.starts_with(identifier_prefix.as_str())
            })
            .count()
    }

    pub fn sequence_stepping_type(&self) -> &SeqStepType {
        &self.sequence_stepping_type
    }

    pub fn get_diffusion_diffusion_params(&self) -> Option<DiffusionGenerationParams> {
        self.multimodal.diffusion_params()
    }

    pub fn eos_tokens(&self) -> &[u32] {
        &self.eos_tokens
    }

    // === Unified Reasoning Support ===

    /// Get the active reasoning mode, if any.
    pub fn reasoning_mode(&self) -> Option<ReasoningMode> {
        self.reasoning_mode
    }

    /// Whether any reasoning parser needs special tokens in decoded text.
    pub fn needs_special_tokens(&self) -> bool {
        self.reasoning_parser.is_some()
    }

    /// Enable reasoning with the given parser and mode.
    pub fn enable_reasoning(&mut self, mode: ReasoningMode, parser: Box<dyn ReasoningParser>) {
        self.reasoning_parser = Some(parser);
        self.reasoning_mode = Some(mode);
    }

    /// Check if this sequence is in Harmony mode
    pub fn is_harmony_mode(&self) -> bool {
        self.reasoning_mode == Some(ReasoningMode::Harmony)
    }

    /// Get the reasoning content delta since last call (for streaming).
    pub fn get_reasoning_content_delta(&mut self) -> Option<String> {
        self.reasoning_parser.as_mut()?.get_reasoning_delta()
    }

    /// Get the response content delta since last call (for streaming).
    pub fn get_response_content_delta(&mut self) -> Option<String> {
        self.reasoning_parser.as_mut()?.get_content_delta()
    }

    /// Get accumulated reasoning content (for non-streaming).
    pub fn get_reasoning_content(&self) -> Option<String> {
        self.reasoning_parser.as_ref()?.reasoning_content()
    }

    /// Get accumulated response content (for non-streaming).
    pub fn get_response_content(&self) -> Option<String> {
        self.reasoning_parser.as_ref()?.content()
    }

    /// Finalize all reasoning parsers at end of stream.
    pub fn finalize_reasoning(&mut self) {
        if let Some(ref mut p) = self.reasoning_parser {
            p.finalize();
        }
    }

    /// Check if the reasoning parser has detected any tool calls
    pub fn has_harmony_tool_calls(&self) -> bool {
        self.reasoning_parser
            .as_ref()
            .is_some_and(|p| p.has_tool_calls())
    }

    /// Get all tool calls from the reasoning parser (finalizes any pending tool call)
    pub fn get_harmony_tool_calls(&mut self) -> Vec<crate::reasoning_parsers::HarmonyToolCall> {
        self.reasoning_parser
            .as_mut()
            .map(|p| p.finalize_tool_calls())
            .unwrap_or_default()
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
    speech_pcms: Vec<(Arc<Vec<f32>>, usize, usize)>, // (pcm, rate, channels)
    raw_choices: Vec<(Vec<Tensor>, Vec<u32>)>,
    embedding_choices: Vec<Vec<f32>>,
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
            speech_pcms: Vec::new(),
            raw_choices: Vec::new(),
            embedding_choices: Vec::new(),
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
            completion_tokens: self.total_toks.saturating_sub(self.total_prompt_toks),
            prompt_tokens: self.total_prompt_toks,
            total_tokens: self.total_toks,
            avg_tok_per_sec: if self.total_time > 0 {
                (self.total_toks as f32 / self.total_time as f32) * 1000.
            } else {
                0.0
            },
            avg_prompt_tok_per_sec: if self.total_prompt_time > 0 {
                (self.total_prompt_toks as f32 / self.total_prompt_time as f32) * 1000.
            } else {
                0.0
            },
            avg_compl_tok_per_sec: if self.total_completion_time > 0 {
                (self.total_toks.saturating_sub(self.total_prompt_toks) as f32
                    / self.total_completion_time as f32)
                    * 1000.
            } else {
                0.0
            },
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

    pub async fn maybe_send_embedding_done_response(
        &self,
        sender: Sender<Response>,
    ) -> Result<(), SendError<Response>> {
        if self.embedding_choices.len() == self.n_choices {
            assert_eq!(self.embedding_choices.len(), 1);
            let embeddings = self.embedding_choices[0].clone();
            let prompt_tokens = self.total_prompt_toks;
            let total_tokens = self.total_toks;
            sender
                .send(Response::Embeddings {
                    embeddings,
                    prompt_tokens,
                    total_tokens,
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

    pub async fn maybe_send_speech_response(
        &self,
        sender: Sender<Response>,
    ) -> Result<(), SendError<Response>> {
        assert_eq!(self.speech_pcms.len(), 1);

        let (pcm, rate, channels) = self.speech_pcms[0].clone();
        sender
            .send(Response::Speech {
                pcm,
                rate,
                channels,
            })
            .await?;

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
                    created: seq.creation_time() as u128,
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
                    created: seq.creation_time() as u128,
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc::channel;

    fn make_test_sequence() -> Sequence {
        let (tx, _rx) = channel(1);
        let sampler =
            Sampler::new(None, 0, None, None, None, None, None, 32, 1.0, 0.0, vec![]).unwrap();
        let group = Arc::new(Mutex::new(SequenceGroup::new(1, false, true, None)));

        Sequence::new_waiting(
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            "prompt".to_string(),
            0,
            0,
            0,
            tx,
            sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            vec![],
        )
    }

    #[test]
    fn prefill_v2_normal_sets_prefix_cache_len_for_multimodal_trimming() {
        let mut seq = make_test_sequence();
        seq.set_mm_features(vec![
            MultiModalFeature {
                identifier: "img:123".to_string(),
                offset: 0,
                length: 3,
            },
            MultiModalFeature {
                identifier: "img:456".to_string(),
                offset: 4,
                length: 3,
            },
            MultiModalFeature {
                identifier: "audio:789".to_string(),
                offset: 7,
                length: 1,
            },
        ]);

        let seq = seq.prefill_v2_normal(vec![], vec![7, 8], 4);

        assert_eq!(seq.prefix_cache_len(), 4);
        assert_eq!(seq.count_prefix_cached_mm_items_by_kind("img"), 1);
        assert_eq!(seq.count_prefix_cached_mm_items_by_kind("audio"), 0);
    }
}
