use crate::{
    get_mut_arcmutex, get_mut_group,
    paged_attention::block_hash::{MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind},
    pipeline::{text_models_inputs_processor::PagedAttentionMeta, LayerCaches},
    reasoning_parsers::{ReasoningMode, ReasoningParser},
    response::{ChatCompletionChunkResponse, Choice, ChunkChoice, Response, SYSTEM_FINGERPRINT},
    sampler::{Logprobs, Sampler},
    speculative::{SpeculativeProposalDistribution, SpeculativeTokens},
    AdapterGenerationId, AdapterLease, AudioInput, ChatCompletionResponse, PromptTokensDetails,
    Usage, VideoInput,
};
use crate::{
    pipeline::{DiffusionGenerationParams, KvCache},
    response::CompletionChoice,
    tools::ToolCallState,
    CompletionChunkChoice, CompletionChunkResponse, CompletionResponse, ImageChoice,
    ImageGenerationResponse, ImageGenerationResponseFormat,
};
use candle_core::Tensor;
use rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::{
    collections::{HashSet, VecDeque},
    fmt::Display,
    hash::{DefaultHasher, Hash, Hasher},
    ops::Range,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::sync::{
    mpsc::{error::SendError, Sender},
    Mutex, MutexGuard,
};

pub type SeqPreallocatedCache = Vec<Option<(Tensor, Tensor)>>;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ActiveMultimodalWindow {
    item_range: Range<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillTokenView {
    PrefixInclusive,
    SuffixOnly,
}

struct PrefillTokens {
    tokens: Vec<u32>,
    view: PrefillTokenView,
}

struct PendingStreamingEmission {
    bytes: Vec<u8>,
    logprobs: Logprobs,
}

pub(crate) struct StreamingEmission {
    pub text: String,
    pub bytes: Vec<u8>,
    pub logprobs: Logprobs,
}

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

impl StopReason {
    fn metric_label(self) -> &'static str {
        match self {
            StopReason::Eos => "stop",
            StopReason::Length(_) | StopReason::ModelLength(_) => "length",
            StopReason::StopTok(_) | StopReason::StopString { .. } => "stop",
            StopReason::Canceled => "canceled",
            StopReason::GeneratedImage => "generated_image",
            StopReason::GeneratedSpeech => "generated_speech",
            StopReason::ToolCalls => "tool_calls",
        }
    }
}

fn find_earliest_stop_string(bytes: &[u8], stop_strings: &[String]) -> Option<(usize, usize)> {
    stop_strings
        .iter()
        .enumerate()
        .filter_map(|(idx, stop)| {
            let pos = if stop.is_empty() {
                Some(0)
            } else {
                galil_seiferas::gs_find(bytes, stop.as_bytes())
            }?;
            Some((idx, pos))
        })
        .min_by_key(|(idx, pos)| (*pos, *idx))
}

fn longest_stop_prefix_suffix(bytes: &[u8], stop_strings: &[String]) -> usize {
    stop_strings
        .iter()
        .map(|stop| {
            let stop = stop.as_bytes();
            let max_len = bytes.len().min(stop.len().saturating_sub(1));
            (1..=max_len)
                .rev()
                .find(|len| bytes.ends_with(&stop[..*len]))
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

fn has_incomplete_utf8_tail(bytes: &[u8]) -> bool {
    let mut consumed = 0;
    while consumed < bytes.len() {
        match std::str::from_utf8(&bytes[consumed..]) {
            Ok(_) => return false,
            Err(error) => {
                consumed += error.valid_up_to();
                let Some(invalid_len) = error.error_len() else {
                    return true;
                };
                consumed += invalid_len;
            }
        }
    }
    false
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

    fn clone_audios_range(&self, range: Range<usize>) -> Vec<AudioInput> {
        self.audios[range].to_vec()
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
        let hashes = input_images.iter().map(|image| {
            let mut hasher = DefaultHasher::new();
            image.width().hash(&mut hasher);
            image.height().hash(&mut hasher);
            image.color().hash(&mut hasher);
            image.as_bytes().hash(&mut hasher);
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

    fn clone_images_range(&self, range: Range<usize>) -> Vec<image::DynamicImage> {
        self.images[range].to_vec()
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

pub struct SequenceVideos {
    videos: Vec<VideoInput>,
    hashes: Vec<u64>,
}

impl SequenceVideos {
    fn new(input_videos: Vec<VideoInput>) -> Self {
        // Store per-frame hashes (not per-video) so they align 1:1 with
        // per-frame token ranges from `find_image_placeholder_ranges`.
        let hashes = input_videos.iter().flat_map(|v| v.frame_hashes()).collect();
        Self {
            videos: input_videos,
            hashes,
        }
    }

    fn clone_videos(&self) -> Vec<VideoInput> {
        self.videos.clone()
    }

    fn clone_frames_range(&self, range: Range<usize>) -> Vec<VideoInput> {
        let mut videos = Vec::new();
        let mut cursor = 0usize;
        for video in &self.videos {
            let next = cursor + video.frames.len();
            if range.start < next && range.end > cursor {
                let start = range.start.saturating_sub(cursor).min(video.frames.len());
                let end = range.end.saturating_sub(cursor).min(video.frames.len());
                if start < end {
                    videos.push(VideoInput {
                        frames: video.frames[start..end].to_vec(),
                        fps: video.fps,
                        total_num_frames: video.total_num_frames,
                        sampled_indices: video.sampled_indices[start..end].to_vec(),
                    });
                }
            }
            cursor = next;
            if cursor >= range.end {
                break;
            }
        }
        videos
    }

    fn videos(&self) -> &[VideoInput] {
        &self.videos
    }

    fn videos_mut(&mut self) -> &mut Vec<VideoInput> {
        &mut self.videos
    }

    fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    fn keep_num_videos(&mut self, videos_to_keep: usize) {
        if self.videos.len() > videos_to_keep {
            let start = self.videos.len() - videos_to_keep;
            self.videos = self.videos[start..].to_vec();
        }
    }

    fn keep_num_video_frames(&mut self, video_frames_to_keep: usize) {
        let frame_count = self.videos.iter().map(|video| video.frames.len()).sum();
        if frame_count > video_frames_to_keep {
            self.videos = self.clone_frames_range(frame_count - video_frames_to_keep..frame_count);
        }
    }
}

// Holds all multimodal (vision/diffusion) data for a Sequence.
pub struct MultimodalData {
    pub input_images: Option<SequenceImages>,
    pub input_audios: Option<SequenceAudios>,
    pub input_videos: Option<SequenceVideos>,
    pub cached_pixel_values: Option<Tensor>,
    pub cached_pixel_attention_mask: Option<Tensor>,
    pub cached_spatial_shapes: Option<Tensor>,
    pub cached_num_crops: Option<Vec<usize>>,
    pub cached_img_thw: Option<Tensor>,
    pub cached_vid_thw: Option<Tensor>,
    /// Complete image grid metadata, including prefix-cached images.
    pub rope_img_grid_thw: Option<Tensor>,
    /// Complete video grid metadata, including prefix-cached videos.
    pub rope_vid_grid_thw: Option<Tensor>,
    /// Fixed offset between token indices and post-media MRoPE positions.
    pub mrope_position_delta: Option<i64>,
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
        input_videos: Option<Vec<VideoInput>>,
        image_gen_response_format: Option<ImageGenerationResponseFormat>,
        diffusion_params: Option<DiffusionGenerationParams>,
        image_gen_save_file: Option<PathBuf>,
    ) -> Self {
        MultimodalData {
            input_images: input_images.map(SequenceImages::new),
            input_audios: input_audios.map(SequenceAudios::new),
            input_videos: input_videos.map(SequenceVideos::new),
            cached_pixel_values: None,
            cached_pixel_attention_mask: None,
            cached_spatial_shapes: None,
            cached_num_crops: None,
            cached_img_thw: None,
            cached_vid_thw: None,
            rope_img_grid_thw: None,
            rope_vid_grid_thw: None,
            mrope_position_delta: None,
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

    pub fn clone_images_range(&self, range: Range<usize>) -> Option<Vec<image::DynamicImage>> {
        self.input_images
            .as_ref()
            .map(|imgs| imgs.clone_images_range(range))
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

    pub fn clone_audios_range(&self, range: Range<usize>) -> Option<Vec<AudioInput>> {
        self.input_audios
            .as_ref()
            .map(|a| a.clone_audios_range(range))
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

    pub fn take_videos(&mut self) -> Option<Vec<VideoInput>> {
        if self.has_changed_prompt {
            if let Some(input_videos) = self.input_videos.as_mut() {
                let mut videos = Vec::new();
                std::mem::swap(&mut videos, input_videos.videos_mut());
                Some(videos)
            } else {
                None
            }
        } else {
            self.input_videos.as_ref().map(|v| v.clone_videos())
        }
    }

    pub fn clone_videos(&self) -> Option<Vec<VideoInput>> {
        self.input_videos.as_ref().map(|v| v.clone_videos())
    }

    pub fn clone_frames_range(&self, range: Range<usize>) -> Option<Vec<VideoInput>> {
        self.input_videos
            .as_ref()
            .map(|v| v.clone_frames_range(range))
    }

    pub fn videos(&self) -> Option<&[VideoInput]> {
        self.input_videos.as_ref().map(|v| v.videos())
    }

    pub fn video_hashes(&self) -> Option<&[u64]> {
        self.input_videos.as_ref().map(|v| v.hashes())
    }

    pub fn has_videos(&self) -> bool {
        self.input_videos
            .as_ref()
            .is_some_and(|v| !v.videos().is_empty())
    }

    pub fn keep_num_videos(&mut self, videos_to_keep: usize) {
        if let Some(vids) = self.input_videos.as_mut() {
            vids.keep_num_videos(videos_to_keep)
        }
    }

    pub fn keep_num_video_frames(&mut self, video_frames_to_keep: usize) {
        if let Some(vids) = self.input_videos.as_mut() {
            vids.keep_num_video_frames(video_frames_to_keep)
        }
    }

    pub fn keep_num_images(&mut self, images_to_keep: usize) {
        if let Some(imgs) = self.input_images.as_mut() {
            imgs.keep_num_images(images_to_keep);
        }
        // Invalidate preprocessed pixel value cache, the trimmed image set
        // no longer matches the cached tensor dimensions (used by Qwen VL models).
        self.cached_pixel_values = None;
        self.cached_pixel_attention_mask = None;
        self.cached_spatial_shapes = None;
        self.cached_num_crops = None;
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
/// Used by multimodal model input processors to find where each image's placeholder
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

pub fn find_placeholder_delimited_ranges(
    tokens: &[u32],
    placeholder_id: u32,
    start_id: u32,
    end_id: u32,
) -> Vec<(usize, usize)> {
    find_image_placeholder_ranges(tokens, placeholder_id)
        .into_iter()
        .map(|(offset, length)| {
            let placeholder_end = offset + length;
            let start = tokens[..=offset].iter().rposition(|&tok| tok == start_id);
            let end = tokens[placeholder_end..]
                .iter()
                .position(|&tok| tok == end_id)
                .map(|pos| placeholder_end + pos);
            match (start, end) {
                (Some(start), Some(end)) if start < offset && placeholder_end <= end => {
                    (start, end - start + 1)
                }
                _ => (offset, length),
            }
        })
        .collect()
}

pub fn clamp_prefix_cache_len_for_mm_features(
    prefix_len: usize,
    block_size: usize,
    features: &[MultiModalFeature],
) -> usize {
    if prefix_len == 0 || block_size == 0 {
        return prefix_len;
    }

    let mut prefix_len = prefix_len;
    loop {
        let next = features
            .iter()
            .filter(|feature| feature.offset < prefix_len && prefix_len < feature.end())
            .map(|feature| (feature.offset / block_size) * block_size)
            .min()
            .unwrap_or(prefix_len);
        if next == prefix_len {
            return prefix_len;
        }
        prefix_len = next;
    }
}

#[derive(Default)]
pub struct MultimodalPromptLayout {
    features: Vec<MultiModalFeature>,
}

impl MultimodalPromptLayout {
    pub fn extend_ranges(
        &mut self,
        ranges: &[(usize, usize)],
        hashes: &[u64],
        kind: MultimodalKind,
        attention_policy: MultimodalAttentionPolicy,
    ) {
        for (item_idx, (&(offset, length), hash)) in
            (self.next_item_index(kind)..).zip(ranges.iter().zip(hashes.iter()))
        {
            self.features.push(MultiModalFeature {
                kind,
                item_range: item_idx..item_idx + 1,
                hashes: vec![*hash],
                offset,
                length,
                attention_policy,
                splittable: false,
            });
        }
    }

    pub fn into_features(mut self) -> Vec<MultiModalFeature> {
        self.features.sort_by_key(|feature| feature.offset);
        self.features
    }

    fn next_item_index(&self, kind: MultimodalKind) -> usize {
        self.features
            .iter()
            .filter(|feature| feature.kind == kind)
            .map(|feature| feature.item_range.end)
            .max()
            .unwrap_or(0)
    }
}

pub fn build_mm_features_from_ranges(
    ranges: &[(usize, usize)],
    hashes: &[u64],
    kind: MultimodalKind,
) -> Vec<MultiModalFeature> {
    build_mm_features_from_ranges_with_policy(
        ranges,
        hashes,
        kind,
        MultimodalAttentionPolicy::Causal,
    )
}

pub fn build_mm_features_from_ranges_with_policy(
    ranges: &[(usize, usize)],
    hashes: &[u64],
    kind: MultimodalKind,
    attention_policy: MultimodalAttentionPolicy,
) -> Vec<MultiModalFeature> {
    let mut layout = MultimodalPromptLayout::default();
    layout.extend_ranges(ranges, hashes, kind, attention_policy);
    layout.into_features()
}

pub struct Sequence {
    // Metadata, const
    id: usize,
    prompt_len: usize,
    max_len: Option<usize>,
    timestamp: u128,
    sampler: Arc<Sampler>,
    sampling_rng: Option<Arc<std::sync::Mutex<Isaac64Rng>>>,
    stop_tokens: Vec<u32>,
    stop_strings: Vec<String>,
    ignore_eos: bool,
    return_logprobs: bool,
    stream_logprobs: bool,
    responder: Sender<Response>,
    response_index: usize,
    creation_time: u64,
    prompt: String,
    sequence_stepping_type: SeqStepType,
    pub(crate) return_raw_logits: bool,
    token_offset: usize,
    eos_tokens: Vec<u32>,
    adapter: Option<AdapterLease>,

    // Multimodal data (images, diffusion settings, pixel caches)
    pub multimodal: MultimodalData,

    // Completion requests
    suffix: Option<String>,
    prefix: Option<String>,

    // Speculative
    staged_speculative_tokens: SpeculativeTokens,
    staged_speculative_distribution: Option<SpeculativeProposalDistribution>,

    // Prefix caching
    prefill_prompt_toks: Option<PrefillTokens>,
    /// Number of tokens at the start of the prompt that are cached (KV already computed).
    /// These tokens should be skipped during prefill.
    prefix_cache_len: usize,
    prefix_cache_hit_recorded: bool,
    block_hash_revision: u64,
    /// Number of logical tokens represented in model/cache state.
    num_computed_tokens: usize,
    /// Denoising-loop time inside the latest block-generation step; booked as completion
    /// time even when the step was a prompt step (the encoder prefill is the prompt part).
    pending_denoise_time_ms: u128,

    // Cache
    normal_cache: Vec<Option<KvCache>>,
    normal_draft_cache: Vec<Option<KvCache>>,
    scaling_cache: Option<Tensor>,
    cache: LayerCaches,
    draft_cache: LayerCaches,
    xlora_cache: Option<LayerCaches>,
    /// For hybrid models: index into the recurrent state pool
    recurrent_state_idx: Option<usize>,

    // Preallocated KV cache templates, keyed by layer.
    seq_preallocated_cache: Option<SeqPreallocatedCache>,

    // Mutables
    tokens: Vec<u32>,
    logprobs: Vec<Logprobs>,
    cumulative_logprob: f32,
    last_logprob: f32,
    last_completion_bytes_len: usize,
    last_is_done: Option<StopReason>,
    completion_bytes: Vec<u8>,
    stop_pending_bytes: Vec<u8>,
    stop_pending_emissions: VecDeque<PendingStreamingEmission>,
    ready_streaming_emissions: VecDeque<PendingStreamingEmission>,
    stream_idx: usize,
    pub recognizer: SequenceRecognizer,
    scheduling_urgency: usize, // The number of passes since scheduling

    // GPU things
    pub prompt_tok_per_sec: f32,
    pub prompt_timestamp: Option<u128>,
    pub total_prompt_time: Option<u128>,
    pub total_completion_time: Option<u128>,
    pub step_start_instant: Option<Instant>,
    step_timing_kind: Option<StepTimingKind>,
    group: Arc<Mutex<SequenceGroup>>,
    state: RwLock<SequenceState>,

    // Tool calls
    pub(crate) tool_call_state: Option<ToolCallState>,

    // Tag-based reasoning parser.
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
    reasoning_mode: Option<ReasoningMode>,
}

#[derive(Clone, Copy)]
enum StepTimingKind {
    Prompt,
    Completion,
}

impl Sequence {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_waiting(
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
        input_videos: Option<Vec<VideoInput>>,
        // Paged attention
        block_size: Option<usize>,
        //
        tool_call_state: Option<ToolCallState>,
        image_gen_response_format: Option<ImageGenerationResponseFormat>,
        sequence_stepping_type: SeqStepType,
        diffusion_params: Option<DiffusionGenerationParams>,
        image_gen_save_file: Option<PathBuf>,
        // Preallocated KV cache templates, keyed by layer.
        seq_preallocated_cache: Option<SeqPreallocatedCache>,
        //
        return_raw_logits: bool,
        ignore_eos: bool,
        eos_tokens: Vec<u32>,
        sampling_seed: Option<u64>,
    ) -> Self {
        let prompt_len = tokens.len();
        let _ = block_size; // Block management handled by KVCacheManager
        let stream_logprobs = return_logprobs
            && group
                .try_lock()
                .expect("new sequence group must not be locked")
                .is_streaming;
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
            sampling_rng: sampling_seed
                .map(|seed| Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(seed)))),
            stop_tokens,
            stop_strings,
            ignore_eos,
            max_len,
            return_logprobs,
            stream_logprobs,
            prompt_tok_per_sec: 0.,
            prompt_timestamp: None,
            group,
            scaling_cache: None,
            response_index,
            creation_time,
            recognizer,
            prefill_prompt_toks: None,
            prefix_cache_len: 0,
            prefix_cache_hit_recorded: false,
            block_hash_revision: 0,
            num_computed_tokens: 0,
            pending_denoise_time_ms: 0,
            suffix,
            prefix,
            cumulative_logprob: 0.,
            completion_bytes: Vec::new(),
            stop_pending_bytes: Vec::new(),
            stop_pending_emissions: VecDeque::new(),
            ready_streaming_emissions: VecDeque::new(),
            stream_idx: 0,
            last_completion_bytes_len: 0,
            last_logprob: 0.0,
            last_is_done: None,
            staged_speculative_tokens: SpeculativeTokens::default(),
            staged_speculative_distribution: None,
            scheduling_urgency: 0,
            // Multimodal data
            multimodal: MultimodalData::new(
                input_images,
                input_audios,
                input_videos,
                image_gen_response_format,
                diffusion_params,
                image_gen_save_file,
            ),
            tool_call_state,
            sequence_stepping_type,
            return_raw_logits,
            token_offset: 0,
            eos_tokens,
            adapter: None,
            total_prompt_time: None,
            total_completion_time: None,
            step_start_instant: None,
            step_timing_kind: None,
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
        self.prefill_prompt_toks = Some(PrefillTokens {
            tokens: toks,
            view: PrefillTokenView::SuffixOnly,
        });
        self.set_state(SequenceState::RunningPrefillPrompt);
        self.token_offset = offset;
        self.prefix_cache_len = offset;
        self
    }

    /// This is the number of tokens. If the KV cache is Some, then it will use that.
    pub fn len(&self) -> usize {
        if let Some(prefill) = &self.prefill_prompt_toks {
            return prefill.tokens.len();
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

    pub fn generated_len(&self) -> usize {
        self.tokens.len().saturating_sub(self.prompt_len)
    }

    pub fn max_generation_len(&self, max_model_len: usize) -> usize {
        self.max_len.unwrap_or(max_model_len)
    }

    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop_tokens
    }

    pub fn id(&self) -> &usize {
        &self.id
    }

    pub(crate) fn bind_adapter(&mut self, adapter: AdapterLease) {
        assert!(self.adapter.is_none(), "sequence adapter is already bound");
        self.adapter = Some(adapter);
        self.block_hash_revision = self.block_hash_revision.wrapping_add(1);
    }

    pub(crate) fn adapter_lease(&self) -> Option<&AdapterLease> {
        self.adapter.as_ref()
    }

    pub fn adapter_generation(&self) -> Option<AdapterGenerationId> {
        self.adapter.as_ref().map(AdapterLease::generation)
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
            SequenceState::Error
                | SequenceState::FinishedAborted
                | SequenceState::FinishedIgnored
                | SequenceState::Done(_)
        )
    }

    pub fn get_toks(&self) -> &[u32] {
        if let Some(prefill) = &self.prefill_prompt_toks {
            return &prefill.tokens;
        }
        &self.tokens
    }

    pub fn is_chunked_prefill_view(&self) -> bool {
        self.prefill_prompt_toks.is_some() && !self.mm_features().is_empty()
    }

    fn prefill_token_view(&self) -> Option<PrefillTokenView> {
        self.prefill_prompt_toks
            .as_ref()
            .map(|prefill| prefill.view)
    }

    pub(crate) fn has_suffix_only_prefill_toks(&self) -> bool {
        self.prefill_token_view() == Some(PrefillTokenView::SuffixOnly)
    }

    pub(crate) fn active_prompt_query_range(&self) -> Option<Range<usize>> {
        let view = self.prefill_token_view()?;
        let prefix_len = self.prefix_cache_len();
        let token_len = self.get_toks().len();
        Some(match view {
            PrefillTokenView::PrefixInclusive => prefix_len.min(token_len)..token_len,
            PrefillTokenView::SuffixOnly => prefix_len..prefix_len.saturating_add(token_len),
        })
    }

    pub(crate) fn active_prompt_local_query_range(&self) -> Option<Range<usize>> {
        let view = self.prefill_token_view()?;
        let prefix_len = self.prefix_cache_len();
        let token_len = self.get_toks().len();
        Some(match view {
            PrefillTokenView::PrefixInclusive => prefix_len.min(token_len)..token_len,
            PrefillTokenView::SuffixOnly => 0..token_len,
        })
    }

    pub(crate) fn prompt_position_source_toks(&self) -> &[u32] {
        match self.prefill_token_view() {
            Some(PrefillTokenView::SuffixOnly) => &self.tokens,
            _ => self.get_toks(),
        }
    }

    pub(crate) fn active_multimodal_item_range(
        &self,
        kind: MultimodalKind,
    ) -> Option<Range<usize>> {
        self.active_multimodal_window(kind)
            .map(|window| window.item_range)
    }

    pub(crate) fn active_local_multimodal_item_range(
        &self,
        kind: MultimodalKind,
        available_items: usize,
    ) -> Option<Range<usize>> {
        let range = self.active_multimodal_item_range(kind)?;
        if self.prefill_token_view()? == PrefillTokenView::PrefixInclusive {
            return (range.end <= available_items).then_some(range);
        }
        let total_items = self
            .mm_features()
            .iter()
            .filter(|feature| feature.kind == kind)
            .map(|feature| feature.item_range.end)
            .max()
            .unwrap_or(0);
        let retained_start = total_items.saturating_sub(available_items);
        (retained_start <= range.start && range.end <= total_items)
            .then(|| range.start - retained_start..range.end - retained_start)
    }

    fn active_multimodal_window(&self, kind: MultimodalKind) -> Option<ActiveMultimodalWindow> {
        if self.mm_features().is_empty() {
            return None;
        }

        let query = self.active_prompt_query_range()?;
        let mut first = None;
        let mut last = None;
        for feature in self
            .mm_features()
            .iter()
            .filter(|feature| feature.kind == kind)
        {
            if feature.overlaps(query.start, query.end) {
                first = Some(first.map_or(feature.item_range.start, |idx: usize| {
                    idx.min(feature.item_range.start)
                }));
                last = Some(last.map_or(feature.item_range.end, |idx: usize| {
                    idx.max(feature.item_range.end)
                }));
            }
        }
        first.zip(last).and_then(|(start, end)| {
            (start < end).then_some(ActiveMultimodalWindow {
                item_range: start..end,
            })
        })
    }

    pub(crate) fn active_staged_speculative_tokens(&self) -> &SpeculativeTokens {
        &self.staged_speculative_tokens
    }

    pub(crate) fn active_staged_speculative_len(&self) -> usize {
        self.staged_speculative_tokens.len()
    }

    pub(crate) fn set_staged_speculative(
        &mut self,
        tokens: impl Into<SpeculativeTokens>,
        distribution: Option<SpeculativeProposalDistribution>,
    ) {
        self.staged_speculative_tokens = tokens.into();
        self.staged_speculative_distribution = distribution;
    }

    pub(crate) fn take_staged_speculative_tokens(&mut self) -> SpeculativeTokens {
        std::mem::take(&mut self.staged_speculative_tokens)
    }

    pub(crate) fn take_staged_speculative_distribution(
        &mut self,
    ) -> Option<SpeculativeProposalDistribution> {
        self.staged_speculative_distribution.take()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn staged_speculative_distribution(
        &self,
    ) -> Option<&SpeculativeProposalDistribution> {
        self.staged_speculative_distribution.as_ref()
    }

    pub(crate) fn clear_staged_speculative_tokens(&mut self) {
        self.staged_speculative_tokens = SpeculativeTokens::default();
        self.staged_speculative_distribution = None;
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

    pub(crate) fn record_prefix_cache_hit(&mut self) -> bool {
        if !matches!(self.sequence_stepping_type, SeqStepType::PromptAndDecode)
            || self.prefix_cache_hit_recorded
        {
            return false;
        }
        self.prefix_cache_hit_recorded = true;
        true
    }

    pub fn clip_prefix_cache_len_for_non_causal_mm_features(&mut self, block_size: usize) {
        if block_size == 0 || self.prefix_cache_len == 0 {
            return;
        }
        let mut prefix_len = self.prefix_cache_len;
        for feature in self.mm_features() {
            if feature.attention_policy == MultimodalAttentionPolicy::NonCausal
                && feature.offset < prefix_len
                && prefix_len < feature.end()
            {
                prefix_len = prefix_len.min((feature.offset / block_size) * block_size);
            }
        }
        self.prefix_cache_len = prefix_len;
    }

    pub fn clip_prefix_cache_len_for_mm_features(&mut self, block_size: usize) {
        if block_size == 0 || self.prefix_cache_len == 0 {
            return;
        }
        let mut prefix_len = self.prefix_cache_len;
        loop {
            let next = self
                .mm_features()
                .iter()
                .filter(|feature| {
                    (!feature.splittable
                        || feature.attention_policy == MultimodalAttentionPolicy::NonCausal)
                        && feature.offset < prefix_len
                        && prefix_len < feature.end()
                })
                .map(|feature| (feature.offset / block_size) * block_size)
                .min()
                .unwrap_or(prefix_len);
            if next == prefix_len {
                break;
            }
            prefix_len = next;
        }
        self.prefix_cache_len = prefix_len;
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
        self.clear_staged_speculative_tokens();
        self.num_computed_tokens = 0;
        self.bump_block_hash_revision();

        if let Some(metadata) = paged_attn_metadata {
            self.prefix_cache_len = 0;
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

    pub fn preallocated_cache(&self) -> Option<&SeqPreallocatedCache> {
        self.seq_preallocated_cache.as_ref()
    }

    pub fn normal_cache(&mut self) -> &mut Vec<Option<KvCache>> {
        &mut self.normal_cache
    }

    pub fn normal_cache_ref(&self) -> &[Option<KvCache>] {
        &self.normal_cache
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

    pub fn block_hash_revision(&self) -> u64 {
        self.block_hash_revision
    }

    pub fn num_computed_tokens(&self) -> usize {
        self.num_computed_tokens.min(self.len())
    }

    pub fn set_num_computed_tokens(&mut self, len: usize) {
        self.num_computed_tokens = len.min(self.len());
    }

    pub fn advance_num_computed_tokens(&mut self, amount: usize) {
        self.set_num_computed_tokens(self.num_computed_tokens.saturating_add(amount));
    }

    pub fn num_uncomputed_tokens(&self) -> usize {
        self.len().saturating_sub(self.num_computed_tokens())
    }

    pub(crate) fn add_pending_denoise_time(&mut self, time: std::time::Duration) {
        self.pending_denoise_time_ms = self
            .pending_denoise_time_ms
            .saturating_add(time.as_millis());
    }

    fn bump_block_hash_revision(&mut self) {
        self.block_hash_revision = self.block_hash_revision.wrapping_add(1);
    }

    pub fn is_xlora(&self) -> bool {
        self.xlora_cache.is_some()
    }

    pub fn sampler(&self) -> Arc<Sampler> {
        self.sampler.clone()
    }

    pub(crate) fn sampling_rng(
        &self,
        fallback: &Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Arc<std::sync::Mutex<Isaac64Rng>> {
        self.sampling_rng.as_ref().unwrap_or(fallback).clone()
    }

    /// Add a some prefill tokens. Only meant for internal speculative decoding usage.
    pub fn set_prefill_toks(&mut self, toks: Vec<u32>) {
        if let Some(prefill) = self.prefill_prompt_toks.as_mut() {
            prefill.tokens = toks;
        } else {
            self.prefill_prompt_toks = Some(PrefillTokens {
                tokens: toks,
                view: PrefillTokenView::PrefixInclusive,
            });
        }
    }

    pub fn has_prefill_toks(&self) -> bool {
        self.prefill_prompt_toks.is_some()
    }

    /// Remove the prefill tokens.
    pub fn reset_prefill_toks(&mut self) {
        self.prefill_prompt_toks = None
    }

    fn push_stop_pending_emission(&mut self, bytes: Vec<u8>, logprobs: &Logprobs) {
        if bytes.is_empty() {
            return;
        }
        self.stop_pending_bytes.extend_from_slice(&bytes);
        if self.stream_logprobs {
            self.stop_pending_emissions
                .push_back(PendingStreamingEmission {
                    bytes,
                    logprobs: logprobs.clone(),
                });
        }
    }

    fn commit_stop_pending_prefix(&mut self, len: usize) {
        debug_assert!(len <= self.stop_pending_bytes.len());
        if len > 0 {
            if let Some(parser) = self.reasoning_parser.as_mut() {
                parser.process_bytes(&self.stop_pending_bytes[..len]);
            }
        }
        self.completion_bytes
            .extend_from_slice(&self.stop_pending_bytes[..len]);
        self.stop_pending_bytes.drain(..len);

        if !self.stream_logprobs {
            return;
        }

        let mut remaining = len;
        while remaining > 0 {
            let mut emission = self
                .stop_pending_emissions
                .pop_front()
                .expect("pending stop bytes must have matching logprobs");
            if emission.bytes.len() <= remaining {
                remaining -= emission.bytes.len();
                self.ready_streaming_emissions.push_back(emission);
            } else {
                let suffix = emission.bytes.split_off(remaining);
                let suffix_logprobs = emission.logprobs.clone();
                self.ready_streaming_emissions.push_back(emission);
                self.stop_pending_emissions
                    .push_front(PendingStreamingEmission {
                        bytes: suffix,
                        logprobs: suffix_logprobs,
                    });
                remaining = 0;
            }
        }
    }

    fn discard_stop_pending_bytes(&mut self) {
        self.stop_pending_bytes.clear();
        self.stop_pending_emissions.clear();
    }

    fn streaming_safe_commit_len(&self, byte_limit: usize) -> usize {
        if !self.stream_logprobs {
            return byte_limit;
        }
        self.stop_pending_emissions
            .iter()
            .map(|emission| emission.bytes.len())
            .scan(0, |total, len| {
                *total += len;
                Some(*total)
            })
            .take_while(|total| *total <= byte_limit)
            .last()
            .unwrap_or(0)
    }

    pub(crate) fn flush_stop_pending_bytes(&mut self) {
        self.commit_stop_pending_prefix(self.stop_pending_bytes.len());
    }

    pub(crate) fn take_ready_streaming_emissions(
        &mut self,
        finalize: bool,
    ) -> Vec<StreamingEmission> {
        let mut emissions = Vec::with_capacity(self.ready_streaming_emissions.len());
        while !self.ready_streaming_emissions.is_empty() {
            let mut bytes = Vec::new();
            let mut group_len = 0;
            let mut is_complete = false;
            for emission in &self.ready_streaming_emissions {
                bytes.extend_from_slice(&emission.bytes);
                group_len += 1;
                if !has_incomplete_utf8_tail(&bytes) {
                    is_complete = true;
                    break;
                }
            }
            if !is_complete && !finalize {
                break;
            }

            let is_first = self.stream_idx == 0;
            self.stream_idx += bytes.len();
            let text = String::from_utf8_lossy(&bytes);
            let text = if is_first {
                text.trim_start().to_string()
            } else {
                text.to_string()
            };
            for idx in 0..group_len {
                let emission = self
                    .ready_streaming_emissions
                    .pop_front()
                    .expect("streaming UTF-8 group length must match the queue");
                emissions.push(StreamingEmission {
                    text: if idx + 1 == group_len {
                        text.clone()
                    } else {
                        String::new()
                    },
                    bytes: emission.bytes,
                    logprobs: emission.logprobs,
                });
            }
        }
        emissions
    }

    pub fn add_token(
        &mut self,
        tok: Logprobs,
        completion_bytes: Vec<u8>,
        mut is_done: Option<StopReason>,
    ) -> Option<StopReason> {
        let stopped_by_token = matches!(
            is_done,
            Some(StopReason::Eos) | Some(StopReason::StopTok(_))
        );
        let stop_strings_are_active = !self
            .tool_call_state
            .as_ref()
            .is_some_and(|state| state.required_tool_call_unsatisfied());
        let stop_string_may_finish = matches!(
            is_done,
            None | Some(StopReason::Length(_)) | Some(StopReason::ModelLength(_))
        );
        let committed_start = self.completion_bytes.len();

        if let Some(ref mut tool_call_state) = self.tool_call_state {
            tool_call_state.observe_token(tok.token, &completion_bytes);
        }
        if stopped_by_token {
            self.flush_stop_pending_bytes();
        } else if stop_strings_are_active && stop_string_may_finish && !self.stop_strings.is_empty()
        {
            self.push_stop_pending_emission(completion_bytes, &tok);
            if let Some((stop_string_idx, pos)) =
                find_earliest_stop_string(&self.stop_pending_bytes, &self.stop_strings)
            {
                self.commit_stop_pending_prefix(pos);
                self.discard_stop_pending_bytes();
                is_done = Some(StopReason::StopString {
                    stop_string_idx,
                    completion_bytes_pos: committed_start + pos,
                });
            } else if is_done.is_some() {
                self.flush_stop_pending_bytes();
            } else {
                let pending_len =
                    longest_stop_prefix_suffix(&self.stop_pending_bytes, &self.stop_strings);
                let committed_len =
                    self.streaming_safe_commit_len(self.stop_pending_bytes.len() - pending_len);
                self.commit_stop_pending_prefix(committed_len);
            }
        } else {
            self.flush_stop_pending_bytes();
            self.push_stop_pending_emission(completion_bytes, &tok);
            self.flush_stop_pending_bytes();
        }

        let committed_bytes = &self.completion_bytes[committed_start..];
        self.last_completion_bytes_len = committed_bytes.len();
        self.last_logprob = tok.logprob;
        self.last_is_done = is_done;

        self.cumulative_logprob += tok.logprob;
        self.tokens.push(tok.token);
        self.logprobs.push(tok);
        self.reset_prefill_toks();
        is_done
    }

    pub fn responder(&self) -> Sender<Response> {
        self.responder.clone()
    }

    pub(crate) fn response_is_closed(&self) -> bool {
        self.responder.is_closed()
    }

    pub fn creation_time(&self) -> u64 {
        self.creation_time
    }

    pub fn set_state(&self, state: SequenceState) {
        if matches!(state, SequenceState::Error) {
            let mut group = get_mut_group!(self);
            group.n_choices = group.n_choices.saturating_sub(1);
            // Count the transition into Error once, not every re-set.
            if !matches!(self.getstate(), SequenceState::Error) {
                metrics::counter!("mistralrs_sequences_completed_total", "reason" => "error")
                    .increment(1);
            }
        }
        if let SequenceState::Done(reason) = &state {
            // Count the transition into Done once, not every re-set.
            if !matches!(self.getstate(), SequenceState::Done(_)) {
                metrics::counter!("mistralrs_sequences_completed_total", "reason" => reason.metric_label())
                    .increment(1);
            }
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
        let required_tool_call_unsatisfied = self
            .tool_call_state
            .as_ref()
            .is_some_and(|state| state.required_tool_call_unsatisfied());
        let is_eos = !self.ignore_eos
            && match eos_tok {
                Some(eos_tok) => eos_tok.contains(&tok),
                None => false,
            };
        if is_eos && !required_tool_call_unsatisfied {
            Some(StopReason::Eos)
        } else if matches!(
            &*self.state.read().unwrap(),
            SequenceState::Done(StopReason::Canceled)
        ) {
            Some(StopReason::Canceled)
        } else if self.stop_tokens.contains(&tok) && !required_tool_call_unsatisfied {
            Some(StopReason::StopTok(tok))
        } else if self.max_len.is_some()
            && self.tokens.len().saturating_sub(self.prompt_len) + 1 >= self.max_len.unwrap()
        {
            // add_token will be called after this check
            Some(StopReason::Length(self.max_len.unwrap()))
        } else if self.tokens.len() >= max_model_len {
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

    #[cfg(feature = "cuda")]
    pub(crate) fn sampling_logprob_required(&self) -> bool {
        get_mut_group!(self).sampling_logprob_required()
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
        let (new_decoded, consumed) = self.decode_streamable_delta();
        if new_decoded.is_some() {
            self.stream_idx += consumed;
        }
        Ok(new_decoded)
    }

    pub fn get_final_delta(&mut self) -> String {
        let is_first = self.stream_idx == 0;
        let decoded = String::from_utf8_lossy(&self.completion_bytes[self.stream_idx..]);
        self.stream_idx = self.completion_bytes.len();
        if is_first {
            decoded.trim_start().to_string()
        } else {
            decoded.to_string()
        }
    }

    /// Peeks at the delta between the last two decoded sequences, but does not advance the stream index.
    pub fn peek_delta(&self) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.decode_streamable_delta().0)
    }

    fn decode_streamable_delta(&self) -> (Option<String>, usize) {
        let is_first = self.stream_idx == 0;
        let pending = &self.completion_bytes[self.stream_idx..];
        let mut consumed = 0;
        while consumed < pending.len() {
            match std::str::from_utf8(&pending[consumed..]) {
                Ok(_) => {
                    consumed = pending.len();
                    break;
                }
                Err(error) => {
                    consumed += error.valid_up_to();
                    let Some(invalid_len) = error.error_len() else {
                        break;
                    };
                    consumed += invalid_len;
                }
            }
        }
        if consumed == 0 {
            return (None, 0);
        }
        let new_decoded = String::from_utf8_lossy(&pending[..consumed]);

        // The first token usually starts with a space. We don't want to add that to the delta.
        // Since we're using the completion_bytes, we need to take care of that ourselves.
        // Had we used HF's Tokenizer, it would have taken care of that for us.
        if is_first {
            return (Some(new_decoded.trim_start().to_string()), consumed);
        }
        (Some(new_decoded.to_string()), consumed)
    }

    pub fn timestamp(&self) -> u128 {
        self.timestamp
    }

    pub fn prompt_timestamp(&self) -> Option<u128> {
        self.prompt_timestamp
    }

    pub fn set_step_start_instant(&mut self) {
        self.start_prompt_timing();
    }

    pub(crate) fn start_prompt_timing(&mut self) {
        self.step_start_instant = Some(Instant::now());
        self.step_timing_kind = Some(StepTimingKind::Prompt);
    }

    pub(crate) fn start_completion_timing(&mut self) {
        self.step_start_instant = Some(Instant::now());
        self.step_timing_kind = Some(StepTimingKind::Completion);
    }

    pub(crate) fn finish_prompt_timing(&mut self, duration: Duration) {
        // Block diffusion denoises the first canvas inside the prompt step; book that share
        // as completion time so prompt T/s reflects the encoder prefill alone.
        let denoise_ms = std::mem::take(&mut self.pending_denoise_time_ms);
        let prompt_ms = duration.as_millis().saturating_sub(denoise_ms);
        let total = self
            .total_prompt_time
            .unwrap_or(0)
            .saturating_add(prompt_ms);
        self.total_prompt_time = Some(total);
        if denoise_ms > 0 {
            self.total_completion_time = Some(
                self.total_completion_time
                    .unwrap_or(0)
                    .saturating_add(denoise_ms),
            );
        }
        self.step_start_instant = None;
        self.step_timing_kind = None;
        if total > 0 {
            #[allow(clippy::cast_precision_loss)]
            {
                self.prompt_tok_per_sec = self.prompt_len as f32 / (total as f32 / 1000.0);
            }
        }
        self.update_time_info();
    }

    pub(crate) fn finish_completion_timing(&mut self, duration: Duration) {
        self.pending_denoise_time_ms = 0;
        let total = self
            .total_completion_time
            .unwrap_or(0)
            .saturating_add(duration.as_millis());
        self.total_completion_time = Some(total);
        self.step_start_instant = None;
        self.step_timing_kind = None;
        self.update_time_info();
    }

    pub(crate) fn update_time_info(&self) {
        let mut prompt_time_ms = self.total_prompt_time.unwrap_or(0);
        let mut completion_time_ms = self.total_completion_time.unwrap_or(0);
        if let (Some(start), Some(kind)) = (self.step_start_instant, self.step_timing_kind) {
            match kind {
                StepTimingKind::Prompt => {
                    let denoise_ms = self.pending_denoise_time_ms;
                    let elapsed_ms = start.elapsed().as_millis();
                    prompt_time_ms += elapsed_ms.saturating_sub(denoise_ms);
                    completion_time_ms += denoise_ms;
                }
                StepTimingKind::Completion => completion_time_ms += start.elapsed().as_millis(),
            }
        }

        let mut group = get_mut_group!(self);
        group.total_prompt_time = prompt_time_ms;
        group.total_completion_time = completion_time_ms;
        group.total_time = prompt_time_ms.saturating_add(completion_time_ms);
        group.total_prompt_toks = self.prompt_len;
        group.total_cached_toks = self.prefix_cache_len();
        group.total_toks = self.len();
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
        if self.is_chunked_prefill_view() {
            let available_items = self.multimodal.images().map_or(0, <[_]>::len);
            return self
                .active_local_multimodal_item_range(MultimodalKind::Image, available_items)
                .and_then(|range| self.multimodal.clone_images_range(range));
        }
        self.multimodal.take_images()
    }

    pub fn clone_images(&self) -> Option<Vec<image::DynamicImage>> {
        self.multimodal.clone_images()
    }

    pub fn images(&self) -> Option<&[image::DynamicImage]> {
        self.multimodal.images()
    }

    pub fn image_hashes(&self) -> Option<&[u64]> {
        self.multimodal.image_hashes().map(|hashes| {
            if let Some(window) = self.active_multimodal_window(MultimodalKind::Image) {
                &hashes[window.item_range]
            } else if self.is_chunked_prefill_view() {
                &hashes[0..0]
            } else {
                hashes
            }
        })
    }

    pub fn has_images(&self) -> bool {
        if self.is_chunked_prefill_view() {
            let available_items = self.multimodal.images().map_or(0, <[_]>::len);
            return self
                .active_local_multimodal_item_range(MultimodalKind::Image, available_items)
                .is_some();
        }
        self.multimodal.has_images()
    }

    pub fn take_audios(&mut self) -> Option<Vec<AudioInput>> {
        if self.is_chunked_prefill_view() {
            let available_items = self.multimodal.audios().map_or(0, <[_]>::len);
            return self
                .active_local_multimodal_item_range(MultimodalKind::Audio, available_items)
                .and_then(|range| self.multimodal.clone_audios_range(range));
        }
        self.multimodal.take_audios()
    }

    pub fn clone_audios(&self) -> Option<Vec<AudioInput>> {
        self.multimodal.clone_audios()
    }

    pub fn audios(&self) -> Option<&[AudioInput]> {
        self.multimodal.audios()
    }

    pub fn audio_hashes(&self) -> Option<&[u64]> {
        self.multimodal.audio_hashes().map(|hashes| {
            if let Some(window) = self.active_multimodal_window(MultimodalKind::Audio) {
                &hashes[window.item_range]
            } else if self.is_chunked_prefill_view() {
                &hashes[0..0]
            } else {
                hashes
            }
        })
    }

    pub fn has_audios(&self) -> bool {
        if self.is_chunked_prefill_view() {
            let available_items = self.multimodal.audios().map_or(0, <[_]>::len);
            return self
                .active_local_multimodal_item_range(MultimodalKind::Audio, available_items)
                .is_some();
        }
        self.multimodal.has_audios()
    }

    /// Keep these last n audios
    pub fn keep_num_audios(&mut self, audios_to_keep: usize) {
        self.multimodal.keep_num_audios(audios_to_keep)
    }

    pub fn take_videos(&mut self) -> Option<Vec<VideoInput>> {
        if self.is_chunked_prefill_view() {
            let available_items = self.multimodal.videos().map_or(0, |videos| {
                videos.iter().map(|video| video.frames.len()).sum()
            });
            return self
                .active_local_multimodal_item_range(MultimodalKind::Video, available_items)
                .and_then(|range| self.multimodal.clone_frames_range(range));
        }
        self.multimodal.take_videos()
    }

    pub fn clone_videos(&self) -> Option<Vec<VideoInput>> {
        self.multimodal.clone_videos()
    }

    pub fn videos(&self) -> Option<&[VideoInput]> {
        self.multimodal.videos()
    }

    pub fn video_hashes(&self) -> Option<&[u64]> {
        self.multimodal.video_hashes().map(|hashes| {
            if let Some(window) = self.active_multimodal_window(MultimodalKind::Video) {
                &hashes[window.item_range]
            } else if self.is_chunked_prefill_view() {
                &hashes[0..0]
            } else {
                hashes
            }
        })
    }

    pub fn has_videos(&self) -> bool {
        if self.is_chunked_prefill_view() {
            let available_items = self.multimodal.videos().map_or(0, |videos| {
                videos.iter().map(|video| video.frames.len()).sum()
            });
            return self
                .active_local_multimodal_item_range(MultimodalKind::Video, available_items)
                .is_some();
        }
        self.multimodal.has_videos()
    }

    pub fn keep_num_videos(&mut self, videos_to_keep: usize) {
        self.multimodal.keep_num_videos(videos_to_keep)
    }

    pub fn keep_num_video_frames(&mut self, video_frames_to_keep: usize) {
        self.multimodal.keep_num_video_frames(video_frames_to_keep)
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
        self.bump_block_hash_revision();
    }

    /// Count the number of multimodal items whose placeholder tokens fall entirely
    /// within the prefix cache. Used by multimodal model inputs_processors to trim
    /// pixel_values so they match only the non-cached image placeholder positions.
    pub fn count_prefix_cached_mm_items(&self) -> usize {
        if self.is_chunked_prefill_view() {
            return 0;
        }
        let prefix_len = self.prefix_cache_len();
        if prefix_len == 0 {
            return 0;
        }
        self.mm_features()
            .iter()
            .filter(|f| f.end() <= prefix_len)
            .count()
    }

    pub fn count_prefix_cached_mm_items_by_kind(&self, kind: MultimodalKind) -> usize {
        if self.is_chunked_prefill_view() {
            return 0;
        }
        let prefix_len = self.prefix_cache_len();
        if prefix_len == 0 {
            return 0;
        }
        self.mm_features()
            .iter()
            .filter(|f| f.end() <= prefix_len && f.kind == kind)
            .map(|f| f.item_range.len())
            .sum()
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

    pub(crate) fn effective_eos_tokens<'a>(
        &self,
        eos_tokens: &'a [u32],
        disable_eos_stop: bool,
    ) -> Option<&'a [u32]> {
        (!disable_eos_stop && !self.ignore_eos).then_some(eos_tokens)
    }

    /// Get the active reasoning mode, if any.
    pub fn reasoning_mode(&self) -> Option<ReasoningMode> {
        self.reasoning_mode
    }

    /// Whether any reasoning parser needs special tokens in decoded text.
    pub fn needs_special_tokens(&self) -> bool {
        self.reasoning_parser.is_some()
            || self
                .tool_call_state
                .as_ref()
                .is_some_and(|state| state.requires_special_tokens())
    }

    /// Enable reasoning with the given parser and mode.
    pub fn enable_reasoning(&mut self, mode: ReasoningMode, parser: Box<dyn ReasoningParser>) {
        self.reasoning_parser = Some(parser);
        self.reasoning_mode = Some(mode);
    }

    pub fn has_reasoning_state(&self) -> bool {
        self.reasoning_parser.is_some()
            || self
                .tool_call_state
                .as_ref()
                .is_some_and(|state| state.has_reasoning())
    }

    /// Get the reasoning content delta since last call (for streaming).
    pub fn get_reasoning_content_delta(&mut self) -> Option<String> {
        if let Some(parser) = self.reasoning_parser.as_mut() {
            parser.get_reasoning_delta()
        } else {
            self.tool_call_state.as_mut()?.reasoning_delta()
        }
    }

    /// Get the response content delta since last call (for streaming).
    pub fn get_response_content_delta(&mut self) -> Option<String> {
        if let Some(parser) = self.reasoning_parser.as_mut() {
            parser.get_content_delta()
        } else {
            self.tool_call_state.as_mut()?.content_delta()
        }
    }

    /// Get accumulated reasoning content (for non-streaming).
    pub fn get_reasoning_content(&self) -> Option<String> {
        if let Some(parser) = self.reasoning_parser.as_ref() {
            parser.reasoning_content()
        } else {
            self.tool_call_state.as_ref()?.reasoning_content()
        }
    }

    /// Get accumulated response content (for non-streaming).
    pub fn get_response_content(&self) -> Option<String> {
        if let Some(parser) = self.reasoning_parser.as_ref() {
            parser.content()
        } else {
            self.tool_call_state.as_ref()?.content()
        }
    }

    /// Finalize parsers at end of stream.
    pub fn finalize_reasoning(&mut self) {
        if let Some(ref mut p) = self.reasoning_parser {
            p.finalize();
        }
        if let Some(ref mut tool_call_state) = self.tool_call_state {
            tool_call_state.finalize();
        }
    }
}

pub struct SequenceGroup {
    n_choices: usize, // The target number of choices to return. Can be decreased if an error is thrown.
    best_of: Option<usize>, // Top n seqs based on cumulative logprobs.
    pub total_prompt_toks: usize,
    pub total_cached_toks: usize,
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
    streaming_active_choices: usize,
    streaming_finished_choices: HashSet<usize>,
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
            total_cached_toks: 0,
            total_toks: 0,
            total_prompt_time: 0,
            total_time: 0,
            total_completion_time: 0,
            chat_streaming_chunks: Vec::new(),
            completion_streaming_chunks: Vec::new(),
            streaming_active_choices: n_choices,
            streaming_finished_choices: HashSet::new(),
            is_streaming,
            is_chat,
            best_of,
        }
    }

    pub fn get_choices(&self) -> &[Choice] {
        &self.choices
    }

    #[cfg(feature = "cuda")]
    fn sampling_logprob_required(&self) -> bool {
        self.n_choices > 1 || self.best_of.is_some_and(|best_of| best_of > 1)
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
            prompt_tokens_details: if self.total_cached_toks > 0 {
                Some(PromptTokensDetails {
                    cached_tokens: self.total_cached_toks,
                })
            } else {
                None
            },
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

    #[allow(clippy::result_large_err)]
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

    #[allow(clippy::result_large_err)]
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

    #[allow(clippy::result_large_err)]
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

    #[allow(clippy::result_large_err)]
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

    #[allow(clippy::result_large_err)]
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
        if !self.is_streaming {
            return Ok(());
        }
        if usage_opt.is_some()
            && self
                .streaming_finished_choices
                .insert(seq.get_response_index())
        {
            self.streaming_active_choices = self.streaming_active_choices.saturating_sub(1);
        }
        let expected_choices = self.streaming_active_choices.min(self.n_choices).max(1);
        if !self.chat_streaming_chunks.is_empty() {
            let has_duplicate_index =
                self.chat_streaming_chunks
                    .iter()
                    .enumerate()
                    .any(|(idx, chunk)| {
                        self.chat_streaming_chunks[..idx]
                            .iter()
                            .any(|prior| prior.index == chunk.index)
                    });
            if self.chat_streaming_chunks.len() < expected_choices && !has_duplicate_index {
                return Ok(());
            }
            let chunks = std::mem::take(&mut self.chat_streaming_chunks);
            let bundle = chunks.len() == expected_choices && !has_duplicate_index;
            let responses = if bundle {
                vec![chunks]
            } else {
                chunks.into_iter().map(|chunk| vec![chunk]).collect()
            };
            let response_count = responses.len();
            for (idx, choices) in responses.into_iter().enumerate() {
                seq.responder()
                    .send(Response::Chunk(ChatCompletionChunkResponse {
                        id: seq.id.to_string(),
                        choices,
                        created: seq.creation_time() as u128,
                        model: model.clone(),
                        system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                        object: "chat.completion.chunk".to_string(),
                        usage: (idx + 1 == response_count)
                            .then(|| usage_opt.clone())
                            .flatten(),
                        adapter_generation: seq
                            .adapter_generation()
                            .map(|generation| generation.to_string()),
                        session_id: None,
                    }))
                    .await?;
            }
        } else if !self.completion_streaming_chunks.is_empty() {
            let has_duplicate_index =
                self.completion_streaming_chunks
                    .iter()
                    .enumerate()
                    .any(|(idx, chunk)| {
                        self.completion_streaming_chunks[..idx]
                            .iter()
                            .any(|prior| prior.index == chunk.index)
                    });
            if self.completion_streaming_chunks.len() < expected_choices && !has_duplicate_index {
                return Ok(());
            }
            let chunks = std::mem::take(&mut self.completion_streaming_chunks);
            let bundle = chunks.len() == expected_choices && !has_duplicate_index;
            let responses = if bundle {
                vec![chunks]
            } else {
                chunks.into_iter().map(|chunk| vec![chunk]).collect()
            };
            for choices in responses {
                seq.responder()
                    .send(Response::CompletionChunk(CompletionChunkResponse {
                        id: seq.id.to_string(),
                        choices,
                        created: seq.creation_time() as u128,
                        model: model.clone(),
                        system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                        object: "text_completion".to_string(),
                        adapter_generation: seq
                            .adapter_generation()
                            .map(|generation| generation.to_string()),
                    }))
                    .await?;
            }
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
    use crate::tools::{
        state::required_tool_call_deadline_tokens, ToolCallFormat, ToolCallState, ToolChoice,
    };
    use crate::{Function, Tool, ToolType};
    use rand::RngCore;
    use std::collections::HashMap;
    use tokio::sync::mpsc::channel;

    #[test]
    fn image_hash_distinguishes_geometry() {
        let bytes = vec![1, 2, 3, 4, 5, 6];
        let wide =
            image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(2, 1, bytes.clone()).unwrap());
        let tall = image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(1, 2, bytes).unwrap());

        assert_eq!(wide.as_bytes(), tall.as_bytes());
        let images = SequenceImages::new(vec![wide, tall]);
        assert_ne!(images.hashes()[0], images.hashes()[1]);
    }

    #[test]
    fn image_hash_distinguishes_color_type() {
        let rgba = image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(1, 1, vec![1, 2, 3, 4]).unwrap(),
        );
        let luma_alpha = image::DynamicImage::ImageLumaA16(
            image::ImageBuffer::<image::LumaA<u16>, Vec<u16>>::from_raw(
                1,
                1,
                vec![u16::from_ne_bytes([1, 2]), u16::from_ne_bytes([3, 4])],
            )
            .unwrap(),
        );

        assert_eq!(rgba.as_bytes(), luma_alpha.as_bytes());
        assert_ne!(rgba.color(), luma_alpha.color());
        let images = SequenceImages::new(vec![rgba, luma_alpha]);
        assert_ne!(images.hashes()[0], images.hashes()[1]);
    }

    fn make_test_sequence() -> Sequence {
        make_test_sequence_with_seed(None)
    }

    fn test_logprobs(token: u32) -> Logprobs {
        Logprobs {
            token,
            logprob: 0.0,
            bytes: None,
            top_logprobs: None,
        }
    }

    fn test_logprobs_with_value(token: u32, logprob: f32) -> Logprobs {
        Logprobs {
            token,
            logprob,
            bytes: None,
            top_logprobs: Some(Vec::new()),
        }
    }

    fn test_usage() -> Usage {
        Usage {
            completion_tokens: 0,
            prompt_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            avg_tok_per_sec: 0.0,
            avg_prompt_tok_per_sec: 0.0,
            avg_compl_tok_per_sec: 0.0,
            total_time_sec: 0.0,
            total_prompt_time_sec: 0.0,
            total_completion_time_sec: 0.0,
        }
    }

    fn test_streaming_chunk(index: usize, finish_reason: Option<&str>) -> ChunkChoice {
        ChunkChoice {
            finish_reason: finish_reason.map(str::to_string),
            stop_sequence: None,
            index,
            delta: crate::Delta {
                content: Some(format!("choice {index}")),
                role: "assistant".to_string(),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
        }
    }

    fn make_test_sequence_with_seed(seed: Option<u64>) -> Sequence {
        let (tx, _rx) = channel(1);
        let sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            32,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
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
            None, // input_videos
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            false,
            vec![],
            seed,
        )
    }

    #[tokio::test]
    async fn remaining_choice_streams_after_another_choice_finishes() {
        let (tx, mut rx) = channel(8);
        let group = Arc::new(Mutex::new(SequenceGroup::new(2, true, true, None)));
        let mut first = make_test_sequence();
        first.responder = tx.clone();
        first.response_index = 0;
        first.group = group.clone();
        let mut second = make_test_sequence();
        second.responder = tx;
        second.response_index = 1;
        second.group = group.clone();

        group
            .lock()
            .await
            .chat_streaming_chunks
            .push(test_streaming_chunk(0, None));
        group
            .lock()
            .await
            .maybe_send_streaming_response(&first, "test".to_string(), None)
            .await
            .unwrap();
        assert!(rx.try_recv().is_err());

        group
            .lock()
            .await
            .chat_streaming_chunks
            .push(test_streaming_chunk(1, None));
        group
            .lock()
            .await
            .maybe_send_streaming_response(&second, "test".to_string(), None)
            .await
            .unwrap();
        let Response::Chunk(response) = rx.recv().await.unwrap() else {
            panic!("expected chat chunk");
        };
        assert_eq!(response.choices.len(), 2);

        group
            .lock()
            .await
            .chat_streaming_chunks
            .push(test_streaming_chunk(0, Some("stop")));
        group
            .lock()
            .await
            .maybe_send_streaming_response(&first, "test".to_string(), Some(test_usage()))
            .await
            .unwrap();
        let Response::Chunk(response) = rx.recv().await.unwrap() else {
            panic!("expected first terminal chunk");
        };
        assert_eq!(response.choices[0].index, 0);

        group
            .lock()
            .await
            .chat_streaming_chunks
            .push(test_streaming_chunk(1, Some("stop")));
        group
            .lock()
            .await
            .maybe_send_streaming_response(&second, "test".to_string(), Some(test_usage()))
            .await
            .unwrap();
        let Response::Chunk(response) = rx.recv().await.unwrap() else {
            panic!("expected remaining terminal chunk");
        };
        assert_eq!(response.choices[0].index, 1);
    }

    #[test]
    fn seeded_sequence_owns_its_sampling_stream() {
        let fallback = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(7)));
        let first = make_test_sequence_with_seed(Some(42));
        let second = make_test_sequence_with_seed(Some(42));

        assert!(!Arc::ptr_eq(&first.sampling_rng(&fallback), &fallback));
        assert_eq!(
            first.sampling_rng(&fallback).lock().unwrap().next_u64(),
            second.sampling_rng(&fallback).lock().unwrap().next_u64()
        );

        let unseeded = make_test_sequence();
        assert!(Arc::ptr_eq(&unseeded.sampling_rng(&fallback), &fallback));
    }

    #[test]
    fn prompt_rate_uses_accumulated_chunk_time() {
        let mut seq = make_test_sequence();
        seq.cache.push(None);
        seq.finish_prompt_timing(Duration::from_secs(1));
        seq.finish_prompt_timing(Duration::from_secs(1));

        assert_eq!(seq.total_prompt_time, Some(2_000));
        assert_eq!(seq.prompt_tok_per_sec, 4.0);
    }

    fn add_test_media(seq: &mut Sequence) {
        seq.multimodal = MultimodalData::new(
            Some(vec![image::DynamicImage::new_rgb8(1, 1)]),
            Some(vec![AudioInput {
                samples: vec![0.0],
                sample_rate: 16_000,
                channels: 1,
            }]),
            Some(vec![VideoInput::from_frames(
                vec![image::DynamicImage::new_rgb8(1, 1)],
                24.0,
                None,
            )]),
            None,
            None,
            None,
        );
    }

    #[test]
    fn ignore_eos_preserves_explicit_stops_and_length_limit() {
        let eos_tokens = [42];
        let mut seq = make_test_sequence();
        seq.ignore_eos = true;

        assert_eq!(seq.effective_eos_tokens(&eos_tokens, false), None);
        assert_eq!(seq.is_done(42, Some(&eos_tokens), 1024), None);

        seq.stop_tokens.push(42);
        assert_eq!(
            seq.is_done(42, Some(&eos_tokens), 1024),
            Some(StopReason::StopTok(42))
        );

        seq.stop_tokens.clear();
        seq.stop_strings.push("done".to_string());
        assert!(matches!(
            seq.add_token(test_logprobs(7), b"done".to_vec(), None),
            Some(StopReason::StopString { .. })
        ));

        seq.stop_strings.clear();
        seq.max_len = Some(1);
        assert_eq!(
            seq.is_done(7, Some(&eos_tokens), 1024),
            Some(StopReason::Length(1))
        );
    }

    #[test]
    fn constrained_token_stop_string_is_removed_before_final_output() {
        let mut seq = make_test_sequence();
        seq.stop_strings.push("STOP_BOUNDARY".to_string());

        let reason = seq.add_token(test_logprobs(7), b"GREEN STOP_BOUNDARY RED".to_vec(), None);

        assert!(matches!(reason, Some(StopReason::StopString { .. })));
        assert_eq!(seq.get_final_delta(), "GREEN ");
        assert!(!String::from_utf8_lossy(seq.completion_bytes()).contains("STOP_BOUNDARY"));
    }

    #[test]
    fn reasoning_parser_only_observes_visible_stop_filtered_bytes() {
        let mut seq = make_test_sequence();
        seq.stop_strings.push("STOP_BOUNDARY".to_string());
        seq.enable_reasoning(
            crate::reasoning_parsers::ReasoningMode::TagBased,
            Box::new(crate::reasoning_parsers::TagReasoningContext::new_think_tags()),
        );

        let reason = seq.add_token(
            test_logprobs(7),
            b"visible STOP_BOUNDARY leaked".to_vec(),
            None,
        );
        seq.finalize_reasoning();

        assert!(matches!(reason, Some(StopReason::StopString { .. })));
        assert_eq!(seq.get_response_content().as_deref(), Some("visible "));
        assert_eq!(seq.get_final_delta(), "visible ");
    }

    #[test]
    fn cross_token_stop_string_is_never_streamed() {
        let mut seq = make_test_sequence();
        seq.stop_strings.push("STOP_BOUNDARY".to_string());

        assert_eq!(
            seq.add_token(test_logprobs(7), b"GREEN STO".to_vec(), None),
            None
        );
        assert_eq!(seq.get_delta().unwrap(), Some("GREEN ".to_string()));

        let reason = seq.add_token(test_logprobs(8), b"P_BOUNDARY RED".to_vec(), None);
        assert!(matches!(reason, Some(StopReason::StopString { .. })));
        assert_eq!(seq.get_final_delta(), "");
        assert!(!String::from_utf8_lossy(seq.completion_bytes()).contains("STOP_BOUNDARY"));
    }

    #[test]
    fn unmatched_stop_prefix_is_flushed_on_eos() {
        let mut seq = make_test_sequence();
        seq.stop_strings.push("STOP_BOUNDARY".to_string());

        assert_eq!(
            seq.add_token(test_logprobs(7), b"GREEN STO".to_vec(), None),
            None
        );
        assert_eq!(seq.get_delta().unwrap(), Some("GREEN ".to_string()));
        assert_eq!(
            seq.add_token(test_logprobs(42), b"<eos>".to_vec(), Some(StopReason::Eos)),
            Some(StopReason::Eos)
        );
        assert_eq!(seq.get_final_delta(), "STO");
        assert!(!String::from_utf8_lossy(seq.completion_bytes()).contains("<eos>"));
    }

    #[test]
    fn stop_token_id_still_excludes_its_decoded_bytes() {
        let mut seq = make_test_sequence();

        assert_eq!(
            seq.add_token(
                test_logprobs(42),
                b"STOP_BOUNDARY".to_vec(),
                Some(StopReason::StopTok(42)),
            ),
            Some(StopReason::StopTok(42))
        );
        assert!(seq.completion_bytes().is_empty());
    }

    #[test]
    fn earliest_stop_string_wins_independent_of_configuration_order() {
        let mut seq = make_test_sequence();
        seq.stop_strings = vec!["RED".to_string(), "STOP".to_string()];

        let reason = seq.add_token(test_logprobs(7), b"GREEN STOP RED".to_vec(), None);

        assert_eq!(
            reason,
            Some(StopReason::StopString {
                stop_string_idx: 1,
                completion_bytes_pos: 6,
            })
        );
        assert_eq!(seq.get_final_delta(), "GREEN ");
    }

    #[test]
    fn stop_string_on_length_boundary_reports_stop() {
        let mut seq = make_test_sequence();
        seq.stop_strings.push("STOP_BOUNDARY".to_string());

        let reason = seq.add_token(
            test_logprobs(7),
            b"GREEN STOP_BOUNDARY RED".to_vec(),
            Some(StopReason::Length(1)),
        );

        assert!(matches!(reason, Some(StopReason::StopString { .. })));
        assert_eq!(reason.unwrap().to_string(), "stop");
        assert_eq!(seq.get_final_delta(), "GREEN ");
    }

    #[test]
    fn tool_parser_observes_a_stop_delimiter_hidden_from_output() {
        let mut seq = make_test_sequence();
        let tools = vec![weather_tool()];
        seq.stop_strings.push("<|eot|>".to_string());
        seq.tool_call_state = Some(
            ToolCallState::new(ToolChoice::Auto, Some(&tools), Some(ToolCallFormat::Atem)).unwrap(),
        );
        let output = b" to=get_weather<|message|><atem:function_calls><atem:invoke name=\"get_weather\"></atem:invoke></atem:function_calls><|eot|>";

        let reason = seq.add_token(test_logprobs(7), output.to_vec(), None);
        let parsed = seq
            .tool_call_state
            .as_mut()
            .unwrap()
            .finalize_for_response("", None, None, None)
            .unwrap();

        assert!(matches!(reason, Some(StopReason::StopString { .. })));
        assert!(!String::from_utf8_lossy(seq.completion_bytes()).contains("<|eot|>"));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn post_add_terminal_transition_flushes_an_unmatched_stop_prefix() {
        let mut seq = make_test_sequence();
        seq.stop_strings.push("STOP_BOUNDARY".to_string());

        assert_eq!(
            seq.add_token(test_logprobs(7), b"GREEN STO".to_vec(), None),
            None
        );
        seq.flush_stop_pending_bytes();

        assert_eq!(seq.get_final_delta(), "GREEN STO");
    }

    #[test]
    fn delayed_stop_prefix_emissions_preserve_source_logprobs() {
        let mut seq = make_test_sequence();
        seq.return_logprobs = true;
        seq.stream_logprobs = true;
        seq.stop_strings.push("STOP_BOUNDARY".to_string());

        assert_eq!(
            seq.add_token(test_logprobs_with_value(11, -1.0), b"STO".to_vec(), None,),
            None
        );
        assert!(seq.take_ready_streaming_emissions(false).is_empty());
        assert_eq!(
            seq.add_token(test_logprobs_with_value(12, -2.0), b"X".to_vec(), None,),
            None
        );

        let emissions = seq.take_ready_streaming_emissions(false);
        assert_eq!(emissions.len(), 2);
        assert_eq!(emissions[0].text, "STO");
        assert_eq!(emissions[0].bytes, b"STO");
        assert_eq!(emissions[0].logprobs.token, 11);
        assert_eq!(emissions[0].logprobs.logprob, -1.0);
        assert_eq!(emissions[1].text, "X");
        assert_eq!(emissions[1].bytes, b"X");
        assert_eq!(emissions[1].logprobs.token, 12);
        assert_eq!(emissions[1].logprobs.logprob, -2.0);
    }

    #[test]
    fn split_utf8_streaming_emissions_preserve_text_and_source_logprobs() {
        let mut seq = make_test_sequence();
        seq.return_logprobs = true;
        seq.stream_logprobs = true;

        assert_eq!(
            seq.add_token(test_logprobs_with_value(11, -1.0), vec![0xc3], None),
            None
        );
        assert!(seq.take_ready_streaming_emissions(false).is_empty());
        assert_eq!(
            seq.add_token(test_logprobs_with_value(12, -2.0), vec![0xa9], None),
            None
        );

        let emissions = seq.take_ready_streaming_emissions(false);
        assert_eq!(emissions.len(), 2);
        assert_eq!(
            emissions
                .iter()
                .map(|emission| emission.text.as_str())
                .collect::<String>(),
            "é"
        );
        assert_eq!(emissions[0].text, "");
        assert_eq!(emissions[0].bytes, [0xc3]);
        assert_eq!(emissions[0].logprobs.token, 11);
        assert_eq!(emissions[0].logprobs.logprob, -1.0);
        assert_eq!(emissions[1].text, "é");
        assert_eq!(emissions[1].bytes, [0xa9]);
        assert_eq!(emissions[1].logprobs.token, 12);
        assert_eq!(emissions[1].logprobs.logprob, -2.0);
    }

    #[test]
    fn engine_eos_override_applies_to_default_sequence() {
        let eos_tokens = [42];
        let seq = make_test_sequence();

        assert_eq!(
            seq.effective_eos_tokens(&eos_tokens, false),
            Some(eos_tokens.as_slice())
        );
        assert_eq!(seq.effective_eos_tokens(&eos_tokens, true), None);
    }

    fn weather_tool() -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: None,
                name: "get_weather".to_string(),
                parameters: None,
                strict: None,
            },
        }
    }

    fn add_required_tool(seq: &mut Sequence) {
        let tool = weather_tool();
        seq.tool_call_state =
            Some(ToolCallState::new(ToolChoice::Required, Some(&[tool]), None).unwrap());
    }

    fn required_tool_call_should_force(seq: &mut Sequence, max_model_len: usize) -> bool {
        let generated = seq.generated_len();
        let max_generation_len = seq.max_generation_len(max_model_len);
        let (_, remaining, _) =
            ToolCallState::required_tool_call_deadline_status(generated, max_generation_len);
        seq.tool_call_state
            .as_mut()
            .and_then(|state| {
                state.maybe_force_required_grammar(remaining, max_generation_len, false)
            })
            .is_some()
    }

    #[test]
    fn required_tool_call_deadline_clamps() {
        assert_eq!(required_tool_call_deadline_tokens(512), 1024);
        assert_eq!(required_tool_call_deadline_tokens(8192), 2048);
        assert_eq!(required_tool_call_deadline_tokens(32768), 4096);
    }

    #[test]
    fn final_delta_flushes_incomplete_utf8() {
        let mut seq = make_test_sequence();
        seq.completion_bytes = vec![b'a', 0xe2, 0x82];

        assert_eq!(seq.get_delta().unwrap(), Some("a".to_string()));
        assert_eq!(seq.stream_idx, 1);
        assert_eq!(seq.get_final_delta(), "\u{fffd}");
        assert_eq!(seq.stream_idx, seq.completion_bytes.len());
    }

    #[test]
    fn replacement_character_is_not_buffered() {
        let mut seq = make_test_sequence();
        seq.completion_bytes = "\u{fffd}".as_bytes().to_vec();

        assert_eq!(seq.get_delta().unwrap(), Some("\u{fffd}".to_string()));
        assert_eq!(seq.stream_idx, seq.completion_bytes.len());
    }

    #[test]
    fn invalid_bytes_do_not_consume_a_trailing_incomplete_character() {
        let mut seq = make_test_sequence();
        seq.completion_bytes = vec![0xff, 0xe2];

        assert_eq!(seq.get_delta().unwrap(), Some("\u{fffd}".to_string()));
        assert_eq!(seq.stream_idx, 1);

        seq.completion_bytes.extend([0x82, 0xac]);
        assert_eq!(seq.get_delta().unwrap(), Some("\u{20ac}".to_string()));
        assert_eq!(seq.stream_idx, seq.completion_bytes.len());
    }

    #[test]
    fn prefix_cache_hit_is_recorded_once_for_generation_sequences() {
        let mut seq = make_test_sequence();
        assert!(seq.record_prefix_cache_hit());
        assert!(!seq.record_prefix_cache_hit());

        seq.sequence_stepping_type = SeqStepType::OneShot;
        seq.prefix_cache_hit_recorded = false;
        assert!(!seq.record_prefix_cache_hit());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sampling_logprob_is_retained_for_multi_choice_groups() {
        assert!(!SequenceGroup::new(1, false, true, None).sampling_logprob_required());
        assert!(SequenceGroup::new(2, false, true, None).sampling_logprob_required());
        assert!(!SequenceGroup::new(1, false, false, Some(1)).sampling_logprob_required());
        assert!(SequenceGroup::new(1, false, false, Some(2)).sampling_logprob_required());
    }

    #[test]
    fn required_tool_call_forces_immediately_when_max_tokens_is_below_deadline() {
        let mut seq = make_test_sequence();
        add_required_tool(&mut seq);
        seq.set_max_len(512);

        assert!(required_tool_call_should_force(&mut seq, 8192));
    }

    #[test]
    fn required_tool_call_forces_at_remaining_deadline() {
        let mut seq = make_test_sequence();
        add_required_tool(&mut seq);
        seq.set_max_len(2048);

        assert!(!required_tool_call_should_force(&mut seq, 8192));
        seq.tokens.extend(std::iter::repeat_n(1, 1024));
        assert!(required_tool_call_should_force(&mut seq, 8192));
    }

    #[test]
    fn chunk_media_view_does_not_double_count_prefix_items() {
        let mut seq = make_test_sequence();
        seq.set_mm_features(vec![
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![123],
                offset: 0,
                length: 3,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 1..2,
                hashes: vec![456],
                offset: 4,
                length: 3,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Audio,
                item_range: 0..1,
                hashes: vec![789],
                offset: 7,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
        ]);

        let seq = seq.prefill_v2_normal(vec![], vec![7, 8], 4);

        assert_eq!(seq.prefix_cache_len(), 4);
        assert_eq!(
            seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Image),
            0
        );
        assert_eq!(
            seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Audio),
            0
        );
        assert_eq!(seq.count_prefix_cached_mm_items(), 0);
    }

    #[test]
    fn usage_reports_cached_prompt_tokens() {
        let seq = make_test_sequence();
        let seq = seq.prefill_v2_normal(vec![], vec![7, 8], 4);
        seq.update_time_info();
        let usage = seq.get_mut_group().get_usage();
        assert_eq!(
            usage.prompt_tokens_details.map(|d| d.cached_tokens),
            Some(4)
        );
    }

    #[test]
    fn usage_omits_prompt_details_without_prefix_cache() {
        let seq = make_test_sequence();
        let mut seq = seq.prefill_v2_normal(vec![], vec![7, 8], 0);
        seq.set_num_computed_tokens(seq.len());
        seq.update_time_info();
        assert!(seq
            .get_mut_group()
            .get_usage()
            .prompt_tokens_details
            .is_none());
    }

    #[test]
    fn usage_does_not_treat_prefill_progress_as_cached_tokens() {
        let seq = make_test_sequence();
        let mut seq = seq.prefill_v2_normal(vec![], vec![1, 2, 3, 4, 5, 6, 7, 8], 4);
        seq.set_num_computed_tokens(seq.len());
        seq.update_time_info();

        let usage = seq.get_mut_group().get_usage();
        assert_eq!(
            usage
                .prompt_tokens_details
                .map(|details| details.cached_tokens),
            Some(4)
        );
    }

    #[test]
    fn non_chunk_media_view_counts_prefix_items() {
        let mut seq = make_test_sequence();
        seq.set_mm_features(vec![MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![123],
            offset: 0,
            length: 3,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        }]);
        seq.set_prefix_cache_len(4);

        assert_eq!(seq.count_prefix_cached_mm_items(), 1);
        assert_eq!(
            seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Image),
            1
        );
    }

    #[test]
    fn multimodal_prefix_placeholder_delimited_ranges_include_wrappers() {
        let tokens = vec![1, 10, 20, 20, 11, 2, 10, 30, 30, 30, 11, 3];
        let img = find_placeholder_delimited_ranges(&tokens, 20, 10, 11);
        let video = find_placeholder_delimited_ranges(&tokens, 30, 10, 11);
        let fallback = find_placeholder_delimited_ranges(&tokens, 2, 99, 100);

        assert_eq!(img, vec![(1, 4)]);
        assert_eq!(video, vec![(6, 5)]);
        assert_eq!(fallback, vec![(5, 1)]);
    }

    #[test]
    fn multimodal_prefix_cache_len_clamps_inside_feature() {
        let features = vec![MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![123],
            offset: 31,
            length: 4,
            attention_policy: MultimodalAttentionPolicy::NonCausal,
            splittable: false,
        }];

        assert_eq!(clamp_prefix_cache_len_for_mm_features(32, 32, &features), 0);
        assert_eq!(
            clamp_prefix_cache_len_for_mm_features(35, 32, &features),
            35
        );
        assert_eq!(
            clamp_prefix_cache_len_for_mm_features(64, 32, &features),
            64
        );
    }

    #[test]
    fn multimodal_prefix_cache_len_clamps_to_a_fixed_point() {
        let features = vec![
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![1],
                offset: 16,
                length: 24,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 1..2,
                hashes: vec![2],
                offset: 40,
                length: 20,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
        ];

        assert_eq!(
            clamp_prefix_cache_len_for_mm_features(50, 16, &features),
            16
        );
    }

    #[test]
    fn unsplittable_causal_feature_clips_sequence_prefix_cache() {
        let mut seq = make_test_sequence();
        seq.set_mm_features(vec![MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![123],
            offset: 31,
            length: 4,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        }]);
        seq.set_prefix_cache_len(32);

        seq.clip_prefix_cache_len_for_mm_features(32);

        assert_eq!(seq.prefix_cache_len(), 0);
    }

    #[test]
    fn noncausal_prefix_clipping_keeps_its_original_scope() {
        let mut seq = make_test_sequence();
        seq.set_mm_features(vec![MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![123],
            offset: 31,
            length: 4,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        }]);
        seq.set_prefix_cache_len(32);

        seq.clip_prefix_cache_len_for_non_causal_mm_features(32);

        assert_eq!(seq.prefix_cache_len(), 32);
    }

    #[test]
    fn prefix_clipping_reaches_a_fixed_point_across_adjacent_features() {
        let mut seq = make_test_sequence();
        seq.set_mm_features(vec![
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![1],
                offset: 16,
                length: 24,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 1..2,
                hashes: vec![2],
                offset: 40,
                length: 20,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
        ]);
        seq.set_prefix_cache_len(50);

        seq.clip_prefix_cache_len_for_mm_features(16);

        assert_eq!(seq.prefix_cache_len(), 16);
    }

    #[test]
    fn chunk_media_accessors_expose_only_the_active_kind() {
        let mut seq = make_test_sequence();
        add_test_media(&mut seq);
        seq.set_mm_features(vec![
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![1],
                offset: 0,
                length: 2,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Audio,
                item_range: 0..1,
                hashes: vec![2],
                offset: 4,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Video,
                item_range: 0..1,
                hashes: vec![3],
                offset: 6,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
        ]);
        seq.set_prefix_cache_len(4);
        seq.set_prefill_toks(vec![1, 2, 3, 4, 5]);

        assert!(!seq.has_images());
        assert!(seq.image_hashes().unwrap().is_empty());
        assert!(seq.take_images().is_none());
        assert!(seq.has_audios());
        assert_eq!(seq.audio_hashes().unwrap().len(), 1);
        assert_eq!(seq.take_audios().unwrap().len(), 1);
        assert!(!seq.has_videos());
        assert!(seq.video_hashes().unwrap().is_empty());
        assert!(seq.take_videos().is_none());
    }

    #[test]
    fn chunk_media_accessors_use_active_local_item_offsets() {
        let mut seq = make_test_sequence();
        seq.multimodal = MultimodalData::new(
            Some(vec![
                image::DynamicImage::new_rgb8(1, 1),
                image::DynamicImage::new_rgb8(2, 1),
            ]),
            None,
            None,
            None,
            None,
            None,
        );
        seq.set_mm_features(vec![
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![1],
                offset: 0,
                length: 2,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 1..2,
                hashes: vec![2],
                offset: 4,
                length: 2,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
        ]);
        seq.set_prefix_cache_len(4);
        seq.set_prefill_toks(vec![1, 2, 3, 4, 5, 6]);

        assert_eq!(seq.active_prompt_query_range(), Some(4..6));
        assert_eq!(seq.active_prompt_local_query_range(), Some(4..6));
        assert_eq!(seq.prompt_position_source_toks(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(
            seq.active_multimodal_item_range(MultimodalKind::Image),
            Some(1..2)
        );
        assert_eq!(
            seq.active_local_multimodal_item_range(MultimodalKind::Image, 2),
            Some(1..2)
        );
        assert_eq!(seq.image_hashes().unwrap().len(), 1);
        assert_eq!(seq.take_images().unwrap()[0].width(), 2);
        assert_eq!(seq.count_prefix_cached_mm_items(), 0);
    }

    #[test]
    fn suffix_only_prefill_view_survives_scheduler_state_transition() {
        let mut seq = make_test_sequence();
        seq.multimodal = MultimodalData::new(
            Some(vec![
                image::DynamicImage::new_rgb8(1, 1),
                image::DynamicImage::new_rgb8(2, 1),
            ]),
            Some(vec![
                AudioInput {
                    samples: vec![1.0],
                    sample_rate: 16_000,
                    channels: 1,
                },
                AudioInput {
                    samples: vec![2.0],
                    sample_rate: 16_000,
                    channels: 1,
                },
            ]),
            Some(vec![
                VideoInput::from_frames(vec![image::DynamicImage::new_rgb8(1, 1)], 24.0, None),
                VideoInput::from_frames(vec![image::DynamicImage::new_rgb8(2, 1)], 24.0, None),
            ]),
            None,
            None,
            None,
        );
        seq.set_mm_features(vec![
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![1],
                offset: 0,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Audio,
                item_range: 0..1,
                hashes: vec![2],
                offset: 1,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Video,
                item_range: 0..1,
                hashes: vec![3],
                offset: 2,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 1..2,
                hashes: vec![4],
                offset: 3,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Audio,
                item_range: 1..2,
                hashes: vec![5],
                offset: 4,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Video,
                item_range: 1..2,
                hashes: vec![6],
                offset: 5,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
        ]);
        seq.keep_num_images(1);
        seq.keep_num_audios(1);
        seq.keep_num_video_frames(1);
        let mut seq = seq.prefill_v2_normal(vec![], vec![4, 5, 6], 3);
        seq.set_state(SequenceState::Waiting);
        seq.set_state(SequenceState::RunningPrompt);

        assert!(seq.has_suffix_only_prefill_toks());
        assert_eq!(seq.active_prompt_query_range(), Some(3..6));
        assert_eq!(seq.active_prompt_local_query_range(), Some(0..3));
        assert_eq!(seq.prompt_position_source_toks(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            seq.active_multimodal_item_range(MultimodalKind::Image),
            Some(1..2)
        );
        assert_eq!(
            seq.active_local_multimodal_item_range(MultimodalKind::Image, 1),
            Some(0..1)
        );
        assert_eq!(seq.image_hashes().unwrap().len(), 1);
        assert_eq!(seq.audio_hashes().unwrap().len(), 1);
        assert_eq!(seq.video_hashes().unwrap().len(), 1);
        assert_eq!(seq.take_images().unwrap()[0].width(), 2);
        assert_eq!(seq.take_audios().unwrap()[0].samples, vec![2.0]);
        assert_eq!(seq.take_videos().unwrap()[0].frames[0].width(), 2);
        assert_eq!(seq.count_prefix_cached_mm_items(), 0);

        seq.reset_prefill_toks();
        assert!(!seq.has_suffix_only_prefill_toks());
        assert_eq!(seq.active_prompt_query_range(), None);
        assert!(!seq.is_chunked_prefill_view());
    }

    #[test]
    fn video_prefix_retention_uses_frame_counts() {
        let mut seq = make_test_sequence();
        seq.multimodal = MultimodalData::new(
            None,
            None,
            Some(vec![VideoInput::from_frames(
                vec![
                    image::DynamicImage::new_rgb8(1, 1),
                    image::DynamicImage::new_rgb8(2, 1),
                    image::DynamicImage::new_rgb8(3, 1),
                ],
                24.0,
                None,
            )]),
            None,
            None,
            None,
        );

        seq.keep_num_video_frames(2);

        let videos = seq.clone_videos().unwrap();
        assert_eq!(videos.len(), 1);
        assert_eq!(videos[0].frames.len(), 2);
        assert_eq!(videos[0].frames[0].width(), 2);
        assert_eq!(seq.multimodal.video_hashes().unwrap().len(), 3);
    }

    #[test]
    fn video_retention_by_item_keeps_complete_videos() {
        let mut seq = make_test_sequence();
        seq.multimodal = MultimodalData::new(
            None,
            None,
            Some(vec![
                VideoInput::from_frames(vec![image::DynamicImage::new_rgb8(1, 1)], 24.0, None),
                VideoInput::from_frames(
                    vec![
                        image::DynamicImage::new_rgb8(2, 1),
                        image::DynamicImage::new_rgb8(3, 1),
                    ],
                    24.0,
                    None,
                ),
            ]),
            None,
            None,
            None,
        );

        seq.keep_num_videos(1);

        let videos = seq.clone_videos().unwrap();
        assert_eq!(videos.len(), 1);
        assert_eq!(videos[0].frames.len(), 2);
        assert_eq!(videos[0].frames[0].width(), 2);
    }

    #[test]
    fn non_chunk_media_accessors_preserve_full_media() {
        let mut seq = make_test_sequence();
        add_test_media(&mut seq);

        assert_eq!(seq.image_hashes().unwrap().len(), 1);
        assert_eq!(seq.take_images().unwrap().len(), 1);
        assert_eq!(seq.audio_hashes().unwrap().len(), 1);
        assert_eq!(seq.take_audios().unwrap().len(), 1);
        assert_eq!(seq.video_hashes().unwrap().len(), 1);
        assert_eq!(seq.take_videos().unwrap().len(), 1);
    }
}
