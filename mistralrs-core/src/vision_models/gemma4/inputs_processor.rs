#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{imageops, DynamicImage, GenericImageView, Rgba, RgbaImage};
use mistralrs_vision::{ApplyTransforms, Rescale, ToTensorNoNorm, Transforms};
use tokenizers::Tokenizer;

use crate::{
    block_diffusion::block_denoising_progress_emitters,
    device_map::DeviceMapper,
    paged_attention::block_hash::{MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind},
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_completion_input_windowed, get_prompt_input,
            PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{
        build_mm_features_from_ranges, build_mm_features_from_ranges_with_policy,
        find_image_placeholder_ranges, Sequence,
    },
    vision_models::gemma4::audio_processing::AudioProcessor,
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        multimodal_layout::{
            MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout,
            PackedMultimodalLayout, RequestMultimodalLayout,
        },
        preprocessor_config::{PreProcessorConfig, ToFilter},
        processor_config::ProcessorConfig,
        ModelInputs,
    },
};

use super::{config::Gemma4BidirectionalAttention, Gemma4SpecificArgs};

// ── Token constants ────────────────────────────────────────────────────────

const IMAGE_TOKEN: &str = "<|image|>";
const BOI_TOKEN: &str = "<|image>";
const EOI_TOKEN: &str = "<image|>";
pub const IMAGE_TOKEN_ID: u32 = 258880;

const AUDIO_TOKEN: &str = "<|audio|>";
const BOA_TOKEN: &str = "<|audio>";
const EOA_TOKEN: &str = "<audio|>";
pub const AUDIO_TOKEN_ID: u32 = 258881;

const VIDEO_TOKEN: &str = "<|video|>";
pub const VIDEO_TOKEN_ID: u32 = 258884;

// ── Processor (public, created by the pipeline loader) ─────────────────────

pub struct Gemma4Processor {
    patch_size: usize,
    pooling_kernel_size: usize,
    default_output_length: usize,
    max_patches: usize,
    audio_seq_length: usize,
    raw_audio_frame_size: Option<usize>,
    video_max_soft_tokens: usize,
    is_unified: bool,
    supports_images: bool,
    supports_audio: bool,
    decode_window: Option<usize>,
    bidirectional_attention: Gemma4BidirectionalAttention,
    vision_attention_on_full_layers: bool,
}

pub struct Gemma4ProcessorSettings {
    pub processor_config: ProcessorConfig,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub default_output_length: usize,
    pub supports_images: bool,
    pub supports_audio: bool,
    pub raw_audio_frame_size: Option<usize>,
    pub is_unified: bool,
    /// Tokens fed per decode step; block-diffusion models set this to the canvas length.
    pub decode_window: Option<usize>,
    pub bidirectional_attention: Gemma4BidirectionalAttention,
    pub vision_attention_on_full_layers: bool,
}

impl Gemma4Processor {
    pub fn new(settings: Gemma4ProcessorSettings) -> Self {
        let Gemma4ProcessorSettings {
            processor_config,
            patch_size,
            pooling_kernel_size,
            default_output_length,
            supports_images,
            supports_audio,
            raw_audio_frame_size,
            is_unified,
            decode_window,
            bidirectional_attention,
            vision_attention_on_full_layers,
        } = settings;
        let max_patches = default_output_length * pooling_kernel_size * pooling_kernel_size;
        let audio_seq_length = processor_config.audio_seq_length.unwrap_or(750);
        let video_max_soft_tokens = processor_config.video_max_soft_tokens.unwrap_or(70);

        Self {
            patch_size,
            pooling_kernel_size,
            default_output_length,
            max_patches,
            audio_seq_length,
            raw_audio_frame_size,
            video_max_soft_tokens,
            is_unified,
            supports_images,
            supports_audio,
            decode_window,
            bidirectional_attention,
            vision_attention_on_full_layers,
        }
    }
}

impl Processor for Gemma4Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        let video_max_patches =
            self.video_max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size;
        Arc::new(Gemma4ImageProcessor {
            patch_size: self.patch_size,
            pooling_kernel_size: self.pooling_kernel_size,
            default_output_length: self.default_output_length,
            max_patches: self.max_patches,
            audio_seq_length: self.audio_seq_length,
            raw_audio_frame_size: self.raw_audio_frame_size,
            video_max_soft_tokens: self.video_max_soft_tokens,
            video_max_patches,
            is_unified: self.is_unified,
            supports_images: self.supports_images,
            supports_audio: self.supports_audio,
            decode_window: self.decode_window,
            bidirectional_attention: self.bidirectional_attention,
            vision_attention_on_full_layers: self.vision_attention_on_full_layers,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            IMAGE_TOKEN,
            BOI_TOKEN,
            EOI_TOKEN,
            AUDIO_TOKEN,
            BOA_TOKEN,
            EOA_TOKEN,
            VIDEO_TOKEN,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::KeepWithAudioAfterText
    }
}

// ── Image processor (InputsProcessor + ImagePreProcessor) ──────────────────

#[allow(dead_code)]
struct Gemma4ImageProcessor {
    patch_size: usize,
    pooling_kernel_size: usize,
    default_output_length: usize,
    max_patches: usize,
    audio_seq_length: usize,
    raw_audio_frame_size: Option<usize>,
    video_max_soft_tokens: usize,
    video_max_patches: usize,
    is_unified: bool,
    supports_images: bool,
    supports_audio: bool,
    decode_window: Option<usize>,
    bidirectional_attention: Gemma4BidirectionalAttention,
    vision_attention_on_full_layers: bool,
}

type UnifiedMediaPreprocessOutput = (Tensor, Tensor, Vec<(u32, u32)>);

fn unified_patch_positions(ph: usize, pw: usize, capacity: usize) -> Result<Vec<i64>> {
    let num_patches = ph
        .checked_mul(pw)
        .ok_or_else(|| candle_core::Error::msg("Gemma4 unified patch count overflow"))?;
    if num_patches > capacity {
        candle_core::bail!(
            "Gemma4 unified media produced {num_patches} patches, exceeding max {capacity}."
        );
    }
    let mut positions = vec![-1i64; capacity * 2];
    for row in 0..ph {
        for col in 0..pw {
            let idx = row * pw + col;
            positions[2 * idx] = col as i64;
            positions[2 * idx + 1] = row as i64;
        }
    }
    Ok(positions)
}

fn convert_to_rgb(image: &DynamicImage) -> DynamicImage {
    if !image.color().has_alpha() {
        return DynamicImage::ImageRgb8(image.to_rgb8());
    }
    let (width, height) = image.dimensions();
    let mut background = RgbaImage::from_pixel(width, height, Rgba([u8::MAX; 4]));
    imageops::overlay(&mut background, &image.to_rgba8(), 0, 0);
    DynamicImage::ImageRgba8(background).into_rgb8().into()
}

impl Gemma4ImageProcessor {
    /// Compute how many vision soft tokens a single image will produce after
    /// aspect-ratio-preserving resize + patch embedding + pooling.
    fn output_tokens_for_size(&self, new_h: usize, new_w: usize) -> usize {
        let ph = new_h / self.patch_size;
        let pw = new_w / self.patch_size;
        let pool_area = self.pooling_kernel_size * self.pooling_kernel_size;
        (ph * pw) / pool_area
    }

    /// Aspect-ratio-preserving resize: compute (new_h, new_w) for a given
    /// original image size, ensuring that the result is a multiple of
    /// `grid_unit = pooling_kernel_size * patch_size` and does not exceed
    /// `max_patches` patches.
    ///
    /// Matches HuggingFace `get_aspect_ratio_preserving_size` including edge
    /// case handling for extreme aspect ratios.
    fn compute_resize_dims(&self, orig_h: usize, orig_w: usize) -> Result<(usize, usize)> {
        if orig_h == 0 || orig_w == 0 {
            candle_core::bail!(
                "Gemma4 image resize: input dimensions must be non-zero, got {orig_h}x{orig_w}"
            );
        }

        let target_px = self.max_patches * self.patch_size * self.patch_size;
        let grid_unit = self.pooling_kernel_size * self.patch_size; // 48
        let pool_area = self.pooling_kernel_size * self.pooling_kernel_size;
        let max_side_length = (self.max_patches / pool_area) * grid_unit;

        let factor = (target_px as f64 / (orig_h as f64 * orig_w as f64)).sqrt();

        let ideal_h = orig_h as f64 * factor;
        let ideal_w = orig_w as f64 * factor;

        let mut new_h = (ideal_h / grid_unit as f64).floor() as usize * grid_unit;
        let mut new_w = (ideal_w / grid_unit as f64).floor() as usize * grid_unit;

        if new_h == 0 && new_w == 0 {
            candle_core::bail!(
                "Gemma4 image resize: both dimensions round to 0 for input {orig_h}x{orig_w}"
            );
        }

        if new_h == 0 {
            new_h = grid_unit;
            new_w = ((orig_w / orig_h) * grid_unit).min(max_side_length);
            new_w = (new_w / grid_unit).max(1) * grid_unit;
        } else if new_w == 0 {
            new_w = grid_unit;
            new_h = ((orig_h / orig_w) * grid_unit).min(max_side_length);
            new_h = (new_h / grid_unit).max(1) * grid_unit;
        }

        if new_h * new_w > target_px {
            candle_core::bail!(
                "Gemma4 image resize: {new_h}x{new_w} = {} pixels exceeds patch budget of {target_px} \
                 for input {orig_h}x{orig_w}",
                new_h * new_w
            );
        }

        Ok((new_h, new_w))
    }

    /// Build the expanded token sequence for a single image:
    /// `<start_of_image>{N * <image_soft_token>}<end_of_image>`
    fn build_image_sequence(&self, num_tokens: usize) -> String {
        let image_tokens = vec![IMAGE_TOKEN.to_string(); num_tokens].join("");
        format!("{BOI_TOKEN}{image_tokens}{EOI_TOKEN}")
    }

    fn raw_image_placeholder_ranges(tokenizer: &Tokenizer, tokens: &[u32]) -> Vec<(usize, usize)> {
        let image_token_id = tokenizer.token_to_id(IMAGE_TOKEN).unwrap_or(IMAGE_TOKEN_ID);
        find_image_placeholder_ranges(tokens, image_token_id)
            .into_iter()
            .filter(|(_, length)| *length == 1)
            .collect()
    }

    fn expand_raw_image_placeholders(
        &self,
        tokenizer: &Tokenizer,
        tokens: &[u32],
        per_image_dims: &[(usize, usize)],
    ) -> anyhow::Result<Option<Vec<u32>>> {
        let ranges = Self::raw_image_placeholder_ranges(tokenizer, tokens);
        if ranges.is_empty() {
            return Ok(None);
        }
        if ranges.len() != per_image_dims.len() {
            anyhow::bail!(
                "Gemma 4 has {} image placeholders but {} image inputs",
                ranges.len(),
                per_image_dims.len()
            );
        }
        let mut expanded = Vec::with_capacity(tokens.len());
        let mut cursor = 0usize;
        for (image_idx, (offset, length)) in ranges.into_iter().enumerate() {
            expanded.extend_from_slice(&tokens[cursor..offset]);
            let (new_h, new_w) = per_image_dims[image_idx];
            let replacement = self.build_image_sequence(self.output_tokens_for_size(new_h, new_w));
            let replacement = tokenizer
                .encode_fast(replacement.as_str(), false)
                .map_err(|err| anyhow::Error::msg(err.to_string()))?;
            expanded.extend_from_slice(replacement.get_ids());
            cursor = offset + length;
        }
        expanded.extend_from_slice(&tokens[cursor..]);
        Ok(Some(expanded))
    }

    fn expand_raw_image_placeholders_for_seq(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut Sequence,
        per_image_dims: &[(usize, usize)],
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<bool> {
        let Some(ids) =
            self.expand_raw_image_placeholders(tokenizer, seq.get_toks(), per_image_dims)?
        else {
            return Ok(false);
        };
        let has_prefill_toks = seq.has_prefill_toks();
        let prompt = tokenizer
            .decode(&ids, false)
            .map_err(|err| anyhow::Error::msg(err.to_string()))?;
        seq.set_initial_prompt(prompt);
        seq.set_toks_and_reallocate(ids.clone(), paged_attn_metadata);
        if has_prefill_toks {
            seq.set_prefill_toks(ids);
        }
        seq.multimodal.has_changed_prompt = true;
        Ok(true)
    }

    fn preprocess_unified_images(
        &self,
        images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<UnifiedMediaPreprocessOutput> {
        self.preprocess_unified_images_at_sizes(images, config, device, None)
    }

    fn preprocess_unified_images_at_sizes(
        &self,
        mut images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
        target_sizes: Option<&[(usize, usize)]>,
    ) -> Result<UnifiedMediaPreprocessOutput> {
        if target_sizes.is_some_and(|sizes| sizes.len() != images.len()) {
            candle_core::bail!("Gemma4 unified media size count does not match input count");
        }
        let do_rescale = config.do_rescale.unwrap_or(true);
        let rescale_factor = config.rescale_factor.unwrap_or(1.0 / 255.0);
        let do_convert_rgb = config.do_convert_rgb.unwrap_or(true);
        let resample = config.resampling.to_filter()?;
        let model_patch_size = self.patch_size * self.pooling_kernel_size;
        let patch_dim = model_patch_size * model_patch_size * 3;

        for image in images.iter_mut() {
            if do_convert_rgb {
                *image = convert_to_rgb(image);
            }
        }

        let mut pixel_values = Vec::new();
        let mut position_ids = Vec::new();
        let mut image_sizes = Vec::new();

        for (image_index, image) in images.into_iter().enumerate() {
            let (w, h) = image.dimensions();
            let (new_h, new_w) = match target_sizes {
                Some(sizes) => sizes[image_index],
                None => self.compute_resize_dims(h as usize, w as usize)?,
            };
            let resized = image.resize_exact(new_w as u32, new_h as u32, resample);
            let transforms = Transforms {
                input: &ToTensorNoNorm,
                inner_transforms: &[&do_rescale.then_some(Rescale {
                    factor: Some(rescale_factor),
                })],
            };
            let tensor = resized.apply(transforms, device)?;
            let (_, h, w) = tensor.dims3()?;
            let ph = h / model_patch_size;
            let pw = w / model_patch_size;
            let num_patches = ph * pw;
            let positions = unified_patch_positions(ph, pw, self.default_output_length)?;

            let patches = tensor
                .reshape((3, ph, model_patch_size, pw, model_patch_size))?
                .permute((1, 3, 2, 4, 0))?
                .reshape((num_patches, patch_dim))?
                .contiguous()?;
            let padded_patches = Tensor::zeros(
                (self.default_output_length, patch_dim),
                patches.dtype(),
                device,
            )?
            .slice_assign(&[0..num_patches, 0..patch_dim], &patches)?;

            let positions = Tensor::from_vec(positions, (self.default_output_length, 2), device)?;

            pixel_values.push(padded_patches.unsqueeze(0)?);
            position_ids.push(positions.unsqueeze(0)?);
            image_sizes.push((new_h as u32, new_w as u32));
        }

        Ok((
            Tensor::cat(&pixel_values, 0)?,
            Tensor::cat(&position_ids, 0)?,
            image_sizes,
        ))
    }

    fn compute_audio_num_tokens(&self, num_mel_frames: usize) -> usize {
        if num_mel_frames == 0 {
            return 0;
        }

        let mut t = num_mel_frames;
        for _ in 0..2 {
            t = (t + 2 - 3) / 2 + 1;
        }
        t.min(self.audio_seq_length)
    }

    /// Build the expanded token sequence for audio:
    /// `<start_of_audio>{N * <audio_soft_token>}<end_of_audio>`
    fn build_audio_sequence(&self, num_tokens: usize) -> String {
        let audio_tokens = vec![AUDIO_TOKEN.to_string(); num_tokens].join("");
        format!("{BOA_TOKEN}{audio_tokens}{EOA_TOKEN}")
    }

    fn expand_raw_audio_placeholders_for_seq(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut Sequence,
        token_counts: &[usize],
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<bool> {
        let mut prompt = tokenizer
            .decode(seq.get_toks(), false)
            .map_err(|err| anyhow::Error::msg(err.to_string()))?;
        let positions = prompt
            .match_indices(AUDIO_TOKEN)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if positions.is_empty() {
            return Ok(false);
        }
        if positions.len() != token_counts.len() {
            anyhow::bail!(
                "Gemma 4 has {} audio placeholders but {} audio inputs",
                positions.len(),
                token_counts.len()
            );
        }
        for (&position, &token_count) in positions.iter().zip(token_counts).rev() {
            let replacement = self.build_audio_sequence(token_count);
            prompt = format!(
                "{}{}{}",
                &prompt[..position],
                replacement,
                &prompt[position + AUDIO_TOKEN.len()..],
            );
        }
        let tokens = tokenizer
            .encode_fast(prompt.as_str(), false)
            .map_err(|err| anyhow::Error::msg(err.to_string()))?;
        seq.set_initial_prompt(prompt);
        seq.set_toks_and_reallocate(tokens.get_ids().to_vec(), paged_attn_metadata);
        Ok(true)
    }

    /// Compute resize dimensions for a video frame (uses smaller patch budget).
    fn compute_video_resize_dims(&self, orig_h: usize, orig_w: usize) -> Result<(usize, usize)> {
        if orig_h == 0 || orig_w == 0 {
            candle_core::bail!(
                "Gemma4 video resize: input dimensions must be non-zero, got {orig_h}x{orig_w}"
            );
        }

        let target_px = self.video_max_patches * self.patch_size * self.patch_size;
        let grid_unit = self.pooling_kernel_size * self.patch_size;
        let pool_area = self.pooling_kernel_size * self.pooling_kernel_size;
        let max_side_length = (self.video_max_patches / pool_area) * grid_unit;

        let factor = (target_px as f64 / (orig_h as f64 * orig_w as f64)).sqrt();

        let ideal_h = orig_h as f64 * factor;
        let ideal_w = orig_w as f64 * factor;

        let mut new_h = (ideal_h / grid_unit as f64).floor() as usize * grid_unit;
        let mut new_w = (ideal_w / grid_unit as f64).floor() as usize * grid_unit;

        if new_h == 0 && new_w == 0 {
            candle_core::bail!(
                "Gemma4 video resize: both dimensions round to 0 for input {orig_h}x{orig_w}"
            );
        }

        if new_h == 0 {
            new_h = grid_unit;
            new_w = ((orig_w / orig_h) * grid_unit).min(max_side_length);
            new_w = (new_w / grid_unit).max(1) * grid_unit;
        } else if new_w == 0 {
            new_w = grid_unit;
            new_h = ((orig_h / orig_w) * grid_unit).min(max_side_length);
            new_h = (new_h / grid_unit).max(1) * grid_unit;
        }

        if new_h * new_w > target_px {
            candle_core::bail!(
                "Gemma4 video resize: {new_h}x{new_w} = {} pixels exceeds patch budget of {target_px} \
                 for input {orig_h}x{orig_w}",
                new_h * new_w
            );
        }

        Ok((new_h, new_w))
    }

    /// Video soft tokens per frame after resize + patch + pool.
    fn video_tokens_for_size(&self, new_h: usize, new_w: usize) -> usize {
        let ph = new_h / self.patch_size;
        let pw = new_w / self.patch_size;
        let pool_area = self.pooling_kernel_size * self.pooling_kernel_size;
        (ph * pw) / pool_area
    }

    /// Build the expanded token sequence for all frames of a single video.
    ///
    /// Format per frame: `"mm:ss <boi>{N × <video_token>}<eoi>"`
    /// All frames are space-joined.
    fn build_video_sequence(&self, timestamps: &[String], tokens_per_frame: usize) -> String {
        let video_tokens = vec![VIDEO_TOKEN.to_string(); tokens_per_frame].join("");
        timestamps
            .iter()
            .map(|ts| format!("{ts} {BOI_TOKEN}{video_tokens}{EOI_TOKEN}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn cached_tokens_for_ranges(prefix_len: usize, ranges: &[(usize, usize)]) -> Vec<usize> {
    ranges
        .iter()
        .map(|&(offset, length)| prefix_len.saturating_sub(offset).min(length))
        .collect()
}

fn cat_padded_audio(mels: &[Tensor], masks: &[Tensor]) -> Result<(Tensor, Tensor)> {
    if mels.len() != masks.len() || mels.is_empty() {
        candle_core::bail!("Gemma 4 audio tensor and mask counts must match");
    }
    let max_frames = mels
        .iter()
        .map(|mel| mel.dim(1))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let mut padded_mels = Vec::with_capacity(mels.len());
    let mut padded_masks = Vec::with_capacity(masks.len());
    for (mel, mask) in mels.iter().zip(masks) {
        let (batch, frames, _) = mel.dims3()?;
        let (mask_batch, mask_frames) = mask.dims2()?;
        if batch != 1 || mask_batch != 1 || frames != mask_frames {
            candle_core::bail!("Gemma 4 audio tensor and mask shapes are inconsistent");
        }
        let padding = max_frames - frames;
        padded_mels.push(mel.pad_with_zeros(1, 0, padding)?);
        if padding == 0 {
            padded_masks.push(mask.clone());
        } else {
            let tail = Tensor::ones((1, padding), mask.dtype(), mask.device())?;
            padded_masks.push(Tensor::cat(&[mask, &tail], 1)?);
        }
    }
    Ok((
        Tensor::cat(&padded_mels, 0)?,
        Tensor::cat(&padded_masks, 0)?,
    ))
}

fn cat_padded_spatial(tensors: &[Tensor]) -> Result<Tensor> {
    if tensors.is_empty() {
        candle_core::bail!("Gemma 4 spatial tensor batch cannot be empty");
    }
    let shapes = tensors
        .iter()
        .map(Tensor::dims4)
        .collect::<Result<Vec<_>>>()?;
    let max_h = shapes.iter().map(|shape| shape.2).max().unwrap_or(0);
    let max_w = shapes.iter().map(|shape| shape.3).max().unwrap_or(0);
    let mut padded = Vec::with_capacity(tensors.len());
    for (tensor, &(batch, _, height, width)) in tensors.iter().zip(&shapes) {
        if batch != 1 {
            candle_core::bail!("Gemma 4 spatial media items must have batch size 1");
        }
        padded.push(
            tensor
                .pad_with_zeros(2, 0, max_h - height)?
                .pad_with_zeros(3, 0, max_w - width)?,
        );
    }
    Tensor::cat(&padded, 0)
}

fn uncached_video_frame_mask(
    ranges: &[(usize, usize)],
    cached_tokens: &[usize],
    frame_count: usize,
) -> Result<Vec<bool>> {
    if ranges.len() != frame_count || cached_tokens.len() != frame_count {
        candle_core::bail!("Gemma 4 video frame metadata length mismatch");
    }
    Ok(ranges
        .iter()
        .zip(cached_tokens)
        .map(|(&(_, length), &cached)| cached < length)
        .collect())
}

fn active_placeholder_ranges(
    token_source: &[u32],
    token_id: u32,
    query: Option<std::ops::Range<usize>>,
) -> Vec<(usize, usize)> {
    let ranges = find_image_placeholder_ranges(token_source, token_id);
    let Some(query) = query else {
        return ranges;
    };
    ranges
        .into_iter()
        .filter(|(offset, length)| *offset < query.end && *offset + *length > query.start)
        .collect()
}

fn active_media_ranges(seq: &Sequence, token_id: u32) -> Vec<(usize, usize)> {
    let query = if seq.is_chunked_prefill_view() {
        seq.active_prompt_query_range()
    } else {
        None
    };
    active_placeholder_ranges(seq.prompt_position_source_toks(), token_id, query)
}

fn media_attention_policy(
    bidirectional_attention: Gemma4BidirectionalAttention,
    kind: MultimodalKind,
) -> MultimodalAttentionPolicy {
    match (bidirectional_attention, kind) {
        (Gemma4BidirectionalAttention::Vision, MultimodalKind::Image | MultimodalKind::Video) => {
            MultimodalAttentionPolicy::NonCausal
        }
        _ => MultimodalAttentionPolicy::Causal,
    }
}

fn validate_sliding_noncausal_ranges<'a>(
    features: impl IntoIterator<Item = &'a MultiModalFeature>,
    sliding_window: Option<usize>,
) -> anyhow::Result<()> {
    let Some(sliding_window) = sliding_window else {
        return Ok(());
    };
    if let Some(feature) = features.into_iter().find(|feature| {
        feature.attention_policy == MultimodalAttentionPolicy::NonCausal
            && feature.length > sliding_window
    }) {
        anyhow::bail!(
            "Gemma 4 media span length {} exceeds its sliding attention window {sliding_window}",
            feature.length
        );
    }
    Ok(())
}

fn rebuild_mm_features(
    seq: &mut Sequence,
    bidirectional_attention: Gemma4BidirectionalAttention,
) -> anyhow::Result<()> {
    let mut features = Vec::new();
    let image_count = seq.images().map_or(0, <[_]>::len);
    let image_hashes = seq.multimodal.image_hashes().unwrap_or_default();
    let image_ranges = find_image_placeholder_ranges(seq.get_toks(), IMAGE_TOKEN_ID);
    if image_hashes.len() != image_count || image_ranges.len() != image_count {
        anyhow::bail!(
            "Gemma 4 has {image_count} image inputs, {} hashes, and {} placeholder ranges",
            image_hashes.len(),
            image_ranges.len()
        );
    }
    features.extend(build_mm_features_from_ranges_with_policy(
        &image_ranges,
        image_hashes,
        MultimodalKind::Image,
        media_attention_policy(bidirectional_attention, MultimodalKind::Image),
    ));

    let audio_count = seq.audios().map_or(0, <[_]>::len);
    let audio_hashes = seq.multimodal.audio_hashes().unwrap_or_default();
    let audio_ranges = find_image_placeholder_ranges(seq.get_toks(), AUDIO_TOKEN_ID);
    if audio_hashes.len() != audio_count || audio_ranges.len() != audio_count {
        anyhow::bail!(
            "Gemma 4 has {audio_count} audio inputs, {} hashes, and {} placeholder ranges",
            audio_hashes.len(),
            audio_ranges.len()
        );
    }
    features.extend(build_mm_features_from_ranges(
        &audio_ranges,
        audio_hashes,
        MultimodalKind::Audio,
    ));

    let video_count = seq.videos().map_or(0, |videos| {
        videos.iter().map(|video| video.frames.len()).sum()
    });
    let video_hashes = seq.multimodal.video_hashes().unwrap_or_default();
    let video_ranges = find_image_placeholder_ranges(seq.get_toks(), VIDEO_TOKEN_ID);
    if video_hashes.len() != video_count || video_ranges.len() != video_count {
        anyhow::bail!(
            "Gemma 4 has {video_count} video frames, {} hashes, and {} placeholder ranges",
            video_hashes.len(),
            video_ranges.len()
        );
    }
    features.extend(build_mm_features_from_ranges_with_policy(
        &video_ranges,
        video_hashes,
        MultimodalKind::Video,
        media_attention_policy(bidirectional_attention, MultimodalKind::Video),
    ));

    features.sort_by_key(|f| f.offset);
    seq.set_mm_features(features);
    Ok(())
}

fn gemma4_layout_items_from_features(
    features: &[MultiModalFeature],
) -> Result<Vec<MultimodalItemLayout>> {
    features
        .iter()
        .enumerate()
        .map(|(item_index, feature)| {
            if feature.hashes.len() != 1 || feature.item_range.len() != 1 {
                candle_core::bail!(
                    "Gemma 4 packed prefill requires one encoder item per placeholder span"
                );
            }
            let placeholder_end = feature
                .offset
                .checked_add(feature.length)
                .ok_or_else(|| candle_core::Error::msg("Gemma 4 placeholder range overflow"))?;
            let placeholder = feature.offset..placeholder_end;
            MultimodalItemLayout::new(
                MultimodalEncoderKey {
                    kind: feature.kind,
                    hash: feature.hashes[0],
                },
                item_index,
                placeholder.clone(),
                feature.attention_policy,
                vec![MultimodalEmbeddingMap::contiguous(placeholder, 0, 0)?],
            )
        })
        .collect()
}

fn gemma4_layout_items(seq: &Sequence) -> Result<Vec<MultimodalItemLayout>> {
    gemma4_layout_items_from_features(seq.mm_features())
}

fn gemma4_packed_layout(
    input_seqs: &[&mut Sequence],
    query_lens: &[usize],
) -> Result<PackedMultimodalLayout> {
    if input_seqs.len() != query_lens.len() {
        candle_core::bail!("Gemma 4 packed multimodal metadata length mismatch");
    }
    let requests = input_seqs
        .iter()
        .zip(query_lens)
        .map(|(seq, &query_len)| {
            if query_len != seq.get_toks().len() {
                candle_core::bail!(
                    "Gemma 4 packed multimodal prefill requires the complete uncached prompt"
                );
            }
            Ok(RequestMultimodalLayout {
                sequence_id: *seq.id(),
                query: 0..query_len,
                items: gemma4_layout_items(seq)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    PackedMultimodalLayout::new(&requests)
}

// ── InputsProcessor ────────────────────────────────────────────────────────

impl InputsProcessor for Gemma4ImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _device: &Device,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        if self.bidirectional_attention == Gemma4BidirectionalAttention::All
            && paged_attn_metadata.is_some()
        {
            anyhow::bail!("Gemma 4 all-token bidirectional attention does not support KV caching");
        }
        let Some(tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "Gemma4ImageProcessor requires a specified tokenizer.",
            ));
        };
        for seq in input_seqs.iter_mut() {
            if seq.multimodal.has_changed_prompt && !seq.mm_features().is_empty() {
                continue;
            }
            let mut changed_prompt = false;
            let raw_audio_placeholder_count =
                find_image_placeholder_ranges(seq.get_toks(), AUDIO_TOKEN_ID)
                    .iter()
                    .filter(|(_, length)| *length == 1)
                    .count();
            if raw_audio_placeholder_count > 0 {
                let audios = seq.audios().unwrap_or_default();
                if raw_audio_placeholder_count != audios.len() {
                    anyhow::bail!(
                        "Gemma 4 has {raw_audio_placeholder_count} audio placeholders but {} audio inputs",
                        audios.len()
                    );
                }
                let config = other_config
                    .as_ref()
                    .expect("Need a PreProcessorConfig config.");
                let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
                let audio_processor = AudioProcessor::new(config);
                let (_, _, frame_counts) = if let Some(frame_size) = self.raw_audio_frame_size {
                    audio_processor.process_raw_frame_audios(audios, &Device::Cpu, frame_size)?
                } else {
                    audio_processor.process_audios(audios, &Device::Cpu)?
                };
                let token_counts = frame_counts
                    .into_iter()
                    .map(|num_frames| {
                        if self.raw_audio_frame_size.is_some() {
                            num_frames
                        } else {
                            self.compute_audio_num_tokens(num_frames)
                        }
                    })
                    .collect::<Vec<_>>();
                changed_prompt |= self.expand_raw_audio_placeholders_for_seq(
                    &tokenizer,
                    seq,
                    &token_counts,
                    paged_attn_metadata.as_deref_mut(),
                )?;
            }

            if self.supports_images {
                if let Some(images) = seq.images() {
                    let per_image_dims = images
                        .iter()
                        .map(|img| {
                            let (w, h) = img.dimensions();
                            self.compute_resize_dims(h as usize, w as usize)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    self.expand_raw_image_placeholders_for_seq(
                        &tokenizer,
                        seq,
                        &per_image_dims,
                        paged_attn_metadata.as_deref_mut(),
                    )?;
                }
            }

            let raw_video_placeholder_count =
                find_image_placeholder_ranges(seq.get_toks(), VIDEO_TOKEN_ID)
                    .iter()
                    .filter(|(_, length)| *length == 1)
                    .count();
            if raw_video_placeholder_count > 0 {
                let videos = seq.videos().unwrap_or_default();
                if raw_video_placeholder_count != videos.len() {
                    anyhow::bail!(
                        "Gemma 4 has {raw_video_placeholder_count} video placeholders but {} video inputs",
                        videos.len()
                    );
                }
                if videos.iter().any(|video| video.frames.is_empty()) {
                    anyhow::bail!("Gemma 4 video inputs must contain at least one frame");
                }
                let mut prompt = tokenizer
                    .decode(seq.get_toks(), false)
                    .expect("Detokenization failed!");
                let original_prompt = prompt.clone();

                for video in videos {
                    let (sample_w, sample_h) = video.frames[0].dimensions();
                    let (new_h, new_w) =
                        self.compute_video_resize_dims(sample_h as usize, sample_w as usize)?;
                    let replacement = self.build_video_sequence(
                        &video.timestamp_strings(),
                        self.video_tokens_for_size(new_h, new_w),
                    );
                    if let Some(pos) = prompt.find(VIDEO_TOKEN) {
                        prompt = format!(
                            "{}{}{}",
                            &prompt[..pos],
                            replacement,
                            &prompt[pos + VIDEO_TOKEN.len()..],
                        );
                    }
                }

                if prompt != original_prompt {
                    seq.set_initial_prompt(prompt.clone());
                    let toks = tokenizer
                        .encode_fast(prompt.as_str(), false)
                        .expect("Tokenization failed!");
                    let ids = toks.get_ids().to_vec();
                    let frame_count = seq.multimodal.video_hashes().map_or(0, <[u64]>::len);
                    let range_count = find_image_placeholder_ranges(&ids, VIDEO_TOKEN_ID).len();
                    if range_count != frame_count {
                        anyhow::bail!(
                            "Gemma 4 has {range_count} video frame placeholders but {frame_count} frames"
                        );
                    }
                    seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_deref_mut());
                    seq.multimodal.has_changed_prompt = true;
                }
            }

            seq.multimodal.has_changed_prompt |= changed_prompt;
            rebuild_mm_features(seq, self.bidirectional_attention)?;
        }

        Ok(())
    }

    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        sliding_window: Option<usize>,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> anyhow::Result<InputProcessorOutput> {
        if is_xlora {
            return Err(anyhow::Error::msg(
                "Cannot make inputs for X-LoRA vision model.",
            ));
        }
        if no_kv_cache {
            return Err(anyhow::Error::msg("Vision model must have kv cache."));
        }
        let Some(tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "Gemma4ImageProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let preprocessor_config: &PreProcessorConfig =
            config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().any(|seq| seq.has_images());
        let has_audios = input_seqs.iter().any(|seq| seq.has_audios());
        let has_videos = input_seqs.iter().any(|seq| seq.has_videos());
        let preserve_media = input_seqs
            .iter()
            .map(|seq| !seq.multimodal.has_changed_prompt)
            .collect::<Vec<_>>();

        let mut changed_sequence_ids = Vec::new();
        let mut image_hashes_accum = Vec::new();
        let mut image_cached_tokens_accum = Vec::new();
        let mut audio_hashes_accum = Vec::new();
        let mut audio_cached_tokens_accum = Vec::new();
        let mut video_pixel_values_accum = Vec::new();
        let mut video_position_ids_accum = Vec::new();
        let mut video_hashes_accum = Vec::new();
        let mut video_cached_tokens_accum = Vec::new();
        let mut video_sizes_accum = Vec::new();

        // ── Audio processing ───────────────────────────────────────────────
        if has_audios && !self.supports_audio {
            return Err(anyhow::Error::msg(
                "This image processor does not support audio.",
            ));
        }

        let (audio_mel, audio_mel_mask) = if has_audios {
            let mut audio_mel_accum = Vec::new();
            let mut audio_mask_accum = Vec::new();
            let audio_processor = AudioProcessor::new(preprocessor_config);

            for (seq_index, seq) in input_seqs.iter_mut().enumerate() {
                if !seq.has_audios() {
                    continue;
                }
                let audios = if preserve_media[seq_index] {
                    seq.clone_audios()
                } else {
                    seq.take_audios()
                };
                if let Some(audios) = audios {
                    let (seq_audio_mel, seq_audio_mask, seq_audio_frame_counts) =
                        if let Some(frame_size) = self.raw_audio_frame_size {
                            audio_processor.process_raw_frame_audios(&audios, device, frame_size)?
                        } else {
                            audio_processor.process_audios(&audios, device)?
                        };
                    let seq_audio_num_tokens = seq_audio_frame_counts
                        .into_iter()
                        .map(|num_frames| {
                            if self.raw_audio_frame_size.is_some() {
                                num_frames
                            } else {
                                self.compute_audio_num_tokens(num_frames)
                            }
                        })
                        .collect::<Vec<_>>();

                    if !seq.multimodal.has_changed_prompt {
                        let mut prompt = tokenizer
                            .decode(seq.get_toks(), false)
                            .expect("Detokenization failed!");

                        let positions: Vec<usize> = prompt
                            .match_indices(AUDIO_TOKEN)
                            .map(|(idx, _)| idx)
                            .collect();

                        for (i, &pos) in positions.iter().enumerate().rev() {
                            let num_tokens = seq_audio_num_tokens
                                .get(i)
                                .copied()
                                .unwrap_or(self.audio_seq_length);
                            let replacement = self.build_audio_sequence(num_tokens);

                            prompt = format!(
                                "{}{}{}",
                                &prompt[..pos],
                                replacement,
                                &prompt[pos + AUDIO_TOKEN.len()..],
                            );
                        }

                        seq.set_initial_prompt(prompt.clone());
                        let toks = tokenizer
                            .encode_fast(prompt.as_str(), false)
                            .expect("Tokenization failed!");

                        let ids = toks.get_ids().to_vec();
                        seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());

                        changed_sequence_ids.push(*seq.id());
                    }

                    let n_audio = audios.len();
                    let audio_ranges = active_media_ranges(seq, AUDIO_TOKEN_ID);
                    let cached_audio_tokens =
                        cached_tokens_for_ranges(seq.prefix_cache_len(), &audio_ranges);
                    let seq_audio_hashes = seq.audio_hashes().unwrap_or(&[]);
                    if n_audio > 0 {
                        for idx in 0..n_audio {
                            let total_tokens = audio_ranges
                                .get(idx)
                                .map(|(_, length)| *length)
                                .unwrap_or_else(|| {
                                    seq_audio_num_tokens
                                        .get(idx)
                                        .copied()
                                        .unwrap_or(self.audio_seq_length)
                                });
                            let cached_tokens = cached_audio_tokens
                                .get(idx)
                                .copied()
                                .unwrap_or(0)
                                .min(total_tokens);
                            if cached_tokens >= total_tokens {
                                continue;
                            }
                            audio_mel_accum.push(seq_audio_mel.get(idx)?.unsqueeze(0)?);
                            audio_mask_accum.push(seq_audio_mask.get(idx)?.unsqueeze(0)?);
                            if let Some(&hash) = seq_audio_hashes.get(idx) {
                                audio_hashes_accum.push(hash);
                            }
                            audio_cached_tokens_accum.push(cached_tokens);
                        }
                    }
                }
            }

            if !audio_mel_accum.is_empty() {
                let (mel, mask) = cat_padded_audio(&audio_mel_accum, &audio_mask_accum)?;
                (Some(mel), Some(mask))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // ── Image processing ───────────────────────────────────────────────
        let pixel_values = if has_images {
            if !self.supports_images {
                return Err(anyhow::Error::msg(
                    "This image processor does not support images.",
                ));
            }

            let mut pixel_values_accum = Vec::new();
            let mut image_position_ids_accum = Vec::new();
            let mut image_sizes_accum = Vec::new();

            for (seq_index, seq) in input_seqs.iter_mut().enumerate() {
                if !seq.has_images() {
                    continue;
                }
                let images = if preserve_media[seq_index] {
                    seq.clone_images()
                } else {
                    seq.take_images()
                }
                .expect("Need to have images by this point.");

                // Compute per-image resize dimensions *before* preprocessing so
                // we can build the correct per-image token sequences.
                let per_image_dims: Vec<(usize, usize)> = images
                    .iter()
                    .map(|img| {
                        let (w, h) = img.dimensions();
                        self.compute_resize_dims(h as usize, w as usize)
                    })
                    .collect::<Result<Vec<_>>>()?;

                let (pixel_values, image_position_ids, image_sizes_all) = if self.is_unified {
                    let (pixel_values, image_position_ids, image_sizes_all) =
                        self.preprocess_unified_images(images, preprocessor_config, device)?;
                    (pixel_values, Some(image_position_ids), image_sizes_all)
                } else {
                    let PreprocessedImages {
                        pixel_values,
                        pixel_attention_mask: _,
                        image_sizes: _,
                        num_img_tokens: _,
                        aspect_ratio_ids: _,
                        aspect_ratio_mask: _,
                        num_tiles: _,
                        image_grid_thw: _,
                        video_grid_thw: _,
                        rows: _,
                        cols: _,
                        pixel_values_list: _,
                        tgt_sizes: _,
                        image_sizes_all,
                        num_crops: _,
                    } = self
                        .preprocess(
                            images,
                            vec![],
                            preprocessor_config,
                            device,
                            (usize::MAX, usize::MAX),
                        )
                        .expect("Preprocessing failed");
                    (pixel_values, None, image_sizes_all.unwrap_or_default())
                };

                self.expand_raw_image_placeholders_for_seq(
                    &tokenizer,
                    seq,
                    &per_image_dims,
                    paged_attn_metadata.as_mut(),
                )?;

                // Per-sequence prefix cache trimming of pixel_values
                let n_images = pixel_values.dim(0).unwrap_or(0);
                let image_ranges = active_media_ranges(seq, IMAGE_TOKEN_ID);
                let cached_image_tokens =
                    cached_tokens_for_ranges(seq.prefix_cache_len(), &image_ranges);
                let seq_image_hashes = seq.image_hashes().unwrap_or(&[]);
                let image_sizes = image_sizes_all;
                for idx in 0..n_images {
                    let total_tokens = image_ranges
                        .get(idx)
                        .map(|(_, length)| *length)
                        .unwrap_or_else(|| {
                            image_sizes
                                .get(idx)
                                .map(|&(h, w)| self.output_tokens_for_size(h as usize, w as usize))
                                .unwrap_or(0)
                        });
                    let cached_tokens = cached_image_tokens
                        .get(idx)
                        .copied()
                        .unwrap_or(0)
                        .min(total_tokens);
                    if cached_tokens >= total_tokens {
                        continue;
                    }
                    pixel_values_accum.push(pixel_values.get(idx)?.unsqueeze(0)?);
                    if let Some(image_position_ids) = image_position_ids.as_ref() {
                        image_position_ids_accum.push(image_position_ids.get(idx)?.unsqueeze(0)?);
                    }
                    if let Some(&size) = image_sizes.get(idx) {
                        image_sizes_accum.push(size);
                    }
                    if let Some(&hash) = seq_image_hashes.get(idx) {
                        image_hashes_accum.push(hash);
                    }
                    image_cached_tokens_accum.push(cached_tokens);
                }
            }

            if pixel_values_accum.is_empty() {
                (None, vec![], None)
            } else {
                let pixel_values = if self.is_unified {
                    Tensor::cat(&pixel_values_accum, 0)?
                } else {
                    cat_padded_spatial(&pixel_values_accum)?
                };
                (
                    Some(pixel_values),
                    image_sizes_accum,
                    if image_position_ids_accum.is_empty() {
                        None
                    } else {
                        Some(Tensor::cat(&image_position_ids_accum, 0)?)
                    },
                )
            }
        } else {
            (None, vec![], None)
        };

        // Video processing.
        let (video_pixel_values, video_position_ids) = if has_videos {
            for (seq_index, seq) in input_seqs.iter_mut().enumerate() {
                if !seq.has_videos() {
                    continue;
                }
                // If this is a new turn (has_changed_prompt is false) and the video
                // placeholders have already been expanded into per-frame soft tokens
                // from a prior turn, skip re-processing.  The KV / encoder caches
                // already hold the embeddings.
                //
                // We must NOT skip when has_changed_prompt is true, because that
                // means we are on a subsequent chunk of the SAME turn, so the frames
                // still need to be encoded for the tokens in this chunk.
                if !seq.multimodal.has_changed_prompt {
                    let toks = seq.get_toks();
                    let video_ranges = find_image_placeholder_ranges(toks, VIDEO_TOKEN_ID);
                    let already_expanded =
                        !video_ranges.is_empty() && video_ranges.iter().all(|(_, len)| *len > 1);
                    if already_expanded {
                        continue;
                    }
                }

                let videos = if preserve_media[seq_index] {
                    seq.clone_videos()
                } else {
                    seq.take_videos()
                };
                if let Some(videos) = videos {
                    let frame_accum_start = video_pixel_values_accum.len();
                    for video in &videos {
                        if video.frames.is_empty() {
                            continue;
                        }

                        // Compute per-frame resize dimensions using video patch budget
                        let (sample_w, sample_h) = video.frames[0].dimensions();
                        let (new_h, new_w) =
                            self.compute_video_resize_dims(sample_h as usize, sample_w as usize)?;
                        let tokens_per_frame = self.video_tokens_for_size(new_h, new_w);
                        let timestamps = video.timestamp_strings();

                        let has_raw_video_placeholder =
                            find_image_placeholder_ranges(seq.get_toks(), VIDEO_TOKEN_ID)
                                .iter()
                                .any(|(_, length)| *length == 1);
                        if has_raw_video_placeholder {
                            let mut prompt = tokenizer
                                .decode(seq.get_toks(), false)
                                .expect("Detokenization failed!");

                            if let Some(pos) = prompt.find(VIDEO_TOKEN) {
                                let replacement =
                                    self.build_video_sequence(&timestamps, tokens_per_frame);
                                prompt = format!(
                                    "{}{}{}",
                                    &prompt[..pos],
                                    replacement,
                                    &prompt[pos + VIDEO_TOKEN.len()..],
                                );
                            }

                            seq.set_initial_prompt(prompt.clone());
                            let toks = tokenizer
                                .encode_fast(prompt.as_str(), false)
                                .expect("Tokenization failed!");
                            let ids = toks.get_ids().to_vec();
                            seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());
                            changed_sequence_ids.push(*seq.id());
                        }

                        if self.is_unified {
                            let sizes = vec![(new_h, new_w); video.frames.len()];
                            let (pixels, positions, processed_sizes) = self
                                .preprocess_unified_images_at_sizes(
                                    video.frames.clone(),
                                    preprocessor_config,
                                    device,
                                    Some(&sizes),
                                )?;
                            for idx in 0..video.frames.len() {
                                video_pixel_values_accum.push(pixels.get(idx)?.unsqueeze(0)?);
                                video_position_ids_accum.push(positions.get(idx)?.unsqueeze(0)?);
                            }
                            video_sizes_accum.extend(processed_sizes);
                        } else {
                            let do_rescale = preprocessor_config.do_rescale.unwrap_or(true);
                            let rescale_factor =
                                preprocessor_config.rescale_factor.unwrap_or(1.0 / 255.0);
                            let resample = preprocessor_config.resampling.to_filter()?;

                            for frame in &video.frames {
                                let frame_rgb = convert_to_rgb(frame);
                                let resized =
                                    frame_rgb.resize_exact(new_w as u32, new_h as u32, resample);

                                let transforms = Transforms {
                                    input: &ToTensorNoNorm,
                                    inner_transforms: &[&do_rescale.then_some(Rescale {
                                        factor: Some(rescale_factor),
                                    })],
                                };

                                let tensor = resized.apply(transforms, device)?;
                                video_pixel_values_accum.push(tensor.unsqueeze(0)?);
                                video_sizes_accum.push((new_h as u32, new_w as u32));
                            }
                        }
                    }

                    // Track per-frame video hashes and cached tokens.
                    // Unlike images (1 hash per image), videos need 1 hash per
                    // frame so the encoder cache can look up individual frames.
                    let video_ranges = active_media_ranges(seq, VIDEO_TOKEN_ID);
                    let cached_video_tokens =
                        cached_tokens_for_ranges(seq.prefix_cache_len(), &video_ranges);

                    let frame_hashes = videos
                        .iter()
                        .flat_map(|video| video.frame_hashes())
                        .collect::<Vec<_>>();
                    let frame_count = video_pixel_values_accum.len() - frame_accum_start;
                    if frame_hashes.len() != frame_count || video_ranges.len() != frame_count {
                        anyhow::bail!(
                            "Gemma 4 has {frame_count} video frames, {} hashes, and {} placeholder ranges",
                            frame_hashes.len(),
                            video_ranges.len()
                        );
                    }
                    let active_frames = uncached_video_frame_mask(
                        &video_ranges,
                        &cached_video_tokens,
                        frame_count,
                    )?;
                    for idx in (0..frame_count).rev() {
                        if !active_frames[idx] {
                            video_pixel_values_accum.remove(frame_accum_start + idx);
                            video_sizes_accum.remove(frame_accum_start + idx);
                            if self.is_unified {
                                video_position_ids_accum.remove(frame_accum_start + idx);
                            }
                        }
                    }
                    for (idx, hash) in frame_hashes.into_iter().enumerate() {
                        if active_frames[idx] {
                            video_hashes_accum.push(hash);
                            video_cached_tokens_accum.push(cached_video_tokens[idx]);
                        }
                    }
                }
            }

            if video_pixel_values_accum.is_empty() {
                (None, None)
            } else if self.is_unified {
                (
                    Some(Tensor::cat(&video_pixel_values_accum, 0)?),
                    Some(Tensor::cat(&video_position_ids_accum, 0)?),
                )
            } else {
                // Pad all frames to the same spatial dimensions
                let max_h = video_sizes_accum.iter().map(|(h, _)| *h).max().unwrap_or(0) as usize;
                let max_w = video_sizes_accum.iter().map(|(_, w)| *w).max().unwrap_or(0) as usize;

                let mut padded = Vec::new();
                for (pv, &(h, w)) in video_pixel_values_accum
                    .iter()
                    .zip(video_sizes_accum.iter())
                {
                    let h = h as usize;
                    let w = w as usize;
                    if h < max_h || w < max_w {
                        let p =
                            pv.pad_with_zeros(2, 0, max_h - h)?
                                .pad_with_zeros(3, 0, max_w - w)?;
                        padded.push(p);
                    } else {
                        padded.push(pv.clone());
                    }
                }
                (Some(Tensor::cat(&padded, 0)?), None)
            }
        } else {
            (None, None)
        };

        for seq in input_seqs.iter_mut() {
            if seq.mm_features().is_empty() {
                rebuild_mm_features(seq, self.bidirectional_attention)?;
            }
            if changed_sequence_ids.contains(seq.id()) {
                seq.multimodal.has_changed_prompt = true;
            }
        }
        if let Some(metadata) = paged_attn_metadata.as_mut() {
            validate_sliding_noncausal_ranges(
                input_seqs.iter().flat_map(|seq| seq.mm_features()),
                metadata.sliding_window,
            )?;
            metadata
                .set_noncausal_mm_context_views(input_seqs, self.vision_attention_on_full_layers);
        }

        // ── Build final model inputs ───────────────────────────────────────
        let text_models_inputs_processor::InnerInputProcessorOutput {
            inputs:
                text_models_inputs_processor::InputMetadata {
                    input,
                    positions,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                },
            seq_indices,
        } = if is_prompt {
            get_prompt_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )?
        } else if let Some(decode_window) = self.decode_window {
            let mut out = get_completion_input_windowed(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
                decode_window,
            )?;
            // The window is re-encoded context, not parallel decode queries: only the last
            // position's logits matter downstream.
            for (start, len) in out.inputs.context_lens.iter_mut() {
                if *len > 1 {
                    *start = *len - 1;
                    *len = 1;
                }
            }
            out
        } else {
            get_completion_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )?
        };
        let block_denoising_progress = block_denoising_progress_emitters(
            Some(tokenizer.clone()),
            input_seqs,
            &seq_indices,
            return_raw_logits,
        );

        let (pixel_values, image_sizes, image_position_ids) = if is_prompt {
            pixel_values
        } else {
            (None, vec![], None)
        };

        let (video_pixel_values, video_position_ids) = if is_prompt {
            (video_pixel_values, video_position_ids)
        } else {
            (None, None)
        };
        let packed_layout = if is_prompt && flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    anyhow::Error::msg("packed Gemma 4 prefill requires logical query lengths")
                })?;
            let layout = gemma4_packed_layout(input_seqs, query_lens)?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "Gemma 4 packed layout has {} tokens but input has {}",
                    layout.token_count(),
                    input.dim(1)?
                );
            }
            Some(layout)
        } else {
            None
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(Gemma4SpecificArgs {
                image_position_ids,
                audio_mel,
                audio_mel_mask,
                image_hashes: if is_prompt {
                    image_hashes_accum
                } else {
                    vec![]
                },
                image_cached_tokens: if is_prompt {
                    image_cached_tokens_accum
                } else {
                    vec![]
                },
                image_sizes,
                audio_hashes: if is_prompt {
                    audio_hashes_accum
                } else {
                    vec![]
                },
                audio_cached_tokens: if is_prompt {
                    audio_cached_tokens_accum
                } else {
                    vec![]
                },
                video_pixel_values,
                video_position_ids,
                video_hashes: if is_prompt {
                    video_hashes_accum
                } else {
                    vec![]
                },
                video_cached_tokens: if is_prompt {
                    video_cached_tokens_accum
                } else {
                    vec![]
                },
                video_sizes: if is_prompt { video_sizes_accum } else { vec![] },
                packed_layout,
                block_denoising_progress,
            }),
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind: if is_prompt {
                crate::pipeline::RecurrentBatchKind::Prefill
            } else {
                crate::pipeline::RecurrentBatchKind::Decode
            },
            adapter_leases: crate::vision_models::adapter_leases(input_seqs, &seq_indices),
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }
}

// ── ImagePreProcessor ──────────────────────────────────────────────────────

impl ImagePreProcessor for Gemma4ImageProcessor {
    // Gemma4 rescales to [0, 1] but does NOT apply ImageNet normalization.
    const DEFAULT_MEAN: [f64; 3] = [0.0, 0.0, 0.0];
    const DEFAULT_STD: [f64; 3] = [1.0, 1.0, 1.0];

    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        // Videos are processed separately in process_inputs(), so we only handle
        // images here.
        let _ = videos;

        let do_rescale = config.do_rescale.unwrap_or(true);
        let rescale_factor = config.rescale_factor.unwrap_or(1.0 / 255.0);
        let do_convert_rgb = config.do_convert_rgb.unwrap_or(true);
        let resample = config.resampling.to_filter()?;

        for image in images.iter_mut() {
            if do_convert_rgb {
                *image = convert_to_rgb(image);
            }
        }

        let mut pixel_values = Vec::new();
        let mut image_sizes = Vec::new();

        for image in images {
            let (w, h) = image.dimensions();
            let (new_h, new_w) = self.compute_resize_dims(h as usize, w as usize)?;

            // resize_exact takes (width, height, filter)
            let resized = image.resize_exact(new_w as u32, new_h as u32, resample);

            let transforms = Transforms {
                input: &ToTensorNoNorm,
                inner_transforms: &[&do_rescale.then_some(Rescale {
                    factor: Some(rescale_factor),
                })],
            };

            let tensor = resized.apply(transforms, device)?;
            pixel_values.push(tensor.unsqueeze(0)?);
            image_sizes.push((new_h as u32, new_w as u32));
        }

        // All images may have different spatial dimensions.  We still need to
        // return a single `pixel_values` tensor.  When sizes differ we pad each
        // image tensor to the batch-maximum height/width so they can be
        // concatenated along dim-0.
        let max_h = image_sizes.iter().map(|(h, _)| *h).max().unwrap_or(0) as usize;
        let max_w = image_sizes.iter().map(|(_, w)| *w).max().unwrap_or(0) as usize;

        let mut padded = Vec::new();
        for (pv, &(h, w)) in pixel_values.iter().zip(image_sizes.iter()) {
            let h = h as usize;
            let w = w as usize;
            if h < max_h || w < max_w {
                // pv shape: [1, 3, h, w] -> pad height and width
                let pad_h = max_h - h;
                let pad_w = max_w - w;
                let p = pv
                    .pad_with_zeros(2, 0, pad_h)?
                    .pad_with_zeros(3, 0, pad_w)?;
                padded.push(p);
            } else {
                padded.push(pv.clone());
            }
        }

        Ok(PreprocessedImages {
            pixel_values: Tensor::cat(&padded, 0)?,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rows: None,
            cols: None,
            pixel_values_list: None,
            tgt_sizes: None,
            image_sizes_all: Some(image_sizes),
            num_crops: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::vision_models::processor_config::ProcessorConfig;
    use image::{DynamicImage, Rgba, RgbaImage};

    #[test]
    fn convert_to_rgb_composites_alpha_on_white() {
        let image = DynamicImage::ImageRgba8(RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgba([10, 20, 30, 0])
            } else {
                Rgba([10, 20, 30, u8::MAX])
            }
        }));

        let rgb = convert_to_rgb(&image).into_rgb8();

        assert_eq!(rgb.get_pixel(0, 0).0, [u8::MAX; 3]);
        assert_eq!(rgb.get_pixel(1, 0).0, [10, 20, 30]);
    }

    fn unified_test_processor() -> Gemma4ImageProcessor {
        Gemma4ImageProcessor {
            patch_size: 2,
            pooling_kernel_size: 1,
            default_output_length: 4,
            max_patches: 4,
            audio_seq_length: 4,
            raw_audio_frame_size: None,
            video_max_soft_tokens: 4,
            video_max_patches: 4,
            is_unified: true,
            supports_images: true,
            supports_audio: false,
            decode_window: None,
            bidirectional_attention: Gemma4BidirectionalAttention::Vision,
            vision_attention_on_full_layers: false,
        }
    }

    #[test]
    fn defaults_audio_seq_length_to_reference_cap() {
        let processor = Gemma4Processor::new(Gemma4ProcessorSettings {
            processor_config: ProcessorConfig::default(),
            patch_size: 16,
            pooling_kernel_size: 3,
            default_output_length: 280,
            supports_images: true,
            supports_audio: true,
            raw_audio_frame_size: None,
            is_unified: false,
            decode_window: None,
            bidirectional_attention: Gemma4BidirectionalAttention::Vision,
            vision_attention_on_full_layers: false,
        });
        assert_eq!(processor.audio_seq_length, 750);
    }

    #[test]
    fn cached_tokens_for_ranges_handles_partial_overlap() {
        let ranges = vec![(5, 4), (12, 3), (20, 2)];

        assert_eq!(cached_tokens_for_ranges(0, &ranges), vec![0, 0, 0]);
        assert_eq!(cached_tokens_for_ranges(7, &ranges), vec![2, 0, 0]);
        assert_eq!(cached_tokens_for_ranges(13, &ranges), vec![4, 1, 0]);
        assert_eq!(cached_tokens_for_ranges(30, &ranges), vec![4, 3, 2]);
    }

    #[test]
    fn normal_suffix_cache_keeps_media_ranges_in_prompt_coordinates() {
        let source = [
            1,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            2,
            3,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            4,
            AUDIO_TOKEN_ID,
            AUDIO_TOKEN_ID,
            5,
            VIDEO_TOKEN_ID,
            VIDEO_TOKEN_ID,
            6,
        ];
        let query = 5..source.len();

        assert_eq!(
            active_placeholder_ranges(&source, IMAGE_TOKEN_ID, Some(query.clone())),
            vec![(5, 2)]
        );
        assert_eq!(
            active_placeholder_ranges(&source, AUDIO_TOKEN_ID, Some(query.clone())),
            vec![(8, 2)]
        );
        assert_eq!(
            active_placeholder_ranges(&source, VIDEO_TOKEN_ID, Some(query)),
            vec![(11, 2)]
        );
    }

    #[test]
    fn attention_mode_only_opens_vision_media() {
        assert_eq!(
            media_attention_policy(Gemma4BidirectionalAttention::Vision, MultimodalKind::Image),
            MultimodalAttentionPolicy::NonCausal
        );
        assert_eq!(
            media_attention_policy(Gemma4BidirectionalAttention::Vision, MultimodalKind::Video),
            MultimodalAttentionPolicy::NonCausal
        );
        assert_eq!(
            media_attention_policy(Gemma4BidirectionalAttention::Vision, MultimodalKind::Audio),
            MultimodalAttentionPolicy::Causal
        );
        assert_eq!(
            media_attention_policy(Gemma4BidirectionalAttention::Causal, MultimodalKind::Image),
            MultimodalAttentionPolicy::Causal
        );
        assert_eq!(
            media_attention_policy(Gemma4BidirectionalAttention::All, MultimodalKind::Image),
            MultimodalAttentionPolicy::Causal
        );
    }

    #[test]
    fn noncausal_media_must_fit_the_sliding_window() {
        let feature = MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset: 0,
            length: 5,
            attention_policy: MultimodalAttentionPolicy::NonCausal,
            splittable: false,
        };

        assert!(validate_sliding_noncausal_ranges([&feature], Some(5)).is_ok());
        assert!(validate_sliding_noncausal_ranges([&feature], Some(4)).is_err());
    }

    #[test]
    fn audio_batches_pad_frames_and_mask_the_tail() {
        let device = Device::Cpu;
        let mels = vec![
            Tensor::from_vec(vec![1f32, 2.], (1, 2, 1), &device).unwrap(),
            Tensor::from_vec(vec![3f32, 4., 5., 6.], (1, 4, 1), &device).unwrap(),
        ];
        let masks = vec![
            Tensor::from_vec(vec![0f32, 0.], (1, 2), &device).unwrap(),
            Tensor::from_vec(vec![0f32, 0., 0., 1.], (1, 4), &device).unwrap(),
        ];
        let (mels, masks) = cat_padded_audio(&mels, &masks).unwrap();

        assert_eq!(mels.dims(), &[2, 4, 1]);
        assert_eq!(
            masks.to_vec2::<f32>().unwrap(),
            vec![vec![0., 0., 1., 1.], vec![0., 0., 0., 1.]]
        );
    }

    #[test]
    fn spatial_batches_pad_across_sequence_boundaries() {
        let device = Device::Cpu;
        let tensors = vec![
            Tensor::zeros((1, 3, 2, 4), candle_core::DType::F32, &device).unwrap(),
            Tensor::zeros((1, 3, 4, 2), candle_core::DType::F32, &device).unwrap(),
        ];

        assert_eq!(cat_padded_spatial(&tensors).unwrap().dims(), &[2, 3, 4, 4]);
    }

    #[test]
    fn packed_layout_preserves_image_audio_and_video_ranges() {
        let features = vec![
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![11],
                offset: 1,
                length: 2,
                attention_policy: MultimodalAttentionPolicy::NonCausal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Audio,
                item_range: 0..1,
                hashes: vec![12],
                offset: 4,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Video,
                item_range: 0..1,
                hashes: vec![13],
                offset: 6,
                length: 2,
                attention_policy: MultimodalAttentionPolicy::NonCausal,
                splittable: false,
            },
        ];
        let layout = PackedMultimodalLayout::new(&[RequestMultimodalLayout {
            sequence_id: 0,
            query: 0..8,
            items: gemma4_layout_items_from_features(&features).unwrap(),
        }])
        .unwrap();
        let text = Tensor::from_vec(
            vec![0f32, 1., 2., 3., 4., 5., 6., 7.],
            (1, 8, 1),
            &Device::Cpu,
        )
        .unwrap();
        let outputs = HashMap::from([
            (
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash: 11,
                },
                vec![Tensor::from_vec(vec![21f32, 22.], (2, 1), &Device::Cpu).unwrap()],
            ),
            (
                MultimodalEncoderKey {
                    kind: MultimodalKind::Audio,
                    hash: 12,
                },
                vec![Tensor::from_vec(vec![24f32], (1, 1), &Device::Cpu).unwrap()],
            ),
            (
                MultimodalEncoderKey {
                    kind: MultimodalKind::Video,
                    hash: 13,
                },
                vec![Tensor::from_vec(vec![26f32, 27.], (2, 1), &Device::Cpu).unwrap()],
            ),
        ]);

        let result = layout
            .splice_embeddings(&text, &outputs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(result, vec![0., 21., 22., 3., 24., 5., 26., 27.]);
    }

    #[test]
    fn packed_layout_rejects_grouped_encoder_items() {
        let feature = MultiModalFeature {
            kind: MultimodalKind::Video,
            item_range: 0..2,
            hashes: vec![1, 2],
            offset: 0,
            length: 4,
            attention_policy: MultimodalAttentionPolicy::NonCausal,
            splittable: false,
        };

        assert!(gemma4_layout_items_from_features(&[feature]).is_err());
    }

    #[test]
    fn unified_patch_positions_preserve_row_major_media_order() {
        assert_eq!(
            unified_patch_positions(2, 3, 8).unwrap(),
            vec![0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, -1, -1, -1, -1]
        );
    }

    #[test]
    fn unified_patch_positions_reject_capacity_overflow() {
        assert!(unified_patch_positions(3, 3, 8).is_err());
    }

    #[test]
    fn unified_video_preprocessing_returns_matching_patches_and_positions() {
        let (pixels, positions, sizes) = unified_test_processor()
            .preprocess_unified_images_at_sizes(
                vec![DynamicImage::new_rgb8(8, 8)],
                &PreProcessorConfig::default(),
                &Device::Cpu,
                Some(&[(2, 4)]),
            )
            .unwrap();

        assert_eq!(pixels.dims(), &[1, 4, 12]);
        assert_eq!(positions.dims(), &[1, 4, 2]);
        assert_eq!(sizes, vec![(2, 4)]);
        assert_eq!(
            positions.flatten_all().unwrap().to_vec1::<i64>().unwrap(),
            vec![0, 0, 1, 0, -1, -1, -1, -1]
        );
    }

    #[test]
    fn cached_video_frames_are_filtered_per_sequence() {
        let ranges = vec![(0, 4), (8, 4), (16, 4), (24, 4)];
        let cached = vec![4, 0, 4, 2];
        assert_eq!(
            uncached_video_frame_mask(&ranges, &cached, 4).unwrap(),
            vec![false, true, false, true]
        );
    }

    #[test]
    fn cached_video_frame_filter_rejects_misaligned_metadata() {
        assert!(uncached_video_frame_mask(&[(0, 4)], &[], 1).is_err());
    }
}
