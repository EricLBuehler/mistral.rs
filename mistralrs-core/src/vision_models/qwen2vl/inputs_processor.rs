use crate::paged_attention::block_hash::{
    MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind,
};
use std::{
    any::Any,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Range,
    sync::Arc,
};

use anyhow::Result;
use candle_core::{Context, DType, Device, IndexOp, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{
    ApplyTensorTransforms, ApplyTransforms, Normalize, TensorTransforms, ToTensor, Transforms,
};
use tokenizers::Tokenizer;

use crate::{
    attention::AttentionMask,
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{build_mm_features_from_ranges, find_placeholder_delimited_ranges, Sequence},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        multimodal_layout::{
            gather_packed_mrope_positions, MropePositionSource, MultimodalEmbeddingMap,
            MultimodalEncoderKey, MultimodalItemLayout, PackedMultimodalLayout,
            RequestMultimodalLayout,
        },
        preprocessor_config::{PreProcessorConfig, ToFilter},
        ModelInputs,
    },
};

use super::Qwen2VLVisionSpecificArgs;

// Input processor
struct Qwen2VLImageProcessor {
    max_edge: Option<u32>,
}
// Processor
pub struct Qwen2VLProcessor {
    max_edge: Option<u32>,
}

impl Qwen2VLProcessor {
    pub const VISION_START: &str = "<|vision_start|>";
    pub const VISION_END: &str = "<|vision_end|>";
    pub const IMAGE_PAD: &str = "<|image_pad|>";
    pub const VIDEO_PAD: &str = "<|video_pad|>";
    pub const PLACEHOLDER: &str = "<|placeholder|>";

    pub fn new(max_edge: Option<u32>) -> Self {
        Self { max_edge }
    }
}

impl Processor for Qwen2VLProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Qwen2VLImageProcessor {
            max_edge: self.max_edge,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[Self::IMAGE_PAD, Self::VIDEO_PAD, Self::PLACEHOLDER]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

fn replace_first_occurrence(text: &str, to_replace: &str, replacement: &str) -> String {
    if let Some(pos) = text.find(to_replace) {
        let mut result = text.to_string();
        result.replace_range(pos..pos + to_replace.len(), replacement);
        result
    } else {
        text.to_string()
    }
}

pub(crate) fn expand_media_placeholders(
    text: &mut String,
    pad: &str,
    placeholder: &str,
    grid: Option<&Tensor>,
    media_rows: usize,
    merge_length: usize,
    kind: MultimodalKind,
) -> Result<()> {
    if merge_length == 0 {
        anyhow::bail!("Qwen merge length must be nonzero");
    }
    let placeholder_count = text.match_indices(pad).count();
    let grid_rows = grid.map(|grid| grid.dim(0)).transpose()?.unwrap_or(0);
    if placeholder_count != grid_rows || grid_rows != media_rows {
        anyhow::bail!(
            "Qwen {kind:?} has {placeholder_count} placeholders, {grid_rows} grid rows, and {media_rows} media rows"
        );
    }
    let mut repetitions = Vec::with_capacity(grid_rows);
    if let Some(grid) = grid {
        for index in 0..grid_rows {
            let row = grid.i(index)?.to_vec1::<u32>()?;
            if row.len() != 3 {
                anyhow::bail!("Qwen {kind:?} grid row must contain t, h, and w");
            }
            let patch_count = row.iter().try_fold(1usize, |product, &value| {
                product
                    .checked_mul(value as usize)
                    .ok_or_else(|| anyhow::Error::msg("Qwen media grid size overflow"))
            })?;
            if patch_count == 0 || patch_count % merge_length != 0 {
                anyhow::bail!(
                    "Qwen {kind:?} grid produces {patch_count} patches for merge length {merge_length}"
                );
            }
            repetitions.push(patch_count / merge_length);
        }
    }
    for repetition in repetitions {
        *text = replace_first_occurrence(text, pad, &placeholder.repeat(repetition));
    }
    *text = text.replace(placeholder, pad);
    Ok(())
}

pub(crate) fn validated_mm_features(
    ranges: &[(usize, usize)],
    hashes: &[u64],
    kind: MultimodalKind,
) -> Result<Vec<MultiModalFeature>> {
    if ranges.len() != hashes.len() {
        anyhow::bail!(
            "Qwen {kind:?} has {} placeholder ranges but {} hashes",
            ranges.len(),
            hashes.len()
        );
    }
    Ok(build_mm_features_from_ranges(ranges, hashes, kind))
}

fn find_sequences(nums: &[u32], needle: u32) -> Vec<(usize, usize)> {
    let mut sequences = Vec::new();
    let mut start = None;

    for (i, &num) in nums.iter().enumerate() {
        if num == needle {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            sequences.push((s, i));
            start = None;
        }
    }

    if let Some(s) = start {
        sequences.push((s, nums.len()));
    }

    sequences
}

pub(crate) fn video_hashes(seq: &Sequence) -> Vec<u64> {
    let hashes = seq
        .clone_videos()
        .unwrap_or_default()
        .iter()
        .map(|video| {
            let mut hasher = DefaultHasher::new();
            video.frame_hashes().hash(&mut hasher);
            hasher.finish()
        })
        .collect::<Vec<_>>();
    if !seq.is_chunked_prefill_view() {
        return hashes;
    }
    seq.active_local_multimodal_item_range(MultimodalKind::Video, hashes.len())
        .and_then(|range| hashes.get(range))
        .unwrap_or_default()
        .to_vec()
}

fn grid_patch_count(grid: Option<&Tensor>) -> Result<usize> {
    Ok(grid
        .map(Tensor::to_vec2::<u32>)
        .transpose()?
        .unwrap_or_default()
        .iter()
        .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
        .sum())
}

pub(crate) fn split_media_pixels(
    pixel_values: &Tensor,
    image_grid_thw: Option<&Tensor>,
    video_grid_thw: Option<&Tensor>,
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let image_patches = grid_patch_count(image_grid_thw)?;
    let video_patches = grid_patch_count(video_grid_thw)?;
    if pixel_values.dim(0)? != image_patches + video_patches {
        anyhow::bail!(
            "Qwen media pixel rows {} do not match image/video grids {}",
            pixel_values.dim(0)?,
            image_patches + video_patches
        );
    }
    let images = (image_patches != 0)
        .then(|| pixel_values.narrow(0, 0, image_patches))
        .transpose()?;
    let videos = (video_patches != 0)
        .then(|| pixel_values.narrow(0, image_patches, video_patches))
        .transpose()?;
    Ok((images, videos))
}

pub(crate) fn select_media_view(
    seq: &Sequence,
    kind: MultimodalKind,
    pixel_values: Option<Tensor>,
    grid_thw: Option<Tensor>,
) -> Result<(Option<Tensor>, Option<Tensor>, usize)> {
    let Some(grid_thw) = grid_thw else {
        if pixel_values.is_some() {
            anyhow::bail!("Qwen media pixels are missing grid metadata");
        }
        return Ok((None, None, 0));
    };
    let item_count = grid_thw.dim(0)?;
    let range = if seq.is_chunked_prefill_view() {
        seq.active_local_multimodal_item_range(kind, item_count)
            .unwrap_or(0..0)
    } else {
        0..item_count
    };
    if range.end > item_count {
        anyhow::bail!(
            "Qwen {:?} media window {:?} exceeds {} grid rows",
            kind,
            range,
            item_count
        );
    }
    let grid_data = grid_thw.to_vec2::<u32>()?;
    let patch_start = grid_data[..range.start]
        .iter()
        .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
        .sum::<usize>();
    let patch_count = grid_data[range.clone()]
        .iter()
        .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
        .sum::<usize>();
    let pixel_values = match (pixel_values, patch_count) {
        (Some(pixel_values), 0) => {
            if pixel_values.dim(0)? < patch_start {
                anyhow::bail!("Qwen media pixel rows do not cover the selected window");
            }
            None
        }
        (Some(pixel_values), patch_count) => {
            Some(pixel_values.narrow(0, patch_start, patch_count)?)
        }
        (None, 0) => None,
        (None, _) => anyhow::bail!("Qwen media grid is missing pixel rows"),
    };
    let selected_count = range.len();
    let grid_thw = (selected_count != 0)
        .then(|| grid_thw.narrow(0, range.start, selected_count))
        .transpose()?;
    Ok((pixel_values, grid_thw, selected_count))
}

pub(crate) fn shift_media_spans(
    spans: &mut Vec<(usize, usize)>,
    prefix_len: usize,
) -> Result<usize> {
    if prefix_len == 0 {
        return Ok(0);
    }
    let mut cached = 0usize;
    for &(start, end) in spans.iter() {
        if end <= prefix_len {
            cached += 1;
        } else if start < prefix_len {
            anyhow::bail!("Qwen prefix cache splits a multimodal item");
        }
    }
    spans.retain(|(_, end)| *end > prefix_len);
    for (start, end) in spans {
        *start -= prefix_len;
        *end -= prefix_len;
    }
    Ok(cached)
}

pub(crate) fn media_data_cached_offset(seq: &Sequence, cached_items: usize) -> usize {
    if seq.is_chunked_prefill_view() {
        0
    } else {
        cached_items
    }
}

pub(crate) fn select_media_batch(
    mut pixel_values: Option<Tensor>,
    mut grid_thw: Option<Tensor>,
    item_counts: &[usize],
    cached_items: &[usize],
    current_items: &[usize],
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    if item_counts.len() != cached_items.len() || item_counts.len() != current_items.len() {
        anyhow::bail!("Qwen per-sequence media metadata length mismatch");
    }
    let Some(grid) = grid_thw.as_ref() else {
        if item_counts.iter().sum::<usize>() != 0 || pixel_values.is_some() {
            anyhow::bail!("Qwen media selection is missing grid metadata");
        }
        return Ok((None, None));
    };
    if grid.dim(0)? != item_counts.iter().sum::<usize>() {
        anyhow::bail!("Qwen media grid rows do not match per-sequence item counts");
    }
    let grid_data = grid.to_vec2::<u32>()?;
    let mut selected_grids = Vec::new();
    let mut selected_pixels = Vec::new();
    let mut grid_offset = 0usize;
    let mut pixel_offset = 0usize;
    for ((&total, &cached), &current) in item_counts.iter().zip(cached_items).zip(current_items) {
        let end = cached
            .checked_add(current)
            .ok_or_else(|| anyhow::Error::msg("Qwen media item range overflow"))?;
        if end > total {
            anyhow::bail!(
                "Qwen media view requests items {cached}..{end} from a sequence with {total}"
            );
        }
        let start_row = grid_offset + cached;
        let patch_start = pixel_offset
            + grid_data[grid_offset..start_row]
                .iter()
                .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
                .sum::<usize>();
        let patch_count = grid_data[start_row..start_row + current]
            .iter()
            .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
            .sum::<usize>();
        if current != 0 {
            selected_grids.push(grid.narrow(0, start_row, current)?);
        }
        if patch_count != 0 {
            let pixels = pixel_values
                .as_ref()
                .ok_or_else(|| anyhow::Error::msg("Qwen media grid is missing pixel rows"))?;
            selected_pixels.push(pixels.narrow(0, patch_start, patch_count)?);
        }
        pixel_offset += grid_data[grid_offset..grid_offset + total]
            .iter()
            .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
            .sum::<usize>();
        grid_offset += total;
    }
    grid_thw = (!selected_grids.is_empty())
        .then(|| Tensor::cat(&selected_grids, 0))
        .transpose()?;
    pixel_values = (!selected_pixels.is_empty())
        .then(|| Tensor::cat(&selected_pixels, 0))
        .transpose()?;
    Ok((pixel_values, grid_thw))
}

fn completed_media_grid(
    seq: &Sequence,
    kind: MultimodalKind,
    grid: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    let Some(grid) = grid else {
        return Ok(None);
    };
    if !seq.is_chunked_prefill_view() {
        return Ok(Some(grid.clone()));
    }
    let token_count = seq.prompt_position_source_toks().len();
    let item_count = seq
        .mm_features()
        .iter()
        .filter(|feature| feature.kind == kind && feature.end() <= token_count)
        .map(|feature| feature.item_range.len())
        .sum::<usize>();
    if item_count > grid.dim(0)? {
        anyhow::bail!("Qwen completed media items exceed the stored MRoPE grid");
    }
    (item_count != 0)
        .then(|| grid.narrow(0, 0, item_count))
        .transpose()
        .map_err(Into::into)
}

pub(crate) struct PromptMropeConfig<'a> {
    packed: bool,
    padded_len: usize,
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    device: &'a Device,
}

impl<'a> PromptMropeConfig<'a> {
    pub(crate) fn new(
        packed: bool,
        padded_len: usize,
        spatial_merge_size: usize,
        image_token_id: u32,
        video_token_id: u32,
        device: &'a Device,
    ) -> Self {
        Self {
            packed,
            padded_len,
            spatial_merge_size,
            image_token_id,
            video_token_id,
            device,
        }
    }
}

pub(crate) fn prompt_mrope(
    input_seqs: &[&mut Sequence],
    query_ranges: &[Range<usize>],
    config: PromptMropeConfig<'_>,
) -> Result<Tensor> {
    let PromptMropeConfig {
        packed,
        padded_len,
        spatial_merge_size,
        image_token_id,
        video_token_id,
        device,
    } = config;
    if input_seqs.len() != query_ranges.len() {
        anyhow::bail!("Qwen MRoPE query count does not match sequence count");
    }
    let mut sources = Vec::with_capacity(input_seqs.len());
    for seq in input_seqs {
        let image_grid = completed_media_grid(
            seq,
            MultimodalKind::Image,
            seq.multimodal.rope_img_grid_thw.as_ref(),
        )?;
        let video_grid = completed_media_grid(
            seq,
            MultimodalKind::Video,
            seq.multimodal.rope_vid_grid_thw.as_ref(),
        )?;
        let full_ids = Tensor::new(seq.prompt_position_source_toks(), device)?.unsqueeze(0)?;
        let (positions, deltas) = super::Qwen2VLModel::compute_rope_index(
            &full_ids,
            image_grid.as_ref(),
            video_grid.as_ref(),
            &AttentionMask::None,
            spatial_merge_size,
            image_token_id,
            video_token_id,
        )?;
        sources.push(MropePositionSource {
            position_ids: positions,
            delta: deltas.flatten_all()?.to_vec1::<i64>()?[0],
        });
    }
    if packed {
        return Ok(gather_packed_mrope_positions(
            &sources,
            query_ranges,
            device,
        )?);
    }

    let mut rows = Vec::with_capacity(sources.len());
    for (source, query) in sources.iter().zip(query_ranges) {
        if query.end > source.position_ids.dim(2)? {
            anyhow::bail!("Qwen MRoPE query range exceeds the sequence position source");
        }
        let positions = source
            .position_ids
            .i((.., 0, query.clone()))?
            .to_dtype(DType::I64)?;
        if positions.dim(1)? > padded_len {
            anyhow::bail!("Qwen MRoPE query is longer than the padded input");
        }
        let padding = padded_len - positions.dim(1)?;
        let positions = if padding == 0 {
            positions
        } else {
            Tensor::cat(
                &[positions, Tensor::ones((3, padding), DType::I64, device)?],
                1,
            )?
        };
        rows.push(positions);
    }
    Ok(Tensor::stack(&rows, 1)?)
}

pub(crate) fn packed_layout(
    input_seqs: &[&mut Sequence],
    query_lens: &[usize],
    continuous_img_pad: &[Vec<(usize, usize)>],
    continuous_vid_pad: &[Vec<(usize, usize)>],
) -> Result<PackedMultimodalLayout> {
    if input_seqs.len() != query_lens.len()
        || input_seqs.len() != continuous_img_pad.len()
        || input_seqs.len() != continuous_vid_pad.len()
    {
        anyhow::bail!("Qwen packed multimodal metadata length mismatch");
    }
    let mut requests = Vec::with_capacity(input_seqs.len());
    for (((seq, &query_len), image_spans), video_spans) in input_seqs
        .iter()
        .zip(query_lens)
        .zip(continuous_img_pad)
        .zip(continuous_vid_pad)
    {
        if query_len != seq.get_toks().len() {
            anyhow::bail!("Qwen packed multimodal prefill requires the complete prompt");
        }
        let image_hashes = seq.image_hashes().unwrap_or_default();
        if image_hashes.len() != image_spans.len() {
            anyhow::bail!(
                "Qwen sequence has {} image hashes but {} image spans",
                image_hashes.len(),
                image_spans.len()
            );
        }
        let video_hashes = video_hashes(seq);
        if video_hashes.len() != video_spans.len() {
            anyhow::bail!(
                "Qwen sequence has {} video hashes but {} video spans",
                video_hashes.len(),
                video_spans.len()
            );
        }
        let mut items = Vec::with_capacity(image_spans.len() + video_spans.len());
        for (item_index, (&hash, &(start, end))) in image_hashes.iter().zip(image_spans).enumerate()
        {
            items.push(MultimodalItemLayout::new(
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash,
                },
                item_index,
                start..end,
                MultimodalAttentionPolicy::Causal,
                vec![MultimodalEmbeddingMap::contiguous(start..end, 0, 0)?],
            )?);
        }
        for (item_index, (&hash, &(start, end))) in video_hashes.iter().zip(video_spans).enumerate()
        {
            items.push(MultimodalItemLayout::new(
                MultimodalEncoderKey {
                    kind: MultimodalKind::Video,
                    hash,
                },
                item_index,
                start..end,
                MultimodalAttentionPolicy::Causal,
                vec![MultimodalEmbeddingMap::contiguous(start..end, 0, 0)?],
            )?);
        }
        requests.push(RequestMultimodalLayout {
            sequence_id: *seq.id(),
            query: Range {
                start: 0,
                end: query_len,
            },
            items,
        });
    }
    Ok(PackedMultimodalLayout::new(&requests)?)
}

impl InputsProcessor for Qwen2VLImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        device: &Device,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        let Some(tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "Qwen2VLImageProcessor requires a specified tokenizer.",
            ));
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        if !input_seqs
            .iter()
            .any(|seq| seq.has_images() || seq.has_videos())
        {
            return Ok(());
        }

        let mut detok_seqs = tokenizer
            .decode_batch(
                &input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                false,
            )
            .expect("Detokenization failed!");
        let mut image_grid_thw_accum = Vec::new();
        let mut video_grid_thw_accum = Vec::new();
        for seq in input_seqs.iter_mut() {
            if !seq.has_images() && !seq.has_videos() {
                image_grid_thw_accum.push(None);
                video_grid_thw_accum.push(None);
                continue;
            }
            let (_, image_grid_thw, video_grid_thw) =
                if let Some(cached_pixel_values) = &seq.multimodal.cached_pixel_values {
                    (
                        cached_pixel_values.clone(),
                        seq.multimodal.cached_img_thw.clone(),
                        seq.multimodal.cached_vid_thw.clone(),
                    )
                } else {
                    let image = if seq.has_images() {
                        Some(self.preprocess(
                            seq.clone_images().unwrap_or_default(),
                            vec![],
                            config,
                            device,
                            (usize::MAX, usize::MAX),
                        )?)
                    } else {
                        None
                    };
                    let video = if seq.has_videos() {
                        Some(
                            self.preprocess(
                                vec![],
                                seq.clone_videos()
                                    .unwrap_or_default()
                                    .into_iter()
                                    .map(|video| video.frames)
                                    .collect(),
                                config,
                                device,
                                (usize::MAX, usize::MAX),
                            )?,
                        )
                    } else {
                        None
                    };
                    let image_grid_thw = image
                        .as_ref()
                        .and_then(|processed| processed.image_grid_thw.clone());
                    let video_grid_thw = video
                        .as_ref()
                        .and_then(|processed| processed.video_grid_thw.clone());
                    let mut pixels = Vec::new();
                    if let Some(image) = image {
                        pixels.push(image.pixel_values);
                    }
                    if let Some(video) = video {
                        pixels.push(video.pixel_values);
                    }
                    let pixel_values = Tensor::cat(&pixels, 0)?;
                    seq.multimodal.cached_pixel_values = Some(pixel_values.clone());
                    seq.multimodal.cached_img_thw = image_grid_thw.clone();
                    seq.multimodal.cached_vid_thw = video_grid_thw.clone();
                    (pixel_values, image_grid_thw, video_grid_thw)
                };
            image_grid_thw_accum.push(image_grid_thw);
            video_grid_thw_accum.push(video_grid_thw);
        }

        for (idx, seq) in input_seqs.iter_mut().enumerate() {
            if seq.multimodal.rope_img_grid_thw.is_none() {
                seq.multimodal.rope_img_grid_thw = image_grid_thw_accum[idx].clone();
            }
            if seq.multimodal.rope_vid_grid_thw.is_none() {
                seq.multimodal.rope_vid_grid_thw = video_grid_thw_accum[idx].clone();
            }
        }

        let merge_length = config.merge_size.expect("Require `merge_size").pow(2);
        for (((text, seq), image_grid), video_grid) in detok_seqs
            .iter_mut()
            .zip(input_seqs.iter())
            .zip(&image_grid_thw_accum)
            .zip(&video_grid_thw_accum)
        {
            if seq.multimodal.has_changed_prompt {
                continue;
            }
            let image_rows = seq.clone_images().unwrap_or_default().len();
            let image_hashes = seq.image_hashes().unwrap_or_default();
            if image_hashes.len() != image_rows {
                anyhow::bail!(
                    "Qwen has {image_rows} image rows but {} image hashes",
                    image_hashes.len()
                );
            }
            let video_rows = seq.clone_videos().unwrap_or_default().len();
            let video_hashes = video_hashes(seq);
            if video_hashes.len() != video_rows {
                anyhow::bail!(
                    "Qwen has {video_rows} video rows but {} video hashes",
                    video_hashes.len()
                );
            }
            expand_media_placeholders(
                text,
                Qwen2VLProcessor::IMAGE_PAD,
                Qwen2VLProcessor::PLACEHOLDER,
                image_grid.as_ref(),
                image_rows,
                merge_length,
                MultimodalKind::Image,
            )?;
            expand_media_placeholders(
                text,
                Qwen2VLProcessor::VIDEO_PAD,
                Qwen2VLProcessor::PLACEHOLDER,
                video_grid.as_ref(),
                video_rows,
                merge_length,
                MultimodalKind::Video,
            )?;
        }

        for (detok, seq) in detok_seqs.into_iter().zip(input_seqs.iter_mut()) {
            if seq.multimodal.has_changed_prompt {
                continue;
            }
            let toks = tokenizer
                .encode_fast(detok.clone(), false)
                .expect("Detokenization failed!");
            let ids = toks.get_ids().to_vec();
            seq.set_initial_prompt(detok);

            if seq.mm_features().is_empty() {
                let mut features = Vec::new();
                let start_id = tokenizer
                    .token_to_id(Qwen2VLProcessor::VISION_START)
                    .context("Qwen tokenizer is missing vision start token")?;
                let end_id = tokenizer
                    .token_to_id(Qwen2VLProcessor::VISION_END)
                    .context("Qwen tokenizer is missing vision end token")?;
                let img_pad_id = tokenizer
                    .token_to_id(Qwen2VLProcessor::IMAGE_PAD)
                    .context("Qwen tokenizer is missing image pad token")?;
                let image_ranges =
                    find_placeholder_delimited_ranges(&ids, img_pad_id, start_id, end_id);
                features.extend(validated_mm_features(
                    &image_ranges,
                    seq.image_hashes().unwrap_or_default(),
                    MultimodalKind::Image,
                )?);
                let vid_pad_id = tokenizer
                    .token_to_id(Qwen2VLProcessor::VIDEO_PAD)
                    .context("Qwen tokenizer is missing video pad token")?;
                let video_ranges =
                    find_placeholder_delimited_ranges(&ids, vid_pad_id, start_id, end_id);
                let hashes = video_hashes(seq);
                features.extend(validated_mm_features(
                    &video_ranges,
                    &hashes,
                    MultimodalKind::Video,
                )?);
                if !features.is_empty() {
                    seq.set_mm_features(features);
                }
            }

            seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_deref_mut());
            seq.multimodal.has_changed_prompt = true;
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
    ) -> Result<InputProcessorOutput> {
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
                "MLlamaInputProcessor requires a specified tokenizer.",
            ));
        };

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
            )
            .unwrap()
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
            )
            .unwrap()
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_media = input_seqs
            .iter()
            .any(|seq| seq.has_images() || seq.has_videos());
        let mut image_item_counts = vec![0usize; input_seqs.len()];
        let mut video_item_counts = vec![0usize; input_seqs.len()];

        let (
            new_input,
            pixel_values,
            pixel_values_videos,
            mut image_grid_thw,
            mut video_grid_thw,
            mut continuous_img_pad,
            mut continuous_vid_pad,
        ) = if has_media {
            let mut image_pixel_values_accum = Vec::new();
            let mut video_pixel_values_accum = Vec::new();
            let mut image_grid_thw_accum = Vec::new();
            let mut video_grid_thw_accum = Vec::new();

            let mut detok_seqs = tokenizer
                .decode_batch(
                    &input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    false,
                )
                .expect("Detokenization failed!");

            for (seq_idx, seq) in input_seqs.iter_mut().enumerate() {
                if !seq.has_images() && !seq.has_videos() {
                    image_grid_thw_accum.push(None);
                    video_grid_thw_accum.push(None);
                    continue;
                }
                let (pixel_values, image_grid_thw, video_grid_thw) =
                    if let Some(cached_pixel_values) = &seq.multimodal.cached_pixel_values {
                        (
                            cached_pixel_values.clone(),
                            seq.multimodal.cached_img_thw.clone(),
                            seq.multimodal.cached_vid_thw.clone(),
                        )
                    } else {
                        let image = if seq.has_images() {
                            Some(self.preprocess(
                                seq.clone_images().unwrap_or_default(),
                                vec![],
                                config,
                                device,
                                (usize::MAX, usize::MAX),
                            )?)
                        } else {
                            None
                        };
                        let video = if seq.has_videos() {
                            Some(
                                self.preprocess(
                                    vec![],
                                    seq.clone_videos()
                                        .unwrap_or_default()
                                        .into_iter()
                                        .map(|video| video.frames)
                                        .collect(),
                                    config,
                                    device,
                                    (usize::MAX, usize::MAX),
                                )?,
                            )
                        } else {
                            None
                        };
                        let image_grid_thw = image
                            .as_ref()
                            .and_then(|processed| processed.image_grid_thw.clone());
                        let video_grid_thw = video
                            .as_ref()
                            .and_then(|processed| processed.video_grid_thw.clone());
                        let mut pixels = Vec::new();
                        if let Some(image) = image {
                            pixels.push(image.pixel_values);
                        }
                        if let Some(video) = video {
                            pixels.push(video.pixel_values);
                        }
                        let pixel_values = Tensor::cat(&pixels, 0)?;
                        seq.multimodal.cached_pixel_values = Some(pixel_values.clone());
                        seq.multimodal.cached_img_thw = image_grid_thw.clone();
                        seq.multimodal.cached_vid_thw = video_grid_thw.clone();
                        (pixel_values, image_grid_thw, video_grid_thw)
                    };

                if seq.multimodal.rope_img_grid_thw.is_none() {
                    seq.multimodal.rope_img_grid_thw = image_grid_thw.clone();
                }
                if seq.multimodal.rope_vid_grid_thw.is_none() {
                    seq.multimodal.rope_vid_grid_thw = video_grid_thw.clone();
                }
                let (image_pixels, video_pixels) = split_media_pixels(
                    &pixel_values,
                    image_grid_thw.as_ref(),
                    video_grid_thw.as_ref(),
                )?;
                let (image_pixels, image_grid_thw, image_count) =
                    select_media_view(seq, MultimodalKind::Image, image_pixels, image_grid_thw)?;
                let (video_pixels, video_grid_thw, video_count) =
                    select_media_view(seq, MultimodalKind::Video, video_pixels, video_grid_thw)?;
                image_item_counts[seq_idx] = image_count;
                video_item_counts[seq_idx] = video_count;
                if let Some(image_pixels) = image_pixels {
                    image_pixel_values_accum.push(image_pixels);
                }
                if let Some(video_pixels) = video_pixels {
                    video_pixel_values_accum.push(video_pixels);
                }
                image_grid_thw_accum.push(image_grid_thw);
                video_grid_thw_accum.push(video_grid_thw);
            }

            if is_prompt {
                let merge_length = config.merge_size.expect("Require `merge_size").pow(2);
                for (seq_idx, (((text, seq), image_grid), video_grid)) in detok_seqs
                    .iter_mut()
                    .zip(input_seqs.iter_mut())
                    .zip(&image_grid_thw_accum)
                    .zip(&video_grid_thw_accum)
                    .enumerate()
                {
                    if seq.multimodal.has_changed_prompt {
                        continue;
                    }
                    let image_rows = image_item_counts[seq_idx];
                    if seq.image_hashes().unwrap_or_default().len() != image_rows {
                        anyhow::bail!(
                            "Qwen has {image_rows} selected image rows but {} image hashes",
                            seq.image_hashes().unwrap_or_default().len()
                        );
                    }
                    let video_rows = video_item_counts[seq_idx];
                    let hashes = video_hashes(seq);
                    if hashes.len() != video_rows {
                        anyhow::bail!(
                            "Qwen has {video_rows} selected video rows but {} video hashes",
                            hashes.len()
                        );
                    }
                    expand_media_placeholders(
                        text,
                        Qwen2VLProcessor::IMAGE_PAD,
                        Qwen2VLProcessor::PLACEHOLDER,
                        image_grid.as_ref(),
                        image_rows,
                        merge_length,
                        MultimodalKind::Image,
                    )?;
                    expand_media_placeholders(
                        text,
                        Qwen2VLProcessor::VIDEO_PAD,
                        Qwen2VLProcessor::PLACEHOLDER,
                        video_grid.as_ref(),
                        video_rows,
                        merge_length,
                        MultimodalKind::Video,
                    )?;
                }
            }

            let mut all_ids = Vec::new();
            let mut all_continuous_img_pad = Vec::new();
            let mut all_continuous_vid_pad = Vec::new();
            for (detok, seq) in detok_seqs.into_iter().zip(input_seqs.iter_mut()) {
                let toks = tokenizer
                    .encode_fast(detok.clone(), false)
                    .expect("Detokenization failed!");
                let ids = toks.get_ids().to_vec();

                if !seq.multimodal.has_changed_prompt {
                    seq.set_initial_prompt(detok.clone());

                    let mut features = Vec::new();
                    if seq.mm_features().is_empty() {
                        let start_id = tokenizer
                            .token_to_id(Qwen2VLProcessor::VISION_START)
                            .context("Qwen tokenizer is missing vision start token")?;
                        let end_id = tokenizer
                            .token_to_id(Qwen2VLProcessor::VISION_END)
                            .context("Qwen tokenizer is missing vision end token")?;
                        let img_pad_id = tokenizer
                            .token_to_id(Qwen2VLProcessor::IMAGE_PAD)
                            .context("Qwen tokenizer is missing image pad token")?;
                        let image_ranges =
                            find_placeholder_delimited_ranges(&ids, img_pad_id, start_id, end_id);
                        features.extend(validated_mm_features(
                            &image_ranges,
                            seq.image_hashes().unwrap_or_default(),
                            MultimodalKind::Image,
                        )?);
                        let vid_pad_id = tokenizer
                            .token_to_id(Qwen2VLProcessor::VIDEO_PAD)
                            .context("Qwen tokenizer is missing video pad token")?;
                        let video_ranges =
                            find_placeholder_delimited_ranges(&ids, vid_pad_id, start_id, end_id);
                        let hashes = video_hashes(seq);
                        features.extend(validated_mm_features(
                            &video_ranges,
                            &hashes,
                            MultimodalKind::Video,
                        )?);
                        if !features.is_empty() {
                            seq.set_mm_features(features);
                        }
                    }

                    seq.set_toks_and_reallocate(ids.clone(), paged_attn_metadata.as_mut());
                    seq.multimodal.has_changed_prompt = true;
                }
                all_ids.push(ids.clone());

                let img_pad = tokenizer
                    .token_to_id(Qwen2VLProcessor::IMAGE_PAD)
                    .context("Qwen tokenizer is missing image pad token")?;
                let continuous_img_pad = find_sequences(&ids, img_pad);
                all_continuous_img_pad.push(continuous_img_pad);

                let vid_pad = tokenizer
                    .token_to_id(Qwen2VLProcessor::VIDEO_PAD)
                    .context("Qwen tokenizer is missing video pad token")?;
                let continuous_vid_pad = find_sequences(&ids, vid_pad);
                all_continuous_vid_pad.push(continuous_vid_pad);
            }

            let mut all_ids_new = Vec::new();
            let max_len = all_ids.iter().map(|ids| ids.len()).max().unwrap();
            for ids in all_ids {
                let pad = max_len - ids.len();
                all_ids_new
                    .push(Tensor::new([ids, vec![0; pad]].concat(), input.device()).unwrap());
            }

            (
                Some(Tensor::stack(&all_ids_new, 0).unwrap()),
                (!image_pixel_values_accum.is_empty())
                    .then(|| Tensor::cat(&image_pixel_values_accum, 0))
                    .transpose()?,
                (!video_pixel_values_accum.is_empty())
                    .then(|| Tensor::cat(&video_pixel_values_accum, 0))
                    .transpose()?,
                {
                    let grids = image_grid_thw_accum
                        .iter()
                        .filter_map(Clone::clone)
                        .collect::<Vec<_>>();
                    (!grids.is_empty())
                        .then(|| Tensor::cat(&grids, 0))
                        .transpose()?
                },
                {
                    let grids = video_grid_thw_accum
                        .iter()
                        .filter_map(Clone::clone)
                        .collect::<Vec<_>>();
                    (!grids.is_empty())
                        .then(|| Tensor::cat(&grids, 0))
                        .transpose()?
                },
                all_continuous_img_pad,
                all_continuous_vid_pad,
            )
        } else {
            (
                None,
                None,
                None,
                None,
                None,
                vec![vec![]; input_seqs.len()],
                vec![vec![]; input_seqs.len()],
            )
        };

        let full_input_from_seq = if is_prompt && flash_meta.packed {
            let max_len = input_seqs
                .iter()
                .map(|seq| seq.get_toks().len())
                .max()
                .unwrap_or(0);
            let rows = input_seqs
                .iter()
                .map(|seq| {
                    let mut ids = seq.get_toks().to_vec();
                    ids.resize(max_len, 0);
                    Tensor::new(ids, device)
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
            Some(Tensor::stack(&rows, 0)?)
        } else {
            None
        };
        let (input, input_ids_full) = match (new_input, is_prompt) {
            (Some(new_input), true) => (input, new_input),
            (Some(new_input), false) => (input, new_input),
            (None, _) => (
                input.clone(),
                full_input_from_seq.unwrap_or_else(|| input.clone()),
            ),
        };

        let mut pixel_values = if is_prompt { pixel_values } else { None };
        let mut pixel_values_videos = if is_prompt { pixel_values_videos } else { None };

        let mut per_seq_cached_images: Vec<usize> = vec![0; input_seqs.len()];
        let mut per_seq_current_images: Vec<usize> = vec![0; input_seqs.len()];
        let mut per_seq_cached_videos: Vec<usize> = vec![0; input_seqs.len()];
        let mut per_seq_current_videos: Vec<usize> = vec![0; input_seqs.len()];
        if is_prompt {
            for (seq_idx, (seq, (img_pads, vid_pads))) in input_seqs
                .iter()
                .zip(
                    continuous_img_pad
                        .iter_mut()
                        .zip(continuous_vid_pad.iter_mut()),
                )
                .enumerate()
            {
                let local_prefix = seq
                    .active_prompt_local_query_range()
                    .map_or(seq.prefix_cache_len(), |query| query.start);
                let cached_images = shift_media_spans(img_pads, local_prefix)?;
                let cached_videos = shift_media_spans(vid_pads, local_prefix)?;
                per_seq_cached_images[seq_idx] = media_data_cached_offset(seq, cached_images);
                per_seq_cached_videos[seq_idx] = media_data_cached_offset(seq, cached_videos);
                per_seq_current_images[seq_idx] = img_pads.len();
                per_seq_current_videos[seq_idx] = vid_pads.len();
            }
            (pixel_values, image_grid_thw) = select_media_batch(
                pixel_values,
                image_grid_thw,
                &image_item_counts,
                &per_seq_cached_images,
                &per_seq_current_images,
            )?;
            (pixel_values_videos, video_grid_thw) = select_media_batch(
                pixel_values_videos,
                video_grid_thw,
                &video_item_counts,
                &per_seq_cached_videos,
                &per_seq_current_videos,
            )?;
        }

        let seqlens = input_seqs.iter().map(|seq| seq.len()).collect::<Vec<_>>();

        let rope_img_grid_thw = {
            let grids: Vec<_> = input_seqs
                .iter()
                .filter_map(|seq| seq.multimodal.rope_img_grid_thw.clone())
                .collect();
            if grids.is_empty() {
                None
            } else {
                Some(Tensor::cat(&grids, 0).unwrap())
            }
        };
        let rope_vid_grid_thw = {
            let grids: Vec<_> = input_seqs
                .iter()
                .filter_map(|seq| seq.multimodal.rope_vid_grid_thw.clone())
                .collect();
            if grids.is_empty() {
                None
            } else {
                Some(Tensor::cat(&grids, 0).unwrap())
            }
        };

        let mut image_hashes = Vec::new();
        let mut selected_video_hashes = Vec::new();
        if is_prompt {
            for (seq_idx, seq) in input_seqs.iter().enumerate() {
                let hashes = seq.image_hashes().unwrap_or_default();
                let cached = per_seq_cached_images[seq_idx];
                let current = per_seq_current_images[seq_idx];
                let selected = hashes.get(cached..cached + current).ok_or_else(|| {
                    anyhow::Error::msg("Qwen image hashes do not cover the selected media window")
                })?;
                image_hashes.extend_from_slice(selected);

                let hashes = video_hashes(seq);
                let cached = per_seq_cached_videos[seq_idx];
                let current = per_seq_current_videos[seq_idx];
                let selected = hashes.get(cached..cached + current).ok_or_else(|| {
                    anyhow::Error::msg("Qwen video hashes do not cover the selected media window")
                })?;
                selected_video_hashes.extend_from_slice(selected);
            }
        }
        let packed_layout = if is_prompt && flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| anyhow::Error::msg("packed Qwen prefill requires query lengths"))?;
            let layout = packed_layout(
                input_seqs,
                query_lens,
                &continuous_img_pad,
                &continuous_vid_pad,
            )?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "Qwen packed layout has {} tokens but input has {}",
                    layout.token_count(),
                    input.dim(1)?
                );
            }
            Some(layout)
        } else {
            None
        };
        let needs_prompt_mrope = is_prompt
            && input_seqs.iter().any(|seq| {
                seq.multimodal.rope_img_grid_thw.is_some()
                    || seq.multimodal.rope_vid_grid_thw.is_some()
            });
        let prompt_position_ids = if needs_prompt_mrope {
            let image_token_id = tokenizer
                .token_to_id(Qwen2VLProcessor::IMAGE_PAD)
                .ok_or_else(|| anyhow::Error::msg("Qwen tokenizer is missing image pad token"))?;
            let video_token_id = tokenizer
                .token_to_id(Qwen2VLProcessor::VIDEO_PAD)
                .ok_or_else(|| anyhow::Error::msg("Qwen tokenizer is missing video pad token"))?;
            let query_ranges = input_seqs
                .iter()
                .map(|seq| {
                    seq.active_prompt_query_range()
                        .unwrap_or(0..seq.prompt_position_source_toks().len())
                })
                .collect::<Vec<_>>();
            Some(prompt_mrope(
                input_seqs,
                &query_ranges,
                PromptMropeConfig::new(
                    flash_meta.packed,
                    input.dim(1)?,
                    config.merge_size.context("Qwen requires merge_size")?,
                    image_token_id,
                    video_token_id,
                    device,
                ),
            )?)
        } else {
            None
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(Qwen2VLVisionSpecificArgs {
                input_ids_full,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
                rope_img_grid_thw,
                rope_vid_grid_thw,
                seqlens,
                continuous_img_pad,
                continuous_vid_pad,
                image_hashes,
                video_hashes: selected_video_hashes,
                packed_layout,
                prompt_position_ids,
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

impl Qwen2VLImageProcessor {
    fn smart_resize(
        &self,
        height: usize,
        width: usize,
        factor: usize,
        min_pixels: usize,
        max_pixels: usize,
    ) -> candle_core::Result<(usize, usize)> {
        if height < factor || width < factor {
            candle_core::bail!(
                "height:{} or width:{} must be larger than factor:{}",
                height,
                width,
                factor
            );
        } else if (height.max(width) as f64 / height.min(width) as f64) > 200.0 {
            candle_core::bail!(
                "absolute aspect ratio must be smaller than 200, got {:.2}",
                height.max(width) as f64 / height.min(width) as f64
            );
        }

        let mut h_bar = (height as f64 / factor as f64).round() as usize * factor;
        let mut w_bar = (width as f64 / factor as f64).round() as usize * factor;

        if h_bar * w_bar > max_pixels {
            let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize) * factor;
        } else if h_bar * w_bar < min_pixels {
            let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }

        Ok((h_bar, w_bar))
    }

    // patches and t,h,w
    fn preprocess_inner(
        &self,
        images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
        (mut height, mut width): (u32, u32),
    ) -> candle_core::Result<(Tensor, (u32, u32, u32))> {
        let mut processed_images = Vec::new();

        for mut image in images {
            image = image.resize_exact(
                width,
                height,
                config
                    .resampling
                    .map(|resample| Some(resample).to_filter())
                    .unwrap_or(Ok(FilterType::CatmullRom))?,
            );
            image = DynamicImage::ImageRgb8(image.to_rgb8());
            if config.do_resize.is_none() || config.do_resize.is_some_and(|x| x) {
                let (resized_height, resized_width) = self.smart_resize(
                    height as usize,
                    width as usize,
                    config.patch_size.context("Require `patch_size`.")?
                        * config.merge_size.context("Require `merge_size`")?,
                    config.min_pixels.context("Require `min_pixels`")?,
                    config.max_pixels.context("Require `max_pixels`")?,
                )?;
                height = resized_height as u32;
                width = resized_width as u32;
                image = image.resize_exact(
                    resized_width as u32,
                    resized_height as u32,
                    config
                        .resampling
                        .map(|resample| Some(resample).to_filter())
                        .unwrap_or(Ok(FilterType::CatmullRom))?,
                );
            }

            let to_tensor_rescale = Transforms {
                input: &ToTensor,
                inner_transforms: &[],
            };
            let image = image.apply(to_tensor_rescale, device)?;

            let transforms = TensorTransforms {
                inner_transforms: &[&Normalize {
                    mean: config.image_mean.unwrap_or(Self::DEFAULT_MEAN).to_vec(),
                    std: config.image_std.unwrap_or(Self::DEFAULT_STD).to_vec(),
                }],
            };
            let image = <Tensor as ApplyTensorTransforms>::apply(&image, transforms, device)?;

            processed_images.push(image);
        }

        let temporal_patch_size = config
            .temporal_patch_size
            .context("Require `temporal_patch_size")?;
        let remainder = processed_images.len() % temporal_patch_size;
        if remainder != 0 {
            let pad = temporal_patch_size - remainder;
            let last = processed_images.last().unwrap().clone();
            for _ in 0..pad {
                processed_images.push(last.clone());
            }
        }

        let mut patches = Tensor::stack(&processed_images, 0)?;
        let patch_size = config.patch_size.context("Require `patch_size")?;
        let merge_size = config.merge_size.context("Require `merge_size")?;
        // Image
        if patches.dim(0)? == 1 {
            patches = patches.repeat((temporal_patch_size, 1, 1, 1))?;
        }
        let channel = patches.dim(1)?;
        let grid_t = patches.dim(0)? / temporal_patch_size;
        let grid_h = height as usize / patch_size;
        let grid_w = width as usize / patch_size;
        patches = patches.reshape(&[
            grid_t,
            temporal_patch_size,
            channel,
            grid_h / merge_size,
            merge_size,
            patch_size,
            grid_w / merge_size,
            merge_size,
            patch_size,
        ])?;
        patches = patches.permute([0, 3, 6, 4, 7, 2, 1, 5, 8])?;
        let flattened_patches = patches.reshape((
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        ))?;

        Ok((
            flattened_patches,
            (grid_t as u32, grid_h as u32, grid_w as u32),
        ))
    }
}

impl ImagePreProcessor for Qwen2VLImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_, _): (usize, usize),
    ) -> candle_core::Result<PreprocessedImages> {
        let mut pixel_values = Vec::new();
        let mut vision_grid_thw = Vec::new();

        if !images.is_empty() {
            if let Some(max_edge) = self.max_edge {
                images = mistralrs_vision::pad_to_max_edge(&images, max_edge);
            }

            for image in images {
                let (w, h) = image.dimensions();
                let (patches, (t, gh, gw)) =
                    self.preprocess_inner(vec![image], config, device, (h, w))?;
                pixel_values.push(patches);
                vision_grid_thw.push(Tensor::new(&[t, gh, gw], &Device::Cpu)?);
            }
            let pixel_values = Tensor::cat(&pixel_values, 0)?;
            let vision_grid_thw = Tensor::stack(&vision_grid_thw, 0)?;
            return Ok(PreprocessedImages {
                pixel_values,
                pixel_attention_mask: None,
                image_sizes: None,
                num_img_tokens: None,
                aspect_ratio_ids: None,
                aspect_ratio_mask: None,
                num_tiles: None,
                image_grid_thw: Some(vision_grid_thw),
                video_grid_thw: None,
                rows: None,
                cols: None,
                pixel_values_list: None,
                tgt_sizes: None,
                image_sizes_all: None,
                num_crops: None,
            });
        }

        if !videos.is_empty() {
            for images in videos {
                let (w, h) = images[0].dimensions();
                let (patches, (t, gh, gw)) =
                    self.preprocess_inner(images, config, device, (h, w))?;
                pixel_values.push(patches);
                vision_grid_thw.push(Tensor::new(&[t, gh, gw], &Device::Cpu)?);
            }
            let pixel_values = Tensor::cat(&pixel_values, 0)?;
            let vision_grid_thw = Tensor::stack(&vision_grid_thw, 0)?;
            return Ok(PreprocessedImages {
                pixel_values,
                pixel_attention_mask: None,
                image_sizes: None,
                num_img_tokens: None,
                aspect_ratio_ids: None,
                aspect_ratio_mask: None,
                num_tiles: None,
                image_grid_thw: None,
                video_grid_thw: Some(vision_grid_thw),
                rows: None,
                cols: None,
                pixel_values_list: None,
                tgt_sizes: None,
                image_sizes_all: None,
                num_crops: None,
            });
        }
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_expansion_requires_exact_placeholder_grid_and_media_counts() -> Result<()> {
        let grid = Tensor::new(&[[1u32, 2, 2], [1, 2, 4]], &Device::Cpu)?;
        let mut text = format!(
            "a{}b{}c",
            Qwen2VLProcessor::IMAGE_PAD,
            Qwen2VLProcessor::IMAGE_PAD
        );
        expand_media_placeholders(
            &mut text,
            Qwen2VLProcessor::IMAGE_PAD,
            Qwen2VLProcessor::PLACEHOLDER,
            Some(&grid),
            2,
            4,
            MultimodalKind::Image,
        )?;
        assert_eq!(text.match_indices(Qwen2VLProcessor::IMAGE_PAD).count(), 3);

        let one_grid = Tensor::new(&[[1u32, 2, 2]], &Device::Cpu)?;
        let mut excess = format!(
            "{}{}",
            Qwen2VLProcessor::IMAGE_PAD,
            Qwen2VLProcessor::IMAGE_PAD
        );
        let original = excess.clone();
        assert!(expand_media_placeholders(
            &mut excess,
            Qwen2VLProcessor::IMAGE_PAD,
            Qwen2VLProcessor::PLACEHOLDER,
            Some(&one_grid),
            1,
            4,
            MultimodalKind::Image,
        )
        .is_err());
        assert_eq!(excess, original);

        let mut missing = Qwen2VLProcessor::IMAGE_PAD.to_string();
        assert!(expand_media_placeholders(
            &mut missing,
            Qwen2VLProcessor::IMAGE_PAD,
            Qwen2VLProcessor::PLACEHOLDER,
            Some(&grid),
            2,
            4,
            MultimodalKind::Image,
        )
        .is_err());
        assert!(expand_media_placeholders(
            &mut missing,
            Qwen2VLProcessor::IMAGE_PAD,
            Qwen2VLProcessor::PLACEHOLDER,
            Some(&one_grid),
            2,
            4,
            MultimodalKind::Image,
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn malformed_grid_and_range_hash_cardinality_fail_closed() -> Result<()> {
        let malformed_grid = Tensor::new(&[[1u32, 2]], &Device::Cpu)?;
        let mut text = Qwen2VLProcessor::VIDEO_PAD.to_string();
        assert!(expand_media_placeholders(
            &mut text,
            Qwen2VLProcessor::VIDEO_PAD,
            Qwen2VLProcessor::PLACEHOLDER,
            Some(&malformed_grid),
            1,
            4,
            MultimodalKind::Video,
        )
        .is_err());
        assert!(validated_mm_features(&[(1, 2), (5, 2)], &[11], MultimodalKind::Image).is_err());
        assert!(validated_mm_features(&[(1, 2)], &[11, 12], MultimodalKind::Image).is_err());
        Ok(())
    }

    #[test]
    fn split_pixels_keeps_image_video_order() -> Result<()> {
        let pixels = Tensor::arange(0f32, 10f32, &Device::Cpu)?.reshape((5, 2))?;
        let image_grid = Tensor::new(&[[1u32, 1, 2]], &Device::Cpu)?;
        let video_grid = Tensor::new(&[[1u32, 1, 3]], &Device::Cpu)?;
        let (images, videos) = split_media_pixels(&pixels, Some(&image_grid), Some(&video_grid))?;

        assert_eq!(
            images.unwrap().to_vec2::<f32>()?,
            vec![vec![0., 1.], vec![2., 3.]]
        );
        assert_eq!(
            videos.unwrap().to_vec2::<f32>()?,
            vec![vec![4., 5.], vec![6., 7.], vec![8., 9.]]
        );
        Ok(())
    }

    #[test]
    fn media_batch_selection_handles_heterogeneous_sequence_offsets() -> Result<()> {
        let pixels = Tensor::arange(0f32, 15f32, &Device::Cpu)?.reshape((15, 1))?;
        let grid = Tensor::new(
            &[[1u32, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 5]],
            &Device::Cpu,
        )?;
        let (pixels, grid) =
            select_media_batch(Some(pixels), Some(grid), &[3, 2], &[1, 1], &[1, 1])?;

        assert_eq!(
            grid.unwrap().to_vec2::<u32>()?,
            vec![vec![1, 1, 2], vec![1, 1, 5]]
        );
        assert_eq!(
            pixels.unwrap().flatten_all()?.to_vec1::<f32>()?,
            vec![1., 2., 10., 11., 12., 13., 14.]
        );
        Ok(())
    }

    #[test]
    fn media_span_shift_rejects_split_items() -> Result<()> {
        let mut split = vec![(2, 5)];
        assert!(shift_media_spans(&mut split, 3).is_err());

        let mut spans = vec![(0, 2), (4, 7)];
        assert_eq!(shift_media_spans(&mut spans, 2)?, 1);
        assert_eq!(spans, vec![(2, 5)]);
        Ok(())
    }
}
