use std::{any::Any, sync::Arc};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{ApplyTransforms, ToTensorNoNorm, Transforms};
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::{MultimodalAttentionPolicy, MultimodalKind},
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{
        build_mm_features_from_ranges, find_image_delimited_ranges, find_image_placeholder_ranges,
        Sequence,
    },
    video_input::VideoInput,
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        multimodal_layout::{
            MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout,
            PackedMultimodalLayout, RequestMultimodalLayout,
        },
        preprocessor_config::{PreProcessorConfig, ToFilter},
        qwen2vl::{
            media_data_cached_offset, select_media_batch, select_media_view, shift_media_spans,
            split_media_pixels, video_hashes,
        },
        ModelInputs,
    },
};

use super::MuseGlimmerSpecificArgs;

const DEFAULT_PATCH_SIZE: usize = 14;
const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;
const DEFAULT_MERGE_SIZE: usize = 2;
const DEFAULT_MAX_IMAGE_TOKENS: usize = 4096;
const DEFAULT_MAX_VIDEO_FRAME_TOKENS: usize = 144;
const DEFAULT_MAX_VIDEO_FRAMES: usize = 96;
const DEFAULT_VIDEO_FPS: f64 = 2.0;
const DEFAULT_IMAGE_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
const DEFAULT_IMAGE_STD: [f64; 3] = [0.5, 0.5, 0.5];
const DEFAULT_RESCALE_FACTOR: f64 = 1.0 / 255.0;

#[derive(Clone)]
struct MuseGlimmerProcessorSettings {
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    max_image_tokens: usize,
    max_video_frame_tokens: usize,
    image_mean: [f64; 3],
    image_std: [f64; 3],
    rescale_factor: f64,
    do_resize: bool,
    do_rescale: bool,
    do_normalize: bool,
    resampling: FilterType,
    gguf_collapsed_temporal: bool,
}

impl MuseGlimmerProcessorSettings {
    fn from_config(config: &PreProcessorConfig, gguf_collapsed_temporal: bool) -> Result<Self> {
        let patch_size = config.patch_size.unwrap_or(DEFAULT_PATCH_SIZE);
        let temporal_patch_size = config
            .temporal_patch_size
            .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE);
        let merge_size = config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE);
        if patch_size == 0 || temporal_patch_size == 0 || merge_size == 0 {
            anyhow::bail!("Muse-Glimmer patch and merge sizes must be nonzero");
        }
        Ok(Self {
            patch_size,
            temporal_patch_size,
            merge_size,
            max_image_tokens: config.max_image_tokens.unwrap_or(DEFAULT_MAX_IMAGE_TOKENS),
            max_video_frame_tokens: DEFAULT_MAX_VIDEO_FRAME_TOKENS,
            image_mean: config.image_mean.unwrap_or(DEFAULT_IMAGE_MEAN),
            image_std: config.image_std.unwrap_or(DEFAULT_IMAGE_STD),
            rescale_factor: config.rescale_factor.unwrap_or(DEFAULT_RESCALE_FACTOR),
            do_resize: config.do_resize.unwrap_or(true),
            do_rescale: config.do_rescale.unwrap_or(true),
            do_normalize: config.do_normalize.unwrap_or(true),
            resampling: Some(config.resampling.unwrap_or(1)).to_filter()?,
            gguf_collapsed_temporal,
        })
    }

    fn merge_length(&self) -> usize {
        self.merge_size.pow(2)
    }
}

struct MuseGlimmerImageProcessor {
    settings: Arc<MuseGlimmerProcessorSettings>,
    max_edge: Option<u32>,
}

pub struct MuseGlimmerProcessor {
    settings: Arc<MuseGlimmerProcessorSettings>,
    max_edge: Option<u32>,
}

impl MuseGlimmerProcessor {
    pub const IMAGE_TOKEN: &str = "<|patch|>";
    pub const IMAGE_START: &str = "<|image_start|>";
    pub const IMAGE_END: &str = "<|image_end|>";
    pub const VIDEO_TOKEN: &str = "<|video|>";
    pub const VIDEO_START: &str = "<|vid_start|>";
    pub const VIDEO_END: &str = "<|vid_end|>";
    pub const VIDEO_SEPARATOR: &str = "<|vid_frame_separator|>";

    pub fn new(
        config: &PreProcessorConfig,
        max_edge: Option<u32>,
        gguf_collapsed_temporal: bool,
    ) -> Result<Self> {
        Ok(Self {
            settings: Arc::new(MuseGlimmerProcessorSettings::from_config(
                config,
                gguf_collapsed_temporal,
            )?),
            max_edge,
        })
    }
}

impl Processor for MuseGlimmerProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MuseGlimmerImageProcessor {
            settings: self.settings.clone(),
            max_edge: self.max_edge,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            Self::IMAGE_TOKEN,
            Self::IMAGE_START,
            Self::IMAGE_END,
            Self::VIDEO_TOKEN,
            Self::VIDEO_START,
            Self::VIDEO_END,
            Self::VIDEO_SEPARATOR,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

#[derive(Clone)]
struct SampledVideo {
    frames: Vec<DynamicImage>,
    timestamps: Vec<f64>,
}

struct PreparedMediaBatch {
    image_pixels: Option<Tensor>,
    video_pixels: Option<Tensor>,
    image_grid: Option<Tensor>,
    video_grid: Option<Tensor>,
    image_item_counts: Vec<usize>,
    video_item_counts: Vec<usize>,
    per_seq_image_grids: Vec<Option<Tensor>>,
    per_seq_video_grids: Vec<Option<Tensor>>,
}

fn replace_occurrences(text: &str, needle: &str, replacements: &[String]) -> Result<String> {
    let parts = text.split(needle).collect::<Vec<_>>();
    let count = parts.len().saturating_sub(1);
    if count != replacements.len() {
        anyhow::bail!(
            "Muse-Glimmer prompt has {count} `{needle}` placeholders for {} media items",
            replacements.len()
        );
    }
    let extra = replacements.iter().map(String::len).sum::<usize>();
    let mut output = String::with_capacity(text.len() + extra);
    output.push_str(parts[0]);
    for (replacement, part) in replacements.iter().zip(parts.iter().skip(1)) {
        output.push_str(replacement);
        output.push_str(part);
    }
    Ok(output)
}

fn find_sequences(tokens: &[u32], needle: u32) -> Vec<(usize, usize)> {
    find_image_placeholder_ranges(tokens, needle)
        .into_iter()
        .map(|(start, len)| (start, start + len))
        .collect()
}

fn tensor_grid_rows(grid: Option<&Tensor>) -> Result<Vec<[usize; 3]>> {
    grid.map(Tensor::to_vec2::<u32>)
        .transpose()?
        .unwrap_or_default()
        .into_iter()
        .map(|row| {
            if row.len() != 3 {
                anyhow::bail!("Muse-Glimmer media grids must contain t, h, and w");
            }
            Ok([row[0] as usize, row[1] as usize, row[2] as usize])
        })
        .collect()
}

fn validate_grid_tokens(row: [usize; 3], merge_length: usize) -> Result<usize> {
    let patches = row.into_iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(value)
            .ok_or_else(|| anyhow::anyhow!("Muse-Glimmer media grid size overflow"))
    })?;
    if patches == 0 || !patches.is_multiple_of(merge_length) {
        anyhow::bail!(
            "Muse-Glimmer media grid produces {patches} patches for merge length {merge_length}"
        );
    }
    Ok(patches / merge_length)
}

fn image_replacements(grid: Option<&Tensor>, merge_length: usize) -> Result<Vec<String>> {
    tensor_grid_rows(grid)?
        .into_iter()
        .map(|row| {
            let tokens = validate_grid_tokens(row, merge_length)?;
            Ok(format!(
                "{}{}{}",
                MuseGlimmerProcessor::IMAGE_START,
                MuseGlimmerProcessor::IMAGE_TOKEN.repeat(tokens),
                MuseGlimmerProcessor::IMAGE_END
            ))
        })
        .collect()
}

fn video_replacements(
    grid: Option<&Tensor>,
    videos: &[VideoInput],
    settings: &MuseGlimmerProcessorSettings,
) -> Result<Vec<String>> {
    let rows = tensor_grid_rows(grid)?;
    if rows.len() != videos.len() {
        anyhow::bail!(
            "Muse-Glimmer has {} video grids for {} videos",
            rows.len(),
            videos.len()
        );
    }
    rows.into_iter()
        .zip(videos)
        .map(|(row, video)| {
            let sampled = sample_video(video, settings.temporal_patch_size);
            let tokens_per_group =
                validate_grid_tokens([1, row[1], row[2]], settings.merge_length())?;
            let mut timestamps = sampled
                .timestamps
                .iter()
                .step_by(settings.temporal_patch_size)
                .copied()
                .take(row[0])
                .collect::<Vec<_>>();
            while timestamps.len() < row[0] {
                timestamps.push(timestamps.last().copied().unwrap_or(0.0));
            }
            let mut replacement = MuseGlimmerProcessor::VIDEO_START.to_string();
            for (index, timestamp) in timestamps.into_iter().enumerate() {
                replacement.push_str(&format!("Time: {timestamp:.1}s"));
                replacement.push_str(&MuseGlimmerProcessor::VIDEO_TOKEN.repeat(tokens_per_group));
                replacement.push_str(if index + 1 == row[0] {
                    MuseGlimmerProcessor::VIDEO_END
                } else {
                    MuseGlimmerProcessor::VIDEO_SEPARATOR
                });
            }
            Ok(replacement)
        })
        .collect()
}

fn sample_video(video: &VideoInput, temporal_patch_size: usize) -> SampledVideo {
    if video.frames.is_empty() {
        return SampledVideo {
            frames: Vec::new(),
            timestamps: Vec::new(),
        };
    }
    let fps = if video.fps.is_finite() && video.fps > 0.0 {
        video.fps
    } else {
        24.0
    };
    let total_frames = video.total_num_frames.max(video.frames.len());
    let mut count = ((total_frames as f64 * DEFAULT_VIDEO_FPS / fps) as usize)
        .min(DEFAULT_MAX_VIDEO_FRAMES)
        .min(total_frames);
    count = count.max(temporal_patch_size);
    count = count / temporal_patch_size * temporal_patch_size;
    count = count.min(total_frames).min(video.frames.len()).max(1);

    let indices = if count == 1 {
        vec![0]
    } else {
        (0..count)
            .map(|index| index * (video.frames.len() - 1) / (count - 1))
            .collect()
    };
    SampledVideo {
        frames: indices
            .iter()
            .map(|&index| video.frames[index].clone())
            .collect(),
        timestamps: indices
            .iter()
            .map(|&index| video.sampled_indices.get(index).copied().unwrap_or(index) as f64 / fps)
            .collect(),
    }
}

fn shift_video_spans(
    spans: &mut Vec<(usize, usize)>,
    grid: Option<&Tensor>,
    prefix_len: usize,
) -> Result<(usize, usize)> {
    let rows = tensor_grid_rows(grid)?;
    let expected = rows.iter().map(|row| row[0]).sum::<usize>();
    if spans.len() != expected {
        anyhow::bail!(
            "Muse-Glimmer has {} video placeholder groups for {expected} temporal groups",
            spans.len()
        );
    }
    let mut cursor = 0usize;
    let mut cached = 0usize;
    let mut current = 0usize;
    for row in rows {
        let item = &spans[cursor..cursor + row[0]];
        let all_cached = item.iter().all(|(_, end)| *end <= prefix_len);
        let all_current = item.iter().all(|(start, _)| *start >= prefix_len);
        if all_cached {
            cached += 1;
        } else if all_current {
            current += 1;
        } else {
            anyhow::bail!("Muse-Glimmer prefix cache splits a video item");
        }
        cursor += row[0];
    }
    spans.retain(|(_, end)| *end > prefix_len);
    for (start, end) in spans.iter_mut() {
        *start -= prefix_len;
        *end -= prefix_len;
    }
    Ok((cached, current))
}

fn packed_layout(
    input_seqs: &[&mut Sequence],
    query_lens: &[usize],
    image_spans: &[Vec<(usize, usize)>],
    video_spans: &[Vec<(usize, usize)>],
    image_grids: &[Option<Tensor>],
    video_grids: &[Option<Tensor>],
    merge_length: usize,
) -> Result<PackedMultimodalLayout> {
    let count = input_seqs.len();
    if [
        query_lens.len(),
        image_spans.len(),
        video_spans.len(),
        image_grids.len(),
        video_grids.len(),
    ]
    .into_iter()
    .any(|len| len != count)
    {
        anyhow::bail!("Muse-Glimmer packed multimodal metadata length mismatch");
    }

    let mut requests = Vec::with_capacity(count);
    for (((((seq, &query_len), image_spans), video_spans), image_grid), video_grid) in input_seqs
        .iter()
        .zip(query_lens)
        .zip(image_spans)
        .zip(video_spans)
        .zip(image_grids)
        .zip(video_grids)
    {
        if query_len != seq.get_toks().len() {
            anyhow::bail!("Muse-Glimmer packed prefill requires the complete prompt");
        }
        let image_rows = tensor_grid_rows(image_grid.as_ref())?;
        let image_hashes = seq.image_hashes().unwrap_or_default();
        if image_rows.len() != image_spans.len() || image_hashes.len() != image_spans.len() {
            anyhow::bail!("Muse-Glimmer packed image metadata does not align");
        }
        let mut items = Vec::new();
        for (item_index, ((&hash, &(start, end)), row)) in image_hashes
            .iter()
            .zip(image_spans)
            .zip(image_rows)
            .enumerate()
        {
            let expected = validate_grid_tokens(row, merge_length)?;
            if end - start != expected {
                anyhow::bail!("Muse-Glimmer image placeholders do not match the image grid");
            }
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

        let video_rows = tensor_grid_rows(video_grid.as_ref())?;
        let hashes = video_hashes(seq);
        if video_rows.len() != hashes.len() {
            anyhow::bail!("Muse-Glimmer packed video grids do not align with videos");
        }
        let mut span_cursor = 0usize;
        for (item_index, (&hash, row)) in hashes.iter().zip(video_rows).enumerate() {
            let groups = row[0];
            let item_spans = video_spans
                .get(span_cursor..span_cursor + groups)
                .ok_or_else(|| anyhow::anyhow!("Muse-Glimmer video spans do not cover the grid"))?;
            let tokens_per_group = validate_grid_tokens([1, row[1], row[2]], merge_length)?;
            let mut source_start = 0usize;
            let mut maps = Vec::with_capacity(groups);
            for &(start, end) in item_spans {
                if end - start != tokens_per_group {
                    anyhow::bail!("Muse-Glimmer video placeholders do not match the video grid");
                }
                maps.push(MultimodalEmbeddingMap::contiguous(
                    start..end,
                    source_start,
                    0,
                )?);
                source_start += tokens_per_group;
            }
            let placeholder = item_spans
                .first()
                .zip(item_spans.last())
                .map(|(first, last)| first.0..last.1)
                .ok_or_else(|| anyhow::anyhow!("Muse-Glimmer video grid has no temporal groups"))?;
            items.push(MultimodalItemLayout::new(
                MultimodalEncoderKey {
                    kind: MultimodalKind::Video,
                    hash,
                },
                item_index,
                placeholder,
                MultimodalAttentionPolicy::Causal,
                maps,
            )?);
            span_cursor += groups;
        }
        if span_cursor != video_spans.len() {
            anyhow::bail!("Muse-Glimmer video spans exceed the video grids");
        }
        requests.push(RequestMultimodalLayout {
            sequence_id: *seq.id(),
            query: 0..query_len,
            items,
        });
    }
    Ok(PackedMultimodalLayout::new(&requests)?)
}

impl MuseGlimmerImageProcessor {
    fn smart_resize(
        height: usize,
        width: usize,
        patch_size: usize,
        max_tokens: usize,
    ) -> Result<(usize, usize)> {
        if height == 0 || width == 0 || patch_size == 0 || max_tokens == 0 {
            anyhow::bail!("Muse-Glimmer resize dimensions and limits must be nonzero");
        }
        let mut ideal_h = height as f64 / patch_size as f64;
        let mut ideal_w = width as f64 / patch_size as f64;
        let ratio = ideal_w / ideal_h;
        if ideal_h * ideal_w > max_tokens as f64 {
            ideal_h = (max_tokens as f64 / ratio).sqrt();
            ideal_w = ideal_h * ratio;
        }
        let hs = [ideal_h.floor() as usize, ideal_h.ceil() as usize];
        let ws = [ideal_w.floor() as usize, ideal_w.ceil() as usize];
        let mut candidates = Vec::new();
        for h in hs {
            for w in ws {
                if h > 0 && w > 0 && h * w <= max_tokens && !candidates.contains(&(h, w)) {
                    candidates.push((h, w));
                }
            }
        }
        if candidates.is_empty() {
            candidates.push((
                ideal_h.round().max(1.0) as usize,
                ideal_w.round().max(1.0) as usize,
            ));
        }
        let source_ratio = height as f64 / width as f64;
        let (grid_h, grid_w) = candidates
            .into_iter()
            .min_by(|(ah, aw), (bh, bw)| {
                let a = (*ah as f64 / *aw as f64 - source_ratio).abs();
                let b = (*bh as f64 / *bw as f64 - source_ratio).abs();
                a.total_cmp(&b)
            })
            .expect("resize candidates are nonempty");
        Ok((grid_h * patch_size, grid_w * patch_size))
    }

    fn maybe_limit_edge(&self, image: DynamicImage) -> DynamicImage {
        let Some(max_edge) = self.max_edge else {
            return image;
        };
        let (width, height) = image.dimensions();
        if width <= max_edge && height <= max_edge {
            image
        } else {
            image.thumbnail(max_edge, max_edge)
        }
    }

    fn transform_image(&self, image: DynamicImage, max_tokens: usize) -> Result<Tensor> {
        let image = self.maybe_limit_edge(DynamicImage::ImageRgb8(image.to_rgb8()));
        let (width, height) = image.dimensions();
        let image = if self.settings.do_resize {
            let factor = self.settings.patch_size * self.settings.merge_size;
            let (target_h, target_w) =
                Self::smart_resize(height as usize, width as usize, factor, max_tokens)?;
            image.resize_exact(target_w as u32, target_h as u32, self.settings.resampling)
        } else {
            image
        };
        let to_tensor = Transforms {
            input: &ToTensorNoNorm,
            inner_transforms: &[],
        };
        let mut tensor = image.apply(to_tensor, &Device::Cpu)?;
        if self.settings.do_rescale {
            tensor = tensor.affine(self.settings.rescale_factor, 0.0)?;
        }
        if self.settings.do_normalize {
            let channels = tensor.dim(0)?;
            if channels != 3 {
                anyhow::bail!("Muse-Glimmer expects RGB images");
            }
            let mean = Tensor::new(&self.settings.image_mean, tensor.device())?
                .to_dtype(tensor.dtype())?
                .reshape((3, 1, 1))?;
            let std = Tensor::new(&self.settings.image_std, tensor.device())?
                .to_dtype(tensor.dtype())?
                .reshape((3, 1, 1))?;
            tensor = tensor.broadcast_sub(&mean)?.broadcast_div(&std)?;
        }
        Ok(tensor)
    }

    fn patchify(&self, mut frames: Vec<Tensor>, device: &Device) -> Result<(Tensor, [u32; 3])> {
        if frames.is_empty() {
            anyhow::bail!("Muse-Glimmer video input contains no frames");
        }
        let temporal = self.settings.temporal_patch_size;
        let remainder = frames.len() % temporal;
        if remainder != 0 {
            let last = frames.last().expect("frames are nonempty").clone();
            frames.extend((0..temporal - remainder).map(|_| last.clone()));
        }
        let frames = Tensor::stack(&frames, 0)?.to_device(device)?;
        let (_, channels, height, width) = frames.dims4()?;
        let patch = self.settings.patch_size;
        if !height.is_multiple_of(patch) || !width.is_multiple_of(patch) {
            anyhow::bail!("Muse-Glimmer resized dimensions must be divisible by patch size");
        }
        let grid_t = frames.dim(0)? / temporal;
        let grid_h = height / patch;
        let grid_w = width / patch;
        let patches = frames
            .reshape(&[grid_t, temporal, channels, grid_h, patch, grid_w, patch])?
            .permute([0, 3, 5, 1, 2, 4, 6])?
            .reshape((
                grid_t * grid_h * grid_w,
                temporal * channels * patch * patch,
            ))?;
        Ok((patches, [grid_t as u32, grid_h as u32, grid_w as u32]))
    }

    fn preprocess_images(
        &self,
        images: Vec<DynamicImage>,
        device: &Device,
    ) -> Result<PreprocessedImages> {
        let mut pixels = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len());
        for image in images {
            let frame = self.transform_image(image, self.settings.max_image_tokens)?;
            let temporal = self.settings.temporal_patch_size;
            let frames = (0..temporal).map(|_| frame.clone()).collect();
            let (patches, grid) = self.patchify(frames, device)?;
            pixels.push(patches);
            grids.push(Tensor::new(&grid, &Device::Cpu)?);
        }
        self.preprocessed(pixels, grids, true)
    }

    fn preprocess_videos(
        &self,
        videos: Vec<VideoInput>,
        device: &Device,
    ) -> Result<PreprocessedImages> {
        if self.settings.gguf_collapsed_temporal && !videos.is_empty() {
            anyhow::bail!(
                "Muse-Glimmer GGUF projectors collapse temporal patch weights and cannot process video; use the original safetensors model"
            );
        }
        let mut pixels = Vec::with_capacity(videos.len());
        let mut grids = Vec::with_capacity(videos.len());
        for video in videos {
            let sampled = sample_video(&video, self.settings.temporal_patch_size);
            let mut frames = Vec::with_capacity(sampled.frames.len());
            for frame in sampled.frames {
                frames.push(self.transform_image(frame, self.settings.max_video_frame_tokens)?);
            }
            let (patches, grid) = self.patchify(frames, device)?;
            pixels.push(patches);
            grids.push(Tensor::new(&grid, &Device::Cpu)?);
        }
        self.preprocessed(pixels, grids, false)
    }

    fn preprocessed(
        &self,
        pixels: Vec<Tensor>,
        grids: Vec<Tensor>,
        image: bool,
    ) -> Result<PreprocessedImages> {
        let pixel_values = Tensor::cat(&pixels, 0)?;
        let grid = Tensor::stack(&grids, 0)?;
        Ok(PreprocessedImages {
            pixel_values,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: image.then_some(grid.clone()),
            video_grid_thw: (!image).then_some(grid),
            rows: None,
            cols: None,
            pixel_values_list: None,
            tgt_sizes: None,
            image_sizes_all: None,
            num_crops: None,
        })
    }

    fn cached_media(
        &self,
        seq: &mut Sequence,
        device: &Device,
    ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
        if let Some(pixels) = &seq.multimodal.cached_pixel_values {
            return Ok((
                pixels.clone(),
                seq.multimodal.cached_img_thw.clone(),
                seq.multimodal.cached_vid_thw.clone(),
            ));
        }
        let image = seq
            .has_images()
            .then(|| self.preprocess_images(seq.clone_images().unwrap_or_default(), device))
            .transpose()?;
        let video = seq
            .has_videos()
            .then(|| self.preprocess_videos(seq.clone_videos().unwrap_or_default(), device))
            .transpose()?;
        let image_grid = image
            .as_ref()
            .and_then(|value| value.image_grid_thw.clone());
        let video_grid = video
            .as_ref()
            .and_then(|value| value.video_grid_thw.clone());
        let pixels = image
            .into_iter()
            .chain(video)
            .map(|value| value.pixel_values)
            .collect::<Vec<_>>();
        if pixels.is_empty() {
            anyhow::bail!("Muse-Glimmer media sequence contains no media data");
        }
        let pixels = Tensor::cat(&pixels, 0)?;
        seq.multimodal.cached_pixel_values = Some(pixels.clone());
        seq.multimodal.cached_img_thw = image_grid.clone();
        seq.multimodal.cached_vid_thw = video_grid.clone();
        Ok((pixels, image_grid, video_grid))
    }

    fn expand_sequence(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut Sequence,
        device: &Device,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        if seq.multimodal.has_changed_prompt || (!seq.has_images() && !seq.has_videos()) {
            return Ok(());
        }
        let (_, image_grid, video_grid) = self.cached_media(seq, device)?;
        let mut prompt = tokenizer
            .decode(seq.get_toks(), false)
            .map_err(anyhow::Error::msg)?;
        let images = seq.clone_images().unwrap_or_default();
        let image_replacements =
            image_replacements(image_grid.as_ref(), self.settings.merge_length())?;
        if images.len() != image_replacements.len() {
            anyhow::bail!("Muse-Glimmer image grids do not match the supplied images");
        }
        prompt = replace_occurrences(
            &prompt,
            MuseGlimmerProcessor::IMAGE_TOKEN,
            &image_replacements,
        )?;
        let videos = seq.clone_videos().unwrap_or_default();
        let video_replacements = video_replacements(video_grid.as_ref(), &videos, &self.settings)?;
        prompt = replace_occurrences(
            &prompt,
            MuseGlimmerProcessor::VIDEO_TOKEN,
            &video_replacements,
        )?;

        let ids = tokenizer
            .encode_fast(prompt.clone(), false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        seq.set_initial_prompt(prompt);
        if seq.mm_features().is_empty() {
            let image_start = tokenizer
                .token_to_id(MuseGlimmerProcessor::IMAGE_START)
                .context("Muse-Glimmer tokenizer is missing image start token")?;
            let image_end = tokenizer
                .token_to_id(MuseGlimmerProcessor::IMAGE_END)
                .context("Muse-Glimmer tokenizer is missing image end token")?;
            let video_start = tokenizer
                .token_to_id(MuseGlimmerProcessor::VIDEO_START)
                .context("Muse-Glimmer tokenizer is missing video start token")?;
            let video_end = tokenizer
                .token_to_id(MuseGlimmerProcessor::VIDEO_END)
                .context("Muse-Glimmer tokenizer is missing video end token")?;
            let image_ranges = find_image_delimited_ranges(&ids, image_start, image_end);
            let video_ranges = find_image_delimited_ranges(&ids, video_start, video_end);
            let image_hashes = seq.image_hashes().unwrap_or_default();
            let video_hashes = video_hashes(seq);
            if image_ranges.len() != image_hashes.len() || video_ranges.len() != video_hashes.len()
            {
                anyhow::bail!("Muse-Glimmer media placeholders do not match the supplied media");
            }
            let mut features =
                build_mm_features_from_ranges(&image_ranges, image_hashes, MultimodalKind::Image);
            features.extend(build_mm_features_from_ranges(
                &video_ranges,
                &video_hashes,
                MultimodalKind::Video,
            ));
            if !features.is_empty() {
                seq.set_mm_features(features);
            }
        }
        seq.set_toks_and_reallocate(ids, paged_attn_metadata);
        seq.multimodal.has_changed_prompt = true;
        Ok(())
    }

    fn prepare_media_batch(
        &self,
        input_seqs: &mut [&mut Sequence],
        device: &Device,
    ) -> Result<PreparedMediaBatch> {
        let mut image_pixels = Vec::new();
        let mut video_pixels = Vec::new();
        let mut image_grids = Vec::with_capacity(input_seqs.len());
        let mut video_grids = Vec::with_capacity(input_seqs.len());
        let mut image_item_counts = vec![0; input_seqs.len()];
        let mut video_item_counts = vec![0; input_seqs.len()];

        for (index, seq) in input_seqs.iter_mut().enumerate() {
            let cached = if let Some(pixels) = seq.multimodal.cached_pixel_values.clone() {
                Some((
                    pixels,
                    seq.multimodal.cached_img_thw.clone(),
                    seq.multimodal.cached_vid_thw.clone(),
                ))
            } else if seq.has_images() || seq.has_videos() {
                Some(self.cached_media(seq, device)?)
            } else {
                None
            };
            let Some((cached_pixels, image_grid, video_grid)) = cached else {
                image_grids.push(None);
                video_grids.push(None);
                continue;
            };
            let (images, videos) =
                split_media_pixels(&cached_pixels, image_grid.as_ref(), video_grid.as_ref())?;
            let (images, image_grid, image_count) =
                select_media_view(seq, MultimodalKind::Image, images, image_grid)?;
            let (videos, video_grid, video_count) =
                select_media_view(seq, MultimodalKind::Video, videos, video_grid)?;
            image_item_counts[index] = image_count;
            video_item_counts[index] = video_count;
            if let Some(images) = images {
                image_pixels.push(images.to_device(device)?);
            }
            if let Some(videos) = videos {
                video_pixels.push(videos.to_device(device)?);
            }
            image_grids.push(image_grid);
            video_grids.push(video_grid);
        }

        let cat_pixels = |pixels: Vec<Tensor>| {
            (!pixels.is_empty())
                .then(|| Tensor::cat(&pixels, 0))
                .transpose()
        };
        let cat_grids = |grids: &[Option<Tensor>]| {
            let tensors = grids.iter().filter_map(Clone::clone).collect::<Vec<_>>();
            (!tensors.is_empty())
                .then(|| Tensor::cat(&tensors, 0))
                .transpose()
        };
        Ok(PreparedMediaBatch {
            image_pixels: cat_pixels(image_pixels)?,
            video_pixels: cat_pixels(video_pixels)?,
            image_grid: cat_grids(&image_grids)?,
            video_grid: cat_grids(&video_grids)?,
            image_item_counts,
            video_item_counts,
            per_seq_image_grids: image_grids,
            per_seq_video_grids: video_grids,
        })
    }
}

impl InputsProcessor for MuseGlimmerImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        let tokenizer = tokenizer.context("Muse-Glimmer requires a tokenizer")?;
        for seq in input_seqs {
            self.expand_sequence(&tokenizer, seq, device, paged_attn_metadata.as_deref_mut())?;
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
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        if is_xlora {
            anyhow::bail!("Muse-Glimmer does not support X-LoRA");
        }
        if no_kv_cache {
            anyhow::bail!("Muse-Glimmer requires the KV cache");
        }
        let tokenizer = tokenizer.context("Muse-Glimmer requires a tokenizer")?;
        if is_prompt {
            for seq in input_seqs.iter_mut() {
                self.expand_sequence(&tokenizer, seq, device, paged_attn_metadata.as_mut())?;
            }
        }
        let mut media = self.prepare_media_batch(input_seqs, device)?;

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
                input_seqs.iter().map(|seq| seq.get_toks()).collect(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )?
        } else {
            get_completion_input(
                input_seqs.iter().map(|seq| seq.get_toks()).collect(),
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

        let image_token = tokenizer
            .token_to_id(MuseGlimmerProcessor::IMAGE_TOKEN)
            .context("Muse-Glimmer tokenizer is missing image token")?;
        let video_token = tokenizer
            .token_to_id(MuseGlimmerProcessor::VIDEO_TOKEN)
            .context("Muse-Glimmer tokenizer is missing video token")?;
        let mut image_spans = input_seqs
            .iter()
            .map(|seq| find_sequences(seq.get_toks(), image_token))
            .collect::<Vec<_>>();
        let mut video_spans = input_seqs
            .iter()
            .map(|seq| find_sequences(seq.get_toks(), video_token))
            .collect::<Vec<_>>();

        let packed_layout = if is_prompt && flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .context("packed Muse-Glimmer prefill requires query lengths")?;
            let layout = packed_layout(
                input_seqs,
                query_lens,
                &image_spans,
                &video_spans,
                &media.per_seq_image_grids,
                &media.per_seq_video_grids,
                self.settings.merge_length(),
            )?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "Muse-Glimmer packed layout has {} tokens but input has {}",
                    layout.token_count(),
                    input.dim(1)?
                );
            }
            Some(layout)
        } else {
            None
        };

        let mut cached_images = vec![0; input_seqs.len()];
        let mut current_images = vec![0; input_seqs.len()];
        let mut cached_videos = vec![0; input_seqs.len()];
        let mut current_videos = vec![0; input_seqs.len()];
        if is_prompt {
            for index in 0..input_seqs.len() {
                let seq = &input_seqs[index];
                let prefix = seq
                    .active_prompt_local_query_range()
                    .map_or(seq.prefix_cache_len(), |query| query.start);
                let cached = shift_media_spans(&mut image_spans[index], prefix)?;
                cached_images[index] = media_data_cached_offset(seq, cached);
                current_images[index] = image_spans[index].len();
                let (cached, current) = shift_video_spans(
                    &mut video_spans[index],
                    media.per_seq_video_grids[index].as_ref(),
                    prefix,
                )?;
                cached_videos[index] = media_data_cached_offset(seq, cached);
                current_videos[index] = current;
            }
            (media.image_pixels, media.image_grid) = select_media_batch(
                media.image_pixels,
                media.image_grid,
                &media.image_item_counts,
                &cached_images,
                &current_images,
            )?;
            (media.video_pixels, media.video_grid) = select_media_batch(
                media.video_pixels,
                media.video_grid,
                &media.video_item_counts,
                &cached_videos,
                &current_videos,
            )?;
        } else {
            media.image_pixels = None;
            media.video_pixels = None;
            media.image_grid = None;
            media.video_grid = None;
            image_spans.iter_mut().for_each(Vec::clear);
            video_spans.iter_mut().for_each(Vec::clear);
        }

        let mut image_hashes = Vec::new();
        let mut selected_video_hashes = Vec::new();
        if is_prompt {
            for (index, seq) in input_seqs.iter().enumerate() {
                let hashes = seq.image_hashes().unwrap_or_default();
                image_hashes.extend_from_slice(
                    hashes
                        .get(cached_images[index]..cached_images[index] + current_images[index])
                        .context("Muse-Glimmer image hashes do not cover the media window")?,
                );
                let hashes = video_hashes(seq);
                selected_video_hashes.extend_from_slice(
                    hashes
                        .get(cached_videos[index]..cached_videos[index] + current_videos[index])
                        .context("Muse-Glimmer video hashes do not cover the media window")?,
                );
            }
        }

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values: media.image_pixels,
            model_specific_args: Box::new(MuseGlimmerSpecificArgs {
                pixel_values_videos: media.video_pixels,
                image_grid_thw: media.image_grid,
                video_grid_thw: media.video_grid,
                continuous_img_pad: image_spans,
                continuous_vid_pad: video_spans,
                image_hashes,
                video_hashes: selected_video_hashes,
                packed_layout,
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

impl ImagePreProcessor for MuseGlimmerImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = DEFAULT_IMAGE_MEAN;
    const DEFAULT_STD: [f64; 3] = DEFAULT_IMAGE_STD;

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        _config: &PreProcessorConfig,
        device: &Device,
        _size: (usize, usize),
    ) -> candle_core::Result<PreprocessedImages> {
        if !images.is_empty() {
            self.preprocess_images(images, device)
                .map_err(candle_core::Error::msg)
        } else {
            let videos = videos
                .into_iter()
                .map(|frames| VideoInput::from_frames(frames, 24.0, None))
                .collect();
            self.preprocess_videos(videos, device)
                .map_err(candle_core::Error::msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;
    use tokio::sync::{mpsc::channel, Mutex};

    use crate::{
        sampler::Sampler,
        sequence::{SeqStepType, SequenceGroup, SequenceRecognizer},
    };

    fn settings() -> Arc<MuseGlimmerProcessorSettings> {
        Arc::new(MuseGlimmerProcessorSettings {
            patch_size: 1,
            temporal_patch_size: 2,
            merge_size: 1,
            max_image_tokens: 4096,
            max_video_frame_tokens: 144,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            rescale_factor: 1.0,
            do_resize: false,
            do_rescale: false,
            do_normalize: false,
            resampling: FilterType::Lanczos3,
            gguf_collapsed_temporal: false,
        })
    }

    fn sequence_with_image(image: DynamicImage) -> Sequence {
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
        Sequence::new_waiting(
            vec![1],
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
            Arc::new(Mutex::new(SequenceGroup::new(1, false, true, None))),
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            Some(vec![image]),
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
            false,
            vec![],
            None,
        )
    }

    #[test]
    fn resize_matches_transformers_grid() -> Result<()> {
        assert_eq!(
            MuseGlimmerImageProcessor::smart_resize(480, 640, 28, 4096)?,
            (476, 644)
        );
        assert_eq!(
            MuseGlimmerImageProcessor::smart_resize(4000, 3000, 28, 4096)?,
            (2044, 1540)
        );
        Ok(())
    }

    #[test]
    fn video_sampling_matches_transformers_linspace() {
        let video = VideoInput {
            frames: vec![DynamicImage::new_rgb8(1, 1); 12],
            fps: 4.0,
            total_num_frames: 12,
            sampled_indices: (0..12).collect(),
        };
        let sampled = sample_video(&video, 2);
        assert_eq!(sampled.frames.len(), 6);
        assert_eq!(sampled.timestamps, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.75]);
    }

    #[test]
    fn video_prefix_selection_keeps_temporal_groups_together() -> Result<()> {
        let grid = Tensor::new(&[[2u32, 2, 2], [3, 2, 2]], &Device::Cpu)?;
        let mut spans = vec![(2, 3), (5, 6), (10, 11), (13, 14), (16, 17)];
        assert_eq!(shift_video_spans(&mut spans, Some(&grid), 8)?, (1, 1));
        assert_eq!(spans, vec![(2, 3), (5, 6), (8, 9)]);

        let mut split = vec![(2, 3), (5, 6), (10, 11), (13, 14), (16, 17)];
        assert!(shift_video_spans(&mut split, Some(&grid), 12).is_err());
        Ok(())
    }

    #[test]
    fn patch_rows_are_temporal_then_channel() -> Result<()> {
        let processor = MuseGlimmerImageProcessor {
            settings: settings(),
            max_edge: None,
        };
        let first = Tensor::new(&[[[1f32]], [[2.]], [[3.]]], &Device::Cpu)?;
        let second = Tensor::new(&[[[4f32]], [[5.]], [[6.]]], &Device::Cpu)?;
        let (patches, grid) = processor.patchify(vec![first, second], &Device::Cpu)?;
        assert_eq!(grid, [1, 1, 1]);
        assert_eq!(
            patches.to_dtype(DType::F32)?.to_vec2::<f32>()?,
            vec![vec![1., 2., 3., 4., 5., 6.]]
        );
        Ok(())
    }

    #[test]
    fn image_and_video_replacements_match_hf_protocol() -> Result<()> {
        let image_grid = Tensor::new(&[[1u32, 2, 2]], &Device::Cpu)?;
        let image = image_replacements(Some(&image_grid), 4)?;
        assert_eq!(image, vec!["<|image_start|><|patch|><|image_end|>"]);

        let video_grid = Tensor::new(&[[2u32, 2, 2]], &Device::Cpu)?;
        let video = VideoInput {
            frames: vec![DynamicImage::new_rgb8(1, 1); 4],
            fps: 2.0,
            total_num_frames: 4,
            sampled_indices: vec![0, 1, 2, 3],
        };
        let replacement = video_replacements(Some(&video_grid), &[video], &settings())?;
        assert_eq!(
            replacement[0],
            "<|vid_start|>Time: 0.0s<|video|><|video|><|video|><|video|><|vid_frame_separator|>Time: 1.0s<|video|><|video|><|video|><|video|><|vid_end|>"
        );
        Ok(())
    }

    #[test]
    fn gguf_temporal_collapse_rejects_video() -> Result<()> {
        let mut settings = (*settings()).clone();
        settings.gguf_collapsed_temporal = true;
        let processor = MuseGlimmerImageProcessor {
            settings: Arc::new(settings),
            max_edge: None,
        };
        let video = VideoInput::from_frames(vec![DynamicImage::new_rgb8(1, 1)], 1.0, None);
        assert!(processor
            .preprocess_videos(vec![video], &Device::Cpu)
            .is_err());
        Ok(())
    }

    #[test]
    fn prefix_cache_invalidation_rebuilds_retained_image_pixels() -> Result<()> {
        let processor = MuseGlimmerImageProcessor {
            settings: settings(),
            max_edge: None,
        };
        let mut seq = sequence_with_image(DynamicImage::new_rgb8(1, 1));
        processor.cached_media(&mut seq, &Device::Cpu)?;
        seq.multimodal.has_changed_prompt = true;
        seq.keep_num_images(1);
        assert!(seq.multimodal.cached_pixel_values.is_none());

        let batch = processor.prepare_media_batch(&mut [&mut seq], &Device::Cpu)?;
        assert_eq!(batch.image_pixels.unwrap().dims(), &[1, 6]);
        assert_eq!(batch.image_grid.unwrap().to_vec2::<u32>()?, [[1, 1, 1]]);
        assert_eq!(batch.image_item_counts, [1]);
        Ok(())
    }
}
