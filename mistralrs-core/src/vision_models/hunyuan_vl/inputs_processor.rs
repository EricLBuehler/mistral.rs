use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::MultimodalKind,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{build_mm_features_from_ranges, find_placeholder_delimited_ranges, Sequence},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
        ModelInputs,
    },
};
use anyhow::Result;
use candle_core::{Device, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{ApplyTransforms, Normalize, Rescale, ToTensorNoNorm, Transforms};
use std::{any::Any, sync::Arc};
use tokenizers::Tokenizer;

use super::HunyuanVLVisionSpecificArgs;

struct HunyuanVLImageProcessor {
    max_edge: Option<u32>,
}

impl HunyuanVLImageProcessor {
    const DEFAULT_PATCH_SIZE: usize = 16;
    const DEFAULT_MERGE_SIZE: usize = 2;
    const DEFAULT_MIN_PIXELS: usize = 256 * 256;
    // 1.5 budgets 16M pixels for a single image; `max_edge` still caps this when set.
    const DEFAULT_MAX_PIXELS: usize = 4096 * 4096;
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    fn patch_size(config: &PreProcessorConfig) -> usize {
        config.patch_size.unwrap_or(Self::DEFAULT_PATCH_SIZE)
    }

    fn merge_size(config: &PreProcessorConfig) -> usize {
        config.merge_size.unwrap_or(Self::DEFAULT_MERGE_SIZE)
    }

    fn min_pixels(config: &PreProcessorConfig) -> usize {
        config.min_pixels.unwrap_or(Self::DEFAULT_MIN_PIXELS)
    }

    fn max_pixels(&self, config: &PreProcessorConfig) -> usize {
        let max_pixels = config.max_pixels.unwrap_or(Self::DEFAULT_MAX_PIXELS);
        if let Some(max_edge) = self.max_edge {
            max_pixels.min(max_edge as usize * max_edge as usize)
        } else {
            max_pixels
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
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
                "height:{height} or width:{width} must be larger than factor:{factor}"
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
        Ok((h_bar.max(factor), w_bar.max(factor)))
    }

    #[allow(clippy::cast_precision_loss)]
    fn preprocess_inner(
        &self,
        image: DynamicImage,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> candle_core::Result<(Tensor, (u32, u32, u32))> {
        let mut image = DynamicImage::ImageRgb8(image.to_rgb8());
        let (width, height) = image.dimensions();
        let factor = Self::patch_size(config) * Self::merge_size(config);
        let (height, width) = if config.do_resize.unwrap_or(true) {
            self.smart_resize(
                height as usize,
                width as usize,
                factor,
                Self::min_pixels(config),
                self.max_pixels(config),
            )?
        } else {
            (height as usize, width as usize)
        };
        if config.do_resize.unwrap_or(true) {
            image = image.resize_exact(
                u32::try_from(width).map_err(candle_core::Error::wrap)?,
                u32::try_from(height).map_err(candle_core::Error::wrap)?,
                config
                    .resampling
                    .map(|resample| Some(resample).to_filter())
                    .unwrap_or(Ok(FilterType::Lanczos3))?,
            );
        }

        let do_rescale = config.do_rescale.unwrap_or(true);
        let rescale_factor = config.rescale_factor.unwrap_or(1.0 / 255.0);
        let do_normalize = config.do_normalize.unwrap_or(true);
        let image_mean = config.image_mean.unwrap_or(Self::DEFAULT_MEAN);
        let image_std = config.image_std.unwrap_or(Self::DEFAULT_STD);

        let transforms = Transforms {
            input: &ToTensorNoNorm,
            inner_transforms: &[
                &do_rescale.then_some(Rescale {
                    factor: Some(rescale_factor),
                }),
                &do_normalize.then(|| Normalize {
                    mean: image_mean.to_vec(),
                    std: image_std.to_vec(),
                }),
            ],
        };
        let image = image.apply(transforms, device)?;

        let patch = Self::patch_size(config);
        let grid_h = height / patch;
        let grid_w = width / patch;
        let patches = image
            .reshape((3, grid_h, patch, grid_w, patch))?
            .permute((1, 3, 0, 2, 4))?
            .reshape((grid_h * grid_w, 3 * patch * patch))?;

        Ok((
            patches,
            (
                1,
                u32::try_from(grid_h).map_err(candle_core::Error::wrap)?,
                u32::try_from(grid_w).map_err(candle_core::Error::wrap)?,
            ),
        ))
    }

    fn image_tokens_for_grid(config: &PreProcessorConfig, grid: &[u32]) -> usize {
        let merge = u32::try_from(Self::merge_size(config)).unwrap_or(u32::MAX);
        let patch_h = grid[1] / merge;
        let patch_w = grid[2] / merge;
        (patch_h * (patch_w + 1) + 2) as usize
    }
}

pub struct HunyuanVLProcessor {
    max_edge: Option<u32>,
}

impl HunyuanVLProcessor {
    pub const IMAGE_START: &str = "<｜hy_place▁holder▁no▁100｜>";
    pub const IMAGE_END: &str = "<｜hy_place▁holder▁no▁101｜>";
    pub const IMAGE_PAD: &str = "<｜hy_place▁holder▁no▁102｜>";
    pub const PLACEHOLDER: &str = "<|placeholder|>";

    pub fn new(max_edge: Option<u32>) -> Self {
        Self { max_edge }
    }
}

impl Processor for HunyuanVLProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(HunyuanVLImageProcessor {
            max_edge: self.max_edge,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[Self::IMAGE_PAD, Self::PLACEHOLDER]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
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

impl InputsProcessor for HunyuanVLImageProcessor {
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
        let tokenizer = tokenizer.expect("HunyuanVL requires a tokenizer");
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        for seq in input_seqs {
            if !seq.has_images() || seq.multimodal.has_changed_prompt {
                continue;
            }
            if seq.multimodal.cached_pixel_values.is_none() {
                let processed = self.preprocess(
                    seq.clone_images().expect("Need images by this point."),
                    vec![],
                    config,
                    device,
                    (usize::MAX, usize::MAX),
                )?;
                seq.multimodal.cached_pixel_values = Some(processed.pixel_values);
                seq.multimodal.cached_img_thw = processed.image_grid_thw;
            }
            let grid = seq
                .multimodal
                .cached_img_thw
                .as_ref()
                .expect("Missing image grid");
            seq.multimodal.rope_img_grid_thw = Some(grid.clone());
            let grids = grid.to_vec2::<u32>()?;
            let mut text = tokenizer
                .decode(seq.get_toks(), false)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            let mut image = 0;
            while text.contains(HunyuanVLProcessor::IMAGE_PAD) {
                let tokens = Self::image_tokens_for_grid(config, &grids[image]);
                text = replace_first_occurrence(
                    &text,
                    HunyuanVLProcessor::IMAGE_PAD,
                    &HunyuanVLProcessor::PLACEHOLDER.repeat(tokens),
                );
                image += 1;
            }
            text = text.replace(
                HunyuanVLProcessor::PLACEHOLDER,
                HunyuanVLProcessor::IMAGE_PAD,
            );
            let ids = tokenizer
                .encode_fast(text.clone(), false)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
                .get_ids()
                .to_vec();
            let image_pad = tokenizer
                .token_to_id(HunyuanVLProcessor::IMAGE_PAD)
                .unwrap();
            let start = tokenizer
                .token_to_id(HunyuanVLProcessor::IMAGE_START)
                .unwrap();
            let end = tokenizer
                .token_to_id(HunyuanVLProcessor::IMAGE_END)
                .unwrap();
            let ranges = find_placeholder_delimited_ranges(&ids, image_pad, start, end);
            let hashes = seq.image_hashes().expect("Missing image hashes");
            if ranges.len() != hashes.len() {
                anyhow::bail!("HunyuanVL image placeholders do not match the supplied images");
            }
            let features = build_mm_features_from_ranges(&ranges, hashes, MultimodalKind::Image);
            seq.set_mm_features(features);
            seq.set_initial_prompt(text);
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
            anyhow::bail!("Cannot make inputs for X-LoRA vision model.");
        }
        if no_kv_cache {
            anyhow::bail!("Vision model must have kv cache.");
        }
        if is_prompt {
            self.prepare_for_paged_prompt_planning(
                tokenizer.clone(),
                input_seqs,
                device,
                other_config.clone(),
                paged_attn_metadata.as_mut(),
            )?;
        }
        let tokenizer = tokenizer.expect("HunyuanVL requires a tokenizer");
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
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
        let seq_len = input.dim(1)?;
        let image_pad = tokenizer
            .token_to_id(HunyuanVLProcessor::IMAGE_PAD)
            .unwrap();
        let mut pixels = Vec::new();
        let mut selected_grids = Vec::new();
        let mut spans = Vec::new();
        let mut rope_positions = Vec::new();
        for (seq, &offset) in input_seqs.iter().zip(&positions) {
            let full_spans = if is_prompt {
                find_sequences(seq.prompt_position_source_toks(), image_pad)
            } else {
                Vec::new()
            };
            let grids = if is_prompt {
                seq.multimodal
                    .cached_img_thw
                    .as_ref()
                    .map(Tensor::to_vec2::<u32>)
                    .transpose()?
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let layout = super::layout::ImageChunkLayout::new(
                &full_spans,
                &grids,
                offset..offset + seq_len,
                Self::merge_size(config),
            )?;
            if !layout.images.is_empty() {
                let patch_start = grids[..layout.images.start]
                    .iter()
                    .map(|grid| grid.iter().map(|n| *n as usize).product::<usize>())
                    .sum();
                let patch_len = grids[layout.images.clone()]
                    .iter()
                    .map(|grid| grid.iter().map(|n| *n as usize).product::<usize>())
                    .sum();
                pixels.push(
                    seq.multimodal
                        .cached_pixel_values
                        .as_ref()
                        .expect("Missing image pixels")
                        .narrow(0, patch_start, patch_len)?,
                );
                selected_grids.push(seq.multimodal.cached_img_thw.as_ref().unwrap().narrow(
                    0,
                    layout.images.start,
                    layout.images.len(),
                )?);
            }
            rope_positions.push(Tensor::from_vec(
                layout.positions.into_iter().flatten().collect::<Vec<_>>(),
                (super::layout::POSITION_PLANES, 1, seq_len),
                device,
            )?);
            spans.push(layout.spans);
        }
        let pixel_values = (!pixels.is_empty())
            .then(|| Tensor::cat(&pixels, 0))
            .transpose()?;
        let image_grid_thw = (!selected_grids.is_empty())
            .then(|| Tensor::cat(&selected_grids, 0))
            .transpose()?;
        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values,
                model_specific_args: Box::new(HunyuanVLVisionSpecificArgs {
                    position_ids: Some(Tensor::cat(&rope_positions, 1)?),
                    image_grid_thw,
                    continuous_img_pad: spans,
                }),
                paged_attn_meta,
                flash_meta,
                recurrent_batch_kind: if is_prompt {
                    crate::pipeline::RecurrentBatchKind::Prefill
                } else {
                    crate::pipeline::RecurrentBatchKind::Decode
                },
                adapter_leases: crate::vision_models::adapter_leases(input_seqs, &seq_indices),
            }),
            seq_indices,
        })
    }
}

impl ImagePreProcessor for HunyuanVLImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = Self::DEFAULT_MEAN;
    const DEFAULT_STD: [f64; 3] = Self::DEFAULT_STD;

    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        _bs: (usize, usize),
    ) -> candle_core::Result<PreprocessedImages> {
        if !videos.is_empty() {
            candle_core::bail!("HunyuanVL video inputs are not supported yet.");
        }
        let mut pixel_values = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len());
        for image in images.drain(..) {
            let (pixels, grid) = self.preprocess_inner(image, config, device)?;
            pixel_values.push(pixels);
            grids.push(Tensor::new(&[grid.0, grid.1, grid.2], device)?);
        }
        Ok(PreprocessedImages {
            pixel_values: Tensor::cat(&pixel_values, 0)?,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: Some(Tensor::stack(&grids, 0)?),
            video_grid_thw: None,
            rows: None,
            cols: None,
            pixel_values_list: None,
            tgt_sizes: None,
            image_sizes_all: None,
            num_crops: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREPROCESSOR: &str = r#"{
        "do_convert_rgb": true,
        "do_normalize": true,
        "do_rescale": true,
        "do_resize": true,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "max_pixels": 16777216,
        "merge_size": 2,
        "min_pixels": 262144,
        "patch_size": 16,
        "resample": 1,
        "rescale_factor": 0.00392156862745098,
        "size": {"longest_edge": 4194304, "shortest_edge": 262144},
        "temporal_patch_size": 1
    }"#;

    fn config() -> PreProcessorConfig {
        PreProcessorConfig::from_processor_config_json(PREPROCESSOR).unwrap()
    }

    fn processing_factor(config: &PreProcessorConfig) -> usize {
        HunyuanVLImageProcessor::patch_size(config) * HunyuanVLImageProcessor::merge_size(config)
    }

    #[test]
    fn checkpoint_max_pixels_overrides_the_stale_size_entry() {
        let config = config();
        let processor = HunyuanVLImageProcessor { max_edge: None };
        assert_eq!(processor.max_pixels(&config), 16 * 1024 * 1024);
        assert_eq!(HunyuanVLImageProcessor::min_pixels(&config), 512 * 512);
        assert_eq!(processing_factor(&config), 32);
    }

    #[test]
    fn four_k_square_images_keep_the_full_budget() {
        let config = config();
        let processor = HunyuanVLImageProcessor { max_edge: None };
        let max_pixels = processor.max_pixels(&config);
        let min_pixels = HunyuanVLImageProcessor::min_pixels(&config);
        let factor = processing_factor(&config);
        assert_eq!(
            processor
                .smart_resize(4096, 4096, factor, min_pixels, max_pixels)
                .unwrap(),
            (4096, 4096)
        );
        assert_eq!(
            processor
                .smart_resize(8000, 8000, factor, min_pixels, max_pixels)
                .unwrap(),
            (4096, 4096)
        );
        let (height, width) = processor
            .smart_resize(9000, 7000, factor, min_pixels, max_pixels)
            .unwrap();
        assert!(height % factor == 0 && width % factor == 0);
        assert!(height * width <= max_pixels);
    }

    #[test]
    fn explicit_max_edge_caps_the_budget() {
        let config = config();
        let processor = HunyuanVLImageProcessor {
            max_edge: Some(2048),
        };
        assert_eq!(processor.max_pixels(&config), 2048 * 2048);
        assert_eq!(
            processor
                .smart_resize(
                    8000,
                    8000,
                    processing_factor(&config),
                    512 * 512,
                    processor.max_pixels(&config)
                )
                .unwrap(),
            (2048, 2048)
        );
    }

    #[test]
    fn image_token_count_follows_the_merged_grid() {
        let config = config();
        assert_eq!(
            HunyuanVLImageProcessor::image_tokens_for_grid(&config, &[1, 256, 256]),
            16_514
        );
        assert_eq!(
            HunyuanVLImageProcessor::image_tokens_for_grid(&config, &[1, 64, 64]),
            1058
        );
    }

    const SOLID_RGB: [u8; 3] = [10, 20, 30];
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];
    const CONFIG_RESCALE_FACTOR: f64 = 0.003_921_568_627_450_98;

    fn solid_image(rgb: [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(32, 32, image::Rgb(rgb)))
    }

    fn pixel_config(do_rescale: bool, do_normalize: bool) -> PreProcessorConfig {
        let mut config = config();
        config.do_resize = Some(false);
        config.do_rescale = Some(do_rescale);
        config.do_normalize = Some(do_normalize);
        config.rescale_factor = Some(CONFIG_RESCALE_FACTOR);
        config.image_mean = Some(DEFAULT_MEAN);
        config.image_std = Some(DEFAULT_STD);
        config
    }

    fn expected_channels(
        rgb: [u8; 3],
        do_rescale: bool,
        rescale_factor: f64,
        do_normalize: bool,
        image_mean: [f64; 3],
        image_std: [f64; 3],
    ) -> [f64; 3] {
        let mut expected = [0f64; 3];
        for (channel, slot) in expected.iter_mut().enumerate() {
            let mut value = f64::from(rgb[channel]);
            if do_rescale {
                value *= rescale_factor;
            }
            if do_normalize {
                value = (value - image_mean[channel]) / image_std[channel];
            }
            *slot = value;
        }
        expected
    }

    fn assert_solid_pixels(config: &PreProcessorConfig, expected: [f64; 3]) {
        let processor = HunyuanVLImageProcessor { max_edge: None };
        let preprocessed = processor
            .preprocess(
                vec![solid_image(SOLID_RGB)],
                Vec::new(),
                config,
                &Device::Cpu,
                (0, 1),
            )
            .unwrap();
        let grid = preprocessed
            .image_grid_thw
            .as_ref()
            .unwrap()
            .to_vec2::<u32>()
            .unwrap();
        assert_eq!(grid, vec![vec![1, 2, 2]]);
        let patches = preprocessed.pixel_values.to_vec2::<f32>().unwrap();
        assert_eq!(patches.len(), 4);
        for row in &patches {
            assert_eq!(row.len(), 3 * 16 * 16);
            for (channel, expected_value) in expected.iter().enumerate() {
                for index in 0..16 * 16 {
                    let actual = row[channel * 16 * 16 + index];
                    assert!(
                        (f64::from(actual) - expected_value).abs() <= 1e-4,
                        "channel {channel} element {index}: {actual} != {expected_value}"
                    );
                }
            }
        }
    }

    #[test]
    fn default_config_keeps_the_previous_normalized_pixels() {
        use mistralrs_vision::ImageTransform;
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_fn(32, 32, |x, y| {
            let value = (y * 32 + x).to_le_bytes()[0];
            image::Rgb([value, value.wrapping_add(85), value.wrapping_add(170)])
        }));
        let legacy = mistralrs_vision::ToTensor
            .map(&image, &Device::Cpu)
            .unwrap();
        let legacy = Normalize {
            mean: DEFAULT_MEAN.to_vec(),
            std: DEFAULT_STD.to_vec(),
        }
        .map(&legacy, &Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        for explicit_defaults in [false, true] {
            let mut config = pixel_config(true, true);
            if !explicit_defaults {
                config.do_rescale = None;
                config.rescale_factor = None;
                config.do_normalize = None;
                config.image_mean = None;
                config.image_std = None;
            }
            let processor = HunyuanVLImageProcessor { max_edge: None };
            let preprocessed = processor
                .preprocess(
                    vec![image.clone()],
                    Vec::new(),
                    &config,
                    &Device::Cpu,
                    (0, 1),
                )
                .unwrap();
            let pixels = preprocessed
                .pixel_values
                .reshape((2, 2, 3, 16, 16))
                .unwrap()
                .permute((2, 0, 3, 1, 4))
                .unwrap()
                .contiguous()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(pixels, legacy);
        }
    }

    #[test]
    fn rescale_and_normalize_switches_follow_the_config() {
        for (do_rescale, do_normalize) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            let config = pixel_config(do_rescale, do_normalize);
            assert_solid_pixels(
                &config,
                expected_channels(
                    SOLID_RGB,
                    do_rescale,
                    CONFIG_RESCALE_FACTOR,
                    do_normalize,
                    DEFAULT_MEAN,
                    DEFAULT_STD,
                ),
            );
        }
    }

    #[test]
    fn custom_rescale_factor_and_mean_std_are_used() {
        let mut rescaled = pixel_config(true, false);
        rescaled.rescale_factor = Some(0.5);
        assert_solid_pixels(
            &rescaled,
            expected_channels(SOLID_RGB, true, 0.5, false, [0.0; 3], [1.0; 3]),
        );

        let mut normalized = pixel_config(false, true);
        let mean = [1.0, 2.0, 3.0];
        let std = [2.0, 4.0, 5.0];
        normalized.image_mean = Some(mean);
        normalized.image_std = Some(std);
        assert_solid_pixels(
            &normalized,
            expected_channels(SOLID_RGB, false, 1.0, true, mean, std),
        );
    }

    #[test]
    fn disabled_switches_ignore_their_parameters() {
        // Normalize off: a zero std would produce inf/nan if it were still applied.
        let mut no_normalize = pixel_config(true, false);
        no_normalize.image_mean = Some([100.0; 3]);
        no_normalize.image_std = Some([0.0; 3]);
        assert_solid_pixels(
            &no_normalize,
            expected_channels(
                SOLID_RGB,
                true,
                CONFIG_RESCALE_FACTOR,
                false,
                [100.0; 3],
                [0.0; 3],
            ),
        );

        // Rescale off: a zero factor would blank the image if it were still applied.
        let mut no_rescale = pixel_config(false, true);
        no_rescale.rescale_factor = Some(0.0);
        let mean = [1.0, 2.0, 3.0];
        let std = [2.0, 4.0, 5.0];
        no_rescale.image_mean = Some(mean);
        no_rescale.image_std = Some(std);
        assert_solid_pixels(
            &no_rescale,
            expected_channels(SOLID_RGB, false, 0.0, true, mean, std),
        );
    }

    #[test]
    fn resize_and_patch_packing_are_untouched() {
        let processor = HunyuanVLImageProcessor { max_edge: None };
        let preprocessed = processor
            .preprocess(
                vec![solid_image(SOLID_RGB)],
                Vec::new(),
                &config(),
                &Device::Cpu,
                (0, 1),
            )
            .unwrap();
        let grid = preprocessed
            .image_grid_thw
            .unwrap()
            .to_vec2::<u32>()
            .unwrap();
        assert_eq!(grid, vec![vec![1, 32, 32]]);
        assert_eq!(preprocessed.pixel_values.dims(), &[1024, 768]);
    }
}
