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
use candle_core::{Device, IndexOp, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{
    ApplyTensorTransforms, ApplyTransforms, Normalize, TensorTransforms, ToTensor, Transforms,
};
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
    const DEFAULT_MAX_PIXELS: usize = 2048 * 2048;
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
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("HunyuanVLImageProcessor requires a specified tokenizer.");
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs
            .iter()
            .any(|seq| seq.has_images() || seq.multimodal.rope_img_grid_thw.is_some());
        let (new_input, pixel_values, image_grid_thw, continuous_img_pad) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut image_grid_thw_accum = Vec::new();
            let mut per_seq_image_grid_thw = Vec::with_capacity(input_seqs.len());
            let mut detok_seqs = tokenizer
                .decode_batch(
                    &input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    false,
                )
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            for seq in input_seqs.iter_mut() {
                if !seq.has_images() {
                    per_seq_image_grid_thw.push(seq.multimodal.rope_img_grid_thw.clone());
                    continue;
                }
                let (pixel_values, image_grid_thw) =
                    if let Some(cached_pixel_values) = &seq.multimodal.cached_pixel_values {
                        (
                            cached_pixel_values.clone(),
                            seq.multimodal.cached_img_thw.clone(),
                        )
                    } else {
                        let PreprocessedImages {
                            pixel_values,
                            image_grid_thw,
                            ..
                        } = self.preprocess(
                            seq.clone_images().expect("Need images by this point."),
                            vec![],
                            config,
                            device,
                            (usize::MAX, usize::MAX),
                        )?;
                        seq.multimodal.cached_pixel_values = Some(pixel_values.clone());
                        seq.multimodal.cached_img_thw = image_grid_thw.clone();
                        (pixel_values, image_grid_thw)
                    };
                let Some(image_grid_thw) = image_grid_thw else {
                    anyhow::bail!("HunyuanVL image sequence is missing image grid metadata.");
                };
                if seq.multimodal.rope_img_grid_thw.is_none() {
                    seq.multimodal.rope_img_grid_thw = Some(image_grid_thw.clone());
                }
                pixel_values_accum.push(pixel_values);
                image_grid_thw_accum.push(image_grid_thw.clone());
                per_seq_image_grid_thw.push(seq.multimodal.rope_img_grid_thw.clone());
            }

            if is_prompt {
                for ((text, seq_grid), seq) in detok_seqs
                    .iter_mut()
                    .zip(per_seq_image_grid_thw.iter())
                    .zip(input_seqs.iter())
                {
                    if seq.multimodal.has_changed_prompt {
                        continue;
                    }
                    let Some(seq_grid) = seq_grid else {
                        continue;
                    };
                    let mut image_idx = 0usize;
                    while text.contains(HunyuanVLProcessor::IMAGE_PAD) {
                        let grid = seq_grid.i(image_idx)?.to_vec1::<u32>()?;
                        let num_tokens = Self::image_tokens_for_grid(config, &grid);
                        *text = replace_first_occurrence(
                            text,
                            HunyuanVLProcessor::IMAGE_PAD,
                            &HunyuanVLProcessor::PLACEHOLDER.repeat(num_tokens),
                        );
                        image_idx += 1;
                    }
                    *text = text.replace(
                        HunyuanVLProcessor::PLACEHOLDER,
                        HunyuanVLProcessor::IMAGE_PAD,
                    );
                }
            }

            let img_pad = tokenizer
                .encode_fast(HunyuanVLProcessor::IMAGE_PAD, false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .get_ids()[0];
            let mut all_ids = Vec::new();
            let mut all_continuous_img_pad = Vec::new();
            for (detok, seq) in detok_seqs.into_iter().zip(input_seqs.iter_mut()) {
                let toks = tokenizer
                    .encode_fast(detok.clone(), false)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let ids = toks.get_ids().to_vec();
                if !seq.multimodal.has_changed_prompt {
                    seq.set_initial_prompt(detok);
                    if seq.mm_features().is_empty() {
                        if let (Some(hashes), Some(start_id), Some(end_id)) = (
                            seq.image_hashes().map(|h| h.to_vec()),
                            tokenizer.token_to_id(HunyuanVLProcessor::IMAGE_START),
                            tokenizer.token_to_id(HunyuanVLProcessor::IMAGE_END),
                        ) {
                            let ranges =
                                find_placeholder_delimited_ranges(&ids, img_pad, start_id, end_id);
                            let features = build_mm_features_from_ranges(
                                &ranges,
                                &hashes,
                                MultimodalKind::Image,
                            );
                            if !features.is_empty() {
                                seq.set_mm_features(features);
                            }
                        }
                    }
                    seq.set_toks_and_reallocate(ids.clone(), paged_attn_metadata.as_mut());
                    seq.multimodal.has_changed_prompt = true;
                }
                all_continuous_img_pad.push(find_sequences(&ids, img_pad));
                all_ids.push(ids);
            }

            let max_len = all_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);
            let mut all_ids_new = Vec::new();
            for mut ids in all_ids {
                ids.resize(max_len, 0);
                all_ids_new.push(Tensor::new(ids, device)?);
            }

            (
                Some(Tensor::stack(&all_ids_new, 0)?),
                if pixel_values_accum.is_empty() {
                    None
                } else {
                    Some(Tensor::cat(&pixel_values_accum, 0)?)
                },
                if image_grid_thw_accum.is_empty() {
                    None
                } else {
                    Some(Tensor::cat(&image_grid_thw_accum, 0)?)
                },
                all_continuous_img_pad,
            )
        } else {
            (None, None, None, vec![vec![]; input_seqs.len()])
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
            )?
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

        let input_ids_full = new_input.unwrap_or_else(|| input.clone());
        let seqlens = input_seqs.iter().map(|seq| seq.len()).collect::<Vec<_>>();
        let rope_img_grid_thw = {
            let grids = input_seqs
                .iter()
                .filter_map(|seq| seq.multimodal.rope_img_grid_thw.clone())
                .collect::<Vec<_>>();
            if grids.is_empty() {
                None
            } else {
                Some(Tensor::cat(&grids, 0)?)
            }
        };
        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values: if is_prompt { pixel_values } else { None },
            model_specific_args: Box::new(HunyuanVLVisionSpecificArgs {
                input_ids_full,
                image_grid_thw,
                rope_img_grid_thw,
                seqlens,
                continuous_img_pad,
            }),
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind: if is_prompt {
                crate::pipeline::RecurrentBatchKind::Prefill
            } else {
                crate::pipeline::RecurrentBatchKind::Decode
            },
        });

        Ok(InputProcessorOutput {
            inputs,
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
