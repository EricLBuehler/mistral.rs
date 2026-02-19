use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{build_mm_features_from_ranges, find_image_placeholder_ranges, Sequence},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
        ModelInputs,
    },
};
use anyhow::Result;
use candle_core::{Context, Device, IndexOp, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{
    ApplyTensorTransforms, ApplyTransforms, Normalize, TensorTransforms, ToTensor, Transforms,
};
use std::{any::Any, sync::Arc};
use tokenizers::Tokenizer;

use super::Qwen2_5VLVisionSpecificArgs;

// Input processor
struct Qwen2_5VLImageProcessor {
    max_edge: Option<u32>,
}
// Processor
pub struct Qwen2_5VLProcessor {
    max_edge: Option<u32>,
}

impl Qwen2_5VLProcessor {
    pub const VISION_START: &str = "<|vision_start|>";
    pub const VISION_END: &str = "<|vision_end|>";
    pub const IMAGE_PAD: &str = "<|image_pad|>";
    pub const VIDEO_PAD: &str = "<|video_pad|>";
    pub const PLACEHOLDER: &str = "<|placeholder|>";

    pub fn new(max_edge: Option<u32>) -> Self {
        Self { max_edge }
    }
}

impl Processor for Qwen2_5VLProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Qwen2_5VLImageProcessor {
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

// index + needle length
fn find_substring_indices(haystack: &str, needle: &str) -> Vec<usize> {
    let mut indices = Vec::new();
    let mut start = 0;

    while let Some(pos) = haystack[start..].find(needle) {
        let index = start + pos;
        indices.push(index + needle.len());
        start = index + needle.len(); // Move past the last found occurrence
    }

    indices
}

impl InputsProcessor for Qwen2_5VLImageProcessor {
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

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().all(|seq| seq.has_images());

        let (
            new_input,
            pixel_values,
            mut image_grid_thw,
            mut video_grid_thw,
            mut continuous_img_pad,
            mut continuous_vid_pad,
            input_ids_searching,
            image_nums,
            video_nums,
        ) = if has_images {
            let mut pixel_values_accum = Vec::new();
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

            for seq in input_seqs.iter_mut() {
                let (pixel_values, image_grid_thw, video_grid_thw) =
                    if let Some(cached_pixel_values) = &seq.multimodal.cached_pixel_values {
                        (
                            cached_pixel_values.clone(),
                            seq.multimodal.cached_img_thw.clone(),
                            seq.multimodal.cached_vid_thw.clone(),
                        )
                    } else {
                        let PreprocessedImages {
                            pixel_values,
                            pixel_attention_mask: _,
                            image_sizes: _,
                            num_img_tokens: _,
                            aspect_ratio_ids: _,
                            aspect_ratio_mask: _,
                            num_tiles: _,
                            image_grid_thw,
                            video_grid_thw,
                            rows: _,
                            cols: _,
                            pixel_values_list: _,
                            tgt_sizes: _,
                            image_sizes_all: _,
                            num_crops: _,
                        } = self
                            .preprocess(
                                seq.clone_images()
                                    .expect("Need to have images by this point."),
                                vec![],
                                config,
                                device,
                                (usize::MAX, usize::MAX), // Don't use it here...
                            )
                            .expect("Preprocessing failed");

                        seq.multimodal.cached_pixel_values = Some(pixel_values.clone());
                        seq.multimodal.cached_img_thw = image_grid_thw.clone();
                        seq.multimodal.cached_vid_thw = video_grid_thw.clone();
                        (pixel_values, image_grid_thw, video_grid_thw)
                    };

                pixel_values_accum.push(pixel_values.unsqueeze(0).unwrap());
                image_grid_thw_accum.push(image_grid_thw);
                video_grid_thw_accum.push(video_grid_thw);
            }

            // Cache the complete grid_thw for MRoPE position computation.
            // Set once during the first inputs processor call when ALL images are present.
            // Unlike cached_img_thw, this is never cleared by keep_num_images, so it
            // remains valid even after prefix caching trims the image set.
            for (idx, seq) in input_seqs.iter_mut().enumerate() {
                if seq.multimodal.rope_img_grid_thw.is_none() {
                    seq.multimodal.rope_img_grid_thw = image_grid_thw_accum[idx].clone();
                }
                if seq.multimodal.rope_vid_grid_thw.is_none() {
                    seq.multimodal.rope_vid_grid_thw = video_grid_thw_accum[idx].clone();
                }
            }

            let image_grid_thw_accum = if image_grid_thw_accum.iter().any(|img| img.is_none()) {
                None
            } else {
                Some(
                    image_grid_thw_accum
                        .into_iter()
                        .map(|img| img.unwrap())
                        .collect::<Vec<_>>(),
                )
            };

            let video_grid_thw_accum = if video_grid_thw_accum.iter().any(|img| img.is_none()) {
                None
            } else {
                Some(
                    video_grid_thw_accum
                        .into_iter()
                        .map(|img| img.unwrap())
                        .collect::<Vec<_>>(),
                )
            };

            if is_prompt {
                if let Some(ref image_grid_thw_accum) = image_grid_thw_accum {
                    let merge_length = config.merge_size.expect("Require `merge_size").pow(2);
                    for ((batch, text), seq) in
                        detok_seqs.iter_mut().enumerate().zip(input_seqs.iter_mut())
                    {
                        if seq.multimodal.has_changed_prompt {
                            continue;
                        }
                        let mut index = 0;
                        while text.contains(Qwen2_5VLProcessor::IMAGE_PAD) {
                            *text = replace_first_occurrence(
                                text,
                                Qwen2_5VLProcessor::IMAGE_PAD,
                                &Qwen2_5VLProcessor::PLACEHOLDER.repeat(
                                    image_grid_thw_accum[batch]
                                        .i(index)
                                        .unwrap()
                                        .to_vec1::<u32>()
                                        .unwrap()
                                        .iter()
                                        .product::<u32>()
                                        as usize
                                        / merge_length,
                                ),
                            );
                            index += 1;
                        }
                        *text = text.replace(
                            Qwen2_5VLProcessor::PLACEHOLDER,
                            Qwen2_5VLProcessor::IMAGE_PAD,
                        );
                    }
                }

                if let Some(ref video_grid_thw_accum) = video_grid_thw_accum {
                    let merge_length = config.merge_size.expect("Require `merge_size").pow(2);
                    let mut index = 0;
                    for ((batch, text), seq) in
                        detok_seqs.iter_mut().enumerate().zip(input_seqs.iter_mut())
                    {
                        if seq.multimodal.has_changed_prompt {
                            continue;
                        }
                        while text.contains(Qwen2_5VLProcessor::VIDEO_PAD) {
                            *text = replace_first_occurrence(
                                text,
                                Qwen2_5VLProcessor::VIDEO_PAD,
                                &Qwen2_5VLProcessor::PLACEHOLDER.repeat(
                                    video_grid_thw_accum[batch]
                                        .i(index)
                                        .unwrap()
                                        .to_vec1::<u32>()
                                        .unwrap()
                                        .iter()
                                        .product::<u32>()
                                        as usize
                                        / merge_length,
                                ),
                            );
                            index += 1;
                        }
                        *text = text.replace(
                            Qwen2_5VLProcessor::PLACEHOLDER,
                            Qwen2_5VLProcessor::VIDEO_PAD,
                        );
                    }
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

                    // Build mm_features for position-aware prefix cache hashing
                    if seq.mm_features().is_empty() {
                        if let (Some(hashes), Some(img_pad_id)) = (
                            seq.image_hashes().map(|h| h.to_vec()),
                            tokenizer.token_to_id(Qwen2_5VLProcessor::IMAGE_PAD),
                        ) {
                            let ranges = find_image_placeholder_ranges(&ids, img_pad_id);
                            seq.set_mm_features(build_mm_features_from_ranges(
                                &ranges, &hashes, "img",
                            ));
                        }
                    }

                    seq.set_toks_and_reallocate(ids.clone(), paged_attn_metadata.as_mut());
                    seq.multimodal.has_changed_prompt = true;
                }
                all_ids.push(ids.clone());

                let img_pad = tokenizer
                    .encode_fast(Qwen2_5VLProcessor::IMAGE_PAD, false)
                    .expect("Detokenization failed!")
                    .get_ids()
                    .to_vec();
                let continuous_img_pad = find_sequences(&ids, img_pad[0]);
                all_continuous_img_pad.push(continuous_img_pad);

                let vid_pad = tokenizer
                    .encode_fast(Qwen2_5VLProcessor::VIDEO_PAD, false)
                    .expect("Detokenization failed!")
                    .get_ids()
                    .to_vec();
                let continuous_vid_pad = find_sequences(&ids, vid_pad[0]);
                all_continuous_vid_pad.push(continuous_vid_pad);
            }

            let mut input_ids_searching = Vec::new();
            let mut image_nums = Vec::new();
            let mut video_nums = Vec::new();
            for (seq, ids) in input_seqs.iter().zip(&all_ids) {
                let prompt = seq.get_initial_prompt();
                let match_indices =
                    find_substring_indices(prompt, Qwen2_5VLProcessor::VISION_START);
                image_nums.push(
                    match_indices
                        .iter()
                        .filter(|&&idx| {
                            prompt[idx..idx + Qwen2_5VLProcessor::IMAGE_PAD.len()]
                                == *Qwen2_5VLProcessor::IMAGE_PAD
                        })
                        .count(),
                );
                video_nums.push(
                    match_indices
                        .iter()
                        .filter(|&&idx| {
                            prompt[idx..idx + Qwen2_5VLProcessor::VIDEO_PAD.len()]
                                == *Qwen2_5VLProcessor::VIDEO_PAD
                        })
                        .count(),
                );

                input_ids_searching.push(ids.to_vec());
            }

            let mut all_ids_new = Vec::new();
            let max_len = all_ids.iter().map(|ids| ids.len()).max().unwrap();
            for ids in all_ids {
                let pad = max_len - ids.len();
                all_ids_new.push(Tensor::new([ids, vec![0; pad]].concat(), device).unwrap());
            }

            (
                Some(Tensor::stack(&all_ids_new, 0).unwrap()),
                Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                image_grid_thw_accum.map(|img| Tensor::cat(&img, 0).unwrap()),
                video_grid_thw_accum.map(|vid| Tensor::cat(&vid, 0).unwrap()),
                all_continuous_img_pad,
                all_continuous_vid_pad,
                input_ids_searching,
                image_nums,
                video_nums,
            )
        } else {
            (
                None,
                None,
                None,
                None,
                vec![],
                vec![],
                vec![vec![]; input_seqs.len()],
                vec![0; input_seqs.len()],
                vec![0; input_seqs.len()],
            )
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
            )
            .unwrap()
        };

        let (input, input_ids_full) = match (new_input, is_prompt) {
            (Some(new_input), true) => (input, new_input),
            (Some(new_input), false) => (input, new_input),
            (None, _) => (input.clone(), input.clone()),
        };

        let mut pixel_values = if is_prompt { pixel_values } else { None };

        // Adjust continuous pad ranges for prefix caching: drop cached ranges, shift new ones.
        // Also trim pixel_values and grid_thw to exclude cached images/videos so the vision
        // encoder only produces embeddings for the non-cached ones.
        let mut per_seq_cached_images: Vec<usize> = vec![0; input_seqs.len()];
        if is_prompt {
            let mut total_cached_images = 0usize;
            let mut total_cached_videos = 0usize;
            for (seq_idx, (seq, (img_pads, vid_pads))) in input_seqs
                .iter()
                .zip(
                    continuous_img_pad
                        .iter_mut()
                        .zip(continuous_vid_pad.iter_mut()),
                )
                .enumerate()
            {
                let prefix_len = seq.prefix_cache_len();
                if prefix_len > 0 {
                    let img_before = img_pads.len();
                    img_pads.retain(|(start, _)| *start >= prefix_len);
                    let cached = img_before - img_pads.len();
                    total_cached_images += cached;
                    per_seq_cached_images[seq_idx] = cached;
                    for (start, end) in img_pads.iter_mut() {
                        *start -= prefix_len;
                        *end -= prefix_len;
                    }
                    let vid_before = vid_pads.len();
                    vid_pads.retain(|(start, _)| *start >= prefix_len);
                    total_cached_videos += vid_before - vid_pads.len();
                    for (start, end) in vid_pads.iter_mut() {
                        *start -= prefix_len;
                        *end -= prefix_len;
                    }
                }
            }
            if total_cached_images > 0 {
                let n_seqs = input_seqs.len().max(1);
                let per_seq_cached = total_cached_images / n_seqs;
                if let Some(ref grid) = image_grid_thw {
                    let total_grid = grid.dim(0).unwrap();
                    let grid_per_seq = total_grid / n_seqs;
                    let remaining_per_seq = grid_per_seq.saturating_sub(per_seq_cached);
                    if remaining_per_seq > 0 {
                        let trimmed: Vec<Tensor> = (0..n_seqs)
                            .map(|i| {
                                grid.narrow(0, i * grid_per_seq + per_seq_cached, remaining_per_seq)
                                    .unwrap()
                            })
                            .collect();
                        image_grid_thw = Some(Tensor::cat(&trimmed, 0).unwrap());
                    } else {
                        image_grid_thw = None;
                    }
                }
                if let Some(ref pv) = pixel_values {
                    let n_imgs = pv.dim(1).unwrap();
                    let remaining = n_imgs.saturating_sub(per_seq_cached);
                    if remaining > 0 {
                        pixel_values = Some(pv.narrow(1, per_seq_cached, remaining).unwrap());
                    } else {
                        pixel_values = None;
                    }
                }
            }
            if total_cached_videos > 0 {
                let n_seqs = input_seqs.len().max(1);
                let per_seq_cached_vids = total_cached_videos / n_seqs;
                if let Some(ref grid) = video_grid_thw {
                    let total_grid = grid.dim(0).unwrap();
                    let grid_per_seq = total_grid / n_seqs;
                    let remaining_per_seq = grid_per_seq.saturating_sub(per_seq_cached_vids);
                    if remaining_per_seq > 0 {
                        let trimmed: Vec<Tensor> = (0..n_seqs)
                            .map(|i| {
                                grid.narrow(
                                    0,
                                    i * grid_per_seq + per_seq_cached_vids,
                                    remaining_per_seq,
                                )
                                .unwrap()
                            })
                            .collect();
                        video_grid_thw = Some(Tensor::cat(&trimmed, 0).unwrap());
                    } else {
                        video_grid_thw = None;
                    }
                }
            }
        }

        let seqlens = input_seqs.iter().map(|seq| seq.len()).collect::<Vec<_>>();

        // Collect the complete rope grids from per-sequence cached values.
        // These cover ALL images/videos in the full sequence (including prefix-cached ones)
        // and are used for MRoPE position computation in get_rope_index.
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

        let image_hashes: Vec<u64> = if is_prompt {
            input_seqs
                .iter()
                .enumerate()
                .flat_map(|(seq_idx, seq)| {
                    seq.image_hashes()
                        .map(|h| {
                            let cached = per_seq_cached_images[seq_idx];
                            if cached < h.len() {
                                h[cached..].to_vec()
                            } else {
                                vec![]
                            }
                        })
                        .unwrap_or_default()
                })
                .collect()
        } else {
            vec![]
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(Qwen2_5VLVisionSpecificArgs {
                input_ids_full,
                image_grid_thw,
                video_grid_thw,
                rope_img_grid_thw,
                rope_vid_grid_thw,
                seqlens,
                continuous_img_pad,
                continuous_vid_pad,
                input_ids_searching,
                image_nums,
                video_nums,
                image_hashes,
            }),
            paged_attn_meta,
            flash_meta,
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }
}

impl Qwen2_5VLImageProcessor {
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
                height,
                width,
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

        let mut patches = Tensor::stack(&processed_images, 0)?;
        let temporal_patch_size = config
            .temporal_patch_size
            .context("Require `temporal_patch_size")?;
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

impl ImagePreProcessor for Qwen2_5VLImageProcessor {
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

            let mut height = 0;
            let mut width = 0;
            for image in &images {
                let (w, h) = image.dimensions();
                if w > width {
                    width = w;
                }
                if h > height {
                    height = h;
                }
            }

            for image in images {
                let (patches, (t, h, w)) =
                    self.preprocess_inner(vec![image], config, device, (height, width))?;
                pixel_values.push(patches);
                vision_grid_thw.push(Tensor::new(&[t, h, w], &Device::Cpu)?);
            }
            let pixel_values = Tensor::stack(&pixel_values, 0)?;
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
            let mut height = 0;
            let mut width = 0;
            for image in &videos {
                let (w, h) = image[0].dimensions();
                if w > width {
                    width = w;
                }
                if h > height {
                    height = h;
                }
            }

            for images in videos {
                let (patches, (t, h, w)) =
                    self.preprocess_inner(images, config, device, (height, width))?;
                pixel_values.push(patches);
                vision_grid_thw.push(Tensor::new(&[t, h, w], &Device::Cpu)?);
            }
            let pixel_values = Tensor::stack(&pixel_values, 0)?;
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
