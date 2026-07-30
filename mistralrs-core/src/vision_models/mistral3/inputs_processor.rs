#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::paged_attention::block_hash::{MultimodalAttentionPolicy, MultimodalKind};
use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};
use mistralrs_vision::{ApplyTransforms, Normalize, Rescale, ToTensorNoNorm, Transforms};
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{build_mm_features_from_ranges, Sequence},
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

use super::Mistral3SpecificArgs;

fn find_mistral3_image_ranges(
    tokens: &[u32],
    image_token_id: u32,
    image_end_token_id: u32,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut position = 0usize;
    while position < tokens.len() {
        if tokens[position] != image_token_id {
            position += 1;
            continue;
        }
        let start = position;
        while position < tokens.len() && tokens[position] != image_end_token_id {
            position += 1;
        }
        if position < tokens.len() {
            ranges.push((start, position - start + 1));
        }
        position += 1;
    }
    ranges
}

fn cat_padded_mistral3_images(tensors: &[Tensor]) -> Result<Tensor> {
    if tensors.is_empty() {
        candle_core::bail!("Mistral 3 image tensor batch cannot be empty");
    }
    let shapes = tensors
        .iter()
        .map(Tensor::dims4)
        .collect::<Result<Vec<_>>>()?;
    let max_height = shapes.iter().map(|shape| shape.2).max().unwrap_or(0);
    let max_width = shapes.iter().map(|shape| shape.3).max().unwrap_or(0);
    let padded = tensors
        .iter()
        .zip(shapes)
        .map(|(tensor, (_, _, height, width))| {
            tensor
                .pad_with_zeros(2, 0, max_height - height)?
                .pad_with_zeros(3, 0, max_width - width)
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&padded, 0)
}

struct Mistral3ImageProcessor {
    image_break_token: String,
    image_end_token: String,
    image_token: String,
    patch_size: usize,
    spatial_merge_size: usize,
}

pub struct Mistral3Processor {
    image_break_token: String,
    image_end_token: String,
    image_token: String,
    patch_size: usize,
    spatial_merge_size: usize,
}

fn mistral3_packed_layout(
    input_seqs: &[&mut Sequence],
    query_lens: &[usize],
    image_sizes_by_sequence: &[Vec<(u32, u32)>],
    image_token_id: u32,
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<PackedMultimodalLayout> {
    if input_seqs.len() != query_lens.len() || input_seqs.len() != image_sizes_by_sequence.len() {
        candle_core::bail!("Mistral 3 packed multimodal metadata length mismatch");
    }
    let mut requests = Vec::with_capacity(input_seqs.len());
    for ((seq, &query_len), image_sizes) in input_seqs
        .iter()
        .zip(query_lens)
        .zip(image_sizes_by_sequence)
    {
        let tokens = seq.get_toks();
        if query_len != tokens.len() {
            candle_core::bail!(
                "Mistral 3 packed multimodal prefill requires the complete uncached prompt"
            );
        }
        let hashes = seq.image_hashes().unwrap_or_default();
        if hashes.len() != image_sizes.len() {
            candle_core::bail!(
                "Mistral 3 sequence has {} image hashes but {} image sizes",
                hashes.len(),
                image_sizes.len()
            );
        }
        let destinations = tokens
            .iter()
            .enumerate()
            .filter_map(|(position, token)| (*token == image_token_id).then_some(position))
            .collect::<Vec<_>>();
        let mut destination_offset = 0usize;
        let mut items = Vec::with_capacity(hashes.len());
        for (item_index, (&hash, &(height, width))) in hashes.iter().zip(image_sizes).enumerate() {
            let height_tokens = height as usize / (patch_size * spatial_merge_size);
            let width_tokens = width as usize / (patch_size * spatial_merge_size);
            let token_count = height_tokens * width_tokens;
            let end = destination_offset
                .checked_add(token_count)
                .ok_or_else(|| candle_core::Error::msg("Mistral 3 image token count overflow"))?;
            let item_destinations = destinations
                .get(destination_offset..end)
                .ok_or_else(|| {
                    candle_core::Error::msg(
                        "Mistral 3 image placeholders do not match encoder output size",
                    )
                })?
                .to_vec();
            let placeholder_start = *item_destinations.first().ok_or_else(|| {
                candle_core::Error::msg("Mistral 3 image has no placeholder tokens")
            })?;
            let placeholder_end = item_destinations
                .last()
                .and_then(|position| position.checked_add(1))
                .ok_or_else(|| candle_core::Error::msg("Mistral 3 placeholder range overflow"))?;
            items.push(MultimodalItemLayout::new(
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash,
                },
                item_index,
                placeholder_start..placeholder_end,
                MultimodalAttentionPolicy::Causal,
                vec![MultimodalEmbeddingMap::new(
                    item_destinations,
                    (0..token_count).collect(),
                    0,
                )?],
            )?);
            destination_offset = end;
        }
        if destination_offset != destinations.len() {
            candle_core::bail!(
                "Mistral 3 sequence has {} unmatched image placeholder tokens",
                destinations.len() - destination_offset
            );
        }
        requests.push(RequestMultimodalLayout {
            sequence_id: *seq.id(),
            query: 0..query_len,
            items,
        });
    }
    PackedMultimodalLayout::new(&requests)
}

impl Mistral3Processor {
    pub fn new(processor_config: ProcessorConfig) -> Self {
        Self {
            image_break_token: processor_config.image_break_token.unwrap().clone(),
            image_end_token: processor_config.image_end_token.unwrap().clone(),
            image_token: processor_config.image_token.unwrap().clone(),
            patch_size: processor_config.patch_size.unwrap(),
            spatial_merge_size: processor_config.spatial_merge_size.unwrap(),
        }
    }
}

impl Processor for Mistral3Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Mistral3ImageProcessor {
            image_break_token: self.image_break_token.clone(),
            image_end_token: self.image_end_token.clone(),
            image_token: self.image_token.clone(),
            patch_size: self.patch_size,
            spatial_merge_size: self.spatial_merge_size,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl InputsProcessor for Mistral3ImageProcessor {
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
        let tokenizer = tokenizer.ok_or_else(|| {
            anyhow::Error::msg("Mistral3ImageProcessor requires a specified tokenizer.")
        })?;
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        for seq in input_seqs {
            if !seq.has_images() || seq.multimodal.has_changed_prompt {
                continue;
            }
            let image_sizes = self.planned_image_sizes(seq.images().unwrap_or_default(), config)?;
            self.prepare_prompt(
                &tokenizer,
                seq,
                &image_sizes,
                paged_attn_metadata.as_deref_mut(),
            )?;
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
                "Idefics3ImageProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().any(|seq| seq.has_images());
        let mut image_sizes_by_sequence = vec![Vec::new(); input_seqs.len()];

        let (pixel_values, image_sizes) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut image_sizes_accum = Vec::new();

            for (seq_idx, seq) in input_seqs.iter_mut().enumerate() {
                if !seq.has_images() {
                    continue;
                }
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
                } = self.preprocess(
                    seq.take_images()
                        .expect("Need to have images by this point."),
                    vec![],
                    config,
                    device,
                    (usize::MAX, usize::MAX),
                )?;
                let image_sizes_all = image_sizes_all.ok_or_else(|| {
                    anyhow::Error::msg("Mistral 3 preprocessing omitted image sizes")
                })?;
                image_sizes_by_sequence[seq_idx] = image_sizes_all.clone();

                self.prepare_prompt(
                    &tokenizer,
                    seq,
                    &image_sizes_all,
                    paged_attn_metadata.as_mut(),
                )?;

                // Per-sequence prefix cache trimming of pixel_values and image_sizes
                let cached = seq.count_prefix_cached_mm_items();
                let n_images = pixel_values.dim(0)?;
                if cached < n_images {
                    if cached > 0 {
                        pixel_values_accum.push(pixel_values.narrow(
                            0,
                            cached,
                            n_images - cached,
                        )?);
                        image_sizes_accum.extend_from_slice(&image_sizes_all[cached..]);
                    } else {
                        pixel_values_accum.push(pixel_values.clone());
                        image_sizes_accum.extend_from_slice(&image_sizes_all);
                    }
                }
            }

            if pixel_values_accum.is_empty() {
                (None, None)
            } else {
                (
                    Some(cat_padded_mistral3_images(&pixel_values_accum)?),
                    Some(image_sizes_accum),
                )
            }
        } else {
            (None, None)
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

        let pixel_values = if is_prompt { pixel_values } else { None };
        let image_sizes = if is_prompt { image_sizes } else { None };

        let image_hashes: Vec<u64> = if is_prompt {
            input_seqs
                .iter()
                .flat_map(|seq| {
                    seq.image_hashes()
                        .map(|h| {
                            let cached = seq.count_prefix_cached_mm_items();
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
        let packed_layout = if is_prompt && flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    anyhow::Error::msg("packed Mistral 3 prefill requires logical query lengths")
                })?;
            let image_token_id = tokenizer.token_to_id(&self.image_token).ok_or_else(|| {
                anyhow::Error::msg("Mistral 3 tokenizer is missing the image token")
            })?;
            let layout = mistral3_packed_layout(
                input_seqs,
                query_lens,
                &image_sizes_by_sequence,
                image_token_id,
                self.patch_size,
                self.spatial_merge_size,
            )?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "Mistral 3 packed layout has {} tokens but input has {}",
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
            model_specific_args: Box::new(Mistral3SpecificArgs {
                image_sizes,
                image_hashes,
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

impl Mistral3ImageProcessor {
    fn resized_dimensions(
        &self,
        mut height: usize,
        mut width: usize,
        max_height: usize,
        max_width: usize,
        patch_size: usize,
    ) -> (usize, usize) {
        let ratio = (height as f64 / max_height as f64).max(width as f64 / max_width as f64);
        if ratio > 1. {
            height = (height as f64 / ratio).floor() as usize;
            width = (width as f64 / ratio).floor() as usize;
        }

        let num_height_tokens = (height - 1) / patch_size + 1;
        let num_width_tokens = (width - 1) / patch_size + 1;
        (
            num_height_tokens * patch_size,
            num_width_tokens * patch_size,
        )
    }

    fn planned_image_sizes(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<Vec<(u32, u32)>> {
        if !config.do_resize.unwrap() {
            return Ok(images
                .iter()
                .map(|image| {
                    let (width, height) = image.dimensions();
                    (height, width)
                })
                .collect());
        }

        let size = config.size.as_ref().unwrap();
        let (max_height, max_width) = if size.contains_key("longest_edge") {
            (size["longest_edge"] as usize, size["longest_edge"] as usize)
        } else if size.contains_key("height") && size.contains_key("width") {
            (size["height"] as usize, size["width"] as usize)
        } else {
            candle_core::bail!("Size must be a map of `longest_edge` or `height` and `width`.");
        };
        let patch_size = config.patch_size.unwrap();
        images
            .iter()
            .map(|image| {
                let (width, height) = image.dimensions();
                let (height, width) = self.resized_dimensions(
                    height as usize,
                    width as usize,
                    max_height,
                    max_width,
                    patch_size,
                );
                Ok((u32::try_from(height)?, u32::try_from(width)?))
            })
            .collect()
    }

    fn image_replacement(&self, height: u32, width: u32) -> anyhow::Result<String> {
        let num_height_tokens = (height as usize) / (self.patch_size * self.spatial_merge_size);
        let num_width_tokens = (width as usize) / (self.patch_size * self.spatial_merge_size);
        if num_height_tokens == 0 || num_width_tokens == 0 {
            anyhow::bail!(
                "Mistral 3 image size {height}x{width} is too small for its patch geometry"
            );
        }

        let mut replace_tokens = vec![
            [
                vec![self.image_token.clone(); num_width_tokens],
                vec![self.image_break_token.clone()],
            ]
            .concat();
            num_height_tokens
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        *replace_tokens.last_mut().unwrap() = self.image_end_token.clone();
        Ok(replace_tokens.join(""))
    }

    fn expand_prompt(&self, prompt: &str, image_sizes: &[(u32, u32)]) -> anyhow::Result<String> {
        let fragments = prompt.split(&self.image_token).collect::<Vec<_>>();
        let placeholder_count = fragments.len() - 1;
        if placeholder_count != image_sizes.len() {
            anyhow::bail!(
                "Mistral 3 has {placeholder_count} image placeholders but {} image inputs",
                image_sizes.len()
            );
        }

        let mut expanded = String::with_capacity(prompt.len());
        for (fragment, &(height, width)) in fragments.iter().zip(image_sizes) {
            expanded.push_str(fragment);
            expanded.push_str(&self.image_replacement(height, width)?);
        }
        expanded.push_str(fragments.last().unwrap());
        Ok(expanded)
    }

    fn prepare_prompt(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut Sequence,
        image_sizes: &[(u32, u32)],
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        if seq.multimodal.has_changed_prompt {
            return Ok(());
        }

        let prompt = tokenizer
            .decode(seq.get_toks(), false)
            .map_err(|err| anyhow::Error::msg(err.to_string()))?;
        let prompt = self.expand_prompt(&prompt, image_sizes)?;
        let tokens = tokenizer
            .encode_fast(prompt.as_str(), false)
            .map_err(|err| anyhow::Error::msg(err.to_string()))?;
        let ids = tokens.get_ids().to_vec();

        let hashes = seq.image_hashes().unwrap_or_default();
        let image_token_id = tokenizer
            .token_to_id(&self.image_token)
            .ok_or_else(|| anyhow::Error::msg("Mistral 3 tokenizer is missing the image token"))?;
        let image_end_token_id = tokenizer
            .token_to_id(&self.image_end_token)
            .ok_or_else(|| {
                anyhow::Error::msg("Mistral 3 tokenizer is missing the image end token")
            })?;
        let ranges = find_mistral3_image_ranges(&ids, image_token_id, image_end_token_id);
        if ranges.len() != hashes.len() {
            anyhow::bail!(
                "Mistral 3 has {} expanded image ranges but {} image inputs",
                ranges.len(),
                hashes.len()
            );
        }
        seq.set_mm_features(build_mm_features_from_ranges(
            &ranges,
            hashes,
            MultimodalKind::Image,
        ));

        let has_prefill_toks = seq.has_prefill_toks();
        seq.set_initial_prompt(prompt);
        seq.set_toks_and_reallocate(ids.clone(), paged_attn_metadata);
        if has_prefill_toks {
            seq.set_prefill_toks(ids);
        }
        seq.multimodal.has_changed_prompt = true;
        Ok(())
    }
}

impl ImagePreProcessor for Mistral3ImageProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/image_processing_pixtral.py
    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        assert!(videos.is_empty());

        let do_resize = config.do_resize.unwrap();
        let do_rescale = config.do_rescale.unwrap();
        let rescale_factor = config.rescale_factor.unwrap();
        let do_normalize = config.do_normalize.unwrap();
        let image_mean = config.image_mean.unwrap_or(Self::DEFAULT_MEAN);
        let image_std = config.image_std.unwrap_or(Self::DEFAULT_STD);
        let do_convert_rgb = config.do_convert_rgb.unwrap_or(true);
        let patch_size = config.patch_size.unwrap();
        let size = config.size.as_ref().unwrap();
        let resample = config.resampling.to_filter()?;

        let default_to_square = config.default_to_square.unwrap();
        assert!(default_to_square);

        let mut pixel_values = Vec::new();
        let mut image_sizes = Vec::new();

        let (max_height, max_width) = if size.contains_key("longest_edge") {
            (size["longest_edge"] as usize, size["longest_edge"] as usize)
        } else if size.contains_key("height") && size.contains_key("width") {
            (size["height"] as usize, size["width"] as usize)
        } else {
            candle_core::bail!("Size must be a map of `longest_edge` or `height` and `width`.");
        };

        for image in images.iter_mut() {
            let (width, height) = image.dimensions();

            // Convert to rgb
            if do_convert_rgb {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            if do_resize {
                let (height, width) = self.resized_dimensions(
                    height as usize,
                    width as usize,
                    max_height,
                    max_width,
                    patch_size,
                );
                *image = image.resize_exact(width as u32, height as u32, resample);
            }

            let (width, height) = image.dimensions();

            image_sizes.push((height, width));
        }

        images = mistralrs_vision::pad_to_max_image_size(images);

        for image in images.iter_mut() {
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
            pixel_values.push(image.unsqueeze(0)?);
        }

        Ok(PreprocessedImages {
            pixel_values: Tensor::cat(&pixel_values, 0)?,
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
    use candle_core::{DType, Device, Tensor};

    use super::{cat_padded_mistral3_images, find_mistral3_image_ranges, Mistral3ImageProcessor};

    fn processor() -> Mistral3ImageProcessor {
        Mistral3ImageProcessor {
            image_break_token: "B".to_string(),
            image_end_token: "E".to_string(),
            image_token: "I".to_string(),
            patch_size: 2,
            spatial_merge_size: 1,
        }
    }

    #[test]
    fn image_ranges_keep_grid_breaks_with_their_image() {
        let tokens = [9, 1, 1, 2, 1, 1, 3, 8, 1, 2, 1, 3, 7];
        assert_eq!(
            find_mistral3_image_ranges(&tokens, 1, 3),
            vec![(1, 6), (8, 4)]
        );
    }

    #[test]
    fn prompt_expansion_uses_each_image_size_in_order() {
        assert_eq!(
            processor()
                .expand_prompt("aI b I c", &[(2, 4), (4, 2)])
                .unwrap(),
            "aIIE b IBIE c"
        );
    }

    #[test]
    fn prompt_expansion_rejects_media_count_mismatch() {
        let error = processor()
            .expand_prompt("aI b I c", &[(2, 4)])
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("2 image placeholders but 1 image inputs"));
    }

    #[test]
    fn planned_resize_matches_patch_rounding() {
        assert_eq!(
            processor().resized_dimensions(100, 200, 100, 100, 16),
            (64, 112)
        );
    }

    #[test]
    fn image_batches_pad_across_sequence_boundaries() {
        let device = Device::Cpu;
        let images = vec![
            Tensor::zeros((1, 3, 2, 4), DType::F32, &device).unwrap(),
            Tensor::zeros((2, 3, 4, 2), DType::F32, &device).unwrap(),
        ];

        assert_eq!(
            cat_padded_mistral3_images(&images).unwrap().dims(),
            &[3, 3, 4, 4]
        );
    }
}
