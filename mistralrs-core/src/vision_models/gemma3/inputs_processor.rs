#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    any::Any,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Range,
    sync::Arc,
};

use candle_core::{Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};
use itertools::Itertools;
use mistralrs_vision::{ApplyTransforms, Normalize, Rescale, ToTensorNoNorm, Transforms};
use regex::Regex;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::{MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind},
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{find_image_placeholder_ranges, Sequence},
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

use super::Gemma3SpecificArgs;

struct Gemma3ImageProcessor {
    full_image_sequence: String,
    supports_images: bool,
}

const IMAGE_TOKEN: &str = "<image_soft_token>";
const BOI_TOKEN: &str = "<start_of_image>";
const EOI_TOKEN: &str = "<end_of_image>";

fn expanded_image_hashes(raw_hashes: &[u64], num_crops: &[usize]) -> Result<Vec<u64>> {
    if raw_hashes.len() != num_crops.len() {
        candle_core::bail!(
            "Gemma 3 has {} image hashes but {} crop counts",
            raw_hashes.len(),
            num_crops.len()
        );
    }
    let mut hashes =
        Vec::with_capacity(raw_hashes.len() + num_crops.iter().copied().sum::<usize>());
    for (&hash, &crop_count) in raw_hashes.iter().zip(num_crops) {
        hashes.push(hash);
        for crop_index in 0..crop_count {
            let mut hasher = DefaultHasher::new();
            "gemma3-pan-and-scan".hash(&mut hasher);
            hash.hash(&mut hasher);
            crop_index.hash(&mut hasher);
            hashes.push(hasher.finish());
        }
    }
    Ok(hashes)
}

fn gemma3_mm_features(
    tokens: &[u32],
    image_token_id: u32,
    raw_hashes: &[u64],
    num_crops: &[usize],
) -> Result<Vec<MultiModalFeature>> {
    let ranges = find_image_placeholder_ranges(tokens, image_token_id);
    let expanded_hashes = expanded_image_hashes(raw_hashes, num_crops)?;
    if ranges.len() != expanded_hashes.len() {
        candle_core::bail!(
            "Gemma 3 has {} image placeholder spans but {} encoder items",
            ranges.len(),
            expanded_hashes.len()
        );
    }

    let mut features = Vec::with_capacity(ranges.len());
    let mut expanded_index = 0usize;
    for (raw_index, &crop_count) in num_crops.iter().enumerate() {
        for _ in 0..=crop_count {
            let (offset, length) = ranges[expanded_index];
            features.push(MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: raw_index..raw_index + 1,
                hashes: vec![expanded_hashes[expanded_index]],
                offset,
                length,
                attention_policy: MultimodalAttentionPolicy::NonCausal,
                splittable: false,
            });
            expanded_index += 1;
        }
    }
    Ok(features)
}

fn validate_paged_noncausal_ranges<'a>(
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
            "Gemma 3 media span length {} exceeds its sliding attention window {sliding_window}",
            feature.length
        );
    }
    Ok(())
}

fn active_expanded_image_indices(
    expanded_hashes: &[u64],
    features: &[MultiModalFeature],
    query: Range<usize>,
) -> Result<Vec<usize>> {
    if query.start > query.end {
        candle_core::bail!("Gemma 3 active query range is reversed");
    }
    let active_hashes = features
        .iter()
        .filter(|feature| {
            feature.kind == MultimodalKind::Image && feature.overlaps(query.start, query.end)
        })
        .flat_map(|feature| feature.hashes.iter().copied())
        .collect::<Vec<_>>();
    let mut used = vec![false; expanded_hashes.len()];
    let mut indices = Vec::with_capacity(active_hashes.len());
    for hash in active_hashes {
        let index = expanded_hashes
            .iter()
            .enumerate()
            .find_map(|(index, candidate)| (!used[index] && *candidate == hash).then_some(index))
            .ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "Gemma 3 active encoder hash {hash} is missing from preprocessed images"
                ))
            })?;
        used[index] = true;
        indices.push(index);
    }
    Ok(indices)
}

fn gemma3_layout_items(
    tokens: &[u32],
    image_token_id: u32,
    image_hashes: &[u64],
) -> Result<Vec<MultimodalItemLayout>> {
    let ranges = find_image_placeholder_ranges(tokens, image_token_id);
    if ranges.len() != image_hashes.len() {
        candle_core::bail!(
            "Gemma 3 has {} image placeholder spans but {} encoder outputs",
            ranges.len(),
            image_hashes.len()
        );
    }
    ranges
        .into_iter()
        .zip(image_hashes)
        .enumerate()
        .map(|(item_index, ((offset, length), &hash))| {
            let placeholder = offset..offset + length;
            MultimodalItemLayout::new(
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash,
                },
                item_index,
                placeholder.clone(),
                MultimodalAttentionPolicy::NonCausal,
                vec![MultimodalEmbeddingMap::contiguous(placeholder, 0, 0)?],
            )
        })
        .collect()
}

fn gemma3_packed_layout(
    input_seqs: &[&mut Sequence],
    query_lens: &[usize],
    image_hashes_by_sequence: &[Vec<u64>],
    image_token_id: u32,
) -> Result<PackedMultimodalLayout> {
    if input_seqs.len() != query_lens.len() || input_seqs.len() != image_hashes_by_sequence.len() {
        candle_core::bail!("Gemma 3 packed multimodal metadata length mismatch");
    }
    let requests = input_seqs
        .iter()
        .zip(query_lens)
        .zip(image_hashes_by_sequence)
        .map(|((seq, &query_len), image_hashes)| {
            if query_len != seq.get_toks().len() {
                candle_core::bail!(
                    "Gemma 3 packed multimodal prefill requires the complete uncached prompt"
                );
            }
            Ok(RequestMultimodalLayout {
                sequence_id: *seq.id(),
                query: 0..query_len,
                items: gemma3_layout_items(seq.get_toks(), image_token_id, image_hashes)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    PackedMultimodalLayout::new(&requests)
}

pub struct Gemma3Processor {
    full_image_sequence: String,
    supports_images: bool,
}

impl Gemma3Processor {
    pub fn new(processor_config: ProcessorConfig, supports_images: bool) -> Self {
        let image_tokens_expanded =
            vec![IMAGE_TOKEN.to_string(); processor_config.image_seq_len.unwrap_or(256)].join("");
        let full_image_sequence = format!("\n\n{BOI_TOKEN}{image_tokens_expanded}{EOI_TOKEN}\n\n");

        Self {
            full_image_sequence,
            supports_images,
        }
    }
}

impl Processor for Gemma3Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Gemma3ImageProcessor {
            full_image_sequence: self.full_image_sequence.clone(),
            supports_images: self.supports_images,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[BOI_TOKEN, EOI_TOKEN, IMAGE_TOKEN]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl InputsProcessor for Gemma3ImageProcessor {
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
    ) -> anyhow::Result<()> {
        let Some(tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "Gemma3ImageProcessor requires a specified tokenizer.",
            ));
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        if !input_seqs.iter().any(|seq| seq.has_images()) {
            return Ok(());
        }
        if !self.supports_images {
            return Err(anyhow::Error::msg(
                "This image processor does not support images.",
            ));
        }

        let re = Regex::new(BOI_TOKEN).unwrap();
        for seq in input_seqs.iter_mut() {
            if !seq.has_images() {
                continue;
            }
            if seq.multimodal.has_changed_prompt {
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
                image_sizes_all: _,
                num_crops,
            } = self
                .preprocess(
                    seq.clone_images()
                        .expect("Need to have images by this point."),
                    vec![],
                    config,
                    device,
                    (usize::MAX, usize::MAX),
                )
                .expect("Preprocessing failed");
            let num_crops = num_crops.unwrap();
            seq.multimodal.cached_pixel_values = Some(pixel_values);
            seq.multimodal.cached_num_crops = Some(num_crops.clone());

            let mut prompt = tokenizer
                .decode(seq.get_toks(), false)
                .expect("Detokenization failed!");
            let image_indexes = re
                .find_iter(&prompt)
                .map(|mat| mat.start())
                .collect::<Vec<_>>();
            for (num, idx) in num_crops.iter().copied().zip(image_indexes).rev() {
                if num != 0 {
                    let formatted_image_text = format!(
                        "Here is the original image {BOI_TOKEN} and here are some crops to help you see better {}",
                        vec![BOI_TOKEN.to_string(); num].join(" ")
                    );
                    prompt = format!(
                        "{}{formatted_image_text}{}",
                        &prompt[..idx],
                        &prompt[idx + BOI_TOKEN.len()..]
                    );
                }
            }
            prompt = prompt.replace(BOI_TOKEN, &self.full_image_sequence);

            seq.set_initial_prompt(prompt.clone());
            let toks = tokenizer
                .encode_fast(prompt, false)
                .expect("Detokenization failed!");
            let ids = toks.get_ids().to_vec();
            if seq.mm_features().is_empty() {
                if let (Some(hashes), Some(img_tok_id)) = (
                    seq.image_hashes().map(|h| h.to_vec()),
                    tokenizer.token_to_id(IMAGE_TOKEN),
                ) {
                    seq.set_mm_features(gemma3_mm_features(&ids, img_tok_id, &hashes, &num_crops)?);
                }
            }

            seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_deref_mut());
            seq.multimodal.has_changed_prompt = true;
        }

        if let Some(metadata) = paged_attn_metadata.as_ref() {
            validate_paged_noncausal_ranges(
                input_seqs.iter().flat_map(|seq| seq.mm_features()),
                metadata.sliding_window,
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
                "Gemma3ImageProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().any(|seq| seq.has_images());
        let mut image_hashes_by_sequence = vec![Vec::new(); input_seqs.len()];

        let pixel_values = if has_images {
            if !self.supports_images {
                return Err(anyhow::Error::msg(
                    "This image processor does not support images.",
                ));
            }

            let mut pixel_values_accum = Vec::new();
            let re = Regex::new(BOI_TOKEN).unwrap();
            for (seq_index, seq) in input_seqs.iter_mut().enumerate() {
                if !seq.has_images() {
                    continue;
                }
                let is_chunked_view = seq.is_chunked_prefill_view();
                let cached_pixel_values = seq.multimodal.cached_pixel_values.clone();
                let (pixel_values, num_crops, uses_full_media_set) =
                    if let Some(cached_pixel_values) = cached_pixel_values {
                        let num_crops =
                            seq.multimodal.cached_num_crops.clone().ok_or_else(|| {
                                anyhow::Error::msg(
                                    "Gemma 3 cached pixels are missing crop metadata",
                                )
                            })?;
                        (cached_pixel_values, num_crops, true)
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
                            image_sizes_all: _,
                            num_crops,
                        } = self
                            .preprocess(
                                seq.take_images()
                                    .expect("Need to have images by this point."),
                                vec![],
                                config,
                                device,
                                (usize::MAX, usize::MAX), // Don't use it here...
                            )
                            .expect("Preprocessing failed");
                        let num_crops =
                            num_crops.expect("Gemma 3 preprocessing omitted crop counts");
                        if !is_chunked_view {
                            seq.multimodal.cached_pixel_values = Some(pixel_values.clone());
                            seq.multimodal.cached_num_crops = Some(num_crops.clone());
                        }
                        (pixel_values, num_crops, false)
                    };

                if !seq.multimodal.has_changed_prompt {
                    let mut prompt = tokenizer
                        .decode(seq.get_toks(), false)
                        .expect("Detokenization failed!");
                    let image_indexes = re
                        .find_iter(&prompt)
                        .map(|mat| mat.start())
                        .collect::<Vec<_>>();
                    for (num, idx) in num_crops.iter().copied().zip(image_indexes).rev() {
                        if num != 0 {
                            let formatted_image_text = format!(
                                "Here is the original image {BOI_TOKEN} and here are some crops to help you see better {}", vec![BOI_TOKEN.to_string(); num].join(" ")
                            );
                            prompt = format!(
                                "{}{formatted_image_text}{}",
                                &prompt[..idx],
                                &prompt[idx + BOI_TOKEN.len()..]
                            );
                        }
                    }
                    prompt = prompt.replace(BOI_TOKEN, &self.full_image_sequence);
                    seq.set_initial_prompt(prompt.clone());
                    let toks = tokenizer
                        .encode_fast(prompt, false)
                        .expect("Detokenization failed!");

                    let ids = toks.get_ids().to_vec();

                    if seq.mm_features().is_empty() {
                        if let (Some(hashes), Some(img_tok_id)) = (
                            seq.image_hashes().map(|h| h.to_vec()),
                            tokenizer.token_to_id(IMAGE_TOKEN),
                        ) {
                            seq.set_mm_features(gemma3_mm_features(
                                &ids, img_tok_id, &hashes, &num_crops,
                            )?);
                        }
                    }

                    seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());
                    seq.multimodal.has_changed_prompt = true;
                }

                let raw_hashes = if is_chunked_view && uses_full_media_set {
                    seq.multimodal.image_hashes().unwrap_or_default()
                } else {
                    seq.image_hashes().unwrap_or_default()
                };
                let expanded_hashes = expanded_image_hashes(raw_hashes, &num_crops)?;
                let n_images = pixel_values.dim(0)?;
                if expanded_hashes.len() != n_images {
                    anyhow::bail!(
                        "Gemma 3 has {} encoder hashes but {} preprocessed images",
                        expanded_hashes.len(),
                        n_images
                    );
                }
                if is_chunked_view {
                    let query = seq.active_prompt_query_range().ok_or_else(|| {
                        anyhow::Error::msg("Gemma 3 media view is missing its prompt query")
                    })?;
                    let active_indices =
                        active_expanded_image_indices(&expanded_hashes, seq.mm_features(), query)?;
                    if active_indices.is_empty() {
                        anyhow::bail!("Gemma 3 chunk has images but no active encoder items");
                    }
                    let selected_hashes = active_indices
                        .iter()
                        .map(|&index| expanded_hashes[index])
                        .collect::<Vec<_>>();
                    let active_indices = active_indices
                        .into_iter()
                        .map(|index| u32::try_from(index).map_err(candle_core::Error::wrap))
                        .collect::<Result<Vec<_>>>()?;
                    let active_count = active_indices.len();
                    let active_indices = Tensor::from_vec(active_indices, active_count, device)?;
                    pixel_values_accum.push(pixel_values.index_select(&active_indices, 0)?);
                    image_hashes_by_sequence[seq_index] = selected_hashes;
                } else {
                    image_hashes_by_sequence[seq_index] = expanded_hashes;
                    let cached = seq.count_prefix_cached_mm_items();
                    if cached < n_images {
                        if cached > 0 {
                            pixel_values_accum.push(pixel_values.narrow(
                                0,
                                cached,
                                n_images - cached,
                            )?);
                        } else {
                            pixel_values_accum.push(pixel_values.clone());
                        }
                    }
                }
            }

            if pixel_values_accum.is_empty() {
                None
            } else {
                Some(Tensor::cat(&pixel_values_accum, 0)?)
            }
        } else {
            None
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

        let pixel_values = if is_prompt { pixel_values } else { None };

        let image_hashes: Vec<u64> = if is_prompt {
            input_seqs
                .iter()
                .zip(&image_hashes_by_sequence)
                .flat_map(|(seq, hashes)| {
                    let cached = seq.count_prefix_cached_mm_items();
                    hashes.get(cached..).unwrap_or_default().to_vec()
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
                    anyhow::Error::msg("packed Gemma 3 prefill requires logical query lengths")
                })?;
            let image_token_id = tokenizer.token_to_id(IMAGE_TOKEN).ok_or_else(|| {
                anyhow::Error::msg("Gemma 3 tokenizer is missing the image token")
            })?;
            let layout = gemma3_packed_layout(
                input_seqs,
                query_lens,
                &image_hashes_by_sequence,
                image_token_id,
            )?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "Gemma 3 packed layout has {} tokens but input has {}",
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
            model_specific_args: Box::new(Gemma3SpecificArgs {
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

impl Gemma3ImageProcessor {
    fn pan_and_scan(
        &self,
        image: &DynamicImage,
        pan_and_scan_min_crop_size: usize,
        pan_and_scan_max_num_crops: usize,
        pan_and_scan_min_ratio_to_activate: f64,
    ) -> Vec<DynamicImage> {
        let (width, height) = image.dimensions();

        let (num_crops_w, num_crops_h) = if width >= height {
            if (width as f64 / height as f64) < pan_and_scan_min_ratio_to_activate {
                return vec![];
            }

            // Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            let mut num_crops_w = (width as f64 / height as f64 + 0.5).floor() as usize;
            num_crops_w = num_crops_w
                .min((width as f64 / pan_and_scan_min_crop_size as f64).floor() as usize);

            // Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_w = num_crops_w.max(2);
            num_crops_w = num_crops_w.min(pan_and_scan_max_num_crops);

            (num_crops_w, 1)
        } else {
            if (height as f64 / width as f64) < pan_and_scan_min_ratio_to_activate {
                return vec![];
            }

            // Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            let mut num_crops_h = (height as f64 / width as f64 + 0.5).floor() as usize;
            num_crops_h = num_crops_h
                .min((height as f64 / pan_and_scan_min_crop_size as f64).floor() as usize);

            // Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_h = num_crops_h.max(2);
            num_crops_h = num_crops_h.min(pan_and_scan_max_num_crops);

            (1, num_crops_h)
        };

        let crop_size_w = (width as f64 / num_crops_w as f64).ceil() as usize;
        let crop_size_h = (height as f64 / num_crops_h as f64).ceil() as usize;

        if crop_size_w.min(crop_size_h) < pan_and_scan_min_crop_size {
            return vec![];
        }

        let crop_positions_w = (0..num_crops_w)
            .map(|i| i * crop_size_w)
            .collect::<Vec<_>>();
        let crop_positions_h = (0..num_crops_h)
            .map(|i| i * crop_size_h)
            .collect::<Vec<_>>();

        let mut image_crops = Vec::new();
        for (pos_h, pos_w) in crop_positions_h
            .into_iter()
            .cartesian_product(crop_positions_w)
        {
            image_crops.push(image.crop_imm(
                pos_w as u32,
                pos_h as u32,
                crop_size_w as u32,
                crop_size_h as u32,
            ));
        }

        image_crops
    }

    fn process_images_for_pan_and_scan(
        &self,
        images: Vec<DynamicImage>,
        pan_and_scan_min_crop_size: usize,
        pan_and_scan_max_num_crops: usize,
        pan_and_scan_min_ratio_to_activate: f64,
    ) -> (Vec<DynamicImage>, Vec<usize>) {
        let mut pas_images_list = Vec::new();
        let mut num_crops = Vec::new();

        for image in images {
            let pas_images = self.pan_and_scan(
                &image,
                pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate,
            );
            num_crops.push(pas_images.len());
            pas_images_list.extend([vec![image], pas_images].concat());
        }

        (pas_images_list, num_crops)
    }
}

impl ImagePreProcessor for Gemma3ImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

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
        let size = config.size.as_ref().unwrap();
        let (height, width) = (size["height"], size["width"]);
        let resample = config.resampling.to_filter()?;
        let do_rescale = config.do_rescale.unwrap();
        let rescale_factor = config.rescale_factor.unwrap();
        let do_normalize = config.do_normalize.unwrap();
        let image_mean = config.image_mean.unwrap_or(Self::DEFAULT_MEAN);
        let image_std = config.image_std.unwrap_or(Self::DEFAULT_STD);
        let do_convert_rgb = config.do_convert_rgb.unwrap_or(true);
        let do_pan_and_scan = config.do_pan_and_scan.unwrap_or(false);
        // https://github.com/huggingface/transformers/blob/ea219ed164bead55a5513e8cfaa17a25d5613b9e/src/transformers/models/gemma3/processing_gemma3.py#L42
        let pan_and_scan_min_crop_size = config.pan_and_scan_min_crop_size.unwrap_or(256);
        let pan_and_scan_max_num_crops = config.pan_and_scan_max_num_crops.unwrap_or(4);
        let pan_and_scan_min_ratio_to_activate =
            config.pan_and_scan_min_ratio_to_activate.unwrap_or(1.2);

        for image in images.iter_mut() {
            // Convert to rgb
            if do_convert_rgb {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }
        }

        let image_count = images.len();
        let num_crops = if do_pan_and_scan {
            let (new_images, num_crops) = self.process_images_for_pan_and_scan(
                images,
                pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate,
            );
            images = new_images;
            num_crops
        } else {
            vec![0; image_count]
        };

        let mut pixel_values = Vec::new();
        for mut image in images {
            if do_resize {
                image = image.resize_exact(width, height, resample);
            }

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
            image_sizes_all: None,
            num_crops: Some(num_crops),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn expanded_hashes_keep_originals_and_distinguish_crops() {
        let hashes = expanded_image_hashes(&[7, 9], &[2, 0]).unwrap();

        assert_eq!(hashes.len(), 4);
        assert_eq!(hashes[0], 7);
        assert_eq!(hashes[3], 9);
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[1], hashes[2]);
    }

    #[test]
    fn pan_and_scan_spans_keep_the_raw_image_item_index() {
        let features = gemma3_mm_features(&[1, 5, 5, 2, 5, 5, 3, 5, 5], 5, &[17], &[2]).unwrap();

        assert_eq!(features.len(), 3);
        assert!(features.iter().all(|feature| feature.item_range == (0..1)));
        assert!(features
            .iter()
            .all(|feature| { feature.attention_policy == MultimodalAttentionPolicy::NonCausal }));
        assert_eq!(
            features
                .iter()
                .map(|feature| (feature.offset, feature.length))
                .collect::<Vec<_>>(),
            vec![(1, 2), (4, 2), (7, 2)]
        );
    }

    #[test]
    fn paged_noncausal_media_must_fit_the_sliding_window() {
        let feature = MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset: 0,
            length: 5,
            attention_policy: MultimodalAttentionPolicy::NonCausal,
            splittable: false,
        };

        assert!(validate_paged_noncausal_ranges([&feature], Some(5)).is_ok());
        assert!(validate_paged_noncausal_ranges([&feature], Some(4)).is_err());
    }

    #[test]
    fn chunk_selects_only_the_active_pan_and_scan_output() {
        let raw_hash = 17;
        let hashes = expanded_image_hashes(&[raw_hash], &[2]).unwrap();
        let features = gemma3_mm_features(&[5, 5, 1, 5, 5, 2, 5, 5], 5, &[raw_hash], &[2]).unwrap();

        assert_eq!(
            active_expanded_image_indices(&hashes, &features, 3..5).unwrap(),
            vec![1]
        );
        assert_eq!(
            active_expanded_image_indices(&hashes, &features, 0..8).unwrap(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn later_chunk_selects_from_the_full_cached_media_set() {
        let hashes = expanded_image_hashes(&[7, 9], &[1, 0]).unwrap();
        let features = gemma3_mm_features(&[5, 5, 1, 5, 5, 2, 5, 5], 5, &[7, 9], &[1, 0]).unwrap();

        assert_eq!(
            active_expanded_image_indices(&hashes, &features, 6..8).unwrap(),
            vec![2]
        );
    }

    #[test]
    fn active_output_selection_consumes_duplicate_hashes_once() {
        let features = vec![
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![7],
                offset: 0,
                length: 2,
                attention_policy: MultimodalAttentionPolicy::NonCausal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 1..2,
                hashes: vec![7],
                offset: 3,
                length: 2,
                attention_policy: MultimodalAttentionPolicy::NonCausal,
                splittable: false,
            },
        ];

        assert_eq!(
            active_expanded_image_indices(&[7, 7], &features, 0..5).unwrap(),
            vec![0, 1]
        );
    }

    #[test]
    fn packed_layout_splices_media_without_touching_text_request() {
        let image_hash = 31;
        let layout = PackedMultimodalLayout::new(&[
            RequestMultimodalLayout {
                sequence_id: 10,
                query: 0..3,
                items: gemma3_layout_items(&[4, 5, 5], 5, &[image_hash]).unwrap(),
            },
            RequestMultimodalLayout {
                sequence_id: 11,
                query: 0..2,
                items: vec![],
            },
        ])
        .unwrap();
        let text = Tensor::from_vec(
            vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9.],
            (1, 5, 2),
            &Device::Cpu,
        )
        .unwrap();
        let encoder = Tensor::from_vec(vec![20f32, 21., 30., 31.], (2, 2), &Device::Cpu).unwrap();
        let outputs = HashMap::from([(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: image_hash,
            },
            vec![encoder],
        )]);

        let result = layout
            .splice_embeddings(&text, &outputs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(result, vec![0., 1., 20., 21., 30., 31., 6., 7., 8., 9.]);
    }
}
