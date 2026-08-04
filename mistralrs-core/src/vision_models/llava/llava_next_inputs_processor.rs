#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use crate::paged_attention::block_hash::{MultiModalFeature, MultimodalKind};
use std::sync::Arc;
use std::{any::Any, ops::Range};

use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use image::GenericImageView;
use itertools::Itertools;
use regex_automata::meta::Regex;
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::pipeline::text_models_inputs_processor::{
    get_completion_input, get_prompt_input, PagedAttentionMeta,
};
use crate::pipeline::{
    text_models_inputs_processor, InputProcessorOutput, InputsProcessor, InputsProcessorType,
    MessagesAction, Processor,
};
use crate::sequence::{build_mm_features_from_ranges, Sequence};
use crate::vision_models::image_processor::{self, ImagePreProcessor, PreprocessedImages};
use crate::vision_models::llava::config::Config as LLaVANextConfig;
use crate::vision_models::preprocessor_config::{PreProcessorConfig, ToFilter};
use crate::vision_models::{
    multimodal_layout::{
        MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout, PackedMultimodalLayout,
        RequestMultimodalLayout,
    },
    preprocessor_config, ModelInputs,
};

use super::llava_next::LLaVANextVisionSpecificArgs;
use super::utils::{
    calculate_unpad, divide_to_samples, get_anyres_image_grid_shape, get_num_samples,
    resize_and_pad_image, select_best_resolution, LLaVAImageProcessor,
};

pub struct LLaVANextProcessor {
    inputs_processor: Arc<LLaVANextInputProcessor>,
}

impl Processor for LLaVANextProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        self.inputs_processor.clone()
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl LLaVANextProcessor {
    pub fn new(config: &str) -> Self {
        let model_config =
            serde_json::from_str::<LLaVANextConfig>(config).expect("Failed to parse model config.");
        let image_tag_splitter = Regex::new(r"<image>").expect("Failed to compile split regex.");
        let inputs_processor = Arc::new(LLaVANextInputProcessor {
            image_tag_splitter,
            model_config: model_config.clone(),
        });
        Self { inputs_processor }
    }
}

pub struct LLaVANextInputProcessor {
    image_tag_splitter: Regex,
    model_config: LLaVANextConfig,
}

type LLaVANextPromptTokens = (Vec<i64>, Vec<(usize, usize)>);

impl LLaVANextInputProcessor {
    pub fn get_num_image_tokens(cfg: &LLaVANextConfig, image_size: (u32, u32)) -> usize {
        let patch_size = cfg.vision_config.patch_size;
        let image_grid_pinpoints = cfg.image_grid_pinpoints.clone().unwrap();
        let anyres_grid_shape =
            get_anyres_image_grid_shape(image_size, &image_grid_pinpoints, patch_size as u32);
        let patch_per_side = cfg.vision_config.image_size / patch_size;
        let unpad_shape = calculate_unpad(anyres_grid_shape, image_size);
        patch_per_side * patch_per_side + (unpad_shape.0 as usize + 1) * (unpad_shape.1 as usize)
    }
}

fn llava_next_prompt_tokens(
    tokenizer: &Tokenizer,
    image_tag_splitter: &Regex,
    detokenized: &str,
    image_token_counts: &[usize],
) -> anyhow::Result<LLaVANextPromptTokens> {
    let splits = image_tag_splitter
        .split(detokenized)
        .map(|span| &detokenized[span.range()])
        .collect::<Vec<_>>();
    let tag_count = splits.len().saturating_sub(1);
    if tag_count != image_token_counts.len() {
        anyhow::bail!(
            "LLaVA-Next prompt has {tag_count} image tags but {} images",
            image_token_counts.len()
        );
    }
    let prompt_chunks = splits
        .into_iter()
        .map(|chunk| {
            tokenizer
                .encode_fast(chunk, false)
                .map_err(|error| anyhow::Error::msg(error.to_string()))
                .map(|encoding| {
                    encoding
                        .get_ids()
                        .iter()
                        .map(|token| i64::from(*token))
                        .collect::<Vec<_>>()
                })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let image_pads = image_token_counts
        .iter()
        .enumerate()
        .map(|(index, &token_count)| {
            if token_count == 0 {
                anyhow::bail!("LLaVA-Next image placeholder cannot be empty");
            }
            let mut pad = vec![0; token_count];
            pad[0] = -(index as i64 + 1);
            Ok(pad)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut ranges = Vec::with_capacity(image_pads.len());
    let mut offset = 0usize;
    for (chunk, pad) in prompt_chunks.iter().zip(&image_pads) {
        offset += chunk.len();
        ranges.push((offset, pad.len()));
        offset += pad.len();
    }
    let mut tokens = Vec::new();
    for part in prompt_chunks.into_iter().interleave(image_pads) {
        tokens.extend(part);
    }
    Ok((tokens, ranges))
}

fn restore_llava_next_image_markers(
    tokens: &[u32],
    features: &[MultiModalFeature],
    query: Range<usize>,
    local_query: Range<usize>,
) -> anyhow::Result<Vec<i64>> {
    if query.len() != local_query.len() {
        anyhow::bail!("LLaVA-Next prompt coordinate ranges have different lengths");
    }
    let mut tokens = tokens
        .iter()
        .map(|token| i64::from(*token))
        .collect::<Vec<_>>();
    for feature in features
        .iter()
        .filter(|feature| feature.kind == MultimodalKind::Image)
    {
        if !query.contains(&feature.offset) {
            continue;
        }
        if feature.item_range.len() != 1 {
            anyhow::bail!("LLaVA-Next image feature must describe exactly one image");
        }
        let local = local_query.start + feature.offset - query.start;
        let token = tokens
            .get_mut(local)
            .ok_or_else(|| anyhow::Error::msg("LLaVA-Next image marker is outside the query"))?;
        *token = -(feature.item_range.start as i64 + 1);
    }
    Ok(tokens)
}

fn llava_next_layout_items(features: &[MultiModalFeature]) -> Result<Vec<MultimodalItemLayout>> {
    features
        .iter()
        .filter(|feature| feature.kind == MultimodalKind::Image)
        .map(|feature| {
            if feature.item_range.len() != 1 || feature.hashes.len() != 1 {
                candle_core::bail!("LLaVA-Next image feature must describe exactly one image");
            }
            let placeholder = feature.offset..feature.end();
            MultimodalItemLayout::new(
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash: feature.hashes[0],
                },
                feature.item_range.start,
                placeholder.clone(),
                feature.attention_policy,
                vec![MultimodalEmbeddingMap::contiguous(placeholder, 0, 0)?],
            )
        })
        .collect()
}

fn llava_next_packed_layout(
    input_seqs: &[&mut Sequence],
    query_lens: &[usize],
) -> Result<PackedMultimodalLayout> {
    if input_seqs.len() != query_lens.len() {
        candle_core::bail!("LLaVA-Next packed multimodal metadata length mismatch");
    }
    let requests = input_seqs
        .iter()
        .zip(query_lens)
        .map(|(seq, &query_len)| {
            if query_len != seq.get_toks().len() {
                candle_core::bail!(
                    "LLaVA-Next packed multimodal prefill requires the complete uncached prompt"
                );
            }
            Ok(RequestMultimodalLayout {
                sequence_id: *seq.id(),
                query: 0..query_len,
                items: llava_next_layout_items(seq.mm_features())?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    PackedMultimodalLayout::new(&requests)
}

// Copy from phi3_inputs_processor. different is (1) calculate of num_image_token (2) process_anyres_image (3)image_ids_pad
impl InputsProcessor for LLaVANextInputProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let Some(tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "LLaVAInputProcessor requires a specified tokenizer.",
            ));
        };
        if !input_seqs.iter().any(|seq| seq.has_images()) {
            return Ok(());
        }

        for seq in input_seqs.iter_mut() {
            if seq.multimodal.has_changed_prompt {
                continue;
            }
            let Some(images) = seq.images() else {
                continue;
            };
            if images.is_empty() {
                continue;
            }

            let detokenized = tokenizer
                .decode(seq.get_toks(), false)
                .map_err(|error| anyhow::Error::msg(error.to_string()))?;
            let image_token_counts = images
                .iter()
                .map(|image| Self::get_num_image_tokens(&self.model_config, image.dimensions()))
                .collect::<Vec<_>>();
            let (tokens, image_ranges) = llava_next_prompt_tokens(
                &tokenizer,
                &self.image_tag_splitter,
                &detokenized,
                &image_token_counts,
            )?;
            let hashes = seq.image_hashes().unwrap_or_default().to_vec();
            if hashes.len() != image_ranges.len() {
                anyhow::bail!(
                    "LLaVA-Next has {} image hashes but {} image placeholders",
                    hashes.len(),
                    image_ranges.len()
                );
            }
            let new_ids = tokens
                .iter()
                .map(|x| if *x < 0 { 0u32 } else { *x as u32 })
                .collect::<Vec<_>>();
            let new_prompt = tokenizer
                .decode(&new_ids, false)
                .map_err(|error| anyhow::Error::msg(error.to_string()))?;
            seq.set_initial_prompt(new_prompt);

            if seq.mm_features().is_empty() {
                seq.set_mm_features(build_mm_features_from_ranges(
                    &image_ranges,
                    &hashes,
                    MultimodalKind::Image,
                ));
            }

            seq.set_toks_and_reallocate(new_ids, paged_attn_metadata.as_deref_mut());
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
                "LLaVAInputProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config
            .clone()
            .expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        let crop_size = (
            *config.crop_size.as_ref().unwrap().get("width").unwrap(),
            *config.crop_size.as_ref().unwrap().get("height").unwrap(),
        );

        let has_images = input_seqs.iter().any(|seq| seq.has_images());
        if !has_images {
            return text_models_inputs_processor::TextInputsProcessor
                .process_inputs(
                    Some(tokenizer),
                    input_seqs,
                    is_prompt,
                    is_xlora,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    sliding_window,
                    other_config,
                    paged_attn_metadata,
                    mapper,
                )
                .map(|metadata| {
                    let InputProcessorOutput {
                        inputs,
                        seq_indices,
                    } = metadata;

                    let text_models_inputs_processor::ModelInputs {
                        input_ids,
                        input_ids_full: _,
                        seqlen_offsets,
                        seqlen_offsets_full: _,
                        context_lens,
                        position_ids,
                        paged_attn_meta,
                        flash_meta,
                        flash_meta_full: _,
                        recurrent_batch_kind,
                        adapter_leases,
                    } = *inputs
                        .downcast::<text_models_inputs_processor::ModelInputs>()
                        .expect("Downcast failed.");

                    let inputs: Box<dyn Any> = Box::new(ModelInputs {
                        input_ids,
                        seqlen_offsets,
                        context_lens,
                        position_ids,
                        pixel_values: None,
                        model_specific_args: Box::new(LLaVANextVisionSpecificArgs {
                            image_sizes: None,
                            num_image_tokens: None,
                            num_image_samples: None,
                            image_hashes: vec![],
                            packed_layout: None,
                        }),
                        paged_attn_meta,
                        flash_meta,
                        recurrent_batch_kind,
                        adapter_leases,
                    });
                    InputProcessorOutput {
                        inputs,
                        seq_indices,
                    }
                });
        }

        let mut pixel_values_accum = Vec::new();
        let mut image_sizes = Vec::new();
        let mut image_token_counts = vec![Vec::new(); input_seqs.len()];
        let mut num_image_samples = Vec::new();
        for (seq_idx, seq) in input_seqs.iter_mut().enumerate() {
            if !seq.has_images() {
                continue;
            }
            let cached = seq.count_prefix_cached_mm_items();
            let images = seq
                .take_images()
                .expect("Need to have images by this point.");
            for image in images.into_iter().skip(cached) {
                let expected_samples = get_num_samples(
                    image.dimensions(),
                    self.model_config
                        .image_grid_pinpoints
                        .as_ref()
                        .expect("LLaVA-Next requires image_grid_pinpoints"),
                    crop_size,
                ) as usize;
                let PreprocessedImages {
                    pixel_values,
                    image_sizes: Some(image_size),
                    num_img_tokens: Some(num_img_tokens),
                    ..
                } = self.preprocess(
                    vec![image],
                    vec![],
                    config,
                    device,
                    (usize::MAX, usize::MAX),
                )?
                else {
                    anyhow::bail!("LLaVA-Next preprocessing omitted required image metadata");
                };
                if pixel_values.dim(0)? != expected_samples || num_img_tokens.len() != 1 {
                    anyhow::bail!("LLaVA-Next preprocessing returned inconsistent image metadata");
                }
                pixel_values_accum.push(pixel_values);
                image_sizes.push(image_size);
                image_token_counts[seq_idx].push(num_img_tokens[0]);
                num_image_samples.push(expected_samples);
            }
        }
        let pixel_values = if pixel_values_accum.is_empty() {
            None
        } else {
            Some(Tensor::cat(&pixel_values_accum, 0)?)
        };
        let image_sizes = (!image_sizes.is_empty()).then_some(image_sizes);
        let num_image_tokens_flat = image_token_counts
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();

        let mut toks = Vec::new();
        let detokenized = tokenizer
            .decode_batch(
                &input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                false,
            )
            .map_err(|error| anyhow::Error::msg(error.to_string()))?;

        for ((detokenized, seq), image_token_counts) in detokenized
            .into_iter()
            .zip(input_seqs.iter_mut())
            .zip(image_token_counts)
        {
            if seq.multimodal.has_changed_prompt {
                let query = seq
                    .active_prompt_query_range()
                    .unwrap_or(0..seq.get_toks().len());
                let local_query = seq
                    .active_prompt_local_query_range()
                    .unwrap_or(0..seq.get_toks().len());
                toks.push(restore_llava_next_image_markers(
                    seq.get_toks(),
                    seq.mm_features(),
                    query,
                    local_query,
                )?);
                continue;
            }
            if image_token_counts.is_empty() {
                toks.push(
                    seq.get_toks()
                        .iter()
                        .map(|token| i64::from(*token))
                        .collect(),
                );
                continue;
            }
            let (input_ids, image_ranges) = llava_next_prompt_tokens(
                &tokenizer,
                &self.image_tag_splitter,
                &detokenized,
                &image_token_counts,
            )?;
            let hashes = seq.image_hashes().unwrap_or_default().to_vec();
            if hashes.len() != image_ranges.len() {
                anyhow::bail!(
                    "LLaVA-Next has {} image hashes but {} image placeholders",
                    hashes.len(),
                    image_ranges.len()
                );
            }
            let new_ids = input_ids
                .iter()
                .map(|x| if *x < 0 { 0u32 } else { *x as u32 })
                .collect::<Vec<_>>();
            let new_prompt = tokenizer
                .decode(&new_ids, false)
                .map_err(|error| anyhow::Error::msg(error.to_string()))?;
            seq.set_initial_prompt(new_prompt);

            if seq.mm_features().is_empty() {
                seq.set_mm_features(build_mm_features_from_ranges(
                    &image_ranges,
                    &hashes,
                    MultimodalKind::Image,
                ));
            }

            seq.set_toks_and_reallocate(new_ids, paged_attn_metadata.as_mut());
            seq.multimodal.has_changed_prompt = true;
            toks.push(input_ids);
        }

        let metadata = if is_prompt {
            get_prompt_input(
                toks.iter().map(Vec::as_slice).collect(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )
        } else {
            get_completion_input(
                toks.iter().map(Vec::as_slice).collect(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
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
        } = metadata?;
        let image_hashes = input_seqs
            .iter()
            .flat_map(|seq| {
                let cached = seq.count_prefix_cached_mm_items();
                seq.image_hashes()
                    .unwrap_or_default()
                    .iter()
                    .copied()
                    .skip(cached)
            })
            .collect::<Vec<_>>();
        let selected_image_count = image_sizes.as_ref().map_or(0, Vec::len);
        if image_hashes.len() != selected_image_count
            || num_image_tokens_flat.len() != selected_image_count
            || num_image_samples.len() != selected_image_count
        {
            anyhow::bail!("LLaVA-Next image metadata does not match the selected media");
        }
        if let Some(pixels) = &pixel_values {
            if pixels.dim(0)? != num_image_samples.iter().sum::<usize>() {
                anyhow::bail!("LLaVA-Next pixel samples do not match image metadata");
            }
        }
        let packed_layout = if is_prompt && flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    anyhow::Error::msg("packed LLaVA-Next prefill requires logical query lengths")
                })?;
            let layout = llava_next_packed_layout(input_seqs, query_lens)?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "LLaVA-Next packed layout has {} tokens but input has {}",
                    layout.token_count(),
                    input.dim(1)?
                );
            }
            Some(layout)
        } else {
            None
        };
        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values,
                model_specific_args: Box::new(LLaVANextVisionSpecificArgs {
                    image_sizes,
                    num_image_tokens: Some(num_image_tokens_flat),
                    num_image_samples: Some(num_image_samples),
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
            }),
            seq_indices,
        })
    }
}

impl ImagePreProcessor for LLaVANextInputProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];
    fn preprocess(
        &self,
        images: Vec<image::DynamicImage>,
        videos: Vec<Vec<image::DynamicImage>>,
        config: &preprocessor_config::PreProcessorConfig,
        device: &candle_core::Device,
        (_, _): (usize, usize),
    ) -> candle_core::Result<image_processor::PreprocessedImages> {
        if images.len() > 1 {
            candle_core::bail!("Can only process one image per batch"); // This is no different from phi3_input_processor
        };
        assert!(videos.is_empty());

        let resized_size = *config.size.as_ref().unwrap().get("shortest_edge").unwrap() as usize;
        let image = images[0].clone();
        let original_size = image.dimensions();
        let image_grid_pinpoints = self.model_config.image_grid_pinpoints.clone().unwrap();
        let best_resolution = select_best_resolution(original_size, &image_grid_pinpoints);
        // Here I didn't use mistral_vision::Transform, because a lot transformations are before turning the image into a tensor
        let image_padded = resize_and_pad_image(&image, best_resolution);
        let filter = config.resampling.to_filter()?;
        let image_original_resize =
            image.resize_exact(resized_size as u32, resized_size as u32, filter);
        let mut samples = vec![image_original_resize];
        for patch in divide_to_samples(
            &image_padded,
            (
                *config.crop_size.as_ref().unwrap().get("width").unwrap(),
                *config.crop_size.as_ref().unwrap().get("height").unwrap(),
            ),
        ) {
            samples.push(patch);
        }
        let image_mean = config
            .image_mean
            .unwrap_or(Self::DEFAULT_MEAN)
            .map(|x| x as f32);
        let image_std = config
            .image_std
            .unwrap_or(Self::DEFAULT_STD)
            .map(|x| x as f32);
        let pixel_values = samples
            .iter()
            .map(|x| {
                LLaVAImageProcessor::process_one_image(
                    x,
                    config,
                    resized_size as u32,
                    filter,
                    DType::BF16,
                    device,
                    &image_mean,
                    &image_std,
                )
            })
            .collect::<Result<Vec<Tensor>>>()?;
        let pixel_values = Tensor::stack(&pixel_values, 0)?;

        Ok(image_processor::PreprocessedImages {
            pixel_values,
            pixel_attention_mask: None,
            image_sizes: Some((original_size.0 as usize, original_size.1 as usize)),
            num_img_tokens: Some(vec![LLaVANextInputProcessor::get_num_image_tokens(
                &self.model_config,
                original_size,
            )]),
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
            num_crops: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};

    use super::*;
    use crate::paged_attention::block_hash::MultimodalAttentionPolicy;
    use crate::vision_models::multimodal_layout::MultimodalEncoderOutputs;

    fn feature(item_range: Range<usize>) -> MultiModalFeature {
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range,
            hashes: vec![23],
            offset: 1,
            length: 2,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        }
    }

    #[test]
    fn packed_layout_splices_media_and_preserves_text_request() {
        let layout = PackedMultimodalLayout::new(&[
            RequestMultimodalLayout {
                sequence_id: 1,
                query: 0..4,
                items: llava_next_layout_items(&[feature(0..1)]).unwrap(),
            },
            RequestMultimodalLayout {
                sequence_id: 2,
                query: 0..2,
                items: vec![],
            },
        ])
        .unwrap();
        let text = Tensor::from_vec(
            vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            (1, 6, 2),
            &Device::Cpu,
        )
        .unwrap();
        let outputs: MultimodalEncoderOutputs = HashMap::from([(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 23,
            },
            vec![Tensor::from_vec(vec![20f32, 21., 30., 31.], (2, 2), &Device::Cpu).unwrap()],
        )]);

        assert_eq!(
            layout
                .splice_embeddings(&text, &outputs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0., 1., 20., 21., 30., 31., 6., 7., 8., 9., 10., 11.]
        );
    }

    #[test]
    fn suffix_only_marker_uses_local_coordinates() {
        let mut feature = feature(0..1);
        feature.offset = 4;
        feature.length = 1;

        assert_eq!(
            restore_llava_next_image_markers(&[7, 0, 8], &[feature], 3..6, 0..3).unwrap(),
            vec![7, -1, 8]
        );
    }

    #[test]
    fn packed_layout_rejects_grouped_image_feature() {
        assert!(llava_next_layout_items(&[feature(0..2)]).is_err());
    }
}
