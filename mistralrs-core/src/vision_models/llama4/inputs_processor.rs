#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    any::Any,
    collections::{HashMap, HashSet},
    ops::Range,
    sync::Arc,
};

use candle_core::{Context, Device, IndexOp, Result, Tensor, D};
use image::DynamicImage;
use itertools::Itertools;
use mistralrs_vision::{
    ApplyTensorTransforms, ApplyTransforms, Normalize, Rescale, TensorTransforms, ToTensorNoNorm,
    Transforms,
};
use ordered_float::NotNan;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::{MultiModalFeature, MultimodalKind},
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{build_mm_features_from_ranges, find_image_delimited_ranges, Sequence},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        multimodal_layout::{
            MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout,
            PackedMultimodalLayout, RequestMultimodalLayout,
        },
        preprocessor_config::PreProcessorConfig,
        processor_config::ProcessorConfig,
        ModelInputs,
    },
};

use super::Llama4ModelSpecificArgs;

pub(crate) const IMAGE_TOKEN: &str = "<|image|>";
const IMAGE_START: &str = "<|image_start|>";
const IMAGE_END: &str = "<|image_end|>";
const PATCH: &str = "<|patch|>";
const TILE_X_SEP: &str = "<|tile_x_separator|>";
const TILE_Y_SEP: &str = "<|tile_y_separator|>";

// Input processor
pub struct Llama4ImageProcessor {
    pub patch_size: usize,
    pub downsample_ratio: usize,
}

impl Llama4ImageProcessor {
    pub fn new(patch_size: Option<usize>, pixel_shuffle_ratio: Option<f32>) -> Self {
        Self {
            patch_size: patch_size.unwrap_or(14),
            downsample_ratio: (1. / pixel_shuffle_ratio.unwrap_or(0.5).powi(2)).round() as usize,
        }
    }
}

// Processor
pub struct Llama4Processor {
    patch_size: usize,
    downsample_ratio: usize,
}

impl Llama4Processor {
    pub fn new(cfg: &ProcessorConfig) -> Self {
        Self {
            patch_size: cfg.patch_size.unwrap_or(14),
            downsample_ratio: (1. / cfg.pixel_shuffle_ratio.unwrap_or(0.5).powi(2)).round()
                as usize,
        }
    }
}

impl Processor for Llama4Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Llama4ImageProcessor {
            patch_size: self.patch_size,
            downsample_ratio: self.downsample_ratio,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            IMAGE_START,
            IMAGE_END,
            PATCH,
            TILE_X_SEP,
            TILE_Y_SEP,
            IMAGE_TOKEN,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl Llama4ImageProcessor {
    fn prompt_split_image(&self, aspect_ratio: &Tensor, num_patches_per_chunk: usize) -> String {
        let mut img_string = IMAGE_START.to_string();
        let aspect_ratio = aspect_ratio.to_vec1::<u32>().unwrap();
        let (ratio_h, ratio_w) = (aspect_ratio[0] as usize, aspect_ratio[1] as usize);
        if ratio_h * ratio_w > 1 {
            for _yy in 0..ratio_h {
                for xx in 0..ratio_w {
                    img_string.push_str(&PATCH.repeat(num_patches_per_chunk));
                    if xx < ratio_w - 1 {
                        img_string.push_str(TILE_X_SEP);
                    }
                }
                img_string.push_str(TILE_Y_SEP);
            }
        }
        img_string.push_str(IMAGE_TOKEN);
        img_string.push_str(&PATCH.repeat(num_patches_per_chunk));
        img_string.push_str(IMAGE_END);
        img_string
    }
}

fn llama4_mm_features(
    tokens: &[u32],
    image_start_id: u32,
    image_end_id: u32,
    image_hashes: &[u64],
) -> Result<Vec<MultiModalFeature>> {
    let ranges = find_image_delimited_ranges(tokens, image_start_id, image_end_id);
    if ranges.len() != image_hashes.len() {
        candle_core::bail!(
            "Llama4 has {} image spans but {} image inputs",
            ranges.len(),
            image_hashes.len()
        );
    }
    Ok(build_mm_features_from_ranges(
        &ranges,
        image_hashes,
        MultimodalKind::Image,
    ))
}

fn llama4_tile_counts(aspect_ratios: &Tensor) -> Result<Vec<usize>> {
    let ratios = aspect_ratios.to_vec2::<u32>()?;
    ratios
        .into_iter()
        .map(|ratio| {
            if ratio.len() != 2 || ratio[0] == 0 || ratio[1] == 0 {
                candle_core::bail!("Llama4 image has an invalid aspect ratio");
            }
            let local_tiles = (ratio[0] as usize)
                .checked_mul(ratio[1] as usize)
                .ok_or_else(|| candle_core::Error::msg("Llama4 tile count overflow"))?;
            Ok(if local_tiles > 1 { local_tiles + 1 } else { 1 })
        })
        .collect()
}

fn llama4_tile_range(tile_counts: &[usize], item_range: Range<usize>) -> Result<Range<usize>> {
    if item_range.start > item_range.end || item_range.end > tile_counts.len() {
        candle_core::bail!("Llama4 image selection is outside the preprocessed images");
    }
    let start = tile_counts[..item_range.start]
        .iter()
        .try_fold(0usize, |sum, count| sum.checked_add(*count))
        .ok_or_else(|| candle_core::Error::msg("Llama4 tile offset overflow"))?;
    let len = tile_counts[item_range]
        .iter()
        .try_fold(0usize, |sum, count| sum.checked_add(*count))
        .ok_or_else(|| candle_core::Error::msg("Llama4 tile count overflow"))?;
    Ok(start..start + len)
}

fn llama4_item_selection(
    is_chunked: bool,
    active_item_range: Option<Range<usize>>,
    active_local_range: Option<Range<usize>>,
    cached_items: usize,
    encoded_items: usize,
    total_items: usize,
) -> Result<Option<(Range<usize>, Range<usize>)>> {
    if encoded_items > total_items || cached_items > total_items {
        candle_core::bail!("Llama4 image selection metadata is inconsistent");
    }
    let selection = if is_chunked {
        active_item_range.and_then(|original| {
            if encoded_items == total_items {
                Some((original.clone(), original))
            } else {
                active_local_range.map(|local| (local, original))
            }
        })
    } else {
        let retained_start = total_items - encoded_items;
        let original_start = cached_items.max(retained_start);
        (original_start < total_items).then_some((
            original_start - retained_start..encoded_items,
            original_start..total_items,
        ))
    };
    if let Some((encoded, original)) = &selection {
        if encoded.start > encoded.end
            || encoded.end > encoded_items
            || original.start > original.end
            || original.end > total_items
            || encoded.len() != original.len()
        {
            candle_core::bail!("Llama4 active image range is outside the retained images");
        }
    }
    Ok(selection)
}

fn llama4_layout_items(
    tokens: &[u32],
    features: &[MultiModalFeature],
    patch_token_id: u32,
) -> Result<Vec<MultimodalItemLayout>> {
    features
        .iter()
        .filter(|feature| feature.kind == MultimodalKind::Image)
        .map(|feature| {
            if feature.item_range.len() != 1 || feature.hashes.len() != 1 {
                candle_core::bail!("Llama4 image feature must describe exactly one image");
            }
            let placeholder = feature.offset..feature.end();
            let item_tokens = tokens.get(placeholder.clone()).ok_or_else(|| {
                candle_core::Error::msg("Llama4 image feature is outside the prompt")
            })?;
            let destination_positions = item_tokens
                .iter()
                .enumerate()
                .filter_map(|(offset, &token)| {
                    (token == patch_token_id).then_some(placeholder.start + offset)
                })
                .collect::<Vec<_>>();
            if destination_positions.is_empty() {
                candle_core::bail!("Llama4 image feature has no patch placeholders");
            }
            let source_positions = (0..destination_positions.len()).collect();
            MultimodalItemLayout::new(
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash: feature.hashes[0],
                },
                feature.item_range.start,
                placeholder,
                feature.attention_policy,
                vec![MultimodalEmbeddingMap::new(
                    destination_positions,
                    source_positions,
                    0,
                )?],
            )
        })
        .collect()
}

fn llama4_image_metadata(
    tokens: &[u32],
    features: &[MultiModalFeature],
    patch_token_id: u32,
    item_range: Range<usize>,
) -> Result<(Vec<u64>, Vec<usize>)> {
    let mut hashes = Vec::with_capacity(item_range.len());
    let mut token_counts = Vec::with_capacity(item_range.len());
    for item_index in item_range {
        let mut matching = features.iter().filter(|feature| {
            feature.kind == MultimodalKind::Image
                && feature.item_range == (item_index..item_index + 1)
        });
        let feature = matching
            .next()
            .ok_or_else(|| candle_core::Error::msg("Llama4 image feature is missing"))?;
        if matching.next().is_some() || feature.hashes.len() != 1 {
            candle_core::bail!("Llama4 image feature metadata is ambiguous");
        }
        let placeholder = feature.offset..feature.end();
        let item_tokens = tokens
            .get(placeholder)
            .ok_or_else(|| candle_core::Error::msg("Llama4 image feature is outside the prompt"))?;
        let token_count = item_tokens
            .iter()
            .filter(|&&token| token == patch_token_id)
            .count();
        if token_count == 0 {
            candle_core::bail!("Llama4 image feature has no patch placeholders");
        }
        hashes.push(feature.hashes[0]);
        token_counts.push(token_count);
    }
    Ok((hashes, token_counts))
}

fn llama4_prompt_query(seq: &Sequence, query_len: usize) -> Result<Range<usize>> {
    if let Some(query) = seq.active_prompt_query_range() {
        if query.len() != query_len {
            candle_core::bail!(
                "Llama4 active prompt has {} tokens but packed metadata has {query_len}",
                query.len()
            );
        }
        return Ok(query);
    }
    let token_count = seq.prompt_position_source_toks().len();
    let start = token_count
        .checked_sub(query_len)
        .ok_or_else(|| candle_core::Error::msg("Llama4 packed query is longer than the prompt"))?;
    Ok(start..token_count)
}

fn validate_llama4_image_spans(query: &Range<usize>, items: &[MultimodalItemLayout]) -> Result<()> {
    for item in items {
        let overlaps_query =
            item.placeholder.start < query.end && query.start < item.placeholder.end;
        if overlaps_query
            && (query.start > item.placeholder.start || query.end < item.placeholder.end)
        {
            candle_core::bail!(
                "Llama4 image item {} must be scheduled as a complete span",
                item.item_index
            );
        }
    }
    Ok(())
}

fn llama4_packed_layout(
    input_seqs: &[&mut Sequence],
    query_lens: &[usize],
    patch_token_id: u32,
) -> Result<PackedMultimodalLayout> {
    if input_seqs.len() != query_lens.len() {
        candle_core::bail!("Llama4 packed multimodal metadata length mismatch");
    }
    let requests = input_seqs
        .iter()
        .zip(query_lens)
        .map(|(seq, &query_len)| {
            let tokens = seq.prompt_position_source_toks();
            let query = llama4_prompt_query(seq, query_len)?;
            let items = llama4_layout_items(tokens, seq.mm_features(), patch_token_id)?;
            validate_llama4_image_spans(&query, &items)?;
            Ok(RequestMultimodalLayout {
                sequence_id: *seq.id(),
                query,
                items,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    PackedMultimodalLayout::new(&requests)
}

impl InputsProcessor for Llama4ImageProcessor {
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
        let tokenizer = tokenizer.ok_or_else(|| {
            anyhow::Error::msg("Llama4InputProcessor requires a specified tokenizer.")
        })?;
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        for seq in input_seqs.iter_mut() {
            self.prepare_sequence(
                &tokenizer,
                seq,
                config,
                device,
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
                "Llama4InputProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        if is_prompt {
            for seq in input_seqs.iter_mut() {
                self.prepare_sequence(
                    &tokenizer,
                    seq,
                    config,
                    device,
                    paged_attn_metadata.as_mut(),
                )?;
            }
        }

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

        let patch_token_id = tokenizer
            .token_to_id(PATCH)
            .ok_or_else(|| anyhow::Error::msg("Llama4 tokenizer is missing the patch token"))?;
        let mut pixel_values_accum = Vec::new();
        let mut image_hashes = Vec::new();
        let mut tile_counts = Vec::new();
        let mut image_token_counts = Vec::new();
        if is_prompt {
            for seq in input_seqs.iter() {
                let Some(pixel_values) = &seq.multimodal.cached_pixel_values else {
                    continue;
                };
                let sequence_tile_counts =
                    seq.multimodal.cached_num_crops.as_deref().ok_or_else(|| {
                        anyhow::Error::msg("Llama4 cached pixels are missing tile counts")
                    })?;
                let total_items = seq
                    .mm_features()
                    .iter()
                    .filter(|feature| feature.kind == MultimodalKind::Image)
                    .map(|feature| feature.item_range.end)
                    .max()
                    .unwrap_or(sequence_tile_counts.len());
                let active_item_range = seq.active_multimodal_item_range(MultimodalKind::Image);
                let available_items = seq.images().map_or(0, <[_]>::len);
                let active_local_range =
                    seq.active_local_multimodal_item_range(MultimodalKind::Image, available_items);
                let selection = llama4_item_selection(
                    seq.is_chunked_prefill_view(),
                    active_item_range,
                    active_local_range,
                    seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Image),
                    sequence_tile_counts.len(),
                    total_items,
                )?;
                let Some((encoded_item_range, original_item_range)) = selection else {
                    continue;
                };
                let tile_range =
                    llama4_tile_range(sequence_tile_counts, encoded_item_range.clone())?;
                if tile_range.end > pixel_values.dim(0)? {
                    anyhow::bail!("Llama4 selected tiles are outside the cached pixels");
                }
                let selected_tile_counts = sequence_tile_counts[encoded_item_range].to_vec();
                let (selected_hashes, selected_token_counts) = llama4_image_metadata(
                    seq.prompt_position_source_toks(),
                    seq.mm_features(),
                    patch_token_id,
                    original_item_range,
                )?;
                if selected_hashes.len() != selected_tile_counts.len() {
                    anyhow::bail!(
                        "Llama4 selected {} images but has {} tile counts",
                        selected_hashes.len(),
                        selected_tile_counts.len()
                    );
                }
                let patches_per_tile = (pixel_values.dim(D::Minus2)? / self.patch_size)
                    * (pixel_values.dim(D::Minus1)? / self.patch_size)
                    / self.downsample_ratio;
                for (&hash, (&tiles, &tokens)) in selected_hashes
                    .iter()
                    .zip(selected_tile_counts.iter().zip(&selected_token_counts))
                {
                    let expected_tokens = tiles
                        .checked_mul(patches_per_tile)
                        .ok_or_else(|| anyhow::Error::msg("Llama4 image token count overflow"))?;
                    if tokens != expected_tokens {
                        anyhow::bail!(
                            "Llama4 image {hash} has {tokens} placeholders but {expected_tokens} encoder rows"
                        );
                    }
                }
                pixel_values_accum.push(pixel_values.narrow(
                    0,
                    tile_range.start,
                    tile_range.len(),
                )?);
                image_hashes.extend(selected_hashes);
                tile_counts.extend(selected_tile_counts);
                image_token_counts.extend(selected_token_counts);
            }
        }
        let pixel_values = if pixel_values_accum.is_empty() {
            None
        } else {
            Some(Tensor::cat(&pixel_values_accum, 0)?)
        };
        let packed_layout = if is_prompt && flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    anyhow::Error::msg("packed Llama4 prefill requires logical query lengths")
                })?;
            let layout = llama4_packed_layout(input_seqs, query_lens, patch_token_id)?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "Llama4 packed layout has {} tokens but input has {}",
                    layout.token_count(),
                    input.dim(1)?
                );
            }
            Some(layout)
        } else {
            None
        };
        let packed_prefill = flash_meta.packed;

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(Llama4ModelSpecificArgs {
                image_hashes,
                tile_counts,
                image_token_counts,
                packed_layout,
                packed_prefill,
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

impl Llama4ImageProcessor {
    fn prepare_sequence(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut Sequence,
        config: &PreProcessorConfig,
        device: &Device,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let image_count = seq.images().map_or(0, <[_]>::len);
        if image_count == 0 {
            return Ok(());
        }

        let cached = match (
            &seq.multimodal.cached_pixel_values,
            &seq.multimodal.cached_spatial_shapes,
            &seq.multimodal.cached_num_crops,
        ) {
            (Some(pixel_values), Some(aspect_ratios), Some(tile_counts)) => Some((
                pixel_values.clone(),
                aspect_ratios.clone(),
                tile_counts.clone(),
            )),
            _ => None,
        };
        let (pixel_values, aspect_ratios, tile_counts) = if let Some(cached) = cached {
            cached
        } else {
            let PreprocessedImages {
                pixel_values,
                pixel_attention_mask: _,
                image_sizes: _,
                num_img_tokens: _,
                aspect_ratio_ids,
                aspect_ratio_mask: _,
                num_tiles: _,
                image_grid_thw: _,
                video_grid_thw: _,
                rows: _,
                cols: _,
                pixel_values_list: _,
                tgt_sizes: _,
                image_sizes_all: _,
                num_crops: _,
            } = self.preprocess(
                seq.clone_images()
                    .expect("Need to have images by this point."),
                vec![],
                config,
                device,
                (1, image_count),
            )?;
            let aspect_ratios = aspect_ratio_ids
                .ok_or_else(|| anyhow::Error::msg("Llama4 preprocessing omitted aspect ratios"))?;
            let tile_counts = llama4_tile_counts(&aspect_ratios)?;
            seq.multimodal.cached_pixel_values = Some(pixel_values.clone());
            seq.multimodal.cached_spatial_shapes = Some(aspect_ratios.clone());
            seq.multimodal.cached_num_crops = Some(tile_counts.clone());
            (pixel_values, aspect_ratios, tile_counts)
        };

        let total_tiles = tile_counts
            .iter()
            .try_fold(0usize, |sum, count| sum.checked_add(*count))
            .ok_or_else(|| anyhow::Error::msg("Llama4 tile count overflow"))?;
        if total_tiles != pixel_values.dim(0)? || aspect_ratios.dim(0)? != tile_counts.len() {
            anyhow::bail!("Llama4 preprocessed image metadata is inconsistent");
        }
        if seq.multimodal.has_changed_prompt {
            if seq.mm_features().is_empty() {
                anyhow::bail!("Llama4 expanded prompt is missing its image features");
            }
            return Ok(());
        }
        if tile_counts.len() != image_count {
            anyhow::bail!(
                "Llama4 preprocessing produced {} image groups for {image_count} images",
                tile_counts.len()
            );
        }

        let placeholder_count = seq.get_initial_prompt().matches(IMAGE_TOKEN).count();
        if placeholder_count != image_count {
            anyhow::bail!(
                "Llama4 prompt has {placeholder_count} image placeholders but {image_count} images"
            );
        }
        let image_h = pixel_values.dim(D::Minus2)?;
        let image_w = pixel_values.dim(D::Minus1)?;
        let patches_per_tile =
            (image_h / self.patch_size) * (image_w / self.patch_size) / self.downsample_ratio;
        if patches_per_tile == 0 {
            anyhow::bail!("Llama4 image produces no patch embeddings");
        }
        let mut prompt_parts = seq.get_initial_prompt().split(IMAGE_TOKEN);
        let mut prompt = prompt_parts.next().unwrap_or_default().to_string();
        for image_index in 0..image_count {
            prompt.push_str(
                &self.prompt_split_image(&aspect_ratios.i(image_index)?, patches_per_tile),
            );
            prompt.push_str(prompt_parts.next().ok_or_else(|| {
                anyhow::Error::msg("Llama4 image placeholder expansion is incomplete")
            })?);
        }
        if prompt_parts.next().is_some() {
            anyhow::bail!("Llama4 prompt has unmatched image placeholders");
        }

        let ids = tokenizer
            .encode_fast(prompt.clone(), false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let hashes = seq
            .image_hashes()
            .ok_or_else(|| anyhow::Error::msg("Llama4 images are missing content hashes"))?
            .to_vec();
        let image_start_id = tokenizer.token_to_id(IMAGE_START).ok_or_else(|| {
            anyhow::Error::msg("Llama4 tokenizer is missing the image start token")
        })?;
        let image_end_id = tokenizer
            .token_to_id(IMAGE_END)
            .ok_or_else(|| anyhow::Error::msg("Llama4 tokenizer is missing the image end token"))?;
        let features = llama4_mm_features(&ids, image_start_id, image_end_id, &hashes)?;
        seq.set_initial_prompt(prompt);
        seq.set_mm_features(features);
        seq.set_toks_and_reallocate(ids, paged_attn_metadata);
        seq.multimodal.has_changed_prompt = true;
        Ok(())
    }

    fn get_factors(dividend: u32) -> HashSet<u32> {
        let mut factors_set = HashSet::new();

        let sqrt = (dividend as f64).sqrt() as u32;
        for i in 1..=sqrt {
            if dividend.is_multiple_of(i) {
                factors_set.insert(i);
                factors_set.insert(dividend / i);
            }
        }

        factors_set
    }

    fn find_supported_resolutions(
        &self,
        max_num_chunks: usize,
        size: &HashMap<String, u32>,
    ) -> Result<Vec<(u32, u32)>> {
        let height = size["height"];
        let width = size["width"];
        if height != width {
            candle_core::bail!("Expected config size height==width ({height}!={width})");
        }

        let patch_size = height;

        let mut asp_map = HashMap::new();
        for chunk_size in (0..max_num_chunks).rev() {
            let factors = Self::get_factors(chunk_size as u32);
            let asp_ratios = factors
                .into_iter()
                .sorted()
                .map(|factors| (factors, chunk_size as u32 / factors));
            for (h, w) in asp_ratios {
                let ratio_float = h as f32 / w as f32;
                asp_map
                    .entry(NotNan::new(ratio_float).context("f32 is NaN")?)
                    .or_insert_with(Vec::new)
                    .push((h, w));
            }
        }

        // Get the resolutions multiplied by the patch size
        let possible_resolutions = asp_map
            .into_values()
            .flatten()
            .map(|(height, depth)| (height * patch_size, depth * patch_size))
            .collect::<Vec<_>>();

        Ok(possible_resolutions)
    }

    #[allow(clippy::type_complexity)]
    fn group_images_by_shape(
        &self,
        images: &[Tensor],
    ) -> Result<(
        HashMap<(usize, usize), Tensor>,
        HashMap<usize, ((usize, usize), usize)>,
    )> {
        let mut grouped_images = HashMap::new();
        let mut grouped_images_index = HashMap::new();
        for (i, image) in images.iter().enumerate() {
            let (_c, h, w) = image.dims3()?;
            let shape = (h, w);
            grouped_images
                .entry(shape)
                .or_insert_with(Vec::new)
                .push(image.clone());
            grouped_images_index.insert(i, (shape, grouped_images[&shape].len() - 1));
        }
        // Stack images with the same shape
        let mut grouped_images_stack = HashMap::new();
        for (shape, images) in grouped_images {
            grouped_images_stack.insert(shape, Tensor::stack(&images, 0)?);
        }

        Ok((grouped_images_stack, grouped_images_index))
    }

    fn get_best_fit(
        &self,
        (original_height, original_width): (u32, u32),
        possible_resolutions: Vec<(u32, u32)>,
        resize_to_max_canvas: bool,
    ) -> Result<(u32, u32)> {
        // All possible reslns h/w
        let (target_heights, target_widths): (Vec<u32>, Vec<u32>) =
            possible_resolutions.iter().copied().unzip();

        // Scaling factors to resize image without distortion
        let scale_w = target_widths
            .iter()
            .map(|tw| *tw as f32 / original_width as f32);
        let scale_h = target_heights
            .iter()
            .map(|th| *th as f32 / original_height as f32);

        // Min scale between w and h (limiting size -> no distortion)
        let scales = scale_w.zip(scale_h).map(|(w, h)| if h > w { w } else { h });

        // Filter only scales that allow upscaling
        let upscaling_options = scales
            .clone()
            .filter(|s| *s >= 1.)
            .map(|x| NotNan::new(x).unwrap())
            .collect::<Vec<_>>();
        let downscaling_options = scales
            .clone()
            .filter(|s| *s < 1.)
            .map(|x| NotNan::new(x).unwrap())
            .collect::<Vec<_>>();
        let selected_scale = if !upscaling_options.is_empty() {
            if resize_to_max_canvas {
                upscaling_options.into_iter().max().unwrap().into_inner()
            } else {
                upscaling_options.into_iter().min().unwrap().into_inner()
            }
        } else {
            // No upscaling; get min downscaling (max scale for scales < 1)
            downscaling_options.into_iter().max().unwrap().into_inner()
        };

        // All reslns that support this scaling factor
        // Ex. can upscale 224x224, 224x448, 224x672 without distortion
        // If there are multiple resolutions, get the one with minimum area to reduce padding.
        // Sort by increasing areas and take 1.
        let chosen_canvas = possible_resolutions
            .into_iter()
            .zip(scales)
            .filter_map(|(possible, scale)| {
                if scale == selected_scale {
                    Some(possible)
                } else {
                    None
                }
            })
            .sorted_by_key(|(h, w)| h * w)
            .take(1)
            .collect::<Vec<_>>()[0];

        Ok(chosen_canvas)
    }

    fn get_max_res_without_distortion(
        &self,
        image_size: (u32, u32),
        target_size: (u32, u32),
    ) -> (u32, u32) {
        let (original_height, original_width) = image_size;
        let (target_height, target_width) = target_size;

        let scale_w = target_width as f64 / original_width as f64;
        let scale_h = target_height as f64 / original_height as f64;

        if scale_w < scale_h {
            let new_width = target_width;
            // Calculate new height and ensure it doesn't exceed target_height
            let new_height = std::cmp::min(
                (original_height as f64 * scale_w).floor() as u32,
                target_height,
            );
            (new_height, new_width)
        } else {
            let new_height = target_height;
            // Calculate new width and ensure it doesn't exceed target_width
            let new_width = std::cmp::min(
                (original_width as f64 * scale_h).floor() as u32,
                target_width,
            );
            (new_height, new_width)
        }
    }

    fn split_to_tiles(
        &self,
        images: &Tensor,
        num_tiles_h: usize,
        num_tiles_w: usize,
    ) -> Result<Tensor> {
        let (bs, c, h, w) = images.dims4()?;
        let mut images = images.reshape((
            bs,
            c,
            num_tiles_h,
            h / num_tiles_h,
            num_tiles_w,
            w / num_tiles_w,
        ))?;
        images = images.permute((0, 2, 4, 1, 3, 5))?.contiguous()?;
        images.reshape((
            bs,
            num_tiles_h * num_tiles_w,
            c,
            h / num_tiles_h,
            w / num_tiles_w,
        ))
    }

    fn reorder_images(
        &self,
        processed_images: HashMap<(usize, usize), Tensor>,
        grouped_images_index: HashMap<usize, ((usize, usize), usize)>,
    ) -> Result<Vec<Tensor>> {
        grouped_images_index
            .values()
            .map(|(k, v)| processed_images[k].i(*v))
            .collect::<Result<Vec<Tensor>>>()
    }
}

impl ImagePreProcessor for Llama4ImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

    fn preprocess(
        &self,
        images_d: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        assert!(videos.is_empty());

        let max_patches = config.max_patches.unwrap_or(16);
        let size = config.size.clone().unwrap_or(HashMap::from_iter([
            ("height".to_string(), 336),
            ("width".to_string(), 336),
        ]));
        let resize_to_max_canvas = config.resize_to_max_canvas.unwrap_or(false);
        let do_rescale = config.do_rescale.unwrap_or(true);
        let do_normalize = config.do_normalize.unwrap_or(true);

        let possible_resolutions = self.find_supported_resolutions(max_patches, &size)?;

        let mut images = Vec::new();
        for mut image in images_d {
            // Convert to rgb, default to true
            if config.do_convert_rgb.unwrap_or(true) {
                image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            let to_tensor_rescale = Transforms {
                input: &ToTensorNoNorm,
                inner_transforms: &[],
            };
            let image = image.apply(to_tensor_rescale, device)?;
            images.push(image);
        }

        let (grouped_images, grouped_images_index) = self.group_images_by_shape(&images)?;

        let mut grouped_processed_images = HashMap::new();
        let mut grouped_aspect_ratios = HashMap::new();
        for (shape, stacked_images) in grouped_images {
            let image_size = (
                stacked_images.dim(D::Minus2)? as u32,
                stacked_images.dim(D::Minus1)? as u32,
            );
            let target_size = self.get_best_fit(
                image_size,
                possible_resolutions.clone(),
                resize_to_max_canvas,
            )?;
            // If target_size requires upscaling, we might want to limit the upscaling to max_upscaling_size
            let max_upscaling_size = if resize_to_max_canvas {
                None
            } else {
                Some(size["height"])
            };
            let target_size_without_distortion =
                if let Some(max_upscaling_size) = max_upscaling_size {
                    let nt_h = image_size.0.max(max_upscaling_size).min(target_size.0);
                    let nt_w = image_size.1.max(max_upscaling_size).min(target_size.1);
                    (nt_h, nt_w)
                } else {
                    candle_core::bail!("Currently resize_to_max_canvas is assumed!");
                };

            // Resize to target_size while preserving aspect ratio
            let new_size_without_distortion =
                self.get_max_res_without_distortion(image_size, target_size_without_distortion);
            let mut processed_images = stacked_images.interpolate2d(
                new_size_without_distortion.0.max(1) as usize,
                new_size_without_distortion.1.max(1) as usize,
            )?;

            // Pad to target_size to be able to split into tiles
            processed_images = {
                let (target_h, target_w) = target_size;
                let (h, w) = (
                    processed_images.dim(D::Minus2)?,
                    processed_images.dim(D::Minus1)?,
                );
                let paste_x_r = target_w as usize - w;
                let paste_y_r = target_h as usize - h;
                processed_images
                    .pad_with_zeros(D::Minus2, 0, paste_y_r)?
                    .pad_with_zeros(D::Minus1, 0, paste_x_r)?
            };

            let rescale_and_norm_transforms = TensorTransforms {
                inner_transforms: &[
                    &do_rescale.then_some(Rescale {
                        factor: config.rescale_factor,
                    }),
                    &do_normalize.then_some(Normalize {
                        mean: config.image_mean.unwrap_or(Self::DEFAULT_MEAN).to_vec(),
                        std: config.image_std.unwrap_or(Self::DEFAULT_STD).to_vec(),
                    }),
                ],
            };
            processed_images = <Tensor as ApplyTensorTransforms>::apply(
                &processed_images,
                rescale_and_norm_transforms,
                device,
            )?;

            let (ratio_h, ratio_w) = (
                target_size.0 / size["height"],
                target_size.1 / size["width"],
            );
            // Split into tiles
            processed_images =
                self.split_to_tiles(&processed_images, ratio_h as usize, ratio_w as usize)?;
            grouped_processed_images.insert(shape, processed_images.clone());
            grouped_aspect_ratios.insert(
                shape,
                Tensor::new(vec![vec![ratio_h, ratio_w]; stacked_images.dim(0)?], device)?,
            );

            // Add a global tile to the processed tile if there are more than one tiles
            if ratio_h * ratio_w > 1 {
                let mut global_tiles = stacked_images
                    .interpolate2d(size["height"] as usize, size["width"] as usize)?;
                global_tiles = <Tensor as ApplyTensorTransforms>::apply(
                    &global_tiles,
                    rescale_and_norm_transforms,
                    device,
                )?;
                grouped_processed_images.insert(
                    shape,
                    Tensor::cat(&[processed_images, global_tiles.unsqueeze(1)?], 1)?,
                );
            }
        }

        let processed_images =
            self.reorder_images(grouped_processed_images, grouped_images_index.clone())?;
        let aspect_ratios_list =
            self.reorder_images(grouped_aspect_ratios, grouped_images_index.clone())?;

        let processed_images = Tensor::cat(&processed_images, 0)?;
        let aspect_ratios = Tensor::stack(&aspect_ratios_list, 0)?;

        Ok(PreprocessedImages {
            pixel_values: processed_images,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: Some(aspect_ratios),
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
    use super::*;
    use crate::vision_models::multimodal_layout::MultimodalEncoderOutputs;

    #[test]
    fn packed_layout_handles_unequal_media_and_text_rows() {
        let tokens = [1, 3, 9, 3, 2];
        let features = llama4_mm_features(&tokens, 1, 2, &[17]).unwrap();
        let layout = PackedMultimodalLayout::new(&[
            RequestMultimodalLayout {
                sequence_id: 1,
                query: 0..tokens.len(),
                items: llama4_layout_items(&tokens, &features, 3).unwrap(),
            },
            RequestMultimodalLayout {
                sequence_id: 2,
                query: 0..2,
                items: vec![],
            },
        ])
        .unwrap();
        let text = Tensor::zeros((1, 7, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let image = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &Device::Cpu).unwrap();
        let outputs: MultimodalEncoderOutputs = HashMap::from([(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 17,
            },
            vec![image],
        )]);

        let result = layout
            .splice_embeddings(&text, &outputs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            result,
            vec![0., 0., 1., 2., 0., 0., 3., 4., 0., 0., 0., 0., 0., 0.]
        );
    }

    #[test]
    fn tile_selection_handles_cached_prefix_and_active_chunk_items() {
        assert_eq!(llama4_tile_range(&[1, 5, 3], 1..3).unwrap(), 1..9);
        assert_eq!(llama4_tile_range(&[1, 5, 3], 2..3).unwrap(), 6..9);
        assert!(llama4_tile_range(&[1, 5], 1..3).is_err());
        assert_eq!(
            llama4_item_selection(false, None, None, 1, 2, 3).unwrap(),
            Some((0..2, 1..3))
        );
        assert_eq!(
            llama4_item_selection(true, Some(2..3), Some(0..1), 0, 1, 3).unwrap(),
            Some((0..1, 2..3))
        );
    }

    #[test]
    fn image_metadata_selects_the_requested_original_items() {
        let tokens = [1, 3, 2, 8, 1, 3, 3, 2];
        let features = llama4_mm_features(&tokens, 1, 2, &[11, 13]).unwrap();

        assert_eq!(
            llama4_image_metadata(&tokens, &features, 3, 1..2).unwrap(),
            (vec![13], vec![2])
        );
    }

    #[test]
    fn malformed_image_cardinality_is_rejected() {
        assert!(llama4_mm_features(&[1, 3, 2], 1, 2, &[11, 13]).is_err());
        let features = llama4_mm_features(&[1, 9, 2], 1, 2, &[11]).unwrap();
        assert!(llama4_layout_items(&[1, 9, 2], &features, 3).is_err());
    }

    #[test]
    fn image_spans_cannot_be_partially_packed() {
        let tokens = [1, 3, 9, 3, 2, 8];
        let features = llama4_mm_features(&tokens, 1, 2, &[11]).unwrap();
        let items = llama4_layout_items(&tokens, &features, 3).unwrap();

        assert!(validate_llama4_image_spans(&(0..4), &items).is_err());
        assert!(validate_llama4_image_spans(&(0..5), &items).is_ok());
        assert!(validate_llama4_image_spans(&(5..6), &items).is_ok());
    }
}
