#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::attention::AttentionMask;
use crate::layers_masker::CausalMaskConfig;
use std::{
    any::Any,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use mistralrs_quant::{NonZeroOp, ShardedVarBuilder};
use text::Qwen3VLTextModel;
use vision::Qwen3VLVisionModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    layers::CausalMasker,
    layers_masker::PastKvLenCache,
    paged_attention::{
        block_hash::MultimodalKind,
        encoder_cache::{CacheModality, EncoderCacheManager},
        AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata,
    },
    vision_models::multimodal_layout::{
        MultimodalEncoderKey, MultimodalEncoderOutputs, PackedMultimodalLayout,
    },
};

pub(crate) mod config;
pub(crate) mod inputs_processor;
mod text;
pub(crate) mod vision;

pub(crate) use config::Config;
pub(crate) use inputs_processor::Qwen3VLProcessor;

pub struct Qwen3VLModel {
    text: Qwen3VLTextModel,
    vision: Qwen3VLVisionModel,
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

/// Compute 3D MRoPE position IDs and position deltas for Qwen3 VL models.
/// Shared between Qwen3VL models.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_rope_index(
    input_ids: &Tensor,
    image_grid_thw: Option<&Tensor>,
    video_grid_thw: Option<&Tensor>,
    attention_mask: &AttentionMask,
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
) -> Result<(Tensor, Tensor)> {
    if image_grid_thw.is_some() || video_grid_thw.is_some() {
        let batch = input_ids.dim(0)?;
        let seq_len = input_ids.dim(1)?;
        let device = input_ids.device().clone();

        let attention_mask_tensor = match attention_mask {
            AttentionMask::Custom(mask) => mask.clone(),
            _ => Tensor::ones((batch, seq_len), DType::F32, &device)?,
        };
        let attention_mask_vec = attention_mask_tensor.to_vec2::<f32>()?;
        let input_ids_vec = input_ids.to_vec2::<u32>()?;

        let image_grid_data = if let Some(grid) = image_grid_thw {
            let raw = grid.to_vec2::<u32>()?;
            let mut data = Vec::with_capacity(raw.len());
            for row in raw {
                if row.len() != 3 {
                    candle_core::bail!("image_grid_thw entries must have length 3");
                }
                data.push([row[0], row[1], row[2]]);
            }
            Some(data)
        } else {
            None
        };

        let video_grid_data = if let Some(grid) = video_grid_thw {
            let raw = grid.to_vec2::<u32>()?;
            let mut data = Vec::with_capacity(raw.len());
            for row in raw {
                if row.len() != 3 {
                    candle_core::bail!("video_grid_thw entries must have length 3");
                }
                // Timestamps split each video into per-frame vision spans, so the grid splits too.
                for _ in 0..row[0] {
                    data.push([1, row[1], row[2]]);
                }
            }
            Some(data)
        } else {
            None
        };

        let mut image_index = 0usize;
        let mut video_index = 0usize;
        let merge_size = spatial_merge_size as u32;

        let mut position_ids_data = vec![vec![vec![1i64; seq_len]; batch]; 3];
        let mut mrope_position_deltas = Vec::with_capacity(batch);

        for batch_idx in 0..batch {
            let mask_row = &attention_mask_vec[batch_idx];
            let input_row = &input_ids_vec[batch_idx];

            let mut valid_indices = Vec::new();
            let mut filtered_tokens = Vec::new();
            for (idx, (&token, &mask_val)) in input_row.iter().zip(mask_row.iter()).enumerate() {
                if mask_val != 0.0 {
                    valid_indices.push(idx);
                    filtered_tokens.push(token);
                }
            }

            let mut positions_for_valid: Vec<[i64; 3]> = Vec::with_capacity(valid_indices.len());
            let mut max_position_value: Option<i64> = None;

            let mut spans = Vec::new();
            let mut span_idx = 0usize;
            while span_idx < filtered_tokens.len() {
                if filtered_tokens[span_idx] == vision_start_token_id {
                    let mut end_idx = span_idx + 1;
                    while end_idx < filtered_tokens.len()
                        && filtered_tokens[end_idx] != vision_end_token_id
                    {
                        end_idx += 1;
                    }
                    if end_idx == filtered_tokens.len() {
                        candle_core::bail!(
                            "vision_start_token_id without matching vision_end_token_id"
                        );
                    }
                    spans.push((span_idx, end_idx));
                    span_idx = end_idx + 1;
                } else {
                    span_idx += 1;
                }
            }

            let mut max_last_llm_pos_ids: Option<i64> = None;
            let mut cursor = 0usize;

            for (start_idx, end_idx) in spans {
                if start_idx + 1 > end_idx {
                    continue;
                }

                let placeholder_start = filtered_tokens[start_idx + 1..end_idx]
                    .iter()
                    .enumerate()
                    .find_map(|(offset, &tok)| {
                        (tok == image_token_id || tok == video_token_id)
                            .then_some(offset + start_idx + 1)
                    });
                let placeholder_start = match placeholder_start {
                    Some(pos) => pos,
                    None => {
                        candle_core::bail!("vision span missing image/video placeholder tokens");
                    }
                };

                let text_len = placeholder_start.saturating_sub(cursor);
                let st_idx = max_last_llm_pos_ids.unwrap_or(0);
                for offset in 0..text_len {
                    let pos_val = st_idx + offset as i64;
                    positions_for_valid.push([pos_val, pos_val, pos_val]);
                    max_position_value = Some(match max_position_value {
                        Some(current) => current.max(pos_val),
                        None => pos_val,
                    });
                }

                let placeholder_token_id = filtered_tokens[placeholder_start];
                let placeholder_slice = &filtered_tokens[placeholder_start..end_idx];
                if placeholder_slice.is_empty() {
                    candle_core::bail!("vision span placeholder slice is empty");
                }
                if !placeholder_slice
                    .iter()
                    .all(|&tok| tok == placeholder_token_id)
                {
                    candle_core::bail!("Mixed placeholder tokens found within a vision span");
                }
                let placeholder_len = placeholder_slice.len();

                let (grid_t, grid_h, grid_w) = match placeholder_token_id {
                    id if id == image_token_id => {
                        let Some(ref img_grid) = image_grid_data else {
                            candle_core::bail!("image_grid_thw required for image placeholders");
                        };
                        if image_index >= img_grid.len() {
                            candle_core::bail!(
                                "Not enough image_grid_thw entries for placeholders"
                            );
                        }
                        let grid = img_grid[image_index];
                        image_index += 1;
                        if merge_size == 0 || grid[1] % merge_size != 0 || grid[2] % merge_size != 0
                        {
                            candle_core::bail!(
                                "image grid dimensions must be divisible by spatial_merge_size"
                            );
                        }
                        (
                            grid[0] as usize,
                            (grid[1] / merge_size) as usize,
                            (grid[2] / merge_size) as usize,
                        )
                    }
                    id if id == video_token_id => {
                        let Some(ref vid_grid) = video_grid_data else {
                            candle_core::bail!("video_grid_thw required for video placeholders");
                        };
                        if video_index >= vid_grid.len() {
                            candle_core::bail!(
                                "Not enough video_grid_thw entries for placeholders"
                            );
                        }
                        let grid = vid_grid[video_index];
                        video_index += 1;
                        if merge_size == 0 || grid[1] % merge_size != 0 || grid[2] % merge_size != 0
                        {
                            candle_core::bail!(
                                "video grid dimensions must be divisible by spatial_merge_size"
                            );
                        }
                        (
                            grid[0] as usize,
                            (grid[1] / merge_size) as usize,
                            (grid[2] / merge_size) as usize,
                        )
                    }
                    other => {
                        candle_core::bail!("Unexpected placeholder token id {other}");
                    }
                };

                if grid_t == 0 || grid_h == 0 || grid_w == 0 {
                    candle_core::bail!("Zero-sized grid encountered in vision span");
                }

                let expected_len = grid_t * grid_h * grid_w;
                if placeholder_len != expected_len {
                    candle_core::bail!(
                        "Placeholder token count {placeholder_len} does not match expected {expected_len}"
                    );
                }

                let base_offset = st_idx + text_len as i64;
                for t in 0..grid_t {
                    for h in 0..grid_h {
                        for w in 0..grid_w {
                            let t_pos = base_offset + t as i64;
                            let h_pos = base_offset + h as i64;
                            let w_pos = base_offset + w as i64;
                            positions_for_valid.push([t_pos, h_pos, w_pos]);
                            max_position_value = Some(match max_position_value {
                                Some(current) => current.max(t_pos).max(h_pos).max(w_pos),
                                None => t_pos.max(h_pos).max(w_pos),
                            });
                        }
                    }
                }

                let max_dim = std::cmp::max(grid_t, std::cmp::max(grid_h, grid_w)) as i64;
                max_last_llm_pos_ids = Some(base_offset + max_dim);
                cursor = placeholder_start + placeholder_len;
            }

            if cursor < filtered_tokens.len() {
                let text_len = filtered_tokens.len() - cursor;
                let st_idx = max_last_llm_pos_ids.unwrap_or(0);
                for offset in 0..text_len {
                    let pos_val = st_idx + offset as i64;
                    positions_for_valid.push([pos_val, pos_val, pos_val]);
                    max_position_value = Some(match max_position_value {
                        Some(current) => current.max(pos_val),
                        None => pos_val,
                    });
                }
            }

            if positions_for_valid.len() != valid_indices.len() {
                candle_core::bail!(
                    "Mismatch between computed positions ({}) and valid tokens ({})",
                    positions_for_valid.len(),
                    valid_indices.len()
                );
            }

            for (pos_idx, &seq_idx) in valid_indices.iter().enumerate() {
                let [p0, p1, p2] = positions_for_valid[pos_idx];
                position_ids_data[0][batch_idx][seq_idx] = p0;
                position_ids_data[1][batch_idx][seq_idx] = p1;
                position_ids_data[2][batch_idx][seq_idx] = p2;
            }

            let seq_total_len = input_row.len() as i64;
            let max_position_value = max_position_value.unwrap_or(0);
            mrope_position_deltas.push(max_position_value + 1 - seq_total_len);
        }

        let mut flat_positions = Vec::with_capacity(3 * batch * seq_len);
        for plane in position_ids_data.iter().take(3) {
            for row in plane.iter().take(batch) {
                flat_positions.extend_from_slice(row);
            }
        }
        let position_ids = Tensor::from_vec(flat_positions, (3, batch, seq_len), &device)?;
        let mrope_position_deltas = Tensor::from_vec(mrope_position_deltas, (batch, 1), &device)?;

        Ok((position_ids, mrope_position_deltas))
    } else if let AttentionMask::Custom(attention_mask) = attention_mask {
        // candle's cumsum materializes an SxS triangular matrix; at long context that overflows
        // u32 kernel indexing and corrupts device memory, so compute positions host-side.
        let (batch, seq_len) = attention_mask.dims2()?;
        let mask = attention_mask.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let mut flat_positions = vec![0i64; batch * seq_len];
        let mut mrope_position_deltas = Vec::with_capacity(batch);
        for (batch_idx, row) in mask.iter().enumerate() {
            let mut count = 0i64;
            let mut max_position = 0i64;
            for (seq_idx, &mask_val) in row.iter().enumerate() {
                let position = if mask_val != 0.0 {
                    count += 1;
                    count - 1
                } else {
                    1
                };
                max_position = max_position.max(position);
                flat_positions[batch_idx * seq_len + seq_idx] = position;
            }
            mrope_position_deltas.push(max_position + 1 - seq_len as i64);
        }
        let device = attention_mask.device();
        let position_ids =
            Tensor::from_vec(flat_positions, (1, batch, seq_len), device)?.repeat((3, 1, 1))?;
        let mrope_position_deltas = Tensor::from_vec(mrope_position_deltas, (batch, 1), device)?;

        Ok((position_ids, mrope_position_deltas))
    } else {
        let position_ids = Tensor::arange(0i64, input_ids.dim(1)? as i64, input_ids.device())?
            .reshape((1, 1, ()))?
            .repeat((3, input_ids.dim(0)?, 1))?;
        let mrope_position_deltas =
            Tensor::zeros((input_ids.dim(0)?, 1), DType::I64, input_ids.device())?;

        Ok((position_ids, mrope_position_deltas))
    }
}

pub(crate) struct VisualEncoder<'a> {
    vision: &'a Qwen3VLVisionModel,
    cache: &'a Arc<Mutex<EncoderCacheManager>>,
    spatial_merge_size: usize,
}

impl<'a> VisualEncoder<'a> {
    pub(crate) fn new(
        vision: &'a Qwen3VLVisionModel,
        cache: &'a Arc<Mutex<EncoderCacheManager>>,
        spatial_merge_size: usize,
    ) -> Self {
        Self {
            vision,
            cache,
            spatial_merge_size,
        }
    }

    pub(crate) fn encode(
        &self,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
        hashes: &[u64],
        modality: CacheModality,
    ) -> Result<Vec<Vec<Tensor>>> {
        let grid_data = grid_thw.to_vec2::<u32>()?;
        if !hashes.is_empty() && hashes.len() != grid_data.len() {
            candle_core::bail!(
                "Qwen encoder has {} hashes but {} grid rows",
                hashes.len(),
                grid_data.len()
            );
        }
        let patches_per_item = grid_data
            .iter()
            .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
            .collect::<Vec<_>>();
        let output_tokens_per_item = grid_data
            .iter()
            .map(|row| {
                row[0] as usize
                    * (row[1] as usize / self.spatial_merge_size)
                    * (row[2] as usize / self.spatial_merge_size)
            })
            .collect::<Vec<_>>();
        let mut per_item = vec![None; grid_data.len()];
        let mut misses = Vec::new();
        if hashes.is_empty() {
            misses.extend(0..grid_data.len());
        } else {
            let mut cache = self.cache.lock().expect("encoder cache lock poisoned");
            for (index, &hash) in hashes.iter().enumerate() {
                if let Some(outputs) = cache.get(modality, hash) {
                    per_item[index] = Some(outputs);
                } else {
                    misses.push(index);
                }
            }
        }
        if !misses.is_empty() {
            let mut pixel_offset = 0usize;
            let mut miss_pixels = Vec::with_capacity(misses.len());
            let mut miss_grids = Vec::with_capacity(misses.len());
            for (index, &patch_count) in patches_per_item.iter().enumerate() {
                if misses.contains(&index) {
                    miss_pixels.push(pixel_values.narrow(0, pixel_offset, patch_count)?);
                    miss_grids.push(grid_thw.i(index)?);
                }
                pixel_offset += patch_count;
            }
            let (main, deepstack) = self.vision.forward(
                &Tensor::cat(&miss_pixels, 0)?,
                &Tensor::stack(&miss_grids, 0)?,
            )?;
            let mut output_offset = 0usize;
            let mut cache = self.cache.lock().expect("encoder cache lock poisoned");
            for &index in &misses {
                let output_len = output_tokens_per_item[index];
                let mut outputs = vec![main.narrow(0, output_offset, output_len)?];
                for layer in &deepstack {
                    outputs.push(layer.narrow(0, output_offset, output_len)?);
                }
                output_offset += output_len;
                if let Some(&hash) = hashes.get(index) {
                    cache.insert(modality, hash, outputs.clone());
                }
                per_item[index] = Some(outputs);
            }
        }
        per_item
            .into_iter()
            .map(|outputs| {
                outputs
                    .ok_or_else(|| candle_core::Error::msg("Qwen encoder item is missing outputs"))
            })
            .collect()
    }
}

pub(crate) fn concatenate_visual_items(per_item: &[Vec<Tensor>]) -> Result<(Tensor, Vec<Tensor>)> {
    let output_count = per_item
        .first()
        .ok_or_else(|| candle_core::Error::msg("Qwen visual batch is empty"))?
        .len();
    if per_item.iter().any(|outputs| outputs.len() != output_count) {
        candle_core::bail!("Qwen visual items have different DeepStack output counts");
    }
    let main = Tensor::cat(
        &per_item
            .iter()
            .map(|outputs| outputs[0].clone())
            .collect::<Vec<_>>(),
        0,
    )?;
    let mut deepstack = Vec::with_capacity(output_count - 1);
    for output in 1..output_count {
        deepstack.push(Tensor::cat(
            &per_item
                .iter()
                .map(|outputs| outputs[output].clone())
                .collect::<Vec<_>>(),
            0,
        )?);
    }
    Ok((main, deepstack))
}

pub(crate) fn insert_current_visual_outputs(
    encoder_outputs: &mut MultimodalEncoderOutputs,
    kind: MultimodalKind,
    hashes: &[u64],
    outputs: Vec<Vec<Tensor>>,
) -> Result<()> {
    if hashes.len() != outputs.len() {
        candle_core::bail!(
            "Qwen has {} current {kind:?} outputs but {} hashes",
            outputs.len(),
            hashes.len()
        );
    }
    for (&hash, outputs) in hashes.iter().zip(outputs) {
        encoder_outputs.insert(MultimodalEncoderKey { kind, hash }, outputs);
    }
    Ok(())
}

impl Qwen3VLModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        // Support both original HuggingFace naming (model.visual.*) and MLX naming (vision_tower.*)
        let vision_vb = if vb.contains_tensor("vision_tower.patch_embed.proj.weight") {
            vb.pp("vision_tower")
        } else {
            vb.pp("model").pp("visual")
        }
        .without_lora_registry();
        let vision = Qwen3VLVisionModel::new(
            &cfg.vision_config,
            vision_vb.set_device(normal_loading_metadata.real_device.clone()),
        )?;
        // Use top-level quantization_config if present, otherwise fall back to text_config's
        let mut text_config = cfg.text_config.clone();
        if cfg.quantization_config.is_some() {
            text_config.quantization_config = cfg.quantization_config.clone();
        }
        let text = Qwen3VLTextModel::new(
            &text_config,
            vb.clone(),
            cfg.tie_word_embeddings,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            vision,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
            vision_start_token_id: cfg.vision_start_token_id,
            vision_end_token_id: cfg.vision_end_token_id,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        pixel_values: Option<Tensor>,
        pixel_values_videos: Option<Tensor>,
        image_grid_thw: Option<Tensor>,
        video_grid_thw: Option<Tensor>,
        rope_img_grid_thw: Option<Tensor>,
        rope_vid_grid_thw: Option<Tensor>,
        seqlens: Vec<usize>,
        continuous_img_pad: Vec<Vec<(usize, usize)>>,
        continuous_vid_pad: Vec<Vec<(usize, usize)>>,
        image_hashes: &[u64],
        video_hashes: &[u64],
        packed_layout: Option<&PackedMultimodalLayout>,
        prompt_position_ids: Option<&Tensor>,
        ctx: &ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let seqlen_offsets = ctx.seqlen_offsets();
        let mut attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            self.text.dtype,
            &CausalMaskConfig {
                sliding_window: self.text.cfg.sliding_window,
                ..Default::default()
            },
        )?;
        let is_first_chunk = ctx.is_first_prompt_chunk();
        attention_mask = if is_first_chunk {
            attention_mask
        } else {
            AttentionMask::None
        };

        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let (batch_size, seq_len, hidden_dim) = input_embeds.dims3()?;
        let device = input_embeds.device().clone();

        let mut image_mask_opt: Option<Tensor> = None;
        let mut video_mask_opt: Option<Tensor> = None;
        let mut deepstack_image_opt: Option<Vec<Tensor>> = None;
        let mut deepstack_video_opt: Option<Vec<Tensor>> = None;
        let mut packed_encoder_outputs = MultimodalEncoderOutputs::new();

        if let Some(pixel_values) = &pixel_values {
            let Some(image_grid_thw_ref) = image_grid_thw.as_ref() else {
                candle_core::bail!("pixel_values require image_grid_thw");
            };
            let mut pixel_values = pixel_values.clone();
            let ndim = pixel_values.dims().len();
            if ndim > 2 {
                let last_dim = pixel_values.dim(ndim - 1)?;
                pixel_values = pixel_values.reshape(((), last_dim))?;
            }

            let per_image = if image_hashes.is_empty() {
                None
            } else {
                Some(
                    VisualEncoder::new(&self.vision, &self.encoder_cache, self.spatial_merge_size)
                        .encode(
                            &pixel_values,
                            image_grid_thw_ref,
                            image_hashes,
                            CacheModality::Image,
                        )?,
                )
            };
            let (image_embeds, deepstack_image_embeds) = match &per_image {
                Some(outputs) => concatenate_visual_items(outputs)?,
                None => self.vision.forward(&pixel_values, image_grid_thw_ref)?,
            };

            let image_embeds = image_embeds.to_device(&device)?.to_dtype(self.text.dtype)?;
            let deepstack_image_embeds = deepstack_image_embeds
                .into_iter()
                .map(|t| t.to_device(&device)?.to_dtype(self.text.dtype))
                .collect::<Result<Vec<_>>>()?;
            if packed_layout.is_some() {
                insert_current_visual_outputs(
                    &mut packed_encoder_outputs,
                    MultimodalKind::Image,
                    image_hashes,
                    per_image.unwrap_or_default(),
                )?;
            }

            if packed_layout.is_none() {
                let mut offset = 0usize;
                let mut image_mask =
                    Tensor::zeros((batch_size, seq_len), DType::F32, input_ids.device())?;
                let total_expected: usize = continuous_img_pad
                    .iter()
                    .flat_map(|spans| spans.iter().map(|(s, e)| e - s))
                    .sum();
                if image_embeds.dim(0)? != total_expected {
                    candle_core::bail!(
                        "Image embedding length {} does not match placeholder tokens {}",
                        image_embeds.dim(0)?,
                        total_expected
                    );
                }
                for (batch, spans) in continuous_img_pad.iter().enumerate() {
                    for &(start, end) in spans {
                        let len = end - start;
                        let chunk = image_embeds.narrow(0, offset, len)?;
                        offset += len;
                        input_embeds = input_embeds.slice_assign(
                            &[batch..batch + 1, start..end, 0..hidden_dim],
                            &chunk.unsqueeze(0)?,
                        )?;
                        let ones = Tensor::ones((1, len), DType::F32, input_ids.device())?;
                        image_mask =
                            image_mask.slice_assign(&[batch..batch + 1, start..end], &ones)?;
                    }
                }
                image_mask_opt = Some(image_mask.to_dtype(DType::U8)?);
                deepstack_image_opt = Some(deepstack_image_embeds);
            }
        }

        if let Some(pixel_values_videos) = &pixel_values_videos {
            let Some(video_grid_thw_ref) = video_grid_thw.as_ref() else {
                candle_core::bail!("pixel_values_videos require video_grid_thw");
            };
            let mut pixel_values = pixel_values_videos.clone();
            let ndim = pixel_values.dims().len();
            if ndim > 2 {
                let last_dim = pixel_values.dim(ndim - 1)?;
                pixel_values = pixel_values.reshape(((), last_dim))?;
            }
            let (video_embeds, deepstack_video_embeds, per_video) = if packed_layout.is_some() {
                let per_video =
                    VisualEncoder::new(&self.vision, &self.encoder_cache, self.spatial_merge_size)
                        .encode(
                            &pixel_values,
                            video_grid_thw_ref,
                            video_hashes,
                            CacheModality::Video,
                        )?;
                let (main, deepstack) = concatenate_visual_items(&per_video)?;
                (main, deepstack, Some(per_video))
            } else {
                let (main, deepstack) = self.vision.forward(&pixel_values, video_grid_thw_ref)?;
                (main, deepstack, None)
            };
            let video_embeds = video_embeds.to_device(&device)?.to_dtype(self.text.dtype)?;
            let deepstack_video_embeds = deepstack_video_embeds
                .into_iter()
                .map(|t| t.to_device(&device)?.to_dtype(self.text.dtype))
                .collect::<Result<Vec<_>>>()?;
            if let Some(per_video) = per_video {
                insert_current_visual_outputs(
                    &mut packed_encoder_outputs,
                    MultimodalKind::Video,
                    video_hashes,
                    per_video,
                )?;
            }

            if packed_layout.is_none() {
                let mut offset = 0usize;
                let mut video_mask =
                    Tensor::zeros((batch_size, seq_len), DType::F32, input_ids.device())?;
                let total_expected: usize = continuous_vid_pad
                    .iter()
                    .flat_map(|spans| spans.iter().map(|(s, e)| e - s))
                    .sum();
                if video_embeds.dim(0)? != total_expected {
                    candle_core::bail!(
                        "Video embedding length {} does not match placeholder tokens {}",
                        video_embeds.dim(0)?,
                        total_expected
                    );
                }
                for (batch, spans) in continuous_vid_pad.iter().enumerate() {
                    for &(start, end) in spans {
                        let len = end - start;
                        let chunk = video_embeds.narrow(0, offset, len)?;
                        offset += len;
                        input_embeds = input_embeds.slice_assign(
                            &[batch..batch + 1, start..end, 0..hidden_dim],
                            &chunk.unsqueeze(0)?,
                        )?;
                        let ones = Tensor::ones((1, len), DType::F32, input_ids.device())?;
                        video_mask =
                            video_mask.slice_assign(&[batch..batch + 1, start..end], &ones)?;
                    }
                }
                video_mask_opt = Some(video_mask.to_dtype(DType::U8)?);
                deepstack_video_opt = Some(deepstack_video_embeds);
            }
        }

        let (legacy_visual_pos_masks, legacy_deepstack_visual_embeds) = match (
            image_mask_opt,
            deepstack_image_opt,
            video_mask_opt,
            deepstack_video_opt,
        ) {
            (Some(image_mask), Some(image_deepstack), Some(video_mask), Some(video_deepstack)) => {
                let combined =
                    (image_mask.to_dtype(DType::F32)? + video_mask.to_dtype(DType::F32)?)?;
                let visual_mask = combined.gt(0f32)?.to_dtype(DType::U8)?;
                let visual_indices = visual_mask.flatten_all()?.nonzero()?.squeeze(1)?;
                let visual_indices_vec = visual_indices.to_vec1::<i64>()?;

                let image_flat = image_mask
                    .flatten_all()?
                    .to_dtype(DType::U8)?
                    .to_vec1::<u8>()?;
                let num_visual = visual_indices_vec.len();
                if image_deepstack.len() != video_deepstack.len() {
                    candle_core::bail!(
                        "DeepStack image layers ({}) do not match video layers ({})",
                        image_deepstack.len(),
                        video_deepstack.len()
                    );
                }
                let mut combined_layers = Vec::with_capacity(image_deepstack.len());
                for (img_layer, vid_layer) in image_deepstack.iter().zip(video_deepstack.iter()) {
                    let mut rows = Vec::with_capacity(num_visual);
                    let mut img_offset = 0usize;
                    let mut vid_offset = 0usize;
                    for &idx in &visual_indices_vec {
                        let idx = idx as usize;
                        if image_flat[idx] != 0 {
                            rows.push(img_layer.i(img_offset)?);
                            img_offset += 1;
                        } else {
                            rows.push(vid_layer.i(vid_offset)?);
                            vid_offset += 1;
                        }
                    }
                    if img_offset != img_layer.dim(0)? || vid_offset != vid_layer.dim(0)? {
                        candle_core::bail!(
                                "DeepStack feature alignment failed for images ({}/{}) or videos ({}/{})",
                                img_offset,
                                img_layer.dim(0)?,
                                vid_offset,
                                vid_layer.dim(0)?
                            );
                    }
                    let row_refs: Vec<&Tensor> = rows.iter().collect();
                    combined_layers.push(Tensor::stack(&row_refs, 0)?);
                }
                (Some(visual_mask), Some(combined_layers))
            }
            (Some(image_mask), Some(image_deepstack), _, _) => {
                (Some(image_mask), Some(image_deepstack))
            }
            (_, _, Some(video_mask), Some(video_deepstack)) => {
                (Some(video_mask), Some(video_deepstack))
            }
            _ => (None, None),
        };

        let (visual_pos_masks, deepstack_visual_embeds) = if let Some(layout) = packed_layout {
            input_embeds = layout.splice_embeddings(&input_embeds, &packed_encoder_outputs)?;
            let destinations = layout.destination_positions(0);
            let mut visual_mask =
                Tensor::zeros((batch_size * seq_len,), DType::F32, input_ids.device())?;
            if !destinations.is_empty() {
                let indices = Tensor::from_vec(
                    destinations
                        .iter()
                        .map(|position| u32::try_from(*position).map_err(candle_core::Error::wrap))
                        .collect::<Result<Vec<_>>>()?,
                    destinations.len(),
                    input_ids.device(),
                )?;
                visual_mask = visual_mask.scatter_add(
                    &indices,
                    &Tensor::ones(destinations.len(), DType::F32, input_ids.device())?,
                    0,
                )?;
            }
            let visual_mask = visual_mask
                .reshape((batch_size, seq_len))?
                .to_dtype(DType::U8)?;
            let output_count = packed_encoder_outputs
                .values()
                .map(Vec::len)
                .min()
                .unwrap_or(1);
            let mut deepstack = Vec::with_capacity(output_count.saturating_sub(1));
            for output in 1..output_count {
                let outputs = packed_encoder_outputs
                    .iter()
                    .map(|(key, values)| (*key, vec![values[output].clone()]))
                    .collect::<HashMap<_, _>>();
                deepstack.push(layout.gather_output_embeddings(0, &input_embeds, &outputs)?);
            }
            (Some(visual_mask), Some(deepstack))
        } else {
            (legacy_visual_pos_masks, legacy_deepstack_visual_embeds)
        };

        let position_ids = if let Some(position_ids) = prompt_position_ids {
            position_ids.clone()
        } else {
            let mut ropeidx_attn_mask_bs = Vec::new();
            let max_seqlens = *seqlens.iter().max().unwrap();
            for len in &seqlens {
                ropeidx_attn_mask_bs.push(Tensor::new(
                    [vec![1f32; *len], vec![0f32; max_seqlens - len]].concat(),
                    input_ids.device(),
                )?);
            }
            let ropeidx_attn_mask = Tensor::stack(&ropeidx_attn_mask_bs, 0)?;
            let (position_ids, mrope_position_deltas) = get_rope_index(
                input_ids_full,
                rope_img_grid_thw.as_ref(),
                rope_vid_grid_thw.as_ref(),
                &AttentionMask::Custom(ropeidx_attn_mask),
                self.spatial_merge_size,
                self.image_token_id,
                self.video_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
            )?;
            crate::vision_models::mrope_position_ids_for_input(
                &position_ids,
                &mrope_position_deltas,
                input_ids,
                seqlen_offsets,
            )?
        };
        let out = self.text.forward_embeds(
            input_embeds,
            &attention_mask,
            &position_ids,
            ctx,
            visual_pos_masks.as_ref(),
            deepstack_visual_embeds.as_deref(),
        )?;
        Ok(out)
    }
}

pub(crate) struct Qwen3VLVisionSpecificArgs {
    pub input_ids_full: Tensor,
    pub pixel_values_videos: Option<Tensor>,
    pub image_grid_thw: Option<Tensor>, // Some when pixel values are provided
    pub video_grid_thw: Option<Tensor>, // Some when pixel values are provided
    pub rope_img_grid_thw: Option<Tensor>,
    pub rope_vid_grid_thw: Option<Tensor>,
    pub seqlens: Vec<usize>,
    pub continuous_img_pad: Vec<Vec<(usize, usize)>>,
    pub continuous_vid_pad: Vec<Vec<(usize, usize)>>,
    pub image_hashes: Vec<u64>,
    pub video_hashes: Vec<u64>,
    pub(crate) packed_layout: Option<PackedMultimodalLayout>,
    pub(crate) prompt_position_ids: Option<Tensor>,
}

impl crate::speculative::SpeculativeTargetMixin for Qwen3VLModel {}

impl crate::block_diffusion::BlockDiffusionMixin for Qwen3VLModel {}

impl MultimodalModel for Qwen3VLModel {
    fn supports_packed_prefill(&self) -> bool {
        true
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let Qwen3VLVisionSpecificArgs {
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
            video_hashes,
            packed_layout,
            prompt_position_ids,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Qwen3VLVisionSpecificArgs`");
        let pixel_values_video = pixel_values_videos.or_else(|| {
            (image_grid_thw.is_none() && video_grid_thw.is_some())
                .then(|| pixel_values.clone())
                .flatten()
        });
        let pixel_values = (image_grid_thw.is_some()).then_some(pixel_values).flatten();
        let rope_img = rope_img_grid_thw.or(image_grid_thw.clone());
        let rope_vid = rope_vid_grid_thw.or(video_grid_thw.clone());
        self.forward(
            input_ids,
            &input_ids_full,
            pixel_values,
            pixel_values_video,
            image_grid_thw,
            video_grid_thw,
            rope_img,
            rope_vid,
            seqlens,
            continuous_img_pad,
            continuous_vid_pad,
            &image_hashes,
            &video_hashes,
            packed_layout.as_ref(),
            prompt_position_ids.as_ref(),
            ctx,
        )
    }
    fn cache(&self) -> &EitherCache {
        &self.text.cache
    }
    fn device(&self) -> &Device {
        &self.text.device
    }
    fn max_seq_len(&self) -> usize {
        self.text.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.text.cfg
    }
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        assert_eq!(input_ids.dims()[0], 1);
        Box::new(Qwen3VLVisionSpecificArgs {
            input_ids_full: input_ids.clone(),
            pixel_values_videos: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rope_img_grid_thw: None,
            rope_vid_grid_thw: None,
            seqlens: vec![input_ids.dims()[1]],
            continuous_img_pad: vec![],
            continuous_vid_pad: vec![],
            image_hashes: vec![],
            video_hashes: vec![],
            packed_layout: None,
            prompt_position_ids: None,
        })
    }
    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    )> {
        Some(
            self.encoder_cache
                .lock()
                .expect("encoder cache poisoned")
                .counters(),
        )
    }
}

impl IsqModel for Qwen3VLModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let mut tensors = self.text.residual_tensors();
        tensors.extend(self.vision.residual_tensors());
        tensors
    }
}

impl AnyMoeBaseModelMixin for Qwen3VLModel {}

#[cfg(test)]
mod tests {
    use super::*;

    const ENCODER_CACHE_CAPACITY: usize = 32;
    const EVICTION_BATCH_SIZE: usize = ENCODER_CACHE_CAPACITY + 1;

    #[test]
    fn current_deepstack_outputs_outlive_encoder_lru_eviction() -> Result<()> {
        let hashes = (0..EVICTION_BATCH_SIZE as u64).collect::<Vec<_>>();
        let outputs = (0..EVICTION_BATCH_SIZE)
            .map(|index| {
                Ok(vec![
                    Tensor::new(&[index as f32], &Device::Cpu)?,
                    Tensor::new(&[(index + 100) as f32], &Device::Cpu)?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        let mut cache = EncoderCacheManager::new(ENCODER_CACHE_CAPACITY);
        for (&hash, output) in hashes.iter().zip(&outputs) {
            cache.insert(CacheModality::Image, hash, output.clone());
        }
        assert!(cache.get(CacheModality::Image, hashes[0]).is_none());

        let mut packed = MultimodalEncoderOutputs::new();
        insert_current_visual_outputs(&mut packed, MultimodalKind::Image, &hashes, outputs)?;
        assert_eq!(packed.len(), EVICTION_BATCH_SIZE);
        let retained = &packed[&MultimodalEncoderKey {
            kind: MultimodalKind::Image,
            hash: hashes[0],
        }];
        assert_eq!(retained[0].to_vec1::<f32>()?, vec![0.]);
        assert_eq!(retained[1].to_vec1::<f32>()?, vec![100.]);
        Ok(())
    }

    #[test]
    fn video_mrope_consumes_the_full_temporal_grid() -> Result<()> {
        // One [2,4,2] grid row feeds two per-frame vision spans in the timestamped prompt format.
        let input_ids = Tensor::new(&[[10u32, 12, 12, 11, 10, 12, 12, 11, 7]], &Device::Cpu)?;
        let video_grid = Tensor::new(&[[2u32, 4, 2]], &Device::Cpu)?;
        let (positions, delta) = get_rope_index(
            &input_ids,
            None,
            Some(&video_grid),
            &AttentionMask::None,
            2,
            13,
            12,
            10,
            11,
        )?;

        assert_eq!(positions.dims(), &[3, 1, 9]);
        assert_eq!(delta.dims(), &[1, 1]);
        assert_eq!(
            positions.i((0, 0))?.to_vec1::<i64>()?,
            vec![0, 1, 1, 3, 4, 5, 5, 7, 8]
        );
        Ok(())
    }
}
