#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};
use text::Qwen3VLTextModel;
use vision::Qwen3VLVisionModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::CausalMasker,
    layers_masker::{masked_fill, PastKvLenCache},
    paged_attention::{
        encoder_cache::EncoderCacheManager, AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, VisionModel,
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
/// Shared between Qwen3VLModel and Qwen3VLMoEModel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_rope_index(
    input_ids: &Tensor,
    image_grid_thw: Option<&Tensor>,
    video_grid_thw: Option<&Tensor>,
    attention_mask: Option<&Tensor>,
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
            Some(mask) => mask.clone(),
            None => Tensor::ones((batch, seq_len), DType::F32, &device)?,
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
            let mut repeated = Vec::new();
            for row in raw {
                if row.len() != 3 {
                    candle_core::bail!("video_grid_thw entries must have length 3");
                }
                let repeat = row[0] as usize;
                for _ in 0..repeat {
                    repeated.push([1, row[1], row[2]]);
                }
            }
            Some(repeated)
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
    } else if let Some(attention_mask) = attention_mask {
        let position_ids = (attention_mask.to_dtype(DType::F32)?.cumsum(D::Minus1)? - 1f64)?;
        let position_ids = masked_fill(&position_ids, &attention_mask.eq(0f64)?, 1i64)?;
        let position_ids = position_ids.unsqueeze(0)?.repeat((3, 1, 1))?;

        let max_position_ids = position_ids.max(0)?.max_keepdim(D::Minus1)?;
        let mrope_position_deltas =
            ((max_position_ids + 1.)? - attention_mask.dim(D::Minus1)? as f64)?;

        Ok((
            position_ids.to_dtype(DType::I64)?,
            mrope_position_deltas.to_dtype(DType::I64)?,
        ))
    } else {
        let position_ids = Tensor::arange(0i64, input_ids.dim(1)? as i64, input_ids.device())?
            .reshape((1, 1, ()))?
            .repeat((3, input_ids.dim(0)?, 1))?;
        let mrope_position_deltas =
            Tensor::zeros((input_ids.dim(0)?, 1), DType::I64, input_ids.device())?;

        Ok((position_ids, mrope_position_deltas))
    }
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
        };
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
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        image_hashes: &[u64],
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            self.text.cfg.sliding_window,
            self.text.dtype,
            self.text.cfg.num_attn_heads,
        )?;
        let is_first_chunk = metadata
            .as_ref()
            .map(|(_, meta)| meta.is_first_prompt_chunk)
            .unwrap_or(true);
        attention_mask = attention_mask.filter(|_| is_first_chunk);

        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let (batch_size, seq_len, hidden_dim) = input_embeds.dims3()?;
        let device = input_embeds.device().clone();

        let mut image_mask_opt: Option<Tensor> = None;
        let mut video_mask_opt: Option<Tensor> = None;
        let mut deepstack_image_opt: Option<Vec<Tensor>> = None;
        let mut deepstack_video_opt: Option<Vec<Tensor>> = None;

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

            let (image_embeds, deepstack_image_embeds) = if !image_hashes.is_empty() {
                let n_images = image_hashes.len();
                let grid_data = image_grid_thw_ref.to_vec2::<u32>()?;
                let patches_per_image: Vec<usize> = grid_data
                    .iter()
                    .map(|row| row[0] as usize * row[1] as usize * row[2] as usize)
                    .collect();
                let merge = self.spatial_merge_size;
                let output_tokens_per_image: Vec<usize> = grid_data
                    .iter()
                    .map(|row| {
                        (row[0] as usize) * (row[1] as usize / merge) * (row[2] as usize / merge)
                    })
                    .collect();

                // per_image[i] = Some(vec![image_embeds_i, ds_0_i, ds_1_i, ...])
                let mut per_image: Vec<Option<Vec<Tensor>>> = vec![None; n_images];
                let mut miss_indices = Vec::new();
                {
                    let mut guard = self
                        .encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned");
                    for (i, &hash) in image_hashes.iter().enumerate() {
                        if let Some(cached) = guard.get(hash) {
                            per_image[i] = Some(cached);
                        } else {
                            miss_indices.push(i);
                        }
                    }
                }

                if miss_indices.is_empty() {
                    // All cached - reassemble
                    let main_parts: Vec<Tensor> = per_image
                        .iter()
                        .map(|o| o.as_ref().unwrap()[0].clone())
                        .collect();
                    let image_embeds = Tensor::cat(&main_parts, 0)?;
                    let n_ds_layers = per_image[0].as_ref().unwrap().len() - 1;
                    let mut deepstack_layers = Vec::with_capacity(n_ds_layers);
                    for layer_idx in 0..n_ds_layers {
                        let layer_parts: Vec<Tensor> = per_image
                            .iter()
                            .map(|o| o.as_ref().unwrap()[1 + layer_idx].clone())
                            .collect();
                        deepstack_layers.push(Tensor::cat(&layer_parts, 0)?);
                    }
                    (image_embeds, deepstack_layers)
                } else {
                    // Collect miss pixel slices and grid rows
                    let mut miss_pixel_slices = Vec::new();
                    let mut miss_grid_rows = Vec::new();
                    let mut pv_offset = 0usize;
                    for (i, &n_patches) in patches_per_image.iter().enumerate() {
                        if miss_indices.contains(&i) {
                            miss_pixel_slices.push(pixel_values.narrow(0, pv_offset, n_patches)?);
                            miss_grid_rows.push(image_grid_thw_ref.i(i)?);
                        }
                        pv_offset += n_patches;
                    }
                    let miss_pixels = Tensor::cat(&miss_pixel_slices, 0)?;
                    let miss_grid = Tensor::stack(&miss_grid_rows, 0)?;

                    let (encoded_main, encoded_ds) =
                        self.vision.forward(&miss_pixels, &miss_grid)?;

                    // Compute output tokens per miss image
                    let miss_output_tokens: Vec<usize> = miss_indices
                        .iter()
                        .map(|&i| output_tokens_per_image[i])
                        .collect();

                    // Split and cache per-image
                    let mut enc_offset = 0usize;
                    {
                        let mut guard = self
                            .encoder_cache
                            .lock()
                            .expect("encoder cache lock poisoned");
                        for (j, &orig_idx) in miss_indices.iter().enumerate() {
                            let n_out = miss_output_tokens[j];
                            let single_main = encoded_main.narrow(0, enc_offset, n_out)?;
                            let mut cache_entry = vec![single_main.clone()];
                            for ds_layer in &encoded_ds {
                                let single_ds = ds_layer.narrow(0, enc_offset, n_out)?;
                                cache_entry.push(single_ds.clone());
                            }
                            enc_offset += n_out;
                            guard.insert(image_hashes[orig_idx], cache_entry.clone());
                            per_image[orig_idx] = Some(cache_entry);
                        }
                    }

                    // Reassemble all images
                    let main_parts: Vec<Tensor> = per_image
                        .iter()
                        .map(|o| o.as_ref().unwrap()[0].clone())
                        .collect();
                    let image_embeds = Tensor::cat(&main_parts, 0)?;
                    let n_ds_layers = per_image[0].as_ref().unwrap().len() - 1;
                    let mut deepstack_layers = Vec::with_capacity(n_ds_layers);
                    for layer_idx in 0..n_ds_layers {
                        let layer_parts: Vec<Tensor> = per_image
                            .iter()
                            .map(|o| o.as_ref().unwrap()[1 + layer_idx].clone())
                            .collect();
                        deepstack_layers.push(Tensor::cat(&layer_parts, 0)?);
                    }
                    (image_embeds, deepstack_layers)
                }
            } else {
                self.vision.forward(&pixel_values, image_grid_thw_ref)?
            };

            let image_embeds = image_embeds.to_device(&device)?.to_dtype(self.text.dtype)?;
            let deepstack_image_embeds = deepstack_image_embeds
                .into_iter()
                .map(|t| t.to_device(&device)?.to_dtype(self.text.dtype))
                .collect::<Result<Vec<_>>>()?;

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
                    image_mask = image_mask.slice_assign(&[batch..batch + 1, start..end], &ones)?;
                }
            }
            image_mask_opt = Some(image_mask.to_dtype(DType::U8)?);
            deepstack_image_opt = Some(deepstack_image_embeds);
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
            let (video_embeds, deepstack_video_embeds) =
                self.vision.forward(&pixel_values, video_grid_thw_ref)?;
            let video_embeds = video_embeds.to_device(&device)?.to_dtype(self.text.dtype)?;
            let deepstack_video_embeds = deepstack_video_embeds
                .into_iter()
                .map(|t| t.to_device(&device)?.to_dtype(self.text.dtype))
                .collect::<Result<Vec<_>>>()?;

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
                    video_mask = video_mask.slice_assign(&[batch..batch + 1, start..end], &ones)?;
                }
            }
            video_mask_opt = Some(video_mask.to_dtype(DType::U8)?);
            deepstack_video_opt = Some(deepstack_video_embeds);
        }

        let (visual_pos_masks, deepstack_visual_embeds) = match (
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

        let mut ropeidx_attn_mask_bs = Vec::new();
        let max_seqlens = *seqlens.iter().max().unwrap();
        for len in &seqlens {
            ropeidx_attn_mask_bs.push(Tensor::new(
                [vec![1f32; *len], vec![0f32; max_seqlens - len]].concat(),
                input_ids.device(),
            )?);
        }
        let ropeidx_attn_mask = Tensor::stack(&ropeidx_attn_mask_bs, 0)?;
        let ropeidx_input_ids = if attention_mask.is_some() {
            input_ids
        } else {
            input_ids_full
        };

        let (position_ids, mrope_position_deltas) = get_rope_index(
            ropeidx_input_ids,
            rope_img_grid_thw.as_ref(),
            rope_vid_grid_thw.as_ref(),
            Some(&ropeidx_attn_mask),
            self.spatial_merge_size,
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )?;
        let position_ids = if attention_mask.is_some() {
            position_ids
        } else {
            let mut position_ids = Tensor::new(
                seqlen_offsets.iter().map(|x| *x as i64).collect::<Vec<_>>(),
                input_ids.device(),
            )?
            .reshape((1, (), 1))?
            .repeat((3, 1, 1))?;
            position_ids = position_ids.broadcast_add(&mrope_position_deltas.unsqueeze(0)?)?;
            position_ids
        };

        let out = self.text.forward_embeds(
            input_embeds,
            attention_mask.as_ref(),
            &position_ids,
            context_lens,
            metadata,
            flash_params,
            visual_pos_masks.as_ref(),
            deepstack_visual_embeds.as_deref(),
        )?;
        Ok(out)
    }
}

pub(crate) struct Qwen3VLVisionSpecificArgs {
    pub input_ids_full: Tensor,
    pub image_grid_thw: Option<Tensor>, // Some when pixel values are provided
    pub video_grid_thw: Option<Tensor>, // Some when pixel values are provided
    /// Complete image grid THW for ALL images in the full sequence (including prefix-cached ones).
    /// Used for MRoPE position computation. Falls back to `image_grid_thw` if None.
    pub rope_img_grid_thw: Option<Tensor>,
    /// Complete video grid THW for ALL videos in the full sequence (including prefix-cached ones).
    pub rope_vid_grid_thw: Option<Tensor>,
    pub seqlens: Vec<usize>,
    pub continuous_img_pad: Vec<Vec<(usize, usize)>>,
    pub continuous_vid_pad: Vec<Vec<(usize, usize)>>,
    pub image_hashes: Vec<u64>,
}

impl VisionModel for Qwen3VLModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let Qwen3VLVisionSpecificArgs {
            input_ids_full,
            image_grid_thw,
            video_grid_thw,
            rope_img_grid_thw,
            rope_vid_grid_thw,
            seqlens,
            continuous_img_pad,
            continuous_vid_pad,
            image_hashes,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Qwen3VLVisionSpecificArgs`");
        let (pixel_values, pixel_values_video) = match (&image_grid_thw, &video_grid_thw) {
            (Some(_), None) => (pixel_values, None),
            (None, Some(_)) => (None, pixel_values),
            (None, None) => (None, None),
            (Some(_), Some(_)) => {
                candle_core::bail!("Images and videos cannot be provided together.")
            }
        };
        // Use the complete grid (covering all images/videos including prefix-cached ones)
        // for MRoPE position computation. Falls back to current-frame grid.
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
            seqlen_offsets,
            context_lens,
            &image_hashes,
            metadata,
            flash_params,
        )
    }
    fn cache(&self) -> &EitherCache {
        &self.text.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.text.cache
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
            image_grid_thw: None,
            video_grid_thw: None,
            rope_img_grid_thw: None,
            rope_vid_grid_thw: None,
            seqlens: vec![input_ids.dims()[1]],
            continuous_img_pad: vec![],
            continuous_vid_pad: vec![],
            image_hashes: vec![],
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
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        self.text.get_layers()
    }
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let mut tensors = self.text.residual_tensors();
        tensors.extend(self.vision.residual_tensors());
        tensors
    }
}

impl AnyMoeBaseModelMixin for Qwen3VLModel {}
