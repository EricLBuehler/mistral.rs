#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};
use text::Qwen3VLMoETextModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::CausalMasker,
    layers_masker::PastKvLenCache,
    paged_attention::{
        encoder_cache::EncoderCacheManager, AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, VisionModel,
    },
    vision_models::qwen3_vl::{vision::Qwen3VLVisionModel, Qwen3VLVisionSpecificArgs},
};

pub(crate) mod config;
mod text;

pub(crate) use config::Config;
// Re-export the processor from qwen3_vl since the input processing is identical
pub(crate) use crate::vision_models::qwen3_vl::Qwen3VLProcessor as Qwen3VLMoEProcessor;

pub struct Qwen3VLMoEModel {
    text: Qwen3VLMoETextModel,
    vision: Qwen3VLVisionModel,
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Qwen3VLMoEModel {
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
        let text = Qwen3VLMoETextModel::new(
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

        let (position_ids, mrope_position_deltas) = super::qwen3_vl::get_rope_index(
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

impl VisionModel for Qwen3VLMoEModel {
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

impl IsqModel for Qwen3VLMoEModel {
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

impl AnyMoeBaseModelMixin for Qwen3VLMoEModel {}
