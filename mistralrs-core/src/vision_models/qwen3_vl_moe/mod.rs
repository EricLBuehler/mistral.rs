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
use text::Qwen3VLMoETextModel;

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
    vision_models::multimodal_layout::{MultimodalEncoderOutputs, PackedMultimodalLayout},
    vision_models::qwen3_vl::{
        concatenate_visual_items, insert_current_visual_outputs, vision::Qwen3VLVisionModel,
        Qwen3VLVisionSpecificArgs, VisualEncoder,
    },
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
            let (position_ids, mrope_position_deltas) = super::qwen3_vl::get_rope_index(
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

impl crate::speculative::SpeculativeTargetMixin for Qwen3VLMoEModel {}

impl crate::block_diffusion::BlockDiffusionMixin for Qwen3VLMoEModel {}

impl MultimodalModel for Qwen3VLMoEModel {
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

impl IsqModel for Qwen3VLMoEModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let mut tensors = self.text.residual_tensors();
        tensors.extend(self.vision.residual_tensors());
        tensors
    }
}

impl AnyMoeBaseModelMixin for Qwen3VLMoEModel {}
