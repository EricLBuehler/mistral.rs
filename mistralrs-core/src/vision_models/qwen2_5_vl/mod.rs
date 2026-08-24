#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::attention::AttentionMask;
use crate::layers_masker::CausalMaskConfig;
use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use candle_core::{Context, Device, IndexOp, Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;
use text::Qwen2_5VLTextModel;
use vision::Qwen2_5VLVisionModel;

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
    utils::unvarbuilder::UnVarBuilder,
    vision_models::multimodal_layout::{
        gather_packed_mrope_positions, MropePositionSource, MultimodalEncoderOutputs,
        PackedMultimodalLayout,
    },
    vision_models::qwen2vl::insert_current_visual_outputs,
};

mod config;
mod inputs_processor;
mod text;
mod vision;

pub(crate) use config::Config;
pub(crate) use inputs_processor::Qwen2_5VLProcessor;

pub struct Qwen2_5VLModel {
    text: Qwen2_5VLTextModel,
    vision: Qwen2_5VLVisionModel,
    vision_prefix: &'static str,
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Qwen2_5VLModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let (vision_vb, vision_prefix) =
            if vb.contains_tensor("vision_tower.patch_embed.proj.weight") {
                (vb.pp("vision_tower"), "vision_tower")
            } else {
                (vb.pp("visual"), "visual")
            };
        let vision_vb = vision_vb.without_lora_registry();
        let vision = Qwen2_5VLVisionModel::new(
            &cfg.vision_config,
            vision_vb.set_device(normal_loading_metadata.real_device.clone()),
            &normal_loading_metadata.mapper.get_comm_for(0)?,
        )?;
        let text = Qwen2_5VLTextModel::new(
            cfg,
            vb.clone(),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            vision,
            vision_prefix,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn encode_visual_items(
        &self,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
        hashes: &[u64],
        modality: CacheModality,
    ) -> Result<Vec<Tensor>> {
        let grids = grid_thw.to_vec2::<u32>()?;
        if grids.len() != hashes.len() {
            candle_core::bail!(
                "Qwen visual grid count {} does not match hash count {}",
                grids.len(),
                hashes.len()
            );
        }
        let patch_counts = grids
            .iter()
            .map(|grid| grid.iter().map(|value| *value as usize).product::<usize>())
            .collect::<Vec<_>>();
        let output_counts = grids
            .iter()
            .map(|grid| {
                grid[0] as usize
                    * (grid[1] as usize / self.spatial_merge_size)
                    * (grid[2] as usize / self.spatial_merge_size)
            })
            .collect::<Vec<_>>();
        let mut outputs = vec![None; hashes.len()];
        let mut misses = Vec::new();
        {
            let mut cache = self
                .encoder_cache
                .lock()
                .expect("encoder cache lock poisoned");
            for (index, &hash) in hashes.iter().enumerate() {
                if let Some(cached) = cache.get(modality, hash) {
                    outputs[index] = Some(cached[0].clone());
                } else {
                    misses.push(index);
                }
            }
        }
        if !misses.is_empty() {
            let mut pixel_offset = 0;
            let mut miss_pixels = Vec::with_capacity(misses.len());
            let mut miss_grids = Vec::with_capacity(misses.len());
            for (index, &patch_count) in patch_counts.iter().enumerate() {
                if misses.contains(&index) {
                    miss_pixels.push(pixel_values.narrow(0, pixel_offset, patch_count)?);
                    miss_grids.push(grid_thw.i(index)?);
                }
                pixel_offset += patch_count;
            }
            let encoded = self.vision.forward(
                &Tensor::cat(&miss_pixels, 0)?,
                &Tensor::stack(&miss_grids, 0)?,
            )?;
            let mut output_offset = 0;
            let mut cache = self
                .encoder_cache
                .lock()
                .expect("encoder cache lock poisoned");
            for &index in &misses {
                let output = encoded.narrow(0, output_offset, output_counts[index])?;
                output_offset += output_counts[index];
                cache.insert(modality, hashes[index], vec![output.clone()]);
                outputs[index] = Some(output);
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| candle_core::Error::msg("missing Qwen visual output"))
            })
            .collect()
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
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            self.text.dtype,
            &CausalMaskConfig::default(),
        )?;
        let sliding_attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            self.text.dtype,
            &CausalMaskConfig {
                sliding_window: self.text.sliding_window,
                ..Default::default()
            },
        )?;

        let input_embeds = if pixel_values.is_some() || pixel_values_videos.is_some() {
            let mut xs = self.text.embed_tokens(input_ids)?;
            let mut packed_encoder_outputs = MultimodalEncoderOutputs::new();

            if let Some(pixel_values) = pixel_values {
                let mut pixel_values = pixel_values;
                let ndim = pixel_values.dims().len();
                if ndim > 2 {
                    let last_dim = pixel_values.dim(ndim - 1)?;
                    pixel_values = pixel_values.reshape(((), last_dim))?;
                }
                let grid_thw = image_grid_thw
                    .as_ref()
                    .context("pixel_values require image_grid_thw")?;

                let per_image = if image_hashes.is_empty() {
                    None
                } else {
                    Some(self.encode_visual_items(
                        &pixel_values,
                        grid_thw,
                        image_hashes,
                        CacheModality::Image,
                    )?)
                };
                let image_embeds = match &per_image {
                    Some(outputs) => Tensor::cat(outputs, 0)?,
                    None => self.vision.forward(&pixel_values, grid_thw)?,
                }
                .to_dtype(self.text.dtype)?;

                if packed_layout.is_some() {
                    insert_current_visual_outputs(
                        &mut packed_encoder_outputs,
                        MultimodalKind::Image,
                        image_hashes,
                        per_image.unwrap_or_default(),
                    )?;
                } else {
                    let mut offset = 0;
                    for (batch, batch_ids) in continuous_img_pad.iter().enumerate() {
                        for &(start, end) in batch_ids {
                            let len = end - start;
                            xs = xs.slice_assign(
                                &[batch..batch + 1, start..end, 0..xs.dim(2)?],
                                &image_embeds.narrow(0, offset, len)?.unsqueeze(0)?,
                            )?;
                            offset += len;
                        }
                    }
                }
            }

            if let Some(pixel_values_videos) = pixel_values_videos {
                let grid = video_grid_thw
                    .as_ref()
                    .context("pixel_values_videos require video_grid_thw")?;
                let per_video = if video_hashes.is_empty() {
                    None
                } else {
                    Some(self.encode_visual_items(
                        &pixel_values_videos,
                        grid,
                        video_hashes,
                        CacheModality::Video,
                    )?)
                };
                let video_embeds = match &per_video {
                    Some(outputs) => Tensor::cat(outputs, 0)?,
                    None => self.vision.forward(&pixel_values_videos, grid)?,
                }
                .to_dtype(self.text.dtype)?;

                if packed_layout.is_some() {
                    insert_current_visual_outputs(
                        &mut packed_encoder_outputs,
                        MultimodalKind::Video,
                        video_hashes,
                        per_video.unwrap_or_default(),
                    )?;
                } else {
                    let mut offset = 0;
                    for (batch, batch_ids) in continuous_vid_pad.iter().enumerate() {
                        for &(start, end) in batch_ids {
                            let len = end - start;
                            xs = xs.slice_assign(
                                &[batch..batch + 1, start..end, 0..xs.dim(2)?],
                                &video_embeds.narrow(0, offset, len)?.unsqueeze(0)?,
                            )?;
                            offset += len;
                        }
                    }
                }
            }

            if let Some(layout) = packed_layout {
                layout.splice_embeddings(&xs, &packed_encoder_outputs)?
            } else {
                xs
            }
        } else {
            self.text.embed_tokens(input_ids)?
        };

        let decode_position_ids = if rope_img_grid_thw.is_none() && rope_vid_grid_thw.is_none() {
            crate::vision_models::text_decode_mrope_position_ids_from_context(input_ids, ctx)?
        } else {
            None
        };
        let position_ids = if let Some(position_ids) = prompt_position_ids {
            position_ids.clone()
        } else if let Some(position_ids) = decode_position_ids {
            position_ids
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
            let ropeidx_input_ids = if packed_layout.is_some() {
                input_ids_full
            } else if !matches!(attention_mask, AttentionMask::None) {
                input_ids
            } else {
                input_ids_full
            };
            let (position_ids, mrope_position_deltas) =
                crate::vision_models::qwen2vl::Qwen2VLModel::compute_rope_index(
                    ropeidx_input_ids,
                    rope_img_grid_thw.as_ref(),
                    rope_vid_grid_thw.as_ref(),
                    &AttentionMask::Custom(ropeidx_attn_mask),
                    self.spatial_merge_size,
                    self.image_token_id,
                    self.video_token_id,
                )?;

            if packed_layout.is_some() {
                let sources = (0..seqlens.len())
                    .map(|batch| {
                        Ok(MropePositionSource {
                            position_ids: position_ids.i((.., batch, ..))?.unsqueeze(1)?,
                            delta: mrope_position_deltas.i((batch, 0))?.to_scalar::<i64>()?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let queries = seqlens.iter().map(|len| 0..*len).collect::<Vec<_>>();
                gather_packed_mrope_positions(&sources, &queries, input_ids.device())?
            } else if !matches!(attention_mask, AttentionMask::None) {
                position_ids
            } else {
                crate::vision_models::mrope_position_ids_for_input(
                    &position_ids,
                    &mrope_position_deltas,
                    input_ids,
                    seqlen_offsets,
                )?
            }
        };
        let out = self.text.forward_embeds(
            input_embeds,
            &attention_mask,
            &sliding_attention_mask,
            &position_ids,
            ctx,
        )?;
        Ok(out)
    }
}

pub(crate) struct Qwen2_5VLVisionSpecificArgs {
    input_ids_full: Tensor,
    pixel_values_videos: Option<Tensor>,
    image_grid_thw: Option<Tensor>,
    video_grid_thw: Option<Tensor>,
    pub rope_img_grid_thw: Option<Tensor>,
    pub rope_vid_grid_thw: Option<Tensor>,
    seqlens: Vec<usize>,
    continuous_img_pad: Vec<Vec<(usize, usize)>>,
    continuous_vid_pad: Vec<Vec<(usize, usize)>>,
    pub image_hashes: Vec<u64>,
    pub video_hashes: Vec<u64>,
    packed_layout: Option<PackedMultimodalLayout>,
    prompt_position_ids: Option<Tensor>,
}

impl crate::speculative::SpeculativeTargetMixin for Qwen2_5VLModel {}

impl crate::block_diffusion::BlockDiffusionMixin for Qwen2_5VLModel {}

impl MultimodalModel for Qwen2_5VLModel {
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
        let Qwen2_5VLVisionSpecificArgs {
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
            .expect("Cannot downcast into `Qwen2_5VLVisionSpecificArgs`");
        let pixel_values_video = pixel_values_videos.or_else(|| {
            (image_grid_thw.is_none() && video_grid_thw.is_some())
                .then(|| pixel_values.clone())
                .flatten()
        });
        let pixel_values = image_grid_thw.is_some().then_some(pixel_values).flatten();
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
        Box::new(Qwen2_5VLVisionSpecificArgs {
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

impl IsqModel for Qwen2_5VLModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(self.text.residual_tensors());
        uvb.pp(self.vision_prefix)
            .extend(self.vision.residual_tensors());
        uvb.to_safetensors()
    }
}

impl AnyMoeBaseModelMixin for Qwen2_5VLModel {}
