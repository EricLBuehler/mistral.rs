#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::attention::AttentionMask;
use crate::layers_masker::CausalMaskConfig;
use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use candle_core::{Device, Result, Tensor, D};
use mistralrs_quant::ShardedVarBuilder;
use text::Qwen4ExpTextModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    layers::CausalMasker,
    layers_masker::PastKvLenCache,
    paged_attention::{
        encoder_cache::{CacheModality, EncoderCacheManager},
        AttentionImplementation, HybridPagedKvCacheConfig, ModelConfigLike, ModelConfigMetadata,
    },
    pipeline::{
        EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata,
    },
    vision_models::{
        multimodal_layout::PackedMultimodalLayout,
        qwen3_5::packed_visual::{PackedVisualEncoder, PackedVisualInput},
        qwen3_vl::{
            concatenate_visual_items, vision::Qwen3VLVisionModel, Qwen3VLVisionSpecificArgs,
            VisualEncoder,
        },
    },
};

pub(crate) mod config;
mod hyper;
mod ple;
mod qsa;
mod text;

pub(crate) use crate::vision_models::qwen3_vl::Qwen3VLProcessor as Qwen4ExpProcessor;
pub(crate) use config::Config;
#[cfg(feature = "cuda")]
pub(crate) use ple::planned_resident_table_bytes;

/// Hybrid paged config that also budgets for the per-token QSA aux cache beside the KV cache.
pub(crate) struct Qwen4ExpPagedConfig {
    inner: HybridPagedKvCacheConfig,
    aux_dim: usize,
    dense_kv_cap: usize,
}

impl Qwen4ExpPagedConfig {
    pub(crate) fn new(
        base: ModelConfigMetadata,
        layer_types: &[config::LayerType],
        aux_dim: usize,
        dense_kv_cap: usize,
    ) -> Self {
        Self {
            inner: HybridPagedKvCacheConfig::new(
                base,
                layer_types
                    .iter()
                    .map(|ty| matches!(ty, config::LayerType::FullAttention))
                    .collect(),
            )
            .with_uniform_prefix_prefill_attention_features(Default::default()),
            aux_dim,
            dense_kv_cap,
        }
    }
}

impl ModelConfigLike for Qwen4ExpPagedConfig {
    fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len()
    }
    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }
    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }
    fn num_kv_heads(&self) -> usize {
        self.inner.num_kv_heads()
    }
    fn num_attn_heads(&self) -> usize {
        self.inner.num_attn_heads()
    }
    fn k_head_dim(&self) -> usize {
        self.inner.k_head_dim()
    }
    fn v_head_dim(&self) -> usize {
        self.inner.v_head_dim()
    }
    fn attention_backend_kind(&self) -> crate::paged_attention::AttentionBackendKind {
        self.inner.attention_backend_kind()
    }
    fn attention_backend_kind_for_layer(
        &self,
        layer_idx: usize,
    ) -> crate::paged_attention::AttentionBackendKind {
        self.inner.attention_backend_kind_for_layer(layer_idx)
    }
    // Token-major rows keep sparse gathers coalesced; decode never runs FlashInfer's group-limited kernels
    fn kv_cache_layout(&self) -> crate::paged_attention::KvCacheLayout {
        if cfg!(feature = "cuda") {
            crate::paged_attention::KvCacheLayout::FlashInferHnd
        } else {
            self.inner.kv_cache_layout()
        }
    }
    fn kv_cache_layout_for_layer(&self, layer_idx: usize) -> crate::paged_attention::KvCacheLayout {
        if cfg!(feature = "cuda") {
            crate::paged_attention::KvCacheLayout::FlashInferHnd
        } else {
            self.inner.kv_cache_layout_for_layer(layer_idx)
        }
    }
    fn kv_cache_elements_per_token(&self) -> usize {
        self.inner.kv_cache_elements_per_token()
    }
    fn layer_has_paged_kv_cache(&self, layer_idx: usize) -> bool {
        self.inner.layer_has_paged_kv_cache(layer_idx)
    }
    fn layer_kv_cache_elements_per_token(&self, layer_idx: usize) -> Option<usize> {
        self.inner
            .layer_kv_cache_elements_per_token(layer_idx)
            .map(|elements| elements + self.aux_dim)
    }
    fn prefix_prefill_attention_features(
        &self,
        layer_idx: usize,
    ) -> Option<crate::paged_attention::PrefixPrefillAttentionFeatures> {
        self.inner.prefix_prefill_attention_features(layer_idx)
    }
    fn prefix_prefill_kv_len_cap(&self) -> Option<usize> {
        Some(self.dense_kv_cap)
    }
}

pub struct Qwen4ExpModel {
    text: Qwen4ExpTextModel,
    vision: Option<Qwen3VLVisionModel>,
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Qwen4ExpModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vision_vb = if vb.contains_tensor("vision_tower.patch_embed.proj.weight") {
            vb.pp("vision_tower")
        } else {
            vb.pp("model").pp("visual")
        }
        .without_lora_registry();
        let vision = cfg
            .vision_config
            .as_ref()
            .map(|vision_cfg| {
                Qwen3VLVisionModel::new(
                    vision_cfg,
                    vision_vb.set_device(normal_loading_metadata.real_device.clone()),
                )
            })
            .transpose()?;
        // Use top-level quantization_config if present, otherwise fall back to text_config's
        let mut text_config = cfg.text_config.clone();
        if cfg.quantization_config.is_some() {
            text_config.quantization_config = cfg.quantization_config.clone();
        }
        let text = Qwen4ExpTextModel::new(
            &text_config,
            vb.clone(),
            cfg.tie_word_embeddings,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            vision,
            spatial_merge_size: cfg
                .vision_config
                .as_ref()
                .map_or(0, |vision| vision.spatial_merge_size),
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
            vision_start_token_id: cfg.vision_start_token_id,
            vision_end_token_id: cfg.vision_end_token_id,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn vision(&self) -> Result<&Qwen3VLVisionModel> {
        self.vision
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("this Qwen4-Exp checkpoint has no vision tower"))
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
        // Later chunks and decode rows attend through the paged cache, so only the first chunk needs a mask
        let attention_mask = if ctx.is_first_prompt_chunk() {
            CausalMasker.make_causal_mask(
                input_ids,
                &seqlen_offsets as &dyn PastKvLenCache,
                self.text.dtype,
                &CausalMaskConfig {
                    sliding_window: self.text.cfg.sliding_window,
                    ..Default::default()
                },
            )?
        } else {
            AttentionMask::None
        };

        let input_embeds = self.text.embed_tokens(input_ids)?;
        if let Some(layout) = packed_layout {
            let position_ids = prompt_position_ids.ok_or_else(|| {
                candle_core::Error::msg("packed Qwen4-Exp prefill is missing prompt position IDs")
            })?;
            let input_embeds = if pixel_values.is_none() && pixel_values_videos.is_none() {
                input_embeds
            } else {
                PackedVisualEncoder::new(
                    self.vision()?,
                    &self.encoder_cache,
                    self.spatial_merge_size,
                )
                .prepare(PackedVisualInput {
                    input_embeds,
                    pixel_values: pixel_values.as_ref(),
                    pixel_values_videos: pixel_values_videos.as_ref(),
                    image_grid_thw: image_grid_thw.as_ref(),
                    video_grid_thw: video_grid_thw.as_ref(),
                    image_hashes,
                    video_hashes,
                    layout,
                })?
                .input_embeds
            };
            return self.text.forward_embeds(
                input_embeds,
                input_ids,
                &attention_mask,
                position_ids,
                ctx,
            );
        }
        let mut input_embeds = input_embeds;
        let hidden_dim = input_embeds.dim(D::Minus1)?;
        for (media, grid, hashes, pads, modality) in [
            (
                pixel_values.as_ref(),
                image_grid_thw.as_ref(),
                image_hashes,
                &continuous_img_pad,
                CacheModality::Image,
            ),
            (
                pixel_values_videos.as_ref(),
                video_grid_thw.as_ref(),
                video_hashes,
                &continuous_vid_pad,
                CacheModality::Video,
            ),
        ] {
            let Some(media) = media else {
                continue;
            };
            let Some(grid) = grid else {
                candle_core::bail!("{modality:?} pixel values require a grid_thw");
            };
            let mut media = media.clone();
            if media.rank() > 2 {
                let last_dim = media.dim(D::Minus1)?;
                media = media.reshape(((), last_dim))?;
            }
            let vision = self.vision()?;
            let (embeds, _) = if hashes.is_empty() {
                vision.forward(&media, grid)?
            } else {
                let items =
                    VisualEncoder::new(vision, &self.encoder_cache, self.spatial_merge_size)
                        .encode(&media, grid, hashes, modality)?;
                concatenate_visual_items(&items)?
            };
            let embeds = embeds
                .to_device(input_embeds.device())?
                .to_dtype(self.text.dtype)?;
            let expected: usize = pads
                .iter()
                .flat_map(|spans| spans.iter().map(|(s, e)| e - s))
                .sum();
            if embeds.dim(0)? != expected {
                candle_core::bail!(
                    "{modality:?} embedding length {} does not match placeholder tokens {expected}",
                    embeds.dim(0)?
                );
            }
            let mut offset = 0usize;
            for (batch, spans) in pads.iter().enumerate() {
                for &(start, end) in spans {
                    let len = end - start;
                    input_embeds = input_embeds.slice_assign(
                        &[batch..batch + 1, start..end, 0..hidden_dim],
                        &embeds.narrow(0, offset, len)?.unsqueeze(0)?,
                    )?;
                    offset += len;
                }
            }
        }

        let position_ids = if rope_img_grid_thw.is_none() && rope_vid_grid_thw.is_none() {
            // Text-only rows use plain positions on all three MRoPE planes; no host round trip needed
            match crate::vision_models::text_decode_mrope_position_ids_from_context(input_ids, ctx)?
            {
                Some(position_ids) => Some(position_ids),
                None => Some(crate::vision_models::text_mrope_position_ids(
                    input_ids,
                    seqlen_offsets,
                )?),
            }
        } else {
            None
        };
        let position_ids = match position_ids {
            Some(position_ids) => position_ids,
            None => {
                let mut ropeidx_attn_mask_bs = Vec::new();
                let max_seqlens = *seqlens
                    .iter()
                    .max()
                    .ok_or(candle_core::Error::Msg("seqlens is empty".to_string()))?;
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
            }
        };

        self.text
            .forward_embeds(input_embeds, input_ids, &attention_mask, &position_ids, ctx)
    }
}

impl crate::speculative::SpeculativeTargetMixin for Qwen4ExpModel {}

impl crate::block_diffusion::BlockDiffusionMixin for Qwen4ExpModel {}

impl MultimodalModel for Qwen4ExpModel {
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
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        self.text.supports_decode_graphs()
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs_for_args(&self, model_specific_args: &dyn Any) -> bool {
        model_specific_args
            .downcast_ref::<Qwen3VLVisionSpecificArgs>()
            .is_some()
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.text.cfg
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(Qwen4ExpPagedConfig::new(
            self.text.cfg.clone(),
            &self.text.layer_types,
            self.text.aux_dim,
            self.text.dense_kv_cap,
        ))
    }
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        let (batch_size, seq_len) = input_ids.dims2().expect("input ids must be rank 2");
        Box::new(Qwen3VLVisionSpecificArgs {
            input_ids_full: input_ids.clone(),
            pixel_values_videos: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rope_img_grid_thw: None,
            rope_vid_grid_thw: None,
            seqlens: vec![seq_len; batch_size],
            continuous_img_pad: vec![],
            continuous_vid_pad: vec![],
            image_hashes: vec![],
            video_hashes: vec![],
            packed_layout: None,
            prompt_position_ids: None,
        })
    }
    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        Some(&self.encoder_cache)
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

impl IsqModel for Qwen4ExpModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let mut tensors = self.text.residual_tensors();
        if let Some(vision) = &self.vision {
            tensors.extend(vision.residual_tensors());
        }
        tensors
    }
}

impl AnyMoeBaseModelMixin for Qwen4ExpModel {}
