//! PaddleOCR-VL vision model (SigLIP/NaViT encoder + adaptive-MLP connector + ERNIE-4.5-0.3B LM).
//!
//! Ported from PaddleOCR-VL (Apache-2.0, https://github.com/PaddlePaddle/PaddleOCR), following the
//! HuggingFace transformers `modeling_paddleocr_vl` reference and the GGUF ERNIE-4.5 layout.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub mod config;
pub mod connector;
pub mod inputs_processor;
pub mod merge;
pub mod preprocess;
pub mod rope_index;
pub mod text;
pub mod vision;

use std::any::Any;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;

use crate::amoe::AnyMoeBaseModelMixin;
use crate::layers::CausalMasker;
use crate::layers_masker::{CausalMaskConfig, PastKvLenCache};
use crate::paged_attention::{AttentionImplementation, KvCacheLayout, ModelConfigMetadata};
use crate::pipeline::{
    EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalCache, NormalLoadingMetadata,
};

use config::Config;
use connector::Connector;
use merge::Merger;
use rope_index::get_rope_index_batched;
use text::ErnieTextModel;
use vision::VisionModel;

/// The full PaddleOCR-VL-1.5 model: SigLIP/NaViT vision tower -> `mlp_AR` connector ->
/// embed+scatter merge -> ERNIE-4.5-0.3B decoder. Each sub-module is the parity-verified port;
/// `new` assembles them from the checkpoint's top-level prefixes (see `ref/keys.txt`).
pub struct PaddleOcrVlModel {
    vision: VisionModel,
    connector: Connector,
    merger: Merger,
    text: ErnieTextModel,
    cfg: Config,
    // Engine-facing accessor state for the `MultimodalModel` trait: `cache` is the engine KV cache
    // the trait `forward` drives, held here so `text.rs` stays the parity-verified port.
    device: Device,
    max_seq_len: usize,
    config_meta: ModelConfigMetadata,
    cache: EitherCache,
}

impl PaddleOcrVlModel {
    /// `vb` is the checkpoint root. Sub-prefixes: `visual.vision_model.*` (tower), `mlp_AR.*`
    /// (connector), `model.embed_tokens.*` (merge embed), `model.layers/norm.*` + `lm_head.*` (LM).
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let tcfg = cfg.text_config();
        let vcfg = cfg.vision_config();
        // Non-quantized parts (tower/connector/embed) must load on the real compute device, not the
        // cpu staging device ISQ uses for the quantizable LM weights, or their activations reach the
        // cuda quant projections on cpu ("input must live on CUDA"). Mirrors qwen2vl.
        let real_dev = normal_loading_metadata.real_device.clone();
        let vision = VisionModel::load(
            vb.pp("visual")
                .pp("vision_model")
                .set_device(real_dev.clone()),
            &vcfg,
        )?;
        let connector = Connector::load(
            vb.pp("mlp_AR").set_device(real_dev.clone()),
            vcfg.hidden_size,
            vcfg.spatial_merge_size,
            tcfg.hidden_size,
        )?;
        let merger = Merger::load(
            vb.pp("model").set_device(real_dev.clone()),
            tcfg.vocab_size,
            tcfg.hidden_size,
            cfg.image_token_id as i64,
        )?;
        // ErnieTextModel::load consumes the root vb (it does its own .pp("model")/.pp("lm_head")).
        let device = real_dev;
        let text = ErnieTextModel::load(
            vb,
            &tcfg,
            &*normal_loading_metadata.mapper,
            normal_loading_metadata.loading_isq,
            attention_mechanism,
        )?;
        // world_size 1 (CPU f32 parity path, no tensor-parallel sharding); head_dim is the K/V dim.
        let config_meta = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: tcfg.num_hidden_layers,
            hidden_size: tcfg.hidden_size,
            num_attn_heads: tcfg.num_attention_heads,
            num_kv_heads: tcfg.num_key_value_heads,
            sliding_window: None,
            k_head_dim: tcfg.head_dim,
            v_head_dim: tcfg.head_dim,
            kv_cache_layout: KvCacheLayout::Standard,
        };
        Ok(Self {
            vision,
            connector,
            merger,
            text,
            cfg: cfg.clone(),
            device,
            max_seq_len: cfg.max_position_embeddings,
            config_meta,
            cache: EitherCache::Normal(NormalCache::new(
                tcfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
        })
    }
}

// The three mixins are supertraits of `MultimodalModel` but PaddleOCR-VL is a plain dense VLM:
// no MoE experts, no speculative-draft target, no block-diffusion. Empty impls satisfy the bound.
impl AnyMoeBaseModelMixin for PaddleOcrVlModel {}
impl crate::speculative::SpeculativeTargetMixin for PaddleOcrVlModel {}
impl crate::block_diffusion::BlockDiffusionMixin for PaddleOcrVlModel {}

impl IsqModel for PaddleOcrVlModel {
    // The ERNIE LM projections + lm_head are ISQ-quantized (see the loader's `isq_layer_regexes`);
    // everything else is a full-precision residual serialized alongside for a UQFF round-trip.
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let mut tensors = self.text.residual_tensors();
        tensors.extend(self.merger.residual_tensors());
        tensors.extend(self.connector.residual_tensors());
        tensors.extend(self.vision.residual_tensors());
        tensors
    }
}

/// Model-specific args threaded alongside `input_ids` through the engine. `input_ids_full` is each
/// row's whole prompt (so mrope positions/`delta` recompute identically every step, matching
/// qwen3_vl's stateless scheme). `image_grid_thw` holds one `(t, h, w)` patch grid per batch row
/// (empty for a text-only row), and is itself empty only when no row carries an image; mrope needs
/// it on every pass. `vision_rows` names the rows whose patches are in `pixel_values` this pass, in
/// concat order: empty on decode and on prefill chunks that hold no image tokens.
pub(crate) struct PaddleOcrVlVisionSpecificArgs {
    pub input_ids_full: Tensor,
    pub image_grid_thw: Vec<Vec<(usize, usize, usize)>>,
    pub vision_rows: Vec<usize>,
}

impl MultimodalModel for PaddleOcrVlModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let PaddleOcrVlVisionSpecificArgs {
            input_ids_full,
            image_grid_thw,
            vision_rows,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `PaddleOcrVlVisionSpecificArgs`");

        let dev = &self.device;
        let merge = self.cfg.vision_config().spatial_merge_size;
        let image_token_id = self.cfg.image_token_id as i64;
        let seqlen_offsets = ctx.seqlen_offsets();

        // mrope: recompute full-sequence positions from input_ids_full each step (stateless, like
        // qwen3_vl), then slice/continue the current window. One image grid per batch row (empty for
        // text-only / all-text decode).
        let (batch, _full_len) = input_ids_full.dims2()?;
        let grids: Vec<Vec<(usize, usize, usize)>> = if image_grid_thw.is_empty() {
            vec![Vec::new(); batch]
        } else {
            image_grid_thw.clone()
        };
        let (full_pos, deltas) =
            get_rope_index_batched(&input_ids_full, &grids, image_token_id, merge, dev)?;
        let position_ids = crate::vision_models::mrope_position_ids_for_input(
            &full_pos,
            &deltas,
            input_ids,
            seqlen_offsets,
        )?;

        // Each row in `vision_rows` runs vision -> connector over each of its images and scatters the
        // connector rows, concatenated in message order, into that row's placeholders;
        // `pixel_values` is every such image's patches concatenated on dim 0, split back by each
        // grid's t*h*w. Every other row (text-only, decode, or a prefill chunk outside the image
        // span) is a plain on-device token embed of the current window.
        let embeds = if vision_rows.is_empty() {
            self.merger.embed_tokens(input_ids)?
        } else {
            let pv = pixel_values.expect("vision rows without pixel values");
            let text = self.merger.embed_tokens(input_ids)?;
            let mut rows = (0..batch)
                .map(|b| text.narrow(0, b, 1)?.squeeze(0))
                .collect::<Result<Vec<_>>>()?;
            let mut offset = 0;
            for &b in &vision_rows {
                let mut embeds_per_image = Vec::with_capacity(image_grid_thw[b].len());
                for &(t, h, w) in &image_grid_thw[b] {
                    let post_ln =
                        self.vision
                            .forward(&pv.narrow(0, offset, t * h * w)?, t, h, w)?;
                    offset += t * h * w;
                    embeds_per_image.push(self.connector.forward(&post_ln, t, h, w)?);
                }
                let image_embeds = Tensor::cat(&embeds_per_image, 0)?;
                let row_ids = input_ids.narrow(0, b, 1)?.flatten_all()?;
                rows[b] = self.merger.forward(&row_ids, &image_embeds)?;
            }
            Tensor::stack(&rows, 0)?
        };

        // Engine causal mask: batch-aware; `Custom` on prefill, `None` on single-token decode.
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            embeds.dtype(),
            &CausalMaskConfig::default(),
        )?;
        // No `is_first_prompt_chunk` override here: a later prompt chunk carries
        // `prefix_cache_len = chunk.start`, so paged sees `num_cached_tokens` and takes the prefix
        // gather path, which reads causality off this mask. Forcing `AttentionMask::None` there
        // makes it fall back to `flash_params.causal` and attend non-causally within the chunk.

        let mut guard = self.cache.normal();
        // `None` on the NormalCache path, where the text model falls back to Sdpa.
        let paged = ctx.paged_metadata();
        let paged_ref = paged.as_ref().map(|(kv, meta)| (kv.as_slice(), *meta));
        let logits = self
            .text
            .forward(
                &embeds,
                &position_ids,
                &mut guard.0,
                &mask,
                paged_ref,
                Some(ctx.flash_params()),
            )?
            .logits; // [batch, seq, vocab]
        ctx.logits(&logits) // engine slices the wanted rows
    }

    fn device(&self) -> &Device {
        &self.device
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.config_meta
    }
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(PaddleOcrVlVisionSpecificArgs {
            input_ids_full: input_ids.clone(),
            image_grid_thw: Vec::new(),
            vision_rows: Vec::new(),
        })
    }
}
