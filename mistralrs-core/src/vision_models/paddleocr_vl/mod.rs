//! PaddleOCR-VL vision model (SigLIP/NaViT encoder + adaptive-MLP connector + ERNIE-4.5-0.3B LM).
//!
//! Ported from PaddleOCR-VL (Apache-2.0, https://github.com/PaddlePaddle/PaddleOCR), following the
//! HuggingFace transformers `modeling_paddleocr_vl` reference and the GGUF ERNIE-4.5 layout.

// `dead_code`: each submodule keeps numerical-parity-only helpers (the inherent prefill `forward`,
// `causal_mask`, `preprocess_image`, activation-capture fields) that the loader/`MultimodalModel`
// path does not all reach.
#![allow(dead_code)]
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
        let vision = VisionModel::load(vb.pp("visual").pp("vision_model"), &vcfg)?;
        let connector = Connector::load(
            vb.pp("mlp_AR"),
            vcfg.hidden_size,
            vcfg.spatial_merge_size,
            tcfg.hidden_size,
        )?;
        let merger = Merger::load(
            vb.pp("model"),
            tcfg.vocab_size,
            tcfg.hidden_size,
            cfg.image_token_id as i64,
        )?;
        // ErnieTextModel::load consumes the root vb (it does its own .pp("model")/.pp("lm_head")).
        let device = vb.device().clone();
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

/// Model-specific args threaded alongside `input_ids` through the engine. Single image, no video:
/// `input_ids_full` is the whole prompt (so mrope positions/`delta` are recomputed identically on
/// every step, matching qwen3_vl's stateless scheme); `image_grid_thw` is that image's patch grid
/// `(t, h, w)`, `None` for text-only prompts and all decode steps.
pub(crate) struct PaddleOcrVlVisionSpecificArgs {
    pub input_ids_full: Tensor,
    pub image_grid_thw: Option<(usize, usize, usize)>,
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
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `PaddleOcrVlVisionSpecificArgs`");

        let dev = &self.device;
        let merge = self.cfg.vision_config().spatial_merge_size;
        let image_token_id = self.cfg.image_token_id as i64;
        let seqlen_offsets = ctx.seqlen_offsets();

        // mrope: recompute full-sequence positions from input_ids_full each step (stateless, like
        // qwen3_vl), then slice/continue the current window. Image prefill is batch-1; text/decode
        // rows carry no image grid.
        let (batch, _full_len) = input_ids_full.dims2()?;
        let grids: Vec<Vec<(usize, usize, usize)>> = match image_grid_thw {
            Some(grid) => vec![vec![grid]],
            None => vec![Vec::new(); batch],
        };
        let (full_pos, deltas) =
            get_rope_index_batched(&input_ids_full, &grids, image_token_id, merge, dev)?;
        let position_ids = crate::vision_models::mrope_position_ids_for_input(
            &full_pos,
            &deltas,
            input_ids,
            seqlen_offsets,
        )?;

        // Prefill-with-image runs vision -> connector -> masked-scatter merge (batch-1); text/decode
        // is a pure on-device token embed of the current window.
        let embeds = match pixel_values {
            Some(pv) => {
                let (t, h, w) = image_grid_thw.expect("pixel_values require image_grid_thw");
                let vout = self.vision.forward(&pv, t, h, w)?;
                let image_embeds = self.connector.forward(&vout.post_ln, t, h, w)?;
                self.merger
                    .forward(&input_ids.flatten_all()?, &image_embeds)?
                    .unsqueeze(0)?
            }
            None => self.merger.embed_tokens(input_ids)?,
        };

        // Engine causal mask: batch-aware; `Custom` on prefill, `None` on single-token decode.
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            embeds.dtype(),
            &CausalMaskConfig::default(),
        )?;

        let mut guard = self.cache.normal();
        // Paged metadata is threaded through but inert until the loader enables paged attention:
        // `paged_metadata` is None on the NormalCache path, so the text model falls back to Sdpa.
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
            image_grid_thw: None,
        })
    }
}
