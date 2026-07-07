//! PaddleOCR-VL vision model (SigLIP/NaViT encoder + adaptive-MLP connector + ERNIE-4.5-0.3B LM).
//!
//! Ported from PaddleOCR-VL (Apache-2.0, https://github.com/PaddlePaddle/PaddleOCR), following the
//! HuggingFace transformers `modeling_paddleocr_vl` reference and the GGUF ERNIE-4.5 layout.

// `dead_code`: each submodule keeps numerical-parity-only helpers (the inherent prefill `forward`,
// the hand-rolled `KvCache`/`forward_cached`/`causal_mask`, `preprocess_image`, activation-capture
// fields) that the loader/`MultimodalModel` path does not all reach.
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

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;

use crate::amoe::AnyMoeBaseModelMixin;
use crate::paged_attention::{KvCacheLayout, ModelConfigMetadata};
use crate::pipeline::{EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalCache};

use config::Config;
use connector::Connector;
use merge::Merger;
use rope_index::get_rope_index;
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
    // Engine-facing accessor state for the `MultimodalModel` trait. Held on the top-level model,
    // not `ErnieTextModel`, so `text.rs` stays byte-identical to the parity-verified port.
    // `cache` is the engine's KV cache the trait `forward` will drive; the hand-rolled
    // `text::KvCache` is only used by the parity harness.
    device: Device,
    max_seq_len: usize,
    config_meta: ModelConfigMetadata,
    cache: EitherCache,
}

impl PaddleOcrVlModel {
    /// `vb` is the checkpoint root. Sub-prefixes: `visual.vision_model.*` (tower), `mlp_AR.*`
    /// (connector), `model.embed_tokens.*` (merge embed), `model.layers/norm.*` + `lm_head.*` (LM).
    pub fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
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
        let text = ErnieTextModel::load(vb, &tcfg)?;
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

    /// Prefill forward: pixel_values -> vision -> connector -> embed+scatter merge -> ERNIE LM.
    /// Returns logits `[seq, vocab]`.
    /// `grid` is the image `(t,h,w)` and `position_ids` is the `[3,seq]` mrope index (see rope_index).
    /// No decode loop / KV cache here; that rides the `MultimodalModel` wiring.
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid: (usize, usize, usize),
        input_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (t, h, w) = grid;
        let vout = self.vision.forward(pixel_values, t, h, w)?;
        let image_embeds = self.connector.forward(&vout.post_ln, t, h, w)?;
        let embeds = self.merger.forward(input_ids, &image_embeds)?;
        Ok(self.text.forward(&embeds, position_ids)?.logits)
    }
}

// The three mixins are supertraits of `MultimodalModel` but PaddleOCR-VL is a plain dense VLM:
// no MoE experts, no speculative-draft target, no block-diffusion. Empty impls satisfy the bound.
impl AnyMoeBaseModelMixin for PaddleOcrVlModel {}
impl crate::speculative::SpeculativeTargetMixin for PaddleOcrVlModel {}
impl crate::block_diffusion::BlockDiffusionMixin for PaddleOcrVlModel {}

impl IsqModel for PaddleOcrVlModel {
    // ISQ is not wired for the f32 CPU parity path (parity, not quantization, is the goal);
    // empty means "no full-precision residuals reserved", a no-op while ISQ is off.
    // Populate with norms/embeddings/lm_head when an ISQ speed path is added.
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        vec![]
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
        let offset = ctx.seqlen_offsets()[0]; // batch 1: past length, 0 on prefill
        let merge = self.cfg.vision_config().spatial_merge_size;
        let image_token_id = self.cfg.image_token_id as i64;

        let ids_flat = input_ids.flatten_all()?; // batch 1 -> [seq]
        let ids_vec = ids_flat.to_dtype(DType::I64)?.to_vec1::<i64>()?;

        // mrope: full prefill positions [3, full_len] + decode `delta`, recomputed each step from
        // input_ids_full (stateless, like qwen3_vl). Decode position = offset + delta on all 3 rows.
        let full_ids = input_ids_full
            .flatten_all()?
            .to_dtype(DType::I64)?
            .to_vec1::<i64>()?;
        let (full_pos, delta) = match image_grid_thw {
            Some(grid) => get_rope_index(&full_ids, &[grid], image_token_id, merge, dev)?,
            None => {
                let n = full_ids.len();
                let mut v = Vec::with_capacity(3 * n);
                for _ in 0..3 {
                    for j in 0..n as i64 {
                        v.push(j);
                    }
                }
                (Tensor::from_vec(v, (3, n), dev)?, 0i64)
            }
        };
        let position_ids = if offset == 0 {
            full_pos
        } else {
            let n = ids_vec.len();
            let mut cols = Vec::with_capacity(3 * n);
            for _ in 0..3 {
                for j in 0..n as i64 {
                    cols.push(offset as i64 + j + delta);
                }
            }
            Tensor::from_vec(cols, (3, n), dev)?
        };

        // Prefill runs vision -> connector -> masked-scatter merge (needs pixel_values); decode is a
        // pure text embed of the newly generated token(s).
        let embeds = match pixel_values {
            Some(pv) => {
                let (t, h, w) = image_grid_thw.expect("pixel_values require image_grid_thw");
                let vout = self.vision.forward(&pv, t, h, w)?;
                let image_embeds = self.connector.forward(&vout.post_ln, t, h, w)?;
                self.merger.forward(&ids_flat, &image_embeds)?
            }
            None => self.merger.embed(&ids_vec)?,
        };

        let mut guard = self.cache.normal();
        let logits = self
            .text
            .forward_engine(&embeds, &position_ids, &mut guard.0, offset)?
            .logits; // [seq, vocab]
        ctx.logits(&logits.unsqueeze(0)?) // [1, seq, vocab]; engine slices the wanted rows
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
