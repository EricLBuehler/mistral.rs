//! Flint (`FlintForCausalLM`), a MoE model from RecursiveSI.
//!
//! Structure follows the checkpoint's own `config.json` and the upstream llama.cpp
//! implementation (`src/models/flint.cpp`). Five things distinguish it from a plain
//! Qwen-style MoE:
//!
//! - **Attention sinks.** Every layer carries a per-head `softmax_offset` that joins the
//!   softmax as an extra logit attending to nothing.
//! - **Per-layer RoPE.** `no_rope_layer_indices` lists the layers that get no positional
//!   encoding at all. In the shipped checkpoint that set is exactly the full-attention
//!   layers, but the two are separate fields and the GGUF carries them as separate arrays.
//! - **A dense lead.** `mlp_only_layers` names the leading layers that keep a plain SwiGLU
//!   MLP instead of the MoE block.
//! - **A renormalised sigmoid router.** Selection and weighting both use `sigmoid(logits)`,
//!   and the selected weights are then normalised to sum to one. There is no expert bias.
//! - **A shared expert.** Ungated and added in full on every MoE layer.
//!
//! The routed experts live in the checkpoint one expert per directory
//! (`mlp.experts.<i>.<gate|up|down>_proj.weight`), which is the per-expert layout the
//! existing expert gather already handles.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, Module, Result, Tensor};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::layers_masker::CausalMaskConfig;
use crate::moe::{ExpertProjNames, MoEExperts, MoEExpertsConfig};
use crate::{
    attention::{AttentionMask, Sdpa, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        self, embedding_with_legacy_tied_uqff, Activation, CausalMasker, RmsNorm, RotaryEmbedding,
    },
    paged_attention::{ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::PagedAttentionInputMetadata, EitherCache, IsqModel,
        ModelForwardContext, NormalCache, NormalCacheType, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::unvarbuilder::UnVarBuilder,
};

serde_default_fn!(bool, default_tie_word_embeddings, false);

/// Per-layer attention kind, from the checkpoint's `layer_types` array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

impl LayerType {
    fn is_sliding(&self) -> bool {
        matches!(self, Self::SlidingAttention)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: Option<usize>,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    #[serde(default)]
    pub(crate) hidden_act: Activation,
    /// Width of the dense lead MLP, from `intermediate_size`.
    pub(crate) intermediate_size: usize,
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) shared_expert_intermediate_size: usize,
    pub(crate) norm_topk_prob: bool,
    pub(crate) attention_bias: bool,
    #[serde(default = "default_tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) use_sliding_window: bool,
    pub(crate) sliding_window: usize,
    pub(crate) layer_types: Vec<LayerType>,
    /// Layers that keep a dense MLP instead of the MoE block.
    #[serde(default)]
    pub(crate) mlp_only_layers: Vec<usize>,
    /// Layers that get no positional encoding. Empty means "every sliding layer".
    #[serde(default)]
    pub(crate) no_rope_layer_indices: Vec<usize>,
    pub(crate) quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_theta as f32
    }

    /// `layer_types` drives the KV cache geometry, so a length mismatch is a config error.
    pub fn validate(&self) -> Result<()> {
        if self.layer_types.len() != self.num_hidden_layers {
            candle_core::bail!(
                "Flint layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            );
        }
        if self.use_sliding_window && self.sliding_window == 0 {
            candle_core::bail!("Flint use_sliding_window requires a positive sliding_window");
        }
        Ok(())
    }

    /// Whether the layer carries RoPE. The explicit list wins; without one, the
    /// full-attention layers are the ones without positional encoding.
    pub fn layer_has_rope(&self, layer_idx: usize) -> bool {
        if self.no_rope_layer_indices.is_empty() {
            self.layer_types[layer_idx].is_sliding()
        } else {
            !self.no_rope_layer_indices.contains(&layer_idx)
        }
    }

    /// Whether the layer keeps a dense MLP.
    pub fn layer_is_dense(&self, layer_idx: usize) -> bool {
        self.mlp_only_layers.contains(&layer_idx)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    Sliding,
    Full,
}

impl LayerKind {
    fn is_sliding(&self) -> bool {
        matches!(self, Self::Sliding)
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kind: LayerKind,
    /// `None` on the layers that get no positional encoding.
    rotary_emb: Option<Arc<RotaryEmbedding>>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        kind: LayerKind,
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;

        let q_norm = RmsNorm::new(
            head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("q_norm"), false),
        )?;
        let k_norm = RmsNorm::new(
            head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("k_norm"), false),
        )?;

        // The per-head sink is a softmax logit that attends to nothing.
        let sinks = mapper
            .set_device(layer_idx, vb.clone(), false)
            .get((num_heads,), "softmax_offset")?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            kind,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softmax_scale: (head_dim as f32).powf(-0.5),
                softcap: None,
                sliding_window: None,
                sinks: Some(sinks),
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut crate::pipeline::KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let (q, k, v) = {
            let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::AttnProj);
            let (q, k, v) =
                crate::ops::qkv_projections(xs, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
            let (q, k, v) = if q_len != 1 {
                (
                    q.reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                        .transpose(1, 2)?,
                    k.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                        .transpose(1, 2)?,
                    v.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                        .transpose(1, 2)?,
                )
            } else {
                (
                    q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?,
                    k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?,
                    v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?,
                )
            };

            // The QK norms run on every layer; only the rope call is per-layer optional.
            let positions = ctx
                .text_positions(q.device(), q.dim(2)?)?
                .ok_or_else(|| candle_core::Error::msg("Flint missing RoPE positions"))?;
            let (q, k) = match &self.rotary_emb {
                Some(rope) => rope.forward_qk_norm(
                    &q,
                    &k,
                    self.q_norm.weight(),
                    self.k_norm.weight(),
                    self.q_norm.eps(),
                    self.k_norm.eps(),
                    positions,
                )?,
                None => {
                    let q = candle_nn::ops::rms_norm(
                        &q.contiguous()?,
                        self.q_norm.weight(),
                        self.q_norm.eps() as f32,
                    )?;
                    let k = candle_nn::ops::rms_norm(
                        &k.contiguous()?,
                        self.k_norm.weight(),
                        self.k_norm.eps() as f32,
                    )?;
                    (q, k)
                }
            };
            (q, k, v)
        };

        let metadata = ctx.paged_layer(layer_idx);
        let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::Attention);
        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(ctx.flash_params()),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(ctx.flash_params()),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(ctx.flash_params()),
                    &self.sdpa_params,
                )?
            }
        };

        drop(_p);
        attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::AttnProj);
        self.o_proj.forward(&attn_output)
    }
}

/// Sigmoid router: selection and weighting both use `sigmoid(logits)`, with the selected
/// weights renormalised when `norm_topk_prob` is set. There is no expert bias.
struct MoEGate {
    /// Held widened: the reference computes the router in f32, and the selection is decided
    /// on those values.
    gate: Tensor,
    top_k: usize,
    renormalize: bool,
}

impl MoEGate {
    fn new(cfg: &Config, layer_vb: ShardedVarBuilder) -> Result<Self> {
        let gate = layers::linear_no_bias(cfg.hidden_size, cfg.num_experts, layer_vb.pp("gate"))?
            .weight()
            .to_dtype(DType::F32)?
            .contiguous()?;
        Ok(Self {
            gate,
            top_k: cfg.num_experts_per_tok,
            renormalize: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::Router);
        // A single row is a GEMV, which the tiled matmul path does badly.
        let logits = if xs.dim(xs.rank() - 2).is_ok_and(|rows| rows == 1) {
            mistralrs_quant::gemv_cpu::gemv(xs, &self.gate)?
        } else {
            xs.to_dtype(DType::F32)?.matmul(&self.gate.t()?)?
        };
        let topk = crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.top_k,
                score_function: crate::ops::MoeRouterScoreFunction::Sigmoid,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Sigmoid,
                renormalize: self.renormalize,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )?;
        Ok((topk.values, topk.indices))
    }
}

/// The ungated shared expert, added in full to every MoE layer.
struct SharedExpert {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
}

impl SharedExpert {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let inter = cfg.shared_expert_intermediate_size;
        let vb = mapper.set_device(layer_idx, vb, false);
        Ok(Self {
            gate_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                inter,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("gate_proj"),
            )?,
            up_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                inter,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("up_proj"),
            )?,
            down_proj: RowParallelLayer::new(
                inter,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("down_proj"),
            )?,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let y = match self.act_fn {
            Activation::Silu => candle_nn::ops::silu(&gate)?.mul(&up)?,
            Activation::Gelu => Activation::Gelu.forward(&gate)?.mul(&up)?,
            other => candle_core::bail!("unsupported Flint activation {other:?}"),
        };
        self.down_proj.forward(&y)
    }
}

/// The plain SwiGLU MLP the leading layers keep.
struct DenseMlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
}

impl DenseMlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let inter = cfg.intermediate_size;
        let vb = mapper.set_device(layer_idx, vb, false);
        Ok(Self {
            gate_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                inter,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("gate_proj"),
            )?,
            up_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                inter,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("up_proj"),
            )?,
            down_proj: RowParallelLayer::new(
                inter,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("down_proj"),
            )?,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let y = match self.act_fn {
            Activation::Silu => candle_nn::ops::silu(&gate)?.mul(&up)?,
            Activation::Gelu => Activation::Gelu.forward(&gate)?.mul(&up)?,
            other => candle_core::bail!("unsupported Flint activation {other:?}"),
        };
        self.down_proj.forward(&y)
    }
}

struct MoeMlp {
    gate: MoEGate,
    experts: MoEExperts,
    shared: SharedExpert,
}

impl MoeMlp {
    /// `vb` is the `model.layers.N` prefix; the module names are applied here.
    fn new(
        cfg: &Config,
        layer_vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let vb = layer_vb.pp("mlp");
        let gate = MoEGate::new(cfg, vb.clone())?;
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Flint layer device not mapped"))?;

        let experts = MoEExperts::new(
            &MoEExpertsConfig {
                num_experts: cfg.num_experts,
                num_experts_per_tok: cfg.num_experts_per_tok,
                hidden_size: cfg.hidden_size,
                moe_intermediate_size: cfg.moe_intermediate_size,
                expert_proj_names: ExpertProjNames::DEFAULT,
            },
            vb.clone(),
            layer_device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        let shared = SharedExpert::new(cfg, vb.pp("shared_expert"), mapper, layer_idx, comm)?;

        Ok(Self {
            gate,
            experts,
            shared,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let (topk_weights, topk_ids) = self.gate.forward(&xs_flat)?;
        let ys = {
            let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::Routed);
            self.experts.forward(xs, topk_weights, &topk_ids)?
        };
        // The shared expert is ungated and always added in full.
        let shared = {
            let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::Shared);
            self.shared.forward(xs)?
        };
        let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::Combine);
        let ys = (ys + shared)?;
        ys.reshape((b_size, seq_len, hidden_dim))
    }
}

enum LayerMlp {
    Dense(DenseMlp),
    MoE(MoeMlp),
}

impl LayerMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::MoE(mlp) => mlp.forward(xs),
        }
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: LayerMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    /// `vb` is the `model.layers` prefix; the layer index is applied here.
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let kind = if cfg.layer_types[layer_idx].is_sliding() {
            LayerKind::Sliding
        } else {
            LayerKind::Full
        };
        let vb = vb.pp(layer_idx.to_string());

        let self_attn = Attention::new(
            cfg,
            vb.pp("self_attn"),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            kind,
            rotary_emb,
            comm,
        )?;

        let mlp = if cfg.layer_is_dense(layer_idx) {
            LayerMlp::Dense(DenseMlp::new(cfg, vb.pp("mlp"), mapper, layer_idx, comm)?)
        } else {
            LayerMlp::MoE(MoeMlp::new(
                cfg,
                vb.clone(),
                mapper,
                layer_idx,
                loading_isq,
                comm,
            )?)
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut crate::pipeline::KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::Norms);
        let h = self.input_layernorm.forward(xs)?;
        drop(_p);

        let attn = self
            .self_attn
            .forward(&h, attention_mask, kv_cache, ctx, layer_idx)?;

        // The FFN block reads the post-attention residual, which it is also added to.
        let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::Norms);
        let residual = (attn + xs)?;
        let h = self.post_attention_layernorm.forward(&residual)?;
        drop(_p);

        let ffn = self.mlp.forward(&h)?;
        ffn + residual
    }
}

pub struct Model {
    embed_tokens: Arc<dyn QuantMethod>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    dtype: DType,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: Config,
    cfg_metadata: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: crate::paged_attention::AttentionImplementation,
    ) -> Result<Self> {
        cfg.validate()?;
        let vb_m = vb.pp("model");
        let mapper = normal_loading_metadata.mapper;
        let dtype = vb_m.dtype();
        let loading_isq = normal_loading_metadata.loading_isq;
        let head_dim = cfg.head_dim();

        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), loading_isq),
            cfg.tie_word_embeddings
                .then(|| mapper.set_nm_device(vb.pp("lm_head"), loading_isq)),
            &cfg.quantization_config,
        )?;

        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head: Arc<dyn QuantMethod> = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), loading_isq),
            )?
        };

        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer> = crate::utils::progress::NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .run(
            !normal_loading_metadata.real_device.is_metal() && cfg.quantization_config.is_none(),
            |layer_idx| {
                let device = mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device);
                // One rope table serves the layers that get positional encoding.
                let rotary_emb = Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta(),
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    dtype,
                )?);
                let paged_attn = match &attention_mechanism {
                    crate::paged_attention::AttentionImplementation::Eager => None,
                    crate::paged_attention::AttentionImplementation::PagedAttention => Some(
                        PagedAttention::new(head_dim, device, None)
                            .expect("PagedAttention creation failed"),
                    ),
                };
                let comm = mapper
                    .get_comm_for(layer_idx)
                    .expect("Failed to get comm for layer");
                DecoderLayer::new(
                    cfg,
                    vb_l.clone(),
                    &*mapper,
                    layer_idx,
                    loading_isq,
                    paged_attn,
                    cfg.layer_has_rope(layer_idx).then_some(rotary_emb),
                    &comm,
                )
            },
        )?;

        let cache_types: Vec<NormalCacheType> = (0..cfg.num_hidden_layers)
            .map(|i| {
                if cfg.layer_types[i].is_sliding() {
                    NormalCacheType::SlidingWindow {
                        window: cfg.sliding_window,
                    }
                } else {
                    NormalCacheType::Normal {
                        max_seq_len: cfg.max_position_embeddings,
                    }
                }
            })
            .collect();

        let world_size = mapper.get_comm_for(0)?.world_size();
        let cfg_metadata = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: (cfg.num_key_value_heads / world_size).max(1),
            num_attn_heads: cfg.num_attention_heads / world_size,
            sliding_window: None,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            dtype,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            max_seq_len: cfg.max_position_embeddings,
            mapper,
            cfg: cfg.clone(),
            cfg_metadata,
        })
    }

    fn inner_forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut xs = {
            let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::Embed);
            self.embed_tokens.embedding_forward(input_ids, self.dtype)?
        };
        let cache = &mut self.cache.normal().0;

        // One mask of each shape, selected per layer by layer kind.
        let mask_cache = ctx.mask_cache(cache);
        let force_custom = !ctx.flash_params().packed;
        let causal_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig {
                force_custom,
                ..Default::default()
            },
        )?;
        let sliding_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig {
                sliding_window: Some(self.cfg.sliding_window),
                force_custom,
            },
        )?;
        let causal_mask = DeviceMappedMask::new(causal_mask, &*self.mapper)?;
        let sliding_mask = DeviceMappedMask::new(sliding_mask, &*self.mapper)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if layer.self_attn.kind.is_sliding() {
                sliding_mask.get(xs.device())
            } else {
                causal_mask.get(xs.device())
            };
            xs = layer.forward(&xs, &mask, &mut cache[i], ctx, i)?;
        }
        xs = xs.to_device(&self.device)?;

        let xs = {
            let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::Norms);
            self.norm.forward(&xs)?
        };
        let xs = ctx.logits(&xs)?;
        let _p = crate::utils::decode_prof::profile(crate::utils::decode_prof::Phase::LmHead);
        ctx.lm_head(&*self.lm_head, &xs)
    }

    fn residual_tensors_inner(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);
        for (idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            uvb_l
                .pp("self_attn")
                .pp("q_norm")
                .add(&layer.self_attn.q_norm);
            uvb_l
                .pp("self_attn")
                .pp("k_norm")
                .add(&layer.self_attn.k_norm);
            if let Some(sinks) = &layer.self_attn.sdpa_params.sinks {
                uvb_l
                    .pp("self_attn")
                    .add_tensor("softmax_offset", sinks.clone());
            }
            if let LayerMlp::MoE(moe) = &layer.mlp {
                uvb_l.pp("mlp").pp("gate").add(&moe.gate.gate);
            }
        }
        uvb.to_safetensors()
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.residual_tensors_inner()
    }

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        Some(self.residual_tensors_inner())
    }
}

impl NormalModel for Model {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.inner_forward(input_ids, ctx)
    }

    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &crate::pipeline::text_models_inputs_processor::FlashParams,
        _flash_params_full: &crate::pipeline::text_models_inputs_processor::FlashParams,
    ) -> Result<Tensor> {
        candle_core::bail!("Flint does not support X-LoRA")
    }

    fn is_xlora(&self) -> bool {
        false
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
        &self.cfg_metadata
    }
}

impl crate::amoe::AnyMoeBaseModelMixin for Model {}

impl crate::speculative::SpeculativeTargetMixin for Model {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::AttentionImplementation;

    /// The shipped `config.json`, verbatim.
    const REAL_CONFIG: &str = r#"{
  "architectures": [
    "FlintForCausalLM"
  ],
  "model_type": "flint",
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 150000,
  "decoder_sparse_step": 1,
  "eos_token_id": [
    150001,
    150003,
    150005,
    150018
  ],
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 6144,
  "layer_types": [
    "full_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "full_attention"
  ],
  "max_position_embeddings": 262144,
  "mlp_only_layers": [
    0,
    1
  ],
  "moe_intermediate_size": 512,
  "norm_topk_prob": true,
  "num_attention_heads": 32,
  "num_experts": 256,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 49,
  "num_key_value_heads": 4,
  "no_rope_layer_indices": [
    0,
    1,
    5,
    9,
    13,
    17,
    21,
    25,
    29,
    33,
    37,
    41,
    45,
    48
  ],
  "output_router_logits": false,
  "pad_token_id": 150001,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000.0,
  "router_aux_loss_coef": 0.0005,
  "router_score_function": "sigmoid",
  "shared_expert_intermediate_size": 1024,
  "sliding_window": 513,
  "softmax_type": "learnable",
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "use_sliding_window": true,
  "vocab_size": 150144,
  "transformers_version": "5.12.1"
}"#;

    fn real_config() -> Config {
        serde_json::from_str(REAL_CONFIG).expect("shipped Flint config deserializes")
    }

    #[test]
    fn shipped_config_deserializes_and_validates() {
        let cfg = real_config();
        cfg.validate().expect("shipped config passes validation");
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.rope_theta(), 10_000.0);
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 6);
        assert!(cfg.norm_topk_prob);
        assert_eq!(cfg.intermediate_size, 6_144);
        assert_eq!(cfg.shared_expert_intermediate_size, 1_024);
        assert_eq!(cfg.sliding_window, 513);
    }

    /// The shipped checkpoint lists the no-rope layers separately from `layer_types`, and the
    /// two sets have to agree: a layer that is full-attention but still gets RoPE, or the
    /// other way round, silently corrupts the attention.
    #[test]
    fn no_rope_layers_are_exactly_the_full_attention_layers() {
        let cfg = real_config();
        let full: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|i| !cfg.layer_types[*i].is_sliding())
            .collect();
        assert_eq!(full, cfg.no_rope_layer_indices);
        assert_eq!(
            full,
            vec![0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
        );
        for layer in 0..cfg.num_hidden_layers {
            assert_eq!(
                cfg.layer_has_rope(layer),
                cfg.layer_types[layer].is_sliding(),
                "layer {layer}"
            );
        }
    }

    /// The two leading layers keep a dense MLP; every other layer is MoE.
    #[test]
    fn dense_lead_is_the_first_two_layers() {
        let cfg = real_config();
        let dense: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|i| cfg.layer_is_dense(*i))
            .collect();
        assert_eq!(dense, vec![0, 1]);
    }

    /// Without an explicit list, the full-attention layers are the ones with no positional
    /// encoding.
    #[test]
    fn rope_falls_back_to_the_sliding_layers() {
        let mut cfg = real_config();
        cfg.no_rope_layer_indices.clear();
        assert!(!cfg.layer_has_rope(0));
        assert!(cfg.layer_has_rope(2));
        assert!(!cfg.layer_has_rope(5));
    }

    #[test]
    fn short_layer_types_is_a_config_error() {
        let mut cfg = real_config();
        cfg.layer_types.pop();
        assert!(cfg.validate().is_err());
    }

    /// The reference selects on `sigmoid(logits)` and weights with the same value, then
    /// renormalises the selected weights to sum to one.
    #[test]
    fn router_selects_on_sigmoid_and_renormalises() {
        let num_experts = 6;
        let top_k = 3;
        let gate = Tensor::from_vec(
            vec![
                0.0f32, 0.0, // expert 0
                2.0, 0.0, // expert 1
                -1.0, 0.0, // expert 2
                0.5, 0.0, // expert 3
                0.25, 0.0, // expert 4
                -2.0, 0.0, // expert 5
            ],
            (num_experts, 2),
            &Device::Cpu,
        )
        .unwrap();
        let router = MoEGate {
            gate,
            top_k,
            renormalize: true,
        };
        let xs = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &Device::Cpu).unwrap();
        let (weights, indices) = router.forward(&xs).unwrap();

        let scores: Vec<f32> = vec![0.0f32, 2.0, -1.0, 0.5, 0.25, -2.0]
            .into_iter()
            .map(|logit| 1.0 / (1.0 + (-logit).exp()))
            .collect();
        // The top three by sigmoid are experts 1, 3 and 4.
        let mut expected: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1));
        let expected = &expected[..top_k];
        let sum: f32 = expected.iter().map(|(_, score)| score).sum();

        let got_indices = indices.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(
            got_indices,
            expected.iter().map(|(i, _)| *i as u32).collect::<Vec<_>>()
        );
        let got_weights = weights.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (got, (_, score)) in got_weights.iter().zip(expected.iter()) {
            assert!((got - score / sum).abs() < 1e-6, "{got} vs {}", score / sum);
        }
        let total: f32 = got_weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "weights sum to {total}");
    }

    fn small_config() -> Config {
        Config {
            vocab_size: 8,
            hidden_size: 4,
            num_hidden_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: Some(2),
            max_position_embeddings: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            hidden_act: Activation::Silu,
            intermediate_size: 6,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 6,
            shared_expert_intermediate_size: 6,
            norm_topk_prob: true,
            attention_bias: false,
            tie_word_embeddings: false,
            use_sliding_window: true,
            sliding_window: 4,
            // A sliding layer, a full one, then another sliding one.
            layer_types: vec![
                LayerType::SlidingAttention,
                LayerType::FullAttention,
                LayerType::SlidingAttention,
            ],
            // The first two layers keep a dense MLP, so both kinds are built.
            mlp_only_layers: vec![0, 1],
            no_rope_layer_indices: vec![1],
            quantization_config: None,
        }
    }

    /// Builds a model from a synthetic checkpoint whose key set mirrors the real one exactly,
    /// so a renamed or relocated tensor fails here rather than at load time on a 79 GB
    /// checkpoint.
    fn model_for(cfg: &Config) -> Result<Model> {
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let q = cfg.num_attention_heads * head_dim;
        let kv = cfg.num_key_value_heads * head_dim;
        let inter = cfg.moe_intermediate_size;
        let shared = cfg.shared_expert_intermediate_size;

        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        {
            let put = |tensors: &mut std::collections::HashMap<String, Tensor>,
                       name: String,
                       shape: &[usize]|
             -> Result<()> {
                tensors.insert(name, Tensor::ones(shape, DType::F32, &Device::Cpu)?);
                Ok(())
            };
            put(
                &mut tensors,
                "model.embed_tokens.weight".to_string(),
                &[cfg.vocab_size, h],
            )?;
            put(&mut tensors, "model.norm.weight".to_string(), &[h])?;
            put(
                &mut tensors,
                "lm_head.weight".to_string(),
                &[cfg.vocab_size, h],
            )?;

            for layer in 0..cfg.num_hidden_layers {
                let p = format!("model.layers.{layer}");
                for name in ["input_layernorm", "post_attention_layernorm"] {
                    put(&mut tensors, format!("{p}.{name}.weight"), &[h])?;
                }
                for (name, shape) in [
                    ("q_proj", &[q, h][..]),
                    ("k_proj", &[kv, h][..]),
                    ("v_proj", &[kv, h][..]),
                    ("o_proj", &[h, q][..]),
                    ("q_norm", &[head_dim][..]),
                    ("k_norm", &[head_dim][..]),
                ] {
                    put(&mut tensors, format!("{p}.self_attn.{name}.weight"), shape)?;
                }
                // The per-head sink, and it carries no `.weight` suffix.
                put(
                    &mut tensors,
                    format!("{p}.self_attn.softmax_offset"),
                    &[cfg.num_attention_heads],
                )?;
                if cfg.layer_is_dense(layer) {
                    for (name, shape) in [
                        ("gate_proj", &[cfg.intermediate_size, h][..]),
                        ("up_proj", &[cfg.intermediate_size, h][..]),
                        ("down_proj", &[h, cfg.intermediate_size][..]),
                    ] {
                        put(&mut tensors, format!("{p}.mlp.{name}.weight"), shape)?;
                    }
                } else {
                    put(
                        &mut tensors,
                        format!("{p}.mlp.gate.weight"),
                        &[cfg.num_experts, h],
                    )?;
                    for (base, width, count) in [
                        (format!("{p}.mlp.shared_expert"), shared, 1usize),
                        (format!("{p}.mlp.experts"), inter, cfg.num_experts),
                    ] {
                        for expert in 0..count {
                            let base = if count == 1 {
                                base.clone()
                            } else {
                                format!("{base}.{expert}")
                            };
                            put(
                                &mut tensors,
                                format!("{base}.gate_proj.weight"),
                                &[width, h],
                            )?;
                            put(&mut tensors, format!("{base}.up_proj.weight"), &[width, h])?;
                            put(
                                &mut tensors,
                                format!("{base}.down_proj.weight"),
                                &[h, width],
                            )?;
                        }
                    }
                }
            }
        }

        let vb = mistralrs_quant::ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu);
        let metadata = NormalLoadingMetadata {
            mapper: Box::new(crate::device_map::DummyDeviceMapper {
                nm_device: Device::Cpu,
            }),
            loading_isq: false,
            real_device: Device::Cpu,
            multi_progress: Arc::new(indicatif::MultiProgress::new()),
            matformer_slicing_config: None,
            rope_pairing: None,
        };
        Model::new(cfg, vb, true, metadata, AttentionImplementation::Eager)
    }

    /// A missing or misnamed tensor surfaces here, on a 4 KB checkpoint rather than 79 GB.
    #[test]
    fn checkpoint_loads_with_the_shipped_tensor_names() {
        model_for(&small_config()).expect("every shipped tensor name resolves");
    }

    #[test]
    fn tiny_model_runs_a_forward() -> Result<()> {
        use crate::pipeline::text_models_inputs_processor::FlashParams;

        let cfg = small_config();
        let model = model_for(&cfg)?;
        let seq_len = 4;
        let input_ids = Tensor::ones((1, seq_len), DType::U32, &Device::Cpu)?;

        let flash_params = FlashParams::empty(false);
        // One entry per sequence: the position the sequence starts at.
        let seqlen_offsets = [0usize];
        let context_lens = [(0usize, seq_len)];
        let position_ids: Vec<usize> = (0..seq_len).collect();
        let mut ctx = ModelForwardContext::new(
            &seqlen_offsets,
            &context_lens,
            &position_ids,
            None,
            &flash_params,
        );
        let logits = model.forward(&input_ids, &mut ctx)?;
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
        assert!(logits
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()));
        Ok(())
    }
}
