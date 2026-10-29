//! Kolibri-1 (`Kolibri1ForCausalLM`).
//!
//! Full-attention layers take no RoPE. qk-norm still runs, on a contiguous q and k.
//! The residual is the pre-norm stream: `post_attn_norm` and `post_ffn_norm` carry no
//! residual, and their outputs are the deltas the next layer adds. The router selects
//! on `logits + expert_bias` and weights by `sigmoid` of the unbiased logit. The bias
//! is `moe.router.expert_bias`, a sibling of `mlp`. Every layer is MoE.

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
pub struct RopeScaling {
    #[serde(rename = "type")]
    pub rope_type: String,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
}

/// Some checkpoint revisions carry `rope_parameters` instead of a flat `rope_theta`; the
/// plugin reads the object form, so both are accepted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: Option<f64>,
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
    pub(crate) rope_scaling: Option<RopeScaling>,
    pub(crate) rope_parameters: Option<RopeParameters>,
    pub(crate) quantization_config: Option<QuantizedConfig>,
    /// Present in the checkpoint, ignored by the reference plugin. Kept so config parse
    /// tolerates it rather than silently dropping a field the model declares.
    pub(crate) head_dtype: Option<String>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|p| p.rope_theta)
            .unwrap_or(self.rope_theta) as f32
    }

    /// `layer_types` is authoritative: it drives both the KV cache geometry and whether RoPE
    /// runs, so a length mismatch is a config error rather than something to paper over.
    pub fn validate(&self) -> Result<()> {
        if self.layer_types.len() != self.num_hidden_layers {
            candle_core::bail!(
                "Kolibri-1 layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            );
        }
        if self.use_sliding_window && self.sliding_window == 0 {
            candle_core::bail!("Kolibri-1 use_sliding_window requires a positive sliding_window");
        }
        Ok(())
    }
}

/// The shipped fp8 `config.json` lists only the per-layer routers in
/// `modules_to_not_convert`, but `lm_head`, `model.embed_tokens` and `model.norm` also ship
/// without a `weight_scale_inv`. Without the correction below, loading fails with "FP8 module
/// `lm_head` has no `weight_scale_inv` and is not listed in `modules_to_not_convert`" - which
/// is the loader being right about the config and the config being wrong about the tensors.
pub fn correct_fp8_exclusions(cfg: &Config) -> Config {
    let mut cfg = cfg.clone();
    let Some(QuantizedConfig::Fp8 {
        modules_to_not_convert,
        ..
    }) = cfg.quantization_config.as_mut()
    else {
        return cfg;
    };
    for module in ["lm_head", "model.embed_tokens", "model.norm"] {
        if !modules_to_not_convert.iter().any(|m| m == module) {
            modules_to_not_convert.push(module.to_string());
        }
    }
    cfg
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
    /// `None` on full-attention layers: RNoPE means no positional encoding at all.
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
                // The window is carried by the mask.
                sliding_window: None,
                sinks: None,
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

            // qk-norm precedes RoPE in the reference. On RNoPE layers the norm still runs and
            // only the rope call is skipped; `qk_rms_norm_rope` makes the transposed q/k
            // contiguous before the norm, so the norm-only branch has to do the same.
            let positions = ctx
                .text_positions(q.device(), q.dim(2)?)?
                .ok_or_else(|| candle_core::Error::msg("Kolibri-1 missing RoPE positions"))?;
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

        attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        self.o_proj.forward(&attn_output)
    }
}

/// Sigmoid logit add routing: select on `logits + bias`, weight by `sigmoid(logits)`.
///
/// The distinction from selecting on `sigmoid(logits) + bias` is the whole reason this is a
/// custom routing function upstream; the two pick different experts whenever the bias is
/// nonzero. `Raw` keeps the score the top-k sees as the raw logit, and `Sigmoid` with a
/// selection bias is the combination that reproduces it.
struct MoEGate {
    /// Held in f32. A bf16 matmul rounds the logits before top-k, and that changes which
    /// experts are selected.
    gate: Tensor,
    expert_bias: Tensor,
    top_k: usize,
    renormalize: bool,
}

impl MoEGate {
    /// `vb` is the layer prefix, not the `mlp` prefix: the router bias is a sibling of `mlp`
    /// at `model.layers.N.moe.router.expert_bias`, while the router weight is `mlp.gate`.
    fn new(cfg: &Config, layer_vb: ShardedVarBuilder) -> Result<Self> {
        // The checkpoint stores the router in bf16. Widen it before the matmul.
        let gate = layers::linear_no_bias(
            cfg.hidden_size,
            cfg.num_experts,
            layer_vb.pp("mlp").pp("gate"),
        )?
        .weight()
        .to_dtype(DType::F32)?
        .contiguous()?;
        let expert_bias = layer_vb
            .pp("moe")
            .pp("router")
            .get((cfg.num_experts,), "expert_bias")?;
        Ok(Self {
            gate,
            expert_bias,
            top_k: cfg.num_experts_per_tok,
            renormalize: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        // `norm_topk_prob` is false here, so the weights are sigmoid values used as mixture
        // weights with no renormalisation.
        let logits = xs.to_dtype(DType::F32)?.matmul(&self.gate.t()?)?;
        let topk = crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.top_k,
                score_function: crate::ops::MoeRouterScoreFunction::Raw,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Sigmoid,
                renormalize: self.renormalize,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            Some(&self.expert_bias),
            None,
        )?;
        Ok((topk.values, topk.indices))
    }
}

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
            other => candle_core::bail!("unsupported Kolibri-1 activation {other:?}"),
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
    /// `vb` is the layer prefix: `MoEGate` needs it to reach the sibling `moe.router` bias.
    fn new(
        cfg: &Config,
        layer_vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let vb = layer_vb.pp("mlp");
        let gate = MoEGate::new(cfg, layer_vb.clone())?;
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Kolibri-1 layer device not mapped"))?;

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

        let shared = SharedExpert::new(cfg, vb.pp("shared_experts"), mapper, layer_idx, comm)?;

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
        let ys = { self.experts.forward(xs, topk_weights, &topk_ids)? };
        // The shared expert is ungated and always added.
        let shared = { self.shared.forward(xs)? };
        let ys = (ys + shared)?;
        ys.reshape((b_size, seq_len, hidden_dim))
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MoeMlp,
    input_layernorm: RmsNorm,
    post_attn_norm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    post_ffn_norm: RmsNorm,
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
            // RNoPE: the full-attention layers never get a rope table.
            if kind.is_sliding() { rotary_emb } else { None },
            comm,
        )?;

        let mlp = MoeMlp::new(cfg, vb.clone(), mapper, layer_idx, loading_isq, comm)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attn_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attn_norm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            post_ffn_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_ffn_norm"))?,
        })
    }

    /// Sandwich layer. `residual` is threaded explicitly because the residual stream carries
    /// post-normed values here, which the usual pre-norm `xs = attn(norm(xs)) + xs` shape
    /// cannot express.
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut crate::pipeline::KvCache,
        ctx: &mut ModelForwardContext<'_>,
        residual: &mut Option<Tensor>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        // The residual is the pre-norm stream: each of the two residual-carrying norms reads
        // `residual + block output`. `post_attn_norm` and `post_ffn_norm` carry none, and
        // their outputs are the deltas the next layer adds, so `xs` and the residual advance
        // independently.
        let (res, dhn) = {
            match residual {
                // The reference zeroes the residual before layer 0, so the first accumulation is
                // just the embedding itself.
                None => (xs.clone(), self.input_layernorm.forward(xs)?),
                Some(res) => self.input_layernorm.forward_add_rms_norm(xs, res)?,
            }
        };

        let attn = self
            .self_attn
            .forward(&dhn, attention_mask, kv_cache, ctx, layer_idx)?;
        let (res, dhn2) = {
            // post_attn_norm carries no residual: its output is a delta, not a normalized stream.
            let attn_n = self.post_attn_norm.forward(&attn)?;
            self.post_attention_layernorm
                .forward_add_rms_norm(&attn_n, &res)?
        };

        let moe = self.mlp.forward(&dhn2)?;
        // Likewise for post_ffn_norm; this becomes the next layer's block output.
        *residual = Some(res);
        let ff = self.post_ffn_norm.forward(&moe)?;
        Ok(ff)
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
        let cfg = &correct_fp8_exclusions(cfg);
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
        // The checkpoint ships an untied head; tie_word_embeddings is honoured for fine-tunes
        // that set it rather than loading a head that should not exist.
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
                // One rope table serves the sliding layers; the full-attention layers
                // never get one (RNoPE).
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
                    Some(rotary_emb),
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
        let mut xs = { self.embed_tokens.embedding_forward(input_ids, self.dtype)? };
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

        let mut residual: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if layer.self_attn.kind.is_sliding() {
                sliding_mask.get(xs.device())
            } else {
                causal_mask.get(xs.device())
            };
            // The residual rides the same device map as the hidden states, so it has to be
            // carried across explicitly rather than left behind on the previous layer's device.
            if let Some(res) = &mut residual {
                let device = self
                    .mapper
                    .device_for(i, false)
                    .unwrap_or(&self.device)
                    .clone();
                *res = res.to_device(&device)?;
            }
            xs = layer.forward(&xs, &mask, &mut cache[i], ctx, &mut residual, i)?;
        }
        xs = xs.to_device(&self.device)?;

        // The final norm carries the residual too: the logits read norm(residual + hidden).
        let xs = {
            match &residual {
                Some(res) => self.norm.forward_add_rms_norm(&xs, res)?.1,
                None => self.norm.forward(&xs)?,
            }
        };
        let xs = ctx.logits(&xs)?;
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
            uvb_l.pp("post_attn_norm").add(&layer.post_attn_norm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            uvb_l.pp("post_ffn_norm").add(&layer.post_ffn_norm);
            uvb_l
                .pp("self_attn")
                .pp("q_norm")
                .add(&layer.self_attn.q_norm);
            uvb_l
                .pp("self_attn")
                .pp("k_norm")
                .add(&layer.self_attn.k_norm);
            uvb_l
                .pp("mlp")
                .pp("gate")
                .add_tensor("weight", layer.mlp.gate.gate.clone());
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

impl crate::speculative::SpeculativeTargetMixin for Model {}

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
        candle_core::bail!("Kolibri-1 does not support X-LoRA")
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    use crate::paged_attention::AttentionImplementation;
    use crate::pipeline::text_models_inputs_processor::FlashParams;
    use crate::utils::varbuilder_utils::DeviceForLoadTensor;

    /// The shipped `config.json`, verbatim. Keeps the deserializer honest about the field
    /// set the model actually ships, including `layer_types` and the `head_dtype` the plugin
    /// ignores.
    const REAL_CONFIG: &str = r#"{
  "architectures": ["Kolibri1ForCausalLM"],
  "model_type": "kolibri1",
  "hidden_size": 2560,
  "num_hidden_layers": 50,
  "num_attention_heads": 48,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "hidden_act": "silu",
  "max_position_embeddings": 262144,
  "rms_norm_eps": 1e-06,
  "vocab_size": 128000,
  "rope_theta": 10000.0,
  "num_experts": 384,
  "num_experts_per_tok": 6,
  "moe_intermediate_size": 512,
  "shared_expert_intermediate_size": 512,
  "norm_topk_prob": false,
  "attention_bias": false,
  "tie_word_embeddings": false,
  "use_sliding_window": true,
  "sliding_window": 513,
  "layer_types": [
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "full_attention"
  ],
  "eos_token_id": 127906,
  "pad_token_id": 127901,
  "dtype": "bfloat16",
  "head_dtype": "float32"
}"#;

    fn real_config() -> Config {
        serde_json::from_str(REAL_CONFIG).expect("shipped Kolibri-1 config deserializes")
    }

    fn real_config_with_fp8() -> String {
        let mut value: serde_json::Value = serde_json::from_str(REAL_CONFIG).unwrap();
        value["quantization_config"] = serde_json::json!({
          "quant_method": "fp8",
          "activation_scheme": "dynamic",
          "weight_block_size": [128, 128],
          "modules_to_not_convert": ["model.layers.0.mlp.gate"]
        });
        value.to_string()
    }

    #[test]
    fn shipped_config_deserializes_and_validates() {
        let cfg = real_config();
        cfg.validate().expect("shipped config passes validation");
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.rope_theta(), 10_000.0);
        assert!(!cfg.norm_topk_prob);
        assert_eq!(cfg.num_experts_per_tok, 6);
    }

    /// RNoPE is decided per layer by `layer_types`, and the shipped pattern is four sliding
    /// layers followed by one full. Getting this backwards silently corrupts every RoPE
    /// layer, so pin the exact indices.
    #[test]
    fn layer_types_mark_every_fifth_layer_full() {
        let cfg = real_config();
        let full: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|i| !cfg.layer_types[*i].is_sliding())
            .collect();
        assert_eq!(full, vec![4, 9, 14, 19, 24, 29, 34, 39, 44, 49]);
    }

    /// The shipped fp8 config omits three modules it ships unquantized. If the correction is
    /// dropped, loading the real checkpoint fails on `lm_head`.
    #[test]
    fn fp8_exclusions_cover_the_modules_the_checkpoint_ships_bf16() {
        let cfg: Config =
            serde_json::from_str(&real_config_with_fp8()).expect("fp8 config deserializes");
        let cfg = correct_fp8_exclusions(&cfg);
        let QuantizedConfig::Fp8 {
            modules_to_not_convert,
            ..
        } = cfg.quantization_config.as_ref().expect("fp8 kept")
        else {
            panic!("expected fp8");
        };
        for module in ["model.layers.0.mlp.gate", "lm_head", "model.embed_tokens"] {
            assert!(
                modules_to_not_convert.iter().any(|m| m == module),
                "{module} missing"
            );
        }
        // Idempotent: a config that already lists them must not gain duplicates.
        let before = modules_to_not_convert.len();
        let cfg = correct_fp8_exclusions(&cfg);
        let QuantizedConfig::Fp8 {
            modules_to_not_convert,
            ..
        } = cfg.quantization_config.as_ref().expect("fp8 kept")
        else {
            panic!("expected fp8");
        };
        assert_eq!(modules_to_not_convert.len(), before);
    }

    #[test]
    fn short_layer_types_is_a_config_error() {
        let mut cfg = real_config();
        cfg.layer_types.pop();
        assert!(cfg.validate().is_err());
    }

    fn test_tensor<S: Into<candle_core::Shape>>(shape: S) -> Result<Tensor> {
        Tensor::ones(shape, DType::F32, &Device::Cpu)
    }

    fn tiny_config() -> Config {
        small_config()
    }

    fn small_config() -> Config {
        Config {
            vocab_size: 8,
            hidden_size: 4,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: Some(2),
            max_position_embeddings: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            hidden_act: Activation::Silu,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 6,
            shared_expert_intermediate_size: 6,
            norm_topk_prob: false,
            attention_bias: false,
            tie_word_embeddings: false,
            use_sliding_window: true,
            sliding_window: 4,
            // One sliding layer, one full, so both paths are built.
            layer_types: vec![LayerType::SlidingAttention, LayerType::FullAttention],
            rope_scaling: None,
            rope_parameters: None,
            quantization_config: None,
            head_dtype: None,
        }
    }

    fn tiny_model() -> Result<Model> {
        model_for(&tiny_config())
    }

    /// Builds a model from a synthetic checkpoint whose key set mirrors the real one exactly,
    /// so a renamed or relocated tensor fails here rather than at load time on a 156 GB
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
                tensors.insert(name, test_tensor(shape)?);
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
                for name in [
                    "input_layernorm",
                    "post_attn_norm",
                    "post_attention_layernorm",
                    "post_ffn_norm",
                ] {
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
                put(
                    &mut tensors,
                    format!("{p}.mlp.gate.weight"),
                    &[cfg.num_experts, h],
                )?;
                // Sibling of `mlp`, not a child: any other prefix would miss this tensor.
                put(
                    &mut tensors,
                    format!("{p}.moe.router.expert_bias"),
                    &[cfg.num_experts],
                )?;
                for (base, width, count) in [
                    (format!("{p}.mlp.shared_experts"), shared, 1usize),
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

    /// A missing or misnamed tensor surfaces here, on a 4 KB checkpoint rather than 156 GB.
    #[test]
    fn checkpoint_loads_with_the_shipped_tensor_names() {
        tiny_model().expect("every shipped tensor name resolves");
    }

    #[test]
    fn tiny_model_runs_a_forward() -> Result<()> {
        run_forward(&tiny_config(), 4)?;
        Ok(())
    }

    /// The real model differs from `tiny_config` in a handful of dimensions at once. Each of
    /// these is swept on its own so a scale-dependent stall is attributable rather than
    /// guessed at; the whole sweep runs in seconds.
    #[test]
    fn scale_sweep_is_clean() -> Result<()> {
        let base = tiny_config();

        let many_experts = Config {
            num_experts: 384,
            num_experts_per_tok: 6,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            ..base.clone()
        };
        run_forward(&many_experts, 4)?;

        let long_context = Config {
            max_position_embeddings: 262_144,
            sliding_window: 513,
            ..base.clone()
        };
        run_forward(&long_context, 4)?;

        let wide = Config {
            hidden_size: 2560,
            head_dim: Some(128),
            num_attention_heads: 48,
            num_key_value_heads: 4,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            num_experts: 8,
            num_experts_per_tok: 6,
            vocab_size: 512,
            ..base.clone()
        };
        run_forward(&wide, 4)?;

        let deep = Config {
            num_hidden_layers: 50,
            layer_types: (0..50)
                .map(|i| {
                    if i % 5 == 4 {
                        LayerType::FullAttention
                    } else {
                        LayerType::SlidingAttention
                    }
                })
                .collect(),
            ..base.clone()
        };
        run_forward(&deep, 4)?;
        Ok(())
    }

    fn run_forward(cfg: &Config, seq_len: usize) -> Result<()> {
        let model = model_for(cfg)?;
        let mut ids: Vec<Vec<u32>> = vec![vec![0u32; seq_len]];
        let input_ids = Tensor::from_vec(ids.pop().unwrap(), (1, seq_len), &Device::Cpu)?;

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

    /// The residual is the pre-norm stream: `residual += block_output` and the norm reads
    /// that sum. Getting this backwards runs but drifts, which is what the golden check
    /// caught; the two contracts disagree on any non-proportional input.
    #[test]
    fn residual_is_the_pre_norm_stream() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::ones((4,), DType::F32, &device)?;
        let norm = RmsNorm::new(4, 1e-5, {
            let mut tensors = std::collections::HashMap::new();
            tensors.insert("weight".to_string(), weight.clone());
            mistralrs_quant::ShardedSafeTensors::wrap(tensors, DType::F32, device.clone())
        })?;
        let eps = norm.eps() as f32;
        let x = Tensor::new(&[[1f32, 2., 3., 4.]], &device)?;
        // Deliberately not proportional to `x`: rms_norm is scale invariant, so a multiple
        // of the input would make both contracts agree.
        let acc = Tensor::new(&[[10f32, 20., 40., 80.]], &device)?;

        let (res, dhn) = norm.forward_add_rms_norm(&x, &acc)?;
        assert_close(&res, &(&acc + &x)?)?;
        assert_close(
            &dhn,
            &candle_nn::ops::rms_norm(&(&acc + &x)?, &weight, eps)?,
        )?;

        // The post-norm contract normalizes the block output alone, which is a different
        // number; the two must not be interchangeable.
        let post_normed = candle_nn::ops::rms_norm(&x, &weight, eps)?;
        let max_gap = dhn
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .zip(post_normed.flatten_all()?.to_vec1::<f32>()?.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_gap > 1e-3, "contracts collapsed; {max_gap}");
        Ok(())
    }

    fn assert_close(lhs: &Tensor, rhs: &Tensor) -> Result<()> {
        let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
        let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
        for (l, r) in lhs.iter().zip(rhs.iter()) {
            assert!((l - r).abs() < 1e-6, "{l} != {r}");
        }
        Ok(())
    }

    /// Selecting on `logits + bias` and weighting by `sigmoid(logits)` is not the same as
    /// selecting on `sigmoid(logits) + bias`. With a bias that dominates, the two pick
    /// different experts, so pin the behaviour the reference specifies.
    #[test]
    fn routing_selects_on_biased_logits_and_weights_on_unbiased_sigmoid() -> Result<()> {
        let device = Device::Cpu;
        // Expert 0 is the raw winner; the bias flips the selection to expert 1.
        let logits = Tensor::new(&[[2.0f32, 1.9]], &device)?;
        let bias = Tensor::new(&[0.0f32, 1.0], &device)?;

        let topk = crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: 1,
                score_function: crate::ops::MoeRouterScoreFunction::Raw,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Sigmoid,
                renormalize: false,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            Some(&bias),
            None,
        )?;

        assert_eq!(topk.indices.to_vec2::<u32>()?, vec![vec![1u32]]);
        let want = sigmoid(1.9f32);
        let got = topk.values.to_vec2::<f32>()?;
        assert!((got[0][0] - want).abs() < 1e-6, "{got:?} != {want}");
        Ok(())
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    // ---------------------------------------------------------------------------------------
    // Golden validation against the reference CPU forward. Ignored by default: it loads the
    // 156 GB bf16 checkpoint.
    //
    //   cargo test --release -p mistralrs-core kolibri1_matches_reference_goldens -- --ignored
    //
    // The goldens are greedy-decode fingerprints produced by the vLLM plugin transcribed to
    // torch; see vllm.cpp `tests/vllm/models/kolibri1_goldens.json`. They are the only oracle
    // for this model, since transformers has no `Kolibri1ForCausalLM`.
    //
    //   KOLIBRI1_MODEL_DIR     checkpoint directory (default /mnt/models/Aleph-Alpha/Kolibri-1-bf16)
    //   KOLIBRI1_GOLDENS       goldens json (default the path above)
    //   KOLIBRI1_MAX_PROMPTS   stop after N prompts, for a quick smoke run
    // ---------------------------------------------------------------------------------------

    fn golden_paths() -> (std::path::PathBuf, std::path::PathBuf) {
        let model = std::env::var("KOLIBRI1_MODEL_DIR")
            .unwrap_or_else(|_| "/mnt/models/Aleph-Alpha/Kolibri-1-bf16".to_string());
        let goldens = std::env::var("KOLIBRI1_GOLDENS").unwrap_or_else(|_| {
            "/home/lu_zero/Sources/vllm.cpp/tests/vllm/models/kolibri1_goldens.json".to_string()
        });
        (
            std::path::PathBuf::from(model),
            std::path::PathBuf::from(goldens),
        )
    }

    fn load_real_model(dir: &std::path::Path) -> anyhow::Result<Model> {
        let mut filenames: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        filenames.sort();
        assert!(!filenames.is_empty(), "no safetensors in {}", dir.display());

        let config = std::fs::read_to_string(dir.join("config.json"))?;
        let device = Device::Cpu;
        let mapper = Box::new(crate::device_map::DummyDeviceMapper {
            nm_device: device.clone(),
        });
        let vb = crate::utils::varbuilder_utils::from_mmaped_safetensors(
            filenames,
            Vec::new(),
            Some(DType::BF16),
            &device,
            Vec::new(),
            true,
            None,
            |_| true,
            Arc::new(|_: String| DeviceForLoadTensor::Base),
        )?;

        let cfg: Config = serde_json::from_str(&config)?;
        Ok(Model::new(
            &cfg,
            vb,
            false,
            NormalLoadingMetadata {
                mapper,
                loading_isq: false,
                real_device: device,
                multi_progress: Arc::new(indicatif::MultiProgress::new()),
                matformer_slicing_config: None,
                rope_pairing: None,
            },
            AttentionImplementation::Eager,
        )?)
    }

    /// Runs one teacher-forced prefill and returns the final-position logits.
    fn teacher_forced_logits(model: &mut Model, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let device = Device::Cpu;
        let seq_len = ids.len();
        let input_ids = Tensor::from_vec(ids.to_vec(), (1, seq_len), &device)?;

        // Scoped: the forward takes the same mutex again, and it is not reentrant.
        {
            let mut cache = model.cache.normal();
            for layer in cache.0.iter_mut() {
                match layer {
                    crate::kv_cache::KvCache::Normal { k, v } => {
                        k.current_seq_len = 0;
                        v.current_seq_len = 0;
                    }
                    crate::kv_cache::KvCache::Rotating { k, v } => {
                        k.set_len(0)?;
                        v.set_len(0)?;
                    }
                    crate::kv_cache::KvCache::Shared { .. } => {}
                }
            }
        }

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
        // Every position's row, not just the last: the reference chain gives one comparison
        // per generated token, and comparing all of them costs the same single prefill.
        logits.i(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()
    }

    #[test]
    #[ignore = "loads the 156 GB Kolibri-1 checkpoint"]
    fn kolibri1_matches_reference_goldens() -> anyhow::Result<()> {
        let (model_dir, goldens_path) = golden_paths();
        let goldens: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&goldens_path)?)?;
        let mut prompts = goldens["prompts"]
            .as_array()
            .cloned()
            .expect("goldens have prompts");
        if let Ok(limit) = std::env::var("KOLIBRI1_MAX_PROMPTS") {
            prompts.truncate(limit.parse().expect("KOLIBRI1_MAX_PROMPTS is a count"));
        }
        assert!(!prompts.is_empty());

        eprintln!("[phase] loading {}", model_dir.display());
        let mut model = load_real_model(&model_dir)?;
        eprintln!("[phase] loaded; {} prompts", prompts.len());
        let mut worst_logit_gap: f32 = 0.0;
        let mut argmax_matches = 0usize;
        let mut argmax_total = 0usize;
        let mut top1_matches = 0usize;
        let mut chain_matches = 0usize;
        let mut chain_total = 0usize;
        let mut first_step_matches = 0usize;

        for prompt in prompts {
            let mut ids: Vec<u32> = prompt["input_ids"]
                .as_array()
                .expect("input_ids")
                .iter()
                .map(|v| v.as_u64().expect("token id") as u32)
                .collect();
            let generated: Vec<u32> = prompt["generated_ids"]
                .as_array()
                .expect("generated_ids")
                .iter()
                .map(|v| v.as_u64().expect("token id") as u32)
                .collect();

            // Fingerprint prefill: every generated token but the last, so the logits at the
            // final position are the ones the reference recorded.
            ids.extend_from_slice(&generated[..generated.len() - 1]);
            eprintln!(
                "[phase] prompt {:?}: teacher-forced {} tokens",
                prompt["prompt"].as_str().unwrap_or("?"),
                ids.len()
            );
            let logits = teacher_forced_logits(&mut model, &ids)?;
            eprintln!("[phase] prompt done");

            // Position `p` of this prefill predicts token `p + 1`, so position
            // `prompt_len - 1 + step` should reproduce `generated_ids[step]`.
            let prompt_len = ids.len() - (generated.len() - 1);
            for step in 0..(generated.len() - 1) {
                let row = &logits[prompt_len - 1 + step];
                let (argmax, _) =
                    row.iter()
                        .enumerate()
                        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                            if *v > bv {
                                (i, *v)
                            } else {
                                (bi, bv)
                            }
                        });
                chain_total += 1;
                if argmax as u64 == generated[step] as u64 {
                    chain_matches += 1;
                    if step == 0 {
                        first_step_matches += 1;
                    }
                } else {
                    println!(
                        "{}: step {step} argmax {argmax} want {}",
                        prompt["prompt"].as_str().unwrap_or("?"),
                        generated[step]
                    );
                }
            }

            let final_row = &logits[logits.len() - 1];
            let (argmax, _) = final_row.iter().enumerate().fold(
                (0usize, f32::NEG_INFINITY),
                |(bi, bv), (i, v)| {
                    if *v > bv {
                        (i, *v)
                    } else {
                        (bi, bv)
                    }
                },
            );
            argmax_total += 1;
            if argmax as u64 == generated[generated.len() - 1] as u64 {
                argmax_matches += 1;
            }

            if let Some(final_logits) = prompt["final_logits"].as_object() {
                // The goldens also carry sums over the whole vocab row. Those are a scale and
                // offset oracle the top-k cannot give: a ranking flip leaves them alone, but a
                // wrong logit gain or a missing bias term moves them directly.
                let sum: f64 = final_row.iter().map(|v| *v as f64).sum();
                let abs_sum: f64 = final_row.iter().map(|v| (*v as f64).abs()).sum();
                println!(
                    "{}: logits_sum {:.2} (want {:.2})  abs_sum {:.2} (want {:.2})",
                    prompt["prompt"].as_str().unwrap_or("?"),
                    sum,
                    final_logits["logits_sum"].as_f64().expect("logits_sum"),
                    abs_sum,
                    final_logits["abs_sum"].as_f64().expect("abs_sum"),
                );
                let want_ids = final_logits["topk_ids"].as_array().expect("topk_ids");
                let want_vals = final_logits["topk_logits"].as_array().expect("topk_logits");
                let want_top1 = want_ids[0].as_u64().expect("token id");
                if argmax as u64 == want_top1 {
                    top1_matches += 1;
                }
                for (id, want) in want_ids.iter().zip(want_vals.iter()) {
                    let id = id.as_u64().expect("token id") as usize;
                    let want = want.as_f64().expect("logit") as f32;
                    let got = final_row[id];
                    worst_logit_gap = worst_logit_gap.max((got - want).abs());
                    println!(
                        "{}: id {id} want {want:.4} got {got:.4}",
                        prompt["prompt"].as_str().unwrap_or("?")
                    );
                }
            }
        }

        println!(
            "first-step {first_step_matches}/{argmax_total}, \
             chain argmax {chain_matches}/{chain_total}, \
             final-token argmax {argmax_matches}/{argmax_total}, \
             reference top-1 {top1_matches}/{argmax_total}, \
             worst top-k logit gap {worst_logit_gap:.4}"
        );
        // The floors catch a collapse. They do not certify the reference. Final-position top-1
        // needs 6 of 8, and the greedy chain needs a rate of 0.55. The chain is weak because the
        // generated sequences repeat, so one early flip marks every later step. The first step
        // is printed and not gated: the boundary between the 6th and 7th routed expert sits
        // inside the arithmetic's own noise, so the first token of a short prompt follows rounding.
        let fp8 = std::fs::read_to_string(model_dir.join("config.json"))?
            .contains("\"quant_method\": \"fp8\"");
        let chain_rate = chain_matches as f64 / chain_total as f64;
        assert!(
            top1_matches * 4 >= argmax_total * 3,
            "reproduced {top1_matches}/{argmax_total} reference top-1 tokens, fp8 {fp8}"
        );
        assert!(
            chain_rate >= 0.55,
            "chain argmax {chain_matches}/{chain_total} = {:.1}% (repetition chains, so this \
             is a sanity floor rather than a correctness measure), fp8 {fp8}, \
             first step {first_step_matches}/{argmax_total}",
            chain_rate * 100.0
        );
        Ok(())
    }

    /// Per-layer cache geometry has to follow `layer_types`, or the sliding layers allocate
    /// full-context KV and the memory advantage of the 4:1 pattern disappears. Sliding
    /// layers get a rotating cache sized to the window; full layers get a normal one.
    #[test]
    fn cache_types_follow_layer_types() {
        let cfg = tiny_config();
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
        assert!(matches!(
            cache_types[0],
            NormalCacheType::SlidingWindow { window: 4 }
        ));
        assert!(matches!(
            cache_types[1],
            NormalCacheType::Normal { max_seq_len: 8 }
        ));

        let cache = NormalCache::from_types(cache_types);
        let cache = cache.lock().unwrap();
        assert!(matches!(
            cache.0[0],
            crate::kv_cache::KvCache::Rotating { .. }
        ));
        assert!(matches!(
            cache.0[1],
            crate::kv_cache::KvCache::Normal { .. }
        ));
    }

    /// Behavioural probes for the forward pass, none of which need the reference fingerprints.
    ///
    /// The golden gate compares one position per prompt against a chain the reference produced
    /// by greedy decoding, on prompts where this model loops degenerately. That is a weak
    /// signal: it cannot distinguish a broken forward from a numerically close one, and it
    /// needs the goldens to interpret. These probes need neither, and they separate the two
    /// cases - a forward that cannot name a capital is broken regardless of its logit gap.
    #[test]
    #[ignore = "loads the 149 GB Kolibri-1 checkpoint"]
    fn kolibri1_answers_ordinary_prompts() -> anyhow::Result<()> {
        use tokenizers::Tokenizer;

        let (model_dir, _) = golden_paths();
        let tok = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        eprintln!("[probe] loading {}", model_dir.display());
        let mut model = load_real_model(&model_dir)?;

        // Australia is printed and not gated. Fused RMSNorm rounds the scale to the activation
        // dtype before the divide, and that changes this answer.
        for (text, want) in [
            ("The capital of France is", Some("Paris")),
            ("The capital of Australia is", None),
            ("The capital of Japan is", Some("Tokyo")),
        ] {
            let ids = tok
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            let rows = teacher_forced_logits(&mut model, &ids)?;
            let last = rows.last().expect("prefill returned no rows");
            let (top, top_logit) =
                last.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                        if *v > bv {
                            (i, *v)
                        } else {
                            (bi, bv)
                        }
                    });
            let got = tok.decode(&[top as u32], true).unwrap_or_default();
            let margin = top_logit
                - last
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != top)
                    .fold(f32::NEG_INFINITY, |m, (_, v)| m.max(*v));
            println!("{text:?} -> {got:?} ({top_logit:.2}, margin {margin:.2})");
            if let Some(want) = want {
                assert_eq!(got.trim(), want, "prompt {text:?}");
            }
        }
        Ok(())
    }

    /// Free generation and teacher-forcing are the same computation, so they must agree.
    ///
    /// No reference oracle is needed: the chain the incremental path produces is the
    /// reference for the prefill path. If the two disagree, the prefill is the wrong one,
    /// because the incremental path is what the model actually does at serving time.
    ///
    /// The offset scan reports the direction and size of any misalignment rather than just
    /// failing, because a one-position offset is the specific failure this exists to catch.
    #[test]
    #[ignore = "loads a Kolibri-1 checkpoint"]
    fn kolibri1_prefill_agrees_with_incremental() -> anyhow::Result<()> {
        let (model_dir, goldens_path) = golden_paths();
        let goldens: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&goldens_path)?)?;
        // Prompt 2 of the goldens: "The capital of Australia is", whose continuation is varied
        // rather than a repeated token. A degenerate chain cannot show a position shift,
        // because every position looks the same.
        let prompt: Vec<u32> = goldens["prompts"][2]["input_ids"]
            .as_array()
            .expect("input_ids")
            .iter()
            .map(|v| v.as_u64().expect("token id") as u32)
            .collect();

        let mut model = load_real_model(&model_dir)?;
        let n_new = 24usize;

        // Incremental: the KV cache carries the history, one token at a time.
        let mut chain: Vec<u32> = Vec::new();
        {
            let prompt_ids = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &Device::Cpu)?;
            let mut next = prefill_and_decode(&mut model, &prompt_ids, 0)?;
            for step in 0..n_new {
                chain.push(next);
                let ids = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
                next = prefill_and_decode(&mut model, &ids, prompt.len() + step)?;
            }
        }
        println!("[align] chain: {chain:?}");

        // Teacher-forced over the same chain.
        let mut ids = prompt.clone();
        ids.extend_from_slice(&chain[..chain.len() - 1]);
        let rows = teacher_forced_logits(&mut model, &ids)?;

        // Row `prompt.len() - 1 + k` predicts `ids[prompt.len() + k]`, which is `chain[k]`.
        // A prefill that is one position behind would make it predict `chain[k - 1]`.
        let mut best = (0i64, 0usize, 0usize);
        for offset in -2i64..=2 {
            let (mut hits, mut tot) = (0usize, 0usize);
            for k in 0..(chain.len() - 1) {
                let kk = k as i64 + offset;
                if kk < 0 || kk as usize >= chain.len() {
                    continue;
                }
                let row_idx = prompt.len() as i64 - 1 + k as i64;
                if row_idx < 0 || row_idx as usize >= rows.len() {
                    continue;
                }
                let row = &rows[row_idx as usize];
                let (argmax, _) =
                    row.iter()
                        .enumerate()
                        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                            if *v > bv {
                                (i, *v)
                            } else {
                                (bi, bv)
                            }
                        });
                tot += 1;
                if argmax as u32 == chain[kk as usize] {
                    hits += 1;
                }
            }
            println!("  offset {offset:+}: {hits}/{tot}");
            if hits > best.1 {
                best = (offset, hits, tot);
            }
        }
        println!("[align] best offset {}: {}/{}", best.0, best.1, best.2);
        assert_eq!(
            best.0, 0,
            "the teacher-forced prefill is offset by {} position(s): row p predicts chain[p+{}] rather than chain[p+1]",
            best.0, best.0
        );
        Ok(())
    }

    /// One forward with the KV cache positioned so the next token appends at `cached`. Returns
    /// the argmax token so the caller can chain it.
    fn prefill_and_decode(model: &mut Model, ids: &Tensor, cached: usize) -> Result<u32> {
        let seq_len = ids.dim(1)?;
        let flash_params = FlashParams::empty(false);
        let seqlen_offsets = [cached];
        // `context_lens` indexes this forward's own output, so it starts at 0 on a decode step;
        // `cached` belongs in `seqlen_offsets`, which is what places the RoPE positions.
        let context_lens = [(0usize, seq_len)];
        let position_ids: Vec<usize> = (cached..cached + seq_len).collect();
        let mut ctx = ModelForwardContext::new(
            &seqlen_offsets,
            &context_lens,
            &position_ids,
            None,
            &flash_params,
        );
        let logits = model.forward(ids, &mut ctx)?;
        let last = logits.i((.., seq_len - 1, ..))?.flatten_all()?;
        last.argmax(0)?.to_scalar::<u32>()
    }
}
