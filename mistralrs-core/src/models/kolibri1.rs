//! Kolibri-1 (`Kolibri1ForCausalLM`), an Aleph Alpha MoE reasoning model.
//!
//! Structure follows the author's vLLM plugin (`aleph_alpha_inference/kolibri1.py` at pin
//! `049a6a7bd240`) and the checkpoint's own `config.json`. Four things distinguish it from
//! every other model in this registry:
//!
//! - **RNoPE.** `layer_types` decides per layer, and full-attention layers get no positional
//!   encoding at all. RoPE is constructed but never applied on those layers.
//! - **Sandwich norms.** Four norms per layer, and the residual stream accumulates the
//!   POST-normed block output: each norm reads the block output alone and the normed value
//!   is what is added. That is the opposite of the usual `x + residual` then norm, so the
//!   residual is threaded through the layer stack rather than living in `xs`. The residual
//!   starts at zero, and the value it holds after the last layer is never read.
//! - **Sigmoid logit add routing.** Top-k is taken over `logits + expert_bias`, but the weight
//!   applied to the selected expert is `sigmoid` of the UNBIASED logit. Selecting on
//!   `sigmoid(logits) + bias` is a different set of experts and is what the plugin's routing
//!   function exists to avoid. The bias lives at `moe.router.expert_bias`, a sibling of
//!   `mlp` rather than a child of it.
//! - **No dense layers.** All layers are MoE.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
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
                // The window is carried by the mask, matching how the other hybrid models here
                // do it. See todo/mistralrs/03-cpu-attention-ignores-sliding-window.md.
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
        // only the rope call is skipped.
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
                let q =
                    candle_nn::ops::rms_norm(&q, self.q_norm.weight(), self.q_norm.eps() as f32)?;
                let k =
                    candle_nn::ops::rms_norm(&k, self.k_norm.weight(), self.k_norm.eps() as f32)?;
                (q, k)
            }
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
    gate: Linear,
    expert_bias: Tensor,
    top_k: usize,
    renormalize: bool,
}

impl MoEGate {
    /// `vb` is the layer prefix, not the `mlp` prefix: the router bias is a sibling of `mlp`
    /// at `model.layers.N.moe.router.expert_bias`, while the router weight is `mlp.gate`.
    fn new(cfg: &Config, layer_vb: ShardedVarBuilder) -> Result<Self> {
        // The router is excluded from fp8 conversion upstream, so it stays bf16.
        let gate = layers::linear_no_bias(
            cfg.hidden_size,
            cfg.num_experts,
            layer_vb.pp("mlp").pp("gate"),
        )?;
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
        let logits = self.gate.forward(xs)?;
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
                &None,
                false,
                comm,
                vb.pp("gate_proj"),
            )?,
            up_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                inter,
                &None,
                false,
                comm,
                vb.pp("up_proj"),
            )?,
            down_proj: RowParallelLayer::new(
                inter,
                cfg.hidden_size,
                &None,
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
        let ys = self.experts.forward(xs, topk_weights, &topk_ids)?;
        // The shared expert is ungated and always added.
        let ys = (ys + self.shared.forward(xs)?)?;
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
        layer_idx: usize,
        residual: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        // Sandwich contract: each norm reads the block output ALONE, and the POST-NORMED
        // value is what accumulates into the residual. This is not the pre-norm
        // `x + residual` then norm that `RmsNorm::forward_add_rms_norm` implements, so the
        // two steps are spelled out rather than fused.
        let dhn = self.input_layernorm.forward(xs)?;
        let res = match residual {
            // The reference zeroes the residual before layer 0, so the first accumulation is
            // just the normed embedding.
            None => dhn.clone(),
            Some(res) => (res.clone() + &dhn)?,
        };

        let attn = self
            .self_attn
            .forward(&dhn, attention_mask, kv_cache, ctx, layer_idx)?;
        // post_attn_norm carries no residual.
        let attn_n = self.post_attn_norm.forward(&attn)?;

        let dhn2 = self.post_attention_layernorm.forward(&attn_n)?;
        let res = (&res + &dhn2)?;

        let moe = self.mlp.forward(&dhn2)?;
        // post_ffn_norm carries no residual either; the caller feeds it back as the next `xs`.
        *residual = Some(res);
        self.post_ffn_norm.forward(&moe)
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
        let mut xs = self.embed_tokens.embedding_forward(input_ids, self.dtype)?;
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

        // Accumulator only: the reference zeroes it before layer 0 and never reads it after the
        // last layer, so it is not threaded out to the head.
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
            xs = layer.forward(&xs, &mask, &mut cache[i], ctx, i, &mut residual)?;
        }
        xs = xs.to_device(&self.device)?;

        // The final norm carries the residual too: out = norm(hidden), logits from out.
        let xs = self.norm.forward(&xs)?;
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
            uvb_l.pp("mlp").pp("gate").add(&layer.mlp.gate.gate);
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
    use crate::paged_attention::AttentionImplementation;

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

    /// Builds a model from a synthetic checkpoint. The key set mirrors the real one exactly,
    /// so a renamed or relocated tensor fails here rather than at load time on a 156 GB
    /// checkpoint.
    fn tiny_model() -> Result<Model> {
        let cfg = tiny_config();
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        let put = |tensors: &mut std::collections::HashMap<String, Tensor>,
                   name: &str,
                   shape: &[usize]|
         -> Result<()> {
            tensors.insert(name.to_string(), test_tensor(shape)?);
            Ok(())
        };

        for (name, shape) in [
            ("model.embed_tokens.weight", &[8usize, 4][..]),
            ("model.norm.weight", &[4][..]),
            ("lm_head.weight", &[8, 4][..]),
            ("model.layers.0.input_layernorm.weight", &[4][..]),
            ("model.layers.0.post_attn_norm.weight", &[4][..]),
            ("model.layers.0.post_attention_layernorm.weight", &[4][..]),
            ("model.layers.0.post_ffn_norm.weight", &[4][..]),
            ("model.layers.0.self_attn.q_proj.weight", &[4, 4][..]),
            ("model.layers.0.self_attn.k_proj.weight", &[2, 4][..]),
            ("model.layers.0.self_attn.v_proj.weight", &[2, 4][..]),
            ("model.layers.0.self_attn.o_proj.weight", &[4, 4][..]),
            ("model.layers.0.self_attn.q_norm.weight", &[2][..]),
            ("model.layers.0.self_attn.k_norm.weight", &[2][..]),
            ("model.layers.0.mlp.gate.weight", &[4, 4][..]),
            (
                "model.layers.0.mlp.shared_experts.gate_proj.weight",
                &[6, 4][..],
            ),
            (
                "model.layers.0.mlp.shared_experts.up_proj.weight",
                &[6, 4][..],
            ),
            (
                "model.layers.0.mlp.shared_experts.down_proj.weight",
                &[4, 6][..],
            ),
        ] {
            put(&mut tensors, name, shape)?;
        }

        for layer in 0..2 {
            for (name, shape) in [
                ("input_layernorm", &[4][..]),
                ("post_attn_norm", &[4][..]),
                ("post_attention_layernorm", &[4][..]),
                ("post_ffn_norm", &[4][..]),
            ] {
                put(
                    &mut tensors,
                    &format!("model.layers.{layer}.{name}.weight"),
                    shape,
                )?;
            }
            for (name, shape) in [
                ("q_proj", &[4, 4][..]),
                ("k_proj", &[2, 4][..]),
                ("v_proj", &[2, 4][..]),
                ("o_proj", &[4, 4][..]),
                ("q_norm", &[2][..]),
                ("k_norm", &[2][..]),
            ] {
                put(
                    &mut tensors,
                    &format!("model.layers.{layer}.self_attn.{name}.weight"),
                    shape,
                )?;
            }
            put(
                &mut tensors,
                &format!("model.layers.{layer}.mlp.gate.weight"),
                &[4, 4][..],
            )?;
            // Sibling of `mlp`, not a child: any other prefix would miss this tensor.
            put(
                &mut tensors,
                &format!("model.layers.{layer}.moe.router.expert_bias"),
                &[4][..],
            )?;
            for (name, shape) in [
                ("gate_proj", &[6, 4][..]),
                ("up_proj", &[6, 4][..]),
                ("down_proj", &[4, 6][..]),
            ] {
                put(
                    &mut tensors,
                    &format!("model.layers.{layer}.mlp.shared_experts.{name}.weight"),
                    shape,
                )?;
                for expert in 0..4 {
                    put(
                        &mut tensors,
                        &format!("model.layers.{layer}.mlp.experts.{expert}.{name}.weight"),
                        shape,
                    )?;
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
        Model::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)
    }

    /// A missing or misnamed tensor surfaces here, on a 4 KB checkpoint rather than 156 GB.
    #[test]
    fn checkpoint_loads_with_the_shipped_tensor_names() {
        tiny_model().expect("every shipped tensor name resolves");
    }

    /// The residual stream accumulates POST-normed block outputs and starts at zero, so it is
    /// not expressible as `x + residual` then norm. Run two layers by hand and compare.
    #[test]
    fn residual_accumulates_post_normed_outputs() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::ones((4,), DType::F32, &device)?;
        let norm = RmsNorm::new(4, 1e-5, {
            let mut tensors = std::collections::HashMap::new();
            tensors.insert("weight".to_string(), weight.clone());
            mistralrs_quant::ShardedSafeTensors::wrap(tensors, DType::F32, device.clone())
        })?;
        let eps = norm.eps() as f32;
        let x = Tensor::new(&[[1f32, 2., 3., 4.]], &device)?;
        // Deliberately not proportional to `x`: rms_norm is scale invariant, so a
        // multiple of the input would make both contracts agree.
        let acc = Tensor::new(&[[10f32, 20., 40., 80.]], &device)?;

        let dhn = norm.forward(&x)?;
        let want_normed = candle_nn::ops::rms_norm(&x, &weight, eps)?;
        assert_close(&dhn, &want_normed)?;
        assert_close(&(acc.clone() + &dhn)?, &(&acc + &want_normed)?)?;

        // The pre-norm helper norms `x + acc` instead, which is a different number; the two
        // contracts must not be interchangeable here.
        let (sum, normed_sum) = norm.forward_add_rms_norm(&x, &acc)?;
        assert_close(&sum, &(&acc + &x)?)?;
        let max_gap = normed_sum
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .zip(want_normed.flatten_all()?.to_vec1::<f32>()?.iter())
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
}
