#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer, Shard,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::moe::{MoEExperts, MoEExpertsConfig};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{sinks_backend_supports, AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        self, embedding_with_legacy_tied_uqff, Activation, CausalMasker, RmsNorm, RotaryEmbedding,
        Sdpa,
    },
    paged_attention::{
        AttentionBackendKind, AttentionImplementation, KvCacheLayout, ModelConfigLike,
        ModelConfigMetadata, PagedAttention, PrefixPrefillAttentionFeatures,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalCacheType,
        NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

const ROUTER_NORMALIZATION_EPS: f64 = 1e-20;

serde_default_fn!(bool, default_true, true);
serde_default_fn!(usize, default_sparse_step, 1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoftmaxType {
    #[default]
    Learnable,
    Standard,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterScoreFunction {
    #[default]
    Sigmoid,
    Softmax,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) head_dim: Option<usize>,
    pub(crate) quantization_config: Option<QuantizedConfig>,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
    #[serde(default = "default_true")]
    pub(crate) use_sliding_window: bool,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_experts: usize,
    #[serde(default)]
    pub(crate) mlp_only_layers: Vec<usize>,
    #[serde(default = "default_sparse_step")]
    pub(crate) decoder_sparse_step: usize,
    #[serde(default = "default_true")]
    pub(crate) norm_topk_prob: bool,
    pub(crate) num_experts_per_tok: usize,
    #[serde(default)]
    pub(crate) shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub(crate) layer_types: Vec<LayerType>,
    #[serde(default)]
    pub(crate) no_rope_layer_indices: Vec<usize>,
    #[serde(default)]
    pub(crate) attention_bias: bool,
    #[serde(default)]
    pub(crate) softmax_type: SoftmaxType,
    #[serde(default)]
    pub(crate) router_score_function: RouterScoreFunction,
}

impl Config {
    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub(crate) fn sliding_window(&self, layer_idx: usize) -> Option<usize> {
        if self.use_sliding_window
            && self.layer_types.get(layer_idx) == Some(&LayerType::SlidingAttention)
        {
            self.sliding_window
        } else {
            None
        }
    }

    pub(crate) fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    pub(crate) fn has_attention_sinks(&self) -> bool {
        self.softmax_type == SoftmaxType::Learnable
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if !self.layer_types.is_empty() && self.layer_types.len() != self.num_hidden_layers {
            candle_core::bail!("Flint layer_types must have num_hidden_layers entries");
        }
        if self.decoder_sparse_step == 0 {
            candle_core::bail!("Flint decoder_sparse_step must be positive");
        }
        if self.sliding_window == Some(0) {
            candle_core::bail!("Flint sliding_window must be positive");
        }
        Ok(())
    }
}

pub(crate) struct ModelConfig {
    base: ModelConfigMetadata,
    sliding_windows: Vec<Option<usize>>,
    has_sinks: bool,
}

impl ModelConfig {
    pub(crate) fn new(
        base: ModelConfigMetadata,
        sliding_windows: Vec<Option<usize>>,
        has_sinks: bool,
    ) -> Self {
        Self {
            base,
            sliding_windows,
            has_sinks,
        }
    }
}

impl ModelConfigLike for ModelConfig {
    fn max_seq_len(&self) -> usize {
        self.base.max_seq_len
    }
    fn num_layers(&self) -> usize {
        self.base.num_layers
    }
    fn hidden_size(&self) -> usize {
        self.base.hidden_size
    }
    fn num_kv_heads(&self) -> usize {
        self.base.num_kv_heads
    }
    fn num_attn_heads(&self) -> usize {
        self.base.num_attn_heads
    }
    fn k_head_dim(&self) -> usize {
        self.base.k_head_dim
    }
    fn v_head_dim(&self) -> usize {
        self.base.v_head_dim
    }
    fn sliding_window_for_layer(&self, layer_idx: usize) -> Option<usize> {
        self.sliding_windows[layer_idx]
    }
    fn attention_backend_kind(&self) -> AttentionBackendKind {
        self.base.attention_backend_kind()
    }
    fn attention_backend_kind_for_layer(&self, layer_idx: usize) -> AttentionBackendKind {
        self.base.attention_backend_kind_for_layer(layer_idx)
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.base.kv_cache_layout()
    }
    fn kv_cache_layout_for_layer(&self, layer_idx: usize) -> KvCacheLayout {
        self.base.kv_cache_layout_for_layer(layer_idx)
    }
    fn prefix_prefill_attention_features(
        &self,
        layer_idx: usize,
    ) -> Option<PrefixPrefillAttentionFeatures> {
        Some(PrefixPrefillAttentionFeatures {
            has_sinks: self.has_sinks,
            has_sliding_window: self.sliding_windows[layer_idx].is_some(),
            ..Default::default()
        })
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
    rotary_emb: Option<Arc<RotaryEmbedding>>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(ctx: &LayerLoadContext<'_>, vb: ShardedVarBuilder) -> Result<Self> {
        let LayerLoadContext {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            comm,
            ..
        } = *ctx;
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
        let kv_shard = mistralrs_quant::compute_kv_shard(cfg.num_key_value_heads, head_dim, comm)?;
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
        let sliding_window = cfg.sliding_window(layer_idx);
        let q_norm = RmsNorm::new(
            cfg.head_dim(),
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("q_norm"), false),
        )?;
        let k_norm = RmsNorm::new(
            cfg.head_dim(),
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("k_norm"), false),
        )?;
        let sinks = if cfg.has_attention_sinks() {
            Some(
                mapper
                    .set_device(layer_idx, vb.clone(), false)
                    .get_with_hints_dtype(
                        num_heads,
                        "softmax_offset",
                        Shard::Simple {
                            dim: 0,
                            rank: comm.rank(),
                            world_size: comm.world_size(),
                        },
                        vb.dtype(),
                    )?,
            )
        } else {
            None
        };
        let paged_attn = match ctx.attention_mechanism {
            AttentionImplementation::Eager => None,
            AttentionImplementation::PagedAttention => {
                Some(PagedAttention::new(head_dim, ctx.device, None)?)
            }
        };
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
            rotary_emb: (!cfg.no_rope_layer_indices.contains(&layer_idx))
                .then(|| ctx.rotary_emb.clone()),
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                )?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window,
                sinks,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let (mut q, mut k, mut v) =
            crate::ops::qkv_projections(xs, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        (q, k) = if let Some(rotary_emb) = &self.rotary_emb {
            let positions = ctx
                .text_positions(q.device(), q_len)?
                .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
            rotary_emb.forward_qk_norm(
                &q,
                &k,
                self.q_norm.weight(),
                self.k_norm.weight(),
                self.q_norm.eps(),
                self.k_norm.eps(),
                positions,
            )?
        } else {
            (
                self.q_norm.forward(&q.contiguous()?)?,
                self.k_norm.forward(&k.contiguous()?)?,
            )
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
                    assert!(!matches!(attention_mask, AttentionMask::None));
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
                let attention_mask = match attention_mask {
                    AttentionMask::Custom(mask)
                        if mask.dim(candle_core::D::Minus1)? > k.dim(2)? =>
                    {
                        let key_len = k.dim(2)?;
                        AttentionMask::Custom(mask.narrow(
                            candle_core::D::Minus1,
                            mask.dim(candle_core::D::Minus1)? - key_len,
                            key_len,
                        )?)
                    }
                    other => other.clone(),
                };

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    &attention_mask,
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
        let res = self.o_proj.forward(&attn_output)?;
        Ok(res)
    }
}

#[derive(Clone)]
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
}

impl Mlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        i_size: usize,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;

        let gate_proj = ColumnParallelLayer::new(
            hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("gate_proj"),
        )?;
        let up_proj = ColumnParallelLayer::new(
            hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("up_proj"),
        )?;
        let down_proj = RowParallelLayer::new(
            i_size,
            hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_out = self.gate_proj.forward(xs)?;
        let up_out = self.up_proj.forward(xs)?;
        let current_hidden_states = crate::ops::mul_and_act(&gate_out, &up_out, self.act_fn)?;
        let res = self.down_proj.forward(&current_hidden_states)?;
        Ok(res)
    }
}

struct MoeMlp {
    gate: Linear,
    gate_lora: Option<Arc<mistralrs_quant::LoraSiteHandle>>,
    experts: MoEExperts,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    score_function: RouterScoreFunction,
    expert_bias: Option<Tensor>,
    shared_expert: Option<Mlp>,
}

impl MoeMlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
    ) -> Result<Self> {
        let gate_vb = vb.pp("gate").set_device(layer_device.clone());
        let gate = layers::linear_no_bias(cfg.hidden_size, cfg.num_experts, gate_vb.clone())?;
        let gate_lora = mistralrs_quant::register_dynamic_lora_site(
            &gate_vb,
            mistralrs_quant::LoraLinearSpec::replicated(cfg.hidden_size, cfg.num_experts),
        )?;

        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
            expert_proj_names: crate::moe::ExpertProjNames::DEFAULT,
        };

        let expert_bias = if gate_vb.contains_tensor("expert_bias") {
            Some(gate_vb.get_with_hints_dtype(
                cfg.num_experts,
                "expert_bias",
                Default::default(),
                DType::F32,
            )?)
        } else {
            None
        };
        let shared_expert = (cfg.shared_expert_intermediate_size > 0)
            .then(|| {
                Mlp::new(
                    cfg,
                    vb.pp("shared_expert"),
                    comm,
                    cfg.shared_expert_intermediate_size,
                )
            })
            .transpose()?;

        let experts = MoEExperts::new(
            &moe_cfg,
            vb,
            layer_device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        Ok(Self {
            gate,
            gate_lora,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            score_function: cfg.router_score_function,
            expert_bias,
            shared_expert,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        let router_logits = self.gate.forward(&xs_flat)?;
        let router_logits = match &self.gate_lora {
            Some(site) => mistralrs_quant::apply_dynamic_lora_delta(site, &xs_flat, router_logits)?,
            None => router_logits,
        };
        let topk = route_experts(
            &router_logits,
            self.num_experts_per_tok,
            self.score_function,
            self.norm_topk_prob,
            self.expert_bias.as_ref(),
        )?;
        let mut ys = self.experts.forward(xs, topk.values, &topk.indices)?;
        if let Some(shared_expert) = &self.shared_expert {
            ys = (ys + shared_expert.forward(xs)?)?;
        }

        ys.reshape((b_size, seq_len, hidden_dim))
    }

    fn gate(&self) -> &Linear {
        &self.gate
    }
}

fn route_experts(
    logits: &Tensor,
    top_k: usize,
    score_function: RouterScoreFunction,
    renormalize: bool,
    expert_bias: Option<&Tensor>,
) -> Result<crate::ops::TopKOutput> {
    let mut topk = crate::ops::moe_router_topk(
        logits,
        crate::ops::MoeRouterTopKConfig {
            top_k,
            score_function: match score_function {
                RouterScoreFunction::Sigmoid => crate::ops::MoeRouterScoreFunction::Sigmoid,
                RouterScoreFunction::Softmax => crate::ops::MoeRouterScoreFunction::Softmax,
            },
            selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
            renormalize: false,
            norm_min: 0.0,
            output_scale: 1.0,
            logit_clip: None,
        },
        expert_bias,
        None,
    )?;
    if renormalize {
        let denominator =
            (topk.values.sum_keepdim(candle_core::D::Minus1)? + ROUTER_NORMALIZATION_EPS)?;
        topk.values = topk.values.broadcast_div(&denominator)?;
    }
    Ok(topk)
}

struct LayerLoadContext<'a> {
    cfg: &'a Config,
    mapper: &'a dyn DeviceMapper,
    layer_idx: usize,
    loading_isq: bool,
    device: &'a Device,
    comm: &'a Arc<mistralrs_quant::Comm>,
    rotary_emb: &'a Arc<RotaryEmbedding>,
    attention_mechanism: &'a AttentionImplementation,
}

enum MoeOrMlp {
    Moe(MoeMlp),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::Moe(m) => m.forward(xs),
        }
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MoeOrMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(ctx: &LayerLoadContext<'_>, vb: ShardedVarBuilder) -> Result<Self> {
        let LayerLoadContext {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            comm,
            ..
        } = *ctx;
        let self_attn = Attention::new(ctx, vb.pp("self_attn"))?;

        let mlp = if cfg.is_moe_layer(layer_idx) {
            let vb = mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq);
            let layer_device = mapper
                .device_for(layer_idx, false)
                .cloned()
                .unwrap_or_else(|| ctx.device.clone());

            MoeOrMlp::Moe(MoeMlp::new(cfg, vb, layer_device, comm, loading_isq)?)
        } else {
            MoeOrMlp::Mlp(Mlp::new(
                cfg,
                mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
                comm,
                cfg.intermediate_size,
            )?)
        };
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, kv_cache, ctx, layer_idx)?;
        let (residual, xs) = self
            .post_attention_layernorm
            .forward_add_rms_norm(&xs, residual)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

pub struct Model {
    embed_tokens: Arc<dyn QuantMethod>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    dtype: DType,
    sliding_window: Option<usize>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        cfg.validate()?;
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let dtype = vb_m.dtype();

        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb_lm_head.clone(), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;

        let head_dim = cfg.head_dim();
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            if let std::collections::hash_map::Entry::Vacant(entry) = ropes.entry(device.location())
            {
                entry.insert(Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?));
            }
        }

        let load_in_parallel =
            !(normal_loading_metadata.real_device.is_metal() && cfg.quantization_config.is_none());
        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .run(load_in_parallel, |layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                &LayerLoadContext {
                    cfg,
                    mapper: &*mapper,
                    layer_idx,
                    loading_isq: normal_loading_metadata.loading_isq,
                    device,
                    comm: &comm,
                    rotary_emb: &rotary_emb,
                    attention_mechanism: &attention_mechanism,
                },
                vb_l.pp(layer_idx),
            )
        })?;
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            embed_tokens.clone()
        };
        let cache_types = (0..cfg.num_hidden_layers)
            .map(|layer_idx| {
                cfg.sliding_window(layer_idx)
                    .map(|window| NormalCacheType::SlidingWindow { window })
                    .unwrap_or(NormalCacheType::Normal {
                        max_seq_len: cfg.max_position_embeddings,
                    })
            })
            .collect::<Vec<_>>();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            dtype,
            sliding_window: cfg
                .use_sliding_window
                .then_some(cfg.sliding_window)
                .flatten(),
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: None,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
                kv_cache_layout: crate::paged_attention::KvCacheLayout::StandardNoFlashInfer,
            },
            mapper,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward_embeds(
            input_ids,
            self.embed_tokens.embedding_forward(input_ids, self.dtype)?,
            ctx,
        )
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        input_embeds: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut xs = input_embeds;
        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let force_custom =
            !sinks_backend_supports(xs.dtype(), xs.device().location(), self.cfg.k_head_dim);
        let causal_mask = if !force_custom && xs.device().is_cuda() && input_ids.dim(1)? > 1 {
            AttentionMask::CausalFlash
        } else {
            CausalMasker.make_causal_mask(
                input_ids,
                &mask_cache,
                xs.dtype(),
                &CausalMaskConfig {
                    force_custom,
                    ..Default::default()
                },
            )?
        };
        let sliding_mask = if matches!(causal_mask, AttentionMask::CausalFlash) {
            AttentionMask::CausalFlash
        } else {
            CausalMasker.make_causal_mask(
                input_ids,
                &mask_cache,
                xs.dtype(),
                &CausalMaskConfig {
                    sliding_window: self.sliding_window,
                    force_custom,
                },
            )?
        };
        let causal_mask = if ctx.is_first_prompt_chunk() {
            causal_mask
        } else {
            AttentionMask::None
        };
        let sliding_mask = if ctx.is_first_prompt_chunk() {
            sliding_mask
        } else {
            AttentionMask::None
        };
        let causal_mask = DeviceMappedMask::new(causal_mask, &*self.mapper)?;
        let sliding_mask = DeviceMappedMask::new(sliding_mask, &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let mask = if layer.self_attn.sdpa_params.sliding_window.is_some() {
                sliding_mask.get(xs.device())
            } else {
                causal_mask.get(xs.device())
            };
            xs = layer.forward(&xs, &mask, &mut cache[i], ctx, i)?;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        ctx.lm_head(&*self.lm_head, &xs)
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
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
            if let MoeOrMlp::Moe(moe) = &layer.mlp {
                let gate = uvb_l.pp("mlp").pp("gate");
                gate.add(moe.gate());
                if let Some(bias) = &moe.expert_bias {
                    gate.add_tensor("expert_bias", bias.clone());
                }
            }
        }

        uvb.to_safetensors()
    }

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        Some(self.residual_tensors())
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward(input_ids, ctx)
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
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        candle_core::bail!("Flint does not support X-LoRA")
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(ModelConfig::new(
            self.cfg.clone(),
            self.layers
                .iter()
                .map(|layer| layer.self_attn.sdpa_params.sliding_window)
                .collect(),
            self.layers
                .iter()
                .any(|layer| layer.self_attn.sdpa_params.sinks.is_some()),
        ))
    }
    fn supports_packed_prefill(&self) -> bool {
        self.layers.iter().all(|layer| {
            sinks_backend_supports(
                self.dtype,
                layer.self_attn.q_norm.weight().device().location(),
                layer.self_attn.head_dim,
            )
        })
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        true
    }
}

impl AnyMoeBaseModelMixin for Model {}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENS: [u32; 7] = [1, 4, 2, 7, 0, 5, 3];
    const LOGIT_TOLERANCE: f32 = 1e-4;

    fn tiny_config() -> Config {
        Config {
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 6,
            num_hidden_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.,
            sliding_window: Some(3),
            head_dim: Some(4),
            use_sliding_window: true,
            moe_intermediate_size: 4,
            num_experts: 3,
            mlp_only_layers: vec![0],
            decoder_sparse_step: 1,
            norm_topk_prob: true,
            num_experts_per_tok: 2,
            shared_expert_intermediate_size: 5,
            layer_types: vec![
                LayerType::FullAttention,
                LayerType::SlidingAttention,
                LayerType::FullAttention,
            ],
            no_rope_layer_indices: vec![0, 2],
            ..Default::default()
        }
    }

    fn tiny_weights(cfg: &Config) -> Result<HashMap<String, Tensor>> {
        let mut weights = HashMap::new();
        let mut add = |name: String, shape: Vec<usize>| -> Result<()> {
            let seed = name.bytes().map(usize::from).sum::<usize>();
            let norm = name.contains("norm.weight");
            let values = (0..shape.iter().product())
                .map(|i| {
                    let value = (((i * 7 + seed) % 29) as f32 - 14.) / 100.;
                    if norm {
                        1. + value
                    } else {
                        value
                    }
                })
                .collect::<Vec<_>>();
            weights.insert(name, Tensor::from_vec(values, shape, &Device::Cpu)?);
            Ok(())
        };
        add(
            "model.embed_tokens.weight".into(),
            vec![cfg.vocab_size, cfg.hidden_size],
        )?;
        add("model.norm.weight".into(), vec![cfg.hidden_size])?;
        add(
            "lm_head.weight".into(),
            vec![cfg.vocab_size, cfg.hidden_size],
        )?;
        for layer in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                add(format!("{prefix}.{name}.weight"), vec![cfg.hidden_size])?;
            }
            for name in ["q_norm", "k_norm"] {
                add(
                    format!("{prefix}.self_attn.{name}.weight"),
                    vec![cfg.head_dim()],
                )?;
            }
            for (name, heads) in [
                ("q_proj", cfg.num_attention_heads),
                ("k_proj", cfg.num_key_value_heads),
                ("v_proj", cfg.num_key_value_heads),
            ] {
                add(
                    format!("{prefix}.self_attn.{name}.weight"),
                    vec![heads * cfg.head_dim(), cfg.hidden_size],
                )?;
            }
            add(
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim()],
            )?;
            add(
                format!("{prefix}.self_attn.softmax_offset"),
                vec![cfg.num_attention_heads],
            )?;
            let mlps = if cfg.is_moe_layer(layer) {
                add(
                    format!("{prefix}.mlp.gate.weight"),
                    vec![cfg.num_experts, cfg.hidden_size],
                )?;
                if layer == 1 {
                    add(
                        format!("{prefix}.mlp.gate.expert_bias"),
                        vec![cfg.num_experts],
                    )?;
                }
                let mut mlps = (0..cfg.num_experts)
                    .map(|expert| {
                        (
                            format!("{prefix}.mlp.experts.{expert}"),
                            cfg.moe_intermediate_size,
                        )
                    })
                    .collect::<Vec<_>>();
                mlps.push((
                    format!("{prefix}.mlp.shared_expert"),
                    cfg.shared_expert_intermediate_size,
                ));
                mlps
            } else {
                vec![(format!("{prefix}.mlp"), cfg.intermediate_size)]
            };
            for (mlp, intermediate) in mlps {
                for proj in ["gate_proj", "up_proj"] {
                    add(
                        format!("{mlp}.{proj}.weight"),
                        vec![intermediate, cfg.hidden_size],
                    )?;
                }
                add(
                    format!("{mlp}.down_proj.weight"),
                    vec![cfg.hidden_size, intermediate],
                )?;
            }
        }
        Ok(weights)
    }

    fn tiny_model() -> Result<Model> {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg)?;
        let vb = mistralrs_quant::ShardedSafeTensors::wrap(weights, DType::F32, Device::Cpu);
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

    fn forward_tokens(model: &Model, tokens: &[u32], offset: usize) -> Result<Tensor> {
        let input = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)?;
        let offsets = [offset];
        let context_lens = [(0, tokens.len())];
        let positions = [offset + tokens.len()];
        let flash_params = FlashParams::empty(true);
        let mut ctx =
            ModelForwardContext::new(&offsets, &context_lens, &positions, None, &flash_params);
        model.forward(&input, &mut ctx)
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
        let error = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(error < LOGIT_TOLERANCE, "max logit difference: {error}");
        Ok(())
    }

    #[test]
    fn cached_and_chunked_logits_match_full_prefill_across_sliding_window() -> Result<()> {
        let expected = forward_tokens(&tiny_model()?, &TOKENS, 0)?;
        for chunks in [vec![1; TOKENS.len()], vec![2, 3, 2]] {
            let model = tiny_model()?;
            let mut offset = 0;
            let mut logits = Vec::new();
            for count in chunks {
                logits.push(forward_tokens(
                    &model,
                    &TOKENS[offset..offset + count],
                    offset,
                )?);
                offset += count;
            }
            assert_close(&Tensor::cat(&logits, 1)?, &expected)?;
        }
        Ok(())
    }

    #[test]
    fn logits_match_pytorch_reference() -> Result<()> {
        let logits = forward_tokens(&tiny_model()?, &TOKENS, 0)?;
        let expected = Tensor::new(
            &[
                -0.23434460f32,
                -0.25956914,
                0.044062793,
                0.018838255,
                -0.006386241,
                -0.031610753,
                -0.056835264,
                -0.082059756,
            ],
            &Device::Cpu,
        )?
        .reshape((1, 1, 8))?;
        assert_close(&logits.narrow(1, TOKENS.len() - 1, 1)?, &expected)
    }

    #[test]
    fn routing_bias_affects_selection_only() -> Result<()> {
        let logits = Tensor::new(&[[0f32, 1., 2.]], &Device::Cpu)?;
        let bias = Tensor::new(&[2f32, 0., 0.], &Device::Cpu)?;
        let routed = route_experts(&logits, 2, RouterScoreFunction::Sigmoid, true, Some(&bias))?;
        assert_eq!(routed.indices.to_vec2::<u32>()?, vec![vec![0, 2]]);
        let scores = [0.5f32, 1. / (1. + (-2f32).exp())];
        let total = scores.iter().sum::<f32>();
        let expected = Tensor::new(&[[scores[0] / total, scores[1] / total]], &Device::Cpu)?;
        assert_close(&routed.values, &expected)
    }

    #[test]
    fn residuals_preserve_sinks_norms_router_and_optional_bias() -> Result<()> {
        let model = tiny_model()?;
        let residuals = model
            .residual_tensors()
            .into_iter()
            .collect::<HashMap<_, _>>();
        for layer in 0..3 {
            let prefix = format!("model.layers.{layer}");
            assert_eq!(
                residuals[&format!("{prefix}.self_attn.softmax_offset")].dims(),
                &[2]
            );
            for norm in ["q_norm", "k_norm"] {
                assert_eq!(
                    residuals[&format!("{prefix}.self_attn.{norm}.weight")].dims(),
                    &[4]
                );
            }
        }
        assert_eq!(
            residuals["model.layers.1.mlp.gate.expert_bias"].dims(),
            &[3]
        );
        assert!(!residuals.contains_key("model.layers.2.mlp.gate.expert_bias"));
        assert!(model.layers[0].self_attn.rotary_emb.is_none());
        assert!(model.layers[1].self_attn.rotary_emb.is_some());
        assert!(model.layers[2].self_attn.rotary_emb.is_none());
        assert!(model.config().sliding_window.is_none());
        let config = model.model_config();
        assert_eq!(config.sliding_window_for_layer(0), None);
        assert_eq!(config.sliding_window_for_layer(1), Some(3));
        assert_eq!(config.sliding_window_for_layer(2), None);
        assert!(
            config
                .prefix_prefill_attention_features(0)
                .unwrap()
                .has_sinks
        );
        assert!(
            config
                .prefix_prefill_attention_features(1)
                .unwrap()
                .has_sliding_window
        );
        Ok(())
    }

    #[test]
    fn config_rejects_mismatched_layer_types() {
        let mut cfg = tiny_config();
        cfg.layer_types.pop();
        assert!(cfg.validate().is_err());
    }
}
