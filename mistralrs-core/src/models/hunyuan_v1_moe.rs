#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use super::hunyuan_rope::{effective_rope_theta, RopeScalingConfig};
use crate::layers_masker::CausalMaskConfig;
use candle_core::{DType, Device, Module, Result, Tensor};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, MlpLayer},
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding, Activation, CausalMasker, RmsNorm, RotaryEmbedding, Sdpa},
    moe::{MoEExperts, MoEExpertsConfig},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalLoadingMetadata,
        NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, tie_word_embeddings_default, false);
serde_default_fn!(bool, use_cla_default, false);
serde_default_fn!(bool, use_mixed_mlp_moe_default, false);
serde_default_fn!(bool, moe_router_enable_expert_bias_default, false);
serde_default_fn!(bool, moe_router_use_sigmoid_default, false);
serde_default_fn!(bool, norm_topk_prob_default, true);
serde_default_fn!(f64, routed_scaling_factor_default, 1.0);
serde_default_fn!(usize, pretraining_tp_default, 1);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PerLayerValue {
    Scalar(usize),
    Array(Vec<usize>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderMlpKind {
    Dense,
    Moe,
}

impl DecoderMlpKind {
    fn for_num_experts(num_experts: usize) -> Self {
        if num_experts > 1 {
            Self::Moe
        } else {
            Self::Dense
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutingMode {
    Top1,
    TopK(usize),
}

impl RoutingMode {
    fn top_k(self) -> usize {
        match self {
            Self::Top1 => 1,
            Self::TopK(top_k) => top_k,
        }
    }

    fn applies_capacity(self) -> bool {
        matches!(self, Self::TopK(_))
    }

    fn renormalizes_selected_weights(self, norm_topk_prob: bool) -> bool {
        matches!(self, Self::TopK(_)) && norm_topk_prob
    }

    fn output_scale(self, routed_scaling_factor: f64) -> f32 {
        match self {
            Self::Top1 => 1.0,
            Self::TopK(_) => routed_scaling_factor as f32,
        }
    }
}

impl Default for PerLayerValue {
    fn default() -> Self {
        Self::Scalar(0)
    }
}

impl PerLayerValue {
    fn routing_mode(&self, layer_idx: usize) -> RoutingMode {
        match self {
            Self::Scalar(1) => RoutingMode::Top1,
            Self::Scalar(top_k) => RoutingMode::TopK(*top_k),
            Self::Array(top_k) => RoutingMode::TopK(top_k[layer_idx]),
        }
    }

    pub fn get(&self, layer_idx: usize) -> usize {
        match self {
            Self::Scalar(v) => *v,
            Self::Array(arr) => arr[layer_idx],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: Activation,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    pub num_experts: usize,
    #[serde(default)]
    pub num_shared_expert: PerLayerValue,
    #[serde(default)]
    pub moe_topk: PerLayerValue,
    #[serde(default)]
    pub moe_intermediate_size: PerLayerValue,
    #[serde(default = "use_mixed_mlp_moe_default")]
    pub use_mixed_mlp_moe: bool,
    #[serde(default, alias = "first_k_dense_replace")]
    pub moe_layer_num_skipped: usize,
    #[serde(default)]
    pub moe_drop_tokens: bool,
    #[serde(default)]
    pub moe_random_routing_dropped_token: bool,
    #[serde(
        default = "routed_scaling_factor_default",
        alias = "router_scaling_factor"
    )]
    pub routed_scaling_factor: f64,
    #[serde(default = "moe_router_enable_expert_bias_default")]
    pub moe_router_enable_expert_bias: bool,
    #[serde(default = "moe_router_use_sigmoid_default")]
    pub moe_router_use_sigmoid: bool,
    #[serde(default = "norm_topk_prob_default")]
    pub norm_topk_prob: bool,

    #[serde(default = "use_cla_default")]
    pub use_cla: bool,
    #[serde(default)]
    pub cla_share_factor: Option<usize>,

    #[serde(default)]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "pretraining_tp_default")]
    pub pretraining_tp: usize,
    #[serde(default)]
    pub add_classification_head: bool,
    #[serde(default = "tie_word_embeddings_default")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,

    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .filter(|&d| d > 0)
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn effective_rope_theta(&self) -> Result<f64> {
        effective_rope_theta(self.rope_theta, self.head_dim(), self.rope_scaling.as_ref())
    }

    pub(crate) fn uses_moe(&self) -> bool {
        DecoderMlpKind::for_num_experts(self.num_experts) == DecoderMlpKind::Moe
    }

    fn validate(&self) -> Result<()> {
        if !self.uses_moe() {
            return Ok(());
        }
        for (name, value) in [
            ("num_shared_expert", &self.num_shared_expert),
            ("moe_topk", &self.moe_topk),
            ("moe_intermediate_size", &self.moe_intermediate_size),
        ] {
            if let PerLayerValue::Array(values) = value {
                if values.len() != self.num_hidden_layers {
                    candle_core::bail!(
                        "HunYuanMoEV1 {name} has {} entries for {} layers",
                        values.len(),
                        self.num_hidden_layers
                    )
                }
            }
        }
        for (name, value) in [
            ("num_shared_expert", &self.num_shared_expert),
            ("moe_topk", &self.moe_topk),
        ] {
            if let PerLayerValue::Array(values) = value {
                if values.windows(2).any(|pair| pair[0] != pair[1]) {
                    candle_core::bail!(
                        "HunYuanMoEV1 official implementation requires uniform {name} across layers"
                    )
                }
            }
        }
        for layer in 0..self.num_hidden_layers {
            let top_k = self.moe_topk.get(layer);
            if top_k == 0 || top_k > self.num_experts {
                candle_core::bail!(
                    "HunYuanMoEV1 layer {layer} has invalid moe_topk={top_k} for {} experts",
                    self.num_experts
                )
            }
            let moe_intermediate_size = self.moe_intermediate_size.get(layer);
            if moe_intermediate_size != 0 && moe_intermediate_size != self.intermediate_size {
                candle_core::bail!(
                    "HunYuanMoEV1 layer {layer} has moe_intermediate_size={moe_intermediate_size}, but the official implementation uses intermediate_size={}",
                    self.intermediate_size
                )
            }
        }
        if !matches!(self.moe_topk, PerLayerValue::Scalar(1))
            && !self.moe_router_use_sigmoid
            && !self.moe_router_enable_expert_bias
            && (!self.norm_topk_prob || self.routed_scaling_factor != 1.0)
        {
            candle_core::bail!(
                "HunYuanMoEV1 official softmax routing requires norm_topk_prob=true and routed_scaling_factor=1"
            )
        }
        Ok(())
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
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
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(cfg.num_key_value_heads, head_dim, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;
        let (q_norm, k_norm) = if cfg.use_qk_norm {
            (
                Some(RmsNorm::new(
                    head_dim,
                    cfg.rms_norm_eps,
                    mapper.set_device(layer_idx, vb.pp("query_layernorm"), false),
                )?),
                Some(RmsNorm::new(
                    head_dim,
                    cfg.rms_norm_eps,
                    mapper.set_device(layer_idx, vb.pp("key_layernorm"), false),
                )?),
            )
        } else {
            (None, None)
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
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                )?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
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

        let rope_positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
        let (q, k) = self.rotary_emb.forward(&q, &k, rope_positions)?;
        let (q, k) = match (&self.q_norm, &self.k_norm) {
            (Some(q_norm), Some(k_norm)) => (q_norm.forward(&q)?, k_norm.forward(&k)?),
            (None, None) => (q, k),
            _ => unreachable!("Q/K norm configuration must be symmetric"),
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
        let res = self.o_proj.forward(&attn_output)?;
        Ok(res)
    }
}

struct SparseMlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
}

impl SparseMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act_fn: Activation,
        quant_cfg: &Option<QuantizedConfig>,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let gate_proj = ColumnParallelLayer::new(
            hidden_size,
            intermediate_size,
            quant_cfg,
            false,
            comm,
            vb.pp("gate_proj"),
        )?;
        let up_proj = ColumnParallelLayer::new(
            hidden_size,
            intermediate_size,
            quant_cfg,
            false,
            comm,
            vb.pp("up_proj"),
        )?;
        let down_proj = RowParallelLayer::new(
            intermediate_size,
            hidden_size,
            quant_cfg,
            false,
            comm,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_out = self.gate_proj.forward(xs)?;
        let up_out = self.up_proj.forward(xs)?;
        let activated = crate::ops::mul_and_act(&gate_out, &up_out, self.act_fn)?;
        self.down_proj.forward(&activated)
    }
}

fn hunyuan_moe_apply_capacity_mask_cpu(
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    num_experts: usize,
    top_k: usize,
) -> Result<Tensor> {
    let ids = topk_ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
    let mut weights = topk_weights
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;
    if num_experts == 0
        || top_k == 0
        || ids.len() != weights.len()
        || ids.iter().any(|row| row.len() != top_k)
        || weights.iter().any(|row| row.len() != top_k)
    {
        candle_core::bail!("HunYuan MoE capacity mask got invalid routing configuration")
    }

    let capacity = top_k.max(top_k * ids.len() / num_experts);
    let mut counts = vec![0usize; num_experts];
    for priority in 0..top_k {
        for token in 0..ids.len() {
            let expert = ids[token][priority] as usize;
            if expert >= num_experts {
                candle_core::bail!("HunYuan MoE router selected invalid expert {expert}")
            }
            if counts[expert] < capacity {
                counts[expert] += 1;
            } else {
                weights[token][priority] = 0.0;
            }
        }
    }

    Tensor::new(weights, &Device::Cpu)?.to_device(topk_weights.device())
}

fn hunyuan_moe_apply_capacity_mask(
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    num_experts: usize,
    top_k: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if topk_ids.device().is_cuda() && topk_weights.device().is_cuda() {
        return mistralrs_quant::moe::cuda::hunyuan_moe_apply_capacity_mask(
            topk_ids,
            topk_weights,
            num_experts,
            top_k,
        );
    }
    hunyuan_moe_apply_capacity_mask_cpu(topk_ids, topk_weights, num_experts, top_k)
}

fn hunyuan_moe_routed_weights(
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    num_experts: usize,
    routing_mode: RoutingMode,
) -> Result<Tensor> {
    if routing_mode.applies_capacity() {
        hunyuan_moe_apply_capacity_mask(topk_ids, topk_weights, num_experts, routing_mode.top_k())
    } else {
        Ok(topk_weights.clone())
    }
}

struct MoeBlock {
    gate: Arc<dyn QuantMethod>,
    shared_mlp: Option<SparseMlp>,
    experts: MoEExperts,
    num_experts: usize,
    routing_mode: RoutingMode,
    routed_scaling_factor: f64,
    norm_topk_prob: bool,
    router_score_function: crate::ops::MoeRouterScoreFunction,
    router_selection_bias: Option<Tensor>,
}

impl MoeBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        hidden_size: usize,
        num_experts: usize,
        intermediate_size: usize,
        num_shared_expert: usize,
        use_mixed_mlp_moe: bool,
        routing_mode: RoutingMode,
        routed_scaling_factor: f64,
        norm_topk_prob: bool,
        router_score_function: crate::ops::MoeRouterScoreFunction,
        router_selection_bias: Option<Tensor>,
        act_fn: Activation,
        quant_cfg: &Option<QuantizedConfig>,
        loading_isq: bool,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let gate = mistralrs_quant::linear_no_bias(
            hidden_size,
            num_experts,
            &None,
            vb.pp("gate").pp("wg").set_dtype(DType::F32),
        )?;

        let shared_mlp = if use_mixed_mlp_moe && num_shared_expert > 0 {
            let shared_intermediate_size = intermediate_size
                .checked_mul(num_shared_expert)
                .ok_or_else(|| {
                    candle_core::Error::msg("shared expert intermediate size overflow")
                })?;
            Some(SparseMlp::new(
                hidden_size,
                shared_intermediate_size,
                act_fn,
                quant_cfg,
                vb.pp("shared_mlp"),
                comm,
            )?)
        } else {
            None
        };

        let moe_cfg = MoEExpertsConfig {
            num_experts,
            num_experts_per_tok: routing_mode.top_k(),
            hidden_size,
            moe_intermediate_size: intermediate_size,
            expert_proj_names: crate::moe::ExpertProjNames::DEFAULT,
        };
        let experts = MoEExperts::new(
            &moe_cfg,
            vb.clone(),
            vb.device().clone(),
            comm,
            loading_isq,
            quant_cfg,
            act_fn,
        )?;

        Ok(Self {
            gate,
            shared_mlp,
            experts,
            num_experts,
            routing_mode,
            routed_scaling_factor,
            norm_topk_prob,
            router_score_function,
            router_selection_bias,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let router_logits = self.gate.forward(&xs_flat.to_dtype(DType::F32)?)?;
        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.routing_mode.top_k(),
                score_function: self.router_score_function,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: self
                    .routing_mode
                    .renormalizes_selected_weights(self.norm_topk_prob),
                norm_min: f32::EPSILON,
                output_scale: self.routing_mode.output_scale(self.routed_scaling_factor),
                logit_clip: None,
            },
            self.router_selection_bias.as_ref(),
            None,
        )?;
        let routed_weights = hunyuan_moe_routed_weights(
            &topk.indices,
            &topk.values,
            self.num_experts,
            self.routing_mode,
        )?;
        let mut output = self.experts.forward(xs, routed_weights, &topk.indices)?;

        if let Some(shared_mlp) = &self.shared_mlp {
            let shared = shared_mlp.forward(&xs_flat)?.reshape(xs.shape())?;
            output = (output + shared)?;
        }

        Ok(output)
    }
}

enum MoeOrMlp {
    Moe(MoeBlock),
    Mlp(SparseMlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Moe(moe) => moe.forward(xs),
            Self::Mlp(mlp) => mlp.forward(xs),
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
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            comm,
        )?;
        let mlp_vb = mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq);
        let mlp = match DecoderMlpKind::for_num_experts(cfg.num_experts) {
            DecoderMlpKind::Moe => MoeOrMlp::Moe(MoeBlock::new(
                cfg.hidden_size,
                cfg.num_experts,
                cfg.intermediate_size,
                cfg.num_shared_expert.get(layer_idx),
                cfg.use_mixed_mlp_moe,
                cfg.moe_topk.routing_mode(layer_idx),
                if cfg.routed_scaling_factor > 0.0 {
                    cfg.routed_scaling_factor
                } else {
                    1.0
                },
                cfg.norm_topk_prob,
                if cfg.moe_router_use_sigmoid {
                    crate::ops::MoeRouterScoreFunction::Sigmoid
                } else {
                    crate::ops::MoeRouterScoreFunction::Softmax
                },
                if cfg.moe_router_enable_expert_bias {
                    Some(mlp_vb.pp("gate").get_with_hints_dtype(
                        cfg.num_experts,
                        "e_score_correction_bias",
                        Default::default(),
                        DType::F32,
                    )?)
                } else {
                    None
                },
                cfg.hidden_act,
                &cfg.quantization_config,
                loading_isq,
                mlp_vb,
                comm,
            )?),
            DecoderMlpKind::Dense => MoeOrMlp::Mlp(SparseMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.hidden_act,
                &cfg.quantization_config,
                mlp_vb,
                comm,
            )?),
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
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
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
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::new_inner(
            cfg,
            vb_m,
            vb_lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
        cfg: &Config,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        cfg.validate()?;
        if cfg.use_cla {
            candle_core::bail!("HunYuanMoEV1 CLA is not implemented")
        }
        if cfg.attention_bias {
            candle_core::bail!("HunYuanMoEV1 attention_bias=true is not implemented")
        }
        if cfg.mlp_bias {
            candle_core::bail!("HunYuanMoEV1 mlp_bias=true is not implemented")
        }
        if cfg.pretraining_tp != 1 {
            candle_core::bail!("HunYuanMoEV1 pretraining_tp>1 is not implemented")
        }
        if cfg.add_classification_head {
            candle_core::bail!("HunYuanMoEV1 classification head is not implemented")
        }
        if cfg.moe_layer_num_skipped != 0 {
            candle_core::bail!("HunYuanMoEV1 skipped MoE layers are not implemented")
        }
        let rope_theta = cfg.effective_rope_theta()? as f32;
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        let mapper = normal_loading_metadata.mapper;

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        let head_dim = cfg.head_dim();
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_theta,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?),
            );
        }

        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| -> Result<DecoderLayer> {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
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
            ReplicatedLayer::from_linear(
                candle_nn::Linear::new(
                    mapper.cast_nm_device(
                        embed_tokens.embeddings(),
                        normal_loading_metadata.loading_isq,
                    )?,
                    None,
                ),
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
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
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward_embeds(input_ids, self.embed_tokens.forward(input_ids)?, ctx)
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
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig {
                sliding_window: None,
                ..Default::default()
            },
        )?;
        let attention_mask = if ctx.is_first_prompt_chunk() {
            attention_mask
        } else {
            AttentionMask::None
        };
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(&xs, &attention_mask.get(xs.device()), &mut cache[i], ctx, i)?;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        self.lm_head.forward(&xs)
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
            if let (Some(q_norm), Some(k_norm)) = (&layer.self_attn.q_norm, &layer.self_attn.k_norm)
            {
                uvb_l.pp("self_attn").pp("query_layernorm").add(q_norm);
                uvb_l.pp("self_attn").pp("key_layernorm").add(k_norm);
            }
            if let MoeOrMlp::Moe(moe) = &layer.mlp {
                let gate = uvb_l.pp("mlp").pp("gate");
                gate.pp("wg").add(&moe.gate);
                if let Some(bias) = &moe.router_selection_bias {
                    gate.add_tensor("e_score_correction_bias", bias.clone());
                }
            }
        }
        uvb.to_safetensors()
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
        unimplemented!()
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
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        false
    }
}

impl AnyMoeBaseModelMixin for Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        Vec::new()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        Vec::new()
    }
    fn create_anymoe_layers(
        &mut self,
        _additional_vbs: Vec<ShardedVarBuilder>,
        _config: AnyMoeConfig,
        _prefix_mlp: (String, String),
        _layers: Vec<usize>,
        _expert_type: AnyMoeExpertType,
        _gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        candle_core::bail!("AnyMoe is not supported for HunYuanMoEV1 mixed MLP MoE")
    }
    fn amoe_supported(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_top1_and_array_topk_one_use_distinct_routing_modes() {
        assert_eq!(PerLayerValue::Scalar(1).routing_mode(0), RoutingMode::Top1);
        assert_eq!(
            PerLayerValue::Array(vec![1]).routing_mode(0),
            RoutingMode::TopK(1)
        );
    }

    #[test]
    fn array_topk_one_applies_capacity() -> Result<()> {
        let ids = Tensor::new(
            vec![vec![0u32], vec![0u32], vec![0u32], vec![0u32]],
            &Device::Cpu,
        )?;
        let weights = Tensor::ones((4, 1), DType::F32, &Device::Cpu)?;
        let routed = hunyuan_moe_routed_weights(&ids, &weights, 4, RoutingMode::TopK(1))?;

        assert_eq!(
            routed.to_vec2::<f32>()?,
            vec![vec![1f32], vec![0f32], vec![0f32], vec![0f32]]
        );
        Ok(())
    }

    #[test]
    fn zero_or_one_expert_uses_dense_mlp() {
        assert_eq!(DecoderMlpKind::for_num_experts(0), DecoderMlpKind::Dense);
        assert_eq!(DecoderMlpKind::for_num_experts(1), DecoderMlpKind::Dense);
        assert_eq!(DecoderMlpKind::for_num_experts(2), DecoderMlpKind::Moe);
    }

    #[test]
    fn capacity_mask_applies_priority_then_token_order() -> Result<()> {
        let ids = Tensor::new(
            vec![
                vec![0u32, 1u32],
                vec![0u32, 1u32],
                vec![0u32, 1u32],
                vec![0u32, 1u32],
            ],
            &Device::Cpu,
        )?;
        let weights = Tensor::new(
            vec![
                vec![1f32, 2f32],
                vec![3f32, 4f32],
                vec![5f32, 6f32],
                vec![7f32, 8f32],
            ],
            &Device::Cpu,
        )?;

        let masked = hunyuan_moe_apply_capacity_mask_cpu(&ids, &weights, 4, 2)?;
        assert_eq!(
            masked.to_vec2::<f32>()?,
            vec![
                vec![1f32, 2f32],
                vec![3f32, 4f32],
                vec![0f32, 0f32],
                vec![0f32, 0f32],
            ]
        );
        Ok(())
    }

    #[test]
    fn capacity_mask_keeps_decode_routes() -> Result<()> {
        let ids = Tensor::new(vec![vec![3u32, 1u32]], &Device::Cpu)?;
        let weights = Tensor::new(vec![vec![0.25f32, 0.75f32]], &Device::Cpu)?;
        let masked = hunyuan_moe_apply_capacity_mask_cpu(&ids, &weights, 4, 2)?;

        assert_eq!(masked.to_vec2::<f32>()?, weights.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn topk_routing_always_applies_capacity() -> Result<()> {
        let ids = Tensor::new(
            vec![
                vec![0u32, 1u32],
                vec![0u32, 1u32],
                vec![0u32, 1u32],
                vec![0u32, 1u32],
            ],
            &Device::Cpu,
        )?;
        let weights = Tensor::ones((4, 2), DType::F32, &Device::Cpu)?;
        let routed = hunyuan_moe_routed_weights(&ids, &weights, 4, RoutingMode::TopK(2))?;

        assert_eq!(
            routed.to_vec2::<f32>()?,
            vec![
                vec![1f32, 1f32],
                vec![1f32, 1f32],
                vec![0f32, 0f32],
                vec![0f32, 0f32],
            ]
        );
        Ok(())
    }

    #[test]
    fn top1_routing_preserves_all_routes() -> Result<()> {
        let ids = Tensor::new(
            vec![vec![0u32], vec![0u32], vec![0u32], vec![0u32]],
            &Device::Cpu,
        )?;
        let weights = Tensor::ones((4, 1), DType::F32, &Device::Cpu)?;
        let routed = hunyuan_moe_routed_weights(&ids, &weights, 4, RoutingMode::Top1)?;

        assert_eq!(routed.to_vec2::<f32>()?, weights.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn scalar_top1_preserves_selected_softmax_probability() -> Result<()> {
        let routing_mode = PerLayerValue::Scalar(1).routing_mode(0);
        let logits = Tensor::new(vec![vec![3f32.ln(), 0f32]], &Device::Cpu)?;
        let topk = crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: routing_mode.top_k(),
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: routing_mode.renormalizes_selected_weights(true),
                norm_min: f32::EPSILON,
                output_scale: routing_mode.output_scale(2.5),
                logit_clip: None,
            },
            None,
            None,
        )?;

        let weight = topk.values.to_vec2::<f32>()?[0][0];
        assert!((weight - 0.75).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn moe_router_topk_sigmoid_bias_uses_bias_for_selection_only() -> Result<()> {
        let logits = Tensor::new(vec![vec![0f32, 0f32, 0f32]], &Device::Cpu)?;
        let bias = Tensor::new(vec![0f32, 1f32, 2f32], &Device::Cpu)?;
        let topk = crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: 2,
                score_function: crate::ops::MoeRouterScoreFunction::Sigmoid,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: true,
                norm_min: f32::EPSILON,
                output_scale: 2.0,
                logit_clip: None,
            },
            Some(&bias),
            None,
        )?;

        assert_eq!(topk.indices.to_vec2::<u32>()?, vec![vec![2, 1]]);
        assert_eq!(topk.values.to_vec2::<f32>()?, vec![vec![1.0, 1.0]]);
        Ok(())
    }

    #[test]
    fn moe_router_scales_after_renormalization() -> Result<()> {
        let logits = Tensor::new(vec![vec![10f32, 0f32], vec![0f32, 10f32]], &Device::Cpu)?;
        let topk = crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: 1,
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: true,
                norm_min: f32::EPSILON,
                output_scale: 2.5,
                logit_clip: None,
            },
            None,
            None,
        )?;

        assert_eq!(topk.values.to_vec2::<f32>()?, vec![vec![2.5], vec![2.5]]);
        Ok(())
    }
}
