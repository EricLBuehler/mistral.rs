#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::{
    apply_immediate_isq, get_immediate_isq, immediate_isq_match, ColumnParallelLayer,
    IsqCaptureMode, MXFP4Layer, QuantMethod, QuantMethodConfig, QuantizedConfig, ReplicatedLayer,
    RowParallelLayer, Shard, ShardedVarBuilder, UnquantLinear,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{sinks_backend_supports, AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        self, embedding_with_legacy_tied_uqff, CausalMasker, GptOssRotaryEmbedding, RmsNorm,
        RotaryEmbedding, Sdpa,
    },
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalCacheType,
        NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, default_tie_word_embeddings, false);
serde_default_fn!(f32, default_alpha, 1.702);
serde_default_fn!(f32, default_swiglu_limit, 7.0);
serde_default_fn!(f64, default_beta_fast, 32.0);
serde_default_fn!(f64, default_beta_slow, 1.0);

/// YARN rope scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    #[serde(default = "default_beta_fast")]
    pub beta_fast: f64,
    #[serde(default = "default_beta_slow")]
    pub beta_slow: f64,
    #[serde(default)]
    pub truncate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    pub sliding_window: Option<usize>,
    pub head_dim: Option<usize>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub layer_types: Vec<LayerType>,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
    #[serde(default)]
    pub attention_bias: bool,
    pub rope_scaling: Option<RopeScaling>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

/// Wrapper enum for both standard and GPT-OSS YARN rotary embeddings
#[derive(Clone)]
pub enum GptOssRotaryEmbeddingVariant {
    Standard(Arc<RotaryEmbedding>),
    Yarn(Arc<GptOssRotaryEmbedding>),
}

impl GptOssRotaryEmbeddingVariant {
    pub fn forward(&self, q: &Tensor, k: &Tensor, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Standard(rope) => rope.forward(q, k, positions),
            Self::Yarn(rope) => rope.forward(q, k, positions),
        }
    }
}

/// Custom SwiGLU activation: (up + 1) * gate * sigmoid(gate * alpha)
/// With clamping: gate max=limit, up [-limit, limit]
#[allow(dead_code)]
fn gptoss_swiglu(gate: &Tensor, up: &Tensor, alpha: f32, limit: f32) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if gate.device().is_cuda() {
        return mistralrs_quant::gptoss_swiglu_fused(gate, up, alpha, limit);
    }

    let dtype = gate.dtype();
    let limit_d = limit as f64;

    let gate_clamped =
        gate.minimum(&Tensor::full(limit_d as f32, gate.shape(), gate.device())?.to_dtype(dtype)?)?;
    let up_clamped = up.clamp(-limit_d, limit_d)?;

    let gate_scaled = (&gate_clamped * alpha as f64)?;
    let sigmoid_val = candle_nn::ops::sigmoid(&gate_scaled)?;
    let glu = (&gate_clamped * &sigmoid_val)?;

    let up_plus_one = (&up_clamped + 1.0)?;
    up_plus_one.mul(&glu)
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: GptOssRotaryEmbeddingVariant,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    #[allow(dead_code)]
    is_sliding: bool,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: GptOssRotaryEmbeddingVariant,
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
            &None,
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        )?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &None,
            cfg.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &None,
            cfg.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &None,
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;

        let sinks = mapper
            .set_device(layer_idx, vb.clone(), false)
            .get((num_heads,), "sinks")?;

        let is_sliding = matches!(
            cfg.layer_types.get(layer_idx),
            Some(LayerType::SlidingAttention)
        );
        let sliding_window = if is_sliding { cfg.sliding_window } else { None };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
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
                sliding_window,
                sinks: Some(sinks),
            },
            is_sliding,
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
        (q, k) = self.rotary_emb.forward(&q, &k, rope_positions)?;
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

struct GptOssMoE {
    gate: Linear,
    gate_lora: Option<Arc<mistralrs_quant::LoraSiteHandle>>,
    projections: GptOssExpertProjections,
    expert_lora: Option<Arc<mistralrs_quant::LoraExpertSiteHandle>>,
    num_experts_per_tok: usize,
    intermediate_size: usize,
    alpha: f32,
    limit: f32,
}

enum GptOssExpertProjections {
    Interleaved {
        gate_up: Arc<dyn QuantMethod>,
        down: Arc<dyn QuantMethod>,
    },
    Split {
        gate: Arc<dyn QuantMethod>,
        up: Arc<dyn QuantMethod>,
        down: Arc<dyn QuantMethod>,
    },
}

fn load_gpt_oss_expert_projection(
    vb: &ShardedVarBuilder,
    name: &str,
    shape: (usize, usize, usize),
) -> Result<Arc<dyn QuantMethod>> {
    let projection_vb = vb.pp(name);
    if let Some(source) = vb.weight_source() {
        let load_device = mistralrs_quant::weight_source_load_device(&projection_vb);
        if let Some(layer) =
            source.load_linear(&projection_vb.prefix(), &load_device, Shard::default())?
        {
            return apply_immediate_isq(layer, projection_vb);
        }
    }
    let weight = projection_vb.get(shape, "weight")?;
    let bias = if projection_vb.contains_tensor("bias") {
        Some(projection_vb.get((shape.0, shape.1), "bias")?)
    } else {
        None
    };
    let layer = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
        Linear::new(weight, bias),
    ))?) as Arc<dyn QuantMethod>;
    apply_immediate_isq(layer, projection_vb)
}

fn load_gpt_oss_packed_expert_projection(
    num_local_experts: usize,
    in_dim: usize,
    out_dim: usize,
    name: &str,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let projection_vb = vb.pp(name);
    if get_immediate_isq().is_some_and(|params| params.capture == IsqCaptureMode::Immediate) {
        if let Some(target) = immediate_isq_match(&projection_vb).and_then(|matched| matched.ty) {
            if !MXFP4Layer::supports_stacked_isq(target) {
                candle_core::bail!(
                    "Cannot requantize raw GPT-OSS MXFP4 expert `{}` to {target}: that target does not support stacked expert gather. Use a Q*K/Q*_0/Q*_1 target, AFQ, MXFP4, or omit ISQ.",
                    projection_vb.prefix()
                );
            }
        }
    }
    if let Some(source) = vb.weight_source() {
        let load_device = mistralrs_quant::weight_source_load_device(&projection_vb);
        if let Some(layer) =
            source.load_linear(&projection_vb.prefix(), &load_device, Shard::default())?
        {
            return apply_immediate_isq(layer, projection_vb);
        }
    }
    let layer =
        MXFP4Layer::packed_gptoss_linear(num_local_experts, in_dim, out_dim, true, name, vb)?;
    apply_immediate_isq(layer, projection_vb)
}

fn has_interleaved_expert_projection(vb: &ShardedVarBuilder) -> bool {
    if vb.contains_tensor("gate_up_proj_blocks") {
        return true;
    }
    let weight = format!("{}.gate_up_proj.weight", vb.prefix());
    vb.weight_source()
        .is_some_and(|source| source.contains(&weight))
}

impl GptOssMoE {
    fn new(cfg: &Config, vb: ShardedVarBuilder, layer_device: Device) -> Result<Self> {
        let gate_vb = vb.pp("router").set_device(layer_device.clone());
        let gate = layers::linear(cfg.hidden_size, cfg.num_local_experts, gate_vb.clone())?;
        let gate_lora = mistralrs_quant::register_dynamic_lora_site(
            &gate_vb,
            mistralrs_quant::LoraLinearSpec::replicated(cfg.hidden_size, cfg.num_local_experts),
        )?;

        let experts_vb = vb.pp("experts").set_device(layer_device);
        let expert_lora = match experts_vb.lora_registry() {
            Some(registry) => Some(
                registry.register_expert(
                    mistralrs_quant::LoraSiteKey::new(experts_vb.prefix()),
                    mistralrs_quant::LoraExpertSiteSpec::new(
                        cfg.num_local_experts,
                        cfg.hidden_size,
                        cfg.intermediate_size,
                        mistralrs_quant::LoraExpertProjectionNames::new(
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ),
                        mistralrs_quant::Shard::default(),
                        mistralrs_quant::Shard::default(),
                    )?
                    .with_gate_up_order(mistralrs_quant::LoraGateUpOrder::Interleaved),
                    experts_vb.dtype(),
                    experts_vb.device().clone(),
                )?,
            ),
            None => None,
        };

        let projections = if has_interleaved_expert_projection(&experts_vb) {
            GptOssExpertProjections::Interleaved {
                gate_up: load_gpt_oss_packed_expert_projection(
                    cfg.num_local_experts,
                    cfg.hidden_size,
                    cfg.intermediate_size * 2,
                    "gate_up_proj",
                    experts_vb.clone(),
                )?,
                down: load_gpt_oss_packed_expert_projection(
                    cfg.num_local_experts,
                    cfg.intermediate_size,
                    cfg.hidden_size,
                    "down_proj",
                    experts_vb,
                )?,
            }
        } else {
            GptOssExpertProjections::Split {
                gate: load_gpt_oss_expert_projection(
                    &experts_vb,
                    "gate_proj",
                    (
                        cfg.num_local_experts,
                        cfg.intermediate_size,
                        cfg.hidden_size,
                    ),
                )?,
                up: load_gpt_oss_expert_projection(
                    &experts_vb,
                    "up_proj",
                    (
                        cfg.num_local_experts,
                        cfg.intermediate_size,
                        cfg.hidden_size,
                    ),
                )?,
                down: load_gpt_oss_expert_projection(
                    &experts_vb,
                    "down_proj",
                    (
                        cfg.num_local_experts,
                        cfg.hidden_size,
                        cfg.intermediate_size,
                    ),
                )?,
            }
        };

        Ok(Self {
            gate,
            gate_lora,
            projections,
            expert_lora,
            num_experts_per_tok: cfg.num_experts_per_tok,
            intermediate_size: cfg.intermediate_size,
            alpha: cfg.alpha,
            limit: cfg.swiglu_limit,
        })
    }

    fn forward(&self, xs: &Tensor, _layer_idx: usize) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        let router_logits = self.gate.forward(&xs_flat)?;
        let router_logits = match &self.gate_lora {
            Some(site) => mistralrs_quant::apply_dynamic_lora_delta(site, &xs_flat, router_logits)?,
            None => router_logits,
        };

        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.num_experts_per_tok,
                score_function: crate::ops::MoeRouterScoreFunction::Raw,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Softmax,
                renormalize: false,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )?;
        let (topk_weights, topk_ids) = (topk.values, topk.indices);

        let expert_lora = self
            .expert_lora
            .as_ref()
            .map(mistralrs_quant::LoraExpertExecution::current)
            .transpose()?
            .flatten();
        let routed_input = xs_flat.unsqueeze(1)?;

        let activated = match &self.projections {
            GptOssExpertProjections::Interleaved { gate_up, .. } => {
                gate_up.process_routed_stats(&xs_flat, &topk_ids)?;
                let gate_up = gate_up.gather_forward(&routed_input, &topk_ids)?;
                let (num_tokens, topk_dim, _) = gate_up.dims3()?;
                if let Some(lora) = &expert_lora {
                    let (gate, up) = lora.add_gate_up_delta_owned(&xs_flat, gate_up, &topk_ids)?;
                    gptoss_swiglu(&gate, &up, self.alpha, self.limit)?
                } else {
                    let gate_up =
                        gate_up.reshape((num_tokens, topk_dim, self.intermediate_size, 2))?;
                    let gate = gate_up.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
                    let up = gate_up.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
                    gptoss_swiglu(&gate, &up, self.alpha, self.limit)?
                }
            }
            GptOssExpertProjections::Split { gate, up, .. } => {
                gate.process_routed_stats(&xs_flat, &topk_ids)?;
                up.process_routed_stats(&xs_flat, &topk_ids)?;
                let mut gate = gate.gather_forward(&routed_input, &topk_ids)?;
                let mut up = up.gather_forward(&routed_input, &topk_ids)?;
                if let Some(lora) = &expert_lora {
                    gate = lora.add_delta_owned(
                        mistralrs_quant::LoraExpertProjection::Gate,
                        &xs_flat,
                        gate,
                        &topk_ids,
                        None,
                        mistralrs_quant::LoraExpertInputMode::TokenRows,
                    )?;
                    up = lora.add_delta_owned(
                        mistralrs_quant::LoraExpertProjection::Up,
                        &xs_flat,
                        up,
                        &topk_ids,
                        None,
                        mistralrs_quant::LoraExpertInputMode::TokenRows,
                    )?;
                }
                gptoss_swiglu(&gate, &up, self.alpha, self.limit)?
            }
        };

        let down = match &self.projections {
            GptOssExpertProjections::Interleaved { down, .. }
            | GptOssExpertProjections::Split { down, .. } => down,
        };
        down.process_routed_stats(&activated, &topk_ids)?;
        let expert_out = down.gather_forward(&activated, &topk_ids)?;
        let expert_out = match &expert_lora {
            Some(lora) => lora.add_delta_owned(
                mistralrs_quant::LoraExpertProjection::Down,
                &activated,
                expert_out,
                &topk_ids,
                None,
                mistralrs_quant::LoraExpertInputMode::RoutedRows,
            )?,
            None => expert_out,
        };

        let topk_weights = topk_weights
            .to_dtype(expert_out.dtype())?
            .unsqueeze(D::Minus1)?;
        let weighted = expert_out.broadcast_mul(&topk_weights)?;
        let output = weighted.sum(D::Minus2)?;

        output.reshape((b_size, seq_len, hidden_dim))
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: GptOssMoE,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: GptOssRotaryEmbeddingVariant,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        real_device: Device,
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

        let mlp = GptOssMoE::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("mlp"), false),
            real_device,
        )?;

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
        layer_idx: usize,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, kv_cache, ctx, layer_idx)?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs, layer_idx)?;

        residual + xs
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
    #[allow(dead_code)]
    cfg: Config,
    cfg_metadata: ModelConfigMetadata,
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let mapper = normal_loading_metadata.mapper;
        let dtype = vb_m.dtype();

        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq)
            }),
            &None,
        )?;

        let mut ropes: HashMap<_, GptOssRotaryEmbeddingVariant> = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);

            let rope = if let Some(rope_scaling) = &cfg.rope_scaling {
                if rope_scaling.rope_type == "yarn" {
                    GptOssRotaryEmbeddingVariant::Yarn(Arc::new(GptOssRotaryEmbedding::new(
                        cfg.rope_theta,
                        cfg.head_dim(),
                        cfg.max_position_embeddings,
                        rope_scaling.factor,
                        rope_scaling.original_max_position_embeddings,
                        rope_scaling.beta_fast,
                        rope_scaling.beta_slow,
                        rope_scaling.truncate,
                        device,
                        vb_m.dtype(),
                    )?))
                } else {
                    GptOssRotaryEmbeddingVariant::Standard(Arc::new(RotaryEmbedding::new(
                        cfg.rope_theta as f32,
                        cfg.head_dim(),
                        cfg.max_position_embeddings,
                        device,
                        is_gptx,
                        vb_m.dtype(),
                    )?))
                }
            } else {
                GptOssRotaryEmbeddingVariant::Standard(Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    cfg.head_dim(),
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?))
            };
            ropes.insert(device.location(), rope);
        }

        let vb_l = vb_m.pp("layers");

        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .into_iter()
        .map(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device)
                .clone();

            let rotary_emb = ropes
                .get(&device.location())
                .cloned()
                .expect("No RoPE for device");

            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim(), &device, None)?)
                }
            };

            let comm = mapper.get_comm_for(layer_idx)?;

            DecoderLayer::new(
                rotary_emb,
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                device,
                &comm,
            )
        })
        .collect::<Result<Vec<_>>>()?;

        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &None,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            embed_tokens.clone()
        };

        let head_dim = cfg.head_dim();
        let cfg_metadata = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::StandardNoFlashInfer,
        };

        let cache_types: Vec<NormalCacheType> = (0..cfg.num_hidden_layers)
            .map(|layer_idx| match cfg.layer_types.get(layer_idx) {
                Some(LayerType::SlidingAttention) => NormalCacheType::SlidingWindow {
                    window: cfg.sliding_window.unwrap_or(cfg.max_position_embeddings),
                },
                _ => NormalCacheType::Normal {
                    max_seq_len: cfg.max_position_embeddings,
                },
            })
            .collect();

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
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.embedding_forward(input_ids, self.dtype)?;
        let cache = &mut self.cache.normal().0;

        let sliding_window = self.cfg.sliding_window;

        let force_custom_attention_mask = !ctx.flash_params().packed;
        let mask_cache = ctx.mask_cache(cache);
        let causal_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig {
                force_custom: force_custom_attention_mask,
                ..Default::default()
            },
        )?;

        let sliding_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig {
                sliding_window,
                force_custom: force_custom_attention_mask,
            },
        )?;

        let should_use_mask = ctx.is_first_prompt_chunk();
        let causal_mask = if should_use_mask {
            causal_mask
        } else {
            AttentionMask::None
        };
        let sliding_mask = if should_use_mask {
            sliding_mask
        } else {
            AttentionMask::None
        };
        let causal_mask = DeviceMappedMask::new(causal_mask, &*self.mapper)?;
        let sliding_mask = DeviceMappedMask::new(sliding_mask, &*self.mapper)?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;

            let layer_mask = if layer.self_attn.is_sliding {
                let sm = sliding_mask.get(xs.device());
                if matches!(sm, AttentionMask::None) {
                    causal_mask.get(xs.device())
                } else {
                    sm
                }
            } else {
                causal_mask.get(xs.device())
            };

            xs = layer.forward(&xs, &layer_mask, &mut cache[i], i, ctx)?;
        }

        xs = xs.to_device(&self.device)?;
        xs = self.norm.forward(&xs)?;
        let xs = ctx.logits(&xs)?;

        self.lm_head.forward(&xs)
    }

    fn residual_tensors_inner(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);

            let uvb_attn = uvb_l.pp("self_attn");
            if let Some(sinks) = &layer.self_attn.sdpa_params.sinks {
                uvb_attn.add_tensor("sinks", sinks.clone());
            }
            uvb_l.pp("mlp").pp("router").add(&layer.mlp.gate);
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
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
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
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        candle_core::bail!("GPT-OSS does not support X-LoRA")
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
        &self.cfg_metadata
    }

    fn supports_packed_prefill(&self) -> bool {
        sinks_backend_supports(self.dtype, self.device.location(), self.cfg.head_dim())
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
    use mistralrs_quant::{
        uqff_version_tensors, IsqType, QuantizedSerde, ShardedSafeTensors, UqffReader, UqffTensor,
    };

    fn test_tensor<S: Into<candle_core::Shape>>(shape: S) -> Result<Tensor> {
        Tensor::zeros(shape, DType::F32, &Device::Cpu)
    }

    fn append_mxfp4_expert_layer(
        tensors: &mut Vec<UqffTensor>,
        prefix: &str,
        experts: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<()> {
        let layer = MXFP4Layer::from_parts(
            Tensor::zeros((experts, out_dim, in_dim / 2), DType::U8, &Device::Cpu)?,
            Tensor::zeros((experts, out_dim, in_dim / 32), DType::U8, &Device::Cpu)?,
            Some(Tensor::zeros((experts, out_dim), DType::F32, &Device::Cpu)?),
        );
        tensors.extend(layer.serialize_uqff(prefix, IsqType::MXFP4)?);
        Ok(())
    }

    fn residual_test_model() -> Result<Model> {
        let cfg = Config {
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 6,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            head_dim: Some(2),
            tie_word_embeddings: false,
            num_local_experts: 2,
            num_experts_per_tok: 1,
            layer_types: vec![LayerType::FullAttention],
            alpha: 1.702,
            swiglu_limit: 7.,
            attention_bias: true,
            ..Default::default()
        };
        let mut tensors = HashMap::from([
            (
                "model.embed_tokens.weight".to_string(),
                test_tensor((8, 4))?,
            ),
            ("model.norm.weight".to_string(), test_tensor(4)?),
            ("lm_head.weight".to_string(), test_tensor((8, 4))?),
            (
                "model.layers.0.input_layernorm.weight".to_string(),
                test_tensor(4)?,
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                test_tensor(4)?,
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                test_tensor((4, 4))?,
            ),
            (
                "model.layers.0.self_attn.q_proj.bias".to_string(),
                test_tensor(4)?,
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".to_string(),
                test_tensor((2, 4))?,
            ),
            (
                "model.layers.0.self_attn.k_proj.bias".to_string(),
                test_tensor(2)?,
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".to_string(),
                test_tensor((2, 4))?,
            ),
            (
                "model.layers.0.self_attn.v_proj.bias".to_string(),
                test_tensor(2)?,
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".to_string(),
                test_tensor((4, 4))?,
            ),
            (
                "model.layers.0.self_attn.o_proj.bias".to_string(),
                test_tensor(4)?,
            ),
            (
                "model.layers.0.self_attn.sinks".to_string(),
                test_tensor(2)?,
            ),
            (
                "model.layers.0.mlp.router.weight".to_string(),
                test_tensor((2, 4))?,
            ),
            (
                "model.layers.0.mlp.router.bias".to_string(),
                test_tensor(2)?,
            ),
        ]);
        for (name, shape) in [
            ("gate_proj", (2, 6, 4)),
            ("up_proj", (2, 6, 4)),
            ("down_proj", (2, 4, 6)),
        ] {
            tensors.insert(
                format!("model.layers.0.mlp.experts.{name}.weight"),
                test_tensor(shape)?,
            );
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

    fn residual_shapes(residual: Vec<(String, Tensor)>) -> HashMap<String, Vec<usize>> {
        residual
            .into_iter()
            .map(|(name, tensor)| (name, tensor.dims().to_vec()))
            .collect()
    }

    fn quant(weight: Tensor) -> Result<Arc<dyn QuantMethod>> {
        let weight = candle_core::quantized::QTensor::quantize(
            &weight,
            candle_core::quantized::GgmlDType::Q8_0,
        )?;
        Ok(Arc::new(mistralrs_quant::GgufMatMul::new(
            QuantMethodConfig::Gguf {
                q_weight: Arc::new(weight),
                b: None,
            },
        )?))
    }

    #[test]
    fn residual_tensors_cover_all_untracked_checkpoint_tensors() -> Result<()> {
        let model = residual_test_model()?;
        let expected = HashMap::from([
            ("model.norm.weight".to_string(), vec![4]),
            ("model.layers.0.input_layernorm.weight".to_string(), vec![4]),
            (
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                vec![4],
            ),
            ("model.layers.0.self_attn.sinks".to_string(), vec![2]),
            ("model.layers.0.mlp.router.weight".to_string(), vec![2, 4]),
            ("model.layers.0.mlp.router.bias".to_string(), vec![2]),
        ]);

        assert_eq!(residual_shapes(model.residual_tensors()), expected);
        assert_eq!(
            residual_shapes(
                model
                    .residual_tensors_moe_experts_only()
                    .expect("GPT-OSS supports MoQE residual serialization")
            ),
            expected
        );
        Ok(())
    }

    #[test]
    fn packed_experts_load_from_canonical_uqff_layers_without_raw_tensors() -> Result<()> {
        const EXPERTS: usize = 2;
        const HIDDEN: usize = 32;
        const INTERMEDIATE: usize = 32;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpt-oss-experts.uqff");
        let prefix = "model.layers.0.mlp.experts";
        let mut tensors = uqff_version_tensors();
        append_mxfp4_expert_layer(
            &mut tensors,
            &format!("{prefix}.gate_up_proj"),
            EXPERTS,
            INTERMEDIATE * 2,
            HIDDEN,
        )?;
        append_mxfp4_expert_layer(
            &mut tensors,
            &format!("{prefix}.down_proj"),
            EXPERTS,
            HIDDEN,
            INTERMEDIATE,
        )?;
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .unwrap();

        let residual = HashMap::from([
            (
                "model.layers.0.mlp.router.weight".to_string(),
                Tensor::zeros((EXPERTS, HIDDEN), DType::F32, &Device::Cpu)?,
            ),
            (
                "model.layers.0.mlp.router.bias".to_string(),
                Tensor::zeros(EXPERTS, DType::F32, &Device::Cpu)?,
            ),
        ]);
        let vb = ShardedSafeTensors::wrap(residual, DType::F32, Device::Cpu)
            .with_uqff_reader(Arc::new(UqffReader::open(&[path])?))
            .pp("model.layers.0.mlp");
        assert!(!vb.pp("experts").contains_tensor("gate_up_proj_blocks"));

        let cfg = Config {
            hidden_size: HIDDEN,
            intermediate_size: INTERMEDIATE,
            num_local_experts: EXPERTS,
            num_experts_per_tok: 1,
            ..Default::default()
        };
        let moe = GptOssMoE::new(&cfg, vb, Device::Cpu)?;
        match &moe.projections {
            GptOssExpertProjections::Interleaved { gate_up, down } => {
                assert_eq!(gate_up.name(), "mxfp4-layer");
                assert_eq!(down.name(), "mxfp4-layer");
            }
            GptOssExpertProjections::Split { .. } => {
                panic!("canonical UQFF expert layers must remain interleaved")
            }
        }

        let output = moe.forward(&Tensor::zeros((1, 1, HIDDEN), DType::F32, &Device::Cpu)?, 0)?;
        assert_eq!(output.dims(), &[1, 1, HIDDEN]);
        Ok(())
    }

    #[test]
    fn split_expert_forward_records_routed_stats() -> Result<()> {
        let device = Device::Cpu;
        let gate = quant(Tensor::ones((2, 32, 32), DType::F32, &device)?)?;
        let up = quant(Tensor::ones((2, 32, 32), DType::F32, &device)?)?;
        let down = quant(Tensor::ones((2, 32, 32), DType::F32, &device)?)?;
        let mut router = vec![0f32; 64];
        router[0] = 1.0;
        router[32] = -1.0;
        let moe = GptOssMoE {
            gate: Linear::new(Tensor::from_vec(router, (2, 32), &device)?, None),
            gate_lora: None,
            projections: GptOssExpertProjections::Split {
                gate: gate.clone(),
                up: up.clone(),
                down: down.clone(),
            },
            expert_lora: None,
            num_experts_per_tok: 1,
            intermediate_size: 32,
            alpha: 1.702,
            limit: 7.0,
        };
        let mut input = vec![1f32; 64];
        input[32] = -1.0;
        let input = Tensor::from_vec(input, (1, 2, 32), &device)?;

        gate.begin_track_stats()?;
        up.begin_track_stats()?;
        down.begin_track_stats()?;
        moe.forward(&input, 0)?;

        assert_eq!(gate.stats_snapshot(), Some((1, 2)));
        assert_eq!(up.stats_snapshot(), Some((1, 2)));
        assert_eq!(down.stats_snapshot(), Some((1, 2)));
        Ok(())
    }
}
