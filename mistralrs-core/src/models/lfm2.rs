#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, Linear};
use mistralrs_quant::{
    ColumnParallelLayer, Convolution, QuantMethod, QuantizedConfig, ReplicatedLayer,
    RowParallelLayer, ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    },
    layers::{self, embedding, Activation, CausalMasker, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::{CausalMaskConfig, PastKvLenCache},
    moe::{MoEExperts, MoEExpertsConfig},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, ForwardMaskCache, IsqModel, KvCache, ModelForwardContext,
        NormalLoadingMetadata, NormalModel, RecurrentBatchKind,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, default_true, true);
serde_default_fn!(bool, default_false, false);
serde_default_fn!(usize, default_vocab_size, 65536);
serde_default_fn!(usize, default_hidden_size, 1024);
serde_default_fn!(usize, default_intermediate_size, 2560);
serde_default_fn!(usize, default_num_hidden_layers, 14);
serde_default_fn!(usize, default_num_attention_heads, 16);
serde_default_fn!(usize, default_num_key_value_heads, 8);
serde_default_fn!(usize, default_max_position_embeddings, 128000);
serde_default_fn!(usize, default_conv_l_cache, 3);
serde_default_fn!(usize, default_block_multiple_of, 256);
serde_default_fn!(usize, default_moe_intermediate_size, 1792);
serde_default_fn!(usize, default_num_dense_layers, 2);
serde_default_fn!(usize, default_num_experts, 32);
serde_default_fn!(usize, default_num_experts_per_tok, 4);
serde_default_fn!(f64, default_norm_eps, 1e-5);
serde_default_fn!(f64, default_block_ffn_dim_multiplier, 1.0);
serde_default_fn!(f64, default_rope_theta, 1_000_000.0);
serde_default_fn!(f64, default_routed_scaling_factor, 1.0);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: String,
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            rope_theta: default_rope_theta(),
            rope_type: "default".to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default)]
    pub block_ff_dim: Option<usize>,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub rope_parameters: RopeParameters,
    #[serde(default = "default_false")]
    pub conv_bias: bool,
    #[serde(default = "default_conv_l_cache", rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default = "default_block_multiple_of")]
    pub block_multiple_of: usize,
    #[serde(default = "default_block_ffn_dim_multiplier")]
    pub block_ffn_dim_multiplier: f64,
    #[serde(default = "default_true")]
    pub block_auto_adjust_ff_dim: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub tie_embedding: Option<bool>,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: usize,
    #[serde(default = "default_num_dense_layers")]
    pub num_dense_layers: usize,
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_true")]
    pub use_expert_bias: bool,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    pub quantization_config: Option<QuantizedConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Attention,
    Conv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardType {
    Dense,
    Moe,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn intermediate_size(&self) -> usize {
        if self.is_moe() {
            return self.intermediate_size;
        }
        let mut intermediate_size = self.block_ff_dim.unwrap_or(self.intermediate_size);
        if self.block_auto_adjust_ff_dim {
            intermediate_size = 2 * intermediate_size / 3;
            intermediate_size = (self.block_ffn_dim_multiplier * intermediate_size as f64) as usize;
            intermediate_size =
                self.block_multiple_of * intermediate_size.div_ceil(self.block_multiple_of);
        }
        intermediate_size
    }

    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_embedding.unwrap_or(self.tie_word_embeddings)
    }

    pub fn layer_types(&self) -> Vec<LayerType> {
        if self.layer_types.is_empty() {
            return vec![LayerType::Attention; self.num_hidden_layers];
        }
        self.layer_types
            .iter()
            .map(|layer_type| match layer_type.as_str() {
                "full_attention" => LayerType::Attention,
                "conv" => LayerType::Conv,
                _ => LayerType::Attention,
            })
            .collect()
    }

    pub fn is_moe(&self) -> bool {
        self.model_type.as_deref() == Some("lfm2_moe")
            || self
                .architectures
                .iter()
                .any(|architecture| architecture == "Lfm2MoeForCausalLM")
    }

    pub fn feed_forward_type(&self, layer_idx: usize) -> FeedForwardType {
        if self.is_moe() && layer_idx >= self.num_dense_layers {
            FeedForwardType::Moe
        } else {
            FeedForwardType::Dense
        }
    }
}

struct Mlp {
    w1: Arc<dyn QuantMethod>,
    w2: Arc<dyn QuantMethod>,
    w3: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        loading_isq: bool,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
    ) -> Result<Self> {
        let intermediate_size = cfg.intermediate_size();
        let vb = mapper.set_device(layer_idx, vb, loading_isq);
        let w1 = mistralrs_quant::linear_no_bias(
            cfg.hidden_size,
            intermediate_size,
            &cfg.quantization_config,
            vb.pp("w1"),
        )?;
        let w2 = mistralrs_quant::linear_no_bias(
            intermediate_size,
            cfg.hidden_size,
            &cfg.quantization_config,
            vb.pp("w2"),
        )?;
        let w3 = mistralrs_quant::linear_no_bias(
            cfg.hidden_size,
            intermediate_size,
            &cfg.quantization_config,
            vb.pp("w3"),
        )?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

struct MoeMlp {
    gate: Linear,
    gate_lora: Option<Arc<mistralrs_quant::LoraSiteHandle>>,
    experts: MoEExperts,
    expert_bias: Option<Tensor>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
}

impl MoeMlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or_else(|| vb.device().clone());
        let gate_vb = vb.pp("gate").set_device(layer_device.clone());
        let gate = layers::linear_no_bias(cfg.hidden_size, cfg.num_experts, gate_vb.clone())?;
        let gate_lora = mistralrs_quant::register_dynamic_lora_site(
            &gate_vb,
            mistralrs_quant::LoraLinearSpec::replicated(cfg.hidden_size, cfg.num_experts),
        )?;
        let expert_bias = if cfg.use_expert_bias {
            Some(
                mapper
                    .set_device(layer_idx, vb.clone(), false)
                    .get((cfg.num_experts,), "expert_bias")?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };
        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
            expert_proj_names: crate::moe::ExpertProjNames::MIXTRAL,
        };
        let experts = MoEExperts::new(
            &moe_cfg,
            vb,
            layer_device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            Activation::Silu,
        )?;

        Ok(Self {
            gate,
            gate_lora,
            experts,
            expert_bias,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor as f32,
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
        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.num_experts_per_tok,
                score_function: crate::ops::MoeRouterScoreFunction::Sigmoid,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: false,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            self.expert_bias.as_ref(),
            None,
        )?;
        let mut routing_weights = topk.values;
        if self.norm_topk_prob {
            let denominator = (routing_weights.sum_keepdim(D::Minus1)? + 1e-6)?;
            routing_weights = routing_weights.broadcast_div(&denominator)?;
        }
        if self.routed_scaling_factor != 1.0 {
            routing_weights = (routing_weights * self.routed_scaling_factor as f64)?;
        }

        let ys = self.experts.forward(xs, routing_weights, &topk.indices)?;
        ys.reshape((b_size, seq_len, hidden_dim))
    }

    fn gate(&self) -> &Linear {
        &self.gate
    }
}

enum FeedForward {
    Dense(Mlp),
    Moe(MoeMlp),
}

impl FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    out_proj: Arc<dyn QuantMethod>,
    q_layernorm: RmsNorm,
    k_layernorm: RmsNorm,
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
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rotary_emb: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let vb_attn = mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq);
        let head_dim = cfg.head_dim();
        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            vb_attn.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(cfg.num_key_value_heads, head_dim, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_attn.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_attn.pp("v_proj"),
        )?;
        let out_proj = RowParallelLayer::new(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb_attn.pp("out_proj"),
        )?;
        let vb_norms = mapper.set_device(layer_idx, vb.pp("self_attn"), false);
        let q_layernorm = RmsNorm::new(head_dim, cfg.norm_eps, vb_norms.pp("q_layernorm"))?;
        let k_layernorm = RmsNorm::new(head_dim, cfg.norm_eps, vb_norms.pp("k_layernorm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_layernorm,
            k_layernorm,
            num_heads: cfg.num_attention_heads / comm.world_size(),
            num_kv_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
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
        x: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let (mut q, mut k, mut v) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let rope_positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
        (q, k) = self.rotary_emb.forward_qk_norm(
            &q,
            &k,
            self.q_layernorm.weight(),
            self.k_layernorm.weight(),
            self.q_layernorm.eps(),
            self.k_layernorm.eps(),
            rope_positions,
        )?;

        let metadata = ctx.paged_layer(layer_idx);
        let y = match &self.paged_attn {
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

        let y = if !matches!(attention_mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        self.out_proj.forward(&y)
    }
}

struct ShortConv {
    conv: Conv1d,
    in_proj: Arc<dyn QuantMethod>,
    out_proj: Arc<dyn QuantMethod>,
    l_cache: usize,
}

impl ShortConv {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let vb_proj = mapper.set_device(layer_idx, vb.pp("conv"), loading_isq);
        let in_proj = mistralrs_quant::linear_b(
            cfg.hidden_size,
            3 * cfg.hidden_size,
            cfg.conv_bias,
            &cfg.quantization_config,
            vb_proj.pp("in_proj"),
        )?;
        let out_proj = mistralrs_quant::linear_b(
            cfg.hidden_size,
            cfg.hidden_size,
            cfg.conv_bias,
            &cfg.quantization_config,
            vb_proj.pp("out_proj"),
        )?;
        let conv_cfg = Conv1dConfig {
            padding: cfg.conv_l_cache - 1,
            groups: cfg.hidden_size,
            ..Default::default()
        };
        let vb_conv = mapper.set_device(layer_idx, vb.pp("conv").pp("conv"), false);
        let conv = if cfg.conv_bias {
            layers::conv1d(
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.conv_l_cache,
                conv_cfg,
                vb_conv,
            )?
        } else {
            layers::conv1d_no_bias(
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.conv_l_cache,
                conv_cfg,
                vb_conv,
            )?
        };
        Ok(Self {
            conv,
            in_proj,
            out_proj,
            l_cache: cfg.conv_l_cache,
        })
    }

    fn prefill_state(&self, bx: &Tensor) -> Result<Tensor> {
        let (batch, hidden, seq_len) = bx.dims3()?;
        if seq_len >= self.l_cache {
            bx.narrow(2, seq_len - self.l_cache, self.l_cache)
        } else {
            let pad = Tensor::zeros(
                (batch, hidden, self.l_cache - seq_len),
                bx.dtype(),
                bx.device(),
            )?;
            Tensor::cat(&[pad, bx.clone()], 2)
        }
    }

    fn prefill_state_from_existing(&self, bx: &Tensor, conv_state: &Tensor) -> Result<Tensor> {
        let seq_len = bx.dim(2)?;
        if seq_len >= self.l_cache {
            bx.narrow(2, seq_len - self.l_cache, self.l_cache)
        } else {
            let kept = conv_state.narrow(2, seq_len, self.l_cache - seq_len)?;
            Tensor::cat(&[kept, bx.clone()], 2)
        }
    }

    fn cached_prefill_conv(&self, bx: &Tensor, conv_state: &mut Tensor) -> Result<Tensor> {
        let seq_len = bx.dim(2)?;
        let combined = Tensor::cat(&[conv_state.clone(), bx.clone()], 2)?;
        let next_state = self.prefill_state_from_existing(bx, conv_state)?;
        *conv_state = next_state;
        Convolution
            .forward_1d(&self.conv, &combined)?
            .narrow(2, self.l_cache, seq_len)
    }

    fn decode_conv(&self, bx: &Tensor, conv_state: &mut Tensor) -> Result<Tensor> {
        let new_col = bx.narrow(2, bx.dim(2)? - 1, 1)?;
        *conv_state = if self.l_cache > 1 {
            let kept = conv_state.narrow(2, 1, self.l_cache - 1)?;
            Tensor::cat(&[kept, new_col], 2)?
        } else {
            new_col
        };
        let weight = self.conv.weight().squeeze(1)?.unsqueeze(0)?;
        let mut conv_out = (conv_state.clone() * weight)?.sum(D::Minus1)?;
        if let Some(bias) = self.conv.bias() {
            conv_out = conv_out.broadcast_add(&bias.reshape((1, bias.dim(0)?))?)?;
        }
        conv_out.unsqueeze(2)
    }

    fn forward(
        &self,
        x: &Tensor,
        conv_state: &mut Tensor,
        batch_kind: RecurrentBatchKind,
        use_existing_state: bool,
    ) -> Result<Tensor> {
        let (_, seq_len, hidden) = x.dims3()?;
        let projected = self.in_proj.forward(x)?.transpose(1, 2)?;
        let b_proj = projected.narrow(1, 0, hidden)?;
        let c_proj = projected.narrow(1, hidden, hidden)?;
        let x_proj = projected.narrow(1, 2 * hidden, hidden)?;
        let bx = (b_proj * x_proj)?;

        let conv_out = if matches!(batch_kind, RecurrentBatchKind::Decode) && seq_len == 1 {
            self.decode_conv(&bx, conv_state)?
        } else if use_existing_state {
            self.cached_prefill_conv(&bx, conv_state)?
        } else {
            *conv_state = self.prefill_state(&bx)?;
            Convolution
                .forward_1d(&self.conv, &bx)?
                .narrow(2, 0, seq_len)?
        };

        let y = (c_proj * conv_out)?.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&y)
    }
}

enum LayerImpl {
    Attention(Attention),
    Conv(ShortConv),
}

struct DecoderLayer {
    layer_impl: LayerImpl,
    feed_forward: FeedForward,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &Config,
        layer_type: LayerType,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let layer_impl = match layer_type {
            LayerType::Attention => LayerImpl::Attention(Attention::new(
                cfg,
                vb.clone(),
                mapper,
                layer_idx,
                loading_isq,
                rotary_emb.expect("attention layers require RoPE"),
                paged_attn,
                comm,
            )?),
            LayerType::Conv => LayerImpl::Conv(ShortConv::new(
                cfg,
                vb.clone(),
                mapper,
                layer_idx,
                loading_isq,
            )?),
        };
        let feed_forward = match cfg.feed_forward_type(layer_idx) {
            FeedForwardType::Dense => FeedForward::Dense(Mlp::new(
                cfg,
                vb.pp("feed_forward"),
                loading_isq,
                mapper,
                layer_idx,
            )?),
            FeedForwardType::Moe => FeedForward::Moe(MoeMlp::new(
                cfg,
                vb.pp("feed_forward"),
                mapper,
                layer_idx,
                loading_isq,
                comm,
            )?),
        };
        let operator_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.norm_eps,
            mapper.set_device(layer_idx, vb.pp("operator_norm"), false),
        )?;
        let ffn_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.norm_eps,
            mapper.set_device(layer_idx, vb.pp("ffn_norm"), false),
        )?;
        Ok(Self {
            layer_impl,
            feed_forward,
            operator_norm,
            ffn_norm,
        })
    }

    fn forward_attention(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let LayerImpl::Attention(attn) = &self.layer_impl else {
            candle_core::bail!("expected attention layer")
        };
        let residual = x;
        let attn_out = attn.forward(
            &self.operator_norm.forward(x)?,
            attention_mask,
            kv_cache,
            ctx,
            layer_idx,
        )?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let ffn_out = self.feed_forward.forward(&self.ffn_norm.forward(&x)?)?;
        ffn_out + residual
    }

    fn forward_conv(
        &self,
        x: &Tensor,
        conv_state: &mut Tensor,
        batch_kind: RecurrentBatchKind,
        use_existing_state: bool,
    ) -> Result<Tensor> {
        let LayerImpl::Conv(conv) = &self.layer_impl else {
            candle_core::bail!("expected conv layer")
        };
        let residual = x;
        let conv_out = conv.forward(
            &self.operator_norm.forward(x)?,
            conv_state,
            batch_kind,
            use_existing_state,
        )?;
        let x = (conv_out + residual)?;
        let residual = &x;
        let ffn_out = self.feed_forward.forward(&self.ffn_norm.forward(&x)?)?;
        ffn_out + residual
    }
}

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<LayerType>,
    embedding_norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    cache: EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    max_seq_len: usize,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        Self::new_inner(
            cfg,
            vb.pp("model"),
            vb.pp("lm_head"),
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
        if cfg.rope_parameters.rope_type != "default" && !cfg.rope_parameters.rope_type.is_empty() {
            candle_core::bail!(
                "LFM2 rope type `{}` is not supported",
                cfg.rope_parameters.rope_type
            );
        }

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
        let embedding_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.norm_eps,
            mapper.set_nm_device(vb_m.pp("embedding_norm"), false),
        )?;
        let lm_head = if cfg.tie_word_embeddings() {
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
        } else {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        };

        let head_dim = cfg.head_dim();
        let layer_types = cfg.layer_types();
        let mut ropes = HashMap::new();
        for (layer_idx, layer_type) in layer_types.iter().enumerate() {
            if !matches!(layer_type, LayerType::Attention) {
                continue;
            }
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            if let std::collections::hash_map::Entry::Vacant(entry) = ropes.entry(device.location())
            {
                entry.insert(Arc::new(RotaryEmbedding::new(
                    cfg.rope_parameters.rope_theta as f32,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?));
            }
        }

        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let comm = mapper.get_comm_for(layer_idx)?;
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes.get(&device.location()).cloned();
            let paged_attn = match (&attention_mechanism, layer_types[layer_idx]) {
                (AttentionImplementation::PagedAttention, LayerType::Attention) => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
                _ => None,
            };
            DecoderLayer::new(
                cfg,
                layer_types[layer_idx],
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                rotary_emb,
                paged_attn,
                &comm,
            )
        })?;

        let hybrid_layer_types = layer_types
            .iter()
            .map(|layer_type| match layer_type {
                LayerType::Attention => HybridLayerType::Attention,
                LayerType::Conv => HybridLayerType::Recurrent,
            })
            .collect();
        let cache = Arc::new(Mutex::new(HybridCache::new(
            HybridCacheConfig {
                layer_types: hybrid_layer_types,
                max_seq_len: cfg.max_position_embeddings,
                recurrent: RecurrentLayerConfig {
                    conv_dim: cfg.hidden_size,
                    conv_width: cfg.conv_l_cache,
                    state_dims: Vec::new(),
                    recurrent_dtype: None,
                },
            },
            vb_m.dtype(),
            &normal_loading_metadata.real_device,
        )?));

        Ok(Self {
            embed_tokens,
            layers,
            layer_types,
            embedding_norm,
            lm_head,
            cache: EitherCache::Hybrid(cache),
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: None,
                k_head_dim: head_dim,
                v_head_dim: head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            max_seq_len: cfg.max_position_embeddings,
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward_embeds(input_ids, self.embed(input_ids)?, ctx)
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        input_embeds: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut x = input_embeds;
        let mut hybrid_cache = self.cache.hybrid();
        let recurrent_metadata = ctx.recurrent_metadata().cloned();
        if self
            .layer_types
            .iter()
            .any(|layer_type| matches!(layer_type, LayerType::Conv))
            && recurrent_metadata.is_none()
        {
            candle_core::bail!("Hybrid recurrent metadata is required for LFM2 conv layers");
        }

        let mask = if ctx.is_paged() {
            CausalMasker.make_causal_mask(
                input_ids,
                &ForwardMaskCache::Paged(ctx.seqlen_offsets()),
                x.dtype(),
                &CausalMaskConfig::default(),
            )?
        } else {
            CausalMasker.make_causal_mask(
                input_ids,
                &*hybrid_cache as &dyn PastKvLenCache,
                x.dtype(),
                &CausalMaskConfig::default(),
            )?
        };
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = self.mapper.map(x, layer_idx)?;
            match &layer.layer_impl {
                LayerImpl::Attention(_) => {
                    let Some(HybridLayerCache::Attention(kv_cache)) =
                        hybrid_cache.get_mut(layer_idx)
                    else {
                        candle_core::bail!(
                            "Hybrid cache layer {layer_idx} is not attention for LFM2"
                        );
                    };
                    x = layer.forward_attention(
                        &x,
                        &mask.get(x.device()),
                        kv_cache,
                        ctx,
                        layer_idx,
                    )?;
                }
                LayerImpl::Conv(_) => {
                    let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer_idx)
                    else {
                        candle_core::bail!(
                            "Hybrid cache layer {layer_idx} is not recurrent for LFM2"
                        );
                    };
                    let recurrent_metadata = recurrent_metadata
                        .as_ref()
                        .expect("checked above: LFM2 conv layers require recurrent metadata");
                    let indices = recurrent_metadata.state_indices();
                    let mut conv_state = pool.gather_conv_state(indices)?;
                    let use_existing_state = recurrent_metadata.batch_kind()
                        == RecurrentBatchKind::Decode
                        || !ctx.is_first_prompt_chunk();
                    x = layer.forward_conv(
                        &x,
                        &mut conv_state,
                        recurrent_metadata.batch_kind(),
                        use_existing_state,
                    )?;
                    pool.scatter_conv_state_with_host_indices(
                        indices,
                        recurrent_metadata.state_indices_host(),
                        &conv_state,
                    )?;
                }
            }
        }

        let x = x.to_device(&self.device)?;
        let x = self.embedding_norm.forward(&x)?;
        let x = ctx.logits(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("embedding_norm").add(&self.embedding_norm);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("operator_norm").add(&layer.operator_norm);
            uvb_l.pp("ffn_norm").add(&layer.ffn_norm);
            match &layer.layer_impl {
                LayerImpl::Attention(attn) => {
                    uvb_l
                        .pp("self_attn")
                        .pp("q_layernorm")
                        .add(&attn.q_layernorm);
                    uvb_l
                        .pp("self_attn")
                        .pp("k_layernorm")
                        .add(&attn.k_layernorm);
                }
                LayerImpl::Conv(conv) => {
                    uvb_l.pp("conv").pp("conv").add(&conv.conv);
                }
            }
            if let FeedForward::Moe(moe) = &layer.feed_forward {
                uvb_l.pp("feed_forward").pp("gate").add(moe.gate());
                if let Some(expert_bias) = &moe.expert_bias {
                    uvb_l
                        .pp("feed_forward")
                        .add_tensor("expert_bias", expert_bias.clone());
                }
            }
        }
        uvb_m.to_safetensors()
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.residual_tensors_m(UnVarBuilder::new().pp("model"))
    }

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");

        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("embedding_norm").add(&self.embedding_norm);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("operator_norm").add(&layer.operator_norm);
            uvb_l.pp("ffn_norm").add(&layer.ffn_norm);

            match &layer.layer_impl {
                LayerImpl::Attention(attn) => {
                    let uvb_attn = uvb_l.pp("self_attn");
                    uvb_attn.pp("q_proj").add(&attn.q_proj);
                    uvb_attn.pp("k_proj").add(&attn.k_proj);
                    uvb_attn.pp("v_proj").add(&attn.v_proj);
                    uvb_attn.pp("out_proj").add(&attn.out_proj);
                    uvb_attn.pp("q_layernorm").add(&attn.q_layernorm);
                    uvb_attn.pp("k_layernorm").add(&attn.k_layernorm);
                }
                LayerImpl::Conv(conv) => {
                    let uvb_conv = uvb_l.pp("conv");
                    uvb_conv.pp("in_proj").add(&conv.in_proj);
                    uvb_conv.pp("out_proj").add(&conv.out_proj);
                    uvb_conv.pp("conv").add(&conv.conv);
                }
            }

            let uvb_ffn = uvb_l.pp("feed_forward");
            match &layer.feed_forward {
                FeedForward::Dense(mlp) => {
                    uvb_ffn.pp("w1").add(&mlp.w1);
                    uvb_ffn.pp("w2").add(&mlp.w2);
                    uvb_ffn.pp("w3").add(&mlp.w3);
                }
                FeedForward::Moe(moe) => {
                    uvb_ffn.pp("gate").add(moe.gate());
                    if let Some(expert_bias) = &moe.expert_bias {
                        uvb_ffn.add_tensor("expert_bias", expert_bias.clone());
                    }
                }
            }
        }

        Some(uvb.to_safetensors())
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl NormalModel for Model {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
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
        candle_core::bail!("LFM2 does not support X-LoRA forward")
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
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Model {}
