#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::Deserialize;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        embedding, Activation, CausalMasker, DeepSeekV2RopeConfig, DeepSeekV2RotaryEmbedding, Mlp,
        RmsNorm, Sdpa,
    },
    layers_masker::PastKvLenCache,
    mla::{
        mla_cache_forward, mla_decode_forward, should_use_mla_cache, should_use_mla_decode,
        MlaWeights,
    },
    moe::{MoEExperts, MoEExpertsConfig},
    ops::{SplitOp, TopKLastDimOp},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(usize, n_group, 1);
serde_default_fn!(usize, topk_group, 1);

#[derive(Deserialize, Clone, Debug)]
pub struct Glm4MoeLiteConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    #[allow(dead_code)]
    pub(crate) num_key_value_heads: usize,
    pub(crate) q_lora_rank: usize,
    pub(crate) kv_lora_rank: usize,
    pub(crate) qk_nope_head_dim: usize,
    pub(crate) qk_rope_head_dim: usize,
    pub(crate) v_head_dim: usize,
    pub(crate) n_routed_experts: usize,
    pub(crate) n_shared_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    #[serde(default = "first_k_dense_replace")]
    pub(crate) first_k_dense_replace: usize,
    #[serde(default = "routed_scaling_factor")]
    pub(crate) routed_scaling_factor: f64,
    #[serde(default = "n_group")]
    pub(crate) n_group: usize,
    #[serde(default = "topk_group")]
    pub(crate) topk_group: usize,
    #[serde(default = "moe_layer_freq")]
    pub(crate) moe_layer_freq: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f32,
    pub(crate) max_position_embeddings: usize,
    #[serde(default = "hidden_act")]
    pub(crate) hidden_act: Activation,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    #[serde(alias = "quantization")]
    pub(crate) quantization_config: Option<QuantizedConfig>,
}

impl Glm4MoeLiteConfig {
    pub(crate) fn q_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }

    fn softmax_scale(&self) -> f32 {
        1.0 / (self.q_head_dim() as f32).sqrt()
    }
}

enum QProj {
    Lora {
        a: Arc<dyn QuantMethod>,
        norm: RmsNorm,
        b: Arc<dyn QuantMethod>,
    },
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Lora { a, norm, b } => {
                b.forward_autocast(&norm.forward(&a.forward_autocast(xs)?)?)
            }
        }
    }
}

struct Attention {
    q: QProj,
    kv_a_proj_with_mqa: Arc<dyn QuantMethod>,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    cfg: Glm4MoeLiteConfig,
    q_head_dim: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    num_attention_heads: usize,
    mla_weights: MlaWeights,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &Glm4MoeLiteConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let q_head_dim = cfg.q_head_dim();

        // GLM4MoeLite always uses LoRA for Q projection
        let q = {
            let a = ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.q_lora_rank,
                &cfg.quantization_config,
                false,
                mapper.set_device(layer_idx, vb.pp("q_a_proj"), loading_isq),
            )?;
            let norm = RmsNorm::new(
                cfg.q_lora_rank,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("q_a_layernorm"), false),
            )?;
            let b = ColumnParallelLayer::new(
                cfg.q_lora_rank,
                cfg.num_attention_heads * q_head_dim,
                &cfg.quantization_config,
                false,
                comm,
                mapper.set_device(layer_idx, vb.pp("q_b_proj"), loading_isq),
            )?;
            QProj::Lora { a, norm, b }
        };

        let kv_a_proj_with_mqa = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            &cfg.quantization_config,
            false,
            mapper.set_device(layer_idx, vb.pp("kv_a_proj_with_mqa"), loading_isq),
        )?;
        let kv_a_layernorm = RmsNorm::new(
            cfg.kv_lora_rank,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("kv_a_layernorm"), false),
        )?;
        let kv_b_proj = ColumnParallelLayer::new(
            cfg.kv_lora_rank,
            cfg.num_attention_heads * (q_head_dim - cfg.qk_rope_head_dim + cfg.v_head_dim),
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("kv_b_proj"), loading_isq),
        )?;

        let o_proj = RowParallelLayer::new(
            cfg.num_attention_heads * cfg.v_head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;

        let mla_weights = MlaWeights::new(
            paged_attn.is_some(),
            mapper.device_for(layer_idx, loading_isq),
        );

        Ok(Self {
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            cfg: cfg.clone(),
            q_head_dim,
            paged_attn,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            sdpa_params: SdpaParams {
                n_kv_groups: 1,
                softcap: None,
                softmax_scale: cfg.softmax_scale(),
                sliding_window: None,
                sinks: None,
            },
            mla_weights,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;

        let mut q = self.q.forward(xs)?;
        q = q
            .reshape((bs, seq_len, self.num_attention_heads, self.q_head_dim))?
            .transpose(1, 2)?;
        let q_split = q.split(
            &[self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        let q_nope = q_split[0].clone();
        let mut q_pe = q_split[1].clone();

        let mut compressed_kv = self.kv_a_proj_with_mqa.forward_autocast(xs)?;
        let ckv_split = compressed_kv.split(
            &[self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        compressed_kv = ckv_split[0].clone();
        let mut k_pe = ckv_split[1].clone();
        k_pe = k_pe
            .reshape((bs, seq_len, 1, self.cfg.qk_rope_head_dim))?
            .transpose(1, 2)?;

        let ckv = self.kv_a_layernorm.forward(&compressed_kv)?;

        (q_pe, k_pe) = self.rotary_emb.forward(&q_pe, &k_pe, seqlen_offsets)?;

        let use_mla_decode = should_use_mla_decode(
            attention_mask,
            seq_len,
            self.paged_attn.is_some(),
            q_nope.device(),
            &metadata,
        );

        let mut attn_out = if use_mla_decode {
            mla_decode_forward(
                &q_nope,
                &q_pe,
                &ckv,
                &k_pe,
                &metadata,
                &self.mla_weights,
                self.kv_b_proj.as_ref(),
                &self.sdpa_params,
                self.num_attention_heads,
                self.cfg.kv_lora_rank,
                self.cfg.qk_rope_head_dim,
                self.cfg.qk_nope_head_dim,
                self.cfg.v_head_dim,
                bs,
                seq_len,
            )?
        } else {
            let mut kv = self.kv_b_proj.forward_autocast(&ckv)?;
            kv = kv
                .reshape((
                    bs,
                    seq_len,
                    self.num_attention_heads,
                    self.cfg.qk_nope_head_dim + self.cfg.v_head_dim,
                ))?
                .transpose(1, 2)?;

            let kv_split =
                kv.split(&[self.cfg.qk_nope_head_dim, self.cfg.v_head_dim], D::Minus1)?;
            let k_nope = kv_split[0].clone();
            let mut v = kv_split[1].clone();

            let q = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?.contiguous()?;
            let mut k = Tensor::cat(
                &[&k_nope, &k_pe.repeat((1, self.num_attention_heads, 1, 1))?],
                D::Minus1,
            )?
            .contiguous()?;

            let use_mla_cache = should_use_mla_cache(self.paged_attn.is_some(), q.device());

            if use_mla_cache {
                mla_cache_forward(
                    &q,
                    &k,
                    &v,
                    &ckv,
                    &k_pe,
                    attention_mask,
                    seqlen_offsets,
                    &metadata,
                    flash_params,
                    self.kv_b_proj.as_ref(),
                    &self.sdpa_params,
                    self.num_attention_heads,
                    self.cfg.kv_lora_rank,
                    self.cfg.qk_rope_head_dim,
                    self.cfg.qk_nope_head_dim,
                    self.cfg.v_head_dim,
                    bs,
                    seq_len,
                )?
            } else {
                match &self.paged_attn {
                    Some(paged_attn) => match metadata {
                        Some(((key_cache, value_cache), input_metadata)) => {
                            let v = v
                                .pad_with_zeros(
                                    D::Minus1,
                                    0,
                                    self.q_head_dim - self.cfg.v_head_dim,
                                )?
                                .contiguous()?;
                            paged_attn
                                .forward(
                                    &q,
                                    &k,
                                    &v,
                                    attention_mask,
                                    Some(key_cache),
                                    Some(value_cache),
                                    input_metadata,
                                    &self.sdpa_params,
                                    Some(flash_params),
                                )?
                                .narrow(D::Minus1, 0, self.cfg.v_head_dim)?
                        }
                        None => {
                            // If we don't have metadata, we are most likely generating an imatrix so we don't want to populate that.
                            // Generating the dummy metadata with the assumption that we are not generating text (only processing prompts).
                            let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                            // Sanity check.
                            assert!(attention_mask.is_some());
                            let v = v
                                .pad_with_zeros(
                                    D::Minus1,
                                    0,
                                    self.q_head_dim - self.cfg.v_head_dim,
                                )?
                                .contiguous()?;
                            paged_attn
                                .forward(
                                    &q,
                                    &k,
                                    &v,
                                    attention_mask,
                                    None,
                                    None,
                                    &input_metadata,
                                    &self.sdpa_params,
                                    Some(flash_params),
                                )?
                                .narrow(D::Minus1, 0, self.cfg.v_head_dim)?
                        }
                    },
                    None => {
                        (k, v) = kv_cache.append(&k, &v)?;

                        Sdpa.run_attention(
                            &q,
                            &k,
                            &v,
                            attention_mask,
                            Some(flash_params),
                            &self.sdpa_params,
                        )?
                    }
                }
            }
        };

        attn_out = if attention_mask.is_some() {
            attn_out.transpose(1, 2)?.reshape((bs, seq_len, ()))?
        } else {
            attn_out.reshape((bs, seq_len, ()))?
        };

        self.o_proj.forward_autocast(&attn_out)
    }
}

struct Expert {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
    act: Activation,
}

impl Expert {
    fn new(
        cfg: &Glm4MoeLiteConfig,
        vb: ShardedVarBuilder,
        hidden_size: Option<usize>,
        intermediate_size: Option<usize>,
    ) -> Result<Self> {
        let hidden_size = hidden_size.unwrap_or(cfg.hidden_size);
        let intermediate_size = intermediate_size.unwrap_or(cfg.intermediate_size);

        Ok(Self {
            gate: ReplicatedLayer::new(
                hidden_size,
                intermediate_size,
                &cfg.quantization_config,
                false,
                vb.pp("gate_proj"),
            )?,
            up: ReplicatedLayer::new(
                hidden_size,
                intermediate_size,
                &cfg.quantization_config,
                false,
                vb.pp("up_proj"),
            )?,
            down: ReplicatedLayer::new(
                intermediate_size,
                hidden_size,
                &cfg.quantization_config,
                false,
                vb.pp("down_proj"),
            )?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let lhs = self.gate.forward(&xs)?;
        let rhs = self.up.forward(&xs)?;
        let mut res = self
            .down
            .forward(&crate::ops::mul_and_act(&lhs, &rhs, self.act)?)?;
        if self.gate.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct MoeGate {
    weight: Tensor,
    cfg: Glm4MoeLiteConfig,
    top_k: usize,
    n_routed_experts: usize,
    e_score_correction_bias: Tensor,
}

impl MoeGate {
    fn new(
        cfg: &Glm4MoeLiteConfig,
        vb: ShardedVarBuilder,
        n_routed_experts: usize,
    ) -> Result<Self> {
        let weight = vb.get((n_routed_experts, cfg.hidden_size), "weight")?;
        // GLM4MoeLite uses NoAuxTc routing with e_score_correction_bias
        let e_score_correction_bias = vb.get_with_hints_dtype(
            n_routed_experts,
            "e_score_correction_bias",
            Default::default(),
            DType::F32,
        )?;
        Ok(Self {
            weight,
            cfg: cfg.clone(),
            top_k: cfg.num_experts_per_tok,
            n_routed_experts,
            e_score_correction_bias,
        })
    }

    /// (topk_idx, topk_weight)
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bs, seq_len, h) = xs.dims3()?;
        // Compute gating score
        let xs = xs.reshape(((), h))?;
        let logits = xs
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.weight.t()?.to_dtype(DType::F32)?)?;
        // GLM4MoeLite uses sigmoid scoring
        let scores = candle_nn::ops::sigmoid(&logits)?;

        // NoAuxTc routing with e_score_correction_bias
        let scores_for_choice = scores
            .reshape((bs * seq_len, ()))?
            .broadcast_add(&self.e_score_correction_bias.unsqueeze(0)?)?;
        // (n, n_group)
        let group_scores = scores_for_choice
            .reshape((bs * seq_len, self.cfg.n_group, ()))?
            .topk(2)?
            .values
            .sum(D::Minus1)?;
        // (n, topk_group)
        let group_idx = group_scores.topk(self.cfg.topk_group)?.indices;
        // (n, n_group)
        let mut group_mask = group_scores.zeros_like()?;
        // (n, n_group)
        group_mask = group_mask.scatter_add(
            &group_idx,
            &group_idx.ones_like()?.to_dtype(group_mask.dtype())?,
            1,
        )?;
        // (n, e)
        let score_mask = group_mask
            .unsqueeze(D::Minus1)?
            .expand((
                bs * seq_len,
                self.cfg.n_group,
                self.n_routed_experts / self.cfg.n_group,
            ))?
            .reshape((bs * seq_len, ()))?;
        // (n, e)
        // Invert the mask
        let tmp_scores = scores_for_choice.broadcast_mul(&score_mask)?;
        let topk_idx = tmp_scores.topk(self.top_k)?.indices;
        let mut topk_weight = scores.gather(&topk_idx, 1)?;

        // Normalize with sigmoid
        let denominator = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
        topk_weight = topk_weight.broadcast_div(&denominator)?;

        // Must multiply the scaling factor
        topk_weight = (topk_weight * self.cfg.routed_scaling_factor)?;

        Ok((topk_idx, topk_weight))
    }
}

struct Moe {
    experts: MoEExperts,
    shared_experts: Option<Expert>,
    gate: MoeGate,
}

impl Moe {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &Glm4MoeLiteConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        n_shared_experts: usize,
        n_routed_experts: usize,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(real_device);

        let moe_cfg = MoEExpertsConfig {
            num_experts: n_routed_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
        };

        // Use the optimized MoEExperts with automatic backend selection
        let experts = MoEExperts::new(
            &moe_cfg,
            mapper.set_device(layer_idx, vb.clone(), loading_isq),
            layer_device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        // Shared experts are handled separately
        let shared_experts = if n_shared_experts > 0 {
            Some(Expert::new(
                cfg,
                mapper.set_device(layer_idx, vb.pp("shared_experts"), loading_isq),
                None,
                Some(cfg.moe_intermediate_size),
            )?)
        } else {
            None
        };

        let gate = MoeGate::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("gate"), false),
            n_routed_experts,
        )?;

        Ok(Self {
            experts,
            shared_experts,
            gate,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let identity = xs.clone();
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;

        // Get routing weights from custom gate (NoAuxTc with e_score_correction_bias)
        let (topk_idx, topk_weight) = self.gate.forward(xs)?;

        // Forward through routed experts using optimized MoEExperts
        let mut y = self.experts.forward(xs, topk_weight, &topk_idx)?;
        y = y.reshape((b_size, seq_len, hidden_dim))?;

        // Add shared expert output
        if let Some(ref shared_experts) = self.shared_experts {
            y = (y + shared_experts.forward(&identity)?)?;
        }

        Ok(y)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = self.experts.get_isq_layers();
        if let Some(ref mut shared) = self.shared_experts {
            layers.push(&mut shared.gate);
            layers.push(&mut shared.up);
            layers.push(&mut shared.down);
        }
        layers
    }
}

enum MoeOrMlp {
    Moe(Box<Moe>),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn: Attention,
    moe_or_mlp: MoeOrMlp,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &Glm4MoeLiteConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let attn = Attention::new(
            rotary_emb,
            cfg,
            vb.pp("self_attn"),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            comm,
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
        // Layer 0 uses dense MLP (first_k_dense_replace=1 by default), other layers use MoE
        let moe_or_mlp = if layer_idx >= cfg.first_k_dense_replace
            && layer_idx.is_multiple_of(cfg.moe_layer_freq)
        {
            MoeOrMlp::Moe(Box::new(Moe::new(
                cfg,
                vb.pp("mlp"),
                mapper,
                layer_idx,
                loading_isq,
                cfg.n_shared_experts,
                cfg.n_routed_experts,
                comm,
                real_device,
            )?))
        } else {
            MoeOrMlp::Mlp(Mlp::new(
                mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                comm,
            )?)
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attn,
            moe_or_mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .moe_or_mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

pub struct Glm4MoeLite {
    lm_head: Arc<dyn QuantMethod>,
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    cache: EitherCache,
    device: Device,
    max_seq_len: usize,
    cfg: ModelConfigMetadata,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl Glm4MoeLite {
    pub fn new(
        cfg: &Glm4MoeLiteConfig,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");

        let mapper = normal_loading_metadata.mapper;

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(
                    embed_tokens.embeddings(),
                    normal_loading_metadata.loading_isq,
                )?,
                None,
            ))?
        };
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let mut ropes = HashMap::new();
        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: None,
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(DeepSeekV2RotaryEmbedding::new(
                    &rope_cfg,
                    vb.dtype(),
                    device,
                )?),
            );
        }

        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(
                    PagedAttention::new(cfg.v_head_dim, device, None)
                        .expect("Failed to create PagedAttention"),
                ),
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
                normal_loading_metadata.real_device.clone(),
            )
        })?;

        Ok(Self {
            lm_head,
            embed_tokens,
            norm,
            layers,
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            device: normal_loading_metadata.real_device.clone(),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: (cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: None,
                k_head_dim: cfg.q_head_dim(),
                v_head_dim: if matches!(
                    attention_mechanism,
                    AttentionImplementation::PagedAttention
                ) {
                    cfg.q_head_dim()
                } else {
                    cfg.v_head_dim
                },
                kv_cache_layout: if matches!(
                    attention_mechanism,
                    AttentionImplementation::PagedAttention
                ) && {
                    #[cfg(all(feature = "cuda", target_family = "unix"))]
                    {
                        matches!(normal_loading_metadata.real_device, Device::Cuda(_))
                    }
                    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
                    {
                        false
                    }
                } {
                    crate::paged_attention::KvCacheLayout::Mla {
                        kv_lora_rank: cfg.kv_lora_rank,
                        kpe_head_dim: cfg.qk_rope_head_dim,
                    }
                } else {
                    crate::paged_attention::KvCacheLayout::Standard
                },
            },
            mapper,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let cache = &mut self.cache.normal().0;
        let attention_mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        // PagedAttention prompt chunking
        let attention_mask = attention_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask.as_ref().map(|m| m.get(xs.device())),
                seqlen_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                flash_params,
            )?;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let xs = extract_logits(&xs, context_lens)?;
        self.lm_head.forward_autocast(&xs)
    }
}

impl IsqModel for Glm4MoeLite {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match &mut layer.attn.q {
                QProj::Lora { a, norm: _, b } => {
                    tensors.push((a, Some(i)));
                    tensors.push((b, Some(i)));
                }
            }
            tensors.push((&mut layer.attn.kv_a_proj_with_mqa, Some(i)));
            tensors.push((&mut layer.attn.kv_b_proj, Some(i)));
            tensors.push((&mut layer.attn.o_proj, Some(i)));
            match &mut layer.moe_or_mlp {
                MoeOrMlp::Mlp(mlp) => {
                    tensors.push((&mut mlp.gate, Some(i)));
                    tensors.push((&mut mlp.up, Some(i)));
                    tensors.push((&mut mlp.down, Some(i)));
                }
                MoeOrMlp::Moe(moe) => {
                    for layer in moe.get_isq_layers() {
                        tensors.push((layer, Some(i)));
                    }
                }
            }
        }
        (tensors, &*self.mapper)
    }

    fn get_layers_moe_experts_only(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match &mut layer.moe_or_mlp {
                MoeOrMlp::Mlp(mlp) => {
                    tensors.push((&mut mlp.gate, Some(i)));
                    tensors.push((&mut mlp.up, Some(i)));
                    tensors.push((&mut mlp.down, Some(i)));
                }
                MoeOrMlp::Moe(moe) => {
                    for layer in moe.get_isq_layers() {
                        tensors.push((layer, Some(i)));
                    }
                }
            }
        }
        (tensors, &*self.mapper)
    }

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
                .pp("kv_a_layernorm")
                .add(&layer.attn.kv_a_layernorm);

            match &layer.moe_or_mlp {
                MoeOrMlp::Moe(moe) => {
                    uvb_l
                        .pp("mlp")
                        .pp("gate")
                        .add_tensor("weight", moe.gate.weight.clone());
                    uvb_l.pp("mlp").pp("gate").add_tensor(
                        "e_score_correction_bias",
                        moe.gate.e_score_correction_bias.clone(),
                    );
                }
                MoeOrMlp::Mlp(_) => (),
            }

            match &layer.attn.q {
                QProj::Lora { a: _, norm, b: _ } => {
                    uvb_l.pp("self_attn").pp("q_a_layernorm").add(norm);
                }
            }
        }

        uvb.to_safetensors()
    }

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
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
                .pp("kv_a_layernorm")
                .add(&layer.attn.kv_a_layernorm);

            match &layer.moe_or_mlp {
                MoeOrMlp::Moe(moe) => {
                    uvb_l
                        .pp("mlp")
                        .pp("gate")
                        .add_tensor("weight", moe.gate.weight.clone());
                    uvb_l.pp("mlp").pp("gate").add_tensor(
                        "e_score_correction_bias",
                        moe.gate.e_score_correction_bias.clone(),
                    );
                }
                MoeOrMlp::Mlp(_) => (),
            }

            match &layer.attn.q {
                QProj::Lora { a, norm, b } => {
                    uvb_l.pp("self_attn").pp("q_a_proj").add(a);
                    uvb_l.pp("self_attn").pp("q_a_layernorm").add(norm);
                    uvb_l.pp("self_attn").pp("q_b_proj").add(b);
                }
            }
            uvb_l
                .pp("self_attn")
                .pp("kv_a_proj_with_mqa")
                .add(&layer.attn.kv_a_proj_with_mqa);
            uvb_l
                .pp("self_attn")
                .pp("kv_b_proj")
                .add(&layer.attn.kv_b_proj);
            uvb_l.pp("self_attn").pp("o_proj").add(&layer.attn.o_proj);
        }

        Some(uvb.to_safetensors())
    }
}

impl NormalModel for Glm4MoeLite {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
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
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
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
}

impl AnyMoeBaseModelMixin for Glm4MoeLite {}
