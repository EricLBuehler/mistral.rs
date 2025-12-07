#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GPT-OSS Model Implementation
//!
//! Key features:
//! - Attention with per-head "sinks" (attention bias that goes through softmax)
//! - Layer types: alternating "sliding_attention" and "full_attention"
//! - MoE with 32 experts, 4 active per token
//! - Special SwiGLU activation: (up + 1) * gate * sigmoid(gate * alpha), with clamping
//! - MXFP4 quantized expert weights with combined gate_up_proj format

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder, MXFP4Layer,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::DeviceMapper,
    layers::{self, embedding, CausalMasker, MatMul, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::progress::NiceProgressBar,
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

impl Default for LayerType {
    fn default() -> Self {
        LayerType::FullAttention
    }
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

/// YARN rotary embedding implementation
/// Based on HuggingFace transformers YARN implementation
pub struct YarnRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    #[allow(dead_code)]
    attention_scale: f32,
}

impl YarnRotaryEmbedding {
    pub fn new(
        base: f64,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_scaling: &RopeScaling,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let dim = head_dim;
        let factor = rope_scaling.factor;
        let beta_fast = rope_scaling.beta_fast;
        let beta_slow = rope_scaling.beta_slow;
        let original_max_pos = rope_scaling.original_max_position_embeddings;

        // Compute attention scale: 0.1 * ln(factor) + 1.0 for YARN
        let attention_scale = (0.1 * factor.ln() + 1.0) as f32;

        // Compute base inverse frequencies (extrapolation)
        let inv_freq_extrapolation: Vec<f64> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f64 / dim as f64))
            .collect();

        // Compute interpolation frequencies
        let inv_freq_interpolation: Vec<f64> = inv_freq_extrapolation
            .iter()
            .map(|f| f / factor)
            .collect();

        // Compute low and high frequency bounds for the ramp
        let low_freq_wavelen = original_max_pos as f64 / beta_slow;
        let high_freq_wavelen = original_max_pos as f64 / beta_fast;

        // Compute the linear ramp factor for each frequency
        let inv_freq: Vec<f64> = inv_freq_extrapolation
            .iter()
            .zip(inv_freq_interpolation.iter())
            .map(|(&freq_extra, &freq_inter)| {
                let wavelen = 2.0 * std::f64::consts::PI / freq_extra;

                if wavelen < high_freq_wavelen {
                    // High frequency: use extrapolation (original)
                    freq_extra
                } else if wavelen > low_freq_wavelen {
                    // Low frequency: use interpolation (scaled)
                    freq_inter
                } else {
                    // Mid frequency: blend between interpolation and extrapolation
                    let ramp = (wavelen - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen);
                    freq_inter * ramp + freq_extra * (1.0 - ramp)
                }
            })
            .collect();

        let inv_freq_len = inv_freq.len();
        let inv_freq_tensor = Tensor::from_vec(
            inv_freq.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            (1, inv_freq_len),
            device,
        )?;

        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;

        let freqs = t.matmul(&inv_freq_tensor)?;

        // Apply attention scale to sin/cos (matches HF transformers behavior)
        // When applied to both Q and K, the effect on scores is scale^2, but this
        // matches how the model was trained with HF transformers
        let sin = (freqs.sin()? * attention_scale as f64)?.to_dtype(dtype)?;
        let cos = (freqs.cos()? * attention_scale as f64)?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            attention_scale,
        })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, qh, seq_len, n_embd) = q.dims4()?;
        let (_b_sz, kh, _seq_len, _n_embd) = k.dims4()?;

        let rope = candle_nn::rotary_emb::rope;

        if cfg!(feature = "cuda") && qh == kh {
            let (cos, sin) = if seqlen_offsets.len() == 1 {
                (
                    self.cos.narrow(0, seqlen_offsets[0], seq_len)?,
                    self.sin.narrow(0, seqlen_offsets[0], seq_len)?,
                )
            } else {
                let mut cos_s = Vec::new();
                let mut sin_s = Vec::new();
                for offset in seqlen_offsets {
                    cos_s.push(self.cos.narrow(0, *offset, seq_len)?);
                    sin_s.push(self.sin.narrow(0, *offset, seq_len)?);
                }
                (Tensor::cat(&cos_s, 0)?, Tensor::cat(&sin_s, 0)?)
            };

            let q_embed = q.transpose(1, 2)?.flatten(0, 1)?;
            let k_embed = k.transpose(1, 2)?.flatten(0, 1)?;
            mistralrs_quant::rotary::apply_rotary_inplace(
                &q_embed,
                &k_embed,
                &cos,
                &sin,
                true, // is_gpt_neox style
            )?;
            let mut q = q_embed
                .reshape((b_sz, seq_len, qh, n_embd))?
                .transpose(1, 2)?;
            let mut k = k_embed
                .reshape((b_sz, seq_len, kh, n_embd))?
                .transpose(1, 2)?;
            if !(cfg!(feature = "flash-attn") || cfg!(feature = "flash-attn-v3")) {
                q = q.contiguous()?;
                k = k.contiguous()?;
            }
            Ok((q, k))
        } else if seqlen_offsets.len() == 1 {
            let cos = self.cos.narrow(0, seqlen_offsets[0], seq_len)?;
            let sin = self.sin.narrow(0, seqlen_offsets[0], seq_len)?;
            let q_embed = rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else {
            let mut q_embeds = Vec::new();
            let mut k_embeds = Vec::new();
            for (i, offset) in seqlen_offsets.iter().enumerate() {
                let cos = self.cos.narrow(0, *offset, seq_len)?;
                let sin = self.sin.narrow(0, *offset, seq_len)?;
                let q_embed = rope(&q.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                let k_embed = rope(&k.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                q_embeds.push(q_embed);
                k_embeds.push(k_embed);
            }
            Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
        }
    }
}

/// Wrapper enum for both standard and YARN rotary embeddings
#[derive(Clone)]
pub enum GptOssRotaryEmbedding {
    Standard(Arc<RotaryEmbedding>),
    Yarn(Arc<YarnRotaryEmbedding>),
}

impl GptOssRotaryEmbedding {
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Standard(rope) => rope.forward(q, k, seqlen_offsets),
            Self::Yarn(rope) => rope.forward(q, k, seqlen_offsets),
        }
    }
}

/// Custom SwiGLU activation with clamping as used in GPT-OSS
/// Formula: (up + 1) * gate * sigmoid(gate * alpha)
/// With clamping: gate clamped to max=limit (no min), up clamped to [-limit, limit]
fn gptoss_swiglu(gate: &Tensor, up: &Tensor, alpha: f32, limit: f32) -> Result<Tensor> {
    let dtype = gate.dtype();
    let limit_d = limit as f64;

    // Clamp gate to max=limit only (no min bound), up to [-limit, limit]
    // HF: gate.clamp(min=None, max=self.limit)
    let gate_clamped = gate.minimum(&Tensor::full(limit_d, gate.shape(), gate.device())?.to_dtype(dtype)?)?;
    let up_clamped = up.clamp(-limit_d, limit_d)?;

    // glu = gate * sigmoid(gate * alpha)
    let gate_scaled = (&gate_clamped * alpha as f64)?;
    let sigmoid_val = candle_nn::ops::sigmoid(&gate_scaled)?;
    let glu = (&gate_clamped * &sigmoid_val)?;

    // output = (up + 1) * glu
    let up_plus_one = (&up_clamped + 1.0)?;
    up_plus_one.mul(&glu)
}

/// Attention with per-head sinks
struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    sinks: Tensor, // [num_heads] - per-head attention sink values
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: GptOssRotaryEmbedding,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    #[allow(dead_code)]
    is_sliding: bool,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: GptOssRotaryEmbedding,
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

        // Attention projections are NOT quantized in MXFP4 models (in modules_to_not_convert)
        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &None, // Not quantized
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        );
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &None, // Not quantized
            cfg.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &None, // Not quantized
            cfg.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &None, // Not quantized
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;

        // Load sinks: [num_heads]
        let sinks = vb.get((num_heads,), "sinks")?;

        let is_sliding = matches!(
            cfg.layer_types.get(layer_idx),
            Some(LayerType::SlidingAttention)
        );
        let sliding_window = if is_sliding {
            cfg.sliding_window
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            sinks,
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
                ),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window,
            },
            is_sliding,
        })
    }

    /// GPT-OSS specific attention forward with sinks
    /// The sinks are added to attention logits before softmax, then dropped after
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&xs, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&xs, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&xs, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

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

        (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        // For now, use standard SDPA without sinks modification
        // TODO: Implement custom attention kernel with sinks support
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
                    Some(flash_params),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
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
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                // Custom attention with sinks for non-paged attention
                self.attention_with_sinks(&q, &k, &v, attention_mask, seqlen_offsets, flash_params)?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&attn_output, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    /// Attention computation with sinks
    /// Sinks are added as extra logits, go through softmax, then are dropped
    fn attention_with_sinks(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        // Manual attention with sinks
        let (b_sz, num_heads, q_len, _head_dim) = q.dims4()?;
        let (_, _, k_len, _) = k.dims4()?;

        // Repeat KV for grouped query attention
        let k = if self.sdpa_params.n_kv_groups > 1 {
            crate::layers::repeat_kv(k.clone(), self.sdpa_params.n_kv_groups)?
        } else {
            k.clone()
        };
        let v = if self.sdpa_params.n_kv_groups > 1 {
            crate::layers::repeat_kv(v.clone(), self.sdpa_params.n_kv_groups)?
        } else {
            v.clone()
        };

        // Compute attention scores: [b, heads, q_len, k_len]
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.sdpa_params.softmax_scale as f64)?;

        // Apply attention mask (causal) - convert mask to match attn_weights dtype
        let attn_weights = if let Some(mask) = attention_mask {
            // The mask might have shape [b, 1, q_len, total_len] or [b, 1, 1, total_len]
            // We need to narrow the last dim to k_len, but only if it's larger than k_len
            let mask_last_dim = mask.dim(D::Minus1)?;
            let causal_mask = if mask_last_dim > k_len {
                mask.narrow(D::Minus1, 0, k_len)?.to_dtype(attn_weights.dtype())?
            } else {
                // Mask is already the right size or smaller (will broadcast)
                mask.to_dtype(attn_weights.dtype())?
            };
            attn_weights.broadcast_add(&causal_mask)?
        } else {
            attn_weights
        };

        // Apply sliding window mask if configured for this layer
        let attn_weights = if let Some(window) = self.sdpa_params.sliding_window {
            // For sliding window attention, mask positions outside the window
            // Query at absolute position q_abs can only attend to keys where:
            // k_pos >= q_abs - window
            // Since k_pos ranges from 0 to k_len-1 and q_abs = seqlen_offset + q_idx
            let seqlen_offset = seqlen_offsets.first().copied().unwrap_or(0);

            // Create sliding window mask: [q_len, k_len]
            // For each (q_idx, k_idx), mask if k_idx < seqlen_offset + q_idx - window
            let mut mask_data = vec![0.0f32; q_len * k_len];
            for q_idx in 0..q_len {
                let q_abs = seqlen_offset + q_idx;
                let min_k = q_abs.saturating_sub(window);
                for k_idx in 0..k_len {
                    if k_idx < min_k {
                        mask_data[q_idx * k_len + k_idx] = f32::NEG_INFINITY;
                    }
                }
            }

            let sliding_mask = Tensor::from_vec(mask_data, (q_len, k_len), q.device())?
                .to_dtype(attn_weights.dtype())?
                .reshape((1, 1, q_len, k_len))?;

            attn_weights.broadcast_add(&sliding_mask)?
        } else {
            attn_weights
        };

        // Add sinks: reshape [num_heads] -> [1, num_heads, 1, 1] and concat along last dim
        let sinks_expanded = self
            .sinks
            .reshape((1, num_heads, 1, 1))?
            .broadcast_as((b_sz, num_heads, q_len, 1))?
            .to_dtype(attn_weights.dtype())?;

        // Concatenate sinks to attention logits: [b, heads, q_len, k_len + 1]
        let combined_logits = Tensor::cat(&[&attn_weights, &sinks_expanded], D::Minus1)?;

        // Subtract max for numerical stability
        let max_logits = combined_logits.max_keepdim(D::Minus1)?;
        let combined_logits = combined_logits.broadcast_sub(&max_logits)?;

        // Softmax
        let probs = candle_nn::ops::softmax_last_dim(&combined_logits)?;

        // Drop the sink probability (take all but last)
        // narrow creates a non-contiguous view, matmul requires contiguous tensors
        let scores = probs.narrow(D::Minus1, 0, k_len)?.contiguous()?;

        // Compute output
        scores.matmul(&v)
    }
}

/// GPT-OSS MoE layer with combined gate_up_proj and special activation
struct GptOssMoE {
    gate: Linear,
    gate_up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    #[allow(dead_code)]
    num_experts: usize,
    num_experts_per_tok: usize,
    intermediate_size: usize,
    alpha: f32,
    limit: f32,
}

impl GptOssMoE {
    fn new(cfg: &Config, vb: ShardedVarBuilder, layer_device: Device) -> Result<Self> {
        // Router is not quantized
        let gate = layers::linear(
            cfg.hidden_size,
            cfg.num_local_experts,
            vb.pp("router").set_device(layer_device.clone()),
        )?;

        // Load MXFP4 quantized experts
        let experts_vb = vb.pp("experts").set_device(layer_device);

        // gate_up_proj: [num_experts, intermediate_size * 2, hidden_size/2] packed
        // down_proj: [num_experts, hidden_size, intermediate_size/2] packed
        let gate_up_proj = MXFP4Layer::packed_gptoss_linear(
            cfg.num_local_experts,
            cfg.hidden_size,
            cfg.intermediate_size * 2,
            true, // has bias
            "gate_up_proj",
            experts_vb.clone(),
        )?;

        let down_proj = MXFP4Layer::packed_gptoss_linear(
            cfg.num_local_experts,
            cfg.intermediate_size,
            cfg.hidden_size,
            true, // has bias
            "down_proj",
            experts_vb,
        )?;

        Ok(Self {
            gate,
            gate_up_proj,
            down_proj,
            num_experts: cfg.num_local_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            intermediate_size: cfg.intermediate_size,
            alpha: cfg.alpha,
            limit: cfg.swiglu_limit,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let _num_tokens = xs_flat.dim(0)?;

        // Compute routing logits
        let router_logits = self.gate.forward(&xs_flat)?.to_dtype(DType::F32)?;

        // HF: Select top-k from raw logits FIRST, then softmax only on selected values
        // router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        // router_scores = softmax(router_top_value, dim=1)
        let topk_ids_i64 = router_logits
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let topk_logits = router_logits.gather(&topk_ids_i64, D::Minus1)?;

        // Convert to U32 for the CUDA kernel
        let topk_ids = topk_ids_i64.to_dtype(DType::U32)?;

        // Softmax only on the top-k logits (not all logits)
        let topk_weights = candle_nn::ops::softmax_last_dim(&topk_logits)?;

        // Forward through experts using gather_forward
        // gate_up_proj output: [num_tokens, topk, intermediate_size * 2]
        let gate_up = self.gate_up_proj.gather_forward(&xs_flat, &topk_ids)?;

        // Split gate and up - they are INTERLEAVED (gate at even indices, up at odd indices)
        // HF: gate = gate_up[..., ::2], up = gate_up[..., 1::2]
        // gate_up shape: [num_tokens, topk, intermediate_size * 2]
        // Reshape to [num_tokens, topk, intermediate_size, 2] then select
        let (num_tokens, topk_dim, _) = gate_up.dims3()?;
        let gate_up_reshaped = gate_up.reshape((num_tokens, topk_dim, self.intermediate_size, 2))?;
        let gate = gate_up_reshaped.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let up = gate_up_reshaped.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

        // Apply GPT-OSS SwiGLU activation
        let activated = gptoss_swiglu(&gate, &up, self.alpha, self.limit)?;

        // down_proj: [num_tokens, topk, hidden_size]
        let expert_out = self.down_proj.gather_forward(&activated, &topk_ids)?;

        // Weight and sum expert outputs
        // topk_weights: [num_tokens, topk] -> [num_tokens, topk, 1]
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
        rotary_emb: GptOssRotaryEmbedding,
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
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
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

    #[allow(clippy::too_many_arguments)]
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
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
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

        // embed_tokens is NOT quantized (in modules_to_not_convert)
        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &None, // Not quantized
        )?;

        let mut ropes: HashMap<_, GptOssRotaryEmbedding> = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);

            // Use YARN RoPE if configured, otherwise standard RoPE
            let rope = if let Some(rope_scaling) = &cfg.rope_scaling {
                if rope_scaling.rope_type == "yarn" {
                    GptOssRotaryEmbedding::Yarn(Arc::new(YarnRotaryEmbedding::new(
                        cfg.rope_theta,
                        cfg.head_dim(),
                        cfg.max_position_embeddings,
                        rope_scaling,
                        device,
                        vb_m.dtype(),
                    )?))
                } else {
                    // Fallback to standard RoPE for unknown types
                    GptOssRotaryEmbedding::Standard(Arc::new(RotaryEmbedding::new(
                        cfg.rope_theta as f32,
                        cfg.head_dim(),
                        cfg.max_position_embeddings,
                        device,
                        is_gptx,
                        vb_m.dtype(),
                    )?))
                }
            } else {
                GptOssRotaryEmbedding::Standard(Arc::new(RotaryEmbedding::new(
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

        // lm_head is NOT quantized (in modules_to_not_convert)
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &None, // Not quantized
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
            mapper,
            cfg: cfg.clone(),
            cfg_metadata,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn inner_forward(
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
            self.layers[0].self_attn.num_heads,
        )?;
        // PagedAttention prompt chunking
        let attention_mask = attention_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv, m)| (kv[i].clone(), *m)),
                flash_params,
            )?;
        }

        let xs = xs.to_device(&self.device)?;
        let xs = self.norm.forward(&xs)?;
        extract_logits(&MatMul.qmethod_matmul(&xs, &*self.lm_head)?, context_lens)
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut layers = Vec::new();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layers.push((&mut layer.self_attn.q_proj, Some(i)));
            layers.push((&mut layer.self_attn.k_proj, Some(i)));
            layers.push((&mut layer.self_attn.v_proj, Some(i)));
            layers.push((&mut layer.self_attn.o_proj, Some(i)));
            // MoE layers are already MXFP4 quantized, don't ISQ them
        }
        layers.push((&mut self.lm_head, None));
        (layers, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }
}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.inner_forward(input_ids, seqlen_offsets, context_lens, metadata, flash_params)
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
        &self.cfg_metadata
    }
}

impl AnyMoeBaseModelMixin for Model {}
