#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{f32::consts::PI, ops::Mul, str::FromStr, sync::Arc};

use candle_core::{
    quantized::{QMatMul, QTensor},
    Context, DType, Device, IndexOp, Result, Tensor, D,
};
use candle_nn::{
    Conv2d, Conv2dConfig, Embedding, GroupNorm, LayerNorm, LayerNormConfig, Linear, Module,
};
use float8::F8E4M3;
use half::{bf16, f16};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, RowParallelLayer, ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};

pub use crate::attention::Sdpa;
pub use crate::layers_masker::CausalMasker;
pub use crate::layers_utils::repeat_kv;
use crate::{
    amoe::{AnyMoeTrainableLayer, MlpLayer},
    gguf::Content,
    models::llama,
    ops::SplitOp,
    vision_models::{
        gemma3::config::Gemma3TextConfig,
        mllama::{MLlamaRopeScaling, MLlamaRopeType, MLlamaTextConfig},
        phi4::Phi4MMConfig,
    },
};

pub use mistralrs_quant::MatMul;

pub fn embedding(in_size: usize, out_size: usize, vb: ShardedVarBuilder) -> Result<Embedding> {
    let embeddings = vb.get_with_hints((in_size, out_size), "weight", Default::default())?;
    Ok(Embedding::new(embeddings, out_size))
}

pub fn layer_norm<C: Into<LayerNormConfig>>(
    size: usize,
    config: C,
    vb: ShardedVarBuilder,
) -> Result<LayerNorm> {
    let config = config.into();
    let weight = vb.get(size, "weight")?;
    if config.affine {
        let bias = vb.get(size, "bias")?;
        Ok(LayerNorm::new(weight, bias, config.eps))
    } else {
        Ok(LayerNorm::new_no_bias(weight, config.eps))
    }
}

pub fn group_norm(
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    vb: ShardedVarBuilder,
) -> Result<GroupNorm> {
    let weight = vb.get(num_channels, "weight")?;
    let bias = vb.get(num_channels, "bias")?;
    GroupNorm::new(weight, bias, num_channels, num_groups, eps)
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: ShardedVarBuilder,
) -> Result<Conv2d> {
    let ws = vb.get(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
    )?;
    let bs = vb.get(out_channels, "bias")?;
    Ok(Conv2d::new(ws, Some(bs), cfg))
}

pub fn conv2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: ShardedVarBuilder,
) -> Result<Conv2d> {
    let ws = vb.get(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
    )?;
    Ok(Conv2d::new(ws, None, cfg))
}

pub fn linear(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    let bs = vb.get(out_dim, "bias")?;
    Ok(Linear::new(ws, Some(bs)))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(ws, None))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: ShardedVarBuilder,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    eps: f64,
    weight: Tensor,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let w = vb.get(size, "weight")?;
        Ok(Self { eps, weight: w })
    }

    /// Gemma uses weight + 1.0
    pub fn new_gemma(size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let w = vb.get(size, "weight")?;
        let w = (w + 1.0)?;
        Ok(Self { eps, weight: w })
    }

    /// Gemma uses weight + 1.0. Undo for UQFF generation.
    pub fn undo_gemma(&self) -> Result<Self> {
        Ok(Self {
            eps: self.eps,
            weight: (&self.weight - 1.0)?,
        })
    }

    pub fn from_w(w: Tensor, eps: f64) -> Result<Self> {
        Ok(Self { eps, weight: w })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32)
    }
}

#[derive(Debug, Clone)]
pub struct F32RmsNorm {
    w: Tensor,
    eps: f64,
}

impl F32RmsNorm {
    pub fn new(size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            w: vb.get((size,), "weight")?,
            eps,
        })
    }

    pub fn weight(&self) -> &Tensor {
        &self.w
    }
}

impl Module for F32RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let initial_type = xs.dtype();
        let mut xs = xs.to_dtype(DType::F32)?;
        let var = xs.powf(2.)?.mean_keepdim(D::Minus1)?;
        xs = xs.broadcast_mul(&(&var + self.eps)?.recip()?.sqrt()?)?;
        xs.to_dtype(initial_type)?.broadcast_mul(&self.w)
    }
}

#[derive(Debug, Clone)]
pub struct QRmsNorm {
    eps: f64,
    weight: Tensor,
}

impl QRmsNorm {
    pub fn new(scale: QTensor, eps: f32) -> Result<Self> {
        let scale = scale.dequantize(&scale.device())?;
        Ok(Self {
            eps: eps as f64,
            weight: scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32)
    }
}

/// RoPE supporting LongRope
#[derive(Debug, Clone)]
pub struct PhiRotaryEmbedding {
    short_sin: Tensor,
    short_cos: Tensor,
    long_cos: Option<Tensor>,
    long_sin: Option<Tensor>,
    original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaledRopeType {
    #[serde(alias = "su")]
    #[serde(alias = "longrope")]
    Su,
    #[serde(alias = "yarn")]
    Yarn,
    #[serde(alias = "dynamic")]
    Dynamic,
    #[serde(alias = "linear")]
    Linear,
}

impl FromStr for ScaledRopeType {
    type Err = candle_core::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "su" | "longrope" => Ok(Self::Su),
            "yarn" => Ok(Self::Yarn),
            "linear" => Ok(Self::Linear),
            "dynamic" => Ok(Self::Dynamic),
            _ => Err(candle_core::Error::Msg(
                "Expected either `su` or `yarn` scaled RoPE type.".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PhiRopeScalingConfig {
    Classic {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
    },
    Scaled {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
        long_mscale: f64,
        short_mscale: f64,
    },
}

pub struct PhiRopeConfig {
    pub rope_scaling: Option<PhiRopeScalingConfig>,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    pub rope_theta: f64,
    pub head_dim: usize,
    pub partial_rotary_factor: Option<f64>,
}

impl PhiRotaryEmbedding {
    fn new_classic_scaled(
        short_factor: &[f64],
        long_factor: &[f64],
        scaling_type: &ScaledRopeType,
        cfg: &PhiRopeConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor.unwrap_or(1.)) as usize;

        // Calculate scale
        let scale =
            cfg.max_position_embeddings as f64 / cfg.original_max_position_embeddings as f64;
        let scaling_factor = if scale <= 1.0 {
            1.0
        } else {
            match scaling_type {
                ScaledRopeType::Su => {
                    (1.0 + scale.ln() / (cfg.original_max_position_embeddings as f64).ln()).sqrt()
                }
                ScaledRopeType::Yarn => 0.1 * scale.ln() + 1.0,
                _ => candle_core::bail!("Expected either `su` or `yarn` RoPE"),
            }
        };

        // Calculate inv freqs for short, long
        let inv_freq_long = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                (1f64 / (long_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64))) as f32
            })
            .collect::<Vec<_>>();
        let inv_freq_short = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                (1f64 / (short_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64))) as f32
            })
            .collect::<Vec<_>>();
        let inv_freq_len = inv_freq_long.len();

        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;

        // Calculate sin,cos for long
        let inv_freq_long = Tensor::from_vec(inv_freq_long, (1, inv_freq_len), dev)?;
        let freqs_long = t.matmul(&inv_freq_long)?;
        let long_sin = freqs_long.sin()?.mul(scaling_factor)?.to_dtype(dtype)?;
        let long_cos = freqs_long.cos()?.mul(scaling_factor)?.to_dtype(dtype)?;

        // Calculate sin,cos for short
        let inv_freq_short =
            Tensor::from_vec(inv_freq_short, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let freqs_short = t.matmul(&inv_freq_short)?;
        let short_sin = freqs_short.sin()?.mul(scaling_factor)?.to_dtype(dtype)?;
        let short_cos = freqs_short.cos()?.mul(scaling_factor)?.to_dtype(dtype)?;

        Ok(Self {
            short_cos,
            short_sin,
            long_cos: Some(long_cos),
            long_sin: Some(long_sin),
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    fn new_unscaled(cfg: &PhiRopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor.unwrap_or(1.)) as usize;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        Ok(Self {
            short_cos: cos,
            short_sin: sin,
            long_cos: None,
            long_sin: None,
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_scaled(
        short_factor: &[f64],
        long_factor: &[f64],
        scaling_type: &ScaledRopeType,
        long_mscale: f64,
        short_mscale: f64,
        cfg: &PhiRopeConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor.unwrap_or(1.)) as usize;

        if !matches!(scaling_type, ScaledRopeType::Su) {
            candle_core::bail!("Scaled Phi3 RoPE (non-classic scaled, with mscales) must have type `su`/`longrope`.");
        }

        if short_factor.len() != dim / 2 {
            candle_core::bail!(
                "Misaligned length {}, expected {} for `su`/`longrope` short rescale factors",
                short_factor.len(),
                dim / 2
            );
        }
        if long_factor.len() != dim / 2 {
            candle_core::bail!(
                "Misaligned length {}, expected {} for `su`/`longrope` long rescale factors",
                long_factor.len(),
                dim / 2
            );
        }

        // Short cos/sin
        let inv_freq_short: Vec<_> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                1f32 / (short_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)) as f32
            })
            .collect();
        let inv_freq_len_short = inv_freq_short.len();
        let inv_freq_short = Tensor::from_vec(inv_freq_short, (1, inv_freq_len_short), dev)?;
        let t_short = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs_short = t_short.matmul(&inv_freq_short)?;
        let sin_short = (freqs_short.sin()?.to_dtype(dtype)? * short_mscale)?;
        let cos_short = (freqs_short.cos()?.to_dtype(dtype)? * short_mscale)?;

        // Long cos/sin
        let inv_freq_long: Vec<_> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                1f32 / (long_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)) as f32
            })
            .collect();
        let inv_freq_len_long = inv_freq_long.len();
        let inv_freq_long = Tensor::from_vec(inv_freq_long, (1, inv_freq_len_long), dev)?;
        let t_long = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs_long = t_long.matmul(&inv_freq_long)?;
        let sin_long = (freqs_long.sin()?.to_dtype(dtype)? * long_mscale)?;
        let cos_long = (freqs_long.cos()?.to_dtype(dtype)? * long_mscale)?;
        Ok(Self {
            short_cos: cos_short,
            short_sin: sin_short,
            long_cos: Some(cos_long),
            long_sin: Some(sin_long),
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    pub fn new(dtype: DType, cfg: impl Into<PhiRopeConfig>, dev: &Device) -> Result<Self> {
        let cfg: PhiRopeConfig = cfg.into();

        match &cfg.rope_scaling {
            Some(PhiRopeScalingConfig::Classic {
                short_factor,
                long_factor,
                scaling_type,
            }) => {
                Self::new_classic_scaled(short_factor, long_factor, scaling_type, &cfg, dtype, dev)
            }

            Some(PhiRopeScalingConfig::Scaled {
                short_factor,
                long_factor,
                scaling_type,
                long_mscale,
                short_mscale,
            }) => Self::new_scaled(
                short_factor,
                long_factor,
                scaling_type,
                *long_mscale,
                *short_mscale,
                &cfg,
                dtype,
                dev,
            ),

            None => Self::new_unscaled(&cfg, dtype, dev),
        }
    }

    /// Returns (sin, cos) taking into account LongRope
    fn get_long_or_short_sin_cos(&self, position_ids: &[usize]) -> (&Tensor, &Tensor) {
        if self.long_cos.is_none() {
            return (&self.short_sin, &self.short_cos);
        }
        let seq_len = position_ids.iter().max().unwrap() + 1;
        if seq_len > self.original_max_position_embeddings {
            (
                self.long_sin.as_ref().unwrap(),
                self.long_cos.as_ref().unwrap(),
            )
        } else {
            (&self.short_sin, &self.short_cos)
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (sin, cos) = self.get_long_or_short_sin_cos(position_ids);
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let all_same = seqlen_offsets.iter().all(|&x| x == seqlen_offsets[0]);

        let rot_dim = cos.dim(D::Minus1)? * 2;

        // Case for Phi 3 / Phi 4 mini
        if rot_dim != q.dim(D::Minus1)? {
            let rot_dim = cos.dim(D::Minus1)? * 2;
            let q_rot = q.narrow(D::Minus1, 0, rot_dim)?;
            let q_pass = q.narrow(D::Minus1, rot_dim, q.dim(D::Minus1)? - rot_dim)?;
            let k_rot = k.narrow(D::Minus1, 0, rot_dim)?;
            let k_pass = k.narrow(D::Minus1, rot_dim, k.dim(D::Minus1)? - rot_dim)?;

            let (q_rot, k_rot) = if all_same {
                let cos = cos.narrow(0, seqlen_offsets[0], seq_len)?;
                let sin = sin.narrow(0, seqlen_offsets[0], seq_len)?;
                let q_embed = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, &cos, &sin)?;
                let k_embed = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, &cos, &sin)?;
                (q_embed, k_embed)
            } else {
                let mut q_embeds = Vec::new();
                let mut k_embeds = Vec::new();
                for (i, offset) in seqlen_offsets.iter().enumerate() {
                    let cos = cos.narrow(0, *offset, seq_len)?;
                    let sin = sin.narrow(0, *offset, seq_len)?;
                    let q_embed = candle_nn::rotary_emb::rope(
                        &q_rot.i(i)?.unsqueeze(0)?.contiguous()?,
                        &cos,
                        &sin,
                    )?;
                    let k_embed = candle_nn::rotary_emb::rope(
                        &k_rot.i(i)?.unsqueeze(0)?.contiguous()?,
                        &cos,
                        &sin,
                    )?;
                    q_embeds.push(q_embed);
                    k_embeds.push(k_embed);
                }
                let q_rot = Tensor::cat(&q_embeds, 0)?;
                let k_rot = Tensor::cat(&k_embeds, 0)?;
                (q_rot, k_rot)
            };

            Ok((
                Tensor::cat(&[q_rot, q_pass], D::Minus1)?.contiguous()?,
                Tensor::cat(&[k_rot, k_pass], D::Minus1)?.contiguous()?,
            ))
        } else if all_same {
            let cos = cos.narrow(0, seqlen_offsets[0], seq_len)?;
            let sin = sin.narrow(0, seqlen_offsets[0], seq_len)?;
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else {
            let mut q_embeds = Vec::new();
            let mut k_embeds = Vec::new();
            for (i, offset) in seqlen_offsets.iter().enumerate() {
                let cos = cos.narrow(0, *offset, seq_len)?;
                let sin = sin.narrow(0, *offset, seq_len)?;
                let q_embed =
                    candle_nn::rotary_emb::rope(&q.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                let k_embed =
                    candle_nn::rotary_emb::rope(&k.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                q_embeds.push(q_embed);
                k_embeds.push(k_embed);
            }
            Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
        }
    }
}

/// RoPE for Llama3
#[derive(Debug, Clone)]
pub struct Llama3RotaryEmbedding(RotaryEmbedding);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}

fn calculate_default_inv_freq(cfg: &llama::Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

// https://github.com/huggingface/transformers/blob/1392a6867f40a55dfabaf306745c67627598b1af/src/transformers/modeling_rope_utils.py#L298
impl Llama3RotaryEmbedding {
    pub fn new_llama3(
        dtype: DType,
        cfg: &llama::Config,
        dev: &Device,
        is_gpt_neox: bool,
    ) -> Result<Self> {
        match &cfg.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => Ok(Self(RotaryEmbedding::new(
                cfg.rope_theta,
                cfg.hidden_size / cfg.num_attention_heads,
                cfg.max_position_embeddings,
                dev,
                is_gpt_neox,
                dtype,
            )?)),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                let inv_freq = calculate_default_inv_freq(cfg)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>();
                let inv_freq_len = inv_freq.len();
                let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;

                let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
                    .to_dtype(DType::F32)?
                    .reshape((cfg.max_position_embeddings, 1))?;
                let freqs = t.matmul(&inv_freq)?;
                let sin = freqs.sin()?.to_dtype(dtype)?;
                let cos = freqs.cos()?.to_dtype(dtype)?;
                Ok(Self(RotaryEmbedding {
                    sin,
                    cos,
                    is_gpt_neox,
                }))
            }
        }
    }

    pub fn new_mllama3(
        dtype: DType,
        cfg: &MLlamaTextConfig,
        dev: &Device,
        is_gpt_neox: bool,
    ) -> Result<Self> {
        match &cfg.rope_scaling {
            None
            | Some(MLlamaRopeScaling {
                rope_type: MLlamaRopeType::Default,
                ..
            }) => Ok(Self(RotaryEmbedding::new(
                cfg.rope_theta,
                cfg.hidden_size / cfg.num_attention_heads,
                cfg.max_position_embeddings,
                dev,
                is_gpt_neox,
                dtype,
            )?)),
            Some(MLlamaRopeScaling {
                rope_type: MLlamaRopeType::Llama3,
                original_max_position_embeddings,
                factor,
                attention_factor: _,
                beta_fast: _,
                beta_slow: _,
                short_factor: _,
                long_factor: _,
                low_freq_factor,
                high_freq_factor,
            }) => {
                let factor = factor.context("MLlama Llama3 RoPE needs `factor` parameter.")?;
                let low_freq_factor = low_freq_factor
                    .context("MLlama Llama3 RoPE needs `low_freq_factor` parameter.")?;
                let high_freq_factor = high_freq_factor
                    .context("MLlama Llama3 RoPE needs `high_freq_factor` parameter.")?;

                let low_freq_wavelen = *original_max_position_embeddings as f32 / low_freq_factor;
                let high_freq_wavelen = *original_max_position_embeddings as f32 / high_freq_factor;

                let head_dim = cfg.hidden_size / cfg.num_attention_heads;

                let inv_freq = (0..head_dim)
                    .step_by(2)
                    .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / factor
                        } else {
                            let smooth = (*original_max_position_embeddings as f32 / wavelen
                                - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);
                            (1. - smooth) * freq / factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>();
                let inv_freq_len = inv_freq.len();
                let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;

                let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
                    .to_dtype(DType::F32)?
                    .reshape((cfg.max_position_embeddings, 1))?;
                let freqs = t.matmul(&inv_freq)?;
                let sin = freqs.sin()?.to_dtype(dtype)?;
                let cos = freqs.cos()?.to_dtype(dtype)?;
                Ok(Self(RotaryEmbedding {
                    sin,
                    cos,
                    is_gpt_neox,
                }))
            }
            Some(MLlamaRopeScaling {
                rope_type: other, ..
            }) => {
                candle_core::bail!(
                    "MLlama doesn't support any other RoPE type than `llama3`, got {other:?}"
                )
            }
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        self.0.forward(q, k, seqlen_offsets)
    }
}

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L107
#[derive(Debug, Clone)]
pub struct Qwen2VLRotaryEmbedding {
    inv_freq: Tensor,
    mrope_section: Vec<usize>,
}

impl Qwen2VLRotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        device: &Device,
        mrope_section: Vec<usize>,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (inv_freq_len,), device)?.to_dtype(DType::F32)?;
        Ok(Self {
            inv_freq,
            mrope_section,
        })
    }

    /// (cos, sin)
    pub fn compute_cos_sin(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let inv_freq_expanded =
            self.inv_freq
                .reshape((1, 1, (), 1))?
                .repeat((3, position_ids.dim(1)?, 1, 1))?;
        let position_ids_expanded = position_ids.unsqueeze(2)?;
        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded.to_dtype(inv_freq_expanded.dtype())?)?
            .transpose(2, 3)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        let cos = Tensor::cat(
            &cos.split(&self.mrope_section, D::Minus1)?
                .into_iter()
                .enumerate()
                .map(|(i, m)| m.i(i % 3))
                .collect::<Result<Vec<_>>>()?,
            D::Minus1,
        )?
        .squeeze(0)?
        .to_dtype(dtype)?
        .contiguous()?;
        let sin = Tensor::cat(
            &sin.split(&self.mrope_section, D::Minus1)?
                .into_iter()
                .enumerate()
                .map(|(i, m)| m.i(i % 3))
                .collect::<Result<Vec<_>>>()?,
            D::Minus1,
        )?
        .squeeze(0)?
        .to_dtype(dtype)?
        .contiguous()?;

        Ok((cos, sin))
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L203
    pub fn forward(
        &self,
        (cos, sin): &(Tensor, Tensor),
        q: &mut Tensor,
        k: &mut Tensor,
    ) -> Result<()> {
        *q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        *k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2_5VLRotaryEmbedding {
    inv_freq: Tensor,
    mrope_section: Vec<usize>,
}

impl Qwen2_5VLRotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        device: &Device,
        mrope_section: Vec<usize>,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (inv_freq_len,), device)?.to_dtype(DType::F32)?;
        Ok(Self {
            inv_freq,
            mrope_section,
        })
    }

    /// (cos, sin)
    pub fn compute_cos_sin(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let inv_freq_expanded =
            self.inv_freq
                .reshape((1, 1, (), 1))?
                .repeat((3, position_ids.dim(1)?, 1, 1))?;
        let position_ids_expanded = position_ids.unsqueeze(2)?;
        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded.to_dtype(inv_freq_expanded.dtype())?)?
            .transpose(2, 3)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        let cos = Tensor::cat(
            &cos.split(&self.mrope_section, D::Minus1)?
                .into_iter()
                .enumerate()
                .map(|(i, m)| m.i(i % 3))
                .collect::<Result<Vec<_>>>()?,
            D::Minus1,
        )?
        .squeeze(0)?
        .to_dtype(dtype)?
        .contiguous()?;
        let sin = Tensor::cat(
            &sin.split(&self.mrope_section, D::Minus1)?
                .into_iter()
                .enumerate()
                .map(|(i, m)| m.i(i % 3))
                .collect::<Result<Vec<_>>>()?,
            D::Minus1,
        )?
        .squeeze(0)?
        .to_dtype(dtype)?
        .contiguous()?;

        Ok((cos, sin))
    }

    pub fn forward(
        &self,
        (cos, sin): &(Tensor, Tensor),
        q: &mut Tensor,
        k: &mut Tensor,
    ) -> Result<()> {
        *q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        *k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekV2RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum DeepSeekV2RopeScaling {
    Yarn {
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        mscale: f32,
        mscale_all_dim: f32,
        factor: f32,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
    },
    LinearOrDynamic {
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
        factor: f64,
    },
}

pub struct DeepSeekV2RopeConfig {
    pub rope_scaling: Option<DeepSeekV2RopeScaling>,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub qk_rope_head_dim: usize,
}

impl DeepSeekV2RotaryEmbedding {
    fn new_unscaled(cfg: &DeepSeekV2RopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.qk_rope_head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self { sin, cos })
    }

    fn yarn_find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2. * PI)).ln())
            / (2. * base.ln())
    }

    fn yarn_find_correction_range(
        low_rot: f32,
        high_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> (f32, f32) {
        let low =
            Self::yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
        let high =
            Self::yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
        (low.max(0.), high.min(dim as f32 - 1.))
    }

    fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        if min == max {
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/modeling_deepseek.py#L255
            max += 0.001;
        }
        let linear_func =
            ((Tensor::arange(0f32, dim as f32, dev)? - min as f64)? / (max as f64 - min as f64))?;
        linear_func.clamp(0., 1)
    }

    pub(crate) fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
        if scale <= 1. {
            return 1.;
        }
        0.1 * mscale * scale.ln() + 1.
    }

    #[allow(clippy::too_many_arguments)]
    fn new_yarn(
        cfg: &DeepSeekV2RopeConfig,
        dtype: DType,
        dev: &Device,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        factor: f32,
        mscale: f32,
        mscale_all_dim: f32,
    ) -> Result<Self> {
        let freq_extra: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32))
            .collect();
        let freq_extra_len = freq_extra.len();
        let freq_extra = Tensor::from_vec(freq_extra, freq_extra_len, dev)?;
        let freq_inter: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / (factor * cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32)))
            .collect();
        let freq_inter_len = freq_inter.len();
        let freq_inter = Tensor::from_vec(freq_inter, (1, freq_inter_len), dev)?;

        let (low, high) = Self::yarn_find_correction_range(
            beta_fast,
            beta_slow,
            cfg.qk_rope_head_dim,
            cfg.rope_theta,
            original_max_position_embeddings,
        );
        let inv_freq_mask =
            (1. - Self::yarn_linear_ramp_mask(low, high, cfg.qk_rope_head_dim / 2, dev)?)?;
        let inv_freq = freq_inter
            .broadcast_mul(&(1. - &inv_freq_mask)?)?
            .broadcast_add(&freq_extra.broadcast_mul(&inv_freq_mask)?)?;

        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let mscale =
            Self::yarn_get_mscale(factor, mscale) / Self::yarn_get_mscale(factor, mscale_all_dim);
        let sin = (freqs.sin()? * mscale as f64)?.to_dtype(dtype)?;
        let cos = (freqs.cos()? * mscale as f64)?.to_dtype(dtype)?;

        Ok(Self { sin, cos })
    }

    pub fn new(cfg: &DeepSeekV2RopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(DeepSeekV2RopeScaling::LinearOrDynamic {
                scaling_type: _,
                factor: _,
            }) => candle_core::bail!("linear and dynamic rope are not implemented yet!"),
            Some(DeepSeekV2RopeScaling::Yarn {
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                factor,
                mscale,
                mscale_all_dim,
                scaling_type: _,
            }) => Self::new_yarn(
                cfg,
                dtype,
                dev,
                *original_max_position_embeddings,
                *beta_fast,
                *beta_slow,
                *factor,
                *mscale,
                *mscale_all_dim,
            ),
            None => Self::new_unscaled(cfg, dtype, dev),
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let all_same = seqlen_offsets.iter().all(|&x| x == seqlen_offsets[0]);
        if all_same {
            let cos = self.cos.narrow(0, seqlen_offsets[0], seq_len)?;
            let sin = self.sin.narrow(0, seqlen_offsets[0], seq_len)?;
            let q_embed = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else {
            let mut q_embeds = Vec::new();
            let mut k_embeds = Vec::new();
            for (i, offset) in seqlen_offsets.iter().enumerate() {
                let cos = self.cos.narrow(0, *offset, seq_len)?;
                let sin = self.sin.narrow(0, *offset, seq_len)?;
                let q_embed = candle_nn::rotary_emb::rope_i(
                    &q.i(i)?.unsqueeze(0)?.contiguous()?,
                    &cos,
                    &sin,
                )?;
                let k_embed = candle_nn::rotary_emb::rope_i(
                    &k.i(i)?.unsqueeze(0)?.contiguous()?,
                    &cos,
                    &sin,
                )?;
                q_embeds.push(q_embed);
                k_embeds.push(k_embed);
            }
            Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
        }
    }
}

#[derive(Debug, Clone)]
pub struct Phi4MMRotaryEmbedding {
    short_sin: Tensor,
    short_cos: Tensor,
    long_cos: Option<Tensor>,
    long_sin: Option<Tensor>,
    original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phi4MMScaledRopeType {
    #[serde(alias = "longrope")]
    LongRope,
    #[default]
    Default,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Phi4MMRopeScalingConfig {
    short_factor: Option<Vec<f64>>,
    long_factor: Option<Vec<f64>>,
    #[serde(rename = "type")]
    scaling_type: Phi4MMScaledRopeType,
}

impl Phi4MMRotaryEmbedding {
    fn new_unscaled(cfg: &Phi4MMConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = (cfg.head_dim() as f64 * cfg.partial_rotary_factor) as usize;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        Ok(Self {
            short_cos: cos,
            short_sin: sin,
            long_cos: None,
            long_sin: None,
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_longrope(
        short_factor: &[f64],
        long_factor: &[f64],
        cfg: &Phi4MMConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = (cfg.head_dim() as f64 * cfg.partial_rotary_factor) as usize;

        // Calculate scale
        let scale =
            cfg.max_position_embeddings as f64 / cfg.original_max_position_embeddings as f64;
        let scaling_factor = if scale <= 1.0 {
            1.0
        } else {
            (1.0 + scale.ln() / (cfg.original_max_position_embeddings as f64).ln()).sqrt()
        };

        // Short cos/sin
        let inv_freq_short: Vec<_> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                1f32 / (short_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)) as f32
            })
            .collect();
        let inv_freq_len_short = inv_freq_short.len();
        let inv_freq_short = Tensor::from_vec(inv_freq_short, (1, inv_freq_len_short), dev)?;
        let t_short = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs_short = t_short.matmul(&inv_freq_short)?;
        let sin_short = (freqs_short.sin()?.to_dtype(dtype)? * scaling_factor)?;
        let cos_short = (freqs_short.cos()?.to_dtype(dtype)? * scaling_factor)?;

        // Long cos/sin
        let inv_freq_long: Vec<_> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                1f32 / (long_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)) as f32
            })
            .collect();
        let inv_freq_len_long = inv_freq_long.len();
        let inv_freq_long = Tensor::from_vec(inv_freq_long, (1, inv_freq_len_long), dev)?;
        let t_long = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs_long = t_long.matmul(&inv_freq_long)?;
        let sin_long = (freqs_long.sin()?.to_dtype(dtype)? * scaling_factor)?;
        let cos_long = (freqs_long.cos()?.to_dtype(dtype)? * scaling_factor)?;

        Ok(Self {
            short_cos: cos_short,
            short_sin: sin_short,
            long_cos: Some(cos_long),
            long_sin: Some(sin_long),
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    pub fn new(dtype: DType, cfg: &Phi4MMConfig, dev: &Device) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(Phi4MMRopeScalingConfig {
                scaling_type: Phi4MMScaledRopeType::LongRope,
                short_factor: Some(short_factor),
                long_factor: Some(long_factor),
            }) => Self::new_longrope(short_factor, long_factor, cfg, dtype, dev),

            _ => Self::new_unscaled(cfg, dtype, dev),
        }
    }

    /// Returns (sin, cos) taking into account LongRope
    fn get_long_or_short_sin_cos(&self, position_ids: &[usize]) -> (&Tensor, &Tensor) {
        if self.long_cos.is_none() {
            return (&self.short_sin, &self.short_cos);
        }
        let seq_len = position_ids.iter().max().unwrap() + 1;
        if seq_len > self.original_max_position_embeddings {
            (
                self.long_sin.as_ref().unwrap(),
                self.long_cos.as_ref().unwrap(),
            )
        } else {
            (&self.short_sin, &self.short_cos)
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let (sin, cos) = self.get_long_or_short_sin_cos(position_ids);

        let rot_dim = cos.dim(D::Minus1)? * 2;
        let q_rot = q.narrow(D::Minus1, 0, rot_dim)?;
        let q_pass = q.narrow(D::Minus1, rot_dim, q.dim(D::Minus1)? - rot_dim)?;
        let k_rot = k.narrow(D::Minus1, 0, rot_dim)?;
        let k_pass = k.narrow(D::Minus1, rot_dim, k.dim(D::Minus1)? - rot_dim)?;

        let all_same = seqlen_offsets.iter().all(|&x| x == seqlen_offsets[0]);
        let (q_rot, k_rot) = if all_same {
            let cos = cos.narrow(0, seqlen_offsets[0], seq_len)?;
            let sin = sin.narrow(0, seqlen_offsets[0], seq_len)?;
            let q_embed = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, &cos, &sin)?;
            (q_embed, k_embed)
        } else {
            let mut q_embeds = Vec::new();
            let mut k_embeds = Vec::new();
            for (i, offset) in seqlen_offsets.iter().enumerate() {
                let cos = cos.narrow(0, *offset, seq_len)?;
                let sin = sin.narrow(0, *offset, seq_len)?;
                let q_embed = candle_nn::rotary_emb::rope(
                    &q_rot.i(i)?.unsqueeze(0)?.contiguous()?,
                    &cos,
                    &sin,
                )?;
                let k_embed = candle_nn::rotary_emb::rope(
                    &k_rot.i(i)?.unsqueeze(0)?.contiguous()?,
                    &cos,
                    &sin,
                )?;
                q_embeds.push(q_embed);
                k_embeds.push(k_embed);
            }
            let q_rot = Tensor::cat(&q_embeds, 0)?;
            let k_rot = Tensor::cat(&k_embeds, 0)?;
            (q_rot, k_rot)
        };

        Ok((
            Tensor::cat(&[q_rot, q_pass], D::Minus1)?.contiguous()?,
            Tensor::cat(&[k_rot, k_pass], D::Minus1)?.contiguous()?,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct Gemma3RotaryEmbedding(RotaryEmbedding);

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Gemmma3ScaledRopeType {
    #[serde(alias = "linear")]
    Linear,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Gemma3RopeScalingConfig {
    factor: f64,
    rope_type: Gemmma3ScaledRopeType,
}

impl Gemma3RotaryEmbedding {
    fn new_linear(
        cfg: &Gemma3TextConfig,
        factor: f64,
        is_gpt_neox: bool,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let inv_freq = (inv_freq / factor)?;

        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        Ok(Self(RotaryEmbedding {
            cos,
            sin,
            is_gpt_neox,
        }))
    }

    pub fn new(
        is_gpt_neox: bool,
        dtype: DType,
        cfg: &Gemma3TextConfig,
        dev: &Device,
    ) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(Gemma3RopeScalingConfig {
                rope_type: Gemmma3ScaledRopeType::Linear,
                factor,
            }) => Self::new_linear(cfg, *factor, is_gpt_neox, dtype, dev),

            _ => Self::new_linear(cfg, 1.0, is_gpt_neox, dtype, dev),
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        self.0.forward(q, k, seqlen_offsets)
    }
}

#[derive(Debug, Clone)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
    dtype: DType,
}

impl QLinear {
    pub fn new<R: std::io::Read + std::io::Seek>(
        ct: &mut Content<'_, R>,
        name: &str,
        device: &Device,
    ) -> Result<Self> {
        let w = ct.tensor(&format!("{name}.weight"), device)?;
        let b = ct.tensor(&format!("{name}.bias"), device)?;
        let inner = QMatMul::from_qtensor(w)?;
        let bias = b.dequantize(device)?;
        Ok(Self {
            inner,
            bias: Some(bias),
            dtype: DType::F32,
        })
    }

    pub fn from_linear(linear: Linear) -> Self {
        Self {
            inner: QMatMul::Tensor(linear.weight().clone()),
            bias: linear.bias().cloned(),
            dtype: linear.weight().dtype(),
        }
    }

    pub fn from_parts(w: Tensor, b: Option<Tensor>) -> Self {
        let dtype = w.dtype();
        Self {
            inner: QMatMul::Tensor(w),
            bias: b,
            dtype,
        }
    }

    pub fn from_qparts(w: QTensor, b: Option<Tensor>) -> Self {
        if let Some(ref b) = b {
            assert_eq!(b.dtype(), DType::F32);
        }
        Self {
            inner: QMatMul::QTensor(Arc::new(w)),
            bias: b,
            dtype: DType::F32,
        }
    }

    pub fn from_old_and_qmatmul(inner: QMatMul, old: &Self) -> Self {
        Self {
            inner,
            bias: old.bias.clone(),
            dtype: old.dtype,
        }
    }

    pub fn inner(&mut self) -> &mut QMatMul {
        &mut self.inner
    }

    pub fn inner_ref(&self) -> &QMatMul {
        &self.inner
    }

    pub fn is_quant(&self) -> bool {
        matches!(self.inner, QMatMul::QTensor(_))
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn bias_mut(&mut self) -> Option<&mut Tensor> {
        self.bias.as_mut()
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if self.is_quant() {
            xs.to_dtype(DType::F32)?
        } else {
            xs.clone()
        };
        if let Some(bias) = &self.bias {
            self.inner
                .forward(&xs)?
                .broadcast_add(bias)?
                .to_dtype(self.dtype)
        } else {
            self.inner.forward(&xs)?.to_dtype(self.dtype)
        }
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    is_gpt_neox: bool,
}

impl RotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        device: &Device,
        is_gpt_neox: bool,
        dtype: DType,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            is_gpt_neox,
        })
    }

    pub fn new_partial(
        base: f32,
        rot_dim: usize,
        max_position_embeddings: usize,
        device: &Device,
        is_gpt_neox: bool,
        dtype: DType,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..rot_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / rot_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            is_gpt_neox,
        })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;

        let rope = if self.is_gpt_neox {
            candle_nn::rotary_emb::rope
        } else {
            candle_nn::rotary_emb::rope_i
        };

        let all_same = seqlen_offsets.iter().all(|&x| x == seqlen_offsets[0]);
        if all_same {
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

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    Silu,
    Sigmoid,
    HardSigmoid,
    Swiglu,
    Swish,
    HardSwish,
    Elu(f64),
    LeakyRelu(f64),
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
    QuickGelu,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu_erf(),
            // https://github.com/huggingface/transformers/blob/12f043eaeaabfef6f6efea411d98e6f6d3c094b7/src/transformers/activations.py#L49-L78
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Relu2 => xs.relu()?.sqr(),
            Self::Relu6 => xs.clamp(0f32, 6f32),
            Self::Silu => xs.silu(),
            Self::Sigmoid => candle_nn::ops::sigmoid(xs),
            Self::HardSigmoid => candle_nn::ops::hard_sigmoid(xs),
            Self::Swiglu => candle_nn::ops::swiglu(xs),
            Self::Swish => xs * candle_nn::ops::sigmoid(xs)?,
            Self::HardSwish => xs * candle_nn::ops::hard_sigmoid(xs)?,
            &Self::Elu(alpha) => xs.elu(alpha),
            &Self::LeakyRelu(negative_slope) => candle_nn::ops::leaky_relu(xs, negative_slope),
            Self::GeluPytorchTanh => xs.gelu(),
            Self::QuickGelu => xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?),
        }
    }
}

impl TryInto<candle_nn::Activation> for Activation {
    type Error = candle_core::Error;

    fn try_into(self) -> Result<candle_nn::Activation> {
        match self {
            Self::Gelu => Ok(candle_nn::Activation::Gelu),
            Self::Relu => Ok(candle_nn::Activation::Relu),
            Self::Silu => Ok(candle_nn::Activation::Silu),
            Self::NewGelu => Ok(candle_nn::Activation::NewGelu),
            Self::Relu2 => Ok(candle_nn::Activation::Relu2),
            Self::Relu6 => Ok(candle_nn::Activation::Relu6),
            Self::Sigmoid => Ok(candle_nn::Activation::Sigmoid),
            Self::HardSigmoid => Ok(candle_nn::Activation::HardSigmoid),
            Self::Swiglu => Ok(candle_nn::Activation::Swiglu),
            Self::Swish => Ok(candle_nn::Activation::Swish),
            Self::HardSwish => Ok(candle_nn::Activation::HardSwish),
            Self::Elu(x) => Ok(candle_nn::Activation::Elu(x)),
            Self::LeakyRelu(x) => Ok(candle_nn::Activation::LeakyRelu(x)),
            Self::GeluPytorchTanh => Ok(candle_nn::Activation::GeluPytorchTanh),
            Self::QuickGelu => candle_core::bail!("No mapping to candle_nn for QuickGelu"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

pub struct Conv3dNoBias {
    conv2d_1: Conv2d,
    conv2d_2: Conv2d,
}

impl Conv3dNoBias {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_sizes: [usize; 3],
        cfg: Conv3dConfig,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let ws = vb.get(
            (
                out_channels,
                in_channels / cfg.groups,
                kernel_sizes[0],
                kernel_sizes[1],
                kernel_sizes[2],
            ),
            "weight",
        )?;

        // Split on temporal dimension
        // https://github.com/pytorch/pytorch/issues/139066

        let w1 = ws.i((.., .., 0, .., ..))?;
        let w2 = ws.i((.., .., 1, .., ..))?;

        let cfg = Conv2dConfig {
            padding: cfg.padding,
            stride: cfg.stride,
            dilation: cfg.dilation,
            groups: cfg.groups,
        };

        Ok(Self {
            conv2d_1: Conv2d::new(w1.contiguous()?, None, cfg),
            conv2d_2: Conv2d::new(w2.contiguous()?, None, cfg),
        })
    }
}

impl Module for Conv3dNoBias {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs1 = xs.i((.., .., 0, .., ..))?;
        let xs2 = xs.i((.., .., 1, .., ..))?;

        (self.conv2d_1.forward(&xs1)? + self.conv2d_2.forward(&xs2)?)?.unsqueeze(2)
    }
}

pub trait TensorInfExtend {
    fn is_inf(&self) -> Result<Self>
    where
        Self: Sized;
    fn any(&self) -> Result<bool>;
}

impl TensorInfExtend for Tensor {
    fn is_inf(&self) -> Result<Self> {
        self.broadcast_eq(&Tensor::new(f32::INFINITY, self.device())?.to_dtype(self.dtype())?)
    }

    fn any(&self) -> Result<bool> {
        let sum = self.sum_all()?;
        match self.dtype() {
            DType::U8 => Ok(sum.to_scalar::<u8>()? == 0),
            DType::U32 => Ok(sum.to_scalar::<u32>()? == 0),
            DType::I16 => Ok(sum.to_scalar::<i16>()? == 0),
            DType::I32 => Ok(sum.to_scalar::<i32>()? == 0),
            DType::I64 => Ok(sum.to_scalar::<i64>()? == 0),
            DType::F16 => Ok(sum.to_scalar::<half::f16>()? == half::f16::from_f32_const(0.)),
            DType::BF16 => Ok(sum.to_scalar::<half::bf16>()? == half::bf16::from_f32_const(0.)),
            DType::F32 => Ok(sum.to_scalar::<f32>()? == 0.),
            DType::F64 => Ok(sum.to_scalar::<f64>()? == 0.),
            DType::F8E4M3 => Ok(sum.to_scalar::<F8E4M3>()? == F8E4M3::ZERO),
        }
    }
}

pub fn clamp_for_f16(xs: &Tensor) -> Result<Tensor> {
    let mut max = match xs.dtype() {
        DType::U8 => u8::MAX as f32 - 1000.,
        DType::U32 => u32::MAX as f32 - 1000.,
        DType::I16 => i16::MAX as f32 - 1000.,
        DType::I32 => i32::MAX as f32 - 1000.,
        DType::I64 => i64::MAX as f32 - 1000.,
        DType::F16 => half::f16::MAX.to_f32_const() - 1000.,
        DType::BF16 => half::bf16::MAX.to_f32_const() - 1000.,
        DType::F32 => f32::MAX - 1000.,
        DType::F64 => f64::MAX as f32 - 1000.,
        DType::F8E4M3 => F8E4M3::MAX.to_f32() - 1000.,
    };
    if xs.is_inf()?.any()? {
        max -= 1000.;
    }
    xs.clamp(-max, max)
}

pub struct FloatInfo {
    /// Minimum representable value.
    pub min: f64,
    /// Maximum representable value.
    pub max: f64,
    /// The difference between 1.0 and the next smallest representable float larger than 1.0.
    pub eps: f64,
    pub dtype: DType,
}

pub trait GetFloatInfo {
    fn finfo(&self) -> Result<FloatInfo>;
}

impl GetFloatInfo for DType {
    fn finfo(&self) -> Result<FloatInfo> {
        let finfo = match self {
            Self::BF16 => FloatInfo {
                min: bf16::MIN.to_f64(),
                max: bf16::MAX.to_f64(),
                eps: bf16::EPSILON.to_f64(),
                dtype: DType::BF16,
            },
            Self::F16 => FloatInfo {
                min: f16::MIN.to_f64(),
                max: f16::MAX.to_f64(),
                eps: f16::EPSILON.to_f64(),
                dtype: DType::F16,
            },
            Self::F32 => FloatInfo {
                min: f32::MIN as f64,
                max: f32::MAX as f64,
                eps: f32::EPSILON as f64,
                dtype: DType::F32,
            },
            Self::F64 => FloatInfo {
                min: f64::MIN,
                max: f64::MAX,
                eps: f64::EPSILON,
                dtype: DType::F64,
            },
            Self::F8E4M3 => FloatInfo {
                min: F8E4M3::MIN.to_f64(),
                max: F8E4M3::MAX.to_f64(),
                eps: F8E4M3::EPSILON.to_f64(),
                dtype: DType::F8E4M3,
            },
            other => {
                candle_core::bail!("Expected a float type for `GetFloatInfo`, got {other:?}");
            }
        };
        Ok(finfo)
    }
}

#[derive(Clone)]
pub struct Mlp {
    pub gate: Arc<dyn QuantMethod>,
    pub up: Arc<dyn QuantMethod>,
    pub down: Arc<dyn QuantMethod>,
    act: Activation,
    params: Vec<usize>,
}

impl Mlp {
    pub fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        quantization_config: &Option<QuantizedConfig>,
        hidden_act: Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            gate: ColumnParallelLayer::new(
                hidden_size,
                intermediate_size,
                quantization_config,
                false,
                comm,
                vb.pp("gate_proj"),
            )?,
            up: ColumnParallelLayer::new(
                hidden_size,
                intermediate_size,
                quantization_config,
                false,
                comm,
                vb.pp("up_proj"),
            )?,
            down: RowParallelLayer::new(
                intermediate_size,
                hidden_size,
                quantization_config,
                false,
                comm,
                vb.pp("down_proj"),
            )?,
            act: hidden_act,
            params: vec![hidden_size, intermediate_size],
        })
    }

    pub fn replicate(
        params: &[usize],
        vb: ShardedVarBuilder,
        act: Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Self::new(vb, params[0], params[1], &None, act, comm)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let lhs = self.gate.forward(&xs)?;
        let rhs = self.up.forward(&xs)?;
        let mut res = self.down.forward(&candle_nn::ops::mul_and_act(
            &lhs,
            &rhs,
            self.act.try_into()?,
        )?)?;
        if self.gate.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

impl AnyMoeTrainableLayer for Mlp {}

impl MlpLayer for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let lhs = MatMul.qmethod_matmul(&xs, &*self.gate)?;
        let rhs = MatMul.qmethod_matmul(&xs, &*self.up)?;
        let mut res = if matches!(
            self.act,
            Activation::Gelu | Activation::Silu | Activation::Relu
        ) {
            MatMul.qmethod_matmul(
                &candle_nn::ops::mul_and_act(&lhs, &rhs, self.act.try_into()?)?,
                &*self.down,
            )?
        } else {
            MatMul.qmethod_matmul(&(&lhs.apply(&self.act)? * &rhs)?, &*self.down)?
        };
        if self.gate.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.gate, &mut self.up, &mut self.down]
    }
    fn clone(&self) -> Box<dyn MlpLayer> {
        Box::new(Clone::clone(self))
    }
    fn get_params(&self) -> &[usize] {
        &self.params
    }
    fn hidden_act(&self) -> Activation {
        self.act
    }
    // gate, up, down
    fn new_added_delta(&self, deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        let gate = if let Some(ref delta) = deltas[0] {
            self.gate.add_delta_w(delta)?
        } else {
            self.gate.clone()
        };
        let up = if let Some(ref delta) = deltas[1] {
            self.up.add_delta_w(delta)?
        } else {
            self.up.clone()
        };
        let down = if let Some(ref delta) = deltas[2] {
            self.down.add_delta_w(delta)?
        } else {
            self.down.clone()
        };

        Ok(Box::new(Self {
            gate,
            up,
            down,
            act: self.act,
            params: self.params.clone(),
        }))
    }

    fn dtype_device(&self) -> (DType, Device) {
        self.gate.dtype_and_device()
    }
}

pub struct AvgPool2d {
    kernel_size: usize,
    stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.avg_pool2d_with_stride(self.kernel_size, self.stride)
    }
}

/// Applies 2D reflection padding to a tensor of shape (N, C, H, W).
///
/// The `padding` argument is a 4-tuple (pad_left, pad_right, pad_top, pad_bottom).
/// For left padding, it reflects the values from column 1 up to pad_left (in reverse order);
/// for right padding, it reflects from the second-to-last column backwards, and similarly for
/// vertical (height) padding.
pub struct ReflectionPad2d {
    padding: (usize, usize, usize, usize),
}

impl ReflectionPad2d {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self { padding }
    }
}

impl Module for ReflectionPad2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (pad_left, pad_right, pad_top, pad_bottom) = self.padding;

        let (_n, _c, h, w) = xs.dims4()?;

        // --- Horizontal Padding (along width, axis = 3) ---
        // For left padding, we reflect columns 1..=pad_left (in reverse order).
        let left_pad = if pad_left > 0 {
            // Create indices: [pad_left, pad_left-1, ..., 1]
            let indices: Vec<i64> = (1..=pad_left as i64).rev().collect();
            Some(xs.index_select(&Tensor::new(indices, &Device::Cpu)?, 3)?)
        } else {
            None
        };

        // For right padding, we reflect from the right side (excluding the last column).
        let right_pad = if pad_right > 0 {
            // For pad_right == 2, generate indices: [w-2, w-3, ... , w-1-pad_right]
            let start = w as i64 - 2;
            let indices: Vec<i64> = (0..pad_right as i64).map(|i| start - i).collect();
            Some(xs.index_select(&Tensor::new(indices, &Device::Cpu)?, 3)?)
        } else {
            None
        };

        // Concatenate horizontally (along width, dim=3)
        let x_padded_width = match (left_pad, right_pad) {
            (Some(l), Some(r)) => Tensor::cat(&[l, xs.clone(), r], 3)?,
            (Some(l), None) => Tensor::cat(&[l, xs.clone()], 3)?,
            (None, Some(r)) => Tensor::cat(&[xs.clone(), r], 3)?,
            (None, None) => xs.clone(),
        };

        // --- Vertical Padding (along height, axis = 2) ---
        // For top padding, reflect rows 1..=pad_top (in reverse order)
        let top_pad = if pad_top > 0 {
            let indices: Vec<i64> = (1..=pad_top as i64).rev().collect();
            Some(x_padded_width.index_select(&Tensor::new(indices, &Device::Cpu)?, 2)?)
        } else {
            None
        };

        // For bottom padding, reflect from the bottom (excluding the last row)
        let bottom_pad = if pad_bottom > 0 {
            let start = h as i64 - 2;
            let indices: Vec<i64> = (0..pad_bottom as i64).map(|i| start - i).collect();
            Some(x_padded_width.index_select(&Tensor::new(indices, &Device::Cpu)?, 2)?)
        } else {
            None
        };

        // Concatenate vertically (along height, dim=2)
        let x_padded = match (top_pad, bottom_pad) {
            (Some(t), Some(b)) => Tensor::cat(&[t, x_padded_width, b], 2)?,
            (Some(t), None) => Tensor::cat(&[t, x_padded_width], 2)?,
            (None, Some(b)) => Tensor::cat(&[x_padded_width, b], 2)?,
            (None, None) => x_padded_width,
        };

        Ok(x_padded)
    }
}

pub struct ScaledEmbedding {
    scale: f64,
    embedding: Embedding,
}

impl ScaledEmbedding {
    pub fn new(scale: f64, embedding: Embedding) -> Self {
        Self { scale, embedding }
    }

    pub fn embeddings(&self) -> &Tensor {
        self.embedding.embeddings()
    }
}

impl Module for ScaledEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.embedding)? * self.scale
    }
}
