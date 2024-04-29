#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{ops::Mul, str::FromStr};

use candle_core::{quantized::QTensor, DType, Device, Result, Tensor};
use candle_nn::{
    layer_norm::{RmsNormNonQuantized, RmsNormQuantized},
    Module, VarBuilder,
};

use crate::models::phi3;

#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: candle_nn::RmsNorm<RmsNormNonQuantized>,
    eps: f64,
    weight: Tensor,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::rms_norm_non_quant(size, eps, vb)?;
        let w = inner.inner().weight().clone();
        Ok(Self {
            inner,
            eps,
            weight: w,
        })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.device().is_cpu() {
            // Handle device mapping case
            return candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32);
        }
        self.inner.forward(x)
    }
}

#[derive(Debug, Clone)]
pub struct QRmsNorm {
    inner: candle_nn::RmsNorm<RmsNormQuantized>,
}

impl QRmsNorm {
    pub fn new(scale: QTensor, eps: f32) -> Result<Self> {
        let scale = scale.dequantize(&scale.device())?;
        let inner = candle_nn::RmsNorm::<RmsNormQuantized>::new(scale, eps as f64);
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
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

#[derive(Debug, Clone)]
enum ScaledRopeType {
    Su,
    Yarn,
}

impl FromStr for ScaledRopeType {
    type Err = candle_core::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "su" => Ok(Self::Su),
            "yarn" => Ok(Self::Yarn),
            _ => Err(candle_core::Error::Msg(
                "Expected either `su` or `yarn` scaled RoPE type.".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
struct ScaledRopeParams {
    short_factor: Vec<f32>,
    long_factor: Vec<f32>,
    scaling_type: ScaledRopeType,
}

impl PhiRotaryEmbedding {
    pub fn new(dtype: DType, cfg: &phi3::Config, dev: &Device) -> Result<Self> {
        let scaled_params = cfg.rope_scaling.as_ref().map(|r| ScaledRopeParams {
            short_factor: r["short_factor"].clone().left().unwrap(),
            long_factor: r["long_factor"].clone().left().unwrap(),
            scaling_type: r["type"].clone().right().unwrap().parse().unwrap(),
        });
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.head_dim();

        if let Some(scaled_params) = scaled_params {
            // Calculate scale
            let scale =
                cfg.max_position_embeddings as f64 / cfg.original_max_position_embeddings as f64;
            let scaling_factor = if scale <= 1.0 {
                1.0
            } else {
                match scaled_params.scaling_type {
                    ScaledRopeType::Su => (1.0
                        + scale.ln() / (cfg.original_max_position_embeddings as f64).ln())
                    .sqrt(),
                    ScaledRopeType::Yarn => 0.1 * scale.ln() + 1.0,
                }
            };

            // Calculate inv freqs for short, long
            let inv_freq_long: Vec<_> = (0..dim)
                .step_by(2)
                .enumerate()
                .map(|(k, i)| {
                    1f32 / (scaled_params.long_factor[k]
                        * cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
                })
                .collect();
            let inv_freq_short: Vec<_> = (0..dim)
                .step_by(2)
                .enumerate()
                .map(|(k, i)| {
                    1f32 / (scaled_params.short_factor[k]
                        * cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
                })
                .collect();
            let inv_freq_len = inv_freq_long.len();

            let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
                .to_dtype(dtype)?
                .reshape((max_seq_len, 1))?;

            // Calculate sin,cos for long
            let inv_freq_long =
                Tensor::from_vec(inv_freq_long, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
            let freqs_long = t.matmul(&inv_freq_long)?;
            let long_sin = freqs_long.sin()?.mul(scaling_factor)?;
            let long_cos = freqs_long.cos()?.mul(scaling_factor)?;

            // Calculate sin,cos for short
            let inv_freq_short =
                Tensor::from_vec(inv_freq_short, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
            let freqs_short = t.matmul(&inv_freq_short)?;
            let short_sin = freqs_short.sin()?.mul(scaling_factor)?;
            let short_cos = freqs_short.cos()?.mul(scaling_factor)?;

            Ok(Self {
                short_cos,
                short_sin,
                long_cos: Some(long_cos),
                long_sin: Some(long_sin),
                original_max_position_embeddings: cfg.original_max_position_embeddings,
            })
        } else {
            let inv_freq: Vec<_> = (0..dim)
                .step_by(2)
                .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
                .collect();
            let inv_freq_len = inv_freq.len();
            let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
            let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
                .to_dtype(dtype)?
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
    }

    /// Returns (sin, cos) taking into account LongRope
    fn get_long_or_short_sin_cos(&self, seqlen_offsets: &[usize]) -> (&Tensor, &Tensor) {
        if self.long_cos.is_none() {
            return (&self.short_sin, &self.short_cos);
        }
        let seq_len = seqlen_offsets.iter().max().unwrap() + 1;
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
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        let (sin, cos) = self.get_long_or_short_sin_cos(seqlen_offsets);
        for offset in seqlen_offsets {
            let cos = cos.narrow(0, *offset, seq_len)?;
            let sin = sin.narrow(0, *offset, seq_len)?;
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            q_embeds.push(q_embed);
            k_embeds.push(k_embed);
        }
        Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
    }
}
