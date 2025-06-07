use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::layers;

use super::config::ConformerEncoderConfig;

pub struct AbsolutePositionalEncoding {
    pe: Tensor,
    xscale: f64,
}

impl AbsolutePositionalEncoding {
    pub fn new(cfg: &ConformerEncoderConfig, device: &Device) -> Result<Self> {
        let max_len = 5000;

        let mut pe = Tensor::zeros(max_len, DType::F32, device)?;
        let position = Tensor::arange(0u32, max_len as u32, device)?.unsqueeze(1)?;

        let div_term = (Tensor::arange_step(0u32, cfg.attention_dim as u32, 2, device)?
            * -((10000f64).ln() / cfg.attention_dim as f64))?;

        let sin = (position.to_dtype(DType::F32)? * &div_term)?.sin()?;
        let cos = (position.to_dtype(DType::F32)? * &div_term)?.cos()?;

        // Interleave
        let sin_indices = Tensor::from_vec(
            (0..max_len).step_by(2).map(|x| x as u32).collect(),
            max_len / 2,
            device,
        )?;
        let cos_indices = Tensor::from_vec(
            (1..max_len).step_by(2).map(|x| x as u32).collect(),
            max_len / 2,
            device,
        )?;
        pe = pe.index_add(&sin_indices, &sin, D::Minus1)?;
        pe = pe.index_add(&cos_indices, &cos, D::Minus1)?;
        pe = pe.unsqueeze(0)?;

        Ok(Self {
            pe,
            xscale: (cfg.attention_dim as f64).sqrt(),
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.dim(1)? >= self.pe.dim(1)? {
            candle_core::bail!("Need to recompute positional embeds");
        }

        (xs * self.xscale)?.broadcast_mul(&self.pe.i((.., ..xs.dim(1)?))?)
    }
}

pub struct T5RelativeAttentionLogitBias {
    bias_values: Embedding,
    skip_bucketing: bool,
    max_distance: usize,
    symmetric: bool,
}

impl T5RelativeAttentionLogitBias {
    pub fn new(
        num_heads: usize,
        num_buckets: Option<usize>,
        max_distance: usize,
        symmetric: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let skip_bucketing = num_buckets.is_none();
        let mut num_buckets = num_buckets.unwrap_or(max_distance);
        if !symmetric {
            num_buckets *= 2;
        }

        Ok(Self {
            bias_values: layers::embedding(num_buckets, num_heads, vb.pp("bias_values"), &None)?,
            skip_bucketing,
            symmetric,
            max_distance,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let maxpos = x.dim(1)?;
        let device = x.device();

        // Create position matrices
        let context_position = Tensor::arange(0u32, maxpos as u32, device)?.unsqueeze(1)?;
        let memory_position = Tensor::arange(0u32, maxpos as u32, device)?.unsqueeze(0)?;

        // Calculate relative positions
        let relative_position = memory_position.broadcast_sub(&context_position)?;

        // Clip to max distance
        let max_dist = self.max_distance as i64;
        let relative_position = relative_position.clamp(-max_dist, max_dist - 1)?;

        // Map to bias indices
        let bias_idx = if self.skip_bucketing {
            relative_position
        } else {
            unimplemented!("require skip_bucketing");
        };

        let bias_idx = if self.symmetric {
            bias_idx.abs()?
        } else {
            (bias_idx + (self.bias_values.embeddings().dim(0)? as f64 / 2.))?
        };

        // Get bias values
        let t5_rel_att_bias = self.bias_values.forward(&bias_idx)?; // [L, L, H]
        t5_rel_att_bias.permute((2, 0, 1))?.unsqueeze(0) // [1, H, L, L]
    }
}
