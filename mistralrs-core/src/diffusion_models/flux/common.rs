//! Shared utilities for FLUX.1 and FLUX.2 models.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::LazyLock;

use candle_core::{DType, Device, Result, Tensor, D};

use crate::layers::MatMul;
use crate::pipeline::text_models_inputs_processor::FlashParams;

/// Shared FlashParams for diffusion attention (bidirectional, non-causal).
///
/// Avoids allocating empty HashMaps on every attention call (~57 blocks × 20-50 steps).
pub(super) static DIFFUSION_FLASH_PARAMS: LazyLock<FlashParams> = LazyLock::new(|| FlashParams {
    causal: false,
    cumulative_seqlens_q: HashMap::new(),
    cumulative_seqlens_k: HashMap::new(),
    max_q: 0,
    max_k: 0,
});

/// Precompute the inv_freq tensor for a given RoPE dimension and theta.
///
/// This should be called once at model construction time and the result stored,
/// avoiding a CPU-to-GPU sync on every denoising step.
pub(super) fn precompute_inv_freq(dim: usize, theta: usize, device: &Device) -> Result<Tensor> {
    if dim % 2 == 1 {
        candle_core::bail!("dim {dim} is odd")
    }
    let theta = theta as f64;
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();
    Tensor::from_vec(inv_freq, (1, 1, inv_freq_len), device)
}

/// Compute rotary position embeddings using a precomputed inv_freq tensor.
pub(super) fn rope_with_inv_freq(pos: &Tensor, inv_freq: &Tensor) -> Result<Tensor> {
    let inv_freq = inv_freq.to_dtype(pos.dtype())?;
    let freqs = pos.unsqueeze(2)?.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let out = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 3)?;
    let (b, n, d, _ij) = out.dims4()?;
    out.reshape((b, n, d, 2, 2))
}

/// Precomputed frequency tensor for sinusoidal timestep embeddings.
///
/// Avoids repeated `Tensor::arange` + exp on GPU during the denoising loop.
#[derive(Debug, Clone)]
pub(super) struct TimestepFreqs {
    freqs: Tensor,
}

impl TimestepFreqs {
    pub fn new(dim: usize, device: &Device) -> Result<Self> {
        const MAX_PERIOD: f64 = 10000.;
        if dim % 2 == 1 {
            candle_core::bail!("{dim} is odd")
        }
        let half = dim / 2;
        let arange = Tensor::arange(0, half as u32, device)?.to_dtype(DType::F32)?;
        let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
        let freqs = freqs.unsqueeze(0)?;
        Ok(Self { freqs })
    }

    pub fn embed(&self, t: &Tensor, dtype: DType) -> Result<Tensor> {
        const TIME_FACTOR: f64 = 1000.;
        let t = (t * TIME_FACTOR)?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&self.freqs)?;
        Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)
    }
}

/// Simple scaled dot-product attention without batch dimension handling.
///
/// Used by the autoencoders (FLUX.1 and FLUX.2 VAE) where inputs are always 3D/4D
/// with a single head dimension.
pub(super) fn scaled_dot_product_attention_simple(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (MatMul.matmul(q, &k.t()?)? * scale_factor)?;
    MatMul.matmul(&candle_nn::ops::softmax_last_dim(&attn_weights)?, v)
}

/// Diagonal Gaussian distribution for VAE latent sampling.
#[derive(Debug, Clone)]
pub struct DiagonalGaussian {
    sample: bool,
    chunk_dim: usize,
}

impl DiagonalGaussian {
    pub fn new(sample: bool, chunk_dim: usize) -> Result<Self> {
        Ok(Self { sample, chunk_dim })
    }
}

impl candle_nn::Module for DiagonalGaussian {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let chunks = xs.chunk(2, self.chunk_dim)?;
        if self.sample {
            let std = (&chunks[1] * 0.5)?.exp()?;
            &chunks[0] + (std * chunks[0].randn_like(0., 1.))?
        } else {
            Ok(chunks[0].clone())
        }
    }
}
