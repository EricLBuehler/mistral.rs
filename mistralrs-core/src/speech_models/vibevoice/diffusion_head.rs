//! Diffusion Head for generating acoustic latents from LM hidden states.
//!
//! Uses adaptive layer normalization (AdaLN) for conditioning on:
//! - Timestep embeddings
//! - LM hidden states (conditioning signal)
//!
//! The head iteratively denoises random Gaussian noise to produce acoustic latents.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::ShardedVarBuilder;

use super::config::DiffusionHeadConfig;

/// Helper to create a linear layer without bias from ShardedVarBuilder
fn linear_no_bias(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

/// RMS Normalization layer
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        xs.broadcast_mul(&self.weight)
    }
}

/// Timestep Embedder using sinusoidal positional encoding
struct TimestepEmbedder {
    mlp_0: Linear,
    mlp_2: Linear,
    dim: usize,
}

impl TimestepEmbedder {
    fn new(hidden_size: usize, vb: ShardedVarBuilder) -> Result<Self> {
        // Frequency embedding dimension (typically 256)
        let freq_dim = 256;
        let mlp_0 = linear_no_bias(freq_dim, hidden_size, vb.pp("mlp").pp("0"))?;
        let mlp_2 = linear_no_bias(hidden_size, hidden_size, vb.pp("mlp").pp("2"))?;
        Ok(Self {
            mlp_0,
            mlp_2,
            dim: freq_dim,
        })
    }

    /// Create sinusoidal timestep embeddings
    fn timestep_embedding(
        &self,
        timesteps: &Tensor,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let half_dim = self.dim / 2;
        let emb_scale = -(10000.0_f64.ln()) / (half_dim as f64);

        // Create frequency basis (keep as F32 for sin/cos computation)
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| (emb_scale * i as f64).exp() as f32)
            .collect();
        let freqs = Tensor::from_vec(freqs, half_dim, device)?;

        // timesteps: (batch,) -> (batch, 1)
        // freqs: (half_dim,) -> (1, half_dim)
        // args: (batch, half_dim)
        // Keep in F32 for accurate sin/cos computation
        let args = timesteps
            .to_dtype(DType::F32)?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&freqs.unsqueeze(0)?)?;

        // Concatenate sin and cos embeddings, then convert to target dtype
        let sin_emb = args.sin()?;
        let cos_emb = args.cos()?;
        Tensor::cat(&[&sin_emb, &cos_emb], D::Minus1)?.to_dtype(dtype)
    }

    fn forward(&self, timesteps: &Tensor, device: &Device, dtype: DType) -> Result<Tensor> {
        let emb = self.timestep_embedding(timesteps, device, dtype)?;
        let emb = self.mlp_0.forward(&emb)?;
        let emb = emb.silu()?;
        self.mlp_2.forward(&emb)
    }
}

/// SwiGLU Feed-Forward Network
struct SwiGluFfn {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGluFfn {
    fn new(hidden_size: usize, intermediate_size: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Diffusion Head Layer with AdaLN modulation
struct HeadLayer {
    norm: RmsNorm,
    ada_ln_modulation: Linear,
    ffn: SwiGluFfn,
}

impl HeadLayer {
    fn new(hidden_size: usize, ffn_ratio: f32, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let intermediate_size = (hidden_size as f32 * ffn_ratio) as usize;

        let norm = RmsNorm::new(hidden_size, eps, vb.pp("norm"))?;

        // AdaLN modulation outputs: [shift, scale, gate] for FFN = 3 * hidden_size
        let ada_ln_modulation = linear_no_bias(
            hidden_size,
            3 * hidden_size,
            vb.pp("adaLN_modulation").pp("1"),
        )?;

        let ffn = SwiGluFfn::new(hidden_size, intermediate_size, vb.pp("ffn"))?;

        Ok(Self {
            norm,
            ada_ln_modulation,
            ffn,
        })
    }

    fn forward(&self, xs: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let hidden_size = xs.dim(D::Minus1)?;

        // Get modulation parameters from conditioning
        let modulation = cond.silu()?.apply(&self.ada_ln_modulation)?;
        let shift = modulation.narrow(D::Minus1, 0, hidden_size)?;
        let scale = modulation.narrow(D::Minus1, hidden_size, hidden_size)?;
        let gate = modulation.narrow(D::Minus1, 2 * hidden_size, hidden_size)?;

        // Apply AdaLN: x * (1 + scale) + shift
        let xs_norm = self.norm.forward(xs)?;
        let xs_modulated = xs_norm
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;

        // FFN with gating
        let ffn_out = self.ffn.forward(&xs_modulated)?;
        let ffn_gated = ffn_out.broadcast_mul(&gate)?;

        // Residual connection
        xs + ffn_gated
    }
}

/// Final layer with different modulation (no gating, direct output)
/// Note: Unlike HeadLayer, FinalLayer doesn't have a separate norm - it applies
/// AdaLN modulation directly to the input.
struct FinalLayer {
    ada_ln_modulation: Linear,
    linear: Linear,
}

impl FinalLayer {
    fn new(hidden_size: usize, latent_size: usize, vb: ShardedVarBuilder) -> Result<Self> {
        // Final layer only needs shift and scale (2 * hidden_size)
        let ada_ln_modulation = linear_no_bias(
            hidden_size,
            2 * hidden_size,
            vb.pp("adaLN_modulation").pp("1"),
        )?;

        let linear = linear_no_bias(hidden_size, latent_size, vb.pp("linear"))?;

        Ok(Self {
            ada_ln_modulation,
            linear,
        })
    }

    fn forward(&self, xs: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let hidden_size = xs.dim(D::Minus1)?;

        // Get modulation parameters from conditioning
        let modulation = cond.silu()?.apply(&self.ada_ln_modulation)?;
        let shift = modulation.narrow(D::Minus1, 0, hidden_size)?;
        let scale = modulation.narrow(D::Minus1, hidden_size, hidden_size)?;

        // Apply AdaLN modulation directly (no separate norm layer)
        // x_modulated = x * (1 + scale) + shift
        let xs_modulated = xs.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;

        // Project to latent space
        self.linear.forward(&xs_modulated)
    }
}

/// Diffusion Head
///
/// Generates acoustic latents from LM hidden states using diffusion.
/// Architecture:
/// - noisy_images_proj: Projects noisy latents to hidden size
/// - t_embedder: Timestep embedding
/// - cond_proj: Projects conditioning (LM hidden states)
/// - layers: Stack of HeadLayers with AdaLN
/// - final_layer: Output projection to latent space
pub struct DiffusionHead {
    noisy_images_proj: Linear,
    t_embedder: TimestepEmbedder,
    cond_proj: Linear,
    layers: Vec<HeadLayer>,
    final_layer: FinalLayer,
    hidden_size: usize,
    latent_size: usize,
}

impl DiffusionHead {
    pub fn new(cfg: &DiffusionHeadConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let latent_size = cfg.latent_size;

        // Input projection for noisy latents
        let noisy_images_proj =
            linear_no_bias(latent_size, hidden_size, vb.pp("noisy_images_proj"))?;

        // Timestep embedder
        let t_embedder = TimestepEmbedder::new(hidden_size, vb.pp("t_embedder"))?;

        // Condition projection
        let cond_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("cond_proj"))?;

        // Stack of head layers
        let mut layers = Vec::new();
        for i in 0..cfg.head_layers {
            let layer = HeadLayer::new(
                hidden_size,
                cfg.head_ffn_ratio,
                cfg.rms_norm_eps,
                vb.pp("layers").pp(i),
            )?;
            layers.push(layer);
        }

        // Final output layer
        let final_layer = FinalLayer::new(hidden_size, latent_size, vb.pp("final_layer"))?;

        Ok(Self {
            noisy_images_proj,
            t_embedder,
            cond_proj,
            layers,
            final_layer,
            hidden_size,
            latent_size,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `noisy_latents` - Noisy latent codes, shape (batch, seq_len, latent_size)
    /// * `timesteps` - Diffusion timesteps, shape (batch,)
    /// * `condition` - LM hidden states, shape (batch, seq_len, hidden_size)
    ///
    /// # Returns
    /// * Predicted noise or velocity, shape (batch, seq_len, latent_size)
    pub fn forward(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        condition: &Tensor,
    ) -> Result<Tensor> {
        let device = noisy_latents.device();
        let dtype = noisy_latents.dtype();

        // Project noisy latents
        let xs = self.noisy_images_proj.forward(noisy_latents)?;

        // Get timestep embedding
        let t_emb = self.t_embedder.forward(timesteps, device, dtype)?;

        // Project condition
        let cond = self.cond_proj.forward(condition)?;

        // Combine timestep embedding with condition
        // t_emb: (batch, hidden_size) -> (batch, 1, hidden_size)
        // cond: (batch, seq_len, hidden_size)
        let combined_cond = cond.broadcast_add(&t_emb.unsqueeze(1)?)?;

        // Add projected noisy latents to combined condition for initial hidden state
        let mut xs = (xs + &combined_cond)?;

        // Pass through layers
        for layer in &self.layers {
            xs = layer.forward(&xs, &combined_cond)?;
        }

        // Final output
        self.final_layer.forward(&xs, &combined_cond)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn latent_size(&self) -> usize {
        self.latent_size
    }
}
