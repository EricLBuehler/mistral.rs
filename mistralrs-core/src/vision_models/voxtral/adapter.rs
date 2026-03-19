#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::layers::MatMul;

/// Temporal adapter that performs 4x downsampling via reshape + MLP.
///
/// Input: [B, T, encoder_dim] (e.g., [B, T, 1280])
/// Reshape: [B, T/4, encoder_dim*4] (e.g., [B, T/4, 5120])
/// Output: [B, T/4, decoder_dim] (e.g., [B, T/4, 3072])
pub struct VoxtralTemporalAdapter {
    pub(super) w_in: Arc<dyn QuantMethod>,
    pub(super) w_out: Arc<dyn QuantMethod>,
    downsample_factor: usize,
}

impl VoxtralTemporalAdapter {
    pub fn new(
        encoder_dim: usize,
        decoder_dim: usize,
        downsample_factor: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let in_features = encoder_dim * downsample_factor;
        let w_in = mistralrs_quant::linear_b(
            in_features,
            decoder_dim,
            false,
            &None,
            vb.pp("audio_language_projection").pp("0"),
        )?;
        let w_out = mistralrs_quant::linear_b(
            decoder_dim,
            decoder_dim,
            false,
            &None,
            vb.pp("audio_language_projection").pp("2"),
        )?;
        Ok(Self {
            w_in,
            w_out,
            downsample_factor,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        // Truncate to nearest multiple of downsample_factor
        let t_trunc = t - (t % self.downsample_factor);
        let xs = if t_trunc < t {
            xs.narrow(1, 0, t_trunc)?
        } else {
            xs.clone()
        };
        let t_new = t_trunc / self.downsample_factor;
        // Reshape: [B, T_trunc, D] -> [B, T_trunc/factor, D*factor]
        let xs = xs.reshape((b, t_new, d * self.downsample_factor))?;
        // MLP: Linear -> GELU -> Linear
        let xs = MatMul.qmethod_matmul(&xs, &*self.w_in)?;
        let xs = xs.gelu_erf()?;
        MatMul.qmethod_matmul(&xs, &*self.w_out)
    }
}
