use candle_core::{DType, Device, IndexOp, Result, Tensor, D};

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
