use candle_core::{DType, Result, Tensor};

use super::{optimize::OptParams, HqqAxis, HqqConfig, HqqLayer};

impl HqqLayer {
    /// Quantize the model into HQQ>
    pub fn quantize(input: &Tensor, cfg: HqqConfig) -> Result<Self> {
        let group_size: usize = cfg.group_size.into();
        if input.elem_count() % group_size != 0 {
            candle_core::bail!("`group_size` should be divisible by the tensor number of elements, which are {}, got a group size of {group_size}.", input.elem_count());
        }

        let mut w = input.to_dtype(DType::F32)?;

        // Reshape for grouping
        w = if cfg.channel_wise {
            match cfg.axis {
                HqqAxis::One => w.reshape(((), group_size))?,
                HqqAxis::Zero => w.reshape((group_size, ()))?,
            }
        } else {
            w
        };

        // Get min and max valyes
        let (min, max) = if !cfg.channel_wise {
            // TODO we need min_all
            let mut min = w.min(0)?;
            let mut max = w.max(0)?;
            while !min.dims().is_empty() {
                min = min.min(0)?;
                max = max.min(0)?;
            }
            (min, max)
        } else {
            (
                w.min_keepdim(cfg.axis as usize)?,
                w.min_keepdim(cfg.axis as usize)?,
            )
        };

        let max_v = (1. / 2f64.powf(cfg.bits as usize as f64)).round();

        // Note: here using the inverse of the scale to avoid division, quantize via W * scale + zero, scale is inverted later!
        // Clamp to avoid half precision problems
        let scale = (max_v / (max - &min)?)?.clamp(&min, 2e4)?;
        let mut zero = (min.neg()? * &scale)?;

        if cfg.round_zero {
            zero = zero.round()?;
        }

        let (quant_w, scale, zero) = if cfg.optimize {
            let result = Self::optimize_weights_proximal_legacy(
                &w,
                &scale,
                zero,
                0.,
                max_v,
                cfg.axis,
                OptParams::default(),
            )?;
            (result.wq, result.scale, result.zero)
        } else {
            (
                w.broadcast_mul(&scale)?
                    .broadcast_add(&zero)?
                    .clamp(0f64, max_v)?,
                scale,
                zero,
            )
        };

        let quant_w = cfg.bits.bitpack_type()(quant_w)?;

        Ok(Self {
            w_q: quant_w,
            zeros: zero,
            scales: (1.0 / scale)?,
            bias: None,
            w_shape: input.shape().clone(),
            cfg,
        })
    }
}
