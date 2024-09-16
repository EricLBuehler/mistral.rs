use candle_core::{DType, Device, Result, Tensor};

use crate::hqq::optimize::OptResults;

use super::{optimize::OptParams, HqqAxis, HqqConfig, HqqLayer};

impl HqqLayer {
    /// Quantize the model into HQQ
    pub fn quantize(input: &Tensor, device: &Device, cfg: HqqConfig) -> Result<Self> {
        let group_size: usize = cfg.group_size.into();
        if input.elem_count() % group_size != 0 {
            candle_core::bail!("`group_size` should be divisible by the tensor number of elements, which are {}, got a group size of {group_size}.", input.elem_count());
        }

        let mut w = input.clone().to_dtype(DType::F32)?;

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
                max = max.max(0)?;
            }
            (min, max)
        } else {
            (
                w.min_keepdim(cfg.axis as usize)?,
                w.max_keepdim(cfg.axis as usize)?,
            )
        };

        let max_v = (2f64.powf(cfg.bits as usize as f64) - 1.).round();

        // Note: here using the inverse of the scale to avoid division, quantize via W * scale + zero, scale is inverted later!
        // Clamp to avoid half precision problems
        let scale = (max_v / (max - &min)?)?.clamp(0., 2e4)?;
        let mut zero = (min.neg()? * &scale)?;

        if cfg.round_zeros {
            zero = zero.round()?;
        }

        // We only support using optimization!
        /*let (quant_w, scale, zero) = if let Some(optimization_steps) = cfg.optimization_steps {
            let result = Self::optimize_weights_proximal_legacy(
                &w,
                &scale,
                zero,
                0.,
                max_v,
                cfg.axis,
                OptParams::default(optimization_steps),
            )?;
            (result.wq, result.scale, result.zero)
        } else {
            (
                w.broadcast_mul(&scale)?
                    .broadcast_add(&zero)?
                    .clamp(0., max_v)?,
                scale,
                zero,
            )
        };*/
        let OptResults { wq, scale, zero } = Self::optimize_weights_proximal_legacy(
            &w,
            &scale,
            zero,
            0.,
            max_v,
            cfg.axis,
            OptParams::default(cfg.optimization_steps),
        )?;

        let quant_w = cfg.bits.bitpack_type()(wq)?.to_device(device)?;

        let this = Self {
            w_q: quant_w,
            zeros: zero.to_device(device)?,
            scales: (1.0 / scale)?.to_device(device)?,
            bias: None,
            w_shape: input.shape().clone(),
            cfg,
        };
        Ok(this)
    }
}

mod test {
    #[cfg(all(feature = "cuda", test))]
    use candle_core::{Device, Result, Tensor};

    #[cfg(all(feature = "cuda", test))]
    #[test]
    fn test_quantize_hqq() -> Result<()> {
        use candle_core::DType;

        use crate::{HqqAxis, HqqBits, HqqConfig, HqqLayer};

        let dev = Device::new_cuda(0)?;
        let data = Tensor::rand(0., 1., (3072, 3072), &dev)?.to_dtype(DType::F32)?;
        let hqq = HqqLayer::quantize(
            &data,
            &dev,
            HqqConfig {
                bits: HqqBits::Two,
                group_size: 64.try_into()?,
                axis: HqqAxis::Zero,
                optimization_steps: None,
                round_zeros: false,
                channel_wise: true,
            },
        )?;

        let dequant = hqq.dequantize()?;

        dbg!(&(&dequant - &data)?.abs()?.mean_all()?);
        Ok(())
    }
}
