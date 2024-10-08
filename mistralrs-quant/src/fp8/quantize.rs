use candle_core::{DType, Result, Tensor};
use float8::F8E4M3;

use super::FP8Linear;

pub(super) struct QuantizationResult {
    /// Quantized tensor (f8)
    pub(super) qw: Tensor,
    /// Scalar, f32 tensor.
    ///
    /// Convert unquantized to quantized tensor as follows:
    /// `q = x * qs`
    pub(super) quantize_scale: Tensor,
    /// Scalar, f32 tensor. Reciprocal of `quantize_scale`.
    ///
    /// Convert unquantized to quantized tensor as follows:
    /// `x = q * dqs`
    pub(super) dequantize_scale: Tensor,
}

impl FP8Linear {
    /// Returns (quantized f8e4m3, scale)
    pub(super) fn quantize(data: &Tensor, dtype: DType) -> Result<QuantizationResult> {
        let data = data.to_dtype(DType::F32)?;
        let mut absmax = data.clone();
        while !absmax.dims().is_empty() {
            absmax = absmax.max(0)?;
        }
        let max_v = F8E4M3::MAX.to_f64().round();
        let scale = (max_v / absmax)?.clamp(1e-12, f64::INFINITY)?;
        let qw = data.broadcast_mul(&scale)?.to_dtype(dtype)?;
        Ok(QuantizationResult {
            qw,
            quantize_scale: scale.clone(),
            dequantize_scale: scale.recip()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use crate::fp8::FP8Linear;

    use super::QuantizationResult;

    #[test]
    fn test_roundtrip_f8e4m3() -> Result<()> {
        let dev = Device::new_cuda(0)?;

        let data = Tensor::rand(0., 1., (32, 32), &dev)?.to_dtype(DType::F32)?;

        let QuantizationResult {
            qw,
            quantize_scale: _,
            dequantize_scale,
        } = FP8Linear::quantize(&data, DType::F8E4M3)?;

        let dequant = qw.broadcast_mul(&dequantize_scale)?;

        let _diff = (&data - dequant)?.abs()?.mean_all()?;
        Ok(())
    }
}
