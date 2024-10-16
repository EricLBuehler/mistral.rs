use candle_core::{DType, Result, Tensor};
use candle_nn::Linear;
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
    pub(super) fn quantize(data: &Tensor, dtype: DType) -> Result<QuantizationResult> {
        let data = data.to_dtype(DType::BF16)?;
        let mut absmax = data.clone();
        let mut absmin = data.clone();
        while !absmax.dims().is_empty() {
            absmax = absmax.max(0)?;
            absmin = absmin.min(0)?;
        }

        let absmax = absmax.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let absmin = absmin.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let amax = f32::max(absmax.abs(), absmin.abs());

        let max_v = F8E4M3::MAX.to_f32();
        let scale = (max_v / amax).clamp(F8E4M3::MIN.to_f32(), F8E4M3::MAX.to_f32());
        let scale = Tensor::new(scale, data.device())?;
        let qw = data
            .broadcast_mul(&scale.to_dtype(data.dtype())?)?
            .to_dtype(dtype)?;
        Ok(QuantizationResult {
            qw,
            quantize_scale: scale.clone(),
            dequantize_scale: scale.recip()?,
        })
    }

    pub(super) fn dequantize(&self, dtype: DType) -> Result<Linear> {
        let dequant_w = self
            .lin
            .weight()
            .to_dtype(dtype)?
            .broadcast_mul(&self.dequant_w_scale.to_dtype(dtype)?)?;
        Ok(Linear::new(dequant_w, self.lin.bias().cloned()))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{
        quantized::{GgmlDType, QTensor},
        DType, Device, Result, Tensor,
    };

    use crate::fp8::FP8Linear;

    use super::QuantizationResult;

    #[test]
    fn test_roundtrip_f8e4m3() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;

        let data = Tensor::rand(0., 1., (32, 32), &dev)?.to_dtype(DType::F32)?;

        let QuantizationResult {
            qw,
            quantize_scale: _,
            dequantize_scale,
        } = FP8Linear::quantize(&data, DType::F8E4M3)?;

        let dequant = qw.to_dtype(DType::F32)?.broadcast_mul(&dequantize_scale)?;

        let diff1 = (&data - dequant)?.abs()?.mean_all()?;

        println!("{diff1}");

        let q8_0 = QTensor::quantize(&data, GgmlDType::Q8_0)?.dequantize(&dev)?;
        let diff2 = (&data - q8_0)?.abs()?.mean_all()?;

        println!("{diff2}");
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cublaslt_matmul() -> Result<()> {
        use crate::cublaslt::{maybe_init_cublas_lt_wrapper, F8MatmulOutType, CUBLASLT_HANDLE};
        let dev = Device::new_cuda(0)?;

        let w = Tensor::rand(0., 1., (1, 16, 32), &dev)?.to_dtype(DType::F32)?;
        let mut x = Tensor::rand(0., 1., (1, 16, 32), &dev)?.to_dtype(DType::F32)?;

        // Batch matrix multiplication
        maybe_init_cublas_lt_wrapper();

        let handle = CUBLASLT_HANDLE.lock().unwrap().unwrap();

        let QuantizationResult {
            qw,
            quantize_scale: quant_scale,
            dequantize_scale: dequant_a_scale,
        } = FP8Linear::quantize(&w, DType::F8E4M3)?;

        let mut dequant_b_scale = dequant_a_scale.clone();
        if !matches!(x.dtype(), DType::F8E4M3) {
            let QuantizationResult {
                qw,
                quantize_scale: _,
                dequantize_scale,
            } = FP8Linear::quantize(&x, DType::F8E4M3)?;
            x = qw;
            dequant_b_scale = dequantize_scale;
        }

        let a = qw;
        let b = x;

        // FP8 quantized matmul
        let _res = handle.batch_matmul(
            &a,
            &b,
            &dequant_a_scale,
            &dequant_b_scale,
            &quant_scale,
            None,
            None,
            None,
            None,
            None,
            F8MatmulOutType::BF16,
        )?;

        Ok(())
    }
}
