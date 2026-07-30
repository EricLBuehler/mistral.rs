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
        let mut absmax = data.abs()?;
        while !absmax.dims().is_empty() {
            absmax = absmax.max(0)?;
        }

        let max_v = F8E4M3::MAX.to_f64();
        let scale = (max_v / absmax)?
            .clamp(F8E4M3::MIN.to_f32(), F8E4M3::MAX.to_f32())?
            .to_dtype(DType::F32)?;
        let to_cast = data.broadcast_mul(&scale.to_dtype(data.dtype())?)?;
        let qw = if dtype == DType::F8E4M3 {
            crate::scalar_fp8::ops::dtype_to_fp8(&to_cast)?
        } else {
            to_cast.to_dtype(dtype)?
        };
        Ok(QuantizationResult {
            qw,
            quantize_scale: scale.clone(),
            dequantize_scale: scale.recip()?,
        })
    }

    pub(super) fn dequantize(&self, dtype: DType) -> Result<Linear> {
        let weight = self.lin.weight();
        let weight = if weight.dtype() == DType::F8E4M3 && dtype != DType::F8E4M3 {
            crate::scalar_fp8::ops::fp8_to_dtype(weight, dtype)?
        } else {
            weight.to_dtype(dtype)?
        };
        let dequant_w = weight.broadcast_mul(&self.dequant_w_scale.to_dtype(dtype)?)?;
        Ok(Linear::new(dequant_w, self.lin.bias().cloned()))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "metal"))]
    use candle_core::{
        quantized::{GgmlDType, QTensor},
        DType, Device, Result, Tensor,
    };

    #[cfg(not(feature = "metal"))]
    use crate::{fp8::FP8Linear, QuantMethod, QuantMethodConfig};

    #[cfg(not(feature = "metal"))]
    use super::QuantizationResult;

    #[test]
    #[cfg(not(feature = "metal"))]
    fn test_roundtrip_f8e4m3() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;

        let data = Tensor::rand(0f32, 1f32, (32, 32), &dev)?;

        let QuantizationResult {
            qw,
            quantize_scale: _,
            dequantize_scale,
        } = FP8Linear::quantize(&data, DType::F8E4M3)?;

        let dequant = crate::scalar_fp8::ops::fp8_to_dtype(&qw, DType::F32)?
            .broadcast_mul(&dequantize_scale)?;

        let diff1 = (&data - dequant)?.abs()?.mean_all()?;

        println!("{diff1}");

        let q8_0 = QTensor::quantize(&data, GgmlDType::Q8_0)?.dequantize(&dev)?;
        let diff2 = (&data - q8_0)?.abs()?.mean_all()?;

        println!("{diff2}");
        Ok(())
    }

    #[test]
    #[cfg(not(feature = "metal"))]
    fn test_embedding_gathers_before_dequantizing() -> Result<()> {
        let data = (0..185)
            .map(|i| (i as f32 - 92.0) / 37.0)
            .collect::<Vec<_>>();
        let weight = Tensor::from_slice(&data, (5, 37), &Device::Cpu)?;
        let linear = FP8Linear::new(QuantMethodConfig::FP8 {
            lin: candle_nn::Linear::new(weight, None),
            dtype: DType::F8E4M3,
        })?;
        let ids = Tensor::new(&[[4u32, 1], [3, 1]], &Device::Cpu)?;
        let actual = linear.embedding_forward_raw(&ids)?;
        let expected = linear
            .dequantize_w()?
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((2, 2, 37))?;

        assert_eq!(actual.dims(), &[2, 2, 37]);
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cublaslt_matmul() -> Result<()> {
        use crate::cublaslt::{maybe_init_cublas_lt_wrapper, CUBLASLT_CONTROLLER};
        let dev = Device::new_cuda(0)?;

        // Use 128x128 matrices for FP8 tensor core compatibility across GPU architectures
        let w = Tensor::rand(0., 1., (1, 128, 128), &dev)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let mut x = Tensor::rand(0., 1., (1, 128, 128), &dev)?
            .to_dtype(DType::F32)?
            .contiguous()?;

        // Batch matrix multiplication
        maybe_init_cublas_lt_wrapper(x.device().clone());

        let handle = CUBLASLT_CONTROLLER.get_for_device(x.device()).unwrap();

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
        let _res = handle.batch_matmul_f8(
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
        )?;

        Ok(())
    }
}

#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use candle_core::{DType, Device, Result, Tensor};

    use crate::{fp8::FP8Linear, QuantMethod, QuantMethodConfig};

    #[test]
    fn test_metal_embedding_gathers_multiple_rows_before_dequantizing() -> Result<()> {
        let device = Device::new_metal(0)?;
        let data = (0..185)
            .map(|i| (i as f32 - 92.0) / 37.0)
            .collect::<Vec<_>>();
        let weight = Tensor::from_slice(&data, (5, 37), &device)?;
        let linear = FP8Linear::new(QuantMethodConfig::FP8 {
            lin: candle_nn::Linear::new(weight, None),
            dtype: DType::F8E4M3,
        })?;
        let ids = Tensor::new(&[[4u32, 1], [3, 1]], &device)?;
        let actual = linear.embedding_forward_raw(&ids)?;
        let expected = linear
            .dequantize_w()?
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((2, 2, 37))?;
        let max_diff = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let empty_ids = ids.narrow(1, 0, 0)?;
        let empty = linear.embedding_forward_raw(&empty_ids)?;

        assert!(max_diff <= 1e-6, "max_diff={max_diff}");
        assert_eq!(empty.dims(), &[2, 0, 37]);
        assert_eq!(empty.dtype(), DType::F32);
        Ok(())
    }
}
