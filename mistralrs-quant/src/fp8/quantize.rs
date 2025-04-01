use candle_core::{DType, Result, Tensor};
use candle_nn::Linear;
use float8::F8E4M3;
use half::bf16;

use super::FP8Linear;

pub(super) struct QuantizationResult {
    /// Quantized tensor (f8)
    pub(super) qw: Tensor,
    /// Scalar, f32 tensor. Reciprocal of `quantize_scale`.
    ///
    /// Convert unquantized to quantized tensor as follows:
    /// `x = q * dqs`
    pub(super) dequantize_scale: Tensor,
}

impl FP8Linear {
    pub(super) fn quantize(data: &Tensor, dtype: DType) -> Result<QuantizationResult> {
        let data = data.to_dtype(DType::BF16)?.reshape(((), 32))?;
        let amax = data.abs()?.max(1)?.to_dtype(DType::F32)?;

        let max_v = F8E4M3::MAX.to_f64();
        let scale = (max_v / amax)?;

        let to_cast = data.broadcast_mul(&scale.to_dtype(data.dtype())?.unsqueeze(1)?)?;
        let qw = if data.device().is_metal() {
            // Evil hack to allow metal shader to get the double value!
            let transmute_data = to_cast
                .flatten_all()?
                .to_vec1::<bf16>()?
                .into_iter()
                .map(|x| x.to_f64_const().to_bits() as i64)
                .collect::<Vec<_>>();
            Tensor::from_vec(transmute_data, data.shape(), data.device())?.to_dtype(dtype)?
        } else {
            to_cast.to_dtype(dtype)?
        };

        // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e8m0.html
        // This structure implements the datatype for handling 8-bit scale factors of e8m0 kind: interpreted as powers of two with biased exponent.
        // Bias equals to 127, so numbers 0 through 254 represent 2^-127 through 2^127. Number 0xFF = 255 is reserved for NaN.
        let iscale = scale.recip()?;
        let ue8m0_scale =
            ((iscale.log()? / (iscale.ones_like()? * 2.)?.log()?)? + 127.)?.to_dtype(DType::U8)?;
        Ok(QuantizationResult {
            qw,
            dequantize_scale: ue8m0_scale,
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
    fn test_quantize_f8e4m3() -> Result<()> {
        #[cfg(not(feature = "metal"))]
        let dev = Device::cuda_if_available(0)?;
        #[cfg(feature = "metal")]
        let dev = Device::new_metal(0)?;

        let data = Tensor::ones((8, 8), DType::F32, &dev)?;

        let QuantizationResult {
            qw,
            dequantize_scale,
        } = FP8Linear::quantize(&data, DType::F8E4M3)?;

        println!("{data}");
        println!("{qw}");
        println!("{dequantize_scale}");
        Ok(())
    }

    #[test]
    fn test_roundtrip_f8e4m3() -> Result<()> {
        #[cfg(not(feature = "metal"))]
        let dev = Device::cuda_if_available(0)?;
        #[cfg(feature = "metal")]
        let dev = Device::new_metal(0)?;

        let data = Tensor::rand(0f32, 1f32, (32, 32), &dev)?;

        let QuantizationResult {
            qw,
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
        maybe_init_cublas_lt_wrapper(x.device().clone());

        let handle = CUBLASLT_HANDLE.lock().unwrap().unwrap();

        let QuantizationResult {
            qw,
            dequantize_scale: dequant_a_scale,
        } = FP8Linear::quantize(&w, DType::F8E4M3)?;

        let mut dequant_b_scale = dequant_a_scale.clone();
        if !matches!(x.dtype(), DType::F8E4M3) {
            let QuantizationResult {
                qw,

                dequantize_scale,
            } = FP8Linear::quantize(&x, DType::F8E4M3)?;
            x = qw;
            dequant_b_scale = dequantize_scale;
        }

        let a = qw;
        let b = x;

        // FP8 quantized matmul
        let _res = handle.batch_matmul_fp8(
            &a,
            &b,
            &dequant_a_scale,
            &dequant_b_scale,
            None,
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
