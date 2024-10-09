use std::{
    borrow::Cow,
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module};
use quantize::QuantizationResult;

mod quantize;

use crate::{
    cublaslt::{maybe_init_cublas_lt_wrapper, CUBLASLT_HANDLE},
    IsqType, QuantMethod, QuantMethodConfig, QuantizedSerde,
};

#[derive(Debug)]
pub struct FP8Linear {
    lin: Linear,
    dequant_a_scale: Tensor,
    dequant_b_scale: Tensor,
    quant_scale: Tensor,
}

impl QuantMethod for FP8Linear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Gptq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::Unquantized(_) => unreachable!(),
            QuantMethodConfig::FP8 { lin, dtype } => {
                let QuantizationResult {
                    qw,
                    quantize_scale,
                    dequantize_scale,
                } = Self::quantize(lin.weight(), dtype)?;
                Ok(Self {
                    lin: Linear::new(qw, lin.bias().cloned()),
                    dequant_b_scale: dequantize_scale.clone(), // This is probably wrong!
                    dequant_a_scale: dequantize_scale,
                    quant_scale: quantize_scale,
                })
            }
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Batch matrix multiplication
        maybe_init_cublas_lt_wrapper();

        match *CUBLASLT_HANDLE.lock().unwrap() {
            Some(handle) => {
                let n_dims = x.dims().len();
                if n_dims < 3 {
                    candle_core::bail!(
                        "FP8Linear `matmul` via cuBLASlt expects `x` to have at least 3 dimensions"
                    );
                }
                let mut tgt_shape = x.dims().to_vec();
                *tgt_shape.last_mut().unwrap() = self.lin.weight().dim(0)?;

                let mut x = x.flatten_to(D::Minus(3))?;
                let mut dequant_b_scale = self.dequant_b_scale.clone();
                if !matches!(x.dtype(), DType::F8E4M3) {
                    let QuantizationResult {
                        qw,
                        quantize_scale: _,
                        dequantize_scale,
                    } = Self::quantize(&x, DType::F8E4M3)?;
                    x = qw;
                    dequant_b_scale = dequantize_scale;
                }

                // If attention_bias is set, we fuse the add by giving it as the output matrix
                // and setting beta to 1.0
                let beta = match self.lin.bias().is_some() {
                    true => Some(1.0),
                    false => None,
                };

                let a = self.lin.weight().unsqueeze(0)?;
                let b = x;

                dbg!(&b.to_dtype(DType::F32)?.mean_all()?);
                // FP8 quantized matmul
                let res = handle
                    .batch_matmul(
                        &a,
                        &b,
                        &self.dequant_a_scale,
                        &dequant_b_scale,
                        &self.quant_scale,
                        self.lin.bias(),
                        None,
                        beta,
                        None,
                        None,
                    )?
                    .reshape(tgt_shape)?;
                dbg!(&res.to_dtype(DType::F32)?.mean_all()?);
                Ok(res)
            }
            None => {
                // Dequantize matmul
                let dequant_w = self
                    .lin
                    .weight()
                    .to_dtype(DType::BF16)?
                    .broadcast_mul(&self.dequant_a_scale.to_dtype(DType::BF16)?)?;
                // let dequant_x = if matches!(x.dtype(), DType::F8E4M3) {
                //     x.to_dtype(DType::BF16)?.broadcast_mul(&self.dequant_b_scale.to_dtype(DType::BF16)?)?
                // } else {
                //     x.clone()
                // };
                let dequant_x = x.clone();
                let lin = Linear::new(dequant_w, self.lin.bias().cloned());
                let res = lin.forward(&dequant_x)?;
                res.broadcast_mul(&self.quant_scale.to_dtype(DType::BF16)?) //?.to_dtype(DType::F8E4M3)
            }
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        todo!()
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (DType::F8E4M3, self.lin.weight().device().clone())
    }

    fn get_bias_mut(&mut self) -> Option<&mut Tensor> {
        None
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
    ) -> Result<Arc<dyn QuantMethod>> {
        todo!()
    }

    fn get_max_isq_cpu_threads(&self, dtype: IsqType) -> Option<NonZeroUsize> {
        match dtype {
            IsqType::F8E4M3 => {
                todo!()
            }
            IsqType::Q2K
            | IsqType::Q3K
            | IsqType::Q4K
            | IsqType::Q4_0
            | IsqType::Q4_1
            | IsqType::Q5K
            | IsqType::Q5_0
            | IsqType::Q5_1
            | IsqType::Q6K
            | IsqType::Q8K
            | IsqType::Q8_0
            | IsqType::Q8_1
            | IsqType::HQQ4
            | IsqType::HQQ8 => None,
        }
    }
}

// Serialization structure:
//
// -----------------------
// HQFF version, u32, little endian
// -----------------------
// ISQ type (1 for unquantized), u8, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Weight tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------
// [OPTIONAL] Bias tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------

impl QuantizedSerde for FP8Linear {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "fp8-linear"
    }
    fn serialize(&self) -> Result<Cow<[u8]>> {
        todo!()
    }

    fn deserialize(_data: Cow<[u8]>, _device: &Device) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        todo!()
    }
}
