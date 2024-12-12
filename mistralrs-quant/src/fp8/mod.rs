use std::{
    borrow::Cow,
    io::Cursor,
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module};
use quantize::QuantizationResult;

mod quantize;

use crate::{
    cublaslt::{maybe_init_cublas_lt_wrapper, F8MatmulOutType, CUBLASLT_HANDLE},
    utils::{
        deserialize_tensor, read_dtype, serialize_tensor, version_is_compatible, write_dtype,
        HQFF_VERSION,
    },
    IsqType, QuantMethod, QuantMethodConfig, QuantizedSerde, QuantizedSerdeType,
};

#[derive(Debug)]
pub struct FP8Linear {
    lin: Linear,
    dequant_w_scale: Tensor,
    dequant_x_scale: Tensor,
    quant_scale: Tensor,
    /// Quantized type
    dtype: DType,
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
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Bnb { .. } => unreachable!(),
            QuantMethodConfig::FP8 { lin, dtype } => {
                let QuantizationResult {
                    qw,
                    quantize_scale,
                    dequantize_scale,
                } = Self::quantize(lin.weight(), dtype)?;
                Ok(Self {
                    lin: Linear::new(qw, lin.bias().cloned()),
                    dequant_x_scale: dequantize_scale.clone(), // This is probably wrong!
                    dequant_w_scale: dequantize_scale,
                    quant_scale: quantize_scale,
                    dtype,
                })
            }
        }
    }
    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        Ok(self.dequantize(DType::F32)?.weight().clone())
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
                // Set up target shape
                let mut tgt_shape = x.dims().to_vec();
                *tgt_shape.last_mut().unwrap() = self.lin.weight().dim(0)?;

                // Flatten for correct dims
                let mut x = x.flatten_to(D::Minus(3))?;

                // Prepare the b tensor. If it is not quantized, quantize it
                let mut dequant_x_scale = self.dequant_x_scale.clone();
                if !matches!(x.dtype(), DType::F8E4M3) {
                    let QuantizationResult {
                        qw,
                        quantize_scale: _,
                        dequantize_scale,
                    } = Self::quantize(&x, DType::F8E4M3)?;
                    x = qw;
                    dequant_x_scale = dequantize_scale;
                }

                // Handle bias
                let beta = match self.lin.bias().is_some() {
                    true => Some(1.0),
                    false => None,
                };

                // Naming
                let a = self.lin.weight().unsqueeze(0)?;
                let b = x;

                handle
                    .batch_matmul_f8(
                        &a,
                        &b,
                        &self.dequant_w_scale,
                        &dequant_x_scale,
                        &self.quant_scale,
                        self.lin.bias(),
                        None,
                        beta,
                        None,
                        None,
                        F8MatmulOutType::BF16, // Output in bf16 to avoid manual dequant
                    )?
                    .reshape(tgt_shape)
            }
            None => {
                // Dequantize matmul
                let dequant_x = x.clone();
                let lin = self.dequantize(x.dtype())?;
                lin.forward(&dequant_x)
            }
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let dequant = self.dequantize(delta.dtype())?;
        let new = Linear::new((dequant.weight() + delta)?, dequant.bias().cloned());
        Ok(Arc::new(Self::new(QuantMethodConfig::FP8 {
            lin: new,
            dtype: self.dtype,
        })?))
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
        _imatrix_weight: Option<Vec<f32>>,
    ) -> Result<Arc<dyn QuantMethod>> {
        todo!()
    }

    fn get_max_isq_cpu_threads(&self, dtype: IsqType) -> Option<NonZeroUsize> {
        match dtype {
            IsqType::F8E4M3 => None,
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

    fn maybe_to_gguf_quant(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>> {
        Ok(self.clone())
    }
}

// Serialization structure:
//
// -----------------------
// HQFF version, u32, little endian
// -----------------------
// ISQ type (3 for fp8), u8, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Weight tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------
// Dequant W scalar, f32, little endian
// -----------------------
// Dequant X scalar, f32, little endian
// -----------------------
// Quant scalar, f32, little endian
// -----------------------
// Quantization type, u32, little endian
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
        let mut buffer = Vec::new();

        buffer.extend(&HQFF_VERSION.to_le_bytes());

        // ISQ type for fp8 is 3
        buffer.push(QuantizedSerdeType::Fp8 as u8);

        // Has bias
        buffer.push(self.lin.bias().is_some() as u8);

        // Weight
        serialize_tensor(&mut buffer, self.lin.weight())?;

        // Dequant a scale
        buffer.extend(self.dequant_w_scale.to_scalar::<f32>()?.to_le_bytes());
        // Dequant b scale
        buffer.extend(self.dequant_x_scale.to_scalar::<f32>()?.to_le_bytes());
        // Quant scale
        buffer.extend(self.quant_scale.to_scalar::<f32>()?.to_le_bytes());

        // DType
        write_dtype(self.dtype, &mut buffer);

        if let Some(bias) = self.lin.bias() {
            // Bias
            serialize_tensor(&mut buffer, bias)?;
        }

        Ok(Cow::from(buffer))
    }

    fn deserialize(data: Cow<[u8]>, device: &Device) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data.to_vec());

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Fp8 as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Fp8 as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let w = deserialize_tensor(&mut buffer, device)?;

        let dequant_w_scale = Tensor::new(buffer.read_f32::<LittleEndian>()?, device)?;
        let dequant_x_scale = Tensor::new(buffer.read_f32::<LittleEndian>()?, device)?;
        let quant_scale = Tensor::new(buffer.read_f32::<LittleEndian>()?, device)?;

        // DType
        let dtype = read_dtype(&mut buffer)?;

        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            lin: Linear::new(w, b),
            dequant_w_scale,
            dequant_x_scale,
            quant_scale,
            dtype,
        }))
    }
}
