#[cfg(not(feature = "cuda"))]
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod ffi;

use std::{
    borrow::Cow,
    io::{Cursor, Read},
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{
    quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};
use candle_nn::Module;

use crate::{
    generate_isq, generate_isq_imatrix,
    utils::{deserialize_tensor, serialize_tensor, version_is_compatible, UQFF_VERSION},
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, QuantizedSerdeType,
};

#[derive(Debug)]
pub struct GgufMatMul {
    pub(crate) w: QMatMul,
    pub(crate) b: Option<Tensor>,
}

impl QuantMethod for GgufMatMul {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { q_weight, b } => Ok(Self {
                w: QMatMul::from_arc(q_weight)?,
                b,
            }),
            QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. } => unreachable!(),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.w.dequantize_f16()?.to_dtype(DType::F32)
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        let x = self.w.forward(a)?;
        if let Some(ref b) = self.b {
            x.broadcast_add(b)
        } else {
            Ok(x)
        }
    }

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    ///
    /// If `a` is (n_tokens, 1, cols), `self` weights are (n_experts, rows, cols),
    /// then the indices are (n_tokens, n_experts_per_tok).
    fn gather_forward(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        // Use indexed_moe_forward for efficient indexed matmul
        // Expected shapes:
        // - x: (n_tokens, 1, hidden_dim) or (n_tokens, n_experts_per_tok, hidden_dim)
        // - indices: (n_tokens, n_experts_per_tok)
        // - weights (self): (n_experts, out_features, in_features)
        #[cfg(feature = "cuda")]
        let res = cuda::qmatmul_indexed_moe_forward(&self.w, x, indices)?;

        // For CPU and Metal: use dequantize-then-matmul approach
        #[cfg(not(feature = "cuda"))]
        let res = cpu::cpu_indexed_moe_forward(&self.w, x, indices)?;

        if let Some(ref b) = self.b {
            res.broadcast_add(b)
        } else {
            Ok(res)
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        Some(DType::F32)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        match self {
            Self {
                w: QMatMul::Tensor(w),
                b,
            } => Ok(Arc::new(Self {
                w: QMatMul::Tensor((w + delta)?),
                b: b.clone(),
            })),
            Self {
                w: QMatMul::TensorF16(w),
                b,
            } => Ok(Arc::new(Self {
                w: QMatMul::TensorF16((w + delta)?),
                b: b.clone(),
            })),
            Self {
                w: QMatMul::QTensor(w),
                b,
            } => {
                let (w, dtype) = (w.dequantize(&w.device())?, w.dtype());
                let w = QMatMul::QTensor(std::sync::Arc::new(
                    candle_core::quantized::QTensor::quantize(&(w + delta)?, dtype)?,
                ));
                Ok(Arc::new(Self { w, b: b.clone() }))
            }
        }
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        match &self.w {
            QMatMul::QTensor(q) => (DType::F32, q.device()),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => (t.dtype(), t.device().clone()),
        }
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if let Some(dtype) = dtype {
            // F8Q8 is not a GgmlDType, so intercept before try_into()
            if dtype == IsqType::F8Q8 {
                let t = match &self.w {
                    QMatMul::QTensor(q) => q.dequantize(&q.device())?,
                    QMatMul::TensorF16(t) | QMatMul::Tensor(t) => t.clone(),
                };
                let t = t.to_device(&device)?;
                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(Arc::new(crate::F8Q8Linear::from_weight(
                    &t,
                    self.b.clone(),
                )?));
            }
            let t = match &self.w {
                QMatMul::QTensor(q) => q.dequantize(&q.device())?,
                QMatMul::TensorF16(t) | QMatMul::Tensor(t) => t.clone(),
            };
            let dtype = dtype.try_into()?;
            let res = if let Some(imatrix_weight) = imatrix_weight {
                generate_isq_imatrix!(t, imatrix_weight, device, dtype, n_quantized, guard)
            } else {
                generate_isq!(t, device, dtype, n_quantized, guard)
            };
            Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: res,
                b: self.b.clone(),
            })?))
        } else {
            let w = match &self.w {
                QMatMul::QTensor(q) => QMatMul::QTensor(Arc::new(QTensor::quantize(
                    &q.dequantize(&device)?,
                    q.dtype(),
                )?)),
                QMatMul::Tensor(t) => QMatMul::Tensor(t.to_device(&device)?),
                QMatMul::TensorF16(t) => QMatMul::TensorF16(t.to_device(&device)?),
            };
            let b = if let Some(b) = &self.b {
                Some(b.to_device(&device)?)
            } else {
                None
            };
            Ok(Arc::new(GgufMatMul { w, b }))
        }
    }
}

// Serialization structure:
//
// -----------------------
// UQFF version, u32, little endian
// -----------------------
// ISQ type (0 for GGUF), u8, little endian
// -----------------------
// Tensor data length in bytes, u32, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Quantized dtype, u32, little endian
// -----------------------
// Num shape dims, u32, little endian
// -----------------------
// ...
// Array (in original order): quantized weight shape dims, u32, little endian
// ...
// -----------------------
// ...
// Array: quantized weight data, u8s
// ...
// -----------------------
// [OPTIONAL] Bias tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------

impl QuantizedSerde for GgufMatMul {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "gguf"
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.b.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        let mut buffer = match &self.w {
            QMatMul::QTensor(qw) => {
                let w = qw.data()?.to_vec();
                let w_shape = qw.shape().dims();
                let dtype: u32 = match qw.dtype() {
                    GgmlDType::F32 => 0,
                    GgmlDType::F16 => 1,
                    GgmlDType::Q4_0 => 2,
                    GgmlDType::Q4_1 => 3,
                    GgmlDType::Q5_0 => 6,
                    GgmlDType::Q5_1 => 7,
                    GgmlDType::Q8_0 => 8,
                    GgmlDType::Q8_1 => 9,
                    GgmlDType::Q2K => 10,
                    GgmlDType::Q3K => 11,
                    GgmlDType::Q4K => 12,
                    GgmlDType::Q5K => 13,
                    GgmlDType::Q6K => 14,
                    GgmlDType::Q8K => 15,
                    // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
                    GgmlDType::BF16 => 30,
                };

                let mut buffer = Vec::new();

                // Version is always first!
                buffer.extend(&UQFF_VERSION.to_le_bytes());

                // ISQ type for GGUF is 0
                buffer.push(QuantizedSerdeType::Gguf as u8);

                // Length
                buffer.extend(&(w.len() as u32).to_le_bytes());

                // Has bias
                buffer.push(bias.is_some() as u8);

                // Dtype (u32)
                buffer.extend(&dtype.to_le_bytes());

                // Shape
                buffer.extend((w_shape.len() as u32).to_le_bytes());
                for dim in w_shape {
                    buffer.extend((*dim as u32).to_le_bytes());
                }

                // Quantized W Vec<u8> (just append it)
                buffer.extend(&w);

                buffer
            }
            QMatMul::TensorF16(_) | QMatMul::Tensor(_) => {
                candle_core::bail!("Cannot serialize non-quantized")
            }
        };

        if let Some(b) = bias.as_ref() {
            serialize_tensor(&mut buffer, b)?;
        }

        Ok(Cow::from(buffer))
    }

    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Gguf as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Gguf as usize
            );
        }

        let data_len = buffer.read_u32::<LittleEndian>()? as usize;

        let has_bias = buffer.read_u8()? != 0;

        // TODO: keep this in sync with get_isq_type_from_uqff!
        let dtype = buffer.read_u32::<LittleEndian>()?;
        let dtype = match dtype {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => GgmlDType::BF16,
            _ => candle_core::bail!("unknown dtype for quantized weight tensor {dtype}"),
        };

        let n_dims = buffer.read_u32::<LittleEndian>()? as usize;

        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(buffer.read_u32::<LittleEndian>()? as usize)
        }

        let mut tensor_data = vec![0; data_len];
        buffer.read_exact(&mut tensor_data)?;

        let _acquired_load_guard = guard.acquire(device);
        // If we have bias
        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        let w = qtensor_from_ggml(dtype, &tensor_data, dims, device)?;
        Ok(Arc::new(Self {
            w: QMatMul::QTensor(w.into()),
            b,
        }))
    }
    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)> {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Gguf as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Gguf as usize
            );
        }

        let data_len = buffer.read_u32::<LittleEndian>()? as usize;

        let has_bias = buffer.read_u8()? != 0;

        // TODO: keep this in sync with get_isq_type_from_uqff!
        let dtype = buffer.read_u32::<LittleEndian>()?;
        let dtype = match dtype {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => GgmlDType::BF16,
            _ => candle_core::bail!("unknown dtype for quantized weight tensor {dtype}"),
        };

        let n_dims = buffer.read_u32::<LittleEndian>()? as usize;

        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(buffer.read_u32::<LittleEndian>()? as usize)
        }

        let mut tensor_data = vec![0; data_len];
        buffer.read_exact(&mut tensor_data)?;

        let _acquired_load_guard = guard.acquire(device);
        // If we have bias
        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        let w = qtensor_from_ggml(dtype, &tensor_data, dims, device)?;
        Ok((
            Arc::new(Self {
                w: QMatMul::QTensor(w.into()),
                b: None,
            }),
            b,
        ))
    }
}

impl GgufMatMul {
    pub fn get_isq_type_from_uqff(data: Cow<[u8]>) -> Result<IsqType> {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Gguf as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Gguf as usize
            );
        }

        let _ = buffer.read_u32::<LittleEndian>()? as usize;

        let _ = buffer.read_u8()? != 0;

        let dtype = buffer.read_u32::<LittleEndian>()?;
        let dtype = match dtype {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => GgmlDType::BF16,
            _ => candle_core::bail!("unknown dtype for quantized weight tensor {dtype}"),
        };

        IsqType::try_from(dtype)
    }
}
