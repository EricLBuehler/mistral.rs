#[cfg(not(feature = "cuda"))]
mod cpu;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "cuda")]
pub mod fast_mmq;
#[cfg(feature = "cuda")]
pub mod fast_mmvq;
#[cfg(feature = "cuda")]
mod ffi;

use candle_core::{
    quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};
use candle_nn::Module;
use safetensors::tensor::Dtype;
use std::sync::{atomic::AtomicUsize, Arc};

use crate::uqff::{UqffHeaderMatch, UqffLayerHeaderView};
use crate::{
    generate_isq, generate_isq_imatrix, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedSerde, QuantizedSerdeType, Shard, UqffReader, UqffTensor,
};

#[derive(Debug)]
pub struct GgufMatMul {
    pub(crate) w: QMatMul,
    pub(crate) b: Option<Tensor>,
    stats: crate::ImatrixLayerStats,
}

fn ggml_dtype_to_uqff_code(dtype: GgmlDType) -> u32 {
    match dtype {
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
        GgmlDType::BF16 => 30,
    }
}

fn ggml_dtype_from_uqff_code(dtype: u32) -> Result<GgmlDType> {
    match dtype {
        0 => Ok(GgmlDType::F32),
        1 => Ok(GgmlDType::F16),
        2 => Ok(GgmlDType::Q4_0),
        3 => Ok(GgmlDType::Q4_1),
        6 => Ok(GgmlDType::Q5_0),
        7 => Ok(GgmlDType::Q5_1),
        8 => Ok(GgmlDType::Q8_0),
        9 => Ok(GgmlDType::Q8_1),
        10 => Ok(GgmlDType::Q2K),
        11 => Ok(GgmlDType::Q3K),
        12 => Ok(GgmlDType::Q4K),
        13 => Ok(GgmlDType::Q5K),
        14 => Ok(GgmlDType::Q6K),
        15 => Ok(GgmlDType::Q8K),
        30 => Ok(GgmlDType::BF16),
        _ => candle_core::bail!("unknown dtype for quantized weight tensor {dtype}"),
    }
}

fn gguf_dtype_label(dtype: u32) -> String {
    match dtype {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2k",
        11 => "q3k",
        12 => "q4k",
        13 => "q5k",
        14 => "q6k",
        15 => "q8k",
        30 => "bf16",
        _ => "unknown",
    }
    .to_string()
}

impl GgufMatMul {
    pub(crate) fn inspect_uqff_header(layer: &UqffLayerHeaderView<'_>) -> Option<UqffHeaderMatch> {
        const WEIGHT_SUFFIXES: &[&str] =
            &["weight", "weight.format", "weight.dtype", "weight.shape"];
        if layer.exact_weight_suffixes(WEIGHT_SUFFIXES)
            && layer.tensor_dtype("weight", Dtype::U8)
            && layer.scalar("weight.format", Dtype::U8)
            && layer.scalar("weight.dtype", Dtype::U32)
            && layer.u32_vector("weight.shape")
        {
            Some(UqffHeaderMatch {
                serde_type: QuantizedSerdeType::Gguf,
            })
        } else {
            None
        }
    }

    pub(crate) fn stored_label_from_uqff_tensors(
        tensors: &[UqffTensor],
        prefix: &str,
    ) -> Result<String> {
        let dtype = crate::uqff::u32_scalar_with_suffix(tensors, prefix, "weight.dtype")?;
        Ok(gguf_dtype_label(dtype))
    }

    pub fn isq_type_from_uqff_dtype(dtype: u32) -> Result<IsqType> {
        IsqType::try_from(ggml_dtype_from_uqff_code(dtype)?)
    }

    pub(crate) fn block_size_from_uqff_dtype(dtype: u32) -> Result<usize> {
        Ok(ggml_dtype_from_uqff_code(dtype)?.block_size())
    }

    pub fn from_raw_uqff(
        dtype: u32,
        tensor_data: Vec<u8>,
        dims: Vec<usize>,
        b: Option<Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let dtype = ggml_dtype_from_uqff_code(dtype)?;
        let w = qtensor_from_ggml(dtype, &tensor_data, dims, device)?;
        // from_arc densifies float fallback entries, matching what ISQ produces at load
        Ok(Self {
            w: QMatMul::from_arc(w.into())?,
            b,
            stats: crate::ImatrixLayerStats::empty(),
        })
    }

    /// Construct without `QMatMul::from_arc`: densifying would bypass the gather kernels
    /// expert stacks rely on.
    pub(crate) fn from_qtensor(w: QTensor, b: Option<Tensor>) -> Self {
        Self {
            w: QMatMul::QTensor(Arc::new(w)),
            b,
            stats: crate::ImatrixLayerStats::empty(),
        }
    }

    /// Quantize a stacked `[E, out, in]` expert tensor slab-by-slab; `imatrix` is `[in]` shared
    /// or `[E, in]` flattened. Row-block formats never cross expert boundaries, so the
    /// byte-concat equals quantizing the whole stack.
    pub(crate) fn quantize_expert_stack(
        stack: &Tensor,
        ty: IsqType,
        imatrix: Option<&[f32]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<QTensor> {
        use crate::utils::isq::QuantizationBehavior;

        let (num_experts, out_dim, in_dim) = stack.dims3()?;
        // device-resident byte extraction races in-flight work; quantize from a CPU copy
        let stack = stack.to_device(&Device::Cpu)?;
        let requested: GgmlDType = ty.try_into()?;
        let slab0 = stack.get(0)?;
        let dtype = match crate::utils::isq::get_quantization_behaviour(&slab0, requested) {
            QuantizationBehavior::Quantize(dtype) => dtype,
            QuantizationBehavior::Skip => GgmlDType::F32,
        };
        let imatrix_capable = matches!(
            dtype,
            GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
        );
        if let Some(v) = imatrix {
            if v.len() != in_dim && v.len() != num_experts * in_dim {
                crate::log::once_log_warn(format!(
                    "Expert stack imatrix length {} matches neither in_dim {in_dim} nor {num_experts}x{in_dim}; quantizing without it.",
                    v.len()
                ));
            }
        }

        let mut bytes = Vec::new();
        for e in 0..num_experts {
            // expert views share storage with an offset; quantize reads raw storage
            let slab = stack.get(e)?.force_contiguous()?;
            let vector = imatrix.and_then(|v| {
                if v.len() == in_dim {
                    Some(v)
                } else if v.len() == num_experts * in_dim {
                    Some(&v[e * in_dim..(e + 1) * in_dim])
                } else {
                    None
                }
            });
            let qt = match vector {
                Some(v) if imatrix_capable && v.iter().any(|x| *x != 0.0) => {
                    candle_core::quantized::QTensor::quantize_imatrix(&slab, v, dtype)?
                }
                _ => candle_core::quantized::QTensor::quantize(&slab, dtype)?,
            };
            bytes.extend_from_slice(&qt.data()?);
        }

        let _acquired_quantize_guard = guard.acquire(device);
        qtensor_from_ggml(dtype, &bytes, vec![num_experts, out_dim, in_dim], device)
    }

    fn from_uqff(reader: &UqffReader, key: &str, device: &Device, shard: Shard) -> Result<Self> {
        let dtype = reader.load_u32_scalar(&format!("{key}.weight.dtype"))?;
        let mut dims = reader.load_u32_vec(&format!("{key}.weight.shape"))?;
        let mut weight = reader.load_raw_u8(&format!("{key}.weight"))?;
        let range = crate::uqff::shard_range(shard, &dims)?;
        if let Some((dim, start, len)) = range {
            let ggml = ggml_dtype_from_uqff_code(dtype)?;
            weight = crate::uqff::slice_blocked_data(
                &weight,
                &dims,
                ggml.block_size(),
                ggml.type_size(),
                dim,
                start,
                len,
            )?;
            dims[dim] = len;
        }
        let bias = reader.load_bias(key, device, range, dims.len())?;
        Self::from_raw_uqff(dtype, weight, dims, bias, device)
    }

    fn add_bias(&self, x: Tensor) -> Result<Tensor> {
        if let Some(ref b) = self.b {
            x.broadcast_add(b)
        } else {
            Ok(x)
        }
    }

    #[cfg(feature = "cuda")]
    fn uses_fast_mmvq(&self) -> bool {
        matches!(
            &self.w,
            QMatMul::QTensor(q) if q.device().is_cuda() && fast_mmvq::supports(q.dtype())
        )
    }

    #[cfg(feature = "cuda")]
    fn try_fast_forward(&self, a: &Tensor) -> Result<Option<Tensor>> {
        if !self.uses_fast_mmvq() || !matches!(a.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            return Ok(None);
        }

        let flat_batch = a.dims()[..a.dims().len().saturating_sub(1)]
            .iter()
            .product::<usize>();

        let QMatMul::QTensor(q) = &self.w else {
            unreachable!("uses_fast_mmvq() requires QTensor weights")
        };

        // Batch 1-8: use MMVQ (decode kernel)
        if (1..=fast_mmvq::MMVQ_MAX_BATCH).contains(&flat_batch) {
            return Ok(Some(fast_mmvq::plain(q, a)?));
        }

        // Batch > 8: use MMQ (prompt kernel)
        if flat_batch > fast_mmvq::MMVQ_MAX_BATCH {
            return Ok(Some(fast_mmq::plain(q, a)?));
        }

        Ok(None)
    }
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
                stats: crate::ImatrixLayerStats::empty(),
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

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        self.stats.process(a)?;
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = self.try_fast_forward(a)? {
                return self.add_bias(out);
            }
        }

        // Fallback: Candle QMatMul requires F32
        let original_dtype = a.dtype();
        let a_f32 = if original_dtype == DType::F32 {
            a.clone()
        } else {
            a.to_dtype(DType::F32)?
        };
        let x = self.w.forward(&a_f32)?;
        let x = if original_dtype == DType::F32 {
            x
        } else {
            x.to_dtype(original_dtype)?
        };
        self.add_bias(x)
    }

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    ///
    /// If `a` is (n_tokens, 1, cols), `self` weights are (n_experts, rows, cols),
    /// then the indices are (n_tokens, n_experts_per_tok).
    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
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

    #[cfg(feature = "cuda")]
    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        match &self.w {
            candle_core::quantized::QMatMul::QTensor(qt) => Some(qt.clone()),
            _ => None,
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        #[cfg(feature = "cuda")]
        {
            if self.uses_fast_mmvq() {
                return None;
            }
        }
        Some(DType::F32)
    }

    fn has_bias(&self) -> bool {
        self.b.is_some()
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        match self {
            Self {
                w: QMatMul::Tensor(w),
                b,
                stats,
            } => Ok(Arc::new(Self {
                w: QMatMul::Tensor((w + delta)?),
                b: b.clone(),
                stats: stats.clone(),
            })),
            Self {
                w: QMatMul::TensorF16(w),
                b,
                stats,
            } => Ok(Arc::new(Self {
                w: QMatMul::TensorF16((w + delta)?),
                b: b.clone(),
                stats: stats.clone(),
            })),
            Self {
                w: QMatMul::QTensor(w),
                b,
                stats,
            } => {
                let (w, dtype) = (w.dequantize(&w.device())?, w.dtype());
                let w = QMatMul::QTensor(std::sync::Arc::new(
                    candle_core::quantized::QTensor::quantize(&(w + delta)?, dtype)?,
                ));
                Ok(Arc::new(Self {
                    w,
                    b: b.clone(),
                    stats: stats.clone(),
                }))
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
            let res = if let Some(imatrix_weight) = imatrix_weight {
                // routed imatrix vectors are per-expert; stacks quantize slab-by-slab
                if t.rank() == 3 {
                    n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let w = Self::quantize_expert_stack(
                        &t,
                        dtype,
                        Some(&imatrix_weight),
                        &device,
                        guard,
                    )?;
                    return Ok(Arc::new(Self::from_qtensor(w, self.b.clone())));
                }
                let dtype = dtype.try_into()?;
                generate_isq_imatrix!(t, imatrix_weight, device, dtype, n_quantized, guard)
            } else {
                let dtype = dtype.try_into()?;
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
            Ok(Arc::new(GgufMatMul {
                w,
                b,
                stats: self.stats.clone(),
            }))
        }
    }

    fn begin_track_stats(&self) -> Result<()> {
        let dims = match &self.w {
            QMatMul::QTensor(q) => q.shape().dims().to_vec(),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.dims().to_vec(),
        };
        let device = self.dtype_and_device().1;
        // Stacked [E, out, in] expert weights collect per expert via the routed path.
        if dims.len() == 3 {
            self.stats.enable_routed(dims[0], dims[2], &device)
        } else {
            self.stats.enable(*dims.last().unwrap(), &device)
        }
    }

    fn process_routed_stats(&self, x: &Tensor, ids: &Tensor) -> Result<()> {
        self.stats.process_routed(x, ids)
    }

    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        self.stats.snapshot()
    }
    fn end_track_stats(&self) -> Result<Tensor> {
        if self.stats.is_enabled() {
            let imatrix = self.stats.compute_imatrix();
            self.stats.clear()?;
            imatrix
        } else {
            candle_core::bail!("`{}` is not tracking stats.", self.name())
        }
    }
}

impl QuantizedSerde for GgufMatMul {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "gguf"
    }
    fn serialize_uqff(&self, prefix: &str, _ty: IsqType) -> Result<Vec<UqffTensor>> {
        // float fallbacks densify at construction; requantize losslessly so they serialize like the rest
        let densified;
        let qw = match &self.w {
            QMatMul::QTensor(qw) => qw,
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
                let ggml = match t.dtype() {
                    DType::F16 => GgmlDType::F16,
                    DType::BF16 => GgmlDType::BF16,
                    _ => GgmlDType::F32,
                };
                densified = Arc::new(QTensor::quantize(&t.to_dtype(DType::F32)?, ggml)?);
                &densified
            }
        };

        // The dtype tag self-describes each layer, so fallback-quantized layers serialize as-is.
        let w = qw.data()?.into_owned();
        let w_len = w.len();
        let mut data = vec![
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Gguf as u8,
            ),
            UqffTensor::from_raw_u8(format!("{prefix}.weight"), w, vec![w_len]),
            UqffTensor::from_u32_scalar(
                format!("{prefix}.weight.dtype"),
                ggml_dtype_to_uqff_code(qw.dtype()),
            ),
            UqffTensor::from_u32_vec(
                format!("{prefix}.weight.shape"),
                qw.shape().dims().iter().map(|dim| *dim as u32).collect(),
                vec![qw.shape().dims().len()],
            ),
        ];
        if let Some(bias) = &self.b {
            data.push(UqffTensor::from_tensor(format!("{prefix}.bias"), bias)?);
        }
        Ok(data)
    }
    fn deserialize_uqff(
        reader: &UqffReader,
        prefix: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self::from_uqff(reader, prefix, device, shard)?))
    }
    fn isq_type_from_uqff(reader: &UqffReader, prefix: &str) -> Result<IsqType> {
        Self::isq_type_from_uqff_dtype(reader.load_u32_scalar(&format!("{prefix}.weight.dtype"))?)
    }
}
