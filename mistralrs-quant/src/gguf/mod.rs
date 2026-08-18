pub mod archive;
pub mod cpu;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "cuda")]
pub mod fast_mmq;
#[cfg(feature = "cuda")]
pub mod fast_mmvq;
#[cfg(feature = "cuda")]
mod ffi;
#[cfg(all(feature = "cuda", has_marlin_kernels))]
mod packed_affine;
mod weight_source;

pub use weight_source::{
    GgufBindingMap, GgufBindingResolver, GgufTensorBackend, GgufTensorBinding, GgufWeightSource,
};

use candle_core::{
    quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul, QStorage, QTensor},
    DType, Device, Result, Shape, Tensor,
};
use candle_nn::{Linear, Module};
use safetensors::tensor::Dtype;
#[cfg(all(feature = "cuda", has_marlin_kernels))]
use std::sync::OnceLock;
use std::sync::{atomic::AtomicUsize, Arc};

use crate::uqff::{UqffHeaderMatch, UqffLayerHeaderView};
use crate::{
    generate_isq, generate_isq_imatrix, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedSerde, QuantizedSerdeType, Shard, UqffReader, UqffTensor,
};

#[cfg(feature = "cuda")]
pub(crate) const GGUF_AFFINE_MIN_BATCH: usize = 8;

#[cfg(all(feature = "cuda", has_marlin_kernels))]
pub(crate) fn gguf_affine_budget_bytes(device: &Device, dtype: DType) -> usize {
    packed_affine::budget_bytes(device, dtype)
}

#[derive(Debug)]
pub struct GgufMatMul {
    pub(crate) w: QMatMul,
    pub(crate) b: Option<Tensor>,
    stats: crate::ImatrixLayerStats,
    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    packed_affine: OnceLock<Option<Arc<packed_affine::PackedAffine>>>,
    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    _gguf_affine_reservation: Option<packed_affine::Reservation>,
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
    fn from_parts(w: QMatMul, b: Option<Tensor>, stats: crate::ImatrixLayerStats) -> Self {
        #[cfg(all(feature = "cuda", has_marlin_kernels))]
        let reservation = packed_affine::Reservation::new(&w);
        Self {
            w,
            b,
            stats,
            #[cfg(all(feature = "cuda", has_marlin_kernels))]
            packed_affine: OnceLock::new(),
            #[cfg(all(feature = "cuda", has_marlin_kernels))]
            _gguf_affine_reservation: reservation,
        }
    }

    fn qtensor_to_device(q: &Arc<QTensor>, device: &Device) -> Result<Arc<QTensor>> {
        if q.device().same_device(device) {
            return Ok(q.clone());
        }
        let storage = QStorage::from_data(q.data()?, device, q.dtype())?;
        Ok(Arc::new(QTensor::new(storage, q.shape())?))
    }

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
        Ok(Self::from_parts(
            QMatMul::from_arc(w.into())?,
            b,
            crate::ImatrixLayerStats::empty(),
        ))
    }

    /// Construct without `QMatMul::from_arc`: densifying would bypass the gather kernels
    /// expert stacks rely on.
    pub(crate) fn from_qtensor(w: QTensor, b: Option<Tensor>) -> Self {
        Self::from_parts(
            QMatMul::QTensor(Arc::new(w)),
            b,
            crate::ImatrixLayerStats::empty(),
        )
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

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    fn packed_affine_for(
        &self,
        flat_batch: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Option<&Arc<packed_affine::PackedAffine>>> {
        if !packed_affine::enabled() {
            return Ok(None);
        }
        let QMatMul::QTensor(weight) = &self.w else {
            return Ok(None);
        };
        if packed_affine::minimum_batch(weight.dtype()).is_none_or(|min| flat_batch < min) {
            return Ok(None);
        }
        let Some(reservation) = self._gguf_affine_reservation.as_ref() else {
            return Ok(None);
        };
        if !packed_affine::PackedAffine::supports(weight, dtype)
            || !weight.device().same_device(device)
        {
            return Ok(None);
        }
        let packed = self.packed_affine.get_or_init(|| {
            match packed_affine::PackedAffine::new(weight, dtype, Some(reservation)) {
                Ok(packed) => Some(Arc::new(packed)),
                Err(error) => {
                    tracing::debug!("Packed GGUF affine initialization failed: {error}");
                    crate::log::once_log_warn(
                        "Packed GGUF affine acceleration is unavailable for one or more layers; using canonical kernels where needed.",
                    );
                    None
                }
            }
        });
        reservation.consume();
        let Some(packed) = packed else {
            return Ok(None);
        };
        Ok(Some(packed))
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    fn packed_affine(&self, a: &Tensor) -> Result<Option<&Arc<packed_affine::PackedAffine>>> {
        let Some((&_, batch_dims)) = a.dims().split_last() else {
            return Ok(None);
        };
        let flat_batch = batch_dims.iter().product::<usize>();
        let QMatMul::QTensor(weight) = &self.w else {
            return Ok(None);
        };
        let Some(dtype) = packed_affine::PackedAffine::dtype_for_input(weight, a) else {
            return Ok(None);
        };
        self.packed_affine_for(flat_batch, dtype, a.device())
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    fn packed_affine_forward(&self, a: &Tensor) -> Result<Option<Tensor>> {
        let Some(packed) = self.packed_affine(a)? else {
            return Ok(None);
        };
        let dtype = a.dtype();
        let a = if dtype == packed.dtype() {
            a.clone()
        } else {
            a.to_dtype(packed.dtype())?
        };
        let output = packed.forward(&a)?;
        Ok(Some(if output.dtype() == dtype {
            output
        } else {
            output.to_dtype(dtype)?
        }))
    }
}

impl QuantMethod for GgufMatMul {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { q_weight, b } => Ok(Self::from_parts(
                QMatMul::from_arc(q_weight)?,
                b,
                crate::ImatrixLayerStats::empty(),
            )),
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
        match &self.w {
            QMatMul::QTensor(weight) if weight.dtype() == GgmlDType::Q8_1 => {
                weight.dequantize(&weight.device())
            }
            _ => self.w.dequantize_f16()?.to_dtype(DType::F32),
        }
    }

    fn embedding_forward_raw(&self, ids: &Tensor) -> Result<Tensor> {
        self.w.embedding(ids)
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        self.stats.process(a)?;
        #[cfg(all(feature = "cuda", has_marlin_kernels))]
        {
            if let Some(out) = self.packed_affine_forward(a)? {
                return self.add_bias(out);
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = self.try_fast_forward(a)? {
                return self.add_bias(out);
            }
            if let QMatMul::QTensor(weight) = &self.w {
                if weight.device().is_cuda()
                    && matches!(weight.dtype(), GgmlDType::Q8_1 | GgmlDType::Q8K)
                {
                    candle_core::bail!(
                        "CUDA {:?} weights require the packed GGUF affine backend with a tile-compatible shape",
                        weight.dtype()
                    );
                }
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
        // For CPU and Metal: use dequantize-then-matmul approach
        #[cfg(feature = "cuda")]
        let res = if x.device().is_cuda() {
            cuda::qmatmul_indexed_moe_forward(&self.w, x, indices)?
        } else {
            cpu::cpu_indexed_moe_forward(&self.w, x, indices)?
        };

        #[cfg(not(feature = "cuda"))]
        let res = cpu::cpu_indexed_moe_forward(&self.w, x, indices)?;

        if let Some(b) = &self.b {
            if b.rank() == 2 {
                let mut shape = indices.dims().to_vec();
                shape.push(b.dim(1)?);
                let bias = b
                    .index_select(&indices.flatten_all()?, 0)?
                    .reshape(Shape::from(shape))?;
                res.broadcast_add(&bias)
            } else {
                res.broadcast_add(b)
            }
        } else {
            Ok(res)
        }
    }

    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        match &self.w {
            candle_core::quantized::QMatMul::QTensor(qt) => Some(qt.clone()),
            _ => None,
        }
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    fn prepare_gguf_affine_raw(
        &self,
        flat_batch: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<bool> {
        Ok(self.packed_affine_for(flat_batch, dtype, device)?.is_some())
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    fn try_gguf_affine_forward_raw(&self, a: &Tensor) -> Result<Option<Tensor>> {
        let Some(out) = self.packed_affine_forward(a)? else {
            return Ok(None);
        };
        Ok(Some(self.add_bias(out)?))
    }

    fn quantized_act_type(&self) -> Option<DType> {
        #[cfg(feature = "cuda")]
        {
            if self.uses_fast_mmvq() {
                return None;
            }
        }
        #[cfg(all(feature = "cuda", has_marlin_kernels))]
        if matches!(
            &self.w,
            QMatMul::QTensor(q)
                if q.device().is_cuda()
                    && q.shape().rank() == 2
                    && packed_affine::minimum_batch(q.dtype()).is_some()
        ) {
            return None;
        }
        // cpu handles bf16 activations natively (widened once inside the packed matmul)
        if let QMatMul::QTensor(qt) = &self.w {
            if qt.device().is_cpu() {
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
                ..
            } => Ok(Arc::new(Self::from_parts(
                QMatMul::Tensor((w + delta)?),
                b.clone(),
                stats.clone(),
            ))),
            Self {
                w: QMatMul::TensorF16(w),
                b,
                stats,
                ..
            } => Ok(Arc::new(Self::from_parts(
                QMatMul::TensorF16((w + delta)?),
                b.clone(),
                stats.clone(),
            ))),
            Self {
                w: QMatMul::QTensor(w),
                b,
                stats,
                ..
            } => {
                let (w, dtype) = (w.dequantize(&w.device())?, w.dtype());
                let w = QMatMul::QTensor(std::sync::Arc::new(
                    candle_core::quantized::QTensor::quantize(&(w + delta)?, dtype)?,
                ));
                Ok(Arc::new(Self::from_parts(w, b.clone(), stats.clone())))
            }
        }
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        match &self.w {
            QMatMul::QTensor(q) => (DType::F32, q.device()),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => (t.dtype(), t.device().clone()),
        }
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        let (dtype, device, shape) = match &self.w {
            QMatMul::QTensor(q) => (DType::F32, q.device(), q.shape().dims().to_vec()),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
                (t.dtype(), t.device().clone(), t.dims().to_vec())
            }
        };
        if shape.len() == 3 && request.ty.is_some_and(|ty| !ty.supports_stacked_gather()) {
            candle_core::bail!(
                "Cannot quantize stacked GGUF expert weights to {}: that target does not support stacked expert gather. Use a Q*K/Q*_0/Q*_1 target, AFQ, or omit ISQ.",
                request.ty.expect("rank-3 rejection requires an ISQ target")
            );
        }
        Ok(crate::plan_weight_isq(dtype, device, shape, request, true))
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
            if imatrix_weight.is_none() {
                if let QMatMul::QTensor(q) = &self.w {
                    if IsqType::try_from(q.dtype()).ok() == Some(dtype) {
                        let w = QMatMul::QTensor(Self::qtensor_to_device(q, &device)?);
                        let b = self
                            .b
                            .as_ref()
                            .map(|bias| bias.to_device(&device))
                            .transpose()?;
                        return Ok(Arc::new(GgufMatMul::from_parts(w, b, self.stats.clone())));
                    }
                }
            }

            let t = match &self.w {
                QMatMul::QTensor(q) => q.dequantize(&q.device())?,
                QMatMul::TensorF16(t) | QMatMul::Tensor(t) => t.clone(),
            };
            if t.rank() == 3 {
                return crate::utils::isq::quantize_expert_stack_with_bias(
                    t,
                    self.b.clone(),
                    dtype,
                    imatrix_weight,
                    &device,
                    n_quantized,
                    guard,
                );
            }

            if GgmlDType::try_from(dtype).is_err() {
                let unquant = Arc::new(crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
                    Linear::new(t, self.b.clone()),
                ))?) as Arc<dyn QuantMethod>;
                return unquant.apply_isq(Some(dtype), device, n_quantized, imatrix_weight, guard);
            }

            let bias = self
                .b
                .as_ref()
                .map(|bias| bias.to_device(&device))
                .transpose()?;
            let res = if let Some(imatrix_weight) = imatrix_weight {
                let dtype = dtype.try_into()?;
                generate_isq_imatrix!(t, imatrix_weight, device, dtype, n_quantized, guard)
            } else {
                let dtype = dtype.try_into()?;
                generate_isq!(t, device, dtype, n_quantized, guard)
            };
            Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: res,
                b: bias,
            })?))
        } else {
            let w = match &self.w {
                QMatMul::QTensor(q) => QMatMul::QTensor(Self::qtensor_to_device(q, &device)?),
                QMatMul::Tensor(t) => QMatMul::Tensor(t.to_device(&device)?),
                QMatMul::TensorF16(t) => QMatMul::TensorF16(t.to_device(&device)?),
            };
            let b = if let Some(b) = &self.b {
                Some(b.to_device(&device)?)
            } else {
                None
            };
            Ok(Arc::new(GgufMatMul::from_parts(w, b, self.stats.clone())))
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
    fn uqff_type(&self) -> Option<IsqType> {
        match &self.w {
            QMatMul::QTensor(qw) => IsqType::try_from(qw.dtype()).ok(),
            QMatMul::Tensor(_) | QMatMul::TensorF16(_) => None,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EMBEDDING_DIM: usize = 256;
    const TEST_VOCAB_SIZE: usize = 8;

    fn assert_embedding_matches_dequantized_gather(
        device: &Device,
        dtype: GgmlDType,
    ) -> Result<()> {
        let values = (0..TEST_VOCAB_SIZE * TEST_EMBEDDING_DIM)
            .map(|index| ((index % 37) as f32 - 18.0) / 7.0)
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(values, (TEST_VOCAB_SIZE, TEST_EMBEDDING_DIM), device)?;
        let weight = QTensor::quantize(&weight, dtype)?;
        let ids = Tensor::from_vec(vec![7u32, 1, 7, 3, 0, 4], (2, 3), device)?;
        let expected = weight
            .dequantize(device)?
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((2, 3, TEST_EMBEDDING_DIM))?;
        let layer = GgufMatMul::from_qtensor(weight, None);
        let actual = layer.embedding_forward_raw(&ids)?;

        assert_eq!(actual.dims(), &[2, 3, TEST_EMBEDDING_DIM]);
        assert_eq!(actual.dtype(), DType::F32);
        let max_diff = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max_diff <= 1e-6, "{dtype:?}: max_diff={max_diff}");
        Ok(())
    }

    #[test]
    fn q_sensitive_targets_support_embedding_gather() -> Result<()> {
        for dtype in [GgmlDType::Q6K, GgmlDType::Q8_0] {
            assert_embedding_matches_dequantized_gather(&Device::Cpu, dtype)?;
        }
        Ok(())
    }

    #[test]
    fn capture_on_same_device_preserves_packed_weight() -> Result<()> {
        let weight = Tensor::ones((2, 32), DType::F32, &Device::Cpu)?;
        let layer = Arc::new(GgufMatMul::from_qtensor(
            QTensor::quantize(&weight, GgmlDType::Q8_0)?,
            None,
        ));
        let before = layer.get_qtensor().unwrap();
        let captured = layer.apply_isq(
            None,
            Device::Cpu,
            &AtomicUsize::new(0),
            None,
            QuantizeOntoGuard::new(),
        )?;
        let after = captured.get_qtensor().unwrap();

        assert!(Arc::ptr_eq(&before, &after));
        Ok(())
    }

    #[test]
    fn same_target_preserves_rank_two_packed_weight_and_bias() -> Result<()> {
        let weight = Tensor::ones((4, 256), DType::F32, &Device::Cpu)?;
        let bias = Tensor::from_vec(vec![1f32, 2., 3., 4.], 4, &Device::Cpu)?;
        let layer = Arc::new(GgufMatMul::from_qtensor(
            QTensor::quantize(&weight, GgmlDType::Q4K)?,
            Some(bias),
        ));
        let before = layer.get_qtensor().unwrap();
        let before_data = before.data()?.into_owned();
        let n_quantized = AtomicUsize::new(0);
        let preserved = layer.apply_isq(
            Some(IsqType::Q4K),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        let after = preserved.get_qtensor().unwrap();

        assert!(Arc::ptr_eq(&before, &after));
        assert_eq!(after.data()?.into_owned(), before_data);
        assert_eq!(n_quantized.load(std::sync::atomic::Ordering::Relaxed), 0);
        let input = Tensor::zeros((1, 256), DType::F32, &Device::Cpu)?;
        assert_eq!(
            preserved.forward(&input)?.to_vec2::<f32>()?,
            vec![vec![1., 2., 3., 4.]]
        );
        Ok(())
    }

    #[test]
    fn same_target_preserves_rank_three_packed_weight_and_bias() -> Result<()> {
        let weight = Tensor::ones((2, 4, 256), DType::F32, &Device::Cpu)?;
        let bias = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6., 7., 8.], (2, 4), &Device::Cpu)?;
        let layer = Arc::new(GgufMatMul::from_qtensor(
            QTensor::quantize(&weight, GgmlDType::Q4K)?,
            Some(bias),
        ));
        let before = layer.get_qtensor().unwrap();
        let before_data = before.data()?.into_owned();
        let n_quantized = AtomicUsize::new(0);
        let preserved = layer.apply_isq(
            Some(IsqType::Q4K),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        let after = preserved.get_qtensor().unwrap();

        assert!(Arc::ptr_eq(&before, &after));
        assert_eq!(after.data()?.into_owned(), before_data);
        assert_eq!(n_quantized.load(std::sync::atomic::Ordering::Relaxed), 0);
        let input = Tensor::zeros((1, 1, 256), DType::F32, &Device::Cpu)?;
        let indices = Tensor::from_vec(vec![1u32, 0], (1, 2), &Device::Cpu)?;
        assert_eq!(
            preserved
                .gather_forward(&input, &indices)?
                .to_vec3::<f32>()?,
            vec![vec![vec![5., 6., 7., 8.], vec![1., 2., 3., 4.]]]
        );
        Ok(())
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn assert_cross_device_capture_preserves_packed_weight(device: Device) -> Result<()> {
        let weight = Tensor::ones((4, 256), DType::F32, &Device::Cpu)?;
        let layer = Arc::new(GgufMatMul::from_qtensor(
            QTensor::quantize(&weight, GgmlDType::Q4K)?,
            None,
        ));
        let before = layer.get_qtensor().unwrap();
        let before_data = before.data()?.into_owned();
        let n_quantized = AtomicUsize::new(0);
        let captured = layer.apply_isq(
            None,
            device.clone(),
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        let after = captured.get_qtensor().unwrap();

        assert!(after.device().same_device(&device));
        assert_eq!(after.dtype(), before.dtype());
        assert_eq!(after.shape(), before.shape());
        assert_eq!(after.data()?.into_owned(), before_data);
        assert_eq!(n_quantized.load(std::sync::atomic::Ordering::Relaxed), 0);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn capture_to_cuda_preserves_packed_weight() -> Result<()> {
        assert_cross_device_capture_preserves_packed_weight(Device::new_cuda(0)?)
    }

    #[cfg(feature = "metal")]
    #[test]
    fn capture_to_metal_preserves_packed_weight() -> Result<()> {
        assert_cross_device_capture_preserves_packed_weight(Device::new_metal(0)?)
    }

    #[test]
    fn packed_gguf_requantizes_to_non_ggml_isq() -> Result<()> {
        let weight = Tensor::ones((4, 64), DType::F32, &Device::Cpu)?;
        let bias = Tensor::zeros(4, DType::F32, &Device::Cpu)?;
        let layer = Arc::new(GgufMatMul::from_qtensor(
            QTensor::quantize(&weight, GgmlDType::Q8_0)?,
            Some(bias),
        ));
        let n_quantized = AtomicUsize::new(0);
        let quantized = layer.apply_isq(
            Some(IsqType::AFQ4),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;

        assert_eq!(quantized.name(), "afq-layer");
        assert!(quantized.has_bias());
        assert_eq!(quantized.dequantize_w()?.dims(), &[4, 64]);
        assert_eq!(n_quantized.load(std::sync::atomic::Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn packed_expert_stack_requantizes_to_non_ggml_isq() -> Result<()> {
        let weight = Tensor::ones((2, 4, 64), DType::F32, &Device::Cpu)?;
        let bias = Tensor::zeros((2, 4), DType::F32, &Device::Cpu)?;
        let layer = Arc::new(GgufMatMul::from_qtensor(
            QTensor::quantize(&weight, GgmlDType::Q8_0)?,
            Some(bias),
        ));
        let n_quantized = AtomicUsize::new(0);
        let quantized = layer.apply_isq(
            Some(IsqType::AFQ4),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;

        assert_eq!(quantized.name(), "afq-layer");
        assert!(quantized.has_bias());
        assert_eq!(quantized.dequantize_w()?.dims(), &[2, 4, 64]);
        assert_eq!(n_quantized.load(std::sync::atomic::Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn packed_expert_stack_requantizes_to_ggml_once() -> Result<()> {
        let weight = Tensor::ones((2, 4, 32), DType::F32, &Device::Cpu)?;
        let layer = Arc::new(GgufMatMul::from_qtensor(
            QTensor::quantize(&weight, GgmlDType::Q8_0)?,
            None,
        ));
        let n_quantized = AtomicUsize::new(0);
        let quantized = layer.apply_isq(
            Some(IsqType::Q4_0),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;

        assert_eq!(quantized.name(), "gguf");
        assert_eq!(quantized.dequantize_w()?.dims(), &[2, 4, 32]);
        assert_eq!(n_quantized.load(std::sync::atomic::Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn packed_expert_stack_rejects_targets_without_gather() -> Result<()> {
        for ty in [IsqType::F8Q8, IsqType::HQQ4, IsqType::MXFP4] {
            let weight = Tensor::ones((2, 4, 32), DType::F32, &Device::Cpu)?;
            let layer = Arc::new(GgufMatMul::from_qtensor(
                QTensor::quantize(&weight, GgmlDType::Q8_0)?,
                None,
            ));
            let request = crate::IsqRequest {
                ty: Some(ty),
                device: Device::Cpu,
                has_imatrix: false,
                capture: crate::IsqCaptureMode::Immediate,
                consumer: crate::IsqConsumer::UqffWrite,
                module_key: "experts".to_string(),
            };
            let plan_error = layer.plan_isq(&request).unwrap_err();
            assert!(
                plan_error
                    .to_string()
                    .contains("does not support stacked expert gather"),
                "{ty}: {plan_error}"
            );
            let error = layer
                .apply_isq(
                    Some(ty),
                    Device::Cpu,
                    &AtomicUsize::new(0),
                    None,
                    QuantizeOntoGuard::new(),
                )
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("does not support stacked expert gather"),
                "{ty}: {error}"
            );
        }
        Ok(())
    }

    #[test]
    fn uqff_shards_stacked_expert_bias_with_output_dimension() -> Result<()> {
        let weight = Tensor::zeros((2, 4, 64), DType::F32, &Device::Cpu)?;
        let bias = Tensor::from_vec(
            (0..8).map(|value| value as f32).collect::<Vec<_>>(),
            (2, 4),
            &Device::Cpu,
        )?;
        let layer = GgufMatMul::from_qtensor(
            QTensor::quantize(&weight, GgmlDType::Q8_0)?,
            Some(bias.clone()),
        );
        let mut tensors = crate::uqff_version_tensors();
        tensors.extend(layer.serialize_uqff("experts.gate", IsqType::Q8_0)?);
        let dir = tempfile::tempdir().map_err(candle_core::Error::wrap)?;
        let file = dir.path().join("experts.uqff");
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &file,
        )
        .map_err(candle_core::Error::wrap)?;
        let reader = UqffReader::open(&[file])?;

        let output_shard = reader
            .load_linear(
                "experts.gate",
                &Device::Cpu,
                Shard::Simple {
                    dim: 1,
                    rank: 1,
                    world_size: 2,
                },
            )?
            .expect("output shard");
        let indices = Tensor::from_vec(vec![0u32, 1], (1, 2), &Device::Cpu)?;
        let input = Tensor::zeros((1, 1, 64), DType::F32, &Device::Cpu)?;
        let actual = output_shard.gather_forward(&input, &indices)?;
        let expected = bias
            .narrow(1, 2, 2)?
            .contiguous()?
            .index_select(&indices.flatten_all()?, 0)?
            .reshape((1, 2, 2))?;
        assert_eq!(actual.to_vec3::<f32>()?, expected.to_vec3::<f32>()?);

        let input_shard = reader
            .load_linear(
                "experts.gate",
                &Device::Cpu,
                Shard::Simple {
                    dim: 2,
                    rank: 0,
                    world_size: 2,
                },
            )?
            .expect("input shard");
        assert!(!input_shard.has_bias());
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn q_sensitive_targets_support_metal_embedding_gather() -> Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [GgmlDType::Q6K, GgmlDType::Q8_0] {
            assert_embedding_matches_dequantized_gather(&device, dtype)?;
        }
        Ok(())
    }
}
