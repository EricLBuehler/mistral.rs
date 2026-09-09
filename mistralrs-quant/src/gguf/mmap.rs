//! CPU GGUF weights backed directly by the archive mmap.
//!
//! Candle's CPU `QStorage::from_data` copies packed GGML blocks into an owned
//! `Vec`. This method instead dispatches Candle's public GGML kernels over the
//! aligned bytes retained by `GgufArchive`, so loading does not make an
//! anonymous-RAM copy of every quantized tensor.

use std::{
    fmt::Debug,
    mem::{align_of, size_of},
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{
    quantized::{
        k_quants::{
            self, BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0,
            BlockQ5_1, BlockQ6K, BlockQ8K, BlockQ8_0, BlockQ8_1, GgmlType,
        },
        GgmlDType, QStorage, QTensor,
    },
    DType, Device, Error, Result, Shape, Tensor,
};
use candle_nn::Linear;
use half::{bf16, f16};

use super::{archive::GgufArchive, ggml_dtype_to_uqff_code, GgufMatMul};
use crate::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, UnquantLinear,
    UqffTensor,
};

#[derive(Debug)]
pub struct GgufMmapMatMul {
    archive: Arc<GgufArchive>,
    tensor_name: Arc<str>,
    dtype: GgmlDType,
    dims: Vec<usize>,
    bias: Option<Tensor>,
    stats: crate::ImatrixLayerStats,
}

impl GgufMmapMatMul {
    pub fn new(
        archive: Arc<GgufArchive>,
        tensor_name: impl Into<Arc<str>>,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let tensor_name = tensor_name.into();
        let info = archive.tensor_info(&tensor_name)?;
        let dtype = info.dtype().candle_dtype()?;
        let dims = info.shape().to_vec();
        if dims.len() < 2 {
            candle_core::bail!(
                "mmap-backed GGUF linear `{tensor_name}` has rank {}, expected at least 2",
                dims.len()
            );
        }
        let data = archive.tensor_data(&tensor_name)?;
        validate_typed_data(dtype, data.bytes())?;
        Ok(Self {
            archive,
            tensor_name,
            dtype,
            dims,
            bias,
            stats: crate::ImatrixLayerStats::empty(),
        })
    }

    fn bytes(&self) -> Result<&[u8]> {
        Ok(self.archive.tensor_data(&self.tensor_name)?.bytes())
    }

    fn add_bias(&self, output: Tensor) -> Result<Tensor> {
        if let Some(bias) = &self.bias {
            output.broadcast_add(bias)
        } else {
            Ok(output)
        }
    }

    fn owned_method(&self, device: &Device) -> Result<GgufMatMul> {
        let storage = QStorage::from_data(self.bytes()?.into(), device, self.dtype)?;
        let weight = QTensor::new(storage, self.dims.clone())?;
        let bias = self
            .bias
            .as_ref()
            .map(|bias| bias.to_device(device))
            .transpose()?;
        Ok(GgufMatMul::from_qtensor(weight, bias))
    }

    fn output_from_f32(
        &self,
        values: Vec<f32>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Tensor> {
        let output = Tensor::from_vec(values, shape, &Device::Cpu)?;
        if dtype == DType::F32 {
            Ok(output)
        } else {
            output.to_dtype(dtype)
        }
    }
}

impl QuantMethod for GgufMmapMatMul {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("GgufMmapMatMul must be constructed from a GgufArchive")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        let mut output = vec![0f32; elem_count(&self.dims)?];
        dequantize(self.dtype, self.bytes()?, &mut output)?;
        Tensor::from_vec(output, Shape::from(self.dims.clone()), &Device::Cpu)
    }

    fn forward_raw(&self, input: &Tensor) -> Result<Tensor> {
        self.stats.process(input)?;
        if !input.device().is_cpu() {
            candle_core::bail!(
                "mmap-backed GGUF tensor `{}` is CPU-only, but input is on {:?}",
                self.tensor_name,
                input.device()
            );
        }
        if self.dims.len() != 2 {
            candle_core::bail!(
                "mmap-backed GGUF forward requires rank-2 weights, got {:?} for `{}`",
                self.dims,
                self.tensor_name
            );
        }
        if input.rank() < 2 {
            candle_core::bail!(
                "mmap-backed GGUF input has rank {}, expected at least 2",
                input.rank()
            );
        }
        let n = self.dims[0];
        let k = self.dims[1];
        if input.dim(input.rank() - 1)? != k {
            candle_core::bail!(
                "mmap-backed GGUF input shape {:?} is incompatible with weight shape {:?}",
                input.dims(),
                self.dims
            );
        }
        let input_dtype = input.dtype();
        let mut output_shape = input.dims().to_vec();
        *output_shape.last_mut().expect("rank checked above") = n;
        let input = input
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let m = input.len() / k;
        let output_len = m
            .checked_mul(n)
            .ok_or_else(|| Error::msg("GGUF matmul output length overflow"))?;
        let mut output = vec![0f32; output_len];
        matmul(self.dtype, (m, k, n), &input, self.bytes()?, &mut output)?;
        let output = self.output_from_f32(output, Shape::from(output_shape), input_dtype)?;
        self.add_bias(output)
    }

    fn gather_forward_raw(&self, input: &Tensor, indices: &Tensor) -> Result<Tensor> {
        if !input.device().is_cpu() || !indices.device().is_cpu() {
            candle_core::bail!("mmap-backed GGUF expert gather is CPU-only")
        }
        if self.dims.len() != 3 {
            candle_core::bail!(
                "mmap-backed GGUF expert gather requires rank-3 weights, got {:?}",
                self.dims
            );
        }
        let (experts, n, k) = (self.dims[0], self.dims[1], self.dims[2]);
        let index_dims = indices.dims().to_vec();
        let route_count = elem_count(&index_dims)?;
        let ids = indices
            .to_dtype(DType::U32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let (input_rows, mut output_shape) = routed_inputs(input, &index_dims, k)?;
        output_shape.push(n);
        let expected_input_len = route_count
            .checked_mul(k)
            .ok_or_else(|| Error::msg("GGUF routed input length overflow"))?;
        if input_rows.len() != expected_input_len {
            candle_core::bail!(
                "mmap-backed GGUF routed input has {} values, expected {expected_input_len}",
                input_rows.len()
            );
        }

        let bytes = self.bytes()?;
        let expert_elements = n
            .checked_mul(k)
            .ok_or_else(|| Error::msg("GGUF expert element count overflow"))?;
        let expert_bytes = packed_byte_len(expert_elements, self.dtype)?;
        let output_len = route_count
            .checked_mul(n)
            .ok_or_else(|| Error::msg("GGUF routed output length overflow"))?;
        let mut output = vec![0f32; output_len];
        for (route, &expert) in ids.iter().enumerate() {
            let expert = expert as usize;
            if expert >= experts {
                candle_core::bail!(
                    "GGUF expert index {expert} is out of range for {experts} experts"
                );
            }
            let start = expert
                .checked_mul(expert_bytes)
                .ok_or_else(|| Error::msg("GGUF expert byte offset overflow"))?;
            let end = start
                .checked_add(expert_bytes)
                .ok_or_else(|| Error::msg("GGUF expert byte range overflow"))?;
            matmul(
                self.dtype,
                (1, k, n),
                &input_rows[route * k..(route + 1) * k],
                bytes.get(start..end).ok_or_else(|| {
                    Error::msg(format!(
                        "GGUF expert {expert} byte range {start}..{end} exceeds {} bytes",
                        bytes.len()
                    ))
                })?,
                &mut output[route * n..(route + 1) * n],
            )?;
        }
        let output = self.output_from_f32(output, Shape::from(output_shape), input.dtype())?;
        if let Some(bias) = &self.bias {
            if bias.rank() == 2 {
                let mut shape = index_dims;
                shape.push(n);
                let selected = bias
                    .index_select(&indices.flatten_all()?, 0)?
                    .reshape(Shape::from(shape))?;
                output.broadcast_add(&selected)
            } else {
                output.broadcast_add(bias)
            }
        } else {
            Ok(output)
        }
    }

    fn embedding_forward_raw(&self, ids: &Tensor) -> Result<Tensor> {
        if self.dims.len() != 2 {
            candle_core::bail!(
                "mmap-backed GGUF embedding requires rank-2 weights, got {:?}",
                self.dims
            );
        }
        let rows = self.dims[0];
        let hidden = self.dims[1];
        let ids_shape = ids.dims().to_vec();
        let ids = ids
            .to_device(&Device::Cpu)?
            .to_dtype(DType::U32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let row_bytes = packed_byte_len(hidden, self.dtype)?;
        let bytes = self.bytes()?;
        let output_len = ids
            .len()
            .checked_mul(hidden)
            .ok_or_else(|| Error::msg("GGUF embedding output length overflow"))?;
        let mut output = vec![0f32; output_len];
        for (output_row, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= rows {
                candle_core::bail!("GGUF embedding row {id} is out of range for {rows} rows");
            }
            let start = id
                .checked_mul(row_bytes)
                .ok_or_else(|| Error::msg("GGUF embedding byte offset overflow"))?;
            let end = start
                .checked_add(row_bytes)
                .ok_or_else(|| Error::msg("GGUF embedding byte range overflow"))?;
            dequantize(
                self.dtype,
                bytes.get(start..end).ok_or_else(|| {
                    Error::msg(format!(
                        "GGUF embedding row {id} byte range {start}..{end} exceeds {} bytes",
                        bytes.len()
                    ))
                })?,
                &mut output[output_row * hidden..(output_row + 1) * hidden],
            )?;
        }
        let mut output_shape = ids_shape;
        output_shape.push(hidden);
        Tensor::from_vec(output, Shape::from(output_shape), &Device::Cpu)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F32, Device::Cpu)
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        if self.dims.len() == 3 && request.ty.is_some_and(|ty| !ty.supports_stacked_gather()) {
            candle_core::bail!(
                "Cannot quantize stacked GGUF expert weights to {}: that target does not support stacked expert gather.",
                request.ty.expect("checked above")
            );
        }
        Ok(crate::plan_weight_isq(
            DType::F32,
            Device::Cpu,
            self.dims.clone(),
            request,
            true,
        ))
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let dense = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            self.dequantize_w()?,
            self.bias.clone(),
        )))?;
        dense.add_delta_w(delta)
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if dtype.is_none() && device.is_cpu() {
            return Ok(self);
        }
        if dtype.is_some_and(|ty| IsqType::try_from(self.dtype).ok() == Some(ty))
            && imatrix_weight.is_none()
        {
            if device.is_cpu() {
                return Ok(self);
            }
            return Ok(Arc::new(self.owned_method(&device)?));
        }
        let dense = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(self.dequantize_w()?, self.bias.clone()),
        ))?);
        dense.apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    fn begin_track_stats(&self) -> Result<()> {
        if self.dims.len() == 3 {
            self.stats
                .enable_routed(self.dims[0], self.dims[2], &Device::Cpu)
        } else {
            self.stats.enable(*self.dims.last().unwrap(), &Device::Cpu)
        }
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        if !self.stats.is_enabled() {
            candle_core::bail!("`{}` is not tracking stats", self.name())
        }
        let result = self.stats.compute_imatrix();
        self.stats.clear()?;
        result
    }

    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        self.stats.snapshot()
    }

    fn process_routed_stats(&self, input: &Tensor, ids: &Tensor) -> Result<()> {
        self.stats.process_routed(input, ids)
    }
}

impl QuantizedSerde for GgufMmapMatMul {
    fn name(&self) -> &'static str {
        "gguf"
    }

    fn isq_serde_supported(&self) -> bool {
        true
    }

    fn uqff_type(&self) -> Option<IsqType> {
        IsqType::try_from(self.dtype).ok()
    }

    fn serialize_uqff(&self, prefix: &str, _ty: IsqType) -> Result<Vec<UqffTensor>> {
        let bytes = self.bytes()?.to_vec();
        let bytes_len = bytes.len();
        let mut tensors = vec![
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                crate::QuantizedSerdeType::Gguf as u8,
            ),
            UqffTensor::from_raw_u8(format!("{prefix}.weight"), bytes, vec![bytes_len]),
            UqffTensor::from_u32_scalar(
                format!("{prefix}.weight.dtype"),
                ggml_dtype_to_uqff_code(self.dtype),
            ),
            UqffTensor::from_u32_vec(
                format!("{prefix}.weight.shape"),
                self.dims.iter().map(|&dim| dim as u32).collect(),
                vec![self.dims.len()],
            ),
        ];
        if let Some(bias) = &self.bias {
            tensors.push(UqffTensor::from_tensor(format!("{prefix}.bias"), bias)?);
        }
        Ok(tensors)
    }
}

fn elem_count(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .ok_or_else(|| Error::msg("GGUF tensor element count overflow"))
    })
}

fn packed_byte_len(elements: usize, dtype: GgmlDType) -> Result<usize> {
    if !elements.is_multiple_of(dtype.block_size()) {
        candle_core::bail!(
            "GGUF row has {elements} elements, not divisible by {:?} block size {}",
            dtype,
            dtype.block_size()
        );
    }
    elements
        .checked_div(dtype.block_size())
        .and_then(|blocks| blocks.checked_mul(dtype.type_size()))
        .ok_or_else(|| Error::msg("GGUF packed byte length overflow"))
}

fn validate_typed_data(dtype: GgmlDType, bytes: &[u8]) -> Result<()> {
    if !bytes.len().is_multiple_of(dtype.type_size()) {
        candle_core::bail!(
            "GGUF {:?} storage has {} bytes, not divisible by type size {}",
            dtype,
            bytes.len(),
            dtype.type_size()
        );
    }
    let alignment = super::archive::ggml_dtype_alignment(dtype);
    if !(bytes.as_ptr() as usize).is_multiple_of(alignment) {
        candle_core::bail!("GGUF mmap storage for {dtype:?} is not aligned to {alignment} bytes");
    }
    Ok(())
}

fn typed<T>(bytes: &[u8]) -> Result<&[T]> {
    if !bytes.len().is_multiple_of(size_of::<T>())
        || !(bytes.as_ptr() as usize).is_multiple_of(align_of::<T>())
    {
        candle_core::bail!("invalid or unaligned GGUF storage for typed kernel dispatch");
    }
    // SAFETY: GGUF dtype validation guarantees both exact element size and alignment.
    Ok(unsafe {
        std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), bytes.len() / size_of::<T>())
    })
}

fn dispatch<T: GgmlType>(
    mkn: (usize, usize, usize),
    input: &[f32],
    bytes: &[u8],
    output: &mut [f32],
) -> Result<()> {
    k_quants::matmul::<T>(mkn, input, typed::<T>(bytes)?, output)
}

fn matmul(
    dtype: GgmlDType,
    mkn: (usize, usize, usize),
    input: &[f32],
    bytes: &[u8],
    output: &mut [f32],
) -> Result<()> {
    match dtype {
        GgmlDType::F32 => dispatch::<f32>(mkn, input, bytes, output),
        GgmlDType::F16 => dispatch::<f16>(mkn, input, bytes, output),
        GgmlDType::BF16 => dispatch::<bf16>(mkn, input, bytes, output),
        GgmlDType::Q4_0 => dispatch::<BlockQ4_0>(mkn, input, bytes, output),
        GgmlDType::Q4_1 => dispatch::<BlockQ4_1>(mkn, input, bytes, output),
        GgmlDType::Q5_0 => dispatch::<BlockQ5_0>(mkn, input, bytes, output),
        GgmlDType::Q5_1 => dispatch::<BlockQ5_1>(mkn, input, bytes, output),
        GgmlDType::Q8_0 => dispatch::<BlockQ8_0>(mkn, input, bytes, output),
        GgmlDType::Q8_1 => dispatch::<BlockQ8_1>(mkn, input, bytes, output),
        GgmlDType::Q2K => dispatch::<BlockQ2K>(mkn, input, bytes, output),
        GgmlDType::Q3K => dispatch::<BlockQ3K>(mkn, input, bytes, output),
        GgmlDType::Q4K => dispatch::<BlockQ4K>(mkn, input, bytes, output),
        GgmlDType::Q5K => dispatch::<BlockQ5K>(mkn, input, bytes, output),
        GgmlDType::Q6K => dispatch::<BlockQ6K>(mkn, input, bytes, output),
        GgmlDType::Q8K => dispatch::<BlockQ8K>(mkn, input, bytes, output),
    }
}

fn dequantize_typed<T: GgmlType>(bytes: &[u8], output: &mut [f32]) -> Result<()> {
    T::to_float(typed::<T>(bytes)?, output);
    Ok(())
}

fn dequantize(dtype: GgmlDType, bytes: &[u8], output: &mut [f32]) -> Result<()> {
    match dtype {
        GgmlDType::F32 => dequantize_typed::<f32>(bytes, output),
        GgmlDType::F16 => dequantize_typed::<f16>(bytes, output),
        GgmlDType::BF16 => dequantize_typed::<bf16>(bytes, output),
        GgmlDType::Q4_0 => dequantize_typed::<BlockQ4_0>(bytes, output),
        GgmlDType::Q4_1 => dequantize_typed::<BlockQ4_1>(bytes, output),
        GgmlDType::Q5_0 => dequantize_typed::<BlockQ5_0>(bytes, output),
        GgmlDType::Q5_1 => dequantize_typed::<BlockQ5_1>(bytes, output),
        GgmlDType::Q8_0 => dequantize_typed::<BlockQ8_0>(bytes, output),
        GgmlDType::Q8_1 => dequantize_typed::<BlockQ8_1>(bytes, output),
        GgmlDType::Q2K => dequantize_typed::<BlockQ2K>(bytes, output),
        GgmlDType::Q3K => dequantize_typed::<BlockQ3K>(bytes, output),
        GgmlDType::Q4K => dequantize_typed::<BlockQ4K>(bytes, output),
        GgmlDType::Q5K => dequantize_typed::<BlockQ5K>(bytes, output),
        GgmlDType::Q6K => dequantize_typed::<BlockQ6K>(bytes, output),
        GgmlDType::Q8K => dequantize_typed::<BlockQ8K>(bytes, output),
    }
}

fn routed_inputs(
    input: &Tensor,
    index_dims: &[usize],
    hidden: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let values = input
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (token_count, routes, per_route, output_shape) = match (input.dims(), index_dims) {
        ([tokens, 1, h], [index_tokens, routes]) if tokens == index_tokens && *h == hidden => {
            (*tokens, *routes, false, vec![*tokens, *routes])
        }
        ([tokens, input_routes, h], [index_tokens, routes])
            if tokens == index_tokens && input_routes == routes && *h == hidden =>
        {
            (*tokens, *routes, true, vec![*tokens, *routes])
        }
        ([batch, seq, 1, h], [ib, is, routes])
            if batch == ib && seq == is && *h == hidden =>
        {
            (*batch * *seq, *routes, false, vec![*batch, *seq, *routes])
        }
        ([batch, seq, routes, h], [ib, is, iroutes])
            if batch == ib && seq == is && routes == iroutes && *h == hidden =>
        {
            (*batch * *seq, *routes, true, vec![*batch, *seq, *routes])
        }
        ([batch, seq, 1, 1, h], [ib, is, routes])
            if batch == ib && seq == is && *h == hidden =>
        {
            (*batch * *seq, *routes, false, vec![*batch, *seq, *routes])
        }
        _ => candle_core::bail!(
            "mmap-backed GGUF routed input shape {:?} does not match indices shape {:?} and hidden size {hidden}",
            input.dims(),
            index_dims
        ),
    };
    let capacity = token_count
        .checked_mul(routes)
        .and_then(|count| count.checked_mul(hidden))
        .ok_or_else(|| Error::msg("GGUF routed input capacity overflow"))?;
    let mut rows = Vec::with_capacity(capacity);
    if per_route {
        rows.extend_from_slice(&values);
    } else {
        for token in 0..token_count {
            let row = &values[token * hidden..(token + 1) * hidden];
            for _ in 0..routes {
                rows.extend_from_slice(row);
            }
        }
    }
    Ok((rows, output_shape))
}
