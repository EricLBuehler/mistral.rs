use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{
    backend::BackendStorage, CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Device, Layout,
    Result, Shape, Storage, Tensor,
};
use candle_nn::Linear;
use float8::F8E4M3;

use crate::{
    QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedActivation, QuantizedSerde, Shard,
    ShardedVarBuilder,
};

pub(crate) const MXFP8_BLOCK_SIZE: usize = 32;
pub(crate) const MXFP8_ACTIVATION_SCHEME: crate::ActivationQuantizationScheme =
    crate::ActivationQuantizationScheme {
        dtype: DType::F8E4M3,
        block_shape: [1, MXFP8_BLOCK_SIZE],
        scale_encoding: crate::ActivationScaleEncoding::E8M0,
    };
const FP8_MAX: f32 = 448.0;
const VECTOR_GEMV_MIN_K: usize = 512;
const E8M0_BIAS: i32 = 127;
const F32_MANTISSA_BITS: u32 = 23;
const F32_MANTISSA_MASK: u32 = 0x007f_ffff;
const FP8_MAX_MANTISSA: u32 = 0x0060_0000;
const FP8_SCALE_EXPONENT_BIAS: i32 = E8M0_BIAS + 8;

#[cfg(feature = "cuda")]
mod cuda;

#[derive(Debug, Clone)]
pub(crate) struct Mxfp8Linear {
    weight: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
    dtype: DType,
    #[cfg(all(feature = "cuda", feature = "cutile"))]
    native_mma: bool,
}

impl Mxfp8Linear {
    pub(crate) fn from_parts(
        weight: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
        dtype: DType,
    ) -> Result<Self> {
        let dims = weight.dims();
        let Some(&k) = dims.last() else {
            candle_core::bail!("MXFP8 weights cannot be scalar")
        };
        if !(2..=3).contains(&dims.len()) || k == 0 || !k.is_multiple_of(MXFP8_BLOCK_SIZE) {
            candle_core::bail!("MXFP8 requires rank 2 or 3 weights with K divisible by {MXFP8_BLOCK_SIZE}, got {dims:?}");
        }
        let mut scale_shape = dims.to_vec();
        *scale_shape.last_mut().unwrap() /= MXFP8_BLOCK_SIZE;
        if weight.dtype() != MXFP8_ACTIVATION_SCHEME.dtype
            || scales.dtype() != MXFP8_ACTIVATION_SCHEME.scale_encoding.storage_dtype()
            || scales.dims() != scale_shape
        {
            candle_core::bail!(
                "MXFP8 requires E4M3 weights and U8 E8M0 scales with shape {scale_shape:?}"
            );
        }
        if !weight.device().same_device(scales.device()) {
            candle_core::bail!("MXFP8 weights and scales must be on the same device");
        }
        let weight = weight.contiguous()?;
        let scales = scales.contiguous()?;
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        let native_mma = crate::cutile::register_mxfp8_shape(&weight, &scales, dtype);
        Ok(Self {
            #[cfg(all(feature = "cuda", feature = "cutile"))]
            native_mma,
            weight,
            scales,
            bias,
            dtype,
        })
    }

    pub(crate) fn load(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        hints: Shard,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        if !in_dim.is_multiple_of(MXFP8_BLOCK_SIZE) {
            candle_core::bail!("MXFP8 input width must be divisible by {MXFP8_BLOCK_SIZE}");
        }
        let scale_hints = crate::blockwise_fp8::scale_shard_from_weight_shard(
            [out_dim, in_dim],
            [1, MXFP8_BLOCK_SIZE],
            hints,
        )?;
        let weight = vb.get_with_hints_dtype((out_dim, in_dim), "weight", hints, DType::F8E4M3)?;
        let scales = vb.get_with_hints_dtype(
            (out_dim, in_dim / MXFP8_BLOCK_SIZE),
            "weight_scale",
            scale_hints,
            DType::U8,
        )?;
        let bias = if bias {
            let range = crate::shard_range(hints, &[out_dim, in_dim])?;
            let value = vb.get((out_dim,), "bias")?;
            Some(match range {
                Some((0, offset, len)) => value.narrow(0, offset, len)?,
                _ => value,
            })
        } else {
            None
        };
        Self::from_parts(weight, scales, bias, vb.dtype())
    }

    pub(crate) fn stack(layers: Vec<Self>) -> Result<Self> {
        let first = layers
            .first()
            .ok_or_else(|| candle_core::Error::msg("MXFP8 expert stack cannot be empty"))?;
        if layers.iter().any(|layer| {
            layer.bias.is_some()
                || layer.weight.shape() != first.weight.shape()
                || layer.dtype != first.dtype
        }) {
            candle_core::bail!("MXFP8 expert stacking requires matching bias-free projections");
        }
        Self::from_parts(
            Tensor::stack(
                &layers.iter().map(|layer| &layer.weight).collect::<Vec<_>>(),
                0,
            )?,
            Tensor::stack(
                &layers.iter().map(|layer| &layer.scales).collect::<Vec<_>>(),
                0,
            )?,
            None,
            first.dtype,
        )
    }

    pub(crate) fn permute_output_rows(self, rows: &[usize]) -> Result<Self> {
        let gather = |tensor: &Tensor| crate::distributed::layers::select_rows(tensor, rows);
        Self::from_parts(
            gather(&self.weight)?,
            gather(&self.scales)?,
            self.bias.as_ref().map(gather).transpose()?,
            self.dtype,
        )
    }

    pub(crate) fn merge(layers: Vec<Self>) -> Result<Option<crate::PackedLinear>> {
        let Some(first) = layers.first() else {
            return Ok(None);
        };
        if first.weight.rank() != 2 {
            return Ok(None);
        }
        let (_, input_dim) = first.weight.dims2()?;
        if layers.iter().any(|layer| {
            layer.weight.rank() != 2
                || layer.weight.dims()[1] != input_dim
                || layer.bias.is_some()
                || layer.dtype != first.dtype
                || !layer.weight.device().same_device(first.weight.device())
        }) {
            return Ok(None);
        }
        let packed = Self::from_parts(
            Tensor::cat(
                &layers.iter().map(|layer| &layer.weight).collect::<Vec<_>>(),
                0,
            )?,
            Tensor::cat(
                &layers.iter().map(|layer| &layer.scales).collect::<Vec<_>>(),
                0,
            )?,
            None,
            first.dtype,
        )?;
        let rows_per_rank = layers
            .iter()
            .map(|layer| layer.weight.dim(0))
            .collect::<Result<Vec<_>>>()?;
        let mut constituents = Vec::with_capacity(layers.len());
        let mut offset = 0;
        for &rows in &rows_per_rank {
            constituents.push(Arc::new(Self::from_parts(
                packed.weight.narrow(0, offset, rows)?,
                packed.scales.narrow(0, offset, rows)?,
                None,
                first.dtype,
            )?) as Arc<dyn QuantMethod>);
            offset += rows;
        }
        Ok(Some(crate::PackedLinear {
            packed: Arc::new(packed),
            constituents,
            rows_per_rank,
        }))
    }
}

impl QuantMethod for Mxfp8Linear {
    fn new(_method: QuantMethodConfig) -> Result<Self> {
        candle_core::bail!("MXFP8 layers require packed E4M3 weights and E8M0 scales")
    }

    fn embedding_table(&self) -> Option<crate::EmbeddingTable<'_>> {
        use crate::embedding::{RowCodec, ScaleGrid, ScaleType};
        let (rows, columns) = self.weight.dims2().ok()?;
        let grid = ScaleGrid::per_row(ScaleType::E8m0, MXFP8_BLOCK_SIZE);
        Some(
            crate::EmbeddingTable::new(RowCodec::Fp8(Some(grid)), rows, columns, &self.weight)
                .with_scales(&self.scales),
        )
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.weight
            .apply_op2_no_bwd(&self.scales, &Dequantize { dtype: self.dtype })
    }

    fn forward_raw(&self, input: &Tensor) -> Result<Tensor> {
        let (n, k) = self.weight.dims2()?;
        if input.dims().last().copied() != Some(k) || input.rank() < 2 {
            candle_core::bail!(
                "MXFP8 activation shape {:?} does not match weight K={k}",
                input.dims()
            );
        }
        let rows = input.elem_count() / k;
        let mut shape = input.dims().to_vec();
        *shape.last_mut().unwrap() = n;
        if rows == 0 {
            return Tensor::zeros(shape, input.dtype(), input.device());
        }
        if self.activation_quantization_scheme_for(input).is_some() {
            return self.forward_quantized(&self.quantize_activation(input)?);
        }
        let input = input.contiguous()?.reshape((rows, k))?;
        // A16 GEMV avoids an activation-quantization launch for a single vector.
        let input = if input.device().is_cuda() && rows == 1 && k >= VECTOR_GEMV_MIN_K {
            input
        } else {
            input.apply_op1_no_bwd(&Quantize)?
        };
        let output = input
            .apply_op3_no_bwd(&self.weight, &self.scales, &Matmul { indices: None })?
            .reshape(shape)?;
        match &self.bias {
            Some(bias) => output.broadcast_add(bias),
            None => Ok(output),
        }
    }

    fn routed_weights(&self) -> Option<crate::RoutedWeights> {
        let (experts, n, k) = self.weight.dims3().ok()?;
        Some(crate::RoutedWeights::new(
            crate::RoutedCodec::Mxfp8,
            self.weight.clone(),
            self.scales.clone(),
            None,
            self.bias.clone(),
            [experts, n, k],
        ))
    }

    fn gather_forward_raw(&self, input: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (_, _, k) = self.weight.dims3()?;
        let (rows, topk) = indices.dims2()?;
        let (input_rows, input_topk, input_k) = input.dims3()?;
        if input_rows != rows
            || input_k != k
            || (input_topk != 1 && input_topk != topk)
            || indices.dtype() != DType::U32
        {
            candle_core::bail!(
                "MXFP8 gather activation/index shapes do not match the expert weights"
            );
        }
        if let Some(routed) = self.routed_weights() {
            if let Some(output) = crate::try_routed_gather(input, indices, &routed)? {
                return Ok(output);
            }
        }
        #[cfg(all(feature = "cuda", has_fp8_mma_kernels))]
        {
            let view = crate::moe::grouped_fp8::Fp8WeightView {
                weight: &self.weight,
                scales: &self.scales,
                encoding: crate::moe::grouped_fp8::Fp8ScaleEncoding::E8m0,
                block: [1, MXFP8_BLOCK_SIZE],
            };
            if let Some(output) = crate::moe::grouped_fp8::gather(input, indices, view)? {
                return crate::routed_gemv::add_routed_bias(output, self.bias.as_ref(), indices);
            }
        }
        let input = input
            .contiguous()?
            .reshape((rows * input_topk, k))?
            .apply_op1_no_bwd(&Quantize)?;
        let output = input.apply_op3_no_bwd(
            &self.weight,
            &self.scales,
            &Matmul {
                indices: Some(indices.contiguous()?),
            },
        )?;
        crate::routed_gemv::add_routed_bias(output, self.bias.as_ref(), indices)
    }

    fn activation_quantization_scheme_for(
        &self,
        input: &Tensor,
    ) -> Option<crate::ActivationQuantizationScheme> {
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        if self.native_mma
            && input.rank() >= 2
            && input.dtype() == self.dtype
            && input.device().same_device(self.weight.device())
        {
            let k = *self.weight.dims().last()?;
            if input.dims().last() == Some(&k) && input.elem_count() / k > 1 {
                return Some(MXFP8_ACTIVATION_SCHEME);
            }
        }
        let _ = input;
        None
    }

    fn quantize_activation(&self, input: &Tensor) -> Result<QuantizedActivation> {
        if self.activation_quantization_scheme_for(input).is_none() {
            candle_core::bail!(
                "MXFP8 packed activation quantization is unavailable for this input"
            );
        }
        quantize_packed(input)
    }

    fn forward_quantized(&self, activation: &QuantizedActivation) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        if self.native_mma {
            if activation.source_dtype() != self.dtype {
                candle_core::bail!(
                    "MXFP8 packed activation source dtype {:?} does not match layer dtype {:?}",
                    activation.source_dtype(),
                    self.dtype
                );
            }
            let result = crate::cutile::cutile_mxfp8_gemm(activation, &self.weight, &self.scales)?;
            let mut shape = activation.source_shape().to_vec();
            *shape.last_mut().unwrap() = self.weight.dim(0)?;
            let result = result.reshape(shape)?;
            return match &self.bias {
                Some(bias) => result.broadcast_add(bias),
                None => Ok(result),
            };
        }
        let _ = activation;
        candle_core::bail!("MXFP8 packed activation matmul is unavailable on this device")
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }
    fn dtype_and_device(&self) -> (DType, Device) {
        (self.dtype, self.weight.device().clone())
    }
    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            self.dtype,
            self.weight.device().clone(),
            self.weight.dims().to_vec(),
            request,
            true,
        ))
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(crate::UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(
                self.dequantize_w()?.add(delta)?,
                self.bias.clone(),
            )),
        )?))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<crate::IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if dtype.is_none() {
            return Ok(Arc::new(Self::from_parts(
                self.weight.to_device(&device)?,
                self.scales.to_device(&device)?,
                self.bias
                    .as_ref()
                    .map(|bias| bias.to_device(&device))
                    .transpose()?,
                self.dtype,
            )?));
        }
        Arc::new(crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(self.dequantize_w()?, self.bias.clone()),
        ))?)
        .apply_isq(dtype, device, n_quantized, imatrix, guard)
    }
}

impl QuantizedSerde for Mxfp8Linear {
    fn name(&self) -> &'static str {
        "MXFP8"
    }
}

fn e8m0(value: u8) -> f32 {
    match value {
        0 => f32::from_bits(1 << 22),
        u8::MAX => f32::NAN,
        _ => f32::from_bits(u32::from(value) << 23),
    }
}

fn cpu_values(storage: &CpuStorage, layout: &Layout) -> Result<Vec<f32>> {
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::msg("MXFP8 requires contiguous tensors"))?;
    Ok(match storage {
        CpuStorage::F32(values) => values[start..end].to_vec(),
        CpuStorage::F16(values) => values[start..end]
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        CpuStorage::BF16(values) => values[start..end]
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        _ => candle_core::bail!("MXFP8 activations require F32, F16 or BF16"),
    })
}

fn cpu_output(values: Vec<f32>, dtype: DType) -> Result<CpuStorage> {
    Ok(match dtype {
        DType::F32 => CpuStorage::F32(values),
        DType::F16 => CpuStorage::F16(values.into_iter().map(half::f16::from_f32).collect()),
        DType::BF16 => CpuStorage::BF16(values.into_iter().map(half::bf16::from_f32).collect()),
        _ => candle_core::bail!("MXFP8 output requires F32, F16 or BF16"),
    })
}

fn quantization_scale(group: &[f32]) -> u8 {
    let maximum = group
        .iter()
        .map(|value| value.abs())
        .fold(f32::MIN_POSITIVE, f32::max);
    let bits = maximum.to_bits();
    let exponent = ((bits >> F32_MANTISSA_BITS) as i32 - FP8_SCALE_EXPONENT_BIAS
        + i32::from(bits & F32_MANTISSA_MASK > FP8_MAX_MANTISSA))
    .clamp(-127, 127);
    (exponent + E8M0_BIAS) as u8
}

pub(crate) fn quantize_packed(input: &Tensor) -> Result<QuantizedActivation> {
    let k = input.dim(candle_core::D::Minus1)?;
    if input.rank() < 2 || k == 0 || !k.is_multiple_of(MXFP8_BLOCK_SIZE) {
        candle_core::bail!(
            "MXFP8 activation must have rank >= 2 and K divisible by {MXFP8_BLOCK_SIZE}"
        );
    }
    let rows = input.elem_count() / k;
    let flattened = input.contiguous()?.reshape((rows, k))?;
    let (storage, layout) = flattened.storage_and_layout();
    let (values, scales) = match &*storage {
        Storage::Cpu(storage) => {
            let input = cpu_values(storage, layout)?;
            let mut values = Vec::with_capacity(input.len());
            let mut scales = Vec::with_capacity(input.len() / MXFP8_BLOCK_SIZE);
            for group in input.as_chunks::<MXFP8_BLOCK_SIZE>().0 {
                let encoded = quantization_scale(group);
                let scale = e8m0(encoded);
                scales.push(encoded);
                values.extend(
                    group
                        .iter()
                        .map(|value| F8E4M3::from_f32((*value / scale).clamp(-FP8_MAX, FP8_MAX))),
                );
            }
            (
                Storage::Cpu(CpuStorage::F8E4M3(values)),
                Storage::Cpu(CpuStorage::U8(scales)),
            )
        }
        #[cfg(feature = "cuda")]
        Storage::Cuda(storage) => {
            let (values, scales) = cuda::quantize_packed(storage, layout)?;
            (Storage::Cuda(values), Storage::Cuda(scales))
        }
        _ => candle_core::bail!("MXFP8 packed activation quantization requires CPU or CUDA"),
    };
    QuantizedActivation::new(
        Tensor::from((values, Shape::from((rows, k)))),
        Tensor::from((scales, Shape::from((rows, k / MXFP8_BLOCK_SIZE)))),
        input.dims().to_vec(),
        input.dtype(),
        MXFP8_ACTIVATION_SCHEME,
    )
}

struct Quantize;

impl CustomOp1 for Quantize {
    fn name(&self) -> &'static str {
        "mxfp8-quantize-activation"
    }
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let mut values = cpu_values(storage, layout)?;
        for group in values.as_chunks_mut::<MXFP8_BLOCK_SIZE>().0 {
            let scale = e8m0(quantization_scale(group));
            for value in group {
                *value =
                    F8E4M3::from_f32((*value / scale).clamp(-FP8_MAX, FP8_MAX)).to_f32() * scale;
            }
        }
        Ok((cpu_output(values, storage.dtype())?, layout.shape().clone()))
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda::quantize(storage, layout)
    }
}

struct Dequantize {
    dtype: DType,
}

impl CustomOp2 for Dequantize {
    fn name(&self) -> &'static str {
        "mxfp8-dequantize"
    }
    fn cpu_fwd(
        &self,
        weight: &CpuStorage,
        wl: &Layout,
        scales: &CpuStorage,
        sl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F8E4M3(weight) = weight else {
            candle_core::bail!("MXFP8 weight dtype must be E4M3")
        };
        let CpuStorage::U8(scales) = scales else {
            candle_core::bail!("MXFP8 scales must be U8 E8M0")
        };
        let values = (0..wl.shape().elem_count())
            .map(|i| {
                weight[wl.start_offset() + i].to_f32()
                    * e8m0(scales[sl.start_offset() + i / MXFP8_BLOCK_SIZE])
            })
            .collect();
        Ok((cpu_output(values, self.dtype)?, wl.shape().clone()))
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        weight: &candle_core::CudaStorage,
        wl: &Layout,
        scales: &candle_core::CudaStorage,
        sl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda::dequantize(weight, wl, scales, sl, self.dtype)
    }
}

struct Matmul {
    indices: Option<Tensor>,
}

impl CustomOp3 for Matmul {
    fn name(&self) -> &'static str {
        "mxfp8-matmul"
    }
    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        il: &Layout,
        weight: &CpuStorage,
        wl: &Layout,
        scales: &CpuStorage,
        sl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let values = cpu_values(input, il)?;
        let CpuStorage::F8E4M3(weight) = weight else {
            candle_core::bail!("MXFP8 weight dtype must be E4M3")
        };
        let CpuStorage::U8(scales) = scales else {
            candle_core::bail!("MXFP8 scales must be U8 E8M0")
        };
        let (input_rows, k) = il.shape().dims2()?;
        let n = wl.dims()[wl.dims().len() - 2];
        let (rows, topk, ids) = match &self.indices {
            Some(indices) => {
                let (rows, topk) = indices.dims2()?;
                (rows, topk, Some(indices.to_vec2::<u32>()?))
            }
            None => (input_rows, 1, None),
        };
        let mut output = vec![0f32; rows * topk * n];
        for row in 0..rows {
            for slot in 0..topk {
                let expert = ids.as_ref().map_or(0, |ids| ids[row][slot] as usize);
                let input_row = if input_rows == rows {
                    row
                } else {
                    row * topk + slot
                };
                for column in 0..n {
                    let base = (expert * n + column) * k;
                    let mut sum = 0f32;
                    for offset in 0..k {
                        let w = weight[wl.start_offset() + base + offset].to_f32()
                            * e8m0(scales[sl.start_offset() + (base + offset) / MXFP8_BLOCK_SIZE]);
                        sum = values[input_row * k + offset].mul_add(w, sum);
                    }
                    output[(row * topk + slot) * n + column] = sum;
                }
            }
        }
        let shape = if ids.is_some() {
            Shape::from((rows, topk, n))
        } else {
            Shape::from((rows, n))
        };
        Ok((cpu_output(output, input.dtype())?, shape))
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        il: &Layout,
        weight: &candle_core::CudaStorage,
        wl: &Layout,
        scales: &candle_core::CudaStorage,
        sl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda::matmul(cuda::Operands {
            input,
            input_layout: il,
            weight,
            weight_layout: wl,
            scales,
            scale_layout: sl,
            indices: self.indices.as_ref(),
        })
    }
}
