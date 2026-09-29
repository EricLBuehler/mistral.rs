use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Linear;
use float8::F8E4M3;

use crate::{
    Nvfp4ActivationMode, Nvfp4LinearSpec, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedSerde, ScaleConvention, Shard, ShardedVarBuilder, NVFP4_BLOCK_SIZE,
};

#[cfg(all(feature = "cuda", feature = "cutile", has_nvfp4_cutlass_sm121_kernels))]
pub(crate) mod cutlass;

#[cfg(all(
    test,
    feature = "cuda",
    feature = "cutile",
    has_nvfp4_cutlass_sm121_kernels
))]
mod cutlass_tests;

#[cfg(all(
    test,
    feature = "cuda",
    feature = "cutile",
    has_nvfp4_cutlass_sm121_kernels
))]
mod cutlass_random_tests;

const FP4_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const FP4_MAX: f32 = 6.0;
const FP8_MAX: f32 = 448.0;
const VALUES_PER_BYTE: usize = 2;
pub(crate) const NVFP4_ACTIVATION_SCHEME: crate::ActivationQuantizationScheme =
    crate::ActivationQuantizationScheme {
        dtype: DType::U8,
        block_shape: [1, NVFP4_BLOCK_SIZE],
    };

#[derive(Debug)]
pub struct Nvfp4LayerParts {
    /// Packed E2M1 weights [..., N, K / 2], with the first value in the lower nibble.
    pub weight: Tensor,
    /// E4M3 dequantization scales [..., N, K / 16].
    pub scales: Tensor,
    /// F32 dequantization multipliers [..., N], expanded from tensor or fused-group scales.
    pub global_scales: Tensor,
    /// F32 dequantization multiplier: a scalar for dense layers or [E] for experts, present only for W4A4.
    pub input_scale: Option<Tensor>,
    pub activation: Nvfp4ActivationMode,
    pub bias: Option<Tensor>,
    pub dtype: DType,
}

#[derive(Debug)]
pub struct Nvfp4Layer {
    parts: Nvfp4LayerParts,
    input_global_scale: Option<f32>,
    // Value equality is safe only for private calibration allocations.
    input_scale_immutable: bool,
    #[cfg(all(feature = "cuda", feature = "cutile", has_nvfp4_cutlass_sm121_kernels))]
    native: Option<cutlass::State>,
}

#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct Nvfp4InputCalibration<'a> {
    layer: &'a Nvfp4Layer,
}

impl Nvfp4InputCalibration<'_> {
    pub(crate) fn matches(self, other: Self) -> bool {
        self.layer.shares_input_calibration(other.layer)
    }
}

impl Nvfp4Layer {
    pub fn from_parts(parts: Nvfp4LayerParts) -> Result<Self> {
        for scale in [&parts.global_scales]
            .into_iter()
            .chain(parts.input_scale.iter())
        {
            if scale.dtype() != DType::F32 {
                candle_core::bail!("NVFP4 global scales must be F32");
            }
            for value in scale.flatten_all()?.to_vec1::<f32>()? {
                dequant_scale(value, ScaleConvention::Dequantize)?;
            }
        }
        Self::from_normalized_parts(parts, false)
    }

    fn from_normalized_parts(
        mut parts: Nvfp4LayerParts,
        input_scale_immutable: bool,
    ) -> Result<Self> {
        let dims = parts.weight.dims();
        if !matches!(dims.len(), 2 | 3) || parts.weight.dtype() != DType::U8 {
            candle_core::bail!("NVFP4 weights must be rank-2 or rank-3 packed U8 tensors");
        }
        if dims.contains(&0) {
            candle_core::bail!("NVFP4 weight dimensions must be nonzero");
        }
        if !matches!(parts.dtype, DType::BF16 | DType::F16 | DType::F32) {
            candle_core::bail!("NVFP4 output dtype must be BF16, F16, or F32");
        }
        if parts.weight.device().is_cuda() && parts.dtype == DType::F32 {
            candle_core::bail!("NVFP4 CUDA inference requires BF16 or F16 output dtype");
        }
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        if let Device::Cuda(device) = parts.weight.device() {
            if !crate::cutile::nvfp4_supported(device) {
                candle_core::bail!(
                    "NVFP4 CUDA inference requires Blackwell or newer, CUDA 13.3, and a compatible tileiras"
                );
            }
        }
        #[cfg(not(all(feature = "cuda", feature = "cutile")))]
        if parts.weight.device().is_cuda() {
            candle_core::bail!("NVFP4 CUDA inference requires the cuda and cutile features");
        }
        if parts.weight.device().is_metal() {
            candle_core::bail!("NVFP4 accelerator inference requires CUDA");
        }
        let k = dims[dims.len() - 1] * VALUES_PER_BYTE;
        if !k.is_multiple_of(NVFP4_BLOCK_SIZE) {
            candle_core::bail!("NVFP4 input dimension {k} must be divisible by {NVFP4_BLOCK_SIZE}");
        }
        let mut scale_dims = dims.to_vec();
        *scale_dims.last_mut().unwrap() = k / NVFP4_BLOCK_SIZE;
        if parts.scales.dtype() != DType::F8E4M3 || parts.scales.dims() != scale_dims {
            candle_core::bail!("NVFP4 block scales must be F8E4M3 with shape {scale_dims:?}");
        }
        if parts.global_scales.dtype() != DType::F32
            || parts.global_scales.dims() != &dims[..dims.len() - 1]
        {
            candle_core::bail!("NVFP4 global scales must be F32 with one value per weight row");
        }
        if parts.activation == Nvfp4ActivationMode::DynamicBlock && parts.input_scale.is_none() {
            candle_core::bail!("NVFP4 W4A4 requires a calibrated input global scale");
        }
        if parts.activation == Nvfp4ActivationMode::None && parts.input_scale.is_some() {
            candle_core::bail!("NVFP4 W4A16 must not contain an input global scale");
        }
        for tensor in [&parts.scales, &parts.global_scales]
            .into_iter()
            .chain(parts.input_scale.iter())
            .chain(parts.bias.iter())
        {
            if !tensor.device().same_device(parts.weight.device()) {
                candle_core::bail!("NVFP4 tensors must be on the same device");
            }
        }
        if let Some(scale) = &parts.input_scale {
            let expected = if dims.len() == 3 {
                vec![dims[0]]
            } else {
                vec![]
            };
            if scale.dtype() != DType::F32 || scale.dims() != expected {
                candle_core::bail!("NVFP4 input global scale must be F32 with shape {expected:?}");
            }
        }
        if let Some(bias) = &parts.bias {
            if bias.dtype() != parts.dtype || bias.dims() != &dims[..dims.len() - 1] {
                candle_core::bail!("NVFP4 bias must match the output dtype and weight rows");
            }
        }
        parts.weight = parts.weight.contiguous()?;
        parts.scales = crate::utils::contiguous_fp8(&parts.scales)?;
        parts.global_scales = parts.global_scales.contiguous()?;
        parts.input_scale = parts
            .input_scale
            .map(|scale| scale.contiguous())
            .transpose()?;
        parts.bias = parts.bias.map(|bias| bias.contiguous()).transpose()?;
        let input_global_scale = if parts.weight.rank() == 2 {
            parts
                .input_scale
                .as_ref()
                .map(Tensor::to_scalar::<f32>)
                .transpose()?
        } else {
            None
        };
        #[cfg(all(feature = "cuda", feature = "cutile", has_nvfp4_cutlass_sm121_kernels))]
        let native = cutlass::State::new(&parts)?;
        let layer = Self {
            parts,
            input_global_scale,
            input_scale_immutable,
            #[cfg(all(feature = "cuda", feature = "cutile", has_nvfp4_cutlass_sm121_kernels))]
            native,
        };
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        if layer.parts.weight.device().is_cuda() {
            crate::cutile::register_nvfp4_shape(layer.gemm_args(), layer.parts.dtype)?;
        }
        Ok(layer)
    }

    fn shares_input_calibration(&self, other: &Self) -> bool {
        if self.input_global_scale != other.input_global_scale {
            return false;
        }
        match (&self.parts.input_scale, &other.parts.input_scale) {
            (None, None) => true,
            (Some(left), Some(right)) => {
                left.id() == right.id()
                    || (self.input_scale_immutable && other.input_scale_immutable)
            }
            _ => false,
        }
    }

    pub fn load(
        in_dim: usize,
        out_dim: usize,
        spec: Nvfp4LinearSpec,
        bias: bool,
        hints: Shard,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Self::load_shape(vec![out_dim, in_dim], spec, bias, hints, vb)
    }

    pub fn linear_b(
        in_dim: usize,
        out_dim: usize,
        spec: Nvfp4LinearSpec,
        bias: bool,
        hints: Shard,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !vb.contains_tensor(spec.scale_names.weight) {
            return crate::make_dummy_or_error("nvfp4_linear", &vb, &[spec.scale_names.weight]);
        }
        Ok(Arc::new(Self::load(
            in_dim, out_dim, spec, bias, hints, vb,
        )?))
    }

    pub fn load_stacked(
        num_experts: usize,
        in_dim: usize,
        out_dim: usize,
        spec: Nvfp4LinearSpec,
        bias: bool,
        hints: Shard,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Self::load_shape(vec![num_experts, out_dim, in_dim], spec, bias, hints, vb)
    }

    fn load_shape(
        dims: Vec<usize>,
        spec: Nvfp4LinearSpec,
        bias: bool,
        hints: Shard,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let input_axis = dims.len() - 1;
        let k = dims[input_axis];
        if dims.contains(&0) {
            candle_core::bail!("NVFP4 weight dimensions must be nonzero");
        }
        if !k.is_multiple_of(NVFP4_BLOCK_SIZE) {
            candle_core::bail!("NVFP4 input dimension {k} must be divisible by {NVFP4_BLOCK_SIZE}");
        }
        let range = crate::shard_range(hints, &dims)?;
        if let Some((axis, offset, len)) = range {
            if axis == input_axis
                && (!offset.is_multiple_of(NVFP4_BLOCK_SIZE)
                    || !len.is_multiple_of(NVFP4_BLOCK_SIZE))
            {
                candle_core::bail!("NVFP4 input shards must align to {NVFP4_BLOCK_SIZE} elements");
            }
        }
        let packed_shard = |packing: usize| match range {
            Some((axis, offset, len)) => {
                let divisor = if axis == input_axis { packing } else { 1 };
                Shard::Offset {
                    dim: axis,
                    offset: offset / divisor,
                    len: len / divisor,
                }
            }
            None => Shard::default(),
        };
        let mut packed_dims = dims.clone();
        packed_dims[input_axis] /= VALUES_PER_BYTE;
        let weight = vb.get_with_hints_dtype(
            packed_dims,
            spec.scale_names.weight,
            packed_shard(VALUES_PER_BYTE),
            DType::U8,
        )?;
        let mut scale_dims = dims.clone();
        scale_dims[input_axis] /= NVFP4_BLOCK_SIZE;
        let scales = vb.get_with_hints_dtype(
            scale_dims,
            spec.scale_names.block_scale,
            packed_shard(NVFP4_BLOCK_SIZE),
            DType::F8E4M3,
        )?;
        let global = vb.get_unchecked_dtype(spec.scale_names.global_scale, DType::F32)?;
        let rows: usize = dims[..input_axis].iter().product();
        let values = global.flatten_all()?.to_vec1::<f32>()?;
        if values.is_empty()
            || !rows.is_multiple_of(values.len())
            || (dims.len() == 3 && values.len() != 1 && !values.len().is_multiple_of(dims[0]))
        {
            candle_core::bail!(
                "NVFP4 global scale shape {:?} does not match weight rows",
                global.dims()
            );
        }
        let mut normalized = Vec::with_capacity(rows);
        for value in values {
            let value = dequant_scale(value, spec.global_scale)?;
            normalized.extend(std::iter::repeat_n(value, rows / global.elem_count()));
        }
        let global_scales = Tensor::from_vec(normalized, &dims[..input_axis], vb.device())?;
        let global_scales = match range {
            Some((axis, offset, len)) if axis != input_axis => {
                global_scales.narrow(axis, offset, len)?
            }
            _ => global_scales,
        };
        let input_scale = if spec.activation == Nvfp4ActivationMode::DynamicBlock {
            let name = spec.scale_names.activation_scale.ok_or_else(|| {
                candle_core::Error::msg("NVFP4 W4A4 config has no input global scale name")
            })?;
            let source = vb.get_unchecked_dtype(name, DType::F32)?;
            let values = source.flatten_all()?.to_vec1::<f32>()?;
            let count = if dims.len() == 3 { dims[0] } else { 1 };
            if values.len() != 1 && values.len() != count {
                candle_core::bail!(
                    "NVFP4 input scale {name} must be scalar or one value per expert"
                );
            }
            let values = (0..count)
                .map(|i| dequant_scale(values[i % values.len()], spec.global_scale))
                .collect::<Result<Vec<_>>>()?;
            let shape = if dims.len() == 3 { vec![count] } else { vec![] };
            let scale = Tensor::from_vec(values, shape, vb.device())?;
            Some(match range {
                Some((0, offset, len)) if dims.len() == 3 => scale.narrow(0, offset, len)?,
                _ => scale,
            })
        } else {
            None
        };
        let bias = if bias {
            match crate::bias_shard(range, dims.len()) {
                crate::BiasShard::Skip => None,
                crate::BiasShard::Full => Some(vb.get(&dims[..input_axis], "bias")?),
                crate::BiasShard::Narrow { dim, start, len } => Some(vb.get_with_hints(
                    &dims[..input_axis],
                    "bias",
                    Shard::Offset {
                        dim,
                        offset: start,
                        len,
                    },
                )?),
            }
        } else {
            None
        };
        Self::from_normalized_parts(
            Nvfp4LayerParts {
                weight,
                scales,
                global_scales,
                input_scale,
                activation: spec.activation,
                bias,
                dtype: vb.dtype(),
            },
            true,
        )
    }

    pub(crate) fn merge(layers: Vec<Self>) -> Result<Option<crate::PackedLinear>> {
        let Some(first) = layers.first() else {
            return Ok(None);
        };
        if first.parts.weight.rank() != 2 {
            return Ok(None);
        }
        let input_dim = first.parts.weight.dim(1)?;
        for layer in &layers {
            if layer.parts.weight.rank() != 2
                || layer.parts.weight.dim(1)? != input_dim
                || layer.parts.activation != first.parts.activation
                || layer.parts.dtype != first.parts.dtype
                || layer.parts.bias.is_some()
                || !layer
                    .parts
                    .weight
                    .device()
                    .same_device(first.parts.weight.device())
            {
                return Ok(None);
            }
            if !first.shares_input_calibration(layer) {
                return Ok(None);
            }
        }
        let tensors = |select: fn(&Nvfp4LayerParts) -> &Tensor| {
            Tensor::cat(
                &layers
                    .iter()
                    .map(|layer| select(&layer.parts))
                    .collect::<Vec<_>>(),
                0,
            )
        };
        let packed = Self::from_normalized_parts(
            Nvfp4LayerParts {
                weight: tensors(|p| &p.weight)?,
                scales: tensors(|p| &p.scales)?,
                global_scales: tensors(|p| &p.global_scales)?,
                input_scale: first.parts.input_scale.clone(),
                activation: first.parts.activation,
                bias: None,
                dtype: first.parts.dtype,
            },
            layers.iter().all(|layer| layer.input_scale_immutable),
        )?;
        let rows_per_rank = layers
            .iter()
            .map(|layer| layer.parts.weight.dim(0))
            .collect::<Result<Vec<_>>>()?;
        let mut constituents = Vec::with_capacity(layers.len());
        let mut offset = 0;
        for &rows in &rows_per_rank {
            let layer = Self::from_normalized_parts(
                Nvfp4LayerParts {
                    weight: packed.parts.weight.narrow(0, offset, rows)?,
                    scales: packed.parts.scales.narrow(0, offset, rows)?,
                    global_scales: packed.parts.global_scales.narrow(0, offset, rows)?,
                    input_scale: packed.parts.input_scale.clone(),
                    activation: packed.parts.activation,
                    bias: None,
                    dtype: packed.parts.dtype,
                },
                packed.input_scale_immutable,
            )?;
            constituents.push(Arc::new(layer) as Arc<dyn QuantMethod>);
            offset += rows;
        }
        Ok(Some(crate::PackedLinear {
            packed: Arc::new(packed),
            constituents,
            rows_per_rank,
        }))
    }

    pub fn stack(layers: Vec<Self>) -> Result<Self> {
        let first = layers
            .first()
            .ok_or_else(|| candle_core::Error::msg("NVFP4 expert list is empty"))?;
        let activation = first.parts.activation;
        let dtype = first.parts.dtype;
        if layers.iter().any(|layer| {
            layer.parts.weight.rank() != 2
                || layer.parts.activation != activation
                || layer.parts.dtype != dtype
                || layer.parts.bias.is_some() != first.parts.bias.is_some()
        }) {
            candle_core::bail!(
                "NVFP4 expert stack requires rank-2 weights and matching activation modes, output dtypes, and bias presence"
            );
        }
        let tensors = |select: fn(&Nvfp4LayerParts) -> &Tensor| {
            Tensor::stack(
                &layers
                    .iter()
                    .map(|layer| select(&layer.parts))
                    .collect::<Vec<_>>(),
                0,
            )
        };
        let bias = if first.parts.bias.is_some() {
            Some(Tensor::stack(
                &layers
                    .iter()
                    .map(|layer| layer.parts.bias.as_ref().unwrap())
                    .collect::<Vec<_>>(),
                0,
            )?)
        } else {
            None
        };
        let input_scale = if activation == Nvfp4ActivationMode::DynamicBlock {
            Some(Tensor::stack(
                &layers
                    .iter()
                    .map(|layer| layer.parts.input_scale.as_ref().unwrap())
                    .collect::<Vec<_>>(),
                0,
            )?)
        } else {
            None
        };
        Self::from_normalized_parts(
            Nvfp4LayerParts {
                weight: tensors(|p| &p.weight)?,
                scales: tensors(|p| &p.scales)?,
                global_scales: tensors(|p| &p.global_scales)?,
                input_scale,
                activation,
                bias,
                dtype,
            },
            layers.iter().all(|layer| layer.input_scale_immutable),
        )
    }

    #[cfg(all(feature = "cuda", feature = "cutile"))]
    fn gemm_args(&self) -> crate::cutile::Nvfp4GemmArgs<'_> {
        crate::cutile::Nvfp4GemmArgs {
            weights: &self.parts.weight,
            weight_scales: &self.parts.scales,
            weight_global_scale: &self.parts.global_scales,
            activation_global_scale: self.parts.input_scale.as_ref(),
        }
    }

    #[cfg(all(feature = "cuda", feature = "cutile"))]
    fn forward_cuda(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(has_nvfp4_cutlass_sm121_kernels)]
        if let Some(native) = &self.native {
            if native.supports(x.dim(0)?, x.dtype()) {
                let (packed, scales) = crate::cutile::cutile_nvfp4_quantize(
                    x,
                    self.parts.input_scale.as_ref().unwrap(),
                )?;
                let scales = cutlass::swizzle_scales(&scales)?;
                return native.forward(&packed, &scales, x.dtype(), self.gemm_args());
            }
        }
        crate::cutile::cutile_nvfp4(x, self.gemm_args())
    }

    #[cfg(all(feature = "cuda", feature = "cutile"))]
    fn forward_quantized_cuda(&self, activation: &crate::QuantizedActivation) -> Result<Tensor> {
        #[cfg(has_nvfp4_cutlass_sm121_kernels)]
        if let Some(native) = &self.native {
            if native.supports(activation.quantized().dim(0)?, activation.source_dtype()) {
                return native.forward(
                    activation.quantized(),
                    &cutlass::swizzle_scales(activation.scales())?,
                    activation.source_dtype(),
                    self.gemm_args(),
                );
            }
        }
        crate::cutile::cutile_nvfp4_prequantized(
            activation.quantized(),
            activation.scales(),
            activation.source_dtype(),
            self.gemm_args(),
        )
    }

    fn logical_shape(&self) -> Vec<usize> {
        let mut shape = self.parts.weight.dims().to_vec();
        *shape.last_mut().unwrap() *= VALUES_PER_BYTE;
        shape
    }

    fn expert(&self, id: usize) -> Result<Self> {
        Self::from_normalized_parts(
            Nvfp4LayerParts {
                weight: self.parts.weight.i(id)?,
                scales: self.parts.scales.i(id)?,
                global_scales: self.parts.global_scales.i(id)?,
                input_scale: self
                    .parts
                    .input_scale
                    .as_ref()
                    .map(|t| t.i(id))
                    .transpose()?,
                activation: self.parts.activation,
                bias: self.parts.bias.as_ref().map(|t| t.i(id)).transpose()?,
                dtype: self.parts.dtype,
            },
            self.input_scale_immutable,
        )
    }

    fn reference_forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let x = if let Some(global) = &self.parts.input_scale {
            let global = global.to_scalar::<f32>()?;
            let mut values = x.flatten_all()?.to_vec1::<f32>()?;
            for block in values.chunks_mut(NVFP4_BLOCK_SIZE) {
                for value in block.iter_mut() {
                    *value /= global;
                }
                let max = block.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
                let scale = F8E4M3::from_f32((max / FP4_MAX).min(FP8_MAX)).to_f32();
                for value in block {
                    *value = if scale == 0.0 {
                        0.0
                    } else {
                        quantize_fp4(*value / scale) * scale
                    };
                }
            }
            Tensor::from_vec(values, x.shape(), x.device())?
        } else {
            x
        };
        let k = x.dim(D::Minus1)?;
        let rows = x.elem_count() / k;
        let mut shape = x.dims().to_vec();
        *shape.last_mut().unwrap() = self.parts.weight.dim(0)?;
        let weight = self.dequantize_block_scaled()?;
        let output = x.reshape((rows, k))?.matmul(&weight.t()?)?;
        let output = output.broadcast_mul(&self.parts.global_scales)?;
        let output = match &self.parts.input_scale {
            Some(global) => output.broadcast_mul(global)?,
            None => output,
        };
        let output = match &self.parts.bias {
            Some(bias) => output.broadcast_add(&bias.to_dtype(DType::F32)?)?,
            None => output,
        };
        output.reshape(shape)?.to_dtype(dtype)
    }

    fn validate_input(&self, x: &Tensor, weight_rank: usize) -> Result<()> {
        if self.parts.weight.rank() != weight_rank {
            candle_core::bail!("NVFP4 operation requires rank-{weight_rank} weights");
        }
        let k = self.parts.weight.dim(D::Minus1)? * VALUES_PER_BYTE;
        if x.rank() < 2 || x.dims().last() != Some(&k) {
            candle_core::bail!(
                "NVFP4 activation shape {:?} must have rank >= 2 and input dimension {k}",
                x.dims()
            );
        }
        if !x.device().same_device(self.parts.weight.device()) {
            candle_core::bail!("NVFP4 activations and weights must be on the same device");
        }
        if !matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            candle_core::bail!("NVFP4 activations must have BF16, F16, or F32 dtype");
        }
        Ok(())
    }

    fn dequantize_block_scaled(&self) -> Result<Tensor> {
        let shape = self.logical_shape();
        let packed = self
            .parts
            .weight
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let scales = self
            .parts
            .scales
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut values = Vec::with_capacity(packed.len() * VALUES_PER_BYTE);
        for byte in packed {
            for nibble in [byte & 15, byte >> 4] {
                let value_index = values.len();
                let scale = scales[value_index / NVFP4_BLOCK_SIZE];
                values.push(unpack_fp4(nibble) * scale);
            }
        }
        Tensor::from_vec(values, shape, self.parts.weight.device())
    }
}

fn dequant_scale(scale: f32, convention: ScaleConvention) -> Result<f32> {
    let scale = match convention {
        ScaleConvention::Dequantize => scale,
        ScaleConvention::Quantize => scale.recip(),
    };
    if !scale.is_finite() || scale <= 0.0 {
        candle_core::bail!("NVFP4 global scales must be finite and positive, got {scale}");
    }
    Ok(scale)
}

fn unpack_fp4(bits: u8) -> f32 {
    let magnitude = FP4_VALUES[(bits & 7) as usize];
    if bits & 8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn quantize_fp4(value: f32) -> f32 {
    let magnitude = value.abs().min(FP4_MAX);
    let mut best = 0;
    for (i, &candidate) in FP4_VALUES.iter().enumerate().skip(1) {
        let error = (magnitude - candidate).abs();
        let current = (magnitude - FP4_VALUES[best]).abs();
        if error < current || (error == current && i % 2 == 0) {
            best = i;
        }
    }
    FP4_VALUES[best].copysign(value)
}

impl QuantMethod for Nvfp4Layer {
    fn new(_method: QuantMethodConfig) -> Result<Self> {
        candle_core::bail!("Construct NVFP4 layers using Nvfp4Layer::from_parts")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.dequantize_block_scaled()?
            .broadcast_mul(&self.parts.global_scales.unsqueeze(D::Minus1)?)?
            .to_dtype(self.parts.dtype)
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        self.validate_input(x, 2)?;
        if x.elem_count() == 0 {
            let mut shape = x.dims().to_vec();
            *shape.last_mut().unwrap() = self.parts.weight.dim(0)?;
            return Tensor::zeros(shape, x.dtype(), x.device());
        }
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        if x.device().is_cuda() {
            let k = x.dim(D::Minus1)?;
            let rows = x.elem_count() / k;
            let x_flat = x.contiguous()?.reshape((rows, k))?;
            let output = self.forward_cuda(&x_flat)?;
            let output = match &self.parts.bias {
                Some(bias) => output.broadcast_add(bias)?,
                None => output,
            };
            let mut shape = x.dims().to_vec();
            *shape.last_mut().unwrap() = self.parts.weight.dim(0)?;
            return output.reshape(shape);
        }
        if !x.device().is_cpu() {
            candle_core::bail!(
                "NVFP4 accelerator inference requires CUDA 13.3 and the cutile feature"
            );
        }
        self.reference_forward(x)
    }

    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.validate_input(x, 3)?;
        if indices.dtype() != DType::U32 || !indices.device().same_device(x.device()) {
            candle_core::bail!("NVFP4 gather indices must be U32 on the activation device");
        }
        let out_dim = self.parts.weight.dim(1)?;
        let (tokens, topk, input_routes, k, output_shape) = match (x.dims(), indices.dims()) {
            (&[tokens, k], &[indexed_tokens, topk]) if tokens == indexed_tokens => {
                (tokens, topk, 1, k, vec![tokens, topk, out_dim])
            }
            (&[tokens, routes, k], &[indexed_tokens, topk])
                if tokens == indexed_tokens && (routes == 1 || routes == topk) =>
            {
                (tokens, topk, routes, k, vec![tokens, topk, out_dim])
            }
            (&[batch, seq, routes, 1, k], &[indexed_batch, indexed_seq, topk])
                if batch == indexed_batch
                    && seq == indexed_seq
                    && (routes == 1 || routes == topk) =>
            {
                (
                    batch * seq,
                    topk,
                    routes,
                    k,
                    vec![batch, seq, topk, 1, out_dim],
                )
            }
            (&[batch, seq, routes, k], &[indexed_batch, indexed_seq, topk])
                if batch == indexed_batch && seq == indexed_seq && routes == topk =>
            {
                (
                    batch * seq,
                    topk,
                    routes,
                    k,
                    vec![batch, seq, topk, 1, out_dim],
                )
            }
            _ => candle_core::bail!(
                "NVFP4 gather activation shape {:?} does not match indices {:?}",
                x.dims(),
                indices.dims()
            ),
        };
        if indices.elem_count() == 0 {
            return Tensor::zeros(output_shape, x.dtype(), x.device());
        }
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        if x.device().is_cuda() {
            let x = x.contiguous()?.reshape((tokens, input_routes, k))?;
            let indices = indices.contiguous()?.reshape((tokens, topk))?;
            let output = crate::cutile::cutile_nvfp4_gather(&x, &indices, self.gemm_args())?;
            let output = match &self.parts.bias {
                Some(bias) => output.broadcast_add(
                    &bias
                        .index_select(&indices.flatten_all()?, 0)?
                        .reshape(output.shape())?,
                )?,
                None => output,
            };
            return output.reshape(output_shape);
        }
        if !x.device().is_cpu() {
            candle_core::bail!(
                "NVFP4 accelerator inference requires CUDA 13.3 and the cutile feature"
            );
        }
        let x = x.reshape((tokens * input_routes, k))?;
        let ids = indices.flatten_all()?.to_vec1::<u32>()?;
        let mut outputs = Vec::with_capacity(ids.len());
        for (i, id) in ids.into_iter().enumerate() {
            let row = x.i(if input_routes == 1 { i / topk } else { i })?;
            outputs.push(self.expert(id as usize)?.forward_raw(&row.unsqueeze(0)?)?);
        }
        Tensor::cat(&outputs, 0)?.reshape(output_shape)
    }

    fn activation_quantization_scheme_for(
        &self,
        x: &Tensor,
    ) -> Option<crate::ActivationQuantizationScheme> {
        if !x.device().is_cuda()
            || !matches!(x.dtype(), DType::BF16 | DType::F16)
            || self.parts.weight.rank() != 2
            || self.input_global_scale.is_none()
            || self.validate_input(x, 2).is_err()
            || x.elem_count() / x.dim(D::Minus1).ok()? <= 1
        {
            return None;
        }
        Some(NVFP4_ACTIVATION_SCHEME)
    }

    fn nvfp4_input_calibration(&self) -> Option<Nvfp4InputCalibration<'_>> {
        Some(Nvfp4InputCalibration { layer: self })
    }

    fn activation_quantization_global_scale(&self) -> Option<f32> {
        self.input_global_scale
    }

    fn quantize_activation(&self, x: &Tensor) -> Result<crate::QuantizedActivation> {
        if self.activation_quantization_scheme_for(x).is_none() {
            candle_core::bail!(
                "NVFP4 shared activation requires CUDA W4A4 with more than one input row"
            );
        }
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        {
            let source = x.contiguous()?;
            let k = source.dim(D::Minus1)?;
            let rows = source.elem_count() / k;
            let (packed, scales) = crate::cutile::cutile_nvfp4_quantize(
                &source.reshape((rows, k))?,
                self.parts.input_scale.as_ref().unwrap(),
            )?;
            crate::QuantizedActivation::new_nvfp4(
                packed,
                scales,
                &source,
                self.input_global_scale.unwrap(),
            )
        }
        #[cfg(not(all(feature = "cuda", feature = "cutile")))]
        candle_core::bail!("NVFP4 shared activation requires CUDA and cuTile");
    }

    fn forward_quantized(&self, activation: &crate::QuantizedActivation) -> Result<Tensor> {
        let (rows, packed_k) = activation.quantized().dims2()?;
        if activation.scheme() != NVFP4_ACTIVATION_SCHEME
            || self.input_global_scale.is_none()
            || activation.global_scale() != self.input_global_scale
            || self.parts.weight.rank() != 2
            || self.parts.weight.dim(1)? != packed_k
            || !activation
                .quantized()
                .device()
                .same_device(self.parts.weight.device())
            || !activation.quantized().device().is_cuda()
            || rows <= 1
        {
            candle_core::bail!("NVFP4 shared activation does not match this projection's calibration or input shape");
        }
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        {
            let output = self.forward_quantized_cuda(activation)?;
            let output = match &self.parts.bias {
                Some(bias) => output.broadcast_add(bias)?,
                None => output,
            };
            let mut shape = activation.source_shape().to_vec();
            *shape.last_mut().unwrap() = self.parts.weight.dim(0)?;
            output.reshape(shape)
        }
        #[cfg(not(all(feature = "cuda", feature = "cutile")))]
        candle_core::bail!("NVFP4 shared activation requires CUDA and cuTile");
    }

    #[cfg(all(feature = "cuda", feature = "cutile"))]
    fn try_quantize_glu(
        &self,
        gate: &Tensor,
        value: &Tensor,
        activation: crate::GluActivationType,
    ) -> Result<Option<crate::QuantizedActivation>> {
        if self.is_dynamic_lora_active()
            || self.stats_snapshot().is_some()
            || self.activation_quantization_scheme_for(gate) != Some(NVFP4_ACTIVATION_SCHEME)
        {
            return Ok(None);
        }
        let Some((packed, scales)) = crate::cutile::cutile_nvfp4_glu(
            crate::cutile::GluQuantArgs {
                gate,
                value,
                activation_global_scale: self.parts.input_scale.as_ref().unwrap(),
                activation,
            },
            false,
        )?
        else {
            return Ok(None);
        };
        crate::QuantizedActivation::new_nvfp4(
            packed,
            scales,
            gate,
            self.input_global_scale.unwrap(),
        )
        .map(Some)
    }

    #[cfg(all(feature = "cuda", feature = "cutile"))]
    fn try_forward_fused_split_glu(
        &self,
        input: &Tensor,
        split_size: usize,
        activation: crate::GluActivationType,
    ) -> Result<Option<Tensor>> {
        let Some(packed_features) = split_size.checked_mul(2) else {
            return Ok(None);
        };
        if split_size == 0 || input.dims().last().copied() != Some(packed_features) {
            return Ok(None);
        }
        let gate = input.narrow(D::Minus1, 0, split_size)?;
        let value = input.narrow(D::Minus1, split_size, split_size)?;
        let Some(quantized) = self.try_quantize_glu(&gate, &value, activation)? else {
            return Ok(None);
        };
        self.forward_quantized(&quantized).map(Some)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }
    fn dtype_and_device(&self) -> (DType, Device) {
        (self.parts.dtype, self.parts.weight.device().clone())
    }
    fn has_bias(&self) -> bool {
        self.parts.bias.is_some()
    }
    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            self.parts.dtype,
            self.parts.weight.device().clone(),
            self.logical_shape(),
            request,
            true,
        ))
    }
    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(crate::UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(
                self.dequantize_w()?.add(delta)?,
                self.parts.bias.clone(),
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
            return Ok(Arc::new(Self::from_normalized_parts(
                Nvfp4LayerParts {
                    weight: self.parts.weight.to_device(&device)?,
                    scales: self.parts.scales.to_device(&device)?,
                    global_scales: self.parts.global_scales.to_device(&device)?,
                    input_scale: self
                        .parts
                        .input_scale
                        .as_ref()
                        .map(|t| t.to_device(&device))
                        .transpose()?,
                    activation: self.parts.activation,
                    bias: self
                        .parts
                        .bias
                        .as_ref()
                        .map(|t| t.to_device(&device))
                        .transpose()?,
                    dtype: self.parts.dtype,
                },
                self.input_scale_immutable,
            )?));
        }
        Arc::new(crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(self.dequantize_w()?, self.parts.bias.clone()),
        ))?)
        .apply_isq(dtype, device, n_quantized, imatrix, guard)
    }
}

impl QuantizedSerde for Nvfp4Layer {
    fn name(&self) -> &'static str {
        "NVFP4"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_checkpoint_views_are_materialized_at_construction() -> Result<()> {
        const ROWS: usize = 4;
        let weight = Tensor::ones((ROWS, NVFP4_BLOCK_SIZE), DType::U8, &Device::Cpu)?.narrow(
            1,
            NVFP4_BLOCK_SIZE / 2,
            NVFP4_BLOCK_SIZE / 2,
        )?;
        let scales = Tensor::ones((ROWS, 2), DType::F8E4M3, &Device::Cpu)?.narrow(1, 1, 1)?;
        let global_scales = Tensor::ones((ROWS, 2), DType::F32, &Device::Cpu)?
            .narrow(1, 1, 1)?
            .squeeze(1)?;
        assert!(!weight.is_contiguous());
        assert!(!scales.is_contiguous());
        assert!(!global_scales.is_contiguous());
        let layer = Nvfp4Layer::from_parts(Nvfp4LayerParts {
            weight,
            scales,
            bias: Some(global_scales.clone()),
            global_scales,
            input_scale: None,
            activation: Nvfp4ActivationMode::None,
            dtype: DType::F32,
        })?;
        assert!(layer.parts.weight.is_contiguous());
        assert!(layer.parts.scales.is_contiguous());
        assert!(layer.parts.global_scales.is_contiguous());
        assert!(layer.parts.bias.as_ref().unwrap().is_contiguous());
        Ok(())
    }

    fn public_calibrated_layer(scale: Tensor) -> Result<Nvfp4Layer> {
        Nvfp4Layer::from_parts(Nvfp4LayerParts {
            weight: Tensor::full(0x22u8, (2, NVFP4_BLOCK_SIZE / 2), &Device::Cpu)?,
            scales: Tensor::ones((2, 1), DType::F8E4M3, &Device::Cpu)?,
            global_scales: Tensor::ones(2, DType::F32, &Device::Cpu)?,
            input_scale: Some(scale),
            activation: Nvfp4ActivationMode::DynamicBlock,
            bias: None,
            dtype: DType::F32,
        })
    }

    fn checkpoint_calibrated_layer(model_opt: bool, scale: Tensor) -> Result<Nvfp4Layer> {
        const PREFIX: &str = "model.layers.0.self_attn.q_proj";
        let config = if model_opt {
            serde_json::json!({"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16})
        } else {
            let weights = serde_json::json!({"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16, "dynamic": false, "symmetric": true});
            let mut activation = weights.clone();
            activation["dynamic"] = serde_json::json!("local");
            serde_json::json!({"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized", "config_groups": {"group_0": {"targets": ["Linear"], "weights": weights, "input_activations": activation}}})
        };
        let config: crate::QuantizedConfig = serde_json::from_value(config).unwrap();
        let Some(crate::CheckpointLinearSpec::Nvfp4(spec)) = config.resolve_checkpoint(PREFIX)?
        else {
            panic!("fixture must resolve NVFP4");
        };
        let tensors = std::collections::HashMap::from([
            (
                format!("{PREFIX}.{}", spec.scale_names.weight),
                Tensor::full(0x22u8, (2, NVFP4_BLOCK_SIZE / 2), &Device::Cpu)?,
            ),
            (
                format!("{PREFIX}.{}", spec.scale_names.block_scale),
                Tensor::ones((2, 1), DType::F8E4M3, &Device::Cpu)?,
            ),
            (
                format!("{PREFIX}.{}", spec.scale_names.global_scale),
                Tensor::new(1f32, &Device::Cpu)?,
            ),
            (
                format!("{PREFIX}.{}", spec.scale_names.activation_scale.unwrap()),
                scale,
            ),
        ]);
        Nvfp4Layer::load(
            NVFP4_BLOCK_SIZE,
            2,
            spec,
            false,
            Shard::default(),
            crate::ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu).pp(PREFIX),
        )
    }

    #[test]
    fn shared_calibration_rejects_independent_mutable_aliases() -> Result<()> {
        let live = Tensor::new(1f32, &Device::Cpu)?;
        let separate = Tensor::new(1f32, &Device::Cpu)?;
        let first = public_calibrated_layer(live.clone())?;
        let same = public_calibrated_layer(live.clone())?;
        let independent = public_calibrated_layer(separate.clone())?;
        assert!(first.shares_input_calibration(&same));
        assert!(!first.shares_input_calibration(&independent));
        separate
            .reshape(1)?
            .slice_set(&Tensor::new(&[2f32], &Device::Cpu)?, 0, 0)?;
        assert_eq!(independent.input_global_scale, Some(1.0));
        assert_eq!(
            independent
                .parts
                .input_scale
                .as_ref()
                .unwrap()
                .to_scalar::<f32>()?,
            2.0
        );
        assert!(!first.shares_input_calibration(&independent));
        assert!(Nvfp4Layer::merge(vec![first, independent])?.is_none());
        live.reshape(1)?
            .slice_set(&Tensor::new(&[3f32], &Device::Cpu)?, 0, 0)?;
        let first = public_calibrated_layer(live.clone())?;
        let second = public_calibrated_layer(live)?;
        let packed = Nvfp4Layer::merge(vec![first, second])?.unwrap();
        let a = packed.constituents[0].nvfp4_input_calibration().unwrap();
        let b = packed.constituents[1].nvfp4_input_calibration().unwrap();
        assert!(a.matches(b));
        Ok(())
    }

    #[test]
    fn checkpoint_owned_calibration_remains_compatible_after_source_mutation() -> Result<()> {
        for model_opt in [true, false] {
            let source = Tensor::new(1f32, &Device::Cpu)?;
            let first = checkpoint_calibrated_layer(model_opt, source.clone())?;
            let second = checkpoint_calibrated_layer(model_opt, Tensor::new(1f32, &Device::Cpu)?)?;
            let unequal = checkpoint_calibrated_layer(model_opt, Tensor::new(2f32, &Device::Cpu)?)?;
            assert!(first.input_scale_immutable && second.input_scale_immutable);
            assert_ne!(first.parts.input_scale.as_ref().unwrap().id(), source.id());
            assert!(first.shares_input_calibration(&second));
            assert!(!first.shares_input_calibration(&unequal));
            let input = Tensor::ones((2, NVFP4_BLOCK_SIZE), DType::F32, &Device::Cpu)?;
            let before = first.forward(&input)?.to_vec2::<f32>()?;
            source
                .reshape(1)?
                .slice_set(&Tensor::new(&[2f32], &Device::Cpu)?, 0, 0)?;
            assert_eq!(
                first
                    .parts
                    .input_scale
                    .as_ref()
                    .unwrap()
                    .to_scalar::<f32>()?,
                1.0
            );
            assert_eq!(first.forward(&input)?.to_vec2::<f32>()?, before);
            let merged = Nvfp4Layer::merge(vec![first, second])?.unwrap();
            assert!(merged.constituents[0]
                .nvfp4_input_calibration()
                .unwrap()
                .matches(merged.constituents[1].nvfp4_input_calibration().unwrap()));
        }
        Ok(())
    }

    #[test]
    fn calibration_provenance_survives_stack_expert_and_device_clone() -> Result<()> {
        let loaded = || checkpoint_calibrated_layer(true, Tensor::new(1f32, &Device::Cpu)?);
        let known = Nvfp4Layer::stack(vec![loaded()?, loaded()?])?;
        assert!(known.expert(0)?.input_scale_immutable);
        assert!(known.expert(0)?.shares_input_calibration(&known.expert(1)?));
        let mixed = Nvfp4Layer::stack(vec![
            loaded()?,
            public_calibrated_layer(Tensor::new(1f32, &Device::Cpu)?)?,
        ])?;
        assert!(!mixed.expert(0)?.input_scale_immutable);
        assert!(!mixed.expert(0)?.shares_input_calibration(&mixed.expert(1)?));
        let reference = loaded()?;
        let copy = Arc::new(loaded()?).apply_isq(
            None,
            Device::Cpu,
            &AtomicUsize::new(0),
            None,
            QuantizeOntoGuard::new(),
        )?;
        assert!(reference
            .nvfp4_input_calibration()
            .unwrap()
            .matches(copy.nvfp4_input_calibration().unwrap()));
        let unknown = Arc::new(public_calibrated_layer(Tensor::new(1f32, &Device::Cpu)?)?)
            .apply_isq(
                None,
                Device::Cpu,
                &AtomicUsize::new(0),
                None,
                QuantizeOntoGuard::new(),
            )?;
        assert!(!reference
            .nvfp4_input_calibration()
            .unwrap()
            .matches(unknown.nvfp4_input_calibration().unwrap()));
        Ok(())
    }
}
