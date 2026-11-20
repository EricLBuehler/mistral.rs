use std::{collections::HashMap, sync::Arc};

use candle_core::{quantized::GgmlDType, DType, Device, Error, Result, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Linear};

use super::{
    archive::{qtensor_from_gguf_data, GgufArchive, GgufEndian},
    GgufMatMul,
};
use crate::{
    bias_shard, block_pack_factor, shard_range, slice_blocked_data, BiasShard, QuantMethod,
    QuantMethodConfig, QuantizedWeightSource, Shard, ShardedSafeTensors, ShardedVarBuilder,
    TensorShapes, UnquantLinear,
};

const DIRECT_GGUF_DTYPES: &str =
    "F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, and Q2_K through Q8_K";

#[derive(Clone, Debug, PartialEq)]
pub enum GgufTensorBinding {
    Tensor(String),
    Mxfp4Blocks(String),
    Mxfp4Scales(String),
    Slice {
        input: Box<Self>,
        dim: usize,
        start: usize,
        len: usize,
    },
    Concat {
        inputs: Vec<Self>,
        dim: usize,
    },
    Stack {
        inputs: Vec<Self>,
        dim: usize,
    },
    Interleave {
        inputs: Vec<Self>,
        dim: usize,
    },
    Transpose {
        input: Box<Self>,
        dim1: usize,
        dim2: usize,
    },
    Permute {
        input: Box<Self>,
        dims: Vec<usize>,
    },
    Reshape {
        input: Box<Self>,
        dims: Vec<usize>,
    },
    Affine {
        input: Box<Self>,
        mul: f64,
        add: f64,
    },
    Log {
        input: Box<Self>,
    },
    InverseSoftplus {
        input: Box<Self>,
    },
    Cast {
        input: Box<Self>,
        dtype: DType,
    },
}

impl GgufTensorBinding {
    pub fn tensor(name: impl Into<String>) -> Self {
        Self::Tensor(name.into())
    }

    pub fn mxfp4_blocks(name: impl Into<String>) -> Self {
        Self::Mxfp4Blocks(name.into())
    }

    pub fn mxfp4_scales(name: impl Into<String>) -> Self {
        Self::Mxfp4Scales(name.into())
    }

    pub fn slice(self, dim: usize, start: usize, len: usize) -> Self {
        Self::Slice {
            input: Box::new(self),
            dim,
            start,
            len,
        }
    }

    pub fn concat(inputs: Vec<Self>, dim: usize) -> Self {
        Self::Concat { inputs, dim }
    }

    pub fn stack(inputs: Vec<Self>, dim: usize) -> Self {
        Self::Stack { inputs, dim }
    }

    pub fn interleave(inputs: Vec<Self>, dim: usize) -> Self {
        Self::Interleave { inputs, dim }
    }

    pub fn transpose(self, dim1: usize, dim2: usize) -> Self {
        Self::Transpose {
            input: Box::new(self),
            dim1,
            dim2,
        }
    }

    pub fn permute(self, dims: Vec<usize>) -> Self {
        Self::Permute {
            input: Box::new(self),
            dims,
        }
    }

    pub fn reshape(self, dims: Vec<usize>) -> Self {
        Self::Reshape {
            input: Box::new(self),
            dims,
        }
    }

    pub fn affine(self, mul: f64, add: f64) -> Self {
        Self::Affine {
            input: Box::new(self),
            mul,
            add,
        }
    }

    pub fn log(self) -> Self {
        Self::Log {
            input: Box::new(self),
        }
    }

    pub fn inverse_softplus(self) -> Self {
        Self::InverseSoftplus {
            input: Box::new(self),
        }
    }

    pub fn cast(self, dtype: DType) -> Self {
        Self::Cast {
            input: Box::new(self),
            dtype,
        }
    }

    fn direct_tensor(&self) -> Option<&str> {
        match self {
            Self::Tensor(name) => Some(name),
            _ => None,
        }
    }

    fn mark_uploaded(&self, archive: &GgufArchive) {
        match self {
            Self::Tensor(name) | Self::Mxfp4Blocks(name) | Self::Mxfp4Scales(name) => {
                archive.mark_tensor_uploaded(name);
            }
            Self::Slice { input, .. }
            | Self::Transpose { input, .. }
            | Self::Permute { input, .. }
            | Self::Reshape { input, .. }
            | Self::Affine { input, .. }
            | Self::Log { input }
            | Self::InverseSoftplus { input }
            | Self::Cast { input, .. } => input.mark_uploaded(archive),
            Self::Concat { inputs, .. }
            | Self::Stack { inputs, .. }
            | Self::Interleave { inputs, .. } => {
                for input in inputs {
                    input.mark_uploaded(archive);
                }
            }
        }
    }

    fn text_layer_index(&self) -> Option<usize> {
        fn source_layer(name: &str) -> Option<usize> {
            name.strip_prefix("blk.")?.split_once('.')?.0.parse().ok()
        }

        match self {
            Self::Tensor(name) | Self::Mxfp4Blocks(name) | Self::Mxfp4Scales(name) => {
                source_layer(name)
            }
            Self::Slice { input, .. }
            | Self::Transpose { input, .. }
            | Self::Permute { input, .. }
            | Self::Reshape { input, .. }
            | Self::Affine { input, .. }
            | Self::Log { input }
            | Self::InverseSoftplus { input }
            | Self::Cast { input, .. } => input.text_layer_index(),
            Self::Concat { inputs, .. }
            | Self::Stack { inputs, .. }
            | Self::Interleave { inputs, .. } => {
                let layer = inputs.first()?.text_layer_index()?;
                inputs
                    .iter()
                    .all(|input| input.text_layer_index() == Some(layer))
                    .then_some(layer)
            }
        }
    }
}

pub trait GgufBindingResolver: Send + Sync {
    fn native_names(&self) -> Vec<String>;

    fn resolve(&self, native_name: &str) -> Option<GgufTensorBinding>;
}

#[derive(Clone, Debug, Default)]
pub struct GgufBindingMap {
    bindings: HashMap<String, GgufTensorBinding>,
}

impl GgufBindingMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        native_name: impl Into<String>,
        binding: GgufTensorBinding,
    ) -> Option<GgufTensorBinding> {
        self.bindings.insert(native_name.into(), binding)
    }

    pub fn with_binding(
        mut self,
        native_name: impl Into<String>,
        binding: GgufTensorBinding,
    ) -> Self {
        self.insert(native_name, binding);
        self
    }

    pub fn get(&self, native_name: &str) -> Option<&GgufTensorBinding> {
        self.bindings.get(native_name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &GgufTensorBinding)> {
        self.bindings
            .iter()
            .map(|(name, binding)| (name.as_str(), binding))
    }
}

impl FromIterator<(String, GgufTensorBinding)> for GgufBindingMap {
    fn from_iter<T: IntoIterator<Item = (String, GgufTensorBinding)>>(iter: T) -> Self {
        Self {
            bindings: iter.into_iter().collect(),
        }
    }
}

impl GgufBindingResolver for GgufBindingMap {
    fn native_names(&self) -> Vec<String> {
        let mut names = self.bindings.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    fn resolve(&self, native_name: &str) -> Option<GgufTensorBinding> {
        self.bindings.get(native_name).cloned()
    }
}

impl GgufBindingResolver for HashMap<String, GgufTensorBinding> {
    fn native_names(&self) -> Vec<String> {
        let mut names = self.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    fn resolve(&self, native_name: &str) -> Option<GgufTensorBinding> {
        self.get(native_name).cloned()
    }
}

pub struct GgufWeightSource {
    archive: Arc<GgufArchive>,
    bindings: HashMap<String, GgufTensorBinding>,
    shapes: HashMap<String, Vec<usize>>,
    output_dtypes: HashMap<String, DType>,
    dtype: DType,
}

struct PackedBinding {
    dtype: GgmlDType,
    dims: Vec<usize>,
    data: Vec<u8>,
}

impl std::fmt::Debug for GgufWeightSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufWeightSource")
            .field("archive", &self.archive)
            .field("bindings", &self.bindings)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl GgufWeightSource {
    pub fn new(
        archive: Arc<GgufArchive>,
        resolver: &dyn GgufBindingResolver,
        dtype: DType,
    ) -> Result<Self> {
        for shard in archive.shards() {
            if shard.endian() == GgufEndian::Big {
                candle_core::bail!(
                    "direct GGUF loading does not support big-endian shard `{}`",
                    shard.path().display()
                );
            }
        }
        let mut bindings = HashMap::new();
        let mut shapes = HashMap::new();
        let mut output_dtypes = HashMap::new();
        for native_name in resolver.native_names() {
            if bindings.contains_key(&native_name) {
                candle_core::bail!("duplicate native GGUF binding `{native_name}`");
            }
            let binding = resolver.resolve(&native_name).ok_or_else(|| {
                Error::msg(format!(
                    "GGUF binding resolver listed `{native_name}` but did not resolve it"
                ))
            })?;
            validate_binding_storage(&archive, &binding, &native_name)
                .map_err(|err| err.with_path(native_name.clone()))?;
            let shape = infer_binding_shape(&archive, &binding)
                .map_err(|err| err.with_path(native_name.clone()))?;
            let output_dtype = infer_binding_dtype(&archive, &binding, dtype)
                .map_err(|err| err.with_path(native_name.clone()))?;
            bindings.insert(native_name.clone(), binding);
            shapes.insert(native_name.clone(), shape);
            output_dtypes.insert(native_name, output_dtype);
        }
        Ok(Self {
            archive,
            bindings,
            shapes,
            output_dtypes,
            dtype,
        })
    }

    pub fn archive(&self) -> &Arc<GgufArchive> {
        &self.archive
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn bindings(&self) -> &HashMap<String, GgufTensorBinding> {
        &self.bindings
    }

    pub fn tensor_shapes(&self) -> &HashMap<String, Vec<usize>> {
        &self.shapes
    }

    pub fn materialize_tensor(&self, native_name: &str, device: &Device) -> Result<Tensor> {
        let output_dtype = *self
            .output_dtypes
            .get(native_name)
            .expect("every GGUF binding has an output dtype");
        self.materialize_tensor_dtype(native_name, output_dtype, device)
    }

    fn materialize_tensor_dtype(
        &self,
        native_name: &str,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let binding = self.binding(native_name)?;
        let tensor = self
            .materialize_binding(binding, device)?
            .to_dtype(dtype)?
            .contiguous()?;
        if device.is_cuda() {
            binding.mark_uploaded(&self.archive);
        }
        Ok(tensor)
    }

    pub fn sharded_var_builder(self: &Arc<Self>, device: Device) -> ShardedVarBuilder {
        let backend = GgufTensorBackend::new(self.clone());
        ShardedSafeTensors::wrap(backend, self.dtype, device)
            .with_weight_source(self.clone())
            .with_raw_gguf(self.archive.clone())
    }

    fn binding(&self, native_name: &str) -> Result<&GgufTensorBinding> {
        self.bindings
            .get(native_name)
            .ok_or_else(|| Error::msg(format!("cannot find native GGUF binding `{native_name}`")))
    }

    fn layer_key<'a>(&self, key: &'a str) -> &'a str {
        key.strip_suffix(".weight").unwrap_or(key)
    }

    fn weight_name(&self, key: &str) -> String {
        format!("{}.weight", self.layer_key(key))
    }

    fn materialize_binding(&self, binding: &GgufTensorBinding, device: &Device) -> Result<Tensor> {
        match binding {
            GgufTensorBinding::Tensor(name) => {
                let tensor = self.archive.load_qtensor(name, device)?;
                tensor.dequantize(device)
            }
            GgufTensorBinding::Mxfp4Blocks(name) => {
                self.materialize_mxfp4_component(name, false, device)
            }
            GgufTensorBinding::Mxfp4Scales(name) => {
                self.materialize_mxfp4_component(name, true, device)
            }
            GgufTensorBinding::Slice {
                input,
                dim,
                start,
                len,
            } => self
                .materialize_binding(input, device)?
                .narrow(*dim, *start, *len)?
                .force_contiguous(),
            GgufTensorBinding::Concat { inputs, dim } => {
                let tensors = inputs
                    .iter()
                    .map(|input| self.materialize_binding(input, device))
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&tensors, *dim)
            }
            GgufTensorBinding::Stack { inputs, dim } => {
                let tensors = inputs
                    .iter()
                    .map(|input| self.materialize_binding(input, device))
                    .collect::<Result<Vec<_>>>()?;
                Tensor::stack(&tensors, *dim)
            }
            GgufTensorBinding::Interleave { inputs, dim } => {
                let tensors = inputs
                    .iter()
                    .map(|input| self.materialize_binding(input, device))
                    .collect::<Result<Vec<_>>>()?;
                let mut shape = tensors
                    .first()
                    .expect("interleave bindings are non-empty")
                    .dims()
                    .to_vec();
                shape[*dim] *= tensors.len();
                Tensor::stack(&tensors, dim + 1)?.reshape(Shape::from(shape))
            }
            GgufTensorBinding::Transpose { input, dim1, dim2 } => self
                .materialize_binding(input, device)?
                .transpose(*dim1, *dim2),
            GgufTensorBinding::Permute { input, dims } => {
                self.materialize_binding(input, device)?.permute(&dims[..])
            }
            GgufTensorBinding::Reshape { input, dims } => self
                .materialize_binding(input, device)?
                .reshape(Shape::from(dims.clone())),
            GgufTensorBinding::Affine { input, mul, add } => {
                self.materialize_binding(input, device)?.affine(*mul, *add)
            }
            GgufTensorBinding::Log { input } => self.materialize_binding(input, device)?.log(),
            GgufTensorBinding::InverseSoftplus { input } => self
                .materialize_binding(input, device)?
                .exp()?
                .affine(1.0, -1.0)?
                .log(),
            GgufTensorBinding::Cast { input, dtype } => {
                self.materialize_binding(input, device)?.to_dtype(*dtype)
            }
        }
    }

    fn materialize_mxfp4_component(
        &self,
        name: &str,
        scales: bool,
        device: &Device,
    ) -> Result<Tensor> {
        let info = self.archive.tensor_info(name)?;
        validate_mxfp4(info.dtype().raw(), info.shape(), name)?;
        let data = self.archive.tensor_data(name)?;
        let block_count = info.elem_count()? / 32;
        let mut values = Vec::with_capacity(block_count * if scales { 1 } else { 16 });
        for block in data.bytes().as_chunks::<17>().0 {
            if scales {
                values.push(block[0]);
            } else {
                let packed = &block[1..];
                for index in (0..32).step_by(2) {
                    let low = if index < 16 {
                        packed[index] & 0x0f
                    } else {
                        packed[index - 16] >> 4
                    };
                    let high_index = index + 1;
                    let high = if high_index < 16 {
                        packed[high_index] & 0x0f
                    } else {
                        packed[high_index - 16] >> 4
                    };
                    values.push(low | (high << 4));
                }
            }
        }
        let mut shape = info.shape().to_vec();
        let last = shape.last_mut().expect("MXFP4 tensors are non-empty");
        *last /= if scales { 32 } else { 2 };
        Tensor::from_vec(values, Shape::from(shape), device)
    }

    fn load_bias(
        &self,
        key: &str,
        device: &Device,
        range: Option<(usize, usize, usize)>,
        weight_rank: usize,
    ) -> Result<Option<Tensor>> {
        let name = format!("{key}.bias");
        if !self.bindings.contains_key(&name) {
            return Ok(None);
        }
        match bias_shard(range, weight_rank) {
            BiasShard::Skip => Ok(None),
            BiasShard::Full => self.materialize_tensor(&name, device).map(Some),
            BiasShard::Narrow { dim, start, len } => self
                .materialize_tensor(&name, &Device::Cpu)?
                .narrow(dim, start, len)?
                .contiguous()?
                .to_device(device)
                .map(Some),
        }
    }

    fn load_direct_linear(
        &self,
        key: &str,
        source_name: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let info = self.archive.tensor_info(source_name)?;
        let mut dims = info.shape().to_vec();
        if dims.len() < 2 {
            candle_core::bail!(
                "native linear `{key}` resolves to rank-{} GGUF tensor `{source_name}`",
                dims.len()
            );
        }
        let range = shard_range(shard, &dims)?;
        let weight = if let Some((dim, start, len)) = range {
            let shard_info = &self.archive.shards()[info.shard_index()];
            if shard_info.endian() != GgufEndian::Little {
                candle_core::bail!("big-endian GGUF tensor loading is not supported");
            }
            let dtype = info.dtype().candle_dtype()?;
            let data = self.archive.tensor_data(source_name)?;
            let bytes = slice_blocked_data(
                data.bytes(),
                &dims,
                dtype.block_size(),
                dtype.type_size(),
                dim,
                start,
                len,
            )?;
            dims[dim] = len;
            qtensor_from_gguf_data(dtype, &bytes, dims.clone(), device)?
        } else {
            self.archive.load_qtensor(source_name, device)?
        };
        let bias = self.load_bias(key, device, range, dims.len())?;
        Ok(Arc::new(GgufMatMul::from_qtensor(weight, bias)))
    }

    fn load_structural_linear(
        &self,
        key: &str,
        binding: &GgufTensorBinding,
        device: &Device,
        shard: Shard,
    ) -> Result<Option<Arc<dyn QuantMethod>>> {
        let Some(mut packed) = self.materialize_packed_binding(binding)? else {
            return Ok(None);
        };
        if packed.dims.len() < 2 {
            candle_core::bail!(
                "native linear `{key}` resolves to rank-{} transformed GGUF tensor",
                packed.dims.len()
            );
        }
        let range = shard_range(shard, &packed.dims)?;
        if let Some((dim, start, len)) = range {
            packed.data = slice_blocked_data(
                &packed.data,
                &packed.dims,
                packed.dtype.block_size(),
                packed.dtype.type_size(),
                dim,
                start,
                len,
            )?;
            packed.dims[dim] = len;
        }
        let rank = packed.dims.len();
        let weight = qtensor_from_gguf_data(packed.dtype, &packed.data, packed.dims, device)?;
        let bias = self.load_bias(key, device, range, rank)?;
        Ok(Some(Arc::new(GgufMatMul::from_qtensor(weight, bias))))
    }

    fn materialize_packed_binding(
        &self,
        binding: &GgufTensorBinding,
    ) -> Result<Option<PackedBinding>> {
        match binding {
            GgufTensorBinding::Tensor(name) => {
                let info = self.archive.tensor_info(name)?;
                let dtype = info.dtype().candle_dtype()?;
                if matches!(info.dtype().raw(), 0 | 1 | 30) {
                    return Ok(None);
                }
                Ok(Some(PackedBinding {
                    dtype,
                    dims: info.shape().to_vec(),
                    data: self.archive.tensor_data(name)?.bytes().to_vec(),
                }))
            }
            GgufTensorBinding::Slice {
                input,
                dim,
                start,
                len,
            } => {
                let Some(mut packed) = self.materialize_packed_binding(input)? else {
                    return Ok(None);
                };
                packed.data = slice_blocked_data(
                    &packed.data,
                    &packed.dims,
                    packed.dtype.block_size(),
                    packed.dtype.type_size(),
                    *dim,
                    *start,
                    *len,
                )?;
                packed.dims[*dim] = *len;
                Ok(Some(packed))
            }
            GgufTensorBinding::Concat { inputs, dim } => {
                let mut packed = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let Some(input) = self.materialize_packed_binding(input)? else {
                        return Ok(None);
                    };
                    packed.push(input);
                }
                concat_packed_bindings(packed, *dim).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn binding_is_dense(&self, binding: &GgufTensorBinding) -> Result<bool> {
        match binding {
            GgufTensorBinding::Tensor(name) => Ok(matches!(
                self.archive.tensor_info(name)?.dtype().raw(),
                0 | 1 | 30
            )),
            GgufTensorBinding::Slice { input, .. }
            | GgufTensorBinding::Transpose { input, .. }
            | GgufTensorBinding::Permute { input, .. }
            | GgufTensorBinding::Reshape { input, .. }
            | GgufTensorBinding::Affine { input, .. }
            | GgufTensorBinding::Log { input }
            | GgufTensorBinding::InverseSoftplus { input }
            | GgufTensorBinding::Cast { input, .. } => self.binding_is_dense(input),
            GgufTensorBinding::Concat { inputs, .. }
            | GgufTensorBinding::Stack { inputs, .. }
            | GgufTensorBinding::Interleave { inputs, .. } => {
                inputs.iter().try_fold(true, |dense, input| {
                    Ok(dense && self.binding_is_dense(input)?)
                })
            }
            GgufTensorBinding::Mxfp4Blocks(_) | GgufTensorBinding::Mxfp4Scales(_) => Ok(false),
        }
    }

    fn structural_quant_dtype(&self, binding: &GgufTensorBinding) -> Result<Option<GgmlDType>> {
        match binding {
            GgufTensorBinding::Tensor(name) => {
                let info = self.archive.tensor_info(name)?;
                if matches!(info.dtype().raw(), 0 | 1 | 30) {
                    Ok(None)
                } else {
                    info.dtype().candle_dtype().map(Some)
                }
            }
            GgufTensorBinding::Slice { input, .. } => self.structural_quant_dtype(input),
            GgufTensorBinding::Concat { inputs, .. } => {
                let mut dtype = None;
                for input in inputs {
                    let Some(input_dtype) = self.structural_quant_dtype(input)? else {
                        return Ok(None);
                    };
                    if dtype.is_some_and(|dtype| dtype != input_dtype) {
                        return Ok(None);
                    }
                    dtype = Some(input_dtype);
                }
                Ok(dtype)
            }
            _ => Ok(None),
        }
    }

    fn load_dense_linear(
        &self,
        key: &str,
        binding: &GgufTensorBinding,
        device: &Device,
        shard: Shard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let mut weight = self.materialize_binding(binding, &Device::Cpu)?;
        if weight.rank() < 2 {
            candle_core::bail!(
                "native linear `{key}` resolves to rank-{} transformed GGUF tensor",
                weight.rank()
            );
        }
        let range = shard_range(shard, weight.dims())?;
        if let Some((dim, start, len)) = range {
            weight = weight.narrow(dim, start, len)?;
        }
        weight = weight
            .to_dtype(
                *self
                    .output_dtypes
                    .get(&self.weight_name(key))
                    .expect("every GGUF binding has an output dtype"),
            )?
            .contiguous()?
            .to_device(device)?;
        let bias = self.load_bias(key, device, range, weight.rank())?;
        Ok(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(weight, bias)),
        )?))
    }

    fn binding_storage(
        &self,
        native_name: &str,
        binding: &GgufTensorBinding,
        dtype: DType,
    ) -> Result<(usize, usize)> {
        let shape = self
            .shapes
            .get(native_name)
            .ok_or_else(|| Error::msg(format!("GGUF binding `{native_name}` has no shape")))?;
        let logical_elements = checked_elem_count(shape)?;
        let resident_bytes = match binding.direct_tensor() {
            Some(source_name) => {
                let info = self.archive.tensor_info(source_name)?;
                if matches!(info.dtype().raw(), 0 | 1 | 30) {
                    logical_elements
                        .checked_mul(dtype.size_in_bytes())
                        .ok_or_else(|| Error::msg("GGUF dense resident byte estimate overflow"))?
                } else {
                    info.byte_len().ok_or_else(|| {
                        Error::msg(format!("GGUF tensor `{source_name}` has no byte range"))
                    })?
                }
            }
            None => match self.structural_quant_dtype(binding)? {
                Some(ggml_dtype) => packed_byte_len(shape, ggml_dtype)?,
                None => logical_elements
                    .checked_mul(
                        self.output_dtypes
                            .get(native_name)
                            .expect("every GGUF binding has an output dtype")
                            .size_in_bytes(),
                    )
                    .ok_or_else(|| Error::msg("GGUF resident byte estimate overflow"))?,
            },
        };
        Ok((logical_elements, resident_bytes))
    }

    fn binding_pack_factor(
        &self,
        native_name: &str,
        binding: &GgufTensorBinding,
        dtype: DType,
    ) -> Result<usize> {
        let (logical_elements, resident_bytes) =
            self.binding_storage(native_name, binding, dtype)?;
        Ok(block_pack_factor(logical_elements, dtype, resident_bytes))
    }
}

impl QuantizedWeightSource for GgufWeightSource {
    fn release_source_cache(&self) -> Result<()> {
        self.archive.release_source_cache()
    }

    fn contains(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    fn load_linear(
        &self,
        key: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Option<Arc<dyn QuantMethod>>> {
        let key = self.layer_key(key);
        let weight_name = self.weight_name(key);
        let Some(binding) = self.bindings.get(&weight_name) else {
            return Ok(None);
        };
        let layer = match binding.direct_tensor() {
            Some(source_name)
                if matches!(
                    self.archive.tensor_info(source_name)?.dtype().raw(),
                    0 | 1 | 30
                ) =>
            {
                self.load_dense_linear(key, binding, device, shard)
                    .map(Some)
            }
            Some(source_name) => self
                .load_direct_linear(key, source_name, device, shard)
                .map(Some),
            None => match self.load_structural_linear(key, binding, device, shard)? {
                Some(layer) => Ok(Some(layer)),
                None => self
                    .load_dense_linear(key, binding, device, shard)
                    .map(Some),
            },
        }?;
        if device.is_cuda() && layer.is_some() {
            binding.mark_uploaded(&self.archive);
        }
        Ok(layer)
    }

    fn supports_dense_packing(&self, key: &str) -> Result<bool> {
        let key = self.layer_key(key);
        let Some(binding) = self.bindings.get(&self.weight_name(key)) else {
            return Ok(false);
        };
        if self.bindings.contains_key(&format!("{key}.bias")) {
            return Ok(false);
        }
        self.binding_is_dense(binding)
    }

    fn load_optional_tensor(&self, name: &str, device: &Device) -> Result<Option<Tensor>> {
        if !self.bindings.contains_key(name) {
            return Ok(None);
        }
        self.materialize_tensor(name, device).map(Some)
    }

    fn shard_alignment(&self, key: &str) -> Result<usize> {
        let weight_name = self.weight_name(key);
        let binding = self.binding(&weight_name)?;
        if let Some(dtype) = self.structural_quant_dtype(binding)? {
            Ok(dtype.block_size())
        } else if let Some(source_name) = binding.direct_tensor() {
            self.archive
                .tensor_info(source_name)?
                .dtype()
                .block_size()
                .ok_or_else(|| {
                    Error::msg(format!(
                        "GGUF tensor `{source_name}` has unknown shard alignment"
                    ))
                })
        } else {
            Ok(1)
        }
    }

    fn pack_factor(&self, dtype: DType) -> Result<usize> {
        let mut layers = HashMap::<usize, Vec<(usize, usize)>>::new();
        for (native_name, binding) in &self.bindings {
            if !native_name.ends_with(".weight")
                || self
                    .shapes
                    .get(native_name)
                    .is_none_or(|shape| shape.len() < 2)
            {
                continue;
            }
            let Some(layer) = binding.text_layer_index() else {
                continue;
            };
            layers.entry(layer).or_default().push(self.binding_storage(
                native_name,
                binding,
                dtype,
            )?);
        }
        let dtype_bytes = dtype.size_in_bytes();
        let mut global_factor: Option<usize> = None;
        for bindings in layers.into_values() {
            let logical_elements = bindings.iter().try_fold(0usize, |total, (logical, _)| {
                total
                    .checked_add(*logical)
                    .ok_or_else(|| Error::msg("GGUF logical element estimate overflow"))
            })?;
            let resident_bytes = bindings.iter().try_fold(0usize, |total, (_, resident)| {
                total
                    .checked_add(*resident)
                    .ok_or_else(|| Error::msg("GGUF resident byte estimate overflow"))
            })?;
            let mut factor = block_pack_factor(logical_elements, dtype, resident_bytes);
            while factor > 1 {
                let estimated_bytes = bindings.iter().try_fold(0usize, |total, (logical, _)| {
                    let bytes = (logical / factor)
                        .checked_mul(dtype_bytes)
                        .ok_or_else(|| Error::msg("GGUF packed byte estimate overflow"))?;
                    total
                        .checked_add(bytes)
                        .ok_or_else(|| Error::msg("GGUF packed byte estimate overflow"))
                })?;
                if estimated_bytes >= resident_bytes {
                    break;
                }
                factor -= 1;
            }
            global_factor = Some(global_factor.map_or(factor, |global| global.min(factor)));
        }
        Ok(global_factor.unwrap_or(1))
    }

    fn resident_layer_bytes(&self, dtype: DType) -> Result<Option<crate::ResidentLayerBytes>> {
        let mut sizes = crate::ResidentLayerBytes::default();
        for (native_name, binding) in &self.bindings {
            let (_, bytes) = self.binding_storage(native_name, binding, dtype)?;
            match binding.text_layer_index() {
                Some(layer) => *sizes.layers.entry(layer).or_default() += bytes,
                None => sizes.outside += bytes,
            }
        }
        Ok(Some(sizes))
    }

    fn pack_factor_for(&self, key: &str, dtype: DType) -> Result<Option<usize>> {
        let weight_name = self.weight_name(key);
        let Some(binding) = self.bindings.get(&weight_name) else {
            return Ok(None);
        };
        self.binding_pack_factor(&weight_name, binding, dtype)
            .map(Some)
    }
}

fn concat_packed_bindings(inputs: Vec<PackedBinding>, dim: usize) -> Result<PackedBinding> {
    let Some(first) = inputs.first() else {
        candle_core::bail!("cannot concatenate zero packed GGUF bindings");
    };
    let dtype = first.dtype;
    let mut dims = first.dims.clone();
    if dim >= dims.len() {
        candle_core::bail!(
            "cannot concatenate dimension {dim} of rank-{} packed GGUF bindings",
            dims.len()
        );
    }
    for input in &inputs[1..] {
        if input.dtype != dtype
            || input.dims.len() != dims.len()
            || input
                .dims
                .iter()
                .enumerate()
                .any(|(index, size)| index != dim && *size != dims[index])
        {
            candle_core::bail!("cannot concatenate incompatible packed GGUF bindings");
        }
    }

    let block = dtype.block_size();
    let block_bytes = dtype.type_size();
    if inputs
        .iter()
        .any(|input| !input.dims[input.dims.len() - 1].is_multiple_of(block))
    {
        candle_core::bail!("packed GGUF concat inputs must have a block-aligned last dimension");
    }
    let mut data = Vec::with_capacity(
        inputs
            .iter()
            .try_fold(0usize, |size, input| size.checked_add(input.data.len()))
            .ok_or_else(|| Error::msg("packed GGUF concat byte length overflow"))?,
    );
    if dim == dims.len() - 1 {
        if inputs
            .iter()
            .any(|input| !input.dims[dim].is_multiple_of(block))
        {
            candle_core::bail!(
                "concatenating the packed GGUF dimension requires block-aligned inputs"
            );
        }
        let rows = dims[..dim].iter().try_fold(1usize, |count, size| {
            count
                .checked_mul(*size)
                .ok_or_else(|| Error::msg("packed GGUF concat row count overflow"))
        })?;
        for row in 0..rows {
            for input in &inputs {
                let row_bytes = input.dims[dim] / block * block_bytes;
                let start = row * row_bytes;
                data.extend_from_slice(&input.data[start..start + row_bytes]);
            }
        }
    } else {
        let row_bytes = dims[dims.len() - 1] / block * block_bytes;
        let inner = dims[dim + 1..dims.len() - 1]
            .iter()
            .try_fold(1usize, |count, size| {
                count
                    .checked_mul(*size)
                    .ok_or_else(|| Error::msg("packed GGUF concat inner size overflow"))
            })?;
        let chunk_bytes = inner
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::msg("packed GGUF concat chunk size overflow"))?;
        let outer = dims[..dim].iter().try_fold(1usize, |count, size| {
            count
                .checked_mul(*size)
                .ok_or_else(|| Error::msg("packed GGUF concat outer size overflow"))
        })?;
        for outer_index in 0..outer {
            for input in &inputs {
                let len = input.dims[dim] * chunk_bytes;
                let start = outer_index * len;
                data.extend_from_slice(&input.data[start..start + len]);
            }
        }
    }
    dims[dim] = inputs.iter().map(|input| input.dims[dim]).sum();
    Ok(PackedBinding { dtype, dims, data })
}

fn packed_byte_len(dims: &[usize], dtype: GgmlDType) -> Result<usize> {
    let Some(last) = dims.last() else {
        candle_core::bail!("packed GGUF tensors cannot be scalar");
    };
    if !last.is_multiple_of(dtype.block_size()) {
        candle_core::bail!(
            "packed GGUF tensor last dimension {last} is not divisible by {:?} block size {}",
            dtype,
            dtype.block_size()
        );
    }
    let elements = checked_elem_count(dims)?;
    if !elements.is_multiple_of(dtype.block_size()) {
        candle_core::bail!(
            "packed GGUF tensor has {elements} elements, not divisible by {:?} block size {}",
            dtype,
            dtype.block_size()
        );
    }
    (elements / dtype.block_size())
        .checked_mul(dtype.type_size())
        .ok_or_else(|| Error::msg("packed GGUF tensor byte length overflow"))
}

#[derive(Clone, Debug)]
pub struct GgufTensorBackend {
    source: Arc<GgufWeightSource>,
}

impl GgufTensorBackend {
    pub fn new(source: Arc<GgufWeightSource>) -> Self {
        Self { source }
    }

    pub fn source(&self) -> &Arc<GgufWeightSource> {
        &self.source
    }
}

impl TensorShapes for GgufTensorBackend {
    fn tensor_shapes(&self) -> HashMap<String, Vec<usize>> {
        self.source.shapes.clone()
    }
}

impl SimpleBackend for GgufTensorBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let expected = self
            .source
            .shapes
            .get(name)
            .ok_or_else(|| Error::CannotFindTensor {
                path: name.to_string(),
            })?;
        if expected.as_slice() != shape.dims() {
            return Err(Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: shape,
                got: Shape::from(expected.clone()),
            }
            .bt());
        }
        let tensor = self.get_unchecked(name, dtype, device)?;
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        self.source.materialize_tensor_dtype(name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.source.contains(name)
    }
}

fn validate_binding_storage(
    archive: &GgufArchive,
    binding: &GgufTensorBinding,
    native_name: &str,
) -> Result<()> {
    match binding {
        GgufTensorBinding::Tensor(name) => {
            let dtype = archive.tensor_info(name)?.dtype();
            if dtype.candle_dtype().is_err() {
                candle_core::bail!(
                    "GGUF tensor `{name}` uses dtype {} ({}) for native binding `{native_name}`; \
                     direct GGUF loading currently supports {DIRECT_GGUF_DTYPES}, while \
                     MXFP4 is supported only through its explicit GPT-OSS binding",
                    dtype.name(),
                    dtype.raw()
                );
            }
        }
        GgufTensorBinding::Mxfp4Blocks(name) | GgufTensorBinding::Mxfp4Scales(name) => {
            let info = archive.tensor_info(name)?;
            validate_mxfp4(info.dtype().raw(), info.shape(), name)?;
        }
        GgufTensorBinding::Slice { input, .. }
        | GgufTensorBinding::Transpose { input, .. }
        | GgufTensorBinding::Permute { input, .. }
        | GgufTensorBinding::Reshape { input, .. }
        | GgufTensorBinding::Affine { input, .. }
        | GgufTensorBinding::Log { input }
        | GgufTensorBinding::InverseSoftplus { input }
        | GgufTensorBinding::Cast { input, .. } => {
            validate_binding_storage(archive, input, native_name)?;
        }
        GgufTensorBinding::Concat { inputs, .. }
        | GgufTensorBinding::Stack { inputs, .. }
        | GgufTensorBinding::Interleave { inputs, .. } => {
            for input in inputs {
                validate_binding_storage(archive, input, native_name)?;
            }
        }
    }
    Ok(())
}

fn infer_binding_shape(archive: &GgufArchive, binding: &GgufTensorBinding) -> Result<Vec<usize>> {
    match binding {
        GgufTensorBinding::Tensor(name) => Ok(archive.tensor_info(name)?.shape().to_vec()),
        GgufTensorBinding::Mxfp4Blocks(name) => infer_mxfp4_component_shape(archive, name, false),
        GgufTensorBinding::Mxfp4Scales(name) => infer_mxfp4_component_shape(archive, name, true),
        GgufTensorBinding::Slice {
            input,
            dim,
            start,
            len,
        } => {
            let mut shape = infer_binding_shape(archive, input)?;
            let rank = shape.len();
            let size = shape.get_mut(*dim).ok_or_else(|| {
                Error::msg(format!(
                    "cannot slice dimension {dim} of rank-{rank} GGUF binding"
                ))
            })?;
            let end = start
                .checked_add(*len)
                .ok_or_else(|| Error::msg("GGUF binding slice overflow"))?;
            if end > *size {
                candle_core::bail!(
                    "GGUF binding slice {start}..{end} exceeds dimension {dim} of size {size}"
                );
            }
            *size = *len;
            Ok(shape)
        }
        GgufTensorBinding::Concat { inputs, dim } => {
            let mut shapes = inputs
                .iter()
                .map(|input| infer_binding_shape(archive, input))
                .collect::<Result<Vec<_>>>()?;
            let Some(mut shape) = shapes.pop() else {
                candle_core::bail!("cannot concatenate zero GGUF bindings");
            };
            if *dim >= shape.len() {
                candle_core::bail!(
                    "cannot concatenate dimension {dim} of rank-{} GGUF bindings",
                    shape.len()
                );
            }
            for other in shapes {
                if other.len() != shape.len()
                    || other
                        .iter()
                        .enumerate()
                        .any(|(index, size)| index != *dim && *size != shape[index])
                {
                    candle_core::bail!(
                        "cannot concatenate GGUF binding shapes {shape:?} and {other:?} along dimension {dim}"
                    );
                }
                shape[*dim] = shape[*dim]
                    .checked_add(other[*dim])
                    .ok_or_else(|| Error::msg("GGUF concat shape overflow"))?;
            }
            Ok(shape)
        }
        GgufTensorBinding::Stack { inputs, dim } => {
            let shapes = inputs
                .iter()
                .map(|input| infer_binding_shape(archive, input))
                .collect::<Result<Vec<_>>>()?;
            let Some(first) = shapes.first() else {
                candle_core::bail!("cannot stack zero GGUF bindings");
            };
            if shapes.iter().any(|shape| shape != first) {
                candle_core::bail!("cannot stack unequal GGUF binding shapes {shapes:?}");
            }
            if *dim > first.len() {
                candle_core::bail!(
                    "cannot stack at dimension {dim} for rank-{} GGUF bindings",
                    first.len()
                );
            }
            let mut shape = first.clone();
            shape.insert(*dim, shapes.len());
            Ok(shape)
        }
        GgufTensorBinding::Interleave { inputs, dim } => {
            let shapes = inputs
                .iter()
                .map(|input| infer_binding_shape(archive, input))
                .collect::<Result<Vec<_>>>()?;
            let Some(first) = shapes.first() else {
                candle_core::bail!("cannot interleave zero GGUF bindings");
            };
            if *dim >= first.len() {
                candle_core::bail!(
                    "cannot interleave dimension {dim} of rank-{} GGUF bindings",
                    first.len()
                );
            }
            if shapes.iter().any(|shape| shape != first) {
                candle_core::bail!("cannot interleave unequal GGUF binding shapes {shapes:?}");
            }
            let mut shape = first.clone();
            shape[*dim] = shape[*dim]
                .checked_mul(shapes.len())
                .ok_or_else(|| Error::msg("GGUF interleave shape overflow"))?;
            Ok(shape)
        }
        GgufTensorBinding::Transpose { input, dim1, dim2 } => {
            let mut shape = infer_binding_shape(archive, input)?;
            if *dim1 >= shape.len() || *dim2 >= shape.len() {
                candle_core::bail!(
                    "cannot transpose dimensions {dim1} and {dim2} of rank-{} GGUF binding",
                    shape.len()
                );
            }
            shape.swap(*dim1, *dim2);
            Ok(shape)
        }
        GgufTensorBinding::Permute { input, dims } => {
            let shape = infer_binding_shape(archive, input)?;
            if dims.len() != shape.len() {
                candle_core::bail!(
                    "GGUF permutation has {} dimensions for rank-{} binding",
                    dims.len(),
                    shape.len()
                );
            }
            let mut seen = vec![false; shape.len()];
            for &dim in dims {
                if dim >= shape.len() || std::mem::replace(&mut seen[dim], true) {
                    candle_core::bail!("invalid GGUF binding permutation {dims:?}");
                }
            }
            Ok(dims.iter().map(|dim| shape[*dim]).collect())
        }
        GgufTensorBinding::Reshape { input, dims } => {
            let shape = infer_binding_shape(archive, input)?;
            if checked_elem_count(&shape)? != checked_elem_count(dims)? {
                candle_core::bail!("cannot reshape GGUF binding from {shape:?} to {dims:?}");
            }
            Ok(dims.clone())
        }
        GgufTensorBinding::Affine { input, .. } => infer_binding_shape(archive, input),
        GgufTensorBinding::Log { input } => infer_binding_shape(archive, input),
        GgufTensorBinding::InverseSoftplus { input } => infer_binding_shape(archive, input),
        GgufTensorBinding::Cast { input, .. } => infer_binding_shape(archive, input),
    }
}

fn infer_mxfp4_component_shape(
    archive: &GgufArchive,
    name: &str,
    scales: bool,
) -> Result<Vec<usize>> {
    let info = archive.tensor_info(name)?;
    validate_mxfp4(info.dtype().raw(), info.shape(), name)?;
    let mut shape = info.shape().to_vec();
    let last = shape.last_mut().expect("MXFP4 tensors are non-empty");
    *last /= if scales { 32 } else { 2 };
    Ok(shape)
}

fn validate_mxfp4(dtype: u32, shape: &[usize], name: &str) -> Result<()> {
    if dtype != 39 {
        candle_core::bail!("GGUF tensor `{name}` has dtype {dtype}, expected MXFP4 dtype 39");
    }
    let Some(last) = shape.last() else {
        candle_core::bail!("GGUF MXFP4 tensor `{name}` has no dimensions");
    };
    if !last.is_multiple_of(32) {
        candle_core::bail!(
            "GGUF MXFP4 tensor `{name}` last dimension {last} is not divisible by 32"
        );
    }
    Ok(())
}

fn infer_binding_dtype(
    archive: &GgufArchive,
    binding: &GgufTensorBinding,
    source_dtype: DType,
) -> Result<DType> {
    match binding {
        GgufTensorBinding::Tensor(_) => Ok(source_dtype),
        GgufTensorBinding::Mxfp4Blocks(name) | GgufTensorBinding::Mxfp4Scales(name) => {
            let info = archive.tensor_info(name)?;
            validate_mxfp4(info.dtype().raw(), info.shape(), name)?;
            Ok(DType::U8)
        }
        GgufTensorBinding::Slice { input, .. }
        | GgufTensorBinding::Transpose { input, .. }
        | GgufTensorBinding::Permute { input, .. }
        | GgufTensorBinding::Reshape { input, .. } => {
            infer_binding_dtype(archive, input, source_dtype)
        }
        GgufTensorBinding::Concat { inputs, .. }
        | GgufTensorBinding::Stack { inputs, .. }
        | GgufTensorBinding::Interleave { inputs, .. } => {
            let mut dtypes = inputs
                .iter()
                .map(|input| infer_binding_dtype(archive, input, source_dtype));
            let Some(first) = dtypes.next() else {
                candle_core::bail!("cannot combine zero GGUF bindings");
            };
            let first = first?;
            for dtype in dtypes {
                let dtype = dtype?;
                if dtype != first {
                    candle_core::bail!(
                        "cannot combine GGUF bindings with dtypes {first:?} and {dtype:?}"
                    );
                }
            }
            Ok(first)
        }
        GgufTensorBinding::Affine { input, .. }
        | GgufTensorBinding::Log { input }
        | GgufTensorBinding::InverseSoftplus { input } => {
            let dtype = infer_binding_dtype(archive, input, source_dtype)?;
            if dtype == DType::U8 {
                candle_core::bail!("cannot apply a floating-point transform to a U8 GGUF binding");
            }
            Ok(dtype)
        }
        GgufTensorBinding::Cast { input, dtype } => {
            if *dtype == DType::U8 {
                candle_core::bail!("cannot cast a GGUF binding to U8");
            }
            let input_dtype = infer_binding_dtype(archive, input, source_dtype)?;
            if input_dtype == DType::U8 {
                candle_core::bail!("cannot cast a U8 GGUF binding");
            }
            Ok(*dtype)
        }
    }
}

fn checked_elem_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, dim| {
        count
            .checked_mul(*dim)
            .ok_or_else(|| Error::msg("GGUF binding element count overflow"))
    })
}

#[cfg(test)]
#[path = "weight_source_tests.rs"]
mod tests;
