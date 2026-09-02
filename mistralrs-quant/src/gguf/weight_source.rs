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
        self.materialize_binding(binding, device)?
            .to_dtype(dtype)?
            .contiguous()
    }

    pub fn sharded_var_builder(self: &Arc<Self>, device: Device) -> ShardedVarBuilder {
        let backend = GgufTensorBackend::new(self.clone());
        ShardedSafeTensors::wrap(backend, self.dtype, device).with_weight_source(self.clone())
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
                .narrow(*dim, *start, *len),
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
        match binding.direct_tensor() {
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
        }
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
mod tests {
    use std::io::Write;

    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    use tempfile::NamedTempFile;

    use super::*;
    use crate::MXFP4Layer;

    const OUT_DIM: usize = 4;
    const IN_DIM: usize = 64;
    const EXPERTS: usize = 3;
    const EXPERT_ROWS: usize = 2;
    type Mxfp4Fixture = (NamedTempFile, Arc<GgufWeightSource>, Vec<u8>, Vec<u8>);

    fn patterned(rows: usize, cols: usize, offset: usize) -> Result<Tensor> {
        let values = (0..rows * cols)
            .map(|index| ((index + offset) % 29) as f32 / 7.0 - 2.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows, cols), &Device::Cpu)
    }

    fn patterned_experts(offset: usize) -> Result<Tensor> {
        let values = (0..EXPERTS * EXPERT_ROWS * IN_DIM)
            .map(|index| ((index + offset) % 31) as f32 / 9.0 - 1.5)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (EXPERTS, EXPERT_ROWS, IN_DIM), &Device::Cpu)
    }

    fn align(value: usize, alignment: usize) -> usize {
        value.div_ceil(alignment) * alignment
    }

    fn push_tensor_header(bytes: &mut Vec<u8>, name: &str, offset: u64) {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        for dim in [64u64, 2, 2] {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&39u32.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    fn mxfp4_data(seed: u8) -> Vec<u8> {
        let mut data = Vec::new();
        for block in 0..8 {
            data.push(124 + ((block as u8 + seed) % 4));
            for index in 0..16 {
                let low = seed.wrapping_add((block * 3 + index) as u8) & 0x0f;
                let high = seed.wrapping_add((block * 3 + index + 7) as u8) & 0x0f;
                data.push(low | (high << 4));
            }
        }
        data
    }

    fn unsupported_dtype_archive() -> Result<(NamedTempFile, Arc<GgufArchive>)> {
        let name = "unsupported.weight";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&256u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(align(bytes.len(), 32), 0);
        bytes.extend_from_slice(&[0; 66]);
        bytes.resize(align(bytes.len(), 32), 0);

        let mut file = NamedTempFile::new().map_err(Error::wrap)?;
        file.as_file_mut().write_all(&bytes).map_err(Error::wrap)?;
        file.as_file_mut().flush().map_err(Error::wrap)?;
        let archive = Arc::new(GgufArchive::open_file(file.path())?);
        Ok((file, archive))
    }

    fn big_endian_archive() -> Result<(NamedTempFile, Arc<GgufArchive>)> {
        let name = "big.weight";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"FUGG");
        bytes.extend_from_slice(&3u32.to_be_bytes());
        bytes.extend_from_slice(&1u64.to_be_bytes());
        bytes.extend_from_slice(&0u64.to_be_bytes());
        bytes.extend_from_slice(&(name.len() as u64).to_be_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&1u64.to_be_bytes());
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(&0u64.to_be_bytes());
        bytes.resize(align(bytes.len(), 32), 0);
        bytes.extend_from_slice(&0f32.to_be_bytes());
        bytes.resize(align(bytes.len(), 32), 0);

        let mut file = NamedTempFile::new().map_err(Error::wrap)?;
        file.as_file_mut().write_all(&bytes).map_err(Error::wrap)?;
        file.as_file_mut().flush().map_err(Error::wrap)?;
        let archive = Arc::new(GgufArchive::open_file(file.path())?);
        Ok((file, archive))
    }

    fn native_mxfp4_blocks(data: &[u8]) -> Vec<u8> {
        let mut blocks = Vec::new();
        for block in data.as_chunks::<17>().0 {
            for index in (0..32).step_by(2) {
                let nibble = |index: usize| {
                    if index < 16 {
                        block[index + 1] & 0x0f
                    } else {
                        block[index - 15] >> 4
                    }
                };
                blocks.push(nibble(index) | (nibble(index + 1) << 4));
            }
        }
        blocks
    }

    fn interleave_rows(lhs: &[u8], rhs: &[u8], outer: usize, rows: usize, width: usize) -> Vec<u8> {
        let mut values = Vec::with_capacity(lhs.len() + rhs.len());
        for outer_index in 0..outer {
            for row in 0..rows {
                let start = (outer_index * rows + row) * width;
                values.extend_from_slice(&lhs[start..start + width]);
                values.extend_from_slice(&rhs[start..start + width]);
            }
        }
        values
    }

    fn mxfp4_source() -> Result<Mxfp4Fixture> {
        let gate = mxfp4_data(0);
        let up = mxfp4_data(5);
        let second_offset = align(gate.len(), 32);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        push_tensor_header(&mut bytes, "blk.0.ffn_gate_exps.weight", 0);
        push_tensor_header(&mut bytes, "blk.0.ffn_up_exps.weight", second_offset as u64);
        bytes.resize(align(bytes.len(), 32), 0);
        bytes.extend_from_slice(&gate);
        bytes.resize(align(bytes.len(), 32), 0);
        bytes.extend_from_slice(&up);
        bytes.resize(align(bytes.len(), 32), 0);

        let mut file = NamedTempFile::new().map_err(Error::wrap)?;
        file.as_file_mut().write_all(&bytes).map_err(Error::wrap)?;
        file.as_file_mut().flush().map_err(Error::wrap)?;

        let archive = Arc::new(GgufArchive::open_file(file.path())?);
        let gate_blocks = GgufTensorBinding::mxfp4_blocks("blk.0.ffn_gate_exps.weight");
        let up_blocks = GgufTensorBinding::mxfp4_blocks("blk.0.ffn_up_exps.weight");
        let gate_scales = GgufTensorBinding::mxfp4_scales("blk.0.ffn_gate_exps.weight");
        let up_scales = GgufTensorBinding::mxfp4_scales("blk.0.ffn_up_exps.weight");
        let bindings = GgufBindingMap::new()
            .with_binding("gate.blocks", gate_blocks.clone())
            .with_binding("gate.scales", gate_scales.clone())
            .with_binding(
                "native.gate_up_proj_blocks",
                GgufTensorBinding::interleave(vec![gate_blocks, up_blocks], 1)
                    .reshape(vec![2, 4, 2, 16]),
            )
            .with_binding(
                "native.gate_up_proj_scales",
                GgufTensorBinding::interleave(vec![gate_scales, up_scales], 1),
            );
        let source = Arc::new(GgufWeightSource::new(archive, &bindings, DType::BF16)?);
        Ok((file, source, gate, up))
    }

    fn test_source() -> Result<(NamedTempFile, Arc<GgufWeightSource>, Tensor, Tensor)> {
        let weight = patterned(OUT_DIM, IN_DIM, 0)?;
        let other = patterned(OUT_DIM, IN_DIM, 11)?;
        let single_block = patterned(1, GgmlDType::Q4_0.block_size(), 7)?;
        let bias = Tensor::from_vec(vec![0.25f32, 0.5, 0.75, 1.0], OUT_DIM, &Device::Cpu)?;
        let norm = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], OUT_DIM, &Device::Cpu)?;

        let q_weight = QTensor::quantize(&weight, GgmlDType::Q4_0)?;
        let q_other = QTensor::quantize(&other, GgmlDType::Q4_0)?;
        let q_single_block = QTensor::quantize(&single_block, GgmlDType::Q4_0)?;
        let q_bias = QTensor::quantize(&bias, GgmlDType::F32)?;
        let q_norm = QTensor::quantize(&norm, GgmlDType::F32)?;
        let expected_weight = q_weight.dequantize(&Device::Cpu)?;
        let expected_other = q_other.dequantize(&Device::Cpu)?;
        let float_weight = QTensor::quantize(&expected_weight, GgmlDType::F32)?;

        let mut file = NamedTempFile::new().map_err(Error::wrap)?;
        gguf_file::write(
            file.as_file_mut(),
            &[],
            &[
                ("blk.0.weight", &q_weight),
                ("blk.1.weight", &q_other),
                ("blk.0.single.weight", &q_single_block),
                ("blk.float.weight", &float_weight),
                ("blk.0.bias", &q_bias),
                ("norm.weight", &q_norm),
            ],
        )?;
        file.as_file_mut().flush().map_err(Error::wrap)?;

        let archive = Arc::new(GgufArchive::open_file(file.path())?);
        let bindings = GgufBindingMap::new()
            .with_binding(
                "model.linear.weight",
                GgufTensorBinding::tensor("blk.0.weight"),
            )
            .with_binding("model.linear.bias", GgufTensorBinding::tensor("blk.0.bias"))
            .with_binding(
                "model.float.weight",
                GgufTensorBinding::tensor("blk.float.weight"),
            )
            .with_binding(
                "model.single.weight",
                GgufTensorBinding::tensor("blk.0.single.weight"),
            )
            .with_binding(
                "model.slice.weight",
                GgufTensorBinding::tensor("blk.0.weight").slice(0, 1, 2),
            )
            .with_binding(
                "model.concat.weight",
                GgufTensorBinding::concat(
                    vec![
                        GgufTensorBinding::tensor("blk.0.weight").slice(0, 0, 2),
                        GgufTensorBinding::tensor("blk.1.weight").slice(0, 2, 2),
                    ],
                    0,
                ),
            )
            .with_binding(
                "model.stack.weight",
                GgufTensorBinding::stack(
                    vec![
                        GgufTensorBinding::tensor("blk.0.weight"),
                        GgufTensorBinding::tensor("blk.1.weight"),
                    ],
                    0,
                ),
            )
            .with_binding(
                "model.norm.weight",
                GgufTensorBinding::tensor("norm.weight").affine(1.0, 1.0),
            )
            .with_binding(
                "model.log.weight",
                GgufTensorBinding::tensor("norm.weight").log(),
            )
            .with_binding(
                "model.inverse_softplus.weight",
                GgufTensorBinding::tensor("norm.weight").inverse_softplus(),
            )
            .with_binding(
                "model.precise.weight",
                GgufTensorBinding::tensor("norm.weight").affine(1.0, 0.001),
            )
            .with_binding(
                "model.cast.weight",
                GgufTensorBinding::tensor("blk.float.weight").cast(DType::F32),
            )
            .with_binding(
                "model.layers.0.linear.weight",
                GgufTensorBinding::tensor("blk.0.weight"),
            )
            .with_binding(
                "model.layers.0.single.weight",
                GgufTensorBinding::tensor("blk.0.single.weight"),
            )
            .with_binding(
                "model.layers.1.linear.weight",
                GgufTensorBinding::tensor("blk.0.weight").slice(0, 0, OUT_DIM),
            );
        let source = Arc::new(GgufWeightSource::new(archive, &bindings, DType::F32)?);
        Ok((file, source, expected_weight, expected_other))
    }

    fn expert_source() -> Result<(NamedTempFile, Arc<GgufWeightSource>, Tensor, Tensor)> {
        let gate = QTensor::quantize(&patterned_experts(0)?, GgmlDType::Q4_0)?;
        let up = QTensor::quantize(&patterned_experts(17)?, GgmlDType::Q4_0)?;
        let gate_bias = Tensor::from_vec(
            (0..EXPERTS * EXPERT_ROWS)
                .map(|index| index as f32 / 10.0)
                .collect::<Vec<_>>(),
            (EXPERTS, EXPERT_ROWS),
            &Device::Cpu,
        )?;
        let up_bias = Tensor::from_vec(
            (0..EXPERTS * EXPERT_ROWS)
                .map(|index| 1.0 + index as f32 / 10.0)
                .collect::<Vec<_>>(),
            (EXPERTS, EXPERT_ROWS),
            &Device::Cpu,
        )?;
        let q_gate_bias = QTensor::quantize(&gate_bias, GgmlDType::F32)?;
        let q_up_bias = QTensor::quantize(&up_bias, GgmlDType::F32)?;
        let expected = Tensor::cat(
            &[gate.dequantize(&Device::Cpu)?, up.dequantize(&Device::Cpu)?],
            1,
        )?;
        let expected_bias = Tensor::cat(&[gate_bias, up_bias], 1)?;

        let mut file = NamedTempFile::new().map_err(Error::wrap)?;
        gguf_file::write(
            file.as_file_mut(),
            &[],
            &[
                ("experts.gate.weight", &gate),
                ("experts.up.weight", &up),
                ("experts.gate.bias", &q_gate_bias),
                ("experts.up.bias", &q_up_bias),
            ],
        )?;
        file.as_file_mut().flush().map_err(Error::wrap)?;
        let archive = Arc::new(GgufArchive::open_file(file.path())?);
        let bindings = GgufBindingMap::new()
            .with_binding(
                "model.experts.gate_up.weight",
                GgufTensorBinding::concat(
                    vec![
                        GgufTensorBinding::tensor("experts.gate.weight"),
                        GgufTensorBinding::tensor("experts.up.weight"),
                    ],
                    1,
                ),
            )
            .with_binding(
                "model.experts.gate_up.bias",
                GgufTensorBinding::concat(
                    vec![
                        GgufTensorBinding::tensor("experts.gate.bias"),
                        GgufTensorBinding::tensor("experts.up.bias"),
                    ],
                    1,
                ),
            )
            .with_binding(
                "model.experts.gate.weight",
                GgufTensorBinding::tensor("experts.gate.weight").slice(1, 0, EXPERT_ROWS),
            );
        let source = Arc::new(GgufWeightSource::new(archive, &bindings, DType::F32)?);
        Ok((file, source, expected, expected_bias))
    }

    fn assert_close(lhs: &Tensor, rhs: &Tensor) -> Result<()> {
        assert_eq!(lhs.dims(), rhs.dims());
        let max_diff = (lhs - rhs)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max_diff <= 0.003, "max diff {max_diff}");
        Ok(())
    }

    #[test]
    fn direct_bindings_stay_packed_and_shard_before_loading() -> Result<()> {
        let (_file, source, expected, _) = test_source()?;

        let full = source
            .load_linear("model.linear", &Device::Cpu, Shard::default())?
            .unwrap();
        assert_eq!(
            full.get_qtensor().unwrap().shape().dims(),
            [OUT_DIM, IN_DIM]
        );
        assert!(full.has_bias());
        assert_close(&full.dequantize_w()?, &expected)?;

        let output_shard = source
            .load_linear(
                "model.linear",
                &Device::Cpu,
                Shard::Simple {
                    dim: 0,
                    rank: 1,
                    world_size: 2,
                },
            )?
            .unwrap();
        assert_eq!(
            output_shard.get_qtensor().unwrap().shape().dims(),
            [OUT_DIM / 2, IN_DIM]
        );
        assert!(output_shard.has_bias());
        assert_close(
            &output_shard.dequantize_w()?,
            &expected.narrow(0, OUT_DIM / 2, OUT_DIM / 2)?,
        )?;
        let zeros = Tensor::zeros((1, IN_DIM), DType::F32, &Device::Cpu)?;
        let output = output_shard.forward(&zeros)?;
        let expected_bias = Tensor::from_vec(vec![0.75f32, 1.0], (1, 2), &Device::Cpu)?;
        assert_close(&output, &expected_bias)?;

        let input_shard = source
            .load_linear(
                "model.linear",
                &Device::Cpu,
                Shard::Simple {
                    dim: 1,
                    rank: 0,
                    world_size: 2,
                },
            )?
            .unwrap();
        assert_eq!(
            input_shard.get_qtensor().unwrap().shape().dims(),
            [OUT_DIM, IN_DIM / 2]
        );
        assert!(!input_shard.has_bias());
        assert_eq!(source.shard_alignment("model.linear")?, 32);
        assert_eq!(
            source.pack_factor_for("model.linear.weight", DType::F32)?,
            Some(7)
        );
        assert_eq!(
            source.pack_factor_for("model.single.weight", DType::F32)?,
            Some(6)
        );

        let with_weight_suffix = source
            .load_linear("model.linear.weight", &Device::Cpu, Shard::default())?
            .unwrap();
        assert!(with_weight_suffix.has_bias());

        let float = source
            .load_linear("model.float", &Device::Cpu, Shard::default())?
            .unwrap();
        assert!(float.get_qtensor().is_none());
        assert_close(&float.dequantize_w()?, &expected)?;
        assert_eq!(source.pack_factor_for("model.float", DType::F32)?, Some(1));
        assert_eq!(source.pack_factor(DType::F32)?, 6);

        let uniform_bindings = GgufBindingMap::new().with_binding(
            "model.layers.0.linear.weight",
            GgufTensorBinding::tensor("blk.0.weight"),
        );
        let uniform =
            GgufWeightSource::new(source.archive().clone(), &uniform_bindings, DType::F32)?;
        assert_eq!(uniform.pack_factor(DType::F32)?, 7);

        let multimodal_bindings = GgufBindingMap::new()
            .with_binding(
                "model.language_model.layers.0.linear.weight",
                GgufTensorBinding::tensor("blk.0.weight"),
            )
            .with_binding(
                "model.vision_tower.encoder.layers.0.linear.weight",
                GgufTensorBinding::tensor("blk.float.weight"),
            );
        let multimodal =
            GgufWeightSource::new(source.archive().clone(), &multimodal_bindings, DType::F32)?;
        assert_eq!(multimodal.pack_factor(DType::F32)?, 7);
        Ok(())
    }

    #[test]
    fn structural_bindings_stay_packed() -> Result<()> {
        let (_file, source, expected, other) = test_source()?;

        let sliced = source
            .load_linear("model.slice", &Device::Cpu, Shard::default())?
            .unwrap();
        assert!(sliced.get_qtensor().is_some());
        assert_close(&sliced.dequantize_w()?, &expected.narrow(0, 1, 2)?)?;

        let concatenated = source
            .load_linear("model.concat", &Device::Cpu, Shard::default())?
            .unwrap();
        let expected_concat = Tensor::cat(&[expected.narrow(0, 0, 2)?, other.narrow(0, 2, 2)?], 0)?;
        assert!(concatenated.get_qtensor().is_some());
        assert_close(&concatenated.dequantize_w()?, &expected_concat)?;
        assert_eq!(source.pack_factor_for("model.concat", DType::F32)?, Some(7));

        let stacked = source
            .load_linear("model.stack", &Device::Cpu, Shard::default())?
            .unwrap();
        assert_eq!(stacked.dequantize_w()?.dims(), [2, OUT_DIM, IN_DIM]);
        assert!(stacked.get_qtensor().is_none());
        Ok(())
    }

    #[test]
    fn rank_three_concat_preserves_layout_and_selects_expert_bias() -> Result<()> {
        let (_file, source, expected, expected_bias) = expert_source()?;
        let layer = source
            .load_linear("model.experts.gate_up", &Device::Cpu, Shard::default())?
            .unwrap();
        assert_eq!(
            layer.get_qtensor().unwrap().shape().dims(),
            [EXPERTS, EXPERT_ROWS * 2, IN_DIM]
        );
        assert_close(&layer.dequantize_w()?, &expected)?;

        let indices = Tensor::from_vec(vec![0u32, 2, 1, 0], (2, 2), &Device::Cpu)?;
        let zeros = Tensor::zeros((2, 1, IN_DIM), DType::F32, &Device::Cpu)?;
        let actual = layer.gather_forward(&zeros, &indices)?;
        let selected_bias = expected_bias
            .index_select(&indices.flatten_all()?, 0)?
            .reshape((2, 2, EXPERT_ROWS * 2))?;
        assert_close(&actual, &selected_bias)?;

        let gate = source
            .load_linear("model.experts.gate", &Device::Cpu, Shard::default())?
            .unwrap();
        assert!(gate.get_qtensor().is_some());
        assert_close(&gate.dequantize_w()?, &expected.narrow(1, 0, EXPERT_ROWS)?)?;

        let output_shard = source
            .load_linear(
                "model.experts.gate_up",
                &Device::Cpu,
                Shard::Simple {
                    dim: 1,
                    rank: 1,
                    world_size: 2,
                },
            )?
            .unwrap();
        let actual = output_shard.gather_forward(&zeros, &indices)?;
        let expected_bias = expected_bias
            .narrow(1, EXPERT_ROWS, EXPERT_ROWS)?
            .contiguous()?
            .index_select(&indices.flatten_all()?, 0)?
            .reshape((2, 2, EXPERT_ROWS))?;
        assert_close(&actual, &expected_bias)
    }

    #[test]
    fn lazy_backend_materializes_bound_residuals() -> Result<()> {
        let (_file, source, _, _) = test_source()?;
        let expected = Tensor::from_vec(vec![2f32, 3.0, 4.0, 5.0], OUT_DIM, &Device::Cpu)?;

        assert_close(
            &source.materialize_tensor("model.norm.weight", &Device::Cpu)?,
            &expected,
        )?;
        let expected_log = Tensor::from_vec(
            vec![0f32, 2f32.ln(), 3f32.ln(), 4f32.ln()],
            OUT_DIM,
            &Device::Cpu,
        )?;
        assert_close(
            &source.materialize_tensor("model.log.weight", &Device::Cpu)?,
            &expected_log,
        )?;
        let inverse_softplus =
            source.materialize_tensor("model.inverse_softplus.weight", &Device::Cpu)?;
        let reconstructed = inverse_softplus.exp()?.affine(1.0, 1.0)?.log()?;
        let original = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], OUT_DIM, &Device::Cpu)?;
        assert_close(&reconstructed, &original)?;
        let vb = source.sharded_var_builder(Device::Cpu);
        assert_eq!(
            vb.tensor_shape("model.norm.weight"),
            Some([OUT_DIM].as_slice())
        );
        let actual = vb.pp("model").pp("norm").get(OUT_DIM, "weight")?;
        assert_close(&actual, &expected)?;
        assert!(vb.contains_tensor("model.norm.weight"));

        let bf16_source = Arc::new(GgufWeightSource::new(
            source.archive().clone(),
            source.bindings(),
            DType::BF16,
        )?);
        let precise = bf16_source
            .sharded_var_builder(Device::Cpu)
            .pp("model")
            .pp("precise")
            .get_with_hints_dtype(OUT_DIM, "weight", Shard::default(), DType::F32)?;
        let expected_precise =
            Tensor::from_vec(vec![1.001f32, 2.001, 3.001, 4.001], OUT_DIM, &Device::Cpu)?;
        let max_diff = (&precise - expected_precise)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(max_diff < 0.0001, "max diff {max_diff}");

        let cast = bf16_source
            .load_linear("model.cast", &Device::Cpu, Shard::default())?
            .unwrap();
        assert_eq!(cast.dequantize_w()?.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    fn mxfp4_bindings_extract_repack_and_interleave_raw_bytes() -> Result<()> {
        let (_file, source, gate_data, up_data) = mxfp4_source()?;
        let expected_gate_blocks = native_mxfp4_blocks(&gate_data);
        let expected_up_blocks = native_mxfp4_blocks(&up_data);
        let expected_gate_scales = gate_data
            .as_chunks::<17>()
            .0
            .iter()
            .map(|block| block[0])
            .collect::<Vec<_>>();
        let expected_up_scales = up_data
            .as_chunks::<17>()
            .0
            .iter()
            .map(|block| block[0])
            .collect::<Vec<_>>();

        let gate_blocks = source.materialize_tensor("gate.blocks", &Device::Cpu)?;
        let gate_scales = source.materialize_tensor("gate.scales", &Device::Cpu)?;
        assert_eq!(gate_blocks.dtype(), DType::U8);
        assert_eq!(gate_blocks.dims(), [2, 2, 32]);
        assert_eq!(
            gate_blocks.flatten_all()?.to_vec1::<u8>()?,
            expected_gate_blocks
        );
        assert_eq!(gate_scales.dtype(), DType::U8);
        assert_eq!(gate_scales.dims(), [2, 2, 2]);
        assert_eq!(
            gate_scales.flatten_all()?.to_vec1::<u8>()?,
            expected_gate_scales
        );

        let dequantized = MXFP4Layer::from_parts(gate_blocks, gate_scales, None).dequantize_w()?;
        let fp4 = [
            0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
            -6.0,
        ];
        let mut expected_dequantized = Vec::new();
        for block in gate_data.as_chunks::<17>().0 {
            let scale = f32::from_bits(u32::from(block[0]) << 23);
            for packed in &block[1..] {
                expected_dequantized.push(fp4[(packed & 0x0f) as usize] * scale);
            }
            for packed in &block[1..] {
                expected_dequantized.push(fp4[(packed >> 4) as usize] * scale);
            }
        }
        assert_close(
            &dequantized.to_dtype(DType::F32)?,
            &Tensor::from_vec(expected_dequantized, (2, 2, 64), &Device::Cpu)?,
        )?;

        let expected_interleaved_blocks =
            interleave_rows(&expected_gate_blocks, &expected_up_blocks, 2, 2, 32);
        let expected_interleaved_scales =
            interleave_rows(&expected_gate_scales, &expected_up_scales, 2, 2, 2);
        let vb = source.sharded_var_builder(Device::Cpu).pp("native");
        let blocks = vb.get_with_hints_dtype(
            (2, 4, 2, 16),
            "gate_up_proj_blocks",
            Shard::default(),
            DType::U8,
        )?;
        let scales = vb.get_with_hints_dtype(
            (2, 4, 2),
            "gate_up_proj_scales",
            Shard::default(),
            DType::U8,
        )?;
        assert_eq!(
            blocks.flatten_all()?.to_vec1::<u8>()?,
            expected_interleaved_blocks
        );
        assert_eq!(
            scales.flatten_all()?.to_vec1::<u8>()?,
            expected_interleaved_scales
        );
        Ok(())
    }

    #[test]
    fn invalid_binding_shapes_are_rejected_at_construction() -> Result<()> {
        let (_file, source, _, _) = test_source()?;
        let bindings = GgufBindingMap::new().with_binding(
            "bad.weight",
            GgufTensorBinding::tensor("blk.0.weight").slice(0, OUT_DIM, 1),
        );
        assert!(GgufWeightSource::new(source.archive().clone(), &bindings, DType::F32).is_err());
        Ok(())
    }

    #[test]
    fn unsupported_dtypes_are_rejected_at_construction() -> Result<()> {
        let (_file, archive) = unsupported_dtype_archive()?;
        let bindings = GgufBindingMap::new().with_binding(
            "model.layers.0.linear.weight",
            GgufTensorBinding::tensor("unsupported.weight"),
        );
        let error = GgufWeightSource::new(archive, &bindings, DType::F32)
            .expect_err("IQ2_XXS must be rejected before model construction")
            .to_string();
        assert!(error.contains("unsupported.weight"));
        assert!(error.contains("IQ2_XXS (16)"));
        assert!(error.contains("Q2_K through Q8_K"));
        Ok(())
    }

    #[test]
    fn big_endian_archives_are_rejected_at_construction() -> Result<()> {
        let (file, archive) = big_endian_archive()?;
        let bindings = GgufBindingMap::new().with_binding(
            "model.layers.0.linear.weight",
            GgufTensorBinding::tensor("big.weight"),
        );
        let error = GgufWeightSource::new(archive, &bindings, DType::F32)
            .expect_err("big-endian GGUF must be rejected before model construction")
            .to_string();
        assert!(error.contains("big-endian"));
        assert!(error.contains(&file.path().display().to_string()));
        Ok(())
    }
}
