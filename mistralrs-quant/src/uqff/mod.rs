use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::var_builder::{Backend, VarBuilderArgs};
use safetensors::tensor::Dtype;

use crate::{QuantizedSerdeType, QuantizedWeightSource, Shard, ShardedSafeTensors};

mod reader;
mod report;
mod tensor;
mod tracker;

pub use reader::UqffReader;
pub use report::{
    build_output_report_from_layers, build_uqff_report, build_uqff_report_from_artifacts,
    inspect_uqff_artifacts, inspect_uqff_path, stored_type_from_tensors, verify_uqff_artifacts,
    verify_uqff_path, write_uqff_report, QuantizationIssue, QuantizationReport, UqffArtifactFile,
    UqffArtifactGroup, UqffArtifacts, UqffFallbackReport, UqffGeneratedBy, UqffInspection,
    UqffLayerReport, UqffMetadataSummary, UqffOutputReport, UqffReport, UqffReportOptions,
    UqffTensorSummary, UqffVerifyOptions, UqffVerifyResult, UQFF_REPORT_JSON,
};
pub use tensor::UqffTensor;
pub use tracker::{TrackedModule, Tracker};

pub const UQFF_VERSION_MAJOR: u32 = 1;
pub const UQFF_VERSION_MINOR: u32 = 2;
pub const UQFF_VERSION_PATCH: u32 = 0;
pub(crate) const UQFF_VERSION_MAJOR_KEY: &str = "uqff.version.major";
pub(crate) const UQFF_VERSION_MINOR_KEY: &str = "uqff.version.minor";
pub(crate) const UQFF_VERSION_PATCH_KEY: &str = "uqff.version.patch";
pub(crate) const UQFF_WEIGHT_FORMAT_SUFFIX: &str = "weight.format";

#[derive(Clone, Debug)]
pub struct UqffTensorHeader {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct UqffLayerHeaderView<'a> {
    prefix: &'a str,
    tensors: &'a HashMap<String, UqffTensorHeader>,
}

impl<'a> UqffLayerHeaderView<'a> {
    pub fn new(prefix: &'a str, tensors: &'a HashMap<String, UqffTensorHeader>) -> Self {
        Self { prefix, tensors }
    }

    pub fn tensor(&self, suffix: &str) -> Option<&UqffTensorHeader> {
        self.tensors.get(&format!("{}.{suffix}", self.prefix))
    }

    pub fn has(&self, suffix: &str) -> bool {
        self.tensor(suffix).is_some()
    }

    pub fn scalar(&self, suffix: &str, dtype: Dtype) -> bool {
        self.tensor(suffix)
            .is_some_and(|tensor| tensor.dtype == dtype && tensor.shape.is_empty())
    }

    pub fn tensor_dtype(&self, suffix: &str, dtype: Dtype) -> bool {
        self.tensor(suffix)
            .is_some_and(|tensor| tensor.dtype == dtype && !tensor.shape.is_empty())
    }

    pub fn u32_vector(&self, suffix: &str) -> bool {
        self.tensor(suffix)
            .is_some_and(|tensor| tensor.dtype == Dtype::U32 && tensor.shape.len() == 1)
    }

    pub fn exact_weight_suffixes(&self, allowed: &[&str]) -> bool {
        let allowed = allowed.iter().copied().collect::<HashSet<_>>();
        self.tensors
            .keys()
            .filter_map(|name| self.weight_suffix(name))
            .all(|suffix| allowed.contains(suffix))
            && allowed.iter().all(|suffix| self.has(suffix))
    }

    fn weight_suffix<'b>(&self, name: &'b str) -> Option<&'b str> {
        let suffix = name.strip_prefix(self.prefix)?.strip_prefix('.')?;
        if suffix == "weight" || suffix.starts_with("weight.") {
            Some(suffix)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct UqffHeaderMatch {
    pub serde_type: QuantizedSerdeType,
}

/// Version tensors prepended to every UQFF tensor stream.
pub fn uqff_version_tensors() -> Vec<UqffTensor> {
    vec![
        UqffTensor::from_u32_scalar(UQFF_VERSION_MAJOR_KEY, UQFF_VERSION_MAJOR),
        UqffTensor::from_u32_scalar(UQFF_VERSION_MINOR_KEY, UQFF_VERSION_MINOR),
        UqffTensor::from_u32_scalar(UQFF_VERSION_PATCH_KEY, UQFF_VERSION_PATCH),
    ]
}

/// Resolve a shard against logical dims: `None` for a full load, else `(dim, start, len)`.
pub fn shard_range(shard: Shard, dims: &[usize]) -> Result<Option<(usize, usize, usize)>> {
    match shard {
        Shard::Simple {
            dim,
            rank,
            world_size,
        } => {
            let size = *dims.get(dim).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "Cannot shard dimension {dim} of rank-{} tensor.",
                    dims.len()
                ))
            })?;
            if world_size == 0 {
                candle_core::bail!("Shard world size must be non-zero.");
            }
            if rank >= world_size {
                candle_core::bail!("Shard rank {rank} is outside world size {world_size}.");
            }
            if world_size == 1 {
                return Ok(None);
            }
            if !size.is_multiple_of(world_size) {
                candle_core::bail!(
                    "Weight shard dim {dim} of size {size} is not divisible by world size {world_size}."
                );
            }
            let len = size / world_size;
            Ok(Some((dim, rank * len, len)))
        }
        Shard::Offset { dim, offset, len } => {
            let size = *dims.get(dim).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "Cannot shard dimension {dim} of rank-{} tensor.",
                    dims.len()
                ))
            })?;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| candle_core::Error::Msg("Shard range overflow.".to_string()))?;
            if end > size {
                candle_core::bail!(
                    "Shard range {offset}..{end} exceeds dimension {dim} of size {size}."
                );
            }
            if offset == 0 && len == size {
                Ok(None)
            } else {
                Ok(Some((dim, offset, len)))
            }
        }
    }
}

/// How a shard maps onto a layer's bias.
pub enum BiasShard {
    /// Embed the full bias (no shard, or a shard the bias is independent of).
    Full,
    /// Do not embed: the input dim is sharded, so the caller adds the bias post-reduce.
    Skip,
    /// A non-input weight dim is sharded; embed the matching bias slice.
    Narrow {
        dim: usize,
        start: usize,
        len: usize,
    },
}

pub fn bias_shard(range: Option<(usize, usize, usize)>, weight_rank: usize) -> BiasShard {
    if weight_rank < 2 {
        return if range.is_none() {
            BiasShard::Full
        } else {
            BiasShard::Skip
        };
    }
    match range {
        None => BiasShard::Full,
        Some((dim, _, _)) if dim == weight_rank - 1 => BiasShard::Skip,
        Some((dim, start, len)) if dim < weight_rank - 1 => BiasShard::Narrow { dim, start, len },
        Some(_) => BiasShard::Skip,
    }
}

pub(crate) fn tensor_with_suffix<'a>(
    tensors: &'a [UqffTensor],
    prefix: &str,
    suffix: &str,
) -> Option<&'a UqffTensor> {
    let key = format!("{prefix}.{suffix}");
    tensors.iter().find(|tensor| tensor.name() == key)
}

pub(crate) fn u8_scalar_with_suffix(
    tensors: &[UqffTensor],
    prefix: &str,
    suffix: &str,
) -> Result<u8> {
    tensor_with_suffix(tensors, prefix, suffix)
        .ok_or_else(|| candle_core::Error::Msg(format!("Missing `{prefix}.{suffix}`")))?
        .scalar_u8()
}

pub(crate) fn u32_scalar_with_suffix(
    tensors: &[UqffTensor],
    prefix: &str,
    suffix: &str,
) -> Result<u32> {
    tensor_with_suffix(tensors, prefix, suffix)
        .ok_or_else(|| candle_core::Error::Msg(format!("Missing `{prefix}.{suffix}`")))?
        .scalar_u32()
}

/// Slice raw block-quantized data along `dim`. The last dim is packed: `block` elements per
/// `block_bytes` bytes, and rows must be whole blocks.
pub fn slice_blocked_data(
    data: &[u8],
    dims: &[usize],
    block: usize,
    block_bytes: usize,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<Vec<u8>> {
    if block == 0 || block_bytes == 0 {
        candle_core::bail!("Packed block sizes must be non-zero.");
    }
    let Some(&last) = dims.last() else {
        candle_core::bail!("Cannot shard scalar packed data.");
    };
    let size = *dims.get(dim).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "Cannot shard dimension {dim} of rank-{} packed tensor.",
            dims.len()
        ))
    })?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| candle_core::Error::Msg("Packed shard range overflow.".to_string()))?;
    if end > size {
        candle_core::bail!(
            "Packed shard range {start}..{end} exceeds dimension {dim} of size {size}."
        );
    }
    if !last.is_multiple_of(block) {
        candle_core::bail!(
            "Cannot shard block-quantized data: last dim {last} is not a multiple of block size {block}."
        );
    }
    let row_bytes = (last / block)
        .checked_mul(block_bytes)
        .ok_or_else(|| candle_core::Error::Msg("Packed row byte size overflow.".to_string()))?;
    let rows = dims[..dims.len() - 1]
        .iter()
        .try_fold(1usize, |count, size| {
            count.checked_mul(*size).ok_or_else(|| {
                candle_core::Error::Msg("Packed tensor row count overflow.".to_string())
            })
        })?;
    let expected_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| candle_core::Error::Msg("Packed tensor byte size overflow.".to_string()))?;
    if data.len() < expected_bytes {
        candle_core::bail!(
            "Packed tensor needs {expected_bytes} bytes for shape {dims:?}, but only {} are available.",
            data.len()
        );
    }
    if dim == dims.len() - 1 {
        if !start.is_multiple_of(block) || !len.is_multiple_of(block) {
            candle_core::bail!(
                "Sharding the packed dim requires block alignment: start {start}, len {len}, block {block}."
            );
        }
        let off = start / block * block_bytes;
        let sub = len / block * block_bytes;
        let mut out = Vec::with_capacity(rows * sub);
        for row in 0..rows {
            let base = row * row_bytes + off;
            out.extend_from_slice(&data[base..base + sub]);
        }
        Ok(out)
    } else {
        let inner = dims[dim + 1..dims.len() - 1]
            .iter()
            .try_fold(1usize, |count, size| {
                count.checked_mul(*size).ok_or_else(|| {
                    candle_core::Error::Msg("Packed tensor inner size overflow.".to_string())
                })
            })?;
        let chunk_bytes = inner.checked_mul(row_bytes).ok_or_else(|| {
            candle_core::Error::Msg("Packed tensor chunk size overflow.".to_string())
        })?;
        let pre = dims[..dim].iter().try_fold(1usize, |count, size| {
            count.checked_mul(*size).ok_or_else(|| {
                candle_core::Error::Msg("Packed tensor outer size overflow.".to_string())
            })
        })?;
        let capacity = pre
            .checked_mul(len)
            .and_then(|value| value.checked_mul(chunk_bytes))
            .ok_or_else(|| {
                candle_core::Error::Msg("Packed shard byte size overflow.".to_string())
            })?;
        let mut out = Vec::with_capacity(capacity);
        for p in 0..pre {
            let base = (p * dims[dim] + start) * chunk_bytes;
            out.extend_from_slice(&data[base..base + len * chunk_bytes]);
        }
        Ok(out)
    }
}

pub struct QuantizedExpertKeys {
    pub gate: String,
    pub up: String,
    pub down: String,
}

impl QuantizedExpertKeys {
    pub fn new(experts_prefix: &str) -> Self {
        Self {
            gate: format!("{experts_prefix}.gate_proj"),
            up: format!("{experts_prefix}.up_proj"),
            down: format!("{experts_prefix}.down_proj"),
        }
    }
}

pub type UqffExpertKeys = QuantizedExpertKeys;

#[derive(Clone)]
pub struct ShardedVarBuilder {
    base: VarBuilderArgs<'static, ShardedSafeTensors>,
    tracker: Tracker,
    weight_source: Option<Arc<dyn QuantizedWeightSource>>,
    shapes: Option<Arc<HashMap<String, Vec<usize>>>>,
    lora_registry: Option<Arc<crate::LoraLayerRegistry>>,
}

impl ShardedVarBuilder {
    pub fn from_varbuilder(base: VarBuilderArgs<'static, ShardedSafeTensors>) -> Self {
        Self {
            base,
            tracker: Tracker::new(),
            weight_source: None,
            shapes: None,
            lora_registry: None,
        }
    }

    pub(crate) fn with_shapes(mut self, shapes: HashMap<String, Vec<usize>>) -> Self {
        self.shapes = Some(Arc::new(shapes));
        self
    }

    /// Shape of `name` under the current prefix, when the backing store exposes an index.
    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        let prefix = self.base.prefix();
        let full = if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}.{name}")
        };
        self.shapes.as_ref()?.get(&full).map(|s| &s[..])
    }

    pub fn tensor_names(&self) -> Option<Vec<String>> {
        let mut names = self.shapes.as_ref()?.keys().cloned().collect::<Vec<_>>();
        names.sort();
        Some(names)
    }

    pub fn from_self(&self, base: VarBuilderArgs<'static, ShardedSafeTensors>) -> Self {
        Self {
            base,
            tracker: self.tracker.clone(),
            weight_source: self.weight_source.clone(),
            shapes: self.shapes.clone(),
            lora_registry: self.lora_registry.clone(),
        }
    }

    /// Returns the prefix of the `VarBuilder`.
    pub fn prefix(&self) -> String {
        self.base.prefix()
    }

    /// Returns a new `VarBuilder` using the root path.
    pub fn root(&self) -> Self {
        self.from_self(self.base.root())
    }

    /// Returns a new `VarBuilder` with the prefix set to `prefix`.
    pub fn set_prefix(&self, prefix: impl ToString) -> Self {
        self.from_self(self.base.set_prefix(prefix))
    }

    /// Return a new `VarBuilder` adding `s` to the current prefix. This can be think of as `cd`
    /// into a directory.
    pub fn push_prefix<S: ToString>(&self, s: S) -> Self {
        self.from_self(self.base.push_prefix(s))
    }

    /// Short alias for `push_prefix`.
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        self.push_prefix(s)
    }

    /// The device used by default.
    pub fn device(&self) -> &Device {
        self.base.device()
    }

    /// The dtype used by default.
    pub fn dtype(&self) -> DType {
        self.base.dtype()
    }

    /// Clone the VarBuilder tweaking its dtype
    pub fn to_dtype(&self, dtype: DType) -> Self {
        self.from_self(self.base.to_dtype(dtype))
    }

    /// This returns true only if a tensor with the passed in name is available. E.g. when passed
    /// `a`, true is returned if `prefix.a` exists but false is returned if only `prefix.a.b`
    /// exists.
    pub fn contains_tensor(&self, tensor_name: &str) -> bool {
        self.base.contains_tensor(tensor_name)
    }

    /// Retrieve the tensor associated with the given name at the current path.
    pub fn get_with_hints<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        hints: <ShardedSafeTensors as Backend>::Hints,
    ) -> Result<Tensor> {
        self.base.get_with_hints(s, name, hints)
    }

    /// Retrieve the tensor associated with the given name at the current path.
    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Tensor> {
        self.base.get(s, name)
    }

    /// Retrieve the tensor associated with the given name at the current path.
    pub fn get_unchecked(&self, name: &str) -> Result<Tensor> {
        self.base.get_unchecked(name)
    }

    /// Retrieve the tensor associated with the given name & dtype at the current path.
    pub fn get_unchecked_dtype(&self, name: &str, dtype: DType) -> Result<Tensor> {
        self.base.get_unchecked_dtype(name, dtype)
    }

    /// Retrieve the tensor associated with the given name & dtype at the current path.
    pub fn get_with_hints_dtype<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        hints: <ShardedSafeTensors as Backend>::Hints,
        dtype: DType,
    ) -> Result<Tensor> {
        self.base.get_with_hints_dtype(s, name, hints, dtype)
    }

    /// Set the device of the VarBuilder.
    pub fn set_device(self, device: Device) -> Self {
        self.from_self(self.base.clone().set_device(device))
    }

    /// Set the dtype of the VarBuilder.
    pub fn set_dtype(self, dtype: DType) -> Self {
        self.from_self(self.base.clone().set_dtype(dtype))
    }

    pub fn tracker(&self) -> &Tracker {
        &self.tracker
    }

    pub fn with_weight_source(mut self, source: Arc<dyn QuantizedWeightSource>) -> Self {
        self.weight_source = Some(source);
        self
    }

    pub fn weight_source(&self) -> Option<&Arc<dyn QuantizedWeightSource>> {
        self.weight_source.as_ref()
    }

    pub fn with_uqff_reader(self, reader: Arc<UqffReader>) -> Self {
        self.with_weight_source(reader)
    }

    pub fn with_lora_registry(mut self, registry: Arc<crate::LoraLayerRegistry>) -> Self {
        self.lora_registry = Some(registry);
        self
    }

    pub fn without_lora_registry(mut self) -> Self {
        self.lora_registry = None;
        self
    }

    pub fn lora_registry(&self) -> Option<&Arc<crate::LoraLayerRegistry>> {
        self.lora_registry.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestWeightSource;

    impl QuantizedWeightSource for TestWeightSource {
        fn contains(&self, name: &str) -> bool {
            name == "model.weight"
        }

        fn load_linear(
            &self,
            _key: &str,
            _device: &Device,
            _shard: Shard,
        ) -> Result<Option<Arc<dyn crate::QuantMethod>>> {
            Ok(None)
        }

        fn load_optional_tensor(&self, _name: &str, _device: &Device) -> Result<Option<Tensor>> {
            Ok(None)
        }

        fn shard_alignment(&self, _key: &str) -> Result<usize> {
            Ok(1)
        }

        fn pack_factor(&self, _dtype: DType) -> Result<usize> {
            Ok(1)
        }

        fn pack_factor_for(&self, _key: &str, _dtype: DType) -> Result<Option<usize>> {
            Ok(Some(1))
        }
    }

    #[test]
    fn weight_source_survives_builder_transforms() {
        let backend: HashMap<String, Tensor> = HashMap::new();
        let vb = ShardedSafeTensors::wrap(backend, DType::F32, Device::Cpu)
            .with_weight_source(Arc::new(TestWeightSource));
        let vb = vb.pp("model").to_dtype(DType::F64);

        assert!(vb
            .weight_source()
            .is_some_and(|source| source.contains("model.weight")));
    }

    #[test]
    fn lora_registry_can_be_removed_from_one_builder_subtree() {
        let backend: HashMap<String, Tensor> = HashMap::new();
        let registry = Arc::new(crate::LoraLayerRegistry::new());
        let vb = ShardedSafeTensors::wrap(backend, DType::F32, Device::Cpu)
            .with_lora_registry(registry.clone());
        let language = vb.pp("language_model");
        let vision = vb.pp("vision_model").without_lora_registry();

        assert!(language
            .lora_registry()
            .is_some_and(|candidate| Arc::ptr_eq(candidate, &registry)));
        assert!(vision.lora_registry().is_none());
        assert!(vb.lora_registry().is_some());
    }

    #[test]
    fn shard_range_resolves() -> Result<()> {
        let dims = [8, 64];
        assert!(shard_range(Shard::default(), &dims)?.is_none());
        assert_eq!(
            shard_range(
                Shard::Simple {
                    dim: 1,
                    rank: 1,
                    world_size: 2
                },
                &dims
            )?,
            Some((1, 32, 32))
        );
        assert_eq!(
            shard_range(
                Shard::Offset {
                    dim: 0,
                    offset: 2,
                    len: 4
                },
                &dims
            )?,
            Some((0, 2, 4))
        );
        assert!(shard_range(
            Shard::Offset {
                dim: 0,
                offset: 0,
                len: 8
            },
            &dims
        )?
        .is_none());
        assert!(shard_range(
            Shard::Simple {
                dim: 1,
                rank: 0,
                world_size: 3
            },
            &dims
        )
        .is_err());
        Ok(())
    }

    // 2 elements per block, 1 byte per block: byte value encodes the block index.
    fn rows(dims: &[usize]) -> Vec<u8> {
        let blocks: usize = dims.iter().product::<usize>() / 2;
        (0..blocks).map(|b| b as u8).collect()
    }

    #[test]
    fn slice_blocked_last_dim() -> Result<()> {
        let dims = [3, 8];
        let data = rows(&dims);
        let out = slice_blocked_data(&data, &dims, 2, 1, 1, 4, 4)?;
        assert_eq!(out, vec![2, 3, 6, 7, 10, 11]);
        assert!(slice_blocked_data(&data, &dims, 2, 1, 1, 1, 4).is_err());
        Ok(())
    }

    #[test]
    fn slice_blocked_outer_dim() -> Result<()> {
        let dims = [3, 8];
        let data = rows(&dims);
        let out = slice_blocked_data(&data, &dims, 2, 1, 0, 1, 2)?;
        assert_eq!(out, vec![4, 5, 6, 7, 8, 9, 10, 11]);
        Ok(())
    }

    #[test]
    fn slice_blocked_3d_middle_dim() -> Result<()> {
        let dims = [2, 4, 4];
        let data = rows(&dims);
        let out = slice_blocked_data(&data, &dims, 2, 1, 1, 2, 2)?;
        assert_eq!(out, vec![4, 5, 6, 7, 12, 13, 14, 15]);
        let out = slice_blocked_data(&data, &dims, 2, 1, 2, 2, 2)?;
        assert_eq!(out, vec![1, 3, 5, 7, 9, 11, 13, 15]);
        Ok(())
    }

    #[test]
    fn bias_follows_shard_semantics() {
        assert!(matches!(bias_shard(None, 2), BiasShard::Full));
        assert!(matches!(bias_shard(Some((1, 0, 4)), 2), BiasShard::Skip));
        assert!(matches!(
            bias_shard(Some((0, 4, 4)), 2),
            BiasShard::Narrow {
                dim: 0,
                start: 4,
                len: 4
            }
        ));
        assert!(matches!(bias_shard(Some((2, 0, 4)), 3), BiasShard::Skip));
        assert!(matches!(
            bias_shard(Some((1, 4, 4)), 3),
            BiasShard::Narrow {
                dim: 1,
                start: 4,
                len: 4
            }
        ));
        assert!(matches!(
            bias_shard(Some((0, 1, 2)), 3),
            BiasShard::Narrow {
                dim: 0,
                start: 1,
                len: 2
            }
        ));
    }
}
