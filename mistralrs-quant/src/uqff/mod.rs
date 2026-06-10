use std::sync::Arc;

use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::var_builder::{Backend, VarBuilderArgs};

use crate::{Shard, ShardedSafeTensors};

mod reader;
mod tensor;
mod tracker;

pub use reader::UqffReader;
pub use tensor::UqffTensor;
pub use tracker::{TrackedModule, Tracker};

pub const UQFF_VERSION_MAJOR: u32 = 1;
pub const UQFF_VERSION_MINOR: u32 = 0;
pub const UQFF_VERSION_PATCH: u32 = 0;
pub(crate) const UQFF_VERSION_MAJOR_KEY: &str = "uqff.version.major";
pub(crate) const UQFF_VERSION_MINOR_KEY: &str = "uqff.version.minor";
pub(crate) const UQFF_VERSION_PATCH_KEY: &str = "uqff.version.patch";

/// Version tensors prepended to every UQFF tensor stream.
pub fn uqff_version_tensors() -> Vec<UqffTensor> {
    vec![
        UqffTensor::from_u32_scalar(UQFF_VERSION_MAJOR_KEY, UQFF_VERSION_MAJOR),
        UqffTensor::from_u32_scalar(UQFF_VERSION_MINOR_KEY, UQFF_VERSION_MINOR),
        UqffTensor::from_u32_scalar(UQFF_VERSION_PATCH_KEY, UQFF_VERSION_PATCH),
    ]
}

/// Resolve a shard against logical dims: `None` for a full load, else `(dim, start, len)`.
pub(crate) fn shard_range(shard: Shard, dims: &[usize]) -> Result<Option<(usize, usize, usize)>> {
    match shard {
        Shard::Simple { world_size: 1, .. } => Ok(None),
        Shard::Simple {
            dim,
            rank,
            world_size,
        } => {
            let size = dims[dim];
            if !size.is_multiple_of(world_size) {
                candle_core::bail!(
                    "UQFF shard dim {dim} of size {size} is not divisible by world size {world_size}."
                );
            }
            let len = size / world_size;
            Ok(Some((dim, rank * len, len)))
        }
        Shard::Offset { dim, offset, len } => {
            if offset == 0 && len == dims[dim] {
                Ok(None)
            } else {
                Ok(Some((dim, offset, len)))
            }
        }
    }
}

/// How a shard maps onto a layer's bias.
pub(crate) enum BiasShard {
    /// Embed the full bias (no shard, or a shard the bias is independent of).
    Full,
    /// Do not embed: the input dim is sharded, so the caller adds the bias post-reduce.
    Skip,
    /// The output dim is sharded; embed the matching bias slice.
    Narrow(usize, usize),
}

pub(crate) fn bias_shard(range: Option<(usize, usize, usize)>, weight_rank: usize) -> BiasShard {
    match range {
        None => BiasShard::Full,
        Some((dim, _, _)) if dim == weight_rank - 1 => BiasShard::Skip,
        Some((dim, start, len)) if dim == weight_rank - 2 && weight_rank == 2 => {
            BiasShard::Narrow(start, len)
        }
        // 3D expert layers never carry biases; anything else is a format we do not produce.
        Some(_) => BiasShard::Skip,
    }
}

/// Slice raw block-quantized data along `dim`. The last dim is packed: `block` elements per
/// `block_bytes` bytes, and rows must be whole blocks.
pub(crate) fn slice_blocked_data(
    data: &[u8],
    dims: &[usize],
    block: usize,
    block_bytes: usize,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<Vec<u8>> {
    let last = *dims.last().expect("dims is non-empty");
    if !last.is_multiple_of(block) {
        candle_core::bail!(
            "Cannot shard block-quantized data: last dim {last} is not a multiple of block size {block}."
        );
    }
    let row_bytes = last / block * block_bytes;
    if dim == dims.len() - 1 {
        if !start.is_multiple_of(block) || !len.is_multiple_of(block) {
            candle_core::bail!(
                "Sharding the packed dim requires block alignment: start {start}, len {len}, block {block}."
            );
        }
        let off = start / block * block_bytes;
        let sub = len / block * block_bytes;
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let mut out = Vec::with_capacity(rows * sub);
        for row in 0..rows {
            let base = row * row_bytes + off;
            out.extend_from_slice(&data[base..base + sub]);
        }
        Ok(out)
    } else {
        let chunk_bytes: usize =
            dims[dim + 1..dims.len() - 1].iter().product::<usize>() * row_bytes;
        let pre: usize = dims[..dim].iter().product();
        let mut out = Vec::with_capacity(pre * len * chunk_bytes);
        for p in 0..pre {
            let base = (p * dims[dim] + start) * chunk_bytes;
            out.extend_from_slice(&data[base..base + len * chunk_bytes]);
        }
        Ok(out)
    }
}

/// Canonical UQFF names for the three stacked expert layers, shared by every write site and the read probe.
pub struct UqffExpertKeys {
    pub gate: String,
    pub up: String,
    pub down: String,
}

impl UqffExpertKeys {
    pub fn new(experts_prefix: &str) -> Self {
        Self {
            gate: format!("{experts_prefix}.gate_proj"),
            up: format!("{experts_prefix}.up_proj"),
            down: format!("{experts_prefix}.down_proj"),
        }
    }
}

#[derive(Clone)]
pub struct ShardedVarBuilder {
    base: VarBuilderArgs<'static, ShardedSafeTensors>,
    tracker: Tracker,
    uqff_reader: Option<Arc<UqffReader>>,
}

impl ShardedVarBuilder {
    pub fn from_varbuilder(base: VarBuilderArgs<'static, ShardedSafeTensors>) -> Self {
        Self {
            base,
            tracker: Tracker::new(),
            uqff_reader: None,
        }
    }

    pub fn from_self(&self, base: VarBuilderArgs<'static, ShardedSafeTensors>) -> Self {
        Self {
            base,
            tracker: self.tracker.clone(),
            uqff_reader: self.uqff_reader.clone(),
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

    pub fn with_uqff_reader(mut self, reader: Arc<UqffReader>) -> Self {
        self.uqff_reader = Some(reader);
        self
    }

    pub fn uqff_reader(&self) -> Option<&Arc<UqffReader>> {
        self.uqff_reader.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            BiasShard::Narrow(4, 4)
        ));
        assert!(matches!(bias_shard(Some((2, 0, 4)), 3), BiasShard::Skip));
    }
}
