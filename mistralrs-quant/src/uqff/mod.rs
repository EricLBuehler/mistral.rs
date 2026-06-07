use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::var_builder::{Backend, VarBuilderArgs};

use crate::{uqff::tracker::Tracker, ShardedSafeTensors};

mod tracker;

#[derive(Clone)]
pub struct ShardedVarBuilder {
    base: VarBuilderArgs<'static, ShardedSafeTensors>,
    tracker: Tracker,
}

impl ShardedVarBuilder {
    pub fn from_varbuilder(base: VarBuilderArgs<'static, ShardedSafeTensors>) -> Self {
        Self {
            base,
            tracker: Tracker::new(),
        }
    }

    pub fn from_self(&self, base: VarBuilderArgs<'static, ShardedSafeTensors>) -> Self {
        Self {
            base,
            tracker: self.tracker.clone(),
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
}
