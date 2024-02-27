//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::path::Path;

use candle_core::{DType, Device, Error, Var};
use candle_nn::{
    var_builder::{SimpleBackend, VarBuilderArgs},
    VarBuilder, VarMap,
};

use tqdm::Iter;

/// Load tensors into a VarBuilder backed by a VarMap using MmapedSafetensors.
/// Set `silent` to not show a progress bar.
pub(crate) fn from_mmaped_safetensors<'a, P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
    silent: bool,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, Error> {
    let map = VarMap::new();
    {
        let mut ws = map.data().lock().unwrap();

        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(paths)? };

        if silent {
            for (name, _) in tensors.tensors() {
                let tensor = tensors
                    .load(&name, device)?
                    .to_device(device)?
                    .to_dtype(dtype)?;
                ws.insert(name.clone(), Var::from_tensor(&tensor)?);
            }
        } else {
            for (name, _) in tensors.tensors().iter().tqdm() {
                let tensor = tensors
                    .load(name, device)?
                    .to_device(device)?
                    .to_dtype(dtype)?;
                ws.insert(name.clone(), Var::from_tensor(&tensor)?);
            }
        };
    }

    Ok(VarBuilder::from_varmap(&map, dtype, device))
}
