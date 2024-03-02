//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    thread,
};

use candle_core::{DType, Device, Error, Result, Tensor, Var};
use candle_nn::{
    var_builder::{SimpleBackend, VarBuilderArgs},
    VarBuilder, VarMap,
};

use tqdm::Iter;

/// Load tensors into a VarBuilder backed by a VarMap using MmapedSafetensors.
/// Set `silent` to not show a progress bar.
pub(crate) fn from_mmaped_safetensors<'a>(
    paths: Vec<PathBuf>,
    dtype: DType,
    device: &Device,
    silent: bool,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>> {
    let map = VarMap::new();
    {
        let mut ws = map.data().lock().unwrap();

        let mut handles = Vec::new();
        for path in paths {
            let dev = device.clone();
            let handle = thread::spawn(move || {
                fn inner(
                    path: PathBuf,
                    device: Device,
                    silent: bool,
                    dtype: DType,
                ) -> Result<HashMap<String, Var>> {
                    let mut threadmap = HashMap::new();
                    let tensors =
                        unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };

                    if silent {
                        for (mut name, _) in tensors.tensors() {
                            if name.contains("base_model.model.model") {
                                name = name.replace("base_mode.model.model", "model");
                            }
                            let tensor = tensors
                                .load(&name, &device)?
                                .to_device(&device)?
                                .to_dtype(dtype)?;
                            threadmap.insert(name.clone(), Var::from_tensor(&tensor)?);
                        }
                    } else {
                        for (mut name, _) in tensors.tensors().into_iter().tqdm() {
                            if name.contains("base_model.model.model") {
                                name = name.replace("base_mode.model.model", "model");
                            }
                            let tensor = tensors
                                .load(&name, &device)?
                                .to_device(&device)?
                                .to_dtype(dtype)?;
                            threadmap.insert(name.clone(), Var::from_tensor(&tensor)?);
                        }
                    };
                    Ok(threadmap)
                }
                inner(path, dev, silent, dtype)
            });
            handles.push(handle)
        }

        for handle in handles {
            if handle.is_finished() {
                let res = handle.join().unwrap()?;
                ws.extend(res);
            }
        }
    }

    Ok(VarBuilder::from_varmap(&map, dtype, device))
}
