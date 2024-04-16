//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::path::PathBuf;

use candle_core::{DType, Device, Result, Var};
use candle_nn::{
    var_builder::{SimpleBackend, VarBuilderArgs},
    VarBuilder, VarMap,
};

use tqdm::Iter;

/// Load tensors into a VarBuilder backed by a VarMap using MmapedSafetensors.
/// Set `silent` to not show a progress bar.
pub(crate) fn from_mmaped_safetensors<'a>(
    paths: Vec<PathBuf>,
    xlora_paths: Vec<PathBuf>,
    dtype: DType,
    device: &Device,
    silent: bool,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>> {
    let map = VarMap::new();
    {
        let mut ws = map.data().lock().unwrap();

        for path in paths {
            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };

            if silent {
                for (name, _) in tensors.tensors() {
                    let new_name = if name.contains("base_model.model.model") {
                        name.replace("base_model.model.model", "model")
                    } else {
                        name.clone()
                    };
                    let tensor = tensors
                        .load(&name, device)?
                        .to_device(device)?
                        .to_dtype(dtype)?;
                    ws.insert(new_name, Var::from_tensor(&tensor)?);
                }
            } else {
                for (name, _) in tensors.tensors().into_iter().tqdm() {
                    let new_name = if name.contains("base_model.model.model") {
                        name.replace("base_model.model.model", "model")
                    } else {
                        name.clone()
                    };
                    let tensor = tensors
                        .load(&name, device)?
                        .to_device(device)?
                        .to_dtype(dtype)?;
                    ws.insert(new_name, Var::from_tensor(&tensor)?);
                }
            }
        }
        for (i, path) in xlora_paths.into_iter().enumerate() {
            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };

            if silent {
                for (name, _) in tensors.tensors() {
                    if name.contains("internal_xlora_classifier") {
                        continue;
                    }
                    let mut new_name = if name.contains("base_model.model.model") {
                        name.replace("base_model.model.model", "model")
                    } else {
                        name.clone()
                    };
                    let pos = new_name.find(".lora").unwrap();
                    new_name.insert_str(pos + 7, &format!(".{}", i + 1));
                    let tensor = tensors
                        .load(&name, device)?
                        .to_device(device)?
                        .to_dtype(dtype)?;
                    ws.insert(new_name, Var::from_tensor(&tensor)?);
                }
            } else {
                for (name, _) in tensors.tensors().into_iter().tqdm() {
                    if name.contains("internal_xlora_classifier") {
                        continue;
                    }
                    let mut new_name = if name.contains("base_model.model.model") {
                        name.replace("base_model.model.model", "model")
                    } else {
                        name.clone()
                    };
                    let pos = new_name.find(".lora").unwrap();
                    new_name.insert_str(pos + 7, &format!(".{}", i + 1));
                    let tensor = tensors
                        .load(&name, device)?
                        .to_device(device)?
                        .to_dtype(dtype)?;
                    ws.insert(new_name, Var::from_tensor(&tensor)?);
                }
            }
        }
    }

    Ok(VarBuilder::from_varmap(&map, dtype, device))
}
