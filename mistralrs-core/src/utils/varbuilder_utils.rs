//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::{collections::HashMap, path::PathBuf, thread};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    var_builder::{SimpleBackend, VarBuilderArgs},
    VarBuilder,
};

use crate::lora::LoraConfig;
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
    let mut handles = Vec::new();
    for path in paths {
        fn load(
            path: PathBuf,
            silent: bool,
            device: Device,
            dtype: DType,
        ) -> Result<HashMap<String, Tensor>> {
            let mut accum = HashMap::new();
            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };

            if silent {
                for (name, _) in tensors.tensors() {
                    let new_name = if name.contains("base_model.model.model") {
                        name.replace("base_model.model.model", "model")
                    } else {
                        name.clone()
                    };
                    let tensor = tensors
                        .load(&name, &device)?
                        .to_device(&device)?
                        .to_dtype(dtype)?;
                    accum.insert(new_name, tensor);
                }
            } else {
                for (name, _) in tensors.tensors().into_iter().tqdm() {
                    let new_name = if name.contains("base_model.model.model") {
                        name.replace("base_model.model.model", "model")
                    } else {
                        name.clone()
                    };
                    let tensor = tensors
                        .load(&name, &device)?
                        .to_device(&device)?
                        .to_dtype(dtype)?;
                    accum.insert(new_name, tensor);
                }
            }
            Ok(accum)
        }
        let device = device.clone();
        handles.push(thread::spawn(move || load(path, silent, device, dtype)));
    }
    for (i, path) in xlora_paths.into_iter().enumerate() {
        fn load(
            path: PathBuf,
            silent: bool,
            device: Device,
            dtype: DType,
            i: usize,
        ) -> Result<HashMap<String, Tensor>> {
            let mut accum = HashMap::new();
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
                        .load(&name, &device)?
                        .to_device(&device)?
                        .to_dtype(dtype)?;
                    accum.insert(new_name, tensor);
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
                        .load(&name, &device)?
                        .to_device(&device)?
                        .to_dtype(dtype)?;
                    accum.insert(new_name, tensor);
                }
            }
            Ok(accum)
        }
        let device = device.clone();
        handles.push(thread::spawn(move || load(path, silent, device, dtype, i)));
    }
    let mut ws = HashMap::new();
    // Wait until each finishes
    while !handles.iter().all(|h| h.is_finished()) {}
    for h in handles {
        ws.extend(h.join().unwrap()?);
    }
    Ok(VarBuilder::from_tensors(ws, dtype, device))
}

pub(crate) fn load_preload_adapters<'a>(
    paths: &Option<HashMap<String, (PathBuf, LoraConfig)>>,
    dtype: DType,
    device: &Device,
    silent: bool,
) -> Result<Option<HashMap<String, (VarBuilder<'a>, LoraConfig)>>> {
    if let Some(paths) = paths {
        let mut map = HashMap::new();
        for (name, (path, config)) in paths {
            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
            let mut accum = HashMap::new();

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
                    accum.insert(new_name, tensor);
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
                    accum.insert(new_name, tensor);
                }
            }
            map.insert(
                name.clone(),
                (
                    VarBuilder::from_tensors(accum, dtype, device),
                    config.clone(),
                ),
            );
        }
        Ok(Some(map))
    } else {
        Ok(None)
    }
}
