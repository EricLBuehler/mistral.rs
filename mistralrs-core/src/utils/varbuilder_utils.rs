//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::{collections::HashMap, path::PathBuf, thread};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    var_builder::{SimpleBackend, VarBuilderArgs},
    VarBuilder,
};

use crate::lora::LoraConfig;
use crate::utils::progress::IterWithProgress;

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
        let device = device.clone();
        handles.push(thread::spawn(move || {
            Common::load_tensors_from_path(&path, &device, dtype, silent)
        }));
    }
    for path in xlora_paths {
        let device = device.clone();
        handles.push(thread::spawn(move || {
            XLora::load_tensors_from_path(&path, &device, dtype, silent)
        }));
    }

    let mut ws = HashMap::new();
    // Wait until all spawned threads have finished loading tensors:
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
            let loaded_tensors = Common::load_tensors_from_path(path, device, dtype, silent)?;

            map.insert(
                name.clone(),
                (
                    VarBuilder::from_tensors(loaded_tensors, dtype, device),
                    config.clone(),
                ),
            );
        }
        Ok(Some(map))
    } else {
        Ok(None)
    }
}

// Presently this logic only needs to diverge for X-LoRA support via `into_name_key_pairs()`
trait LoadTensors {
    fn load_tensors_from_path(
        path: &PathBuf,
        device: &Device,
        dtype: DType,
        is_silent: bool,
    ) -> Result<HashMap<String, Tensor>> {
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };

        // Extracts the tensor name and processes it, filtering tensors and deriving the key name:
        let names_only = tensors.tensors().into_iter().map(|(name, _)| name);
        let iter = Self::into_name_key_pairs(names_only);

        // Take the filtered list of tensors to load, store with derived lookup key:
        let mut loaded_tensors = HashMap::new();
        for (load_name, key_name) in iter.with_progress(is_silent) {
            let tensor = tensors
                .load(&load_name, device)?
                // TODO: Seems redundant? Tensor was just loaded to this device?
                // .to_device(&device)?
                .to_dtype(dtype)?;

            loaded_tensors.insert(key_name, tensor);
        }

        Ok(loaded_tensors)
    }

    fn into_name_key_pairs(
        tensors: impl Iterator<Item = String>,
    ) -> impl Iterator<Item = (String, String)> {
        tensors.map(|name| {
            let new_name = name.replace("base_model.model.model", "model");

            (name, new_name)
        })
    }
}

struct Common {}
impl LoadTensors for Common {}

struct XLora {}
impl LoadTensors for XLora {
    fn into_name_key_pairs(
        tensors: impl Iterator<Item = String>,
    ) -> impl Iterator<Item = (String, String)> {
        let expectation = "tensor name `{new_name}` should have substring `.lora`";

        tensors
            .filter(|name| !name.contains("internal_xlora_classifier"))
            .enumerate()
            .map(|(i, name)| {
                let mut new_name = name.replace("base_model.model.model", "model");
                // TODO: Add better context to describe intent / requirement:
                let pos = new_name.find(".lora").expect(expectation);
                new_name.insert_str(pos + 7, &format!(".{i}"));

                (name, new_name)
            })
    }
}
