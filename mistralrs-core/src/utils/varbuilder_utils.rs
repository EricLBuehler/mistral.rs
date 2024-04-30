//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
    thread,
};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    var_builder::{SimpleBackend, VarBuilderArgs},
    VarBuilder,
};
use indicatif::MultiProgress;

use crate::{get_mut_arcmutex, utils::new_progress_bar};

/// Load tensors into a VarBuilder backed by a VarMap using MmapedSafetensors.
/// Set `silent` to not show a progress bar.
pub(crate) fn from_mmaped_safetensors<'a>(
    paths: Vec<PathBuf>,
    xlora_paths: Vec<PathBuf>,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>> {
    let mut handles = Vec::new();
    let multi = Arc::new(Mutex::new(MultiProgress::new()));
    for path in paths {
        fn load(
            path: PathBuf,
            device: Device,
            dtype: DType,
            multi: Arc<Mutex<MultiProgress>>,
        ) -> Result<HashMap<String, Tensor>> {
            let mut accum = HashMap::new();
            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };

            let bar = new_progress_bar(tensors.tensors().len() as u64);
            {
                let multi = get_mut_arcmutex!(multi);
                multi.add(bar.clone());
            }
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
                bar.inc(1);
            }
            bar.finish();
            Ok(accum)
        }
        let device = device.clone();
        let multi = multi.clone();
        handles.push(thread::spawn(move || load(path, device, dtype, multi)));
    }
    for (i, path) in xlora_paths.into_iter().enumerate() {
        fn load(
            path: PathBuf,
            device: Device,
            dtype: DType,
            i: usize,
            multi: Arc<Mutex<MultiProgress>>,
        ) -> Result<HashMap<String, Tensor>> {
            let mut accum = HashMap::new();
            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };

            let bar = new_progress_bar(tensors.tensors().len() as u64);
            {
                let multi = get_mut_arcmutex!(multi);
                multi.add(bar.clone());
            }
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
                bar.inc(1);
            }
            bar.finish();
            Ok(accum)
        }
        let device = device.clone();
        let multi = multi.clone();
        handles.push(thread::spawn(move || load(path, device, dtype, i, multi)));
    }
    let mut ws = HashMap::new();
    // Wait until each finishes
    while !handles.iter().all(|h| h.is_finished()) {}
    for h in handles {
        ws.extend(h.join().unwrap()?);
    }
    Ok(VarBuilder::from_tensors(ws, dtype, device))
}
