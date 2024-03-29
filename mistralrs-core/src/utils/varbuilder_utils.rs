//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::{collections::HashMap, path::PathBuf, thread};

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
                            threadmap.insert(new_name, Var::from_tensor(&tensor)?);
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
                            threadmap.insert(new_name, Var::from_tensor(&tensor)?);
                        }
                    };
                    Ok(threadmap)
                }
                inner(path, dev, silent, dtype)
            });
            handles.push(handle)
        }
        for (i, path) in xlora_paths.into_iter().enumerate() {
            let dev = device.clone();
            let handle = thread::spawn(move || {
                fn inner(
                    path: PathBuf,
                    device: Device,
                    silent: bool,
                    dtype: DType,
                    i: usize,
                ) -> Result<HashMap<String, Var>> {
                    let mut threadmap = HashMap::new();
                    let tensors =
                        unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };

                    if silent {
                        for (name, _) in tensors.tensors() {
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
                            threadmap.insert(new_name, Var::from_tensor(&tensor)?);
                        }
                    } else {
                        for (name, _) in tensors.tensors().into_iter().tqdm() {
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
                            threadmap.insert(new_name, Var::from_tensor(&tensor)?);
                        }
                    };
                    Ok(threadmap)
                }
                inner(path, dev, silent, dtype, i)
            });
            handles.push(handle)
        }

        let mut all_finished = true;
        while !all_finished {
            all_finished = true;
            for handle in handles.iter() {
                if !handle.is_finished() {
                    all_finished = false
                }
            }
        }

        for handle in handles {
            let res = handle.join().unwrap()?;
            ws.extend(res);
        }
    }

    Ok(VarBuilder::from_varmap(&map, dtype, device))
}
