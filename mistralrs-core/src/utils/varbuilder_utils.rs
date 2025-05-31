//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    thread::{self, JoinHandle},
};

use candle_core::{pickle::PthTensors, DType, Device, Result, Tensor};
use mistralrs_quant::{safetensors::MmapedSafetensors, ShardedSafeTensors, ShardedVarBuilder};
use regex::Regex;

use crate::lora::LoraConfig;
use crate::utils::progress::IterWithProgress;
use derive_new::new;

trait TensorLoaderBackend {
    fn get_names(&self) -> Vec<String>;
    fn load_name(&self, name: &str, device: &Device, dtype: Option<DType>) -> Result<Tensor>;
}

struct SafetensorBackend(MmapedSafetensors);

impl TensorLoaderBackend for SafetensorBackend {
    fn get_names(&self) -> Vec<String> {
        self.0
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
    }
    fn load_name(&self, name: &str, device: &Device, dtype: Option<DType>) -> Result<Tensor> {
        self.0.load(name, device, dtype)
    }
}

struct PickleBackend(PthTensors);

impl TensorLoaderBackend for PickleBackend {
    fn get_names(&self) -> Vec<String> {
        self.0.tensor_infos().keys().cloned().collect::<Vec<_>>()
    }
    fn load_name(&self, name: &str, device: &Device, _dtype: Option<DType>) -> Result<Tensor> {
        self.0
            .get(name)?
            .ok_or(candle_core::Error::Msg(format!(
                "Could not load tensor {name}"
            )))?
            .to_device(device)
    }
}

pub enum DeviceForLoadTensor {
    Base,
    Idx(usize),
}

/// Load tensors into a VarBuilder backed by a VarMap using MmapedSafetensors.
/// Set `silent` to not show a progress bar.
///
/// # Predicate semantics:
/// - If `regexes` is specified, this will be used in `make_dummy_predicate` based on `.any`
/// - Otherwise, only include keys for which predicate evaluates to true.
#[allow(clippy::too_many_arguments)]
pub(crate) fn from_mmaped_safetensors(
    paths: Vec<PathBuf>,
    xlora_paths: Vec<PathBuf>,
    dtype: Option<DType>,
    base_device: &Device,
    layer_devices: Vec<Option<Device>>,
    silent: bool,
    make_dummy_regexes: Option<Arc<Vec<Regex>>>,
    predicate: impl Fn(String) -> bool + Send + Sync + Clone + 'static,
    get_device_for_tensor: Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>,
) -> Result<ShardedVarBuilder> {
    // No mmap for cuda.
    if xlora_paths.is_empty() && !base_device.is_cuda() || cfg!(feature = "ring") {
        if !silent {
            tracing::info!("Loading model using mmap strategy.");
        }
        return Ok(unsafe {
            ShardedSafeTensors::sharded(
                &paths,
                dtype.unwrap_or(DType::F16),
                base_device,
                make_dummy_regexes,
                Arc::new(predicate),
            )?
        });
    }

    #[allow(clippy::type_complexity)]
    let mut handles: Vec<JoinHandle<Result<HashMap<String, Tensor>>>> = Vec::new();

    for path in paths {
        let base_device = base_device.clone();
        let layer_devices = layer_devices.clone();
        let get_device_for_tensor = get_device_for_tensor.clone();
        if let Some(regexes) = make_dummy_regexes.clone() {
            let predicate = predicate.clone();
            handles.push(thread::spawn(Box::new(move || {
                let loader = Common::new();
                loader.load_tensors_from_path(
                    &path,
                    &base_device,
                    layer_devices,
                    get_device_for_tensor,
                    dtype,
                    silent,
                    predicate,
                    |key| regexes.iter().any(|r| r.is_match(key)),
                )
            })));
        } else {
            let predicate = predicate.clone();
            handles.push(thread::spawn(Box::new(move || {
                let loader = Common::new();
                loader.load_tensors_from_path(
                    &path,
                    &base_device,
                    layer_devices,
                    get_device_for_tensor,
                    dtype,
                    silent,
                    predicate,
                    |_| false,
                )
            })));
        }
    }
    for (i, path) in xlora_paths.into_iter().enumerate() {
        let base_device = base_device.clone();
        let layer_devices = layer_devices.clone();
        let get_device_for_tensor = get_device_for_tensor.clone();
        if let Some(regexes) = make_dummy_regexes.clone() {
            let predicate = predicate.clone();
            handles.push(thread::spawn(Box::new(move || {
                let loader = XLora::new(i + 1);
                loader.load_tensors_from_path(
                    &path,
                    &base_device,
                    layer_devices,
                    get_device_for_tensor,
                    dtype,
                    silent,
                    predicate,
                    |key| regexes.iter().any(|r| r.is_match(key)),
                )
            })));
        } else {
            let predicate = predicate.clone();
            handles.push(thread::spawn(Box::new(move || {
                let loader = XLora::new(i + 1);
                loader.load_tensors_from_path(
                    &path,
                    &base_device,
                    layer_devices,
                    get_device_for_tensor,
                    dtype,
                    silent,
                    predicate,
                    |_| false,
                )
            })));
        }
    }

    let mut ws = HashMap::new();
    // Wait until all spawned threads have finished loading tensors:
    while !handles.iter().all(|h| h.is_finished()) {}
    for h in handles {
        ws.extend(h.join().unwrap()?);
    }

    let backend = Box::new(ws);

    // TODO(EricLBuehler): separation of concerns.
    // This is to have WNA16 for GPTQ which is required. No bf16 for GPTQ
    Ok(ShardedSafeTensors::wrap(
        backend,
        dtype.unwrap_or(DType::F16),
        base_device.clone(),
    ))
}

pub(crate) fn load_preload_adapters(
    paths: &Option<HashMap<String, (PathBuf, LoraConfig)>>,
    dtype: DType,
    device: &Device,
    silent: bool,
) -> Result<Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>> {
    if let Some(paths) = paths {
        let mut map = HashMap::new();
        for (name, (path, config)) in paths {
            let loader = Common::new();
            let loaded_tensors = loader.load_tensors_from_path(
                path,
                device,
                vec![None],
                Arc::new(|_| DeviceForLoadTensor::Base),
                Some(dtype),
                silent,
                |_| true,
                |_| false,
            )?;

            let backend = Box::new(loaded_tensors);

            // TODO(EricLBuehler): separation of concerns.
            // This is to have WNA16 for GPTQ which is required. No bf16 for GPTQ
            let vb = ShardedSafeTensors::wrap(backend, dtype, device.clone());

            map.insert(name.clone(), (vb, config.clone()));
        }
        Ok(Some(map))
    } else {
        Ok(None)
    }
}

// Presently this logic only needs to diverge for X-LoRA support via `get_name_key_pairs()`
trait LoadTensors {
    #[allow(clippy::too_many_arguments)]
    fn load_tensors_from_path(
        &self,
        path: &PathBuf,
        base_device: &Device,
        layer_devices: Vec<Option<Device>>,
        get_device_for_tensor: Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>,
        dtype: Option<DType>,
        is_silent: bool,
        predicate: impl Fn(String) -> bool,
        make_dummy_predicate: impl Fn(&str) -> bool,
    ) -> Result<HashMap<String, Tensor>> {
        let tensors: Box<dyn TensorLoaderBackend> = match path
            .extension()
            .expect("Expected extension")
            .to_str()
            .expect("Expected to convert")
        {
            "safetensors" => Box::new(SafetensorBackend(unsafe {
                MmapedSafetensors::new(path)?
            })),
            "pth" | "pt" | "bin" => Box::new(PickleBackend(
                candle_core::pickle::PthTensors::new(path, None)?
            )),
            other => candle_core::bail!("Unexpected extension `{other}`, this should have been handled by `get_model_paths`."),
        };

        // Extracts the tensor name and processes it, filtering tensors and deriving the key name:
        let names_only = tensors
            .get_names()
            .into_iter()
            .filter(|x| predicate(x.to_string()));
        let iter = self.get_name_key_pairs(names_only).collect::<Vec<_>>();

        // Take the filtered list of tensors to load, store with derived lookup key:
        let mut loaded_tensors = HashMap::new();
        if !iter.is_empty() {
            for (load_name, key_name) in iter.into_iter().with_progress(is_silent) {
                if !make_dummy_predicate(&load_name) {
                    let dev = match get_device_for_tensor(load_name.clone()) {
                        DeviceForLoadTensor::Base => base_device,
                        DeviceForLoadTensor::Idx(i) => layer_devices
                            .get(i)
                            .and_then(|d| d.as_ref())
                            .unwrap_or(base_device),
                    };
                    // If making a dummy, don't add the tensor. `mistralrs_quant` handles this!
                    let tensor = tensors.load_name(&load_name, dev, dtype)?;

                    loaded_tensors.insert(key_name, tensor);
                }
            }
        }

        Ok(loaded_tensors)
    }

    fn get_name_key_pairs(
        &self,
        tensors: impl Iterator<Item = String>,
    ) -> impl Iterator<Item = (String, String)> {
        tensors.map(|name| {
            let new_name = name.replace("base_model.model.model", "model");

            (name, new_name)
        })
    }
}

#[derive(new)]
struct Common {}
impl LoadTensors for Common {}

#[derive(new)]
struct XLora {
    // Matches the associated path instance for reference in `get_name_key_pairs()`
    adapter_index: usize,
}

impl LoadTensors for XLora {
    fn get_name_key_pairs(
        &self,
        tensors: impl Iterator<Item = String>,
    ) -> impl Iterator<Item = (String, String)> {
        let expectation = "tensor name `{new_name}` should have substring `.lora`";

        tensors
            .filter(|name| !name.contains("internal_xlora_classifier"))
            .map(|name| {
                let mut new_name = name.replace("base_model.model.model", "model");
                // TODO: Add better context to describe intent / requirement:
                let pos = new_name.find(".lora").expect(expectation);
                new_name.insert_str(pos + 7, &format!(".{}", self.adapter_index));

                (name, new_name)
            })
    }
}
