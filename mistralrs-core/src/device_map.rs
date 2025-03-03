use std::{fmt::Debug, sync::Arc};

use crate::{
    pipeline::AutoDeviceMapParams,
    utils::{debug::DeviceRepr, log::once_log_info},
    MemoryUsage, Topology, TryIntoDType,
};
use candle_core::{DType, Device, DeviceLocation, Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;
use serde::Deserialize;
use tracing::info;

fn split_range(range: std::ops::Range<usize>, n: usize) -> Vec<std::ops::Range<usize>> {
    assert!(n > 0, "n must be non-zero");

    let total = range.end - range.start;
    let chunk_size = total / n;
    let remainder = total % n;

    let mut chunks = Vec::with_capacity(n);
    let mut start = range.start;

    // Create each chunk. The first `remainder` chunks get an extra element.
    for i in 0..n {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + chunk_size + extra;
        chunks.push(start..end);
        start = end;
    }

    chunks
}

#[derive(Debug, Default, Deserialize, Clone)]
pub struct DeviceLayerMapMetadata {
    pub ordinal: usize,
    pub layers: usize,
}

#[derive(Debug, Clone)]
pub enum DeviceMapSetting {
    /// Manual device mapping.
    Map(DeviceMapMetadata),
    /// Automatic device mapping (recommended).
    Auto(AutoDeviceMapParams),
    /// Dummy device mapping for a NCCL pipeline
    Nccl { devices: Vec<Device> },
    /// Device mapping when using PP (agnostic of TP)
    NcclPipelineParallel {
        devices_and_comms: Vec<(Arc<mistralrs_quant::Comm>, Device)>,
        nm_device: Device,
    },
}

#[derive(Debug, Default, Deserialize, Clone)]
/// Metadata to initialize the device mapper.
pub struct DeviceMapMetadata {
    device_layers: Option<Vec<DeviceLayerMapMetadata>>,
    host_layers: Option<usize>,
}

impl DeviceMapMetadata {
    pub fn from_num_device_layers(device_layers: Vec<DeviceLayerMapMetadata>) -> Self {
        Self {
            device_layers: Some(device_layers),
            host_layers: None,
        }
    }
    /// A device mapper to not map device.
    pub fn dummy() -> Self {
        Self {
            device_layers: None,
            host_layers: None,
        }
    }
}

impl DeviceMapSetting {
    /// A device mapper to not map device.
    pub fn dummy() -> Self {
        Self::Map(DeviceMapMetadata::dummy())
    }
    pub fn into_mapper(
        &self,
        model_layers: usize,
        device: &Device,
        topology: Option<&Topology>,
    ) -> Result<Box<dyn DeviceMapper + Send + Sync>> {
        match self {
            Self::Nccl { devices } => {
                once_log_info("Loading model using a NCCL-parallelized pipeline.");
                Ok(Box::new(NcclDeviceMapper {
                    nm_device: devices[0].clone(),
                    devices: devices.clone(),
                }))
            }
            Self::NcclPipelineParallel {
                devices_and_comms,
                nm_device,
            } => {
                let splits = split_range(0..model_layers, devices_and_comms.len());

                let mut mappings = Vec::new();
                for (split, device) in splits.into_iter().zip(devices_and_comms) {
                    mappings.extend(vec![device.clone(); split.len()]);
                }

                Ok(Box::new(NcclPipelineParallelMapper {
                    mappings,
                    nm_device: nm_device.clone(),
                }))
            }

            Self::Map(DeviceMapMetadata {
                device_layers,
                host_layers,
            }) => {
                if let Some(topology) = topology {
                    if topology.0.iter().all(|x| x.is_none()) {
                        return Ok(Box::new(DummyDeviceMapper {
                            nm_device: device.clone(),
                        }));
                    } else {
                        let layers = topology
                            .0
                            .iter()
                            .map(|layer| {
                                layer
                                    .as_ref()
                                    .map(|x| x.device.clone().unwrap_or(device.clone()))
                                    .unwrap_or(device.clone())
                            })
                            .collect::<Vec<_>>();

                        info!("Loading model according to the following repeating layer mappings based on topology:");
                        for (i, dev) in layers.iter().enumerate() {
                            info!("Layer {i}: {}", dev.device_pretty_repr());
                        }

                        return Ok(Box::new(LayerDeviceMapper {
                            mappings: layers,
                            nm_device: device.clone(),
                        }));
                    }
                }

                // How many device layers
                // Clamp to max of model layers
                let n_device_layers = if let Some(layers) = &device_layers {
                    layers
                        .iter()
                        .map(|metadata| metadata.layers)
                        .sum::<usize>()
                        .clamp(0, model_layers)
                } else {
                    return Ok(Box::new(DummyDeviceMapper {
                        nm_device: device.clone(),
                    }));
                };
                // How many host (cpu) layers, defaulting to automatically filling the rest.
                // If n_device_layers > model_layers, n_host_layers = 0
                let n_host_layers =
                    host_layers.unwrap_or(model_layers.saturating_sub(n_device_layers));
                if n_device_layers + n_host_layers != model_layers {
                    candle_core::bail!("Expected the total number of GPU ({n_device_layers}) and host layers ({n_host_layers}) to sum to the number of model hidden layers ({model_layers})");
                }
                once_log_info(format!("Model has {model_layers} repeating layers."));

                // Handle multi-GPU mapping here
                let mut combined = Vec::with_capacity(model_layers);
                if device_layers
                    .as_ref()
                    .is_some_and(|layers| layers.len() == 1)
                {
                    combined.extend(vec![device.clone(); n_device_layers]);
                } else {
                    let original_seed = if !device.is_cpu() {
                        Some(device.get_current_seed()?)
                    } else {
                        None
                    };
                    for DeviceLayerMapMetadata { ordinal, layers } in
                        device_layers.as_ref().unwrap()
                    {
                        let dev = match device {
                            Device::Cpu => Device::Cpu,
                            Device::Cuda(_) => Device::cuda_if_available(*ordinal)?,
                            Device::Metal(_) => Device::new_metal(*ordinal)?,
                        };
                        if !device.is_cpu() {
                            dev.set_seed(original_seed.unwrap())?;
                        }
                        combined.extend(vec![dev; *layers]);
                    }
                }

                // Always put the CPU layers at the end so that we reduce dtoh and htod copies
                combined.extend(vec![Device::Cpu; n_host_layers]);

                // Sanity
                assert_eq!(combined.len(), model_layers);

                // Print it out
                {
                    once_log_info(
                        "Loading model according to the following repeating layer mappings:",
                    );
                    let mut start_index = 0;
                    let mut current_dev = &combined[0];

                    // Iterate starting from index 1 to detect when the variant changes
                    for (i, variant) in combined.iter().enumerate().skip(1) {
                        // If the variant changes, print the previous continuous block
                        if !variant.same_device(current_dev) {
                            once_log_info(format!(
                                "Layers {}-{}: {} ({} GB)",
                                start_index,
                                i - 1,
                                current_dev.device_pretty_repr(),
                                MemoryUsage
                                    .get_total_memory(current_dev)?
                                    .div_ceil(1024 * 1024 * 1024),
                            ));
                            start_index = i; // start a new range
                            current_dev = variant;
                        }
                    }

                    once_log_info(format!(
                        "Layers {}-{}: {} ({} GB)",
                        start_index,
                        combined.len() - 1,
                        current_dev.device_pretty_repr(),
                        MemoryUsage
                            .get_total_memory(current_dev)?
                            .div_ceil(1024 * 1024 * 1024),
                    ));
                }

                Ok(Box::new(LayerDeviceMapper {
                    mappings: combined,
                    nm_device: device.clone(),
                }))
            }
            Self::Auto(_) => {
                candle_core::bail!(".into_mapper does not work on Auto device map, convert it to a Map with DeviceMappedModelLoader::get_device_layers")
            }
        }
    }
}

pub trait DeviceMapper: Debug {
    // === DURING RUNTIME ===
    /// Map during runtime
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor>;

    // === DURING LOADING TIME ===
    /// If ISQ layer, then do not change the device. *They will do it later in NormalModel::quantize*
    fn set_device<'a>(
        &self,
        layer: usize,
        varbuilder: ShardedVarBuilder<'a>,
        loading_isq: bool,
    ) -> ShardedVarBuilder<'a>;
    /// If ISQ layer, then do not change the device (return None). *They will do it later in NormalModel::quantize*
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device>;
    fn get_unique_devices(&self) -> Vec<Device>;
    /// If ISQ layer, then do not change the device (return None). *They will do it later in NormalModel::quantize*
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor>;
    /// Set non mapped layer device. This is for ISQ + device mapping support
    /// If ISQ layer, then do not change the device. *They will do it later in NormalModel::quantize*
    fn set_nm_device<'a>(
        &self,
        varbuilder: ShardedVarBuilder<'a>,
        loading_isq: bool,
    ) -> ShardedVarBuilder<'a>;
    fn num_device_mapping_layers(&self) -> usize;
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>>;

    // === IMMEDIATELY AFTER INIT ===
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType>;
}

#[derive(Debug)]
/// A device mapper which does device mapping per hidden layer.
pub struct LayerDeviceMapper {
    mappings: Vec<Device>,
    nm_device: Device,
}

impl DeviceMapper for LayerDeviceMapper {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor> {
        input.to_device(&self.mappings[layer])
    }
    fn set_device<'a>(
        &self,
        layer: usize,
        varbuilder: ShardedVarBuilder<'a>,
        loading_isq: bool,
    ) -> ShardedVarBuilder<'a> {
        if loading_isq {
            return varbuilder;
        }
        varbuilder.set_device(self.mappings[layer].clone())
    }
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device> {
        if loading_isq {
            return Some(&self.nm_device);
        }
        self.mappings.get(layer)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        self.mappings.iter().fold(Vec::new(), |mut acc, device| {
            if !acc.iter().any(|d| d.same_device(device)) {
                acc.push(device.clone());
            }
            acc
        })
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            x.to_device(&self.nm_device)
        }
    }
    fn set_nm_device<'a>(
        &self,
        varbuilder: ShardedVarBuilder<'a>,
        loading_isq: bool,
    ) -> ShardedVarBuilder<'a> {
        if loading_isq {
            varbuilder
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&self.mappings.iter().collect::<Vec<_>>())
            .map_err(candle_core::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        self.mappings.len()
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>> {
        let id = mistralrs_quant::Id::new();
        Ok(Arc::new(mistralrs_quant::Comm::from_device(
            id,
            self.device_for(layer_idx, false).unwrap_or(&self.nm_device),
            0,
            1,
        )?))
    }
}

#[derive(Debug)]
pub struct DummyDeviceMapper {
    nm_device: Device,
}

impl DeviceMapper for DummyDeviceMapper {
    fn map(&self, input: Tensor, _: usize) -> Result<Tensor> {
        Ok(input)
    }
    fn set_device<'a>(
        &self,
        _: usize,
        varbuilder: ShardedVarBuilder<'a>,
        loading_isq: bool,
    ) -> ShardedVarBuilder<'a> {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn device_for(&self, _: usize, _loading_isq: bool) -> Option<&Device> {
        Some(&self.nm_device)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        vec![self.nm_device.clone()]
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            x.to_device(&self.nm_device)
        }
    }
    fn set_nm_device<'a>(
        &self,
        varbuilder: ShardedVarBuilder<'a>,
        loading_isq: bool,
    ) -> ShardedVarBuilder<'a> {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&[&self.nm_device])
            .map_err(candle_core::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        // Effectively one layer
        1
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>> {
        let id = mistralrs_quant::Id::new();
        Ok(Arc::new(mistralrs_quant::Comm::from_device(
            id,
            self.device_for(layer_idx, false).unwrap_or(&self.nm_device),
            0,
            1,
        )?))
    }
}

#[derive(Debug)]
pub struct NcclDeviceMapper {
    nm_device: Device,
    devices: Vec<Device>,
}

impl DeviceMapper for NcclDeviceMapper {
    fn map(&self, input: Tensor, _: usize) -> Result<Tensor> {
        Ok(input)
    }
    fn set_device<'a>(
        &self,
        _: usize,
        varbuilder: ShardedVarBuilder<'a>,
        loading_isq: bool,
    ) -> ShardedVarBuilder<'a> {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn device_for(&self, _: usize, _loading_isq: bool) -> Option<&Device> {
        Some(&self.nm_device)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        self.devices.clone()
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            x.to_device(&self.nm_device)
        }
    }
    fn set_nm_device<'a>(
        &self,
        varbuilder: ShardedVarBuilder<'a>,
        loading_isq: bool,
    ) -> ShardedVarBuilder<'a> {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&self.devices.iter().collect::<Vec<_>>())
            .map_err(candle_core::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        // Effectively one layer
        1
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>> {
        let id = mistralrs_quant::Id::new();
        Ok(Arc::new(mistralrs_quant::Comm::from_device(
            id,
            self.device_for(layer_idx, false).unwrap_or(&self.nm_device),
            0,
            1,
        )?))
    }
}

#[derive(Debug)]
/// A device mapper which does device mapping per hidden layer.
pub struct NcclPipelineParallelMapper {
    mappings: Vec<(Arc<mistralrs_quant::Comm>, Device)>,
    nm_device: Device,
}

impl DeviceMapper for NcclPipelineParallelMapper {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor> {
        input.to_device(&self.mappings[layer].1)
    }
    fn set_device<'a>(
        &self,
        layer: usize,
        varbuilder: ShardedVarBuilder<'a>,
        loading_isq: bool,
    ) -> ShardedVarBuilder<'a> {
        if loading_isq {
            return varbuilder;
        }
        varbuilder.set_device(self.mappings[layer].1.clone())
    }
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device> {
        if loading_isq {
            return Some(&self.nm_device);
        }
        self.mappings.get(layer).map(|(_, x)| x)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        self.mappings
            .iter()
            .fold(Vec::new(), |mut acc, (_, device)| {
                if !acc.iter().any(|d| d.same_device(device)) {
                    acc.push(device.clone());
                }
                acc
            })
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            x.to_device(&self.nm_device)
        }
    }
    fn set_nm_device<'a>(
        &self,
        varbuilder: ShardedVarBuilder<'a>,
        loading_isq: bool,
    ) -> ShardedVarBuilder<'a> {
        if loading_isq {
            varbuilder
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&self.mappings.iter().map(|(_, x)| x).collect::<Vec<_>>())
            .map_err(candle_core::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        self.mappings.len()
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>> {
        Ok(self.mappings[layer_idx].0.clone())
    }
}

/// Get all devices on the same device type but different ordinals
pub fn get_all_similar_devices(base: &Device) -> Result<Vec<Device>> {
    let mut devices = Vec::new();
    match base {
        Device::Cpu => return Ok(vec![Device::Cpu]),
        Device::Cuda(_) => {
            let mut ord = 0;
            let DeviceLocation::Cuda { gpu_id: base_ord } = base.location() else {
                candle_core::bail!("location and device do not match");
            };
            loop {
                if base_ord == ord {
                    devices.push(base.clone());
                    ord += 1;
                    continue;
                }
                // Needs to be without a stream as PagedAttention doesn't like it otherwise.
                if let Ok(dev) = Device::new_cuda(ord) {
                    devices.push(dev);
                    ord += 1;
                } else {
                    break;
                }
            }
        }
        #[cfg(not(feature = "metal"))]
        Device::Metal(_) => {
            candle_core::bail!("Not compiled with metal features, but have a metal device.");
        }
        #[cfg(feature = "metal")]
        Device::Metal(_) => {
            let total_ords = metal::Device::all().len();
            let mut ord = 0;
            let DeviceLocation::Metal { gpu_id: base_ord } = base.location() else {
                candle_core::bail!("location and device do not match");
            };
            loop {
                if base_ord == ord {
                    devices.push(base.clone());
                    ord += 1;
                    continue;
                }
                if total_ords == ord {
                    break;
                }
                if let Ok(dev) = Device::new_metal(ord) {
                    devices.push(dev);
                    ord += 1;
                } else {
                    break;
                }
            }
        }
    }
    Ok(devices)
}
