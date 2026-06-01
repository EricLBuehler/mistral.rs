mod mappers;
mod mask;
mod peer;

#[allow(unused_imports)]
pub use mappers::NcclPipelineParallelMapper;
pub use mappers::{DeviceMapper, DummyDeviceMapper, LayerDeviceMapper, NcclDeviceMapper};
pub use mask::DeviceMappedMask;

use std::sync::Arc;

use crate::{pipeline::AutoDeviceMapParams, utils::debug::DeviceRepr, MemoryUsage, Topology};
use candle_core::{Device, DeviceLocation, Result};
use mistralrs_quant::log::once_log_info;
use serde::{Deserialize, Serialize};
use tracing::info;

use peer::CudaPeerAccess;

type MapperBox = Box<dyn DeviceMapper + Send + Sync>;

#[derive(Debug, Default, Deserialize, Serialize, Clone)]
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
    DummyNccl { nm_device: Device },
    /// Real device mapping for a NCCL pipeline
    Nccl {
        nm_device: Device,
        comm: Arc<mistralrs_quant::Comm>,
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

    pub fn device_layers(&self) -> Option<&[DeviceLayerMapMetadata]> {
        self.device_layers.as_deref()
    }

    pub fn host_layers(&self) -> Option<usize> {
        self.host_layers
    }

    pub fn to_cli_spec(&self) -> Option<String> {
        let layers = self.device_layers.as_ref()?;
        if layers.is_empty() {
            return None;
        }
        Some(
            layers
                .iter()
                .map(|l| format!("{}:{}", l.ordinal, l.layers))
                .collect::<Vec<_>>()
                .join(";"),
        )
    }

    fn build_mapper(
        &self,
        model_layers: usize,
        device: &Device,
        topology: Option<&Topology>,
        all_devices: &[Device],
    ) -> Result<MapperBox> {
        if let Some(topology) = topology {
            return self.topology_mapper(topology, device);
        }

        let Some(device_layers) = &self.device_layers else {
            return Ok(dummy_mapper(device));
        };

        let n_device_layers = device_layers
            .iter()
            .map(|metadata| metadata.layers)
            .sum::<usize>()
            .clamp(0, model_layers);
        let n_host_layers = self
            .host_layers
            .unwrap_or(model_layers.saturating_sub(n_device_layers));
        if n_device_layers + n_host_layers != model_layers {
            candle_core::bail!("Expected the total number of GPU ({n_device_layers}) and host layers ({n_host_layers}) to sum to the number of model hidden layers ({model_layers})");
        }
        once_log_info(format!("Model has {model_layers} repeating layers."));

        let mappings = self.manual_mappings(
            model_layers,
            device,
            all_devices,
            device_layers,
            n_device_layers,
            n_host_layers,
        )?;
        log_layer_mappings(&mappings)?;
        layer_mapper(mappings, device)
    }

    fn topology_mapper(&self, topology: &Topology, device: &Device) -> Result<MapperBox> {
        if topology.layers.iter().all(|x| x.is_none()) {
            return Ok(dummy_mapper(device));
        }

        let mappings = topology
            .layers
            .iter()
            .map(|layer| {
                layer
                    .as_ref()
                    .map(|x| x.device.clone().unwrap_or(device.clone()))
                    .unwrap_or(device.clone())
            })
            .collect::<Vec<_>>();

        info!(
            "Loading model according to the following repeating layer mappings based on topology:"
        );
        for (i, dev) in mappings.iter().enumerate() {
            info!("Layer {i}: {}", dev.device_pretty_repr());
        }

        layer_mapper(mappings, device)
    }

    fn manual_mappings(
        &self,
        model_layers: usize,
        device: &Device,
        all_devices: &[Device],
        device_layers: &[DeviceLayerMapMetadata],
        n_device_layers: usize,
        n_host_layers: usize,
    ) -> Result<Vec<Device>> {
        let mut mappings = Vec::with_capacity(model_layers);

        if device_layers.len() == 1 {
            mappings.extend(vec![device.clone(); n_device_layers]);
        } else {
            let original_seed = if !device.is_cpu() {
                Some(device.get_current_seed()?)
            } else {
                None
            };

            for DeviceLayerMapMetadata { ordinal, layers } in device_layers {
                let dev = mapped_device_for_ordinal(device, all_devices, *ordinal)?;
                if let Some(seed) = original_seed {
                    dev.set_seed(seed)?;
                }
                mappings.extend(vec![dev; *layers]);
            }
        }

        mappings.extend(vec![Device::Cpu; n_host_layers]);
        assert_eq!(mappings.len(), model_layers);
        Ok(mappings)
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
        all_devices: &[Device],
    ) -> Result<Box<dyn DeviceMapper + Send + Sync>> {
        match self {
            Self::Nccl { nm_device, comm } => {
                once_log_info("Loading model using a NCCL-parallelized pipeline.");
                Ok(Box::new(NcclDeviceMapper::new(
                    nm_device.clone(),
                    model_layers,
                    Some(comm.clone()),
                )))
            }

            Self::DummyNccl { nm_device } => {
                once_log_info("Loading model using a NCCL-parallelized pipeline.");
                Ok(Box::new(NcclDeviceMapper::new(
                    nm_device.clone(),
                    model_layers,
                    None,
                )))
            }

            Self::Map(metadata) => {
                metadata.build_mapper(model_layers, device, topology, all_devices)
            }
            Self::Auto(_) => {
                candle_core::bail!(".into_mapper does not work on Auto device map, convert it to a Map with DeviceMappedModelLoader::get_device_layers")
            }
        }
    }
}

fn dummy_mapper(device: &Device) -> MapperBox {
    Box::new(DummyDeviceMapper::new(device.clone()))
}

fn layer_mapper(mappings: Vec<Device>, nm_device: &Device) -> Result<MapperBox> {
    let mut peer_devices = mappings.clone();
    peer_devices.push(nm_device.clone());
    let cuda_peer_access = CudaPeerAccess::new(&peer_devices)?;
    Ok(Box::new(LayerDeviceMapper::new(
        mappings,
        nm_device.clone(),
        cuda_peer_access,
    )))
}

fn mapped_device_for_ordinal(
    device: &Device,
    all_devices: &[Device],
    ordinal: usize,
) -> Result<Device> {
    match device.location() {
        DeviceLocation::Cpu => Ok(Device::Cpu),
        DeviceLocation::Cuda { gpu_id } if gpu_id == ordinal => Ok(device.clone()),
        DeviceLocation::Cuda { .. } => all_devices
            .iter()
            .find(|d| d.is_cuda() && device_ordinal(d) == ordinal)
            .cloned()
            .ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "Could not find cuda device with ordinal {ordinal}"
                ))
            }),
        DeviceLocation::Metal { gpu_id } if gpu_id == ordinal => Ok(device.clone()),
        DeviceLocation::Metal { .. } => Device::new_metal(ordinal),
    }
}

fn device_ordinal(device: &Device) -> usize {
    match device.location() {
        DeviceLocation::Cpu => 0,
        DeviceLocation::Cuda { gpu_id } => gpu_id,
        DeviceLocation::Metal { gpu_id } => gpu_id,
    }
}

fn log_layer_mappings(mappings: &[Device]) -> Result<()> {
    if mappings.is_empty() {
        return Ok(());
    }

    once_log_info("Loading model according to the following repeating layer mappings:");
    let mut start_index = 0;
    let mut current_dev = &mappings[0];

    for (i, variant) in mappings.iter().enumerate().skip(1) {
        if !variant.same_device(current_dev) {
            log_layer_mapping_range(start_index, i - 1, current_dev)?;
            start_index = i;
            current_dev = variant;
        }
    }

    log_layer_mapping_range(start_index, mappings.len() - 1, current_dev)
}

fn log_layer_mapping_range(start_index: usize, end_index: usize, device: &Device) -> Result<()> {
    once_log_info(format!(
        "Layers {}-{}: {} ({} GB)",
        start_index,
        end_index,
        device.device_pretty_repr(),
        MemoryUsage
            .query(device)?
            .total()
            .div_ceil(1024 * 1024 * 1024),
    ));
    Ok(())
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
                let dev = Device::new_cuda(ord);
                if let Ok(dev) = dev {
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
            #[cfg(feature = "metal")]
            let total_ords = candle_metal_kernels::metal::Device::all().len();
            #[cfg(not(feature = "metal"))]
            let total_ords = 0;
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
