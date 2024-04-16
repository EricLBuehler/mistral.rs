use candle_core::{Device, Result, Tensor};
use serde::Deserialize;
use tracing::info;

use crate::Pipeline;

#[derive(Debug, Deserialize)]
pub struct DeviceMapMetadata {
    device_layers: usize,
    host_layers: Option<usize>,
}

impl Default for DeviceMapMetadata {
    fn default() -> Self {
        Self {
            device_layers: Default::default(),
            host_layers: Default::default(),
        }
    }
}

pub trait DeviceMapper {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor>;
}

pub struct LayerDeviceMapper {
    mappings: Vec<Device>,
}

impl LayerDeviceMapper {
    pub fn from_metadata(metadata: DeviceMapMetadata, pipeline: &dyn Pipeline) -> Result<Self> {
        let model_layers = pipeline.num_hidden_layers();
        // How many device layers, defaulting to the num model layers
        let n_device_layers = metadata.device_layers;
        // How many host (cpu) layers, defaulting to automatically filling the rest.
        let n_host_layers = metadata
            .host_layers
            .unwrap_or(model_layers - n_device_layers);
        if n_device_layers + n_host_layers != model_layers {
            candle_core::bail!("Expected the number of device ({n_device_layers}) and host layers ({n_host_layers}) to sum to the number of model hidden layers ({model_layers})");
        }
        info!("Using {n_device_layers} layers on device and {n_host_layers} on host.");
        let mut combined = vec![pipeline.device().clone(); n_device_layers];
        // Always put the CPU layers at the end so that we reduce dtoh and htod copies
        combined.extend(vec![Device::Cpu; n_host_layers]);
        Ok(Self { mappings: combined })
    }
}

impl DeviceMapper for LayerDeviceMapper {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor> {
        input.to_device(&self.mappings[layer])
    }
}

pub struct DummyDeviceMapper;

impl DeviceMapper for DummyDeviceMapper {
    fn map(&self, input: Tensor, _: usize) -> Result<Tensor> {
        Ok(input)
    }
}
