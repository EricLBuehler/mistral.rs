use std::fmt::Debug;

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use tracing::info;

use crate::Pipeline;

#[derive(Debug, Default, Deserialize)]
pub struct DeviceMapMetadata {
    device_layers: usize,
    host_layers: Option<usize>,
}

pub trait DeviceMapper: Debug {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor>;
    fn set_device<'a>(&self, layer: usize, varbuilder: VarBuilder<'a>) -> VarBuilder<'a>;
    fn is_dummy(&self) -> bool;
}

pub fn new_dummy_mapper() -> Box<dyn DeviceMapper + Send + Sync> {
    Box::new(DummyDeviceMapper)
}

#[derive(Debug)]
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
    fn set_device<'a>(&self, layer: usize, varbuilder: VarBuilder<'a>) -> VarBuilder<'a> {
        varbuilder.set_device(self.mappings[layer].clone())
    }
    fn is_dummy(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct DummyDeviceMapper;

impl DeviceMapper for DummyDeviceMapper {
    fn map(&self, input: Tensor, _: usize) -> Result<Tensor> {
        Ok(input)
    }
    fn set_device<'a>(&self, _: usize, varbuilder: VarBuilder<'a>) -> VarBuilder<'a> {
        varbuilder
    }
    fn is_dummy(&self) -> bool {
        true
    }
}
