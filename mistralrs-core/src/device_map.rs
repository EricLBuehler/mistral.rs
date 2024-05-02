use std::fmt::Debug;

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use tracing::info;

#[derive(Debug, Default, Deserialize, Clone)]
pub struct DeviceMapMetadata {
    device_layers: Option<usize>,
    host_layers: Option<usize>,
}

impl DeviceMapMetadata {
    pub fn from_num_device_layers(device_layers: usize) -> Self {
        Self {
            device_layers: Some(device_layers),
            host_layers: None,
        }
    }
    pub fn dummy() -> Self {
        Self {
            device_layers: None,
            host_layers: None,
        }
    }
    pub fn is_dummy(&self) -> bool {
        self.device_layers.is_none()
    }
    pub fn into_mapper(
        &self,
        model_layers: usize,
        device: &Device,
    ) -> Result<Box<dyn DeviceMapper + Send + Sync>> {
        // How many device layers
        let n_device_layers = if let Some(n) = self.device_layers {
            n
        } else {
            return Ok(Box::new(DummyDeviceMapper {
                nm_device: device.clone(),
            }));
        };
        // How many host (cpu) layers, defaulting to automatically filling the rest.
        let n_host_layers = self.host_layers.unwrap_or(model_layers - n_device_layers);
        if n_device_layers + n_host_layers != model_layers {
            candle_core::bail!("Expected the number of device ({n_device_layers}) and host layers ({n_host_layers}) to sum to the number of model hidden layers ({model_layers})");
        }
        info!("Using {n_device_layers} layers on device and {n_host_layers} on host.");
        let mut combined = vec![device.clone(); n_device_layers];
        // Always put the CPU layers at the end so that we reduce dtoh and htod copies
        combined.extend(vec![Device::Cpu; n_host_layers]);
        Ok(Box::new(LayerDeviceMapper {
            mappings: combined,
            nm_device: device.clone(),
        }))
    }
}

pub trait DeviceMapper: Debug {
    /// Map during runtime
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor>;
    /// If ISQ layer, then do not change the device. *They will do it later in NormalModel::quantize*
    fn set_device<'a>(
        &self,
        layer: usize,
        varbuilder: VarBuilder<'a>,
        loading_isq: bool,
    ) -> VarBuilder<'a>;
    /// If ISQ layer, then do not change the device (return None). *They will do it later in NormalModel::quantize*
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device>;
    /// Set non mapped layer device. This is for ISQ + device mapping support
    /// If ISQ layer, then do not change the device. *They will do it later in NormalModel::quantize*
    fn set_nm_device<'a>(&self, varbuilder: VarBuilder<'a>, loading_isq: bool) -> VarBuilder<'a>;
}

#[derive(Debug)]
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
        varbuilder: VarBuilder<'a>,
        loading_isq: bool,
    ) -> VarBuilder<'a> {
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
    fn set_nm_device<'a>(&self, varbuilder: VarBuilder<'a>, loading_isq: bool) -> VarBuilder<'a> {
        if loading_isq {
            varbuilder
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
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
        varbuilder: VarBuilder<'a>,
        loading_isq: bool,
    ) -> VarBuilder<'a> {
        if loading_isq {
            varbuilder
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn device_for(&self, _: usize, loading_isq: bool) -> Option<&Device> {
        if loading_isq {
            return Some(&self.nm_device);
        }
        None
    }
    fn set_nm_device<'a>(&self, varbuilder: VarBuilder<'a>, loading_isq: bool) -> VarBuilder<'a> {
        if loading_isq {
            varbuilder
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
}
