use std::{fmt::Debug, str::FromStr};

use anyhow::Result;
use candle_core::Device;
use candle_nn::VarBuilder;

#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use serde::Deserialize;

use super::VisionModel;
use crate::DeviceMapMetadata;

pub trait VisionModelLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        mapper: DeviceMapMetadata,
        loading_isq: bool,
        device: Device,
    ) -> Result<Box<dyn VisionModel + Send + Sync>>;
    fn is_gptx(&self) -> bool;
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>>;
}

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[derive(Clone, Debug, Deserialize)]
/// The architecture to load the vision model as.
pub enum VisionLoaderType {
    #[serde(rename = "idefics2")]
    Idefics2,
}

impl FromStr for VisionLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "idefics2" => Ok(Self::Idefics2),
            a => Err(format!("Unknown architecture `{a}`")),
        }
    }
}

// ======================== Idefics 2 loader

pub struct Idefics2Loader;

impl VisionModelLoader for Idefics2Loader {
    fn load(
        &self,
        _config: &str,
        _use_flash_attn: bool,
        _vb: VarBuilder,
        _mapper: DeviceMapMetadata,
        _loading_isq: bool,
        _device: Device,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, _config: &str, _use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        todo!()
    }
}
