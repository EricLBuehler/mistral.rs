use std::sync::Arc;
use std::{fmt::Debug, str::FromStr};

use anyhow::Result;
use candle_core::Device;
use candle_nn::VarBuilder;

#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use serde::Deserialize;

use super::{Processor, VisionModel};
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;
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
    fn get_processor(
        &self,
        processor_config: ProcessorConfig,
        preprocessor_config: PreProcessorConfig,
    ) -> Arc<dyn Processor + Send + Sync>;
}

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[derive(Clone, Debug, Deserialize)]
/// The architecture to load the vision model as.
pub enum VisionLoaderType {
    #[serde(rename = "phi3v")]
    Phi3V,
}

impl FromStr for VisionLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "idefics2" => Ok(Self::Phi3V),
            a => Err(format!("Unknown architecture `{a}`")),
        }
    }
}
