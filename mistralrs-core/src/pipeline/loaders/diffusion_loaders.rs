use std::{fmt::Debug, str::FromStr};

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use serde::Deserialize;

use super::NormalLoadingMetadata;
use crate::paged_attention::{AttentionImplementation, ModelConfigMetadata};

pub trait DiffusionModel {
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, input_ids: &Tensor) -> candle_core::Result<Tensor>;
    fn device(&self) -> &Device;
    fn max_seq_len(&self) -> usize;
    fn config(&self) -> &ModelConfigMetadata;
}

pub trait DiffusionModelLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn DiffusionModel + Send + Sync>>;
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>>;
    /// Get total num_hidden_layers for the layers which will be device mapped.
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize>;
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, PartialEq)]
/// The architecture to load the vision model as.
pub enum DiffusionLoaderType {
    #[serde(rename = "flux")]
    Flux,
}

impl FromStr for DiffusionLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "flux" => Ok(Self::Flux),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `flux`."
            )),
        }
    }
}

// ======================== Flux loader

/// [`DiffusionLoader`] for a Flux Diffusion model.
///
/// [`DiffusionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.DiffusionLoader.html
pub struct FluxLoader;

impl DiffusionModelLoader for FluxLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn DiffusionModel + Send + Sync>> {
        todo!()
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        todo!()
    }
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        todo!()
    }
}
