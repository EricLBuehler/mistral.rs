use std::{
    collections::HashMap,
    fmt::Debug,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use hf_hub::api::sync::ApiRepo;
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;
use serde::Deserialize;

use tracing::info;

use super::{ModelPaths, NormalLoadingMetadata};
use crate::{
    api_dir_list, api_get_file,
    diffusion_models::{
        flux::{
            self,
            stepper::{FluxStepper, FluxStepperConfig},
        },
        DiffusionGenerationParams,
    },
    lora::LoraConfig,
    paged_attention::AttentionImplementation,
    xlora_models::XLoraConfig,
    Ordering,
};

pub trait DiffusionModel {
    /// This returns a tensor of shape (bs, c, h, w), with values in [0, 255].
    fn forward(
        &mut self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> candle_core::Result<Tensor>;
    fn device(&self) -> &Device;
    fn max_seq_len(&self) -> usize;
}

pub trait DiffusionModelLoader {
    /// If the model is being loaded with `load_model_from_hf` (so manual paths not provided), this will be called.
    fn get_model_paths(&self, api: &ApiRepo, model_id: &Path) -> Result<Vec<PathBuf>>;
    /// If the model is being loaded with `load_model_from_hf` (so manual paths not provided), this will be called.
    fn get_config_filenames(&self, api: &ApiRepo, model_id: &Path) -> Result<Vec<PathBuf>>;
    fn force_cpu_vb(&self) -> Vec<bool>;
    // `configs` and `vbs` should be corresponding. It is up to the implementer to maintain this invaraint.
    fn load(
        &self,
        configs: Vec<String>,
        use_flash_attn: bool,
        vbs: Vec<VarBuilder>,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
        silent: bool,
    ) -> Result<Box<dyn DiffusionModel + Send + Sync>>;
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, PartialEq)]
/// The architecture to load the vision model as.
pub enum DiffusionLoaderType {
    #[serde(rename = "flux")]
    Flux,
    #[serde(rename = "flux-offloaded")]
    FluxOffloaded,
}

impl FromStr for DiffusionLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "flux" => Ok(Self::Flux),
            "flux-offloaded" => Ok(Self::FluxOffloaded),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `flux`."
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DiffusionModelPathsInner {
    pub config_filenames: Vec<PathBuf>,
    pub filenames: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct DiffusionModelPaths(pub DiffusionModelPathsInner);

impl ModelPaths for DiffusionModelPaths {
    fn get_config_filename(&self) -> &PathBuf {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_adapter_filenames(&self) -> &Option<Vec<(String, PathBuf)>> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_adapter_configs(&self) -> &Option<Vec<((String, String), LoraConfig)>> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_classifier_config(&self) -> &Option<XLoraConfig> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_classifier_path(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_ordering(&self) -> &Option<Ordering> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_template_filename(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_lora_preload_adapter_info(&self) -> &Option<HashMap<String, (PathBuf, LoraConfig)>> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
}

// ======================== Flux loader

/// [`DiffusionLoader`] for a Flux Diffusion model.
///
/// [`DiffusionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.DiffusionLoader.html
pub struct FluxLoader {
    pub(crate) offload: bool,
}

impl DiffusionModelLoader for FluxLoader {
    fn get_model_paths(&self, api: &ApiRepo, model_id: &Path) -> Result<Vec<PathBuf>> {
        let regex = Regex::new(r"^flux\d+-(schnell|dev)\.safetensors$")?;
        let flux_name = api_dir_list!(api, model_id)
            .filter(|x| regex.is_match(x))
            .nth(0)
            .with_context(|| "Expected at least 1 .safetensors file matching the FLUX regex, please raise an issue.")?;
        let flux_file = api_get_file!(api, &flux_name, model_id);
        let ae_file = api_get_file!(api, "ae.safetensors", model_id);

        // NOTE(EricLBuehler): disgusting way of doing this but the 0th path is the flux, 1 is ae
        Ok(vec![flux_file, ae_file])
    }
    fn get_config_filenames(&self, api: &ApiRepo, model_id: &Path) -> Result<Vec<PathBuf>> {
        let flux_file = api_get_file!(api, "transformer/config.json", model_id);
        let ae_file = api_get_file!(api, "vae/config.json", model_id);

        // NOTE(EricLBuehler): disgusting way of doing this but the 0th path is the flux, 1 is ae
        Ok(vec![flux_file, ae_file])
    }
    fn force_cpu_vb(&self) -> Vec<bool> {
        vec![self.offload, false]
    }
    fn load(
        &self,
        mut configs: Vec<String>,
        _use_flash_attn: bool,
        mut vbs: Vec<VarBuilder>,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
        silent: bool,
    ) -> Result<Box<dyn DiffusionModel + Send + Sync>> {
        let (vae_cfg, vae_vb) = (configs.remove(1), vbs.remove(1));
        let (flux_cfg, flux_vb) = (configs.remove(0), vbs.remove(0));

        let vae_cfg: flux::autoencoder::Config = serde_json::from_str(&vae_cfg)?;
        let flux_cfg: flux::model::Config = serde_json::from_str(&flux_cfg)?;

        let flux_dtype = flux_vb.dtype();
        if flux_dtype != vae_vb.dtype() {
            anyhow::bail!(
                "Expected VAE and FLUX model VBs to be the same dtype, got {:?} and {flux_dtype:?}",
                vae_vb.dtype()
            );
        }

        Ok(Box::new(FluxStepper::new(
            FluxStepperConfig::default_for_guidance(flux_cfg.guidance_embeds),
            (flux_vb, &flux_cfg),
            (vae_vb, &vae_cfg),
            flux_dtype,
            &normal_loading_metadata.real_device,
            silent,
            self.offload,
        )?))
    }
}
