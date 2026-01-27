use std::{
    fmt::Debug,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

use hf_hub::api::sync::ApiRepo;
use mistralrs_quant::ShardedVarBuilder;
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;
use serde::Deserialize;

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
    paged_attention::AttentionImplementation,
    pipeline::{paths::AdapterPaths, EmbeddingModulePaths},
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

pub trait DiffusionModelLoader: Send + Sync {
    /// If the model is being loaded with `load_model_from_hf` (so manual paths not provided), this will be called.
    fn get_model_paths(&self, api: &ApiRepo, model_id: &Path) -> Result<Vec<PathBuf>>;
    /// If the model is being loaded with `load_model_from_hf` (so manual paths not provided), this will be called.
    fn get_config_filenames(&self, api: &ApiRepo, model_id: &Path) -> Result<Vec<PathBuf>>;
    fn force_cpu_vb(&self) -> Vec<bool>;
    // `configs` and `vbs` should be corresponding. It is up to the implementer to maintain this invaraint.
    fn load(
        &self,
        configs: Vec<String>,
        vbs: Vec<ShardedVarBuilder>,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
        silent: bool,
    ) -> Result<Box<dyn DiffusionModel + Send + Sync>>;
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq)]
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

impl DiffusionLoaderType {
    /// Auto-detect diffusion loader type from a repo file listing.
    /// Extend this when adding new diffusion pipelines.
    pub fn auto_detect_from_files(files: &[String]) -> Option<Self> {
        if Self::matches_flux(files) {
            return Some(Self::Flux);
        }
        None
    }

    fn matches_flux(files: &[String]) -> bool {
        let flux_regex = Regex::new(r"^flux\\d+-(schnell|dev)\\.safetensors$");
        let Ok(flux_regex) = flux_regex else {
            return false;
        };
        let has_transformer = files.iter().any(|f| f == "transformer/config.json");
        let has_vae = files.iter().any(|f| f == "vae/config.json");
        let has_ae = files.iter().any(|f| f == "ae.safetensors");
        let has_flux = files.iter().any(|f| {
            let name = f.rsplit('/').next().unwrap_or(f);
            flux_regex.is_match(name)
        });

        has_transformer && has_vae && has_ae && has_flux
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
    fn get_template_filename(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_chat_template_explicit(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_adapter_paths(&self) -> &AdapterPaths {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]> {
        unreachable!("Use `std::any::Any`.")
    }
}

// ======================== Flux loader

/// [`DiffusionLoader`] for a Flux Diffusion model.
///
/// [`DiffusionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.DiffusionLoader.html
pub struct FluxLoader {
    pub(crate) offload: bool,
}

impl DiffusionModelLoader for FluxLoader {
    fn get_model_paths(&self, api: &ApiRepo, model_id: &Path) -> Result<Vec<PathBuf>> {
        let regex = Regex::new(r"^flux\d+-(schnell|dev)\.safetensors$")?;
        let flux_name = api_dir_list!(api, model_id, true)
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
        mut vbs: Vec<ShardedVarBuilder>,
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
