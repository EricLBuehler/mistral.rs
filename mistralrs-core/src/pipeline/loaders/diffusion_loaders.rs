use std::{
    fmt::Debug,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::pipeline::hf::HubClient;
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
            stepper::{FluxAuxiliaryPaths, FluxStepper, FluxStepperConfig, FluxStepperLoadContext},
        },
        DiffusionGenerationParams,
    },
    paged_attention::AttentionImplementation,
    pipeline::{paths::AdapterPaths, EmbeddingModulePaths},
    utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor},
};

const T5_XXL_SAFETENSOR_FILES: &[&str] =
    &["t5_xxl-shard-0.safetensors", "t5_xxl-shard-1.safetensors"];

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
    fn download_model_paths(
        &self,
        api: &HubClient,
        model_id: &str,
        revision: &str,
    ) -> Result<Box<dyn ModelPaths>>;
    fn load(
        &self,
        paths: &dyn ModelPaths,
        dtype: DType,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
        silent: bool,
    ) -> Result<Box<dyn DiffusionModel + Send + Sync>>;
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq, strum::EnumIter)]
/// The architecture to load the diffusion model as.
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
struct FluxModelPaths {
    flux_config: PathBuf,
    vae_config: PathBuf,
    flux_weights: PathBuf,
    vae_weights: PathBuf,
    auxiliary_paths: FluxAuxiliaryPaths,
}

impl ModelPaths for FluxModelPaths {
    fn files(&self) -> Vec<PathBuf> {
        let mut files = vec![
            self.flux_config.clone(),
            self.vae_config.clone(),
            self.flux_weights.clone(),
            self.vae_weights.clone(),
        ];
        files.extend(self.auxiliary_paths.files());
        files
    }

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

impl FluxLoader {
    fn download_auxiliary_files(api: &HubClient) -> Result<FluxAuxiliaryPaths> {
        let tokenizer = api.repo("EricB/t5_tokenizer", "main");
        let t5 = api.repo("EricB/t5-v1_1-xxl-enc-only", "main");
        let clip = api.repo("openai/clip-vit-large-patch14", "main");
        Ok(FluxAuxiliaryPaths {
            t5_tokenizer: tokenizer.get("t5-v1_1-xxl.tokenizer.json")?,
            t5_config: t5.get("config.json")?,
            t5_weights: T5_XXL_SAFETENSOR_FILES
                .iter()
                .map(|file| t5.get(file))
                .collect::<std::result::Result<_, _>>()?,
            clip_tokenizer: clip.get("tokenizer.json")?,
            clip_config: clip.get("config.json")?,
            clip_weights: clip.get("model.safetensors")?,
        })
    }
}

impl DiffusionModelLoader for FluxLoader {
    fn download_model_paths(
        &self,
        api: &HubClient,
        model_id: &str,
        revision: &str,
    ) -> Result<Box<dyn ModelPaths>> {
        let repo = api.repo(model_id, revision);
        let model_id = Path::new(model_id);
        let regex = Regex::new(r"^flux\d+-(schnell|dev)\.safetensors$")?;
        let flux_name = api_dir_list!(repo, model_id, true, revision)
            .find(|name| regex.is_match(name))
            .with_context(|| "Expected at least 1 .safetensors file matching the FLUX regex, please raise an issue.")?;
        let flux_weights = api_get_file!(repo, &flux_name, model_id, revision);
        let vae_weights = api_get_file!(repo, "ae.safetensors", model_id, revision);
        let flux_config = api_get_file!(repo, "transformer/config.json", model_id, revision);
        let vae_config = api_get_file!(repo, "vae/config.json", model_id, revision);
        let auxiliary_paths = Self::download_auxiliary_files(api)?;
        Ok(Box::new(FluxModelPaths {
            flux_config,
            vae_config,
            flux_weights,
            vae_weights,
            auxiliary_paths,
        }))
    }

    fn load(
        &self,
        paths: &dyn ModelPaths,
        dtype: DType,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
        silent: bool,
    ) -> Result<Box<dyn DiffusionModel + Send + Sync>> {
        let paths = paths
            .as_any()
            .downcast_ref::<FluxModelPaths>()
            .expect("Flux path downcast failed.");
        let flux_cfg = std::fs::read_to_string(&paths.flux_config)?;
        let vae_cfg = std::fs::read_to_string(&paths.vae_config)?;
        let device = &normal_loading_metadata.real_device;
        let load_weights = |path: &PathBuf, force_cpu| {
            let device = if force_cpu { &Device::Cpu } else { device };
            from_mmaped_safetensors(
                vec![path.clone()],
                Vec::new(),
                Some(dtype),
                device,
                vec![None],
                silent,
                None,
                |_| true,
                Arc::new(|_| DeviceForLoadTensor::Base),
            )
        };
        let flux_vb = load_weights(&paths.flux_weights, self.offload)?;
        let vae_vb = load_weights(&paths.vae_weights, false)?;
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
            FluxStepperLoadContext {
                dtype: flux_dtype,
                device,
                silent,
                offloaded: self.offload,
                paths: &paths.auxiliary_paths,
            },
        )?))
    }
}
