use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{Device, DType};
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Cache, Repo, RepoType,
};
use serde::{Deserialize, Serialize};

use crate::device_map::{DeviceLayerMapMetadata, DeviceMapMetadata};
use crate::model_loader::{get_auto_device_map_params, get_model_dtype};
use crate::pipeline::{
    AutoDeviceMapParams, AutoEmbeddingLoader, AutoNormalLoader, AutoVisionLoader,
    DeviceMappedModelLoader, EmbeddingLoaderType, NormalLoaderType, TokenSource, VisionLoaderType,
};
use crate::utils::tokens::get_token;
use crate::{paged_attn_supported, IsqType, ModelSelected, TryIntoDType, GLOBAL_HF_CACHE};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TuneProfile {
    Quality,
    Balanced,
    Fast,
}

#[derive(Debug, Clone)]
pub struct AutoTuneRequest {
    pub model: ModelSelected,
    pub token_source: TokenSource,
    pub hf_revision: Option<String>,
    pub force_cpu: bool,
    pub profile: TuneProfile,
    pub requested_isq: Option<IsqType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneResult {
    pub model_id: String,
    pub profile: TuneProfile,
    pub backend: String,
    pub recommended_isq: Option<IsqType>,
    pub device_layers: Option<Vec<DeviceLayerMapMetadata>>,
    pub device_layers_cli: Option<String>,
    pub paged_attn_mode: Option<String>,
    pub warnings: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TuneBackend {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Copy, Debug)]
enum TuneKind {
    Normal,
    Vision,
    Embedding,
}

fn backend_from_devices(devices: &[Device]) -> TuneBackend {
    if devices.iter().any(|d| matches!(d, Device::Cuda(_))) {
        TuneBackend::Cuda
    } else if devices.iter().any(|d| matches!(d, Device::Metal(_))) {
        TuneBackend::Metal
    } else {
        TuneBackend::Cpu
    }
}

fn backend_name(backend: TuneBackend) -> String {
    match backend {
        TuneBackend::Cpu => "cpu".to_string(),
        TuneBackend::Cuda => "cuda".to_string(),
        TuneBackend::Metal => "metal".to_string(),
    }
}

fn select_devices(force_cpu: bool) -> Result<Vec<Device>> {
    if force_cpu {
        return Ok(vec![Device::Cpu]);
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    {
        if let Ok(dev) = Device::new_cuda(0) {
            return Ok(crate::device_map::get_all_similar_devices(&dev)?);
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(dev) = Device::new_metal(0) {
            return Ok(crate::device_map::get_all_similar_devices(&dev)?);
        }
    }

    Ok(vec![Device::Cpu])
}

fn hf_cache_path_from_model(model: &ModelSelected) -> Option<PathBuf> {
    match model {
        ModelSelected::Plain { hf_cache_path, .. }
        | ModelSelected::Lora { hf_cache_path, .. }
        | ModelSelected::XLora { hf_cache_path, .. }
        | ModelSelected::VisionPlain { hf_cache_path, .. }
        | ModelSelected::Embedding { hf_cache_path, .. }
        | ModelSelected::Run { hf_cache_path, .. } => hf_cache_path.clone(),
        _ => None,
    }
}

fn model_id_from_selected(model: &ModelSelected) -> String {
    match model {
        ModelSelected::Plain { model_id, .. }
        | ModelSelected::Lora { model_id: Some(model_id), .. }
        | ModelSelected::XLora { model_id: Some(model_id), .. }
        | ModelSelected::VisionPlain { model_id, .. }
        | ModelSelected::Embedding { model_id, .. }
        | ModelSelected::Run { model_id, .. } => model_id.clone(),
        ModelSelected::GGUF {
            quantized_model_id,
            ..
        }
        | ModelSelected::GGML {
            quantized_model_id,
            ..
        }
        | ModelSelected::LoraGGUF {
            quantized_model_id,
            ..
        }
        | ModelSelected::XLoraGGUF {
            quantized_model_id,
            ..
        }
        | ModelSelected::LoraGGML {
            quantized_model_id,
            ..
        }
        | ModelSelected::XLoraGGML {
            quantized_model_id,
            ..
        } => quantized_model_id.clone(),
        ModelSelected::DiffusionPlain { model_id, .. } => model_id.clone(),
        ModelSelected::Speech { model_id, .. } => model_id.clone(),
        ModelSelected::Toml { file } => file.clone(),
        ModelSelected::MultiModel { .. } => "multi-model".to_string(),
        _ => "unknown".to_string(),
    }
}

fn load_config_artifacts(
    model_id: &str,
    token_source: &TokenSource,
    hf_revision: Option<String>,
    hf_cache_path: Option<PathBuf>,
) -> Result<(String, bool)> {
    if Path::new(model_id).exists() {
        let config_path = Path::new(model_id).join("config.json");
        let config = std::fs::read_to_string(&config_path).with_context(|| {
            format!("Failed to read config.json at {}", config_path.display())
        })?;
        let sentence_transformers = Path::new(model_id)
            .join("config_sentence_transformers.json")
            .exists();
        return Ok((config, sentence_transformers));
    }

    let cache = hf_cache_path.map(Cache::new).unwrap_or_else(Cache::from_env);
    GLOBAL_HF_CACHE.get_or_init(|| cache.clone());

    let mut api = ApiBuilder::from_cache(cache)
        .with_progress(false)
        .with_token(get_token(token_source)?);
    if let Ok(x) = std::env::var("HF_HUB_CACHE") {
        api = api.with_cache_dir(x.into());
    }
    let api = api.build()?;
    let revision = hf_revision.unwrap_or_else(|| "main".to_string());
    let api = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision,
    ));

    let config_path = api_get_file(&api, model_id, "config.json")?;
    let config = std::fs::read_to_string(&config_path).with_context(|| {
        format!(
            "Failed to read config.json from cache at {}",
            config_path.display()
        )
    })?;

    let sentence_transformers = api_get_file(&api, model_id, "config_sentence_transformers.json")
        .is_ok();

    Ok((config, sentence_transformers))
}

fn api_get_file(api: &ApiRepo, model_id: &str, file: &str) -> Result<PathBuf> {
    let model_id = Path::new(model_id);
    if model_id.exists() {
        let path = model_id.join(file);
        if path.exists() {
            Ok(path)
        } else {
            anyhow::bail!("File {file} not found at {}", model_id.display())
        }
    } else {
        Ok(api.get(file)?)
    }
}

fn infer_kind(config: &str, sentence_transformers: bool) -> Result<TuneKind> {
    if sentence_transformers {
        return Ok(TuneKind::Embedding);
    }
    #[derive(Deserialize)]
    struct AutoConfig {
        architectures: Vec<String>,
    }
    let cfg: AutoConfig = serde_json::from_str(config)?;
    if cfg.architectures.len() != 1 {
        anyhow::bail!("Expected exactly one architecture in config");
    }
    let name = &cfg.architectures[0];
    if VisionLoaderType::from_causal_lm_name(name).is_ok() {
        return Ok(TuneKind::Vision);
    }
    if EmbeddingLoaderType::from_causal_lm_name(name).is_ok() {
        return Ok(TuneKind::Embedding);
    }
    let _ = NormalLoaderType::from_causal_lm_name(name)?;
    Ok(TuneKind::Normal)
}

fn default_candidates(profile: TuneProfile, backend: TuneBackend) -> Vec<IsqType> {
    match backend {
        TuneBackend::Metal => match profile {
            TuneProfile::Quality => vec![IsqType::AFQ8, IsqType::AFQ6, IsqType::AFQ4, IsqType::AFQ3],
            TuneProfile::Balanced => vec![IsqType::AFQ6, IsqType::AFQ4, IsqType::AFQ3],
            TuneProfile::Fast => vec![IsqType::AFQ4, IsqType::AFQ3, IsqType::AFQ2],
        },
        _ => match profile {
            TuneProfile::Quality => vec![IsqType::Q8_0, IsqType::Q6K, IsqType::Q5K, IsqType::Q4K, IsqType::Q3K, IsqType::Q2K],
            TuneProfile::Balanced => vec![IsqType::Q6K, IsqType::Q5K, IsqType::Q4K, IsqType::Q3K],
            TuneProfile::Fast => vec![IsqType::Q4K, IsqType::Q3K, IsqType::Q2K],
        },
    }
}

fn map_for_candidate(
    loader: &dyn DeviceMappedModelLoader,
    config: &str,
    dtype: DType,
    params: &AutoDeviceMapParams,
    devices: &[Device],
    isq: IsqType,
) -> Result<DeviceMapMetadata> {
    let pack_factor = isq.pack_factor(dtype);
    let layer_sizes = loader.layer_sizes_in_bytes(config, dtype, pack_factor, None)?;
    let non_mapped = loader.non_mapped_size_in_bytes(config, dtype, pack_factor, None)?;
    let total = layer_sizes.iter().sum::<usize>() + non_mapped;
    crate::pipeline::get_device_layers_for_loader(
        loader,
        config,
        loader.num_layers(config)?,
        layer_sizes,
        non_mapped,
        total,
        devices,
        dtype,
        params,
        None,
    )
}

pub fn auto_tune(req: AutoTuneRequest) -> Result<AutoTuneResult> {
    let model_id = model_id_from_selected(&req.model);
    match &req.model {
        ModelSelected::GGUF { .. }
        | ModelSelected::GGML { .. }
        | ModelSelected::LoraGGUF { .. }
        | ModelSelected::XLoraGGUF { .. }
        | ModelSelected::LoraGGML { .. }
        | ModelSelected::XLoraGGML { .. } => {
            anyhow::bail!("Auto-tuning is not supported for pre-quantized GGUF/GGML models.");
        }
        ModelSelected::DiffusionPlain { .. } | ModelSelected::Speech { .. } => {
            anyhow::bail!("Auto-tuning is not supported for diffusion or speech models.");
        }
        _ => {}
    }

    let hf_cache_path = hf_cache_path_from_model(&req.model);
    let (config, sentence_transformers) = load_config_artifacts(
        &model_id,
        &req.token_source,
        req.hf_revision.clone(),
        hf_cache_path,
    )?;

    let kind = match &req.model {
        ModelSelected::VisionPlain { .. } => TuneKind::Vision,
        ModelSelected::Embedding { .. } => TuneKind::Embedding,
        _ => infer_kind(&config, sentence_transformers)?,
    };

    let mut params = get_auto_device_map_params(&req.model)?;
    if matches!(kind, TuneKind::Vision) {
        params = params.maybe_promote_to_vision();
    }

    let devices = select_devices(req.force_cpu)?;
    let backend = backend_from_devices(&devices);

    let dtype = {
        let model_dtype = get_model_dtype(&req.model)?;
        let refs = devices.iter().collect::<Vec<_>>();
        model_dtype.try_into_dtype(&refs)?
    };

    let loader_normal = AutoNormalLoader;
    let loader_vision = AutoVisionLoader;
    let loader_embedding = AutoEmbeddingLoader;
    let loader: &dyn DeviceMappedModelLoader = match kind {
        TuneKind::Normal => &loader_normal,
        TuneKind::Vision => &loader_vision,
        TuneKind::Embedding => &loader_embedding,
    };

    let candidates = req
        .requested_isq
        .map(|t| vec![t])
        .unwrap_or_else(|| default_candidates(req.profile, backend));

    let mut warnings = Vec::new();
    let mut notes = Vec::new();

    if matches!(kind, TuneKind::Embedding) {
        notes.push("Detected embedding model configuration.".to_string());
    }
    if matches!(kind, TuneKind::Vision) {
        notes.push("Detected vision model configuration.".to_string());
    }

    for isq in candidates {
        match map_for_candidate(loader, &config, dtype, &params, &devices, isq) {
            Ok(map) => {
                let device_layers = map.device_layers().map(|layers| layers.to_vec());
                let device_layers_cli = map.to_cli_spec();
                let paged_attn_mode = if backend != TuneBackend::Cpu && paged_attn_supported() {
                    Some("auto".to_string())
                } else {
                    Some("off".to_string())
                };
                return Ok(AutoTuneResult {
                    model_id,
                    profile: req.profile,
                    backend: backend_name(backend),
                    recommended_isq: Some(isq),
                    device_layers,
                    device_layers_cli,
                    paged_attn_mode,
                    warnings,
                    notes,
                });
            }
            Err(err) => {
                warnings.push(format!("ISQ {isq:?} does not fit: {err}"));
            }
        }
    }

    anyhow::bail!(
        "No suitable quantization level fits on the available devices. Try a smaller model or enable CPU offload."
    )
}
