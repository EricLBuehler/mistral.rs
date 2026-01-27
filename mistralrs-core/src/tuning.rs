use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device};
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

/// Quality tier for a quantization level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum QualityTier {
    /// Full precision, baseline quality
    Baseline,
    /// Near-lossless (8-bit)
    NearLossless,
    /// Good quality (6-bit, 5-bit)
    Good,
    /// Acceptable quality (4-bit)
    Acceptable,
    /// Degraded quality (3-bit, 2-bit)
    Degraded,
}

/// Fit status for a quantization candidate
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum FitStatus {
    /// Model fits entirely on GPU(s)
    Fits,
    /// Model requires CPU offloading (hybrid)
    Hybrid,
    /// Model doesn't fit even with CPU offload
    TooLarge,
}

/// A tuning candidate with all calculated metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneCandidate {
    /// Quantization type (None = no quantization)
    pub isq: Option<IsqType>,
    /// Display name for the quantization
    pub isq_name: String,
    /// Estimated model size in bytes
    pub estimated_size_bytes: u64,
    /// VRAM usage as percentage (0.0 - 1.0)
    pub vram_usage_percent: f32,
    /// Maximum context length that fits (model-specific calculation)
    pub max_context_tokens: usize,
    /// Whether max_context_tokens is the model's native maximum (not VRAM limited)
    pub context_is_model_max: bool,
    /// Quality tier
    pub quality: QualityTier,
    /// Whether this candidate fits
    pub fit_status: FitStatus,
    /// Device layer mapping (if hybrid)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_layers_cli: Option<String>,
    /// Whether this is the recommended option
    pub recommended: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneResult {
    pub model_id: String,
    pub profile: TuneProfile,
    pub backend: String,
    /// All evaluated candidates with their metrics
    pub candidates: Vec<TuneCandidate>,
    /// The recommended ISQ type
    pub recommended_isq: Option<IsqType>,
    /// Device layers for the recommended option
    pub device_layers: Option<Vec<DeviceLayerMapMetadata>>,
    pub device_layers_cli: Option<String>,
    pub paged_attn_mode: Option<String>,
    /// Copy-paste command for the recommended option
    pub recommended_command: String,
    /// Total VRAM available across all GPUs (bytes)
    pub total_vram_bytes: u64,
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
        | ModelSelected::Lora {
            model_id: Some(model_id),
            ..
        }
        | ModelSelected::XLora {
            model_id: Some(model_id),
            ..
        }
        | ModelSelected::VisionPlain { model_id, .. }
        | ModelSelected::Embedding { model_id, .. }
        | ModelSelected::Run { model_id, .. } => model_id.clone(),
        ModelSelected::GGUF {
            quantized_model_id, ..
        }
        | ModelSelected::GGML {
            quantized_model_id, ..
        }
        | ModelSelected::LoraGGUF {
            quantized_model_id, ..
        }
        | ModelSelected::XLoraGGUF {
            quantized_model_id, ..
        }
        | ModelSelected::LoraGGML {
            quantized_model_id, ..
        }
        | ModelSelected::XLoraGGML {
            quantized_model_id, ..
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
        let config = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config.json at {}", config_path.display()))?;
        let sentence_transformers = Path::new(model_id)
            .join("config_sentence_transformers.json")
            .exists();
        return Ok((config, sentence_transformers));
    }

    let cache = hf_cache_path
        .map(Cache::new)
        .unwrap_or_else(Cache::from_env);
    GLOBAL_HF_CACHE.get_or_init(|| cache.clone());

    let mut api = ApiBuilder::from_cache(cache)
        .with_progress(false)
        .with_token(get_token(token_source)?);
    if let Some(cache_dir) = crate::hf_hub_cache_dir() {
        api = api.with_cache_dir(cache_dir);
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

    let sentence_transformers =
        api_get_file(&api, model_id, "config_sentence_transformers.json").is_ok();

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

/// Get all candidates to evaluate (in order from highest to lowest quality)
fn all_candidates(backend: TuneBackend) -> Vec<Option<IsqType>> {
    match backend {
        TuneBackend::Metal => vec![
            None, // No quantization
            Some(IsqType::AFQ8),
            Some(IsqType::AFQ6),
            Some(IsqType::AFQ4),
            Some(IsqType::AFQ3),
            Some(IsqType::AFQ2),
        ],
        _ => vec![
            None, // No quantization
            Some(IsqType::Q8_0),
            Some(IsqType::Q6K),
            Some(IsqType::Q5K),
            Some(IsqType::Q4K),
            Some(IsqType::Q3K),
            Some(IsqType::Q2K),
        ],
    }
}

fn default_candidates(profile: TuneProfile, backend: TuneBackend) -> Vec<IsqType> {
    match backend {
        TuneBackend::Metal => match profile {
            TuneProfile::Quality => {
                vec![IsqType::AFQ8, IsqType::AFQ6, IsqType::AFQ4, IsqType::AFQ3]
            }
            TuneProfile::Balanced => vec![IsqType::AFQ6, IsqType::AFQ4, IsqType::AFQ3],
            TuneProfile::Fast => vec![IsqType::AFQ4, IsqType::AFQ3, IsqType::AFQ2],
        },
        _ => match profile {
            TuneProfile::Quality => vec![
                IsqType::Q8_0,
                IsqType::Q6K,
                IsqType::Q5K,
                IsqType::Q4K,
                IsqType::Q3K,
                IsqType::Q2K,
            ],
            TuneProfile::Balanced => vec![IsqType::Q6K, IsqType::Q5K, IsqType::Q4K, IsqType::Q3K],
            TuneProfile::Fast => vec![IsqType::Q4K, IsqType::Q3K, IsqType::Q2K],
        },
    }
}

/// Map ISQ type to quality tier
fn quality_tier(isq: Option<IsqType>) -> QualityTier {
    match isq {
        None => QualityTier::Baseline,
        Some(t) => match t {
            IsqType::Q8_0 | IsqType::Q8_1 | IsqType::Q8K | IsqType::AFQ8 | IsqType::HQQ8 => {
                QualityTier::NearLossless
            }
            IsqType::Q6K | IsqType::AFQ6 => QualityTier::Good,
            IsqType::Q5_0 | IsqType::Q5_1 | IsqType::Q5K => QualityTier::Good,
            IsqType::Q4_0 | IsqType::Q4_1 | IsqType::Q4K | IsqType::AFQ4 | IsqType::HQQ4 => {
                QualityTier::Acceptable
            }
            IsqType::Q3K | IsqType::AFQ3 => QualityTier::Degraded,
            IsqType::Q2K | IsqType::AFQ2 => QualityTier::Degraded,
            _ => QualityTier::Acceptable,
        },
    }
}

/// Get display name for ISQ type
fn isq_display_name(isq: Option<IsqType>) -> String {
    match isq {
        None => "None (FP16)".to_string(),
        Some(t) => format!("{t:?}"),
    }
}

/// Get total VRAM across all GPU devices
#[allow(clippy::cast_possible_truncation)]
fn total_vram(devices: &[Device]) -> u64 {
    use crate::MemoryUsage;
    devices
        .iter()
        .filter(|d| !matches!(d, Device::Cpu))
        .filter_map(|d| MemoryUsage.get_total_memory(d).ok())
        .sum::<usize>() as u64
}

/// Get available VRAM across all GPU devices
#[allow(clippy::cast_possible_truncation)]
fn available_vram(devices: &[Device]) -> u64 {
    use crate::MemoryUsage;
    devices
        .iter()
        .filter(|d| !matches!(d, Device::Cpu))
        .filter_map(|d| MemoryUsage.get_memory_available(d).ok())
        .sum::<usize>() as u64
}

/// Calculate the maximum context length that fits in remaining VRAM
/// Uses the model's actual KV cache configuration from ModelConfigLike
/// Returns (max_context, is_at_model_max) where is_at_model_max is true if
/// the context is limited by the model's native max, not VRAM
#[allow(clippy::cast_possible_truncation)]
fn calculate_max_context(
    loader: &dyn DeviceMappedModelLoader,
    config: &str,
    model_size_bytes: u64,
    available_vram_bytes: u64,
    dtype: DType,
) -> Result<(usize, bool)> {
    let model_cfg = loader.model_config(config)?;
    let native_max_seq_len = model_cfg.max_seq_len();

    if model_size_bytes >= available_vram_bytes {
        return Ok((0, false));
    }

    let remaining_bytes = available_vram_bytes - model_size_bytes;

    // KV cache elements per token (from ModelConfigLike trait)
    // This accounts for num_kv_heads, k_head_dim, v_head_dim correctly
    let kv_elems_per_token = model_cfg.kv_cache_elements_per_token();
    let num_layers = model_cfg.num_layers();

    // Total KV cache bytes per token = elements * dtype_size * num_layers
    let dtype_size = dtype.size_in_bytes();
    let kv_bytes_per_token = kv_elems_per_token * dtype_size * num_layers;

    if kv_bytes_per_token == 0 {
        return Ok((native_max_seq_len, true));
    }

    let calculated_max = remaining_bytes as usize / kv_bytes_per_token;

    // Return the minimum of calculated max and model's native max
    let is_at_model_max = calculated_max >= native_max_seq_len;
    Ok((calculated_max.min(native_max_seq_len), is_at_model_max))
}

fn map_for_candidate(
    loader: &dyn DeviceMappedModelLoader,
    config: &str,
    dtype: DType,
    params: &AutoDeviceMapParams,
    devices: &[Device],
    isq: Option<IsqType>,
) -> Result<(DeviceMapMetadata, usize)> {
    let pack_factor = isq.map(|i| i.pack_factor(dtype)).unwrap_or(1);
    let layer_sizes = loader.layer_sizes_in_bytes(config, dtype, pack_factor, None)?;
    let non_mapped = loader.non_mapped_size_in_bytes(config, dtype, pack_factor, None)?;
    let total = layer_sizes.iter().sum::<usize>() + non_mapped;
    let map = crate::pipeline::get_device_layers_for_loader(
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
    )?;
    Ok((map, total))
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
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

    // Get preferred candidates based on profile (for determining recommendation)
    let preferred_candidates: Vec<Option<IsqType>> =
        req.requested_isq.map(|t| vec![Some(t)]).unwrap_or_else(|| {
            default_candidates(req.profile, backend)
                .into_iter()
                .map(Some)
                .collect()
        });

    let mut warnings = Vec::new();
    let mut notes = Vec::new();

    if matches!(kind, TuneKind::Embedding) {
        notes.push("Detected embedding model configuration.".to_string());
    }
    if matches!(kind, TuneKind::Vision) {
        notes.push("Detected vision model configuration.".to_string());
    }

    // Get total VRAM for calculations
    let total_vram_bytes = total_vram(&devices);
    let avail_vram_bytes = available_vram(&devices);

    // Evaluate ALL candidates
    let all_isq_candidates = all_candidates(backend);
    let mut tune_candidates: Vec<TuneCandidate> = Vec::new();
    let mut recommended_idx: Option<usize> = None;

    for isq in all_isq_candidates {
        let result = map_for_candidate(loader, &config, dtype, &params, &devices, isq);

        let (fit_status, estimated_size, device_layers_cli) = match &result {
            Ok((map, total_size)) => {
                let layers = map.device_layers();
                let is_hybrid = layers
                    .map(|l| l.iter().any(|d| d.ordinal == usize::MAX))
                    .unwrap_or(false);
                let status = if is_hybrid {
                    FitStatus::Hybrid
                } else {
                    FitStatus::Fits
                };
                (status, *total_size as u64, map.to_cli_spec())
            }
            Err(_) => {
                // Calculate estimated size even for non-fitting candidates
                let pack_factor = isq.map(|i| i.pack_factor(dtype)).unwrap_or(1);
                let layer_sizes = loader
                    .layer_sizes_in_bytes(&config, dtype, pack_factor, None)
                    .unwrap_or_default();
                let non_mapped = loader
                    .non_mapped_size_in_bytes(&config, dtype, pack_factor, None)
                    .unwrap_or(0);
                let est_size = (layer_sizes.iter().sum::<usize>() + non_mapped) as u64;
                (FitStatus::TooLarge, est_size, None)
            }
        };

        let vram_usage = if total_vram_bytes > 0 {
            (estimated_size as f32) / (total_vram_bytes as f32)
        } else {
            1.0
        };

        let (context_room, context_is_model_max) =
            calculate_max_context(loader, &config, estimated_size, avail_vram_bytes, dtype)
                .unwrap_or((0, false));

        let candidate = TuneCandidate {
            isq,
            isq_name: isq_display_name(isq),
            estimated_size_bytes: estimated_size,
            vram_usage_percent: vram_usage,
            max_context_tokens: context_room,
            context_is_model_max,
            quality: quality_tier(isq),
            fit_status,
            device_layers_cli,
            recommended: false, // Set later
        };

        tune_candidates.push(candidate);
    }

    // Mark the recommended candidate (first from preferred list that fits)
    for pref in &preferred_candidates {
        if let Some(idx) = tune_candidates.iter().position(|c| {
            c.isq == *pref && matches!(c.fit_status, FitStatus::Fits | FitStatus::Hybrid)
        }) {
            tune_candidates[idx].recommended = true;
            recommended_idx = Some(idx);
            break;
        }
    }

    // If no preferred candidate fits, recommend the first that fits
    if recommended_idx.is_none() {
        if let Some(idx) = tune_candidates
            .iter()
            .position(|c| matches!(c.fit_status, FitStatus::Fits | FitStatus::Hybrid))
        {
            tune_candidates[idx].recommended = true;
            recommended_idx = Some(idx);
        }
    }

    // Build the result
    let (recommended_isq, device_layers, device_layers_cli, recommended_command) =
        if let Some(idx) = recommended_idx {
            let rec = &tune_candidates[idx];
            let isq_flag = rec
                .isq
                .map(|i| format!(" --isq {:?}", i).to_lowercase())
                .unwrap_or_default();
            let cmd = format!("mistralrs serve -m {model_id}{isq_flag}");
            (rec.isq, None, rec.device_layers_cli.clone(), cmd)
        } else {
            (None, None, None, format!("mistralrs serve -m {model_id}"))
        };

    let paged_attn_mode = if backend != TuneBackend::Cpu && paged_attn_supported() {
        Some("auto".to_string())
    } else {
        Some("off".to_string())
    };

    // Add warnings for candidates that don't fit
    for c in &tune_candidates {
        if matches!(c.fit_status, FitStatus::TooLarge) && c.isq.is_some() {
            warnings.push(format!(
                "{} ({:.1} GB) exceeds available VRAM",
                c.isq_name,
                c.estimated_size_bytes as f64 / 1e9
            ));
        }
    }

    if recommended_idx.is_none() {
        anyhow::bail!(
            "No suitable quantization level fits on the available devices. Try a smaller model or enable CPU offload."
        );
    }

    Ok(AutoTuneResult {
        model_id,
        profile: req.profile,
        backend: backend_name(backend),
        candidates: tune_candidates,
        recommended_isq,
        device_layers,
        device_layers_cli,
        paged_attn_mode,
        recommended_command,
        total_vram_bytes,
        warnings,
        notes,
    })
}
