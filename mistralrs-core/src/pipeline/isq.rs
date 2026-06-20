use std::{
    borrow::Cow,
    collections::HashSet,
    fs::File,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use indicatif::{ProgressBar, ProgressStyle};
use mistralrs_quant::{IsqBits, IsqType, TrackedModule, UqffTensor};
use regex::Regex;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::info;

use crate::pipeline::EmbeddingModulePaths;
use crate::utils::progress::configure_progress_bar;

pub(crate) const UQFF_RESIDUAL_SAFETENSORS: &str = "residual.safetensors";
pub const UQFF_MULTI_FILE_DELIMITER: &str = ";";

pub(crate) struct WeightLoadingState {
    pub(crate) from_uqff: bool,
    pub(crate) loading_isq: bool,
    pub(crate) immediate_isq: bool,
    pub(crate) write_uqff: bool,
}

pub(crate) enum WeightLoadingMode {
    Uqff,
    ImmediateIsq,
    PostLoadIsq,
    UqffSerialization,
    Plain,
}

impl From<WeightLoadingState> for WeightLoadingMode {
    fn from(state: WeightLoadingState) -> Self {
        if state.from_uqff {
            Self::Uqff
        } else if state.immediate_isq {
            Self::ImmediateIsq
        } else if state.loading_isq {
            Self::PostLoadIsq
        } else if state.write_uqff {
            Self::UqffSerialization
        } else {
            Self::Plain
        }
    }
}

impl WeightLoadingMode {
    pub(crate) fn message(self, target: &'static str) -> Cow<'static, str> {
        match self {
            Self::Uqff => Cow::Borrowed("Loading UQFF model weights."),
            Self::ImmediateIsq => Cow::Owned(format!("Loading {target} weights for quantization.")),
            Self::PostLoadIsq => Cow::Owned(format!("Loading {target} weights for quantization.")),
            Self::UqffSerialization => {
                Cow::Owned(format!("Loading {target} weights for UQFF output."))
            }
            Self::Plain => Cow::Owned(format!("Loading {target} weights.")),
        }
    }
}

pub(crate) fn format_isq_types(types: &[IsqType]) -> String {
    types
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Parse ISQ value.
///
/// If the provided value is a valid integer (one of 2,3,4,5,6,8), the best quantization type will be chosen.
/// Note that the fallback is always a Q/K quantization but on Metal 2,3,4,6,8 uses the fast AFQ.
///
/// One of:
/// - `Q4_0`
/// - `Q4_1`
/// - `Q5_0`
/// - `Q5_1`
/// - `Q8_0`
/// - `Q8_1`
/// - `Q2K`
/// - `Q3K`
/// - `Q4K`
/// - `Q5K`
/// - `Q6K`
/// - `Q8K`
/// - `HQQ1`
/// - `HQQ2`
/// - `HQQ3`
/// - `HQQ4`
/// - `HQQ8`
/// - `AFQ2`
/// - `AFQ3`
/// - `AFQ4`
/// - `AFQ6`
/// - `AFQ8`
pub fn parse_isq_value(s: &str, device: Option<&Device>) -> Result<IsqType, String> {
    let lowered = s.to_lowercase();

    // Numeric shorthands resolve via IsqBits
    if let Ok(bits) = IsqBits::try_from(lowered.as_str()) {
        let tp = match device {
            Some(dev) => bits.resolve(dev),
            None => bits.resolve(&Device::Cpu),
        };
        #[cfg(feature = "cuda")]
        {
            // All IsqBits resolutions are CUDA-safe, so no extra check needed.
        }
        return Ok(tp);
    }

    let tp = match lowered.as_str() {
        "q4_0" => IsqType::Q4_0,
        "q4_1" => IsqType::Q4_1,
        "q5_0" => IsqType::Q5_0,
        "q5_1" => IsqType::Q5_1,
        "q8_0" => IsqType::Q8_0,
        "q8_1" => IsqType::Q8_1,
        "q2k" => IsqType::Q2K,
        "q3k" => IsqType::Q3K,
        "q4k" => IsqType::Q4K,
        "q5k" => IsqType::Q5K,
        "q6k" => IsqType::Q6K,
        "q8k" => IsqType::Q8K,
        "hqq8" => IsqType::HQQ8,
        "hqq4" => IsqType::HQQ4,
        "fp8" => IsqType::F8E4M3,
        "afq8" => IsqType::AFQ8,
        "afq6" => IsqType::AFQ6,
        "afq4" => IsqType::AFQ4,
        "afq3" => IsqType::AFQ3,
        "afq2" => IsqType::AFQ2,
        "f8q8" => IsqType::F8Q8,
        "mxfp4" => IsqType::MXFP4,
        // "hqq3" => IsqType::HQQ3,
        // "hqq2" => IsqType::HQQ2,
        // "hqq1" => IsqType::HQQ1,
        _ => return Err(format!("ISQ type {s} unknown, choose one of `2`, `3`, `4`, `5`, `6`, `8`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `Q8K`, `HQQ8`, `HQQ4`, `FP8`, `AFQ8`, `AFQ6`, `AFQ4`, `AFQ3`, `AFQ2`, `F8Q8`, `MXFP4`.")),
    };
    #[cfg(feature = "cuda")]
    {
        if !matches!(
            tp,
            IsqType::Q4_0
                | IsqType::Q4_1
                | IsqType::Q5_0
                | IsqType::Q5_1
                | IsqType::Q8_0
                | IsqType::Q2K
                | IsqType::Q3K
                | IsqType::Q4K
                | IsqType::Q5K
                | IsqType::Q6K
                | IsqType::HQQ8
                | IsqType::HQQ4
                | IsqType::F8E4M3
                | IsqType::AFQ2
                | IsqType::AFQ3
                | IsqType::AFQ4
                | IsqType::AFQ6
                | IsqType::AFQ8
                | IsqType::F8Q8
                | IsqType::MXFP4 // | IsqType::HQQ3
                                 // | IsqType::HQQ2
                                 // | IsqType::HQQ1
        ) {
            return Err("ISQ type on CUDA must be one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `HQQ8`, `HQQ4`, `FP8`, `AFQ8`, `AFQ6`, `AFQ4`, `AFQ3`, `AFQ2`, `F8Q8`, `MXFP4`".to_string());
        }
    }
    Ok(tp)
}

/// Expand an ISQ specifier into concrete `IsqType` variants.
/// Numeric shorthands (2-8) produce both the non-Metal and Metal variants;
/// explicit method names resolve to a single variant.
pub fn expand_isq_value(s: &str) -> anyhow::Result<Vec<IsqType>> {
    if let Ok(bits) = IsqBits::try_from(s.to_lowercase().as_str()) {
        return Ok(bits.expand());
    }
    let isq = parse_isq_value(s, None).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(vec![isq])
}

/// Given a UQFF filename like `"q4k-0.uqff"`, returns `Some(("q4k", 0))`.
/// Returns `None` for non-sharded filenames like `"model.uqff"` where the
/// suffix after the last `-` is not a number.
pub fn parse_uqff_shard(filename: &str) -> Option<(String, u64)> {
    let stem = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())?;
    let (prefix, suffix) = stem.rsplit_once('-')?;
    let index = suffix.parse::<u64>().ok()?;
    Some((prefix.to_string(), index))
}

/// Expand a single UQFF filename to include all sibling shards.
///
/// Given `"q4k-0.uqff"` and a list of available files, returns
/// `["q4k-0.uqff", "q4k-1.uqff", ...]` for all sequential indices found.
/// Non-sharded filenames (those not matching `{prefix}-{N}.uqff`) are returned as-is.
pub fn expand_uqff_shards(first_file: &str, available_files: &[String]) -> Vec<String> {
    let Some((prefix, _)) = parse_uqff_shard(first_file) else {
        return vec![first_file.to_string()];
    };
    let mut shards = Vec::new();
    for index in 0u64.. {
        let candidate = format!("{prefix}-{index}.uqff");
        if available_files.iter().any(|f| f == &candidate) {
            shards.push(candidate);
        } else {
            break;
        }
    }
    if shards.is_empty() {
        vec![first_file.to_string()]
    } else {
        shards
    }
}

/// Resolve a UQFF shorthand (numeric like `"8"` or ISQ name like `"q4k"`) to an
/// actual UQFF filename from the available files list.
///
/// Returns `Some("q8_0-0.uqff")` if a matching file is found, `None` otherwise.
/// For numeric shorthands, tries all platform variants via `IsqBits::expand()`.
pub fn resolve_uqff_shorthand(input: &str, available_files: &[String]) -> Option<String> {
    let lowered = input.to_lowercase();

    // Try numeric shorthand first (2/3/4/5/6/8)
    if let Ok(bits) = IsqBits::try_from(lowered.as_str()) {
        for isq_type in bits.expand() {
            let candidate = format!("{isq_type}-0.uqff");
            if available_files.iter().any(|f| f == &candidate) {
                return Some(candidate);
            }
        }
        return None;
    }

    // Try explicit ISQ type name (e.g., "q4k", "afq8", "q8_0")
    if let Ok(isq_type) = parse_isq_value(&lowered, None) {
        let candidate = format!("{isq_type}-0.uqff");
        if available_files.iter().any(|f| f == &candidate) {
            return Some(candidate);
        }
    }

    None
}

#[derive(Clone, Debug, Copy, Default, Deserialize, serde::Serialize)]
pub enum IsqOrganization {
    #[default]
    #[serde(rename = "default")]
    Default,
    /// Only quantize MoE experts, if applicable. The enables MoQE.
    /// <https://arxiv.org/abs/2310.02410>
    #[serde(rename = "moqe")]
    MoeExpertsOnly,
}

impl FromStr for IsqOrganization {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "default" => Ok(Self::Default),
            "moqe" => Ok(Self::MoeExpertsOnly),
            other => Err(format!(
                "Expected ISQ organization `default` or `moqe`, got `{other}`"
            )),
        }
    }
}

pub struct UqffFullSer<'a> {
    pub tokenizer: &'a Tokenizer,
    pub template_filename: &'a Option<PathBuf>,
    pub modules: Option<&'a String>,
    pub module_paths: Option<&'a [EmbeddingModulePaths]>,
    pub generation_config: Option<&'a PathBuf>,
    pub config: String,
    pub processor_filename: &'a Option<PathBuf>,
    pub preprocessor_filename: &'a Option<PathBuf>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct UqffWriteConfig {
    pub output: PathBuf,
    #[serde(default)]
    pub types: Vec<IsqType>,
}

impl<'de> Deserialize<'de> for UqffWriteConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Path(PathBuf),
            Config {
                output: PathBuf,
                #[serde(default)]
                types: Vec<IsqType>,
            },
        }

        match Repr::deserialize(deserializer)? {
            Repr::Path(output) => Ok(Self::from_output(output)),
            Repr::Config { output, types } => Ok(Self::with_types(output, types)),
        }
    }
}

impl FromStr for UqffWriteConfig {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::from_output(PathBuf::from(s)))
    }
}

impl From<PathBuf> for UqffWriteConfig {
    fn from(output: PathBuf) -> Self {
        Self::from_output(output)
    }
}

impl UqffWriteConfig {
    pub fn from_output(output: PathBuf) -> Self {
        Self {
            output,
            types: Vec::new(),
        }
    }

    pub fn with_types(output: PathBuf, types: Vec<IsqType>) -> Self {
        Self { output, types }
    }

    /// Build from an ISQ specifier; numeric shorthands expand to all variants, platform-preferred first.
    pub fn expand_from_str(output: PathBuf, spec: &str) -> anyhow::Result<Self> {
        Ok(Self::with_types(output, expand_isq_value(spec)?))
    }
}

pub(crate) struct UqffWriteRequest<'a> {
    pub output: PathBuf,
    pub types: Vec<IsqType>,
    pub layers: Vec<TrackedModule>,
    pub residual: Vec<(String, Tensor)>,
    pub full_ser: UqffFullSer<'a>,
    pub imatrix: std::collections::HashMap<String, Vec<f32>>,
}

const MAX_UQFF_SIZE_BYTES: usize = 10 * 1024 * 1024 * 1024;

pub(crate) fn write_uqff_artifacts(request: UqffWriteRequest<'_>) -> Result<()> {
    let UqffWriteRequest {
        output,
        types,
        mut layers,
        residual,
        full_ser,
        imatrix,
    } = request;

    if types.is_empty() {
        anyhow::bail!("UQFF serialization requires at least one ISQ type.");
    }
    for ty in &types {
        if !ty.supports_uqff() {
            anyhow::bail!("UQFF serialization does not support {ty}.");
        }
    }
    layers.sort_by(|a, b| a.key.cmp(&b.key));

    let mut output_paths = if types.len() == 1 {
        if output.extension().is_none_or(|ext| ext != "uqff") {
            anyhow::bail!("UQFF output path extension must be `.uqff`");
        }
        vec![(types[0], output.clone())]
    } else {
        if output.extension().is_some_and(|ext| ext == "uqff") {
            anyhow::bail!(
                "Multiple UQFF output types require a directory path, not a `.uqff` file."
            );
        }
        std::fs::create_dir_all(&output)?;
        types
            .iter()
            .map(|ty| (*ty, output.join(format!("{ty}.uqff"))))
            .collect::<Vec<_>>()
    };

    // The first requested type becomes the runtime model; serialize it last so earlier passes still see unquantized sources.
    let runtime_ty = types[0];
    output_paths.sort_by_key(|(ty, _)| *ty == runtime_ty);

    let metadata_parent = if output_paths.len() == 1 {
        output_paths[0]
            .1
            .parent()
            .context("Target UQFF path must have a filename!")?
            .to_path_buf()
    } else {
        output.clone()
    };

    for (ty, path) in output_paths {
        write_uqff_type(ty, &path, &layers, ty == runtime_ty, &imatrix)?;
    }
    info!("In-memory model is quantized as {runtime_ty}.");
    write_uqff_metadata(&metadata_parent, residual, full_ser)
}

fn write_uqff_type(
    ty: IsqType,
    serialized: &Path,
    layers: &[TrackedModule],
    swap_runtime: bool,
    imatrix: &std::collections::HashMap<String, Vec<f32>>,
) -> Result<()> {
    tracing::info!(
        "Serializing {} {ty} UQFF layers to `{}`.",
        layers.len(),
        serialized.display()
    );

    let parent = serialized
        .parent()
        .context("Target UQFF path must have a filename!")?;
    std::fs::create_dir_all(parent)?;

    let file_stem = serialized
        .file_stem()
        .context("Target UQFF path must have a file stem!")?
        .to_string_lossy()
        .to_string();

    let bar = ProgressBar::new(layers.len() as u64);
    configure_progress_bar(&bar);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.red/magenta}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    bar.set_message("starting");
    bar.tick();

    let mut seen = HashSet::new();
    let mut current_chunk = Vec::new();
    let mut current_bytes = 0usize;
    let mut shard_index = 0u64;

    for version in mistralrs_quant::uqff_version_tensors() {
        seen.insert(version.name().to_string());
        current_bytes += version.nbytes();
        current_chunk.push(version);
    }

    // Quantization runs on the pool; the writer consumes results in key order so the shard
    // layout stays deterministic while quantize-N+1 overlaps with write-N.
    // Topology-pinned layers keep their type; `ty` is the default for the rest.
    let handles = mistralrs_quant::requantize_tracked(
        layers,
        ty,
        mistralrs_quant::RequantizeResults::CpuStaged,
        |m| m.ty.unwrap_or(ty),
        &|key| imatrix.get(key).cloned(),
    )?;
    let guard = mistralrs_quant::QuantizeOntoGuard::new();
    for (module, rx) in layers.iter().zip(handles.receivers) {
        bar.set_message(module.key.clone());
        bar.tick();
        let layer = rx
            .recv()
            .map_err(|e| anyhow::anyhow!("Requantize channel error: {e}"))??;
        for tensor in layer.serialize_uqff(&module.key, ty)? {
            let name = tensor.name().to_string();
            if !seen.insert(name.clone()) {
                anyhow::bail!("Duplicate UQFF tensor key `{name}`.");
            }
            let tensor_bytes = tensor.nbytes();
            if !current_chunk.is_empty() && current_bytes + tensor_bytes > MAX_UQFF_SIZE_BYTES {
                flush_uqff_shard(
                    parent,
                    &file_stem,
                    &mut shard_index,
                    &mut current_chunk,
                    &mut current_bytes,
                )?;
            }
            current_bytes += tensor_bytes;
            current_chunk.push(tensor);
            if current_bytes >= MAX_UQFF_SIZE_BYTES {
                flush_uqff_shard(
                    parent,
                    &file_stem,
                    &mut shard_index,
                    &mut current_chunk,
                    &mut current_bytes,
                )?;
            }
        }
        if swap_runtime {
            // GGML layers were quantized on CPU for safe byte extraction; upload to the target.
            let target = module.ct.resolve()?.dtype_and_device().1;
            let layer = if layer.dtype_and_device().1.same_device(&target) {
                layer
            } else {
                layer.clone().apply_isq(
                    None,
                    target,
                    &std::sync::atomic::AtomicUsize::new(0),
                    None,
                    guard.clone(),
                )?
            };
            module.ct.replace(layer);
        }
        bar.inc(1);
    }

    flush_uqff_shard(
        parent,
        &file_stem,
        &mut shard_index,
        &mut current_chunk,
        &mut current_bytes,
    )?;
    bar.finish_and_clear();
    Ok(())
}

fn flush_uqff_shard(
    parent: &Path,
    file_stem: &str,
    shard_index: &mut u64,
    current_chunk: &mut Vec<UqffTensor>,
    current_bytes: &mut usize,
) -> Result<()> {
    if current_chunk.is_empty() {
        return Ok(());
    }

    let shard_path = parent.join(format!("{file_stem}-{shard_index}.uqff"));
    info!(
        "Writing shard {} to `{}`",
        shard_index,
        shard_path.display()
    );
    safetensors::serialize_to_file(
        current_chunk.iter().map(|tensor| (tensor.name(), tensor)),
        None,
        &shard_path,
    )?;
    *shard_index += 1;
    current_chunk.clear();
    *current_bytes = 0;
    Ok(())
}

fn write_uqff_metadata(
    metadata_parent: &Path,
    residual: Vec<(String, Tensor)>,
    full_ser: UqffFullSer<'_>,
) -> Result<()> {
    let residual_out = metadata_parent.join(UQFF_RESIDUAL_SAFETENSORS);
    let config_out = metadata_parent.join("config.json");
    let modules_out = metadata_parent.join("modules.json");
    let tokenizer_out = metadata_parent.join("tokenizer.json");
    let tokenizer_cfg_out = metadata_parent.join("tokenizer_config.json");
    let chat_template_jinja_out = metadata_parent.join("chat_template.jinja");
    let gen_cfg_out = metadata_parent.join("generation_config.json");
    let processor_out = metadata_parent.join("processor_config.json");
    let preprocessor_out = metadata_parent.join("preprocessor_config.json");

    info!(
        "Serializing {} residual tensors to `{}`.",
        residual.len(),
        residual_out.display()
    );
    safetensors::serialize_to_file(residual, None, &residual_out)?;

    let UqffFullSer {
        tokenizer,
        template_filename,
        modules,
        module_paths,
        generation_config,
        config,
        processor_filename,
        preprocessor_filename,
    } = full_ser;

    info!("Serializing configuration to `{}`.", config_out.display());
    std::fs::write(config_out, config)?;

    info!("Serializing tokenizer to `{}`.", tokenizer_out.display());
    serde_json::to_writer_pretty(File::create(&tokenizer_out)?, tokenizer)
        .map_err(candle_core::Error::msg)?;

    if let Some(template_filename) = template_filename {
        let template = std::fs::read(template_filename).map_err(candle_core::Error::msg)?;

        if template_filename.extension().map(|e| e.to_str()) == Some(Some("jinja")) {
            info!(
                "Serializing chat template to `{}`.",
                chat_template_jinja_out.display()
            );
            std::fs::write(&chat_template_jinja_out, template).map_err(candle_core::Error::msg)?;

            let sibling_cfg = template_filename
                .parent()
                .map(|dir| dir.join("tokenizer_config.json"));
            if let Some(cfg_path) = sibling_cfg.filter(|p| p.exists()) {
                info!(
                    "Serializing tokenizer config to `{}`.",
                    tokenizer_cfg_out.display()
                );
                std::fs::copy(&cfg_path, &tokenizer_cfg_out).map_err(candle_core::Error::msg)?;
            }
        } else {
            info!(
                "Serializing tokenizer config to `{}`.",
                tokenizer_cfg_out.display()
            );
            std::fs::write(&tokenizer_cfg_out, template).map_err(candle_core::Error::msg)?;
        }
    }

    if let Some(generation_config) = generation_config {
        info!(
            "Serializing generation config to `{}`.",
            gen_cfg_out.display()
        );
        let cfg = std::fs::read(generation_config).map_err(candle_core::Error::msg)?;
        std::fs::write(&gen_cfg_out, cfg).map_err(candle_core::Error::msg)?;
    }

    if let Some(processor_config) = processor_filename {
        info!(
            "Serializing processor config to `{}`.",
            processor_out.display()
        );
        let cfg = std::fs::read(processor_config).map_err(candle_core::Error::msg)?;
        std::fs::write(&processor_out, cfg).map_err(candle_core::Error::msg)?;
    }

    if let Some(preprocessor_config) = preprocessor_filename {
        info!(
            "Serializing preprocessor config to `{}`.",
            preprocessor_out.display()
        );
        let cfg = std::fs::read(preprocessor_config).map_err(candle_core::Error::msg)?;
        std::fs::write(&preprocessor_out, cfg).map_err(candle_core::Error::msg)?;
    }

    if let Some(modules) = modules {
        info!(
            "Serializing modules manifest to `{}`.",
            modules_out.display()
        );
        std::fs::write(&modules_out, modules).map_err(candle_core::Error::msg)?;

        if let Some(module_paths) = module_paths {
            for module in module_paths {
                match module {
                    EmbeddingModulePaths::Transformer { path }
                    | EmbeddingModulePaths::Pooling { path, .. }
                    | EmbeddingModulePaths::Dense { path, .. }
                    | EmbeddingModulePaths::Normalize { path } => {
                        if path.is_empty() {
                            continue;
                        }
                        let module_dir = metadata_parent.join(path.as_str());
                        std::fs::create_dir_all(&module_dir).map_err(candle_core::Error::msg)?;

                        match module {
                            EmbeddingModulePaths::Pooling { config, .. } => {
                                let dest = module_dir.join("config.json");
                                if config != &dest {
                                    std::fs::copy(config, &dest)
                                        .map_err(candle_core::Error::msg)?;
                                }
                            }
                            EmbeddingModulePaths::Dense { config, model, .. } => {
                                let dest_cfg = module_dir.join("config.json");
                                if config != &dest_cfg {
                                    std::fs::copy(config, &dest_cfg)
                                        .map_err(candle_core::Error::msg)?;
                                }
                                let dest_model = module_dir.join("model.safetensors");
                                if model != &dest_model {
                                    std::fs::copy(model, &dest_model)
                                        .map_err(candle_core::Error::msg)?;
                                }
                            }
                            EmbeddingModulePaths::Transformer { .. }
                            | EmbeddingModulePaths::Normalize { .. } => {}
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

pub trait IsqModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)>;

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        None
    }
}

/// Trait for loading models with ISQ.
pub(crate) trait IsqModelLoader {
    /// Regex to match layers which will have standard *immediate* ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }

    /// Regex to match layers which will have standard MoQE *immediate* ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    /// Regex to match layers which will have standard ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }

    /// Regex to match layers which will have standard MoQE ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

/// Map a layer tracking key to its llama.cpp imatrix entry name (`blk.N.attn_q.weight` style).
fn gguf_imatrix_name(key: &str) -> Option<String> {
    if key == "lm_head" || key.ends_with(".lm_head") {
        return Some("output.weight".to_string());
    }
    let index = mistralrs_quant::layer_index_from_prefix(key)?;
    let is_expert = key.contains(".experts");
    let name = if key.ends_with(".self_attn.q_proj") {
        "attn_q"
    } else if key.ends_with(".self_attn.k_proj") {
        "attn_k"
    } else if key.ends_with(".self_attn.v_proj") {
        "attn_v"
    } else if key.ends_with(".self_attn.o_proj") {
        "attn_output"
    } else if key.ends_with(".gate_proj") {
        if is_expert {
            "ffn_gate_exps"
        } else {
            "ffn_gate"
        }
    } else if key.ends_with(".up_proj") {
        if is_expert {
            "ffn_up_exps"
        } else {
            "ffn_up"
        }
    } else if key.ends_with(".down_proj") {
        if is_expert {
            "ffn_down_exps"
        } else {
            "ffn_down"
        }
    } else if key.ends_with(".gate") {
        "ffn_gate_inp"
    } else {
        return None;
    };
    Some(format!("blk.{index}.{name}.weight"))
}

/// Load per-layer imatrix weights for `modules` from a `.cimatrix` (tracking-key keyed) or
/// llama.cpp `.imatrix` file.
pub(crate) fn load_imatrix_map(
    path: &Path,
    modules: &[TrackedModule],
) -> Result<std::collections::HashMap<String, Vec<f32>>> {
    if path.extension().is_some_and(|ext| ext == "cimatrix") {
        info!("Loading collected imatrix file `{}`.", path.display());
        return Ok(mistralrs_quant::CollectedImatrixData::load_imatrix(path)?.0);
    }
    info!("Loading GGUF-format imatrix file `{}`.", path.display());
    let mut data = candle_core::quantized::imatrix_file::load_imatrix(path)?;
    let mut map = std::collections::HashMap::new();
    for module in modules {
        if let Some(name) = gguf_imatrix_name(&module.key) {
            if let Some(values) = data.remove(&name) {
                map.insert(module.key.clone(), values);
            }
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_uqff_shorthand_numeric_q8() {
        let files = vec!["q8_0-0.uqff".to_string(), "config.json".to_string()];
        assert_eq!(
            resolve_uqff_shorthand("8", &files),
            Some("q8_0-0.uqff".to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_numeric_afq8() {
        let files = vec!["afq8-0.uqff".to_string(), "config.json".to_string()];
        assert_eq!(
            resolve_uqff_shorthand("8", &files),
            Some("afq8-0.uqff".to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_prefers_platform_variant() {
        // expand() returns platform-preferred variant first:
        // Metal: [AFQ8, Q8_0], non-Metal: [Q8_0, AFQ8]
        let files = vec!["q8_0-0.uqff".to_string(), "afq8-0.uqff".to_string()];
        let expected = if cfg!(feature = "metal") {
            "afq8-0.uqff"
        } else {
            "q8_0-0.uqff"
        };
        assert_eq!(
            resolve_uqff_shorthand("8", &files),
            Some(expected.to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_numeric_q4() {
        let files = vec!["q4k-0.uqff".to_string()];
        assert_eq!(
            resolve_uqff_shorthand("4", &files),
            Some("q4k-0.uqff".to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_numeric_q5() {
        let files = vec!["q5k-0.uqff".to_string()];
        assert_eq!(
            resolve_uqff_shorthand("5", &files),
            Some("q5k-0.uqff".to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_isq_name() {
        let files = vec!["q4k-0.uqff".to_string(), "q8_0-0.uqff".to_string()];
        assert_eq!(
            resolve_uqff_shorthand("q4k", &files),
            Some("q4k-0.uqff".to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_explicit_filename_returns_none() {
        let files = vec!["q8_0-0.uqff".to_string()];
        assert_eq!(resolve_uqff_shorthand("q8_0-0.uqff", &files), None);
    }

    #[test]
    fn test_resolve_uqff_shorthand_no_match() {
        let files = vec!["config.json".to_string()];
        assert_eq!(resolve_uqff_shorthand("8", &files), None);
    }
}
