//! Model-related argument structs

use clap::{Args, ValueEnum};
use mistralrs_core::{
    AutoDeviceMapParams, HfConfigOverrides, IsqOrganization, LoraAdapterSpec, LoraRuntimeConfig,
    ModelDType, NormalLoaderType, DEFAULT_LORA_MAX_ADAPTERS, DEFAULT_LORA_MAX_BYTES,
    DEFAULT_LORA_MAX_RANK, MAX_LORA_ALIAS_BYTES,
};
use serde::Deserialize;
use std::{
    collections::HashSet,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

const KILOBYTE: u64 = 1_000;
const MEGABYTE: u64 = 1_000_000;
const GIGABYTE: u64 = 1_000_000_000;
const KIBIBYTE: u64 = 1 << 10;
const MEBIBYTE: u64 = 1 << 20;
const GIBIBYTE: u64 = 1 << 30;

/// Model source options
#[derive(Args, Clone, Deserialize)]
pub struct ModelSourceOptions {
    /// Hugging Face model ID or local path to model directory
    #[arg(short = 'm', long)]
    pub model_id: String,

    /// Path to local tokenizer.json file
    #[arg(short = 't', long)]
    pub tokenizer: Option<PathBuf>,

    /// Model architecture (auto-detected if not specified)
    #[arg(short = 'a', long, value_parser = parse_arch)]
    pub arch: Option<NormalLoaderType>,

    /// Model data type
    #[arg(long, default_value = "auto", value_parser = parse_dtype)]
    #[serde(default)]
    pub dtype: ModelDType,

    /// Recursively merged JSON overrides for the Hugging Face model config
    #[arg(long)]
    pub hf_overrides: Option<HfConfigOverrides>,

    /// Runtime model context length
    #[arg(long, value_parser = parse_positive_usize)]
    pub max_model_len: Option<usize>,
}

pub(super) fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|err| format!("invalid positive integer: {err}"))?;
    if value == 0 {
        return Err("value must be greater than zero".to_string());
    }
    Ok(value)
}

/// Format options for model loading
#[derive(Args, Clone, Default, Deserialize)]
pub struct FormatOptions {
    /// Model format: plain (safetensors), GGUF, or GGML.
    /// Auto-detected from `-f` when not specified.
    #[arg(long, value_enum)]
    pub format: Option<ModelFormat>,

    /// GGUF/GGML filename(s); the suffix selects the format (semicolon-separated for multiple)
    #[arg(short = 'f', long)]
    pub quantized_file: Option<String>,

    /// GGUF projector override; auto-selected when unambiguous (semicolon-separated for multiple)
    #[arg(long)]
    pub mmproj: Option<String>,

    /// Optional model ID overriding configuration, tokenizer, and processor assets for a quantized model
    #[arg(long)]
    pub tok_model_id: Option<String>,

    /// GQA value for GGML models
    #[arg(long, default_value_t = 1)]
    #[serde(default = "default_gqa")]
    pub gqa: usize,

    #[doc(hidden)]
    #[arg(skip)]
    #[serde(skip)]
    pub direct_file_only: bool,
}

impl FormatOptions {
    pub(crate) fn normalize(&mut self) -> anyhow::Result<()> {
        let mut format = self.format;
        if self.mmproj.is_some() {
            match format {
                Some(ModelFormat::Plain | ModelFormat::Ggml) => {
                    anyhow::bail!("`--mmproj` requires GGUF format")
                }
                Some(ModelFormat::Gguf) => {}
                None => format = Some(ModelFormat::Gguf),
            }
        }

        let Some(raw_filenames) = self.quantized_file.as_ref() else {
            self.format = format;
            return Ok(());
        };
        let filenames = split_quantized_filenames(raw_filenames)?;
        let mut inferred = None;
        let mut has_unknown = false;
        for filename in &filenames {
            let Some(candidate) = ModelFormat::from_filename(filename) else {
                has_unknown = true;
                continue;
            };
            if inferred.is_some_and(|current| current != candidate) {
                anyhow::bail!(
                    "`--quantized-file` contains mixed GGUF and GGML filenames; use one format per model"
                );
            }
            inferred = Some(candidate);
        }

        match format {
            Some(ModelFormat::Plain) => {
                anyhow::bail!("`--quantized-file` cannot be used with plain model format")
            }
            Some(explicit) => {
                if inferred.is_some_and(|candidate| candidate != explicit) {
                    anyhow::bail!(
                        "`--format {}` conflicts with the `--quantized-file` suffix",
                        explicit.as_str()
                    );
                }
            }
            None if has_unknown => {
                anyhow::bail!(
                    "Cannot infer model format from `--quantized-file`; use `.gguf`/`.ggml` filenames or pass `--format`"
                )
            }
            None => {
                format = inferred;
            }
        }

        self.format = format;
        self.quantized_file = Some(filenames.join(";"));
        Ok(())
    }

    pub(crate) fn derive_local_model_root(&mut self) -> anyhow::Result<String> {
        let raw_filenames = self.quantized_file.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--model-id (-m) is required unless `-f` is provided")
        })?;
        let filenames = split_quantized_filenames(raw_filenames)?;
        let mut root = None;
        let mut basenames = Vec::with_capacity(filenames.len());

        for filename in filenames {
            let path = Path::new(filename);
            if !path.is_file() {
                anyhow::bail!(
                    "Local quantized model file `{}` does not exist or is not a file",
                    path.display()
                );
            }
            let basename = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| anyhow::anyhow!("Invalid quantized model path `{filename}`"))?;
            let parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            if root.as_ref().is_some_and(|current| current != parent) {
                anyhow::bail!(
                    "Quantized model shards must share one parent directory when `--model-id` is omitted"
                );
            }
            root = Some(parent.to_path_buf());
            basenames.push(basename.to_string());
        }

        let root = root.unwrap_or_else(|| PathBuf::from("."));
        self.quantized_file = Some(basenames.join(";"));
        if let Some(mmproj) = self.mmproj.as_ref() {
            let filenames = split_quantized_filenames(mmproj)?;
            let current_dir = std::env::current_dir()?;
            self.mmproj = Some(
                filenames
                    .into_iter()
                    .map(|filename| -> anyhow::Result<String> {
                        let path = Path::new(filename);
                        let path = if path.is_absolute() {
                            if !path.is_file() {
                                anyhow::bail!(
                                    "Local GGUF projector file `{}` does not exist or is not a file",
                                    path.display()
                                );
                            }
                            path.strip_prefix(&root).unwrap_or(path).to_path_buf()
                        } else if path
                            .strip_prefix(&root)
                            .is_ok_and(|relative| path.is_file() && !relative.as_os_str().is_empty())
                        {
                            path.strip_prefix(&root)
                                .expect("projector path was checked against the model root")
                                .to_path_buf()
                        } else if path.parent().is_none_or(|parent| parent.as_os_str().is_empty())
                            && root.join(path).is_file()
                        {
                            path.to_path_buf()
                        } else if path.is_file() {
                            current_dir.join(path)
                        } else if root.join(path).is_file() {
                            path.to_path_buf()
                        } else {
                            anyhow::bail!(
                                "Local GGUF projector file `{}` does not exist relative to the \
                                 current directory or model directory",
                                path.display()
                            );
                        };
                        Ok(path.to_string_lossy().into_owned())
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?
                    .join(";"),
            );
        }
        self.direct_file_only = true;
        Ok(root.to_string_lossy().into_owned())
    }
}

/// Model format type
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFormat {
    /// Plain model (safetensors)
    #[default]
    Plain,
    /// GGUF quantized model
    Gguf,
    /// GGML quantized model
    Ggml,
}

impl ModelFormat {
    fn from_filename(filename: &str) -> Option<Self> {
        match Path::new(filename)
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("gguf") => Some(Self::Gguf),
            Some("ggml") => Some(Self::Ggml),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Gguf => "gguf",
            Self::Ggml => "ggml",
        }
    }
}

fn split_quantized_filenames(value: &str) -> anyhow::Result<Vec<&str>> {
    let filenames = value.split(';').map(str::trim).collect::<Vec<_>>();
    if filenames.is_empty() || filenames.iter().any(|filename| filename.is_empty()) {
        anyhow::bail!("`--quantized-file` must contain nonempty filenames");
    }
    Ok(filenames)
}

/// Adapter options (LoRA/X-LoRA)
#[derive(Args, Clone, Deserialize)]
pub struct AdapterOptions {
    /// Enable dynamic LoRA without preloading an adapter. Supports compatible text and multimodal
    /// language models, including GGUF. Vision, audio, and projector adapters are unsupported.
    #[arg(long, conflicts_with = "xlora")]
    #[serde(default)]
    pub enable_lora: bool,

    /// Preload a language-model LoRA adapter as ALIAS=SOURCE. Supports compatible text and
    /// multimodal language models, including GGUF. Remote adapters use revision main. May be
    /// repeated. Vision, audio, and projector adapters are unsupported.
    #[arg(
        long,
        visible_alias = "lora-modules",
        value_name = "ALIAS=SOURCE|JSON",
        value_parser = parse_lora_adapter,
        num_args = 1..,
        action = clap::ArgAction::Append,
        conflicts_with = "xlora"
    )]
    #[serde(default)]
    pub lora: Vec<LoraAdapterSpec>,

    /// Maximum loaded LoRA aliases and, independently, resident adapter generations
    #[arg(
        long,
        default_value_t = DEFAULT_LORA_MAX_ADAPTERS,
        conflicts_with_all = ["xlora", "legacy_lora"]
    )]
    #[serde(default = "default_lora_max_adapters")]
    pub lora_max_adapters: usize,

    /// Maximum rank accepted for a LoRA adapter
    #[arg(
        long,
        visible_alias = "max-lora-rank",
        default_value_t = DEFAULT_LORA_MAX_RANK,
        conflicts_with_all = ["xlora", "legacy_lora"]
    )]
    #[serde(default = "default_lora_max_rank")]
    pub lora_max_rank: usize,

    /// Maximum memory used by loaded adapters
    #[arg(
        long,
        value_parser = parse_lora_bytes,
        value_name = "BYTES",
        default_value_t = DEFAULT_LORA_MAX_BYTES,
        conflicts_with_all = ["xlora", "legacy_lora"]
    )]
    #[serde(default = "default_lora_max_bytes")]
    pub lora_max_bytes: u64,

    /// Static LoRA adapter source for GGML or a Phi3 GGUF model
    #[arg(
        long,
        value_name = "SOURCE",
        conflicts_with_all = ["enable_lora", "lora", "xlora"]
    )]
    pub legacy_lora: Option<String>,

    /// Ordering JSON file for a legacy raw GGUF or GGML LoRA adapter
    #[arg(long, requires = "legacy_lora")]
    pub legacy_lora_order: Option<PathBuf>,

    /// X-LoRA adapter model ID
    #[arg(long, conflicts_with_all = ["enable_lora", "lora", "legacy_lora"])]
    pub xlora: Option<String>,

    /// X-LoRA ordering JSON file
    #[arg(long, requires = "xlora")]
    pub xlora_order: Option<PathBuf>,

    /// Target non-granular index for X-LoRA
    #[arg(long, requires = "xlora")]
    pub tgt_non_granular_index: Option<usize>,
}

impl AdapterOptions {
    pub fn dynamic_lora_enabled(&self) -> bool {
        self.enable_lora || !self.lora.is_empty()
    }

    pub fn lora_runtime_config(&self) -> LoraRuntimeConfig {
        LoraRuntimeConfig {
            max_adapters: self.lora_max_adapters,
            max_rank: self.lora_max_rank,
            max_bytes: self.lora_max_bytes,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let dynamic_lora = self.dynamic_lora_enabled();
        let legacy_lora = self.legacy_lora.is_some();
        let xlora = self.xlora.is_some();
        if usize::from(dynamic_lora) + usize::from(legacy_lora) + usize::from(xlora) > 1 {
            return Err("dynamic LoRA, legacy LoRA, and X-LoRA are mutually exclusive".to_string());
        }
        if !dynamic_lora && self.lora_runtime_config() != LoraRuntimeConfig::default() {
            return Err(
                "LoRA runtime limits require --enable-lora or at least one --lora preload"
                    .to_string(),
            );
        }
        if legacy_lora != self.legacy_lora_order.is_some() {
            return Err("legacy_lora and legacy_lora_order must be specified together".to_string());
        }
        if xlora != self.xlora_order.is_some() {
            return Err("xlora and xlora_order must be specified together".to_string());
        }
        if self
            .legacy_lora
            .as_ref()
            .is_some_and(|source| source.trim().is_empty())
        {
            return Err("legacy LoRA adapter source must not be empty".to_string());
        }
        if self.lora_max_adapters == 0 {
            return Err("--lora-max-adapters must be greater than zero".to_string());
        }
        if self.lora_max_rank == 0 {
            return Err("--lora-max-rank must be greater than zero".to_string());
        }
        if self.lora_max_bytes == 0 {
            return Err("--lora-max-bytes must be greater than zero".to_string());
        }

        let mut aliases = HashSet::new();
        for adapter in &self.lora {
            validate_lora_adapter(adapter)?;
            if !aliases.insert(adapter.alias.trim()) {
                return Err(format!(
                    "LoRA adapter alias `{}` is specified more than once",
                    adapter.alias
                ));
            }
        }
        Ok(())
    }
}

impl Default for AdapterOptions {
    fn default() -> Self {
        Self {
            enable_lora: false,
            lora: Vec::new(),
            lora_max_adapters: DEFAULT_LORA_MAX_ADAPTERS,
            lora_max_rank: DEFAULT_LORA_MAX_RANK,
            lora_max_bytes: DEFAULT_LORA_MAX_BYTES,
            legacy_lora: None,
            legacy_lora_order: None,
            xlora: None,
            xlora_order: None,
            tgt_non_granular_index: None,
        }
    }
}

#[derive(Args, Clone)]
pub struct MultimodalAdapterOptions {
    /// Enable dynamic LoRA for the language model without preloading an adapter. Vision, audio, and
    /// projector adapters are unsupported.
    #[arg(long)]
    pub enable_lora: bool,

    /// Preload a language-model LoRA adapter as ALIAS=SOURCE. Remote adapters use revision main.
    /// May be repeated. Vision, audio, and projector adapters are unsupported.
    #[arg(
        long,
        visible_alias = "lora-modules",
        value_name = "ALIAS=SOURCE|JSON",
        value_parser = parse_lora_adapter,
        num_args = 1..,
        action = clap::ArgAction::Append
    )]
    pub lora: Vec<LoraAdapterSpec>,

    /// Maximum loaded LoRA aliases and, independently, resident adapter generations
    #[arg(long, default_value_t = DEFAULT_LORA_MAX_ADAPTERS)]
    pub lora_max_adapters: usize,

    /// Maximum rank accepted for a LoRA adapter
    #[arg(
        long,
        visible_alias = "max-lora-rank",
        default_value_t = DEFAULT_LORA_MAX_RANK
    )]
    pub lora_max_rank: usize,

    /// Maximum memory used by loaded adapters
    #[arg(
        long,
        value_parser = parse_lora_bytes,
        value_name = "BYTES",
        default_value_t = DEFAULT_LORA_MAX_BYTES
    )]
    pub lora_max_bytes: u64,
}

impl MultimodalAdapterOptions {
    pub fn dynamic_lora_enabled(&self) -> bool {
        self.enable_lora || !self.lora.is_empty()
    }

    pub fn as_adapter_options(&self) -> AdapterOptions {
        AdapterOptions {
            enable_lora: self.enable_lora,
            lora: self.lora.clone(),
            lora_max_adapters: self.lora_max_adapters,
            lora_max_rank: self.lora_max_rank,
            lora_max_bytes: self.lora_max_bytes,
            ..AdapterOptions::default()
        }
    }

    pub fn from_adapter_options(options: &AdapterOptions) -> Self {
        Self {
            enable_lora: options.enable_lora,
            lora: options.lora.clone(),
            lora_max_adapters: options.lora_max_adapters,
            lora_max_rank: options.lora_max_rank,
            lora_max_bytes: options.lora_max_bytes,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.as_adapter_options().validate()
    }
}

impl Default for MultimodalAdapterOptions {
    fn default() -> Self {
        Self {
            enable_lora: false,
            lora: Vec::new(),
            lora_max_adapters: DEFAULT_LORA_MAX_ADAPTERS,
            lora_max_rank: DEFAULT_LORA_MAX_RANK,
            lora_max_bytes: DEFAULT_LORA_MAX_BYTES,
        }
    }
}

fn parse_lora_adapter(value: &str) -> Result<LoraAdapterSpec, String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct LoraModuleObject {
        name: String,
        path: String,
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        base_model_name: Option<String>,
        #[serde(default, rename = "is_3d_lora_weight")]
        _is_3d_lora_weight: Option<bool>,
    }

    let value = value.trim();
    if value.starts_with('{') {
        let module: LoraModuleObject = serde_json::from_str(value)
            .map_err(|error| format!("invalid LoRA module JSON: {error}"))?;
        let mut adapter = LoraAdapterSpec::new(module.name, module.path);
        if let Some(revision) = module.revision {
            adapter = adapter.with_revision(revision);
        }
        if let Some(base_model_name) = module.base_model_name {
            adapter = adapter.with_base_model_name(base_model_name);
        }
        validate_lora_adapter(&adapter)?;
        return Ok(adapter);
    }

    let (alias, source) = value
        .split_once('=')
        .ok_or_else(|| "expected ALIAS=SOURCE or a JSON object with name and path".to_string())?;
    let adapter = LoraAdapterSpec::new(alias.trim(), source.trim());
    validate_lora_adapter(&adapter)?;
    Ok(adapter)
}

fn parse_lora_bytes(value: &str) -> Result<u64, String> {
    let value = value.trim();
    let digits = value.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return Err("expected a byte count such as 8589934592 or 8GiB".to_string());
    }
    let amount = value[..digits]
        .parse::<u64>()
        .map_err(|error| format!("invalid byte count: {error}"))?;
    let multiplier = match value[digits..].trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "kb" => KILOBYTE,
        "mb" => MEGABYTE,
        "gb" => GIGABYTE,
        "kib" => KIBIBYTE,
        "mib" => MEBIBYTE,
        "gib" => GIBIBYTE,
        suffix => return Err(format!("unsupported byte-size suffix `{suffix}`")),
    };
    amount
        .checked_mul(multiplier)
        .ok_or_else(|| "LoRA byte limit is too large".to_string())
}

fn validate_lora_adapter(adapter: &LoraAdapterSpec) -> Result<(), String> {
    let alias = adapter.alias.trim();
    if alias.is_empty() {
        return Err("LoRA adapter alias must not be empty".to_string());
    }
    if alias.len() > MAX_LORA_ALIAS_BYTES {
        return Err(format!(
            "LoRA adapter alias must not exceed {MAX_LORA_ALIAS_BYTES} bytes"
        ));
    }
    if adapter.source.trim().is_empty() {
        return Err(format!(
            "LoRA adapter source for alias `{}` must not be empty",
            adapter.alias
        ));
    }
    if adapter.revision().is_empty() {
        return Err(format!(
            "LoRA adapter revision for alias `{}` must not be empty",
            adapter.alias
        ));
    }
    if adapter
        .base_model_name
        .as_deref()
        .is_some_and(|model| model.trim().is_empty())
    {
        return Err(format!(
            "LoRA adapter `{}` has an empty base_model_name",
            adapter.alias
        ));
    }
    Ok(())
}

fn default_lora_max_adapters() -> usize {
    DEFAULT_LORA_MAX_ADAPTERS
}

fn default_lora_max_rank() -> usize {
    DEFAULT_LORA_MAX_RANK
}

fn default_lora_max_bytes() -> u64 {
    DEFAULT_LORA_MAX_BYTES
}

/// Quantization options
#[derive(Args, Clone, Default, Deserialize)]
pub struct QuantizationOptions {
    /// Quantization target. Inference commands select a matching GGUF or UQFF artifact when
    /// available. Source checkpoints without a matching UQFF use in-situ quantization. `tune`
    /// evaluates the requested level instead of selecting an artifact. Accepts numeric levels
    /// (`2`, `3`, `4`, `5`, `6`, `8`) or supported quantization names.
    #[arg(long, conflicts_with_all = ["in_situ_quant", "from_uqff"])]
    pub quant: Option<String>,

    /// In-situ quantization target. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or
    /// raw quant names (`q4k`, `q8_0`, etc.). Supports compatible GGUF sources.
    #[arg(long = "isq")]
    #[serde(rename = "isq", alias = "in_situ_quant")]
    pub in_situ_quant: Option<String>,

    /// UQFF artifact to load. Accepts a filename, numeric quantization level (`2`, `3`, `4`, `5`,
    /// `6`, `8`), or quantization type (`q4k`, `afq8`, etc.). Report-declared artifacts and
    /// conventional shard names expand to all of their shards. Use semicolons only to list
    /// disjoint shards manually.
    #[arg(long)]
    pub from_uqff: Option<String>,

    /// ISQ organization strategy: default or moqe
    #[arg(long)]
    pub isq_organization: Option<IsqOrganization>,

    /// imatrix file for enhanced quantization
    #[arg(long)]
    pub imatrix: Option<PathBuf>,

    /// Calibration file for imatrix generation
    #[arg(long, conflicts_with = "imatrix")]
    pub calibration_file: Option<PathBuf>,
}

/// Device and compute options
#[derive(Args, Clone, Default, Deserialize)]
pub struct DeviceOptions {
    /// Force CPU-only execution
    #[arg(long)]
    #[serde(default)]
    pub cpu: bool,

    /// Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20")
    /// Omit for automatic device mapping
    #[arg(short = 'n', long, value_delimiter = ';')]
    pub device_layers: Option<Vec<String>>,

    /// Topology YAML file for device mapping
    #[arg(long)]
    pub topology: Option<PathBuf>,

    /// Custom Hugging Face cache directory
    #[arg(long)]
    pub hf_cache: Option<PathBuf>,

    /// Max sequence length for automatic device mapping
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// Max batch size for automatic device mapping
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
}

/// Multimodal model specific options
#[derive(Args, Clone, Default, Deserialize)]
pub struct MultimodalOptions {
    /// Maximum logical tensor memory retained by the multimodal encoder cache, in MiB
    #[arg(long = "encoder-cache-memory-mb")]
    pub encoder_cache_memory_mb: Option<NonZeroUsize>,

    /// Maximum edge length for image resizing (aspect ratio preserved)
    #[arg(long)]
    pub max_edge: Option<u32>,

    /// Maximum number of images per request
    #[arg(long)]
    pub max_num_images: Option<usize>,

    /// Maximum image dimension for device mapping
    #[arg(long)]
    pub max_image_length: Option<usize>,
}

fn parse_arch(s: &str) -> Result<NormalLoaderType, String> {
    s.parse()
}

fn parse_dtype(s: &str) -> Result<ModelDType, String> {
    s.parse()
}

fn default_gqa() -> usize {
    1
}

fn default_max_seq_len() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN
}

fn default_max_batch_size() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::*;

    #[derive(Parser)]
    struct AdapterCli {
        #[command(flatten)]
        adapter: AdapterOptions,
    }

    #[test]
    fn parses_explicit_lora_alias_and_source() {
        let adapter = parse_lora_adapter("code=org/adapter=revision").unwrap();
        assert_eq!(adapter.alias, "code");
        assert_eq!(adapter.source, "org/adapter=revision");
        assert_eq!(adapter.revision(), "main");
        assert_eq!(adapter.revision, None);
    }

    #[test]
    fn parses_vllm_lora_module_json() {
        let adapter = parse_lora_adapter(
            r#"{"name":"code","path":"org/code-lora","revision":"refs/pr/7","base_model_name":"org/base","is_3d_lora_weight":false}"#,
        )
        .unwrap();
        assert_eq!(adapter.alias, "code");
        assert_eq!(adapter.source, "org/code-lora");
        assert_eq!(adapter.revision(), "refs/pr/7");
        assert_eq!(adapter.base_model_name.as_deref(), Some("org/base"));
    }

    #[test]
    fn vllm_cli_aliases_are_visible_and_repeatable() {
        let cli = AdapterCli::try_parse_from([
            "test",
            "--lora",
            "code=org/code-lora",
            "--lora-modules",
            r#"{"name":"math","path":"org/math-lora"}"#,
            "--max-lora-rank",
            "64",
        ])
        .unwrap();

        assert_eq!(cli.adapter.lora.len(), 2);
        assert_eq!(cli.adapter.lora[0].alias, "code");
        assert_eq!(cli.adapter.lora[1].alias, "math");
        assert_eq!(cli.adapter.lora_max_rank, 64);

        let help = AdapterCli::command().render_long_help().to_string();
        assert!(help.contains("--lora-modules"));
        assert!(help.contains("--max-lora-rank"));
        assert!(!help.contains("--max-loras"));
    }

    #[test]
    fn vllm_cli_accepts_multiple_modules_after_one_option() {
        let cli = AdapterCli::try_parse_from([
            "test",
            "--lora-modules",
            "code=org/code-lora",
            "math=org/math-lora",
        ])
        .unwrap();

        assert_eq!(cli.adapter.lora.len(), 2);
        assert_eq!(cli.adapter.lora[0].alias, "code");
        assert_eq!(cli.adapter.lora[1].alias, "math");
    }

    #[test]
    fn vllm_3d_lora_modules_are_accepted() {
        let adapter =
            parse_lora_adapter(r#"{"name":"moe","path":"org/moe-lora","is_3d_lora_weight":true}"#)
                .unwrap();
        assert_eq!(adapter.alias, "moe");
        assert_eq!(adapter.source, "org/moe-lora");
    }

    #[test]
    fn vllm_base_model_name_is_lineage_metadata() {
        let options = AdapterOptions {
            lora: vec![
                LoraAdapterSpec::new("code", "org/code-lora").with_base_model_name("org/expected")
            ],
            ..AdapterOptions::default()
        };
        options.validate().unwrap();
    }

    #[test]
    fn invalid_vllm_lora_module_json_is_rejected() {
        let error = parse_lora_adapter(r#"{"name":"code"}"#).unwrap_err();
        assert!(error.contains("missing field `path`"));
    }

    #[test]
    fn parses_human_readable_lora_byte_limits() {
        assert_eq!(parse_lora_bytes("8GiB").unwrap(), 8 * GIBIBYTE);
        assert_eq!(parse_lora_bytes("4GB").unwrap(), 4 * GIGABYTE);
        assert!(parse_lora_bytes("1TiB").is_err());
    }

    #[test]
    fn empty_dynamic_runtime_is_explicitly_enabled() {
        let options = AdapterOptions {
            enable_lora: true,
            ..AdapterOptions::default()
        };
        assert!(options.dynamic_lora_enabled());
        assert!(options.lora.is_empty());
    }

    #[test]
    fn duplicate_lora_aliases_are_rejected() {
        let options = AdapterOptions {
            lora: vec![
                LoraAdapterSpec::new("code", "first"),
                LoraAdapterSpec::new("code", "second"),
            ],
            ..AdapterOptions::default()
        };
        assert!(options.validate().unwrap_err().contains("more than once"));
    }

    #[test]
    fn oversized_lora_alias_is_rejected() {
        let options = AdapterOptions {
            lora: vec![LoraAdapterSpec::new(
                "a".repeat(MAX_LORA_ALIAS_BYTES + 1),
                "org/code-lora",
            )],
            ..AdapterOptions::default()
        };
        assert!(options.validate().unwrap_err().contains("must not exceed"));
    }
}
