//! Model-related argument structs

use clap::{Args, ValueEnum};
use mistralrs_core::{
    AutoDeviceMapParams, IsqOrganization, LoraAdapterSpec, LoraRuntimeConfig, ModelDType,
    NormalLoaderType, DEFAULT_LORA_MAX_ADAPTERS, DEFAULT_LORA_MAX_BYTES, DEFAULT_LORA_MAX_RANK,
    MAX_LORA_ALIAS_BYTES,
};
use serde::Deserialize;
use std::{collections::HashSet, path::PathBuf};

const KILOBYTE: u64 = 1_000;
const MEGABYTE: u64 = 1_000_000;
const GIGABYTE: u64 = 1_000_000_000;
const KIBIBYTE: u64 = 1 << 10;
const MEBIBYTE: u64 = 1 << 20;
const GIBIBYTE: u64 = 1 << 30;

/// Model source options
#[derive(Args, Clone, Deserialize)]
pub struct ModelSourceOptions {
    /// HuggingFace model ID or local path to model directory
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
}

/// Format options for model loading
#[derive(Args, Clone, Default, Deserialize)]
pub struct FormatOptions {
    /// Model format: plain (safetensors), gguf, or ggml
    /// Auto-detected if not specified
    #[arg(long, value_enum)]
    pub format: Option<ModelFormat>,

    /// Quantized model filename(s) for GGUF/GGML (semicolon-separated for multiple)
    #[arg(short = 'f', long)]
    pub quantized_file: Option<String>,

    /// Model ID for tokenizer when using quantized format
    #[arg(long)]
    pub tok_model_id: Option<String>,

    /// GQA value for GGML models
    #[arg(long, default_value_t = 1)]
    #[serde(default = "default_gqa")]
    pub gqa: usize,
}

/// Model format type
#[derive(Clone, Copy, ValueEnum, Default, Deserialize)]
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

/// Adapter options (LoRA/X-LoRA)
#[derive(Args, Clone, Deserialize)]
pub struct AdapterOptions {
    /// Enable dynamic LoRA without preloading an adapter. Supports text models. Qwen3.5/3.6 MoE
    /// requires automatic model selection; vision-tower adapters are unsupported.
    #[arg(long, conflicts_with = "xlora")]
    #[serde(default)]
    pub enable_lora: bool,

    /// Preload a language-model LoRA adapter as ALIAS=SOURCE. Remote adapters use revision main. May
    /// be repeated. Qwen3.5/3.6 MoE conditional-generation models require auto model selection;
    /// vision-tower adapters are unsupported.
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

    /// Legacy LoRA adapter source for a raw GGUF or GGML model
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
    /// Quantization front-door: accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or raw quant names (`q4k`, `q8_0`, etc.)
    /// This prefers prebuilt UQFF from `mistralrs-community/<model>-UQFF`, so use `--isq` if you do not want to switch to a prebuilt UQFF.
    #[arg(long, conflicts_with_all = ["in_situ_quant", "from_uqff"])]
    pub quant: Option<String>,

    /// In-situ quantization: accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or raw quant names (`q4k`, `q8_0`, etc.) and quantizes the selected model in-place (in-situ)
    #[arg(long = "isq")]
    #[serde(rename = "isq", alias = "in_situ_quant")]
    pub in_situ_quant: Option<String>,

    /// UQFF file(s) to load from. Accepts numeric shorthands (2, 3, 4, 5, 6, 8)
    /// to auto-detect the appropriate UQFF file (e.g., `--from-uqff 8` finds
    /// q8_0-0.uqff or afq8-0.uqff). Also accepts ISQ type names (e.g., q4k, afq8).
    /// Shards are auto-discovered: specifying the first shard (e.g., q4k-0.uqff)
    /// automatically finds q4k-1.uqff, etc. Use semicolons to separate different
    /// quantizations.
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

    /// Custom HuggingFace cache directory
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
