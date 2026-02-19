//! Model-related argument structs

use clap::{Args, ValueEnum};
use mistralrs_core::{AutoDeviceMapParams, IsqOrganization, ModelDType, NormalLoaderType};
use serde::Deserialize;
use std::path::PathBuf;

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
#[derive(Args, Clone, Default, Deserialize)]
pub struct AdapterOptions {
    /// LoRA adapter model ID(s), semicolon-separated for multiple
    #[arg(long)]
    pub lora: Option<String>,

    /// X-LoRA adapter model ID
    #[arg(long, conflicts_with = "lora")]
    pub xlora: Option<String>,

    /// X-LoRA ordering JSON file
    #[arg(long, requires = "xlora")]
    pub xlora_order: Option<PathBuf>,

    /// Target non-granular index for X-LoRA
    #[arg(long, requires = "xlora")]
    pub tgt_non_granular_index: Option<usize>,
}

/// Quantization options
#[derive(Args, Clone, Default, Deserialize)]
pub struct QuantizationOptions {
    /// In-situ quantization level (e.g., "4", "8", "q4_0", "q4_1", etc.)
    #[arg(long = "isq")]
    pub in_situ_quant: Option<String>,

    /// UQFF file(s) to load from. Shards are auto-discovered: specifying the first
    /// shard (e.g., q4k-0.uqff) automatically finds q4k-1.uqff, etc. Use semicolons
    /// to separate different quantizations (e.g., "q4k-0.uqff;q8_0-0.uqff").
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

/// Vision model specific options
#[derive(Args, Clone, Default, Deserialize)]
pub struct VisionOptions {
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
