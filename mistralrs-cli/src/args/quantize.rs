//! Quantize command argument structs for UQFF generation

use clap::{Args, Subcommand};
use mistralrs_core::{IsqOrganization, ModelDType, NormalLoaderType};
use std::path::PathBuf;

/// Quantize model type selection (base models only, no adapter support)
#[derive(Subcommand, Clone)]
pub enum QuantizeModelType {
    /// Auto-detect model type (recommended)
    Auto {
        #[command(flatten)]
        model: QuantizeModelSourceOptions,

        #[command(flatten)]
        quantization: QuantizeQuantizationOptions,

        #[command(flatten)]
        device: QuantizeDeviceOptions,

        #[command(flatten)]
        output: QuantizeOutputOptions,

        #[command(flatten)]
        vision: QuantizeVisionOptions,
    },

    /// Text generation model with explicit architecture
    Text {
        #[command(flatten)]
        model: QuantizeModelSourceOptions,

        /// Model architecture (required for text models)
        #[arg(short = 'a', long, value_parser = parse_arch)]
        arch: Option<NormalLoaderType>,

        #[command(flatten)]
        quantization: QuantizeQuantizationOptions,

        #[command(flatten)]
        device: QuantizeDeviceOptions,

        #[command(flatten)]
        output: QuantizeOutputOptions,
    },

    /// Vision-language model
    Vision {
        #[command(flatten)]
        model: QuantizeModelSourceOptions,

        #[command(flatten)]
        quantization: QuantizeQuantizationOptions,

        #[command(flatten)]
        device: QuantizeDeviceOptions,

        #[command(flatten)]
        output: QuantizeOutputOptions,

        #[command(flatten)]
        vision: QuantizeVisionOptions,
    },

    /// Embedding model
    Embedding {
        #[command(flatten)]
        model: QuantizeModelSourceOptions,

        #[command(flatten)]
        quantization: QuantizeQuantizationOptions,

        #[command(flatten)]
        device: QuantizeDeviceOptions,

        #[command(flatten)]
        output: QuantizeOutputOptions,
    },
}

/// Model source options for quantization
#[derive(Args, Clone)]
pub struct QuantizeModelSourceOptions {
    /// Model ID to load (HuggingFace repo or local path)
    #[arg(short = 'm', long)]
    pub model_id: String,

    /// Path to local tokenizer.json file
    #[arg(short = 't', long)]
    pub tokenizer: Option<PathBuf>,

    /// Model data type
    #[arg(long, default_value = "auto", value_parser = parse_dtype)]
    pub dtype: ModelDType,
}

/// Quantization options for UQFF generation (ISQ-related only, no from_uqff)
#[derive(Args, Clone)]
pub struct QuantizeQuantizationOptions {
    /// In-situ quantization level (e.g., "4", "8", "q4_0", "q4_1", "q4k", etc.)
    #[arg(long = "isq", required = true)]
    pub in_situ_quant: String,

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

/// Device options for quantization
#[derive(Args, Clone)]
pub struct QuantizeDeviceOptions {
    /// Force CPU-only execution
    #[arg(long)]
    pub cpu: bool,

    /// Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20")
    #[arg(short = 'n', long, value_delimiter = ';')]
    pub device_layers: Option<Vec<String>>,

    /// Topology YAML file for device mapping
    #[arg(long)]
    pub topology: Option<PathBuf>,

    /// Custom HuggingFace cache directory
    #[arg(long)]
    pub hf_cache: Option<PathBuf>,

    /// Max sequence length for automatic device mapping
    #[arg(long, default_value_t = 4096)]
    pub max_seq_len: usize,

    /// Max batch size for automatic device mapping
    #[arg(long, default_value_t = 128)]
    pub max_batch_size: usize,
}

/// Output options for UQFF generation
#[derive(Args, Clone)]
pub struct QuantizeOutputOptions {
    /// Output path for the UQFF file
    #[arg(short = 'o', long = "output", required = true)]
    pub output_path: PathBuf,
}

/// Vision model options for quantization
#[derive(Args, Clone, Default)]
pub struct QuantizeVisionOptions {
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
