//! CLI argument definitions for mistralrs-cli
//!
//! This module provides cleanly organized argument structs using clap's derive macros.
//! Arguments are grouped logically to improve discoverability and reduce duplication.

mod model;
mod paged_attn;
mod quantize;
mod server;

pub use model::*;
pub use paged_attn::*;
pub use quantize::*;
pub use server::*;

use clap::{Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use mistralrs_core::TokenSource;
use serde::Deserialize;
use std::path::PathBuf;

/// Fast LLM inference engine
#[derive(Parser)]
#[command(name = "mistralrs")]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    #[command(flatten)]
    pub global: GlobalOptions,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start HTTP/MCP server and (optionally) the UI at /ui
    Serve {
        #[command(subcommand)]
        model_type: Option<ModelType>,

        /// Default model options (used when model type is not specified)
        #[command(flatten)]
        default_model: DefaultModelOptions,

        #[command(flatten)]
        server: ServerOptions,

        #[command(flatten)]
        runtime: RuntimeOptions,
    },

    /// Run model in interactive mode
    Run {
        #[command(subcommand)]
        model_type: Option<ModelType>,

        /// Default model options (used when model type is not specified)
        #[command(flatten)]
        default_model: DefaultModelOptions,

        #[command(flatten)]
        runtime: RuntimeOptions,

        /// Enable thinking mode for models that support it
        #[arg(long)]
        enable_thinking: bool,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },

    /// Generate UQFF quantized model file
    Quantize {
        #[command(subcommand)]
        model_type: Option<QuantizeModelType>,

        /// Default quantize options (used when model type is not specified)
        #[command(flatten)]
        default_quantize: QuantizeDefaultOptions,
    },

    /// Run system diagnostics and environment checks
    Doctor {
        /// Output JSON instead of human-readable text
        #[arg(long)]
        json: bool,
    },

    /// Recommend quantization + device mapping for a model
    Tune {
        #[command(subcommand)]
        model_type: Option<ModelType>,

        /// Default model options (used when model type is not specified)
        #[command(flatten)]
        default_model: DefaultModelOptions,

        /// Tuning profile (quality, balanced, fast)
        #[arg(long, value_enum, default_value = "balanced")]
        profile: TuneProfileArg,

        /// Output JSON instead of human-readable text
        #[arg(long)]
        json: bool,

        /// Emit a TOML config file with the recommended settings
        #[arg(long)]
        emit_config: Option<PathBuf>,
    },

    /// Authenticate with HuggingFace Hub
    Login {
        /// Provide token directly (non-interactive)
        #[arg(long)]
        token: Option<String>,
    },

    /// Manage the HuggingFace model cache
    Cache {
        #[command(subcommand)]
        cmd: CacheCommand,
    },

    /// Run performance benchmarks
    Bench {
        #[command(subcommand)]
        model_type: Option<ModelType>,

        /// Default model options (used when model type is not specified)
        #[command(flatten)]
        default_model: DefaultModelOptions,

        #[command(flatten)]
        runtime: RuntimeOptions,

        /// Number of tokens in prompt
        #[arg(long, default_value = "512")]
        prompt_len: usize,

        /// Number of tokens to generate
        #[arg(long, default_value = "128")]
        gen_len: usize,

        /// Number of benchmark iterations
        #[arg(long, default_value = "3")]
        iterations: usize,

        /// Number of warmup runs (discarded)
        #[arg(long, default_value = "1")]
        warmup: usize,
    },

    /// Run from a full TOML configuration file
    #[command(name = "from-config")]
    FromConfig {
        /// Path to configuration file (.toml)
        #[arg(short, long)]
        file: PathBuf,
    },
}

/// Cache management subcommands
#[derive(Subcommand, Clone)]
pub enum CacheCommand {
    /// List all cached models
    List,

    /// Delete a specific model from cache
    Delete {
        /// Model ID (e.g., "Qwen/Qwen3-4B")
        #[arg(short = 'm', long)]
        model_id: String,
    },
}

/// Default model options used when no model type subcommand is specified.
/// These mirror the Auto variant's options and are used to construct ModelType::Auto.
#[derive(clap::Args, Clone, Default)]
pub struct DefaultModelOptions {
    /// HuggingFace model ID or local path to model directory
    #[arg(short = 'm', long)]
    pub model_id: Option<String>,

    /// Path to local tokenizer.json file
    #[arg(short = 't', long)]
    pub tokenizer: Option<PathBuf>,

    /// Model architecture (auto-detected if not specified)
    #[arg(short = 'a', long, value_parser = parse_arch)]
    pub arch: Option<mistralrs_core::NormalLoaderType>,

    /// Model data type
    #[arg(long, default_value = "auto", value_parser = parse_dtype)]
    pub dtype: mistralrs_core::ModelDType,

    #[command(flatten)]
    pub format: FormatOptions,

    #[command(flatten)]
    pub adapter: AdapterOptions,

    #[command(flatten)]
    pub quantization: QuantizationOptions,

    #[command(flatten)]
    pub device: DeviceOptions,

    #[command(flatten)]
    pub cache: CacheOptions,

    #[command(flatten)]
    pub vision: VisionOptions,
}

impl DefaultModelOptions {
    /// Convert default options into a ModelType::Auto variant.
    /// Returns an error if model_id is not provided.
    pub fn into_model_type(self) -> anyhow::Result<ModelType> {
        let model_id = self
            .model_id
            .ok_or_else(|| anyhow::anyhow!("--model-id (-m) is required"))?;
        Ok(ModelType::Auto {
            model: ModelSourceOptions {
                model_id,
                tokenizer: self.tokenizer,
                arch: self.arch,
                dtype: self.dtype,
            },
            format: self.format,
            adapter: self.adapter,
            quantization: self.quantization,
            device: self.device,
            cache: self.cache,
            vision: self.vision,
        })
    }
}

/// Get the effective ModelType, using default options if no subcommand was provided.
/// Returns an error if no subcommand is provided and model_id is missing.
pub fn resolve_model_type(
    model_type: Option<ModelType>,
    default_model: DefaultModelOptions,
) -> anyhow::Result<ModelType> {
    match model_type {
        Some(mt) => Ok(mt),
        None => default_model.into_model_type(),
    }
}

fn parse_arch(s: &str) -> Result<mistralrs_core::NormalLoaderType, String> {
    s.parse()
}

fn parse_dtype(s: &str) -> Result<mistralrs_core::ModelDType, String> {
    s.parse()
}

/// Model type selection
#[derive(Subcommand, Clone)]
pub enum ModelType {
    /// Auto-detect model type (recommended)
    Auto {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        format: FormatOptions,

        #[command(flatten)]
        adapter: AdapterOptions,

        #[command(flatten)]
        quantization: QuantizationOptions,

        #[command(flatten)]
        device: DeviceOptions,

        #[command(flatten)]
        cache: CacheOptions,

        #[command(flatten)]
        vision: VisionOptions,
    },

    /// Text generation model with explicit configuration
    Text {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        format: FormatOptions,

        #[command(flatten)]
        adapter: AdapterOptions,

        #[command(flatten)]
        quantization: QuantizationOptions,

        #[command(flatten)]
        device: DeviceOptions,

        #[command(flatten)]
        cache: CacheOptions,
    },

    /// Vision-language model
    Vision {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        format: FormatOptions,

        #[command(flatten)]
        adapter: AdapterOptions,

        #[command(flatten)]
        quantization: QuantizationOptions,

        #[command(flatten)]
        device: DeviceOptions,

        #[command(flatten)]
        cache: CacheOptions,

        #[command(flatten)]
        vision: VisionOptions,
    },

    /// Image generation model (diffusion)
    Diffusion {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        device: DeviceOptions,
    },

    /// Speech synthesis model
    Speech {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        device: DeviceOptions,
    },

    /// Embedding model
    Embedding {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        format: FormatOptions,

        #[command(flatten)]
        quantization: QuantizationOptions,

        #[command(flatten)]
        device: DeviceOptions,

        #[command(flatten)]
        cache: CacheOptions,
    },
}

/// Global options that apply to all commands
#[derive(clap::Args, Clone, Deserialize)]
pub struct GlobalOptions {
    /// Random seed for reproducibility
    #[arg(long, global = true)]
    #[serde(default)]
    pub seed: Option<u64>,

    /// Log all requests and responses to this file
    #[arg(long, short, global = true)]
    #[serde(default)]
    pub log: Option<PathBuf>,

    /// Token source for HuggingFace authentication.
    /// Formats: `literal:<token>`, `env:<var>`, `path:<file>`, `cache`, `none`
    #[arg(long, default_value = "cache", global = true, value_parser = parse_token_source)]
    #[serde(default = "default_token_source")]
    pub token_source: TokenSource,
}

/// Runtime options for inference
#[derive(clap::Args, Clone, Deserialize)]
pub struct RuntimeOptions {
    /// Maximum concurrent sequences
    #[arg(long, default_value_t = 32)]
    #[serde(default = "default_max_seqs")]
    pub max_seqs: usize,

    /// Disable KV cache entirely
    #[arg(long)]
    #[serde(default)]
    pub no_kv_cache: bool,

    /// Number of prefix caches to hold (0 to disable)
    #[arg(long, default_value_t = 16)]
    #[serde(default = "default_prefix_cache_n")]
    pub prefix_cache_n: usize,

    /// Custom chat template file (.json or .jinja)
    #[arg(long, short)]
    #[serde(default)]
    pub chat_template: Option<PathBuf>,

    /// Explicit JINJA template override
    #[arg(long, short)]
    #[serde(default)]
    pub jinja_explicit: Option<PathBuf>,

    /// Enable web search (requires embedding model)
    #[arg(long)]
    #[serde(default)]
    pub enable_search: bool,

    /// Search embedding model to use
    #[arg(long, requires = "enable_search")]
    #[serde(default)]
    pub search_embedding_model: Option<SearchEmbeddingModelArg>,
}

/// Search embedding model options
#[derive(Clone, Copy, ValueEnum, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchEmbeddingModelArg {
    EmbeddingGemma,
}

/// Tuning profile options
#[derive(Clone, Copy, ValueEnum)]
pub enum TuneProfileArg {
    Quality,
    Balanced,
    Fast,
}

impl From<TuneProfileArg> for mistralrs_core::TuneProfile {
    fn from(value: TuneProfileArg) -> Self {
        match value {
            TuneProfileArg::Quality => mistralrs_core::TuneProfile::Quality,
            TuneProfileArg::Balanced => mistralrs_core::TuneProfile::Balanced,
            TuneProfileArg::Fast => mistralrs_core::TuneProfile::Fast,
        }
    }
}

impl From<SearchEmbeddingModelArg> for mistralrs_core::SearchEmbeddingModel {
    fn from(value: SearchEmbeddingModelArg) -> Self {
        match value {
            SearchEmbeddingModelArg::EmbeddingGemma => {
                mistralrs_core::SearchEmbeddingModel::EmbeddingGemma300M
            }
        }
    }
}

impl Default for GlobalOptions {
    fn default() -> Self {
        Self {
            seed: None,
            log: None,
            token_source: TokenSource::CacheToken,
        }
    }
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            max_seqs: 32,
            no_kv_cache: false,
            prefix_cache_n: 16,
            chat_template: None,
            jinja_explicit: None,
            enable_search: false,
            search_embedding_model: None,
        }
    }
}

fn parse_token_source(s: &str) -> Result<TokenSource, String> {
    s.parse()
}

fn default_token_source() -> TokenSource {
    TokenSource::CacheToken
}

fn default_max_seqs() -> usize {
    32
}

fn default_prefix_cache_n() -> usize {
    16
}
