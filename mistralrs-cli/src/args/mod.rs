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
        model_type: ModelType,

        #[command(flatten)]
        server: ServerOptions,

        #[command(flatten)]
        runtime: RuntimeOptions,
    },

    /// Run model in interactive mode
    Run {
        #[command(subcommand)]
        model_type: ModelType,

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
        model_type: QuantizeModelType,
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
        model_type: ModelType,

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

    /// Run from a full TOML configuration file
    #[command(name = "from-config")]
    FromConfig {
        /// Path to configuration file (.toml)
        #[arg(short, long)]
        file: PathBuf,
    },
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
