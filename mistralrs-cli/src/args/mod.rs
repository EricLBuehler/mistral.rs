//! CLI argument definitions for mistralrs-cli
//!
//! This module provides cleanly organized argument structs using clap's derive macros.
//! Arguments are grouped logically to improve discoverability and reduce duplication.

mod model;
mod paged_attn;
mod server;

pub use model::*;
pub use paged_attn::*;
pub use server::*;

use clap::{Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use mistralrs_core::TokenSource;
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
    /// Start HTTP/MCP server with a model
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

    /// Load model configuration from TOML or JSON file
    Config {
        /// Path to configuration file (.toml or .json)
        #[arg(short, long)]
        config: PathBuf,
    },
}

/// Global options that apply to all commands
#[derive(clap::Args, Clone)]
pub struct GlobalOptions {
    /// Random seed for reproducibility
    #[arg(long, global = true)]
    pub seed: Option<u64>,

    /// Log all requests and responses to this file
    #[arg(long, short, global = true)]
    pub log: Option<PathBuf>,

    /// Token source for HuggingFace authentication.
    /// Formats: `literal:<token>`, `env:<var>`, `path:<file>`, `cache`, `none`
    #[arg(long, default_value = "cache", global = true, value_parser = parse_token_source)]
    pub token_source: TokenSource,
}

/// Runtime options for inference
#[derive(clap::Args, Clone)]
pub struct RuntimeOptions {
    /// Maximum concurrent sequences
    #[arg(long, default_value_t = 32)]
    pub max_seqs: usize,

    /// Disable KV cache entirely
    #[arg(long)]
    pub no_kv_cache: bool,

    /// Number of prefix caches to hold (0 to disable)
    #[arg(long, default_value_t = 16)]
    pub prefix_cache_n: usize,

    /// Custom chat template file (.json or .jinja)
    #[arg(long, short)]
    pub chat_template: Option<PathBuf>,

    /// Explicit JINJA template override
    #[arg(long, short)]
    pub jinja_explicit: Option<PathBuf>,

    /// Enable web search (requires embedding model)
    #[arg(long)]
    pub enable_search: bool,

    /// Search embedding model to use
    #[arg(long, requires = "enable_search")]
    pub search_embedding_model: Option<SearchEmbeddingModelArg>,
}

/// Search embedding model options
#[derive(Clone, Copy, ValueEnum)]
pub enum SearchEmbeddingModelArg {
    EmbeddingGemma,
}

fn parse_token_source(s: &str) -> Result<TokenSource, String> {
    s.parse()
}
