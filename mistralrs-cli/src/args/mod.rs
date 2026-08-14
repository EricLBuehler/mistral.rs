//! CLI argument definitions for mistralrs-cli
//!
//! This module provides cleanly organized argument structs using clap's derive macros.
//! Arguments are grouped logically to improve discoverability and reduce duplication.

mod model;
mod paged_attn;
mod quantize;
mod sandbox;
mod server;

pub use model::*;
pub use paged_attn::*;
pub use quantize::*;
pub use sandbox::*;
pub use server::*;

use clap::{Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use mistralrs_core::{ReasoningEffort, TokenSource};
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

        #[command(flatten)]
        agent_options: AgentCliOptions,

        #[command(flatten)]
        sandbox: SandboxOptions,
    },

    /// Run model in interactive mode, or one-shot mode with `-i`
    Run {
        #[command(subcommand)]
        model_type: Option<ModelType>,

        /// Default model options (used when model type is not specified)
        #[command(flatten)]
        default_model: DefaultModelOptions,

        #[command(flatten)]
        runtime: RuntimeOptions,

        #[command(flatten)]
        agent_options: AgentCliOptions,

        #[command(flatten)]
        sandbox: SandboxOptions,

        /// Control thinking mode for models that support it.
        /// Use --thinking or --thinking true to force on, --thinking false to force off.
        /// If both reasoning controls are omitted, effort is unspecified and thinking is enabled.
        #[arg(long, num_args = 0..=1, default_missing_value = "true", value_parser = clap::value_parser!(bool))]
        thinking: Option<bool>,

        /// Set reasoning effort without changing the model's sampling parameters.
        /// Values are off, low, medium, high, or xhigh. "none" is an alias for off.
        #[arg(long)]
        reasoning_effort: Option<ReasoningEffort>,

        /// One-shot text prompt. When provided, sends a single request and exits
        /// instead of entering interactive mode.
        /// Combine with --image, --video, or --audio for multimodal requests.
        #[arg(short = 'i', long)]
        input: Option<String>,

        /// Image URL(s) or file path(s) to include in the request (requires -i).
        /// Can be specified multiple times: --image img1.jpg --image img2.png
        #[arg(long, requires = "input")]
        image: Vec<String>,

        /// Video URL(s) or file path(s) to include in the request (requires -i).
        /// Can be specified multiple times: --video vid1.mp4 --video vid2.webm
        #[arg(long, requires = "input")]
        video: Vec<String>,

        /// Audio URL(s) or file path(s) to include in the request (requires -i).
        /// Can be specified multiple times: --audio audio1.wav --audio audio2.mp3
        #[arg(long, requires = "input")]
        audio: Vec<String>,

        /// LoRA adapter alias to use for requests. Omit to run the base model.
        #[arg(long)]
        adapter: Option<String>,
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

    /// Inspect, report, or verify UQFF artifacts
    Uqff {
        #[command(subcommand)]
        command: UqffCommand,
    },

    /// Run system diagnostics and environment checks
    Doctor {
        /// Output JSON instead of human-readable text
        #[arg(long)]
        json: bool,
    },

    /// Recommend quantization + device mapping for a model.
    /// Rejects `--quant auto`; pass `--quant <level>` or `--isq <level>` to bias
    /// the recommendation toward a specific quantization target. Adapter options
    /// are rejected because adapter memory is not included in the estimate.
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

    /// Authenticate with Hugging Face Hub
    Login {
        /// Provide token directly (non-interactive)
        #[arg(long)]
        token: Option<String>,
    },

    /// Manage the Hugging Face model cache
    Cache {
        #[command(subcommand)]
        cmd: CacheCommand,
    },

    /// Run performance benchmarks for base or LoRA model generation.
    Bench {
        #[command(subcommand)]
        model_type: Option<ModelType>,

        /// Default model options (used when model type is not specified)
        #[command(flatten)]
        default_model: DefaultModelOptions,

        #[command(flatten)]
        runtime: BenchRuntimeOptions,

        /// LoRA adapter alias to benchmark. Omit to benchmark the base model.
        #[arg(long)]
        adapter: Option<String>,

        /// Input lengths used to measure time to first token. Zero skips TTFT. Accepts comma-separated values for sweeps.
        #[arg(long, value_delimiter = ',', default_value = "512")]
        prompt_len: Vec<usize>,

        /// Output tokens per decode request. Values below 2 skip decode metrics.
        #[arg(long, default_value = "128")]
        gen_len: usize,

        /// Input context lengths used to measure decode TPOT. Accepts comma-separated values for sweeps.
        #[arg(long, value_delimiter = ',', default_value = "4")]
        depth: Vec<usize>,

        /// Number of benchmark iterations
        #[arg(long, default_value = "3")]
        iterations: usize,

        /// Number of warmup runs per benchmark case (discarded)
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

    /// Update or migrate an install using the installer
    Update {
        /// Install a specific release tag instead of the latest (e.g. v0.9.1)
        #[arg(long)]
        tag: Option<String>,
    },

    /// Remove an installer-managed install
    Uninstall {
        /// Skip the confirmation prompt
        #[arg(short, long)]
        yes: bool,
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

#[derive(Subcommand, Clone)]
pub enum UqffCommand {
    /// Print or write a UQFF report
    Report {
        /// Hugging Face model ID or local path containing UQFF files
        #[arg(short = 'm', long)]
        model_id: String,

        /// Quant group to inspect, such as 3, q3k, afq3, or all
        #[arg(long)]
        quant: Option<String>,

        /// Hugging Face revision to use
        #[arg(long)]
        revision: Option<String>,

        /// Write uqff_report.json beside the artifacts
        #[arg(long)]
        write: bool,

        /// Print JSON instead of human-readable text
        #[arg(long)]
        json: bool,

        /// Base model ID to include in a written report
        #[arg(long)]
        base_model: Option<String>,

        /// Hugging Face repo ID to include in a written report
        #[arg(long)]
        repo_id: Option<String>,
    },

    /// Validate UQFF artifact structure
    Verify {
        /// Hugging Face model ID or local path containing UQFF files
        #[arg(short = 'm', long)]
        model_id: String,

        /// Quant group to inspect, such as 3, q3k, afq3, or all
        #[arg(long)]
        quant: Option<String>,

        /// Hugging Face revision to use
        #[arg(long)]
        revision: Option<String>,

        /// Print JSON instead of human-readable text
        #[arg(long)]
        json: bool,

        /// Fail on missing report/producer metadata or fallback layers
        #[arg(long)]
        strict: bool,

        /// Allow same-major UQFF files with a newer minor version
        #[arg(long)]
        allow_newer_minor: bool,
    },

    /// Open a UQFF-aware tensor explorer
    Inspect {
        /// Hugging Face model ID or local path containing UQFF files
        #[arg(short = 'm', long)]
        model_id: String,

        /// Quant group to inspect, such as 3, q3k, afq3, or all
        #[arg(long)]
        quant: Option<String>,

        /// Hugging Face revision to use
        #[arg(long)]
        revision: Option<String>,
    },
}

/// Default model options used when no model type subcommand is specified.
/// These mirror the Auto variant's options and are used to construct ModelType::Auto.
#[derive(clap::Args, Clone, Default)]
pub struct DefaultModelOptions {
    /// Hugging Face model ID or local model directory; optional when `-f` names local files
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
    pub multimodal: MultimodalOptions,
}

impl DefaultModelOptions {
    /// Convert default options into a ModelType::Auto variant.
    /// Returns an error if neither model_id nor a local quantized file is provided.
    pub fn into_model_type(mut self) -> anyhow::Result<ModelType> {
        self.format.normalize()?;
        let model_id = match self.model_id {
            Some(model_id) => model_id,
            None => self.format.derive_local_model_root()?,
        };
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
            multimodal: self.multimodal,
        })
    }
}

/// Get the effective ModelType, using default options if no subcommand was provided.
/// Returns an error if no subcommand is provided and model_id is missing.
pub fn resolve_model_type(
    model_type: Option<ModelType>,
    default_model: DefaultModelOptions,
) -> anyhow::Result<ModelType> {
    let mut model_type = match model_type {
        Some(model_type) => model_type,
        None => default_model.into_model_type()?,
    };
    if let Some(format) = model_format_mut(&mut model_type) {
        format.normalize()?;
    }
    Ok(model_type)
}

fn model_format_mut(model_type: &mut ModelType) -> Option<&mut FormatOptions> {
    match model_type {
        ModelType::Auto { format, .. }
        | ModelType::Text { format, .. }
        | ModelType::Multimodal { format, .. }
        | ModelType::Embedding { format, .. } => Some(format),
        ModelType::Diffusion { .. } | ModelType::Speech { .. } => None,
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
        multimodal: MultimodalOptions,
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

    /// Multimodal model
    Multimodal {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        format: FormatOptions,

        #[command(flatten)]
        adapter: MultimodalAdapterOptions,

        #[command(flatten)]
        quantization: QuantizationOptions,

        #[command(flatten)]
        device: DeviceOptions,

        #[command(flatten)]
        cache: CacheOptions,

        #[command(flatten)]
        multimodal: MultimodalOptions,
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

    /// Token source for Hugging Face authentication.
    /// Formats: `literal:<token>`, `env:<var>`, `path:<file>`, `cache`, `none`
    #[arg(long, default_value = "cache", global = true, value_parser = parse_token_source)]
    #[serde(default = "default_token_source")]
    pub token_source: TokenSource,

    /// Increase logging verbosity. Use -v for debug and -vv for trace-level internals.
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count)]
    #[serde(default)]
    pub verbose: u8,
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

    /// Path to a MatFormer config (CSV/JSON describing available slices). See model card.
    #[arg(long)]
    #[serde(default)]
    pub matformer_config_path: Option<PathBuf>,

    /// MatFormer slice to load (must match a slice name in the config file).
    #[arg(long, requires = "matformer_config_path")]
    #[serde(default)]
    pub matformer_slice_name: Option<String>,

    /// MTP assistant model id or path.
    #[arg(long)]
    #[serde(default)]
    pub mtp_model: Option<String>,

    /// Number of MTP draft tokens to propose per target step.
    #[arg(long)]
    #[serde(default)]
    pub mtp_n_predict: Option<usize>,

    /// Path to an MCP client configuration JSON. Also reads `MCP_CONFIG_PATH` if unset.
    #[arg(long)]
    #[serde(default)]
    pub mcp_config: Option<PathBuf>,

    #[arg(skip)]
    #[serde(default)]
    pub agent: bool,

    #[arg(skip)]
    #[serde(default)]
    pub enable_search: bool,

    #[arg(skip)]
    #[serde(default)]
    pub search_embedding_model: Option<SearchEmbeddingModelArg>,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub enable_code_execution: bool,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub enable_shell: bool,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub code_exec_python: Option<PathBuf>,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub code_exec_timeout: Option<u64>,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub code_exec_workdir: Option<PathBuf>,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub shell_path: Option<PathBuf>,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub shell_timeout: Option<u64>,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub shell_workdir: Option<PathBuf>,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub skills_dir: Option<PathBuf>,

    #[arg(skip)]
    #[serde(default, rename = "agent_permission", alias = "code_exec_permission")]
    pub code_exec_permission: CodeExecPermissionArg,
}

#[derive(clap::Args, Clone, Default)]
pub struct AgentCliOptions {
    /// Build a local agent: enables web search, Python code execution, and shell execution, runs the agentic
    /// tool loop with a per-session temp workdir. Equivalent to passing
    /// `--enable-search --enable-code-execution --enable-shell` together.
    #[arg(long, alias = "agentic")]
    pub agent: bool,

    /// Enable web search (requires embedding model)
    #[arg(long)]
    pub enable_search: bool,

    /// Search embedding model to use. Requires `--enable-search` or `--agent`.
    #[arg(long)]
    pub search_embedding_model: Option<SearchEmbeddingModelArg>,

    /// Enable Python code execution tool (WARNING: allows arbitrary code execution)
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub enable_code_execution: bool,

    /// Enable shell execution tool (WARNING: allows arbitrary command execution)
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub enable_shell: bool,

    /// Python interpreter path for code execution. Requires code execution to be on
    /// (via `--enable-code-execution` or `--agent`). Defaults to `python3`.
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub code_exec_python: Option<PathBuf>,

    /// Code execution timeout in seconds (default: 60). Requires code execution to be on.
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub code_exec_timeout: Option<u64>,

    /// Working directory for code execution. Defaults to a temp dir; use "." for cwd.
    /// Requires code execution to be on.
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub code_exec_workdir: Option<PathBuf>,

    /// Shell executable path. Requires shell execution to be on. Defaults to /bin/sh.
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub shell_path: Option<PathBuf>,

    /// Shell execution timeout in seconds (default: 600). Requires shell execution to be on.
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub shell_timeout: Option<u64>,

    /// Root directory for per-session shell working directories. Defaults to temp dirs.
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub shell_workdir: Option<PathBuf>,

    /// Directory for uploaded OpenAI-compatible Skills. Defaults to the system temp directory.
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub skills_dir: Option<PathBuf>,

    /// Agent action permission mode.
    #[arg(long = "agent-permission", alias = "code-exec-permission", value_name = "PERMISSION", value_enum, default_value_t = CodeExecPermissionArg::Auto)]
    pub code_exec_permission: CodeExecPermissionArg,
}

impl AgentCliOptions {
    pub fn apply_to(self, runtime: &mut RuntimeOptions) {
        runtime.agent = self.agent;
        runtime.enable_search = self.enable_search;
        runtime.search_embedding_model = self.search_embedding_model;
        #[cfg(feature = "code-execution")]
        {
            runtime.enable_code_execution = self.enable_code_execution;
            runtime.enable_shell = self.enable_shell;
            runtime.code_exec_python = self.code_exec_python;
            runtime.code_exec_timeout = self.code_exec_timeout;
            runtime.code_exec_workdir = self.code_exec_workdir;
            runtime.shell_path = self.shell_path;
            runtime.shell_timeout = self.shell_timeout;
            runtime.shell_workdir = self.shell_workdir;
            runtime.skills_dir = self.skills_dir;
        }
        runtime.code_exec_permission = self.code_exec_permission;
    }
}

#[derive(clap::Args, Clone, Default)]
pub struct BenchRuntimeOptions {
    /// Disable KV cache entirely
    #[arg(long)]
    pub no_kv_cache: bool,

    /// Path to a MatFormer config (CSV/JSON describing available slices). See model card.
    #[arg(long)]
    pub matformer_config_path: Option<PathBuf>,

    /// MatFormer slice to load (must match a slice name in the config file).
    #[arg(long, requires = "matformer_config_path")]
    pub matformer_slice_name: Option<String>,

    /// MTP assistant model id or path.
    #[arg(long)]
    pub mtp_model: Option<String>,

    /// Number of MTP draft tokens to propose per target step.
    #[arg(long)]
    pub mtp_n_predict: Option<usize>,
}

impl BenchRuntimeOptions {
    pub fn matformer_selection(&self) -> MatformerSelection {
        MatformerSelection {
            config_path: self.matformer_config_path.clone(),
            slice_name: self.matformer_slice_name.clone(),
        }
    }

    pub fn mtp_config(&self) -> Option<mistralrs_core::MtpConfig> {
        self.mtp_model
            .clone()
            .map(|model| mistralrs_core::MtpConfig {
                model,
                n_predict: self.mtp_n_predict,
            })
    }
}

/// Search embedding model options
#[derive(Clone, Copy, ValueEnum, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchEmbeddingModelArg {
    EmbeddingGemma,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CodeExecPermissionArg {
    #[default]
    Auto,
    Ask,
    Deny,
}

/// Tuning profile options
#[derive(Clone, Copy, ValueEnum)]
pub enum TuneProfileArg {
    Quality,
    Balanced,
    Fast,
}

/// Selection of a MatFormer slice (config file + named slice). Used by loaders for
/// models like Gemma 3n that support elastic sizing.
#[derive(Clone, Default)]
pub struct MatformerSelection {
    pub config_path: Option<PathBuf>,
    pub slice_name: Option<String>,
}

impl RuntimeOptions {
    pub fn matformer_selection(&self) -> MatformerSelection {
        MatformerSelection {
            config_path: self.matformer_config_path.clone(),
            slice_name: self.matformer_slice_name.clone(),
        }
    }

    pub fn mtp_config(&self) -> Option<mistralrs_core::MtpConfig> {
        self.mtp_model
            .clone()
            .map(|model| mistralrs_core::MtpConfig {
                model,
                n_predict: self.mtp_n_predict,
            })
    }
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

impl From<CodeExecPermissionArg> for mistralrs_core::CodeExecutionPermission {
    fn from(value: CodeExecPermissionArg) -> Self {
        match value {
            CodeExecPermissionArg::Auto => mistralrs_core::CodeExecutionPermission::Auto,
            CodeExecPermissionArg::Ask => mistralrs_core::CodeExecutionPermission::Ask,
            CodeExecPermissionArg::Deny => mistralrs_core::CodeExecutionPermission::Deny,
        }
    }
}

impl From<CodeExecPermissionArg> for mistralrs_core::AgentPermission {
    fn from(value: CodeExecPermissionArg) -> Self {
        match value {
            CodeExecPermissionArg::Auto => mistralrs_core::AgentPermission::Auto,
            CodeExecPermissionArg::Ask => mistralrs_core::AgentPermission::Ask,
            CodeExecPermissionArg::Deny => mistralrs_core::AgentPermission::Deny,
        }
    }
}

impl Default for GlobalOptions {
    fn default() -> Self {
        Self {
            seed: None,
            log: None,
            token_source: TokenSource::CacheToken,
            verbose: 0,
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
            matformer_config_path: None,
            matformer_slice_name: None,
            mtp_model: None,
            mtp_n_predict: None,
            mcp_config: None,
            agent: false,
            enable_search: false,
            search_embedding_model: None,
            #[cfg(feature = "code-execution")]
            enable_code_execution: false,
            #[cfg(feature = "code-execution")]
            enable_shell: false,
            #[cfg(feature = "code-execution")]
            code_exec_python: None,
            #[cfg(feature = "code-execution")]
            code_exec_timeout: None,
            #[cfg(feature = "code-execution")]
            code_exec_workdir: None,
            #[cfg(feature = "code-execution")]
            shell_path: None,
            #[cfg(feature = "code-execution")]
            shell_timeout: None,
            #[cfg(feature = "code-execution")]
            shell_workdir: None,
            #[cfg(feature = "code-execution")]
            skills_dir: None,
            code_exec_permission: CodeExecPermissionArg::Auto,
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

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::*;

    fn resolve_default_command(command: &str, args: &[&str]) -> anyhow::Result<ModelType> {
        let cli = Cli::try_parse_from(
            ["mistralrs", command]
                .into_iter()
                .chain(args.iter().copied()),
        )?;
        let (model_type, default_model) = match cli.command {
            Command::Run {
                model_type,
                default_model,
                ..
            }
            | Command::Serve {
                model_type,
                default_model,
                ..
            }
            | Command::Bench {
                model_type,
                default_model,
                ..
            } => (model_type, default_model),
            _ => unreachable!(),
        };
        resolve_model_type(model_type, default_model)
    }

    fn resolve_run(args: &[&str]) -> anyhow::Result<ModelType> {
        resolve_default_command("run", args)
    }

    fn resolve_run_error(args: &[&str]) -> anyhow::Error {
        match resolve_run(args) {
            Ok(_) => panic!("expected model resolution to fail"),
            Err(error) => error,
        }
    }

    #[test]
    fn quantized_file_infers_gguf_and_local_root() {
        let filename = format!("mistralrs-{}.GGUF", uuid::Uuid::new_v4());
        fs::write(&filename, []).unwrap();
        let resolved = resolve_run(&["-f", &filename]).unwrap();
        let ModelType::Auto { model, format, .. } = resolved else {
            panic!("expected auto model");
        };
        assert_eq!(model.model_id, ".");
        assert_eq!(format.format, Some(ModelFormat::Gguf));
        assert_eq!(format.quantized_file.as_deref(), Some(filename.as_str()));
        assert!(format.direct_file_only);
        fs::remove_file(filename).unwrap();
    }

    #[test]
    fn direct_gguf_shorthand_is_shared_by_run_serve_and_bench() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("weights")).unwrap();
        let model_path = root.join("weights/model.gguf");
        fs::write(&model_path, []).unwrap();
        let model_path = model_path.to_string_lossy();
        for command in ["run", "serve", "bench"] {
            let resolved = resolve_default_command(command, &["-f", &model_path]).unwrap();
            let ModelType::Auto { model, format, .. } = resolved else {
                panic!("expected auto model");
            };
            assert_eq!(Path::new(&model.model_id), root.join("weights"));
            assert_eq!(format.format, Some(ModelFormat::Gguf));
            assert_eq!(format.quantized_file.as_deref(), Some("model.gguf"));
            assert!(format.direct_file_only);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn direct_gguf_shorthand_rebases_an_explicit_projector() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("weights")).unwrap();
        let model_path = root.join("weights/model.gguf");
        let projector_path = root.join("weights/mmproj-BF16.gguf");
        fs::write(&model_path, []).unwrap();
        fs::write(&projector_path, []).unwrap();
        let model_path = model_path.to_string_lossy();
        for command in ["run", "serve", "bench"] {
            let resolved = resolve_default_command(
                command,
                &["-f", &model_path, "--mmproj", "mmproj-BF16.gguf"],
            )
            .unwrap();
            let ModelType::Auto { model, format, .. } = resolved else {
                panic!("expected auto model");
            };
            assert_eq!(Path::new(&model.model_id), root.join("weights"));
            assert_eq!(format.quantized_file.as_deref(), Some("model.gguf"));
            assert_eq!(format.mmproj.as_deref(), Some("mmproj-BF16.gguf"));
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn direct_text_gguf_accepts_dynamic_lora_options() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let model_path = root.join("model.gguf");
        fs::write(&model_path, []).unwrap();

        let resolved = resolve_run(&[
            "-f",
            &model_path.to_string_lossy(),
            "--lora",
            "code=org/code-lora",
            "--lora-max-rank",
            "64",
        ])
        .unwrap();
        let ModelType::Auto {
            format, adapter, ..
        } = resolved
        else {
            panic!("expected auto model");
        };
        assert_eq!(format.format, Some(ModelFormat::Gguf));
        assert!(format.mmproj.is_none());
        assert_eq!(adapter.lora.len(), 1);
        assert_eq!(adapter.lora[0].alias, "code");
        assert_eq!(adapter.lora[0].source, "org/code-lora");
        assert_eq!(adapter.lora_max_rank, 64);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn direct_gguf_shorthand_preserves_a_cross_directory_projector() {
        let root = PathBuf::from(format!(".mistralrs-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("weights")).unwrap();
        fs::create_dir_all(root.join("projectors")).unwrap();
        let model_path = root.join("weights/model.gguf");
        let projector_path = root.join("projectors/mmproj-BF16.gguf");
        fs::write(&model_path, []).unwrap();
        fs::write(&projector_path, []).unwrap();

        let resolved = resolve_run(&[
            "-f",
            &model_path.to_string_lossy(),
            "--mmproj",
            &projector_path.to_string_lossy(),
        ])
        .unwrap();
        let ModelType::Auto { format, .. } = resolved else {
            panic!("expected auto model");
        };
        let expected_projector = std::env::current_dir().unwrap().join(&projector_path);
        assert_eq!(
            format.mmproj.as_deref(),
            Some(expected_projector.to_string_lossy().as_ref())
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn quantized_shards_derive_common_parent() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let first = root.join("model-00001.gguf");
        let second = root.join("model-00002.GGUF");
        fs::write(&first, []).unwrap();
        fs::write(&second, []).unwrap();
        let files = format!("{}; {}", first.display(), second.display());
        let resolved = resolve_run(&["-f", &files]).unwrap();
        let ModelType::Auto { model, format, .. } = resolved else {
            panic!("expected auto model");
        };
        assert_eq!(Path::new(&model.model_id), root);
        assert_eq!(
            format.quantized_file.as_deref(),
            Some("model-00001.gguf;model-00002.GGUF")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn quantized_shards_without_model_require_common_parent() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("one")).unwrap();
        fs::create_dir_all(root.join("two")).unwrap();
        let first = root.join("one/model-1.gguf");
        let second = root.join("two/model-2.gguf");
        fs::write(&first, []).unwrap();
        fs::write(&second, []).unwrap();
        let files = format!("{};{}", first.display(), second.display());
        let error = resolve_run_error(&["-f", &files]);
        assert!(error.to_string().contains("one parent directory"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn direct_gguf_shorthand_rejects_a_missing_local_file() {
        let missing =
            std::env::temp_dir().join(format!("mistralrs-{}/model.gguf", uuid::Uuid::new_v4()));
        let error = resolve_run_error(&["-f", &missing.to_string_lossy()]);
        assert!(error
            .to_string()
            .contains("does not exist or is not a file"));
    }

    #[test]
    fn remote_quantized_file_infers_ggml() {
        let resolved = resolve_run(&["-m", "org/model", "-f", "model.GGML"]).unwrap();
        let ModelType::Auto { model, format, .. } = resolved else {
            panic!("expected auto model");
        };
        assert_eq!(model.model_id, "org/model");
        assert_eq!(format.format, Some(ModelFormat::Ggml));
    }

    #[test]
    fn mmproj_implies_gguf_for_unknown_main_suffix() {
        let resolved = resolve_run(&[
            "-m",
            "org/model",
            "-f",
            "model.bin",
            "--mmproj",
            "mmproj.gguf",
        ])
        .unwrap();
        let ModelType::Auto { format, .. } = resolved else {
            panic!("expected auto model");
        };
        assert_eq!(format.format, Some(ModelFormat::Gguf));
    }

    #[test]
    fn mmproj_rejects_an_explicit_non_gguf_format() {
        let error = resolve_run_error(&[
            "-m",
            "org/model",
            "--format",
            "ggml",
            "-f",
            "model.ggml",
            "--mmproj",
            "mmproj.gguf",
        ]);
        assert!(error.to_string().contains("requires GGUF"));
    }

    #[test]
    fn format_inference_rejects_mixed_or_unknown_files() {
        let mixed = resolve_run_error(&["-m", "org/model", "-f", "a.gguf;b.ggml"]);
        assert!(mixed.to_string().contains("mixed GGUF and GGML"));

        let unknown = resolve_run_error(&["-m", "org/model", "-f", "model.bin"]);
        assert!(unknown.to_string().contains("Cannot infer model format"));
    }

    #[test]
    fn explicit_format_rejects_contradictory_suffix() {
        let error = resolve_run_error(&["-m", "org/model", "--format", "gguf", "-f", "model.ggml"]);
        assert!(error.to_string().contains("conflicts"));
    }

    #[test]
    fn explicit_ggml_preserves_legacy_bin_filenames() {
        let resolved =
            resolve_run(&["-m", "org/model", "--format", "ggml", "-f", "model.bin"]).unwrap();
        let ModelType::Auto { format, .. } = resolved else {
            panic!("expected auto model");
        };
        assert_eq!(format.format, Some(ModelFormat::Ggml));
        assert_eq!(format.quantized_file.as_deref(), Some("model.bin"));
    }

    #[test]
    fn explicit_model_type_also_infers_format() {
        let resolved = resolve_run(&["auto", "-m", "org/model", "-f", "model.gguf"]).unwrap();
        let ModelType::Auto { format, .. } = resolved else {
            panic!("expected auto model");
        };
        assert_eq!(format.format, Some(ModelFormat::Gguf));
    }

    #[test]
    fn model_id_is_still_required_without_a_quantized_file() {
        let error = resolve_run_error(&[]);
        assert!(error.to_string().contains("--model-id"));

        let resolved = resolve_run(&["-m", "org/plain"]).unwrap();
        let ModelType::Auto { model, format, .. } = resolved else {
            panic!("expected auto model");
        };
        assert_eq!(model.model_id, "org/plain");
        assert_eq!(format.format, None);
    }

    #[test]
    fn run_parses_explicit_lora_preloads_and_request_selection() {
        let cli = Cli::try_parse_from([
            "mistralrs",
            "run",
            "-m",
            "org/base",
            "--enable-lora",
            "--lora",
            "code=org/code-lora",
            "--lora",
            "math=./math-lora",
            "--adapter",
            "code",
        ])
        .unwrap();

        let Command::Run {
            default_model,
            adapter,
            ..
        } = cli.command
        else {
            panic!("expected run command");
        };
        assert_eq!(adapter.as_deref(), Some("code"));
        assert!(default_model.adapter.enable_lora);
        assert_eq!(default_model.adapter.lora.len(), 2);
        assert_eq!(default_model.adapter.lora[1].alias, "math");
    }

    #[test]
    fn run_parses_reasoning_effort() {
        let cli = Cli::try_parse_from([
            "mistralrs",
            "run",
            "-m",
            "org/base",
            "--reasoning-effort",
            "XHIGH",
        ])
        .unwrap();

        let Command::Run {
            reasoning_effort, ..
        } = cli.command
        else {
            panic!("expected run command");
        };
        assert_eq!(reasoning_effort, Some(ReasoningEffort::XHigh));
    }

    #[test]
    fn auto_lora_parses_multimodal_limits() {
        let cli = Cli::try_parse_from([
            "mistralrs",
            "serve",
            "-m",
            "Qwen/Qwen3.6-35B-A3B",
            "--enable-lora",
            "--max-edge",
            "2048",
            "--max-num-images",
            "5",
            "--max-image-length",
            "1536",
        ])
        .unwrap();

        let Command::Serve { default_model, .. } = cli.command else {
            panic!("expected serve command");
        };
        assert!(default_model.adapter.enable_lora);
        assert_eq!(default_model.multimodal.max_edge, Some(2048));
        assert_eq!(default_model.multimodal.max_num_images, Some(5));
        assert_eq!(default_model.multimodal.max_image_length, Some(1536));
    }

    #[test]
    fn lora_preload_requires_an_explicit_alias() {
        let result = Cli::try_parse_from([
            "mistralrs",
            "run",
            "-m",
            "org/base",
            "--lora",
            "org/code-lora",
        ]);
        let error = match result {
            Ok(_) => panic!("expected an invalid LoRA preload"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected ALIAS=SOURCE"));
    }

    #[test]
    fn legacy_raw_lora_does_not_conflict_with_default_runtime_limits() {
        let cli = Cli::try_parse_from([
            "mistralrs",
            "run",
            "-m",
            "org/raw-model",
            "--format",
            "gguf",
            "-f",
            "model.gguf",
            "--legacy-lora",
            "org/legacy-lora",
            "--legacy-lora-order",
            "order.json",
        ]);
        if let Err(error) = cli {
            panic!("{error}");
        }
    }

    #[test]
    fn explicit_multimodal_accepts_dynamic_lora_options() {
        let cli = Cli::try_parse_from([
            "mistralrs",
            "serve",
            "multimodal",
            "-m",
            "org/vision-model",
            "--lora",
            "code=org/code-lora",
        ])
        .unwrap();

        let Command::Serve {
            model_type: Some(ModelType::Multimodal { adapter, .. }),
            ..
        } = cli.command
        else {
            panic!("expected explicit multimodal model");
        };
        assert_eq!(adapter.lora.len(), 1);
        assert_eq!(adapter.lora[0].alias, "code");
        assert_eq!(adapter.lora[0].source, "org/code-lora");
    }

    #[test]
    fn explicit_multimodal_help_lists_only_dynamic_adapter_options() {
        let help = match Cli::try_parse_from(["mistralrs", "serve", "multimodal", "--help"]) {
            Ok(_) => panic!("expected help output"),
            Err(error) => error.to_string(),
        };

        assert!(help.contains("--lora"));
        assert!(help.contains("--enable-lora"));
        assert!(!help.contains("--legacy-lora"));
        assert!(!help.contains("--xlora"));
    }
}
