use anyhow::Result;
use clap::Parser;
use mistralrs_core::{
    initialize_logging, McpClientConfig, ModelSelected, PagedCacheType, TokenSource,
};
use rust_mcp_sdk::schema::LATEST_PROTOCOL_VERSION;
use tokio::join;
use tracing::{error, info};

use mistralrs_server_core::{
    mistralrs_for_server_builder::{
        configure_paged_attn_from_flags, defaults, get_bert_model, MistralRsForServerBuilder,
        ModelConfig,
    },
    mistralrs_server_router_builder::MistralRsServerRouterBuilder,
};

mod interactive_mode;
use interactive_mode::interactive_mode;
mod mcp_server;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// IP to serve on. Defaults to "0.0.0.0"
    #[arg(long)]
    serve_ip: Option<String>,

    /// Integer seed to ensure reproducible random number generation.
    #[arg(short, long)]
    seed: Option<u64>,

    /// Port to serve on.
    #[arg(short, long)]
    port: Option<u16>,

    /// Log all responses and requests to this file
    #[clap(long, short)]
    log: Option<String>,

    /// If a sequence is larger than the maximum model length, truncate the number
    /// of tokens such that the sequence will fit at most the maximum length.
    /// If `max_tokens` is not specified in the request, space for 10 tokens will be reserved instead.
    #[clap(long, short, action)]
    truncate_sequence: bool,

    /// Model selector (used for single-model mode)
    #[clap(subcommand)]
    model: Option<ModelSelected>,

    /// Enable multi-model mode
    #[arg(long)]
    multi_model: bool,

    /// Multi-model configuration file path (JSON format)
    #[arg(long)]
    multi_model_config: Option<String>,

    /// Default model ID to use when no model is specified in requests (multi-model mode only)
    #[arg(long)]
    default_model_id: Option<String>,

    /// Maximum running sequences at any time. If the `tgt_non_granular_index` flag is set for X-LoRA models, this will be set to 1.
    #[arg(long, default_value_t = defaults::MAX_SEQS)]
    max_seqs: usize,

    /// Use no KV cache.
    #[arg(long, default_value_t = defaults::NO_KV_CACHE)]
    no_kv_cache: bool,

    /// Chat template file with a JINJA file with `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.
    /// Used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded.
    #[arg(short, long)]
    chat_template: Option<String>,

    /// Explicit JINJA chat template file (.jinja) to be used. If specified, this overrides all other chat templates.
    #[arg(short, long)]
    jinja_explicit: Option<String>,

    /// Source of the token for authentication.
    /// Can be in the formats: `literal:<value>`, `env:<value>`, `path:<value>`, `cache` to use a cached token, or `none` to use no token.
    /// Defaults to `cache`.
    #[arg(long, default_value_t = defaults::TOKEN_SOURCE, value_parser = parse_token_source)]
    token_source: TokenSource,

    /// Enter interactive mode instead of serving a chat server.
    #[clap(long, short, action)]
    interactive_mode: bool,

    /// Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy.
    #[arg(long, default_value_t = defaults::PREFIX_CACHE_N)]
    prefix_cache_n: usize,

    /// NOTE: This can be omitted to use automatic device mapping!
    /// Number of device layers to load and run on GPU(s). All others will be on the CPU.
    /// If one GPU is used, then this value should be an integer. Otherwise, it follows the following pattern:
    /// ORD:NUM;... Where ORD is a unique device ordinal and NUM is the number of layers for that device.
    #[arg(short, long, value_parser, value_delimiter = ';')]
    num_device_layers: Option<Vec<String>>,

    /// In-situ quantization to apply.
    #[arg(long = "isq")]
    in_situ_quant: Option<String>,

    /// GPU memory to allocate for KV cache with PagedAttention in MBs.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    #[arg(long = "pa-gpu-mem")]
    paged_attn_gpu_mem: Option<usize>,

    /// Percentage of GPU memory to utilize after allocation of KV cache with PagedAttention, from 0 to 1.
    /// If this is not set and the device is CUDA, it will default to `0.9`.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    #[arg(long = "pa-gpu-mem-usage")]
    paged_attn_gpu_mem_usage: Option<f32>,

    /// Total context length to allocate the KV cache for (total number of tokens which the KV cache can hold).
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    /// This is the default setting, and it defaults to the `max-seq-len` specified in after the model type.
    #[arg(long = "pa-ctxt-len")]
    paged_ctxt_len: Option<usize>,

    /// PagedAttention KV cache type (auto or f8e4m3).
    /// Defaults to `auto`.
    #[arg(long = "pa-cache-type", value_parser = parse_cache_type)]
    cache_type: Option<PagedCacheType>,

    /// Block size (number of tokens per block) for PagedAttention. If this is not set and the device is CUDA, it will default to 32.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    #[arg(long = "pa-blk-size")]
    paged_attn_block_size: Option<usize>,

    /// Disable PagedAttention on CUDA. Because PagedAttention is already disabled on Metal, this is only applicable on CUDA.
    #[arg(
        long = "no-paged-attn",
        default_value_t = false,
        conflicts_with = "paged_attn"
    )]
    no_paged_attn: bool,

    /// Enable PagedAttention on Metal. Because PagedAttention is already enabled on CUDA, this is only applicable on Metal.
    #[arg(
        long = "paged-attn",
        default_value_t = false,
        conflicts_with_all = ["no_paged_attn", "cpu"]
    )]
    paged_attn: bool,

    /// Number of tokens to batch the prompt step into. This can help with OOM errors when in the prompt step, but reduces performance.
    #[arg(long = "prompt-batchsize")]
    prompt_chunksize: Option<usize>,

    /// Use CPU only
    #[arg(long)]
    cpu: bool,

    /// Enable searching compatible with the OpenAI `web_search_options` setting. This uses the BERT model specified below or the default.
    #[arg(long = "enable-search")]
    enable_search: bool,

    /// Specify a Hugging Face model ID for a BERT model to assist web searching. Defaults to Snowflake Arctic Embed L.
    #[arg(long = "search-bert-model")]
    search_bert_model: Option<String>,

    /// Enable thinking for interactive mode and models that support it.
    #[arg(long = "enable-thinking")]
    enable_thinking: bool,

    /// Port to serve MCP protocol on
    #[arg(long)]
    mcp_port: Option<u16>,

    /// MCP client configuration file path
    #[arg(long)]
    mcp_config: Option<String>,
}

fn parse_token_source(s: &str) -> Result<TokenSource, String> {
    s.parse()
}

fn parse_cache_type(s: &str) -> Result<PagedCacheType, String> {
    s.parse()
}

/// Load MCP configuration from file path or environment variable
fn load_mcp_config(mcp_config_path: Option<&str>) -> Result<Option<McpClientConfig>> {
    let config_path = if let Some(path) = mcp_config_path {
        Some(path.to_string())
    } else {
        // Check environment variable if no CLI arg provided
        std::env::var("MCP_CONFIG_PATH").ok()
    };

    if let Some(path) = config_path {
        match std::fs::read_to_string(&path) {
            Ok(config_content) => {
                match serde_json::from_str::<McpClientConfig>(&config_content) {
                    Ok(config) => {
                        // Validate configuration
                        if let Err(e) = validate_mcp_config(&config) {
                            error!("MCP configuration validation failed: {}", e);
                            anyhow::bail!("Invalid MCP configuration: {}", e);
                        }

                        info!("Loaded and validated MCP configuration from {}", path);
                        info!("Configured {} MCP servers", config.servers.len());
                        Ok(Some(config))
                    }
                    Err(e) => {
                        error!("Failed to parse MCP configuration: {}", e);
                        error!("Please check your JSON syntax and ensure it matches the MCP configuration schema");
                        anyhow::bail!("Invalid MCP configuration format: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to read MCP configuration file {}: {}", path, e);
                error!("Please ensure the file exists and is readable");
                anyhow::bail!("Cannot read MCP configuration file: {}", e);
            }
        }
    } else {
        Ok(None)
    }
}

/// Validate MCP configuration for common issues
fn validate_mcp_config(config: &McpClientConfig) -> Result<()> {
    use std::collections::HashSet;

    // Check for duplicate server IDs
    let mut seen_ids = HashSet::new();
    for server in &config.servers {
        if !seen_ids.insert(&server.id) {
            anyhow::bail!("Duplicate server ID: {}", server.id);
        }

        // Validate server ID format
        if !server
            .id
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            anyhow::bail!(
                "Invalid server ID '{}': must contain only alphanumeric, hyphen, underscore",
                server.id
            );
        }

        // Validate URLs for HTTP/WebSocket sources
        match &server.source {
            mistralrs_core::McpServerSource::Http { url, .. }
            | mistralrs_core::McpServerSource::WebSocket { url, .. } => {
                // Basic URL validation - check for scheme
                if !url.starts_with("http://")
                    && !url.starts_with("https://")
                    && !url.starts_with("ws://")
                    && !url.starts_with("wss://")
                {
                    anyhow::bail!("Invalid URL for server '{}': must start with http://, https://, ws://, or wss://", server.id);
                }
                if url.len() < 10 {
                    anyhow::bail!("Invalid URL for server '{}': URL too short", server.id);
                }
            }
            mistralrs_core::McpServerSource::Process { command, .. } => {
                if command.is_empty() {
                    anyhow::bail!("Empty command for server '{}'", server.id);
                }
            }
        }
    }

    // Validate global settings
    if let Some(timeout) = config.tool_timeout_secs {
        if timeout == 0 {
            anyhow::bail!("tool_timeout_secs must be greater than 0");
        }
    }

    if let Some(max_calls) = config.max_concurrent_calls {
        if max_calls == 0 {
            anyhow::bail!("max_concurrent_calls must be greater than 0");
        }
    }

    Ok(())
}

/// Load multi-model configuration from file
fn load_multi_model_config(config_path: &str) -> Result<Vec<ModelConfig>> {
    let config_content = std::fs::read_to_string(config_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read multi-model config file {}: {}",
            config_path,
            e
        )
    })?;

    let configs: Vec<ModelConfig> = serde_json::from_str(&config_content)
        .map_err(|e| anyhow::anyhow!("Failed to parse multi-model config: {}", e))?;

    if configs.is_empty() {
        anyhow::bail!("Multi-model configuration file is empty");
    }

    // Validate model IDs are unique
    let mut seen_ids = std::collections::HashSet::new();
    for config in &configs {
        if !seen_ids.insert(&config.model_id) {
            anyhow::bail!("Duplicate model ID in config: {}", config.model_id);
        }
    }

    info!(
        "Loaded multi-model configuration with {} models",
        configs.len()
    );
    Ok(configs)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    initialize_logging();

    // Load MCP configuration if provided
    let mcp_config = load_mcp_config(args.mcp_config.as_deref())?;

    // Validate configuration
    if args.multi_model && args.model.is_some() {
        anyhow::bail!("Cannot specify both --multi-model and a model subcommand. Use either single-model mode (with model subcommand) or multi-model mode (with --multi-model).");
    }

    if !args.multi_model && args.model.is_none() {
        anyhow::bail!("Must specify either a model subcommand for single-model mode or --multi-model for multi-model mode.");
    }

    if args.multi_model && args.multi_model_config.is_none() {
        anyhow::bail!("Multi-model mode requires --multi-model-config to be specified.");
    }

    let paged_attn = configure_paged_attn_from_flags(args.paged_attn, args.no_paged_attn)?;

    let mistralrs = if args.multi_model {
        // Multi-model mode
        let model_configs = load_multi_model_config(args.multi_model_config.as_ref().unwrap())?;

        let mut builder = MistralRsForServerBuilder::new()
            .with_truncate_sequence(args.truncate_sequence)
            .with_max_seqs(args.max_seqs)
            .with_no_kv_cache(args.no_kv_cache)
            .with_token_source(args.token_source)
            .with_interactive_mode(args.interactive_mode)
            .with_prefix_cache_n(args.prefix_cache_n)
            .set_paged_attn(paged_attn)
            .with_cpu(args.cpu)
            .with_enable_search(args.enable_search)
            .with_seed_optional(args.seed)
            .with_log_optional(args.log)
            .with_prompt_chunksize_optional(args.prompt_chunksize)
            .with_mcp_config_optional(mcp_config)
            .with_paged_attn_cache_type(args.cache_type.unwrap_or_default());

        // Add models to builder
        for config in model_configs {
            builder = builder.add_model_config(config);
        }

        // Set default model if specified
        if let Some(default_id) = args.default_model_id {
            builder = builder.with_default_model_id(default_id);
        }

        builder.build_multi_model().await?
    } else {
        // Single-model mode
        MistralRsForServerBuilder::new()
            .with_truncate_sequence(args.truncate_sequence)
            .with_model(args.model.unwrap())
            .with_max_seqs(args.max_seqs)
            .with_no_kv_cache(args.no_kv_cache)
            .with_token_source(args.token_source)
            .with_interactive_mode(args.interactive_mode)
            .with_prefix_cache_n(args.prefix_cache_n)
            .set_paged_attn(paged_attn)
            .with_cpu(args.cpu)
            .with_enable_search(args.enable_search)
            .with_seed_optional(args.seed)
            .with_log_optional(args.log)
            .with_chat_template_optional(args.chat_template)
            .with_jinja_explicit_optional(args.jinja_explicit)
            .with_num_device_layers_optional(args.num_device_layers)
            .with_in_situ_quant_optional(args.in_situ_quant)
            .with_paged_attn_gpu_mem_optional(args.paged_attn_gpu_mem)
            .with_paged_attn_gpu_mem_usage_optional(args.paged_attn_gpu_mem_usage)
            .with_paged_ctxt_len_optional(args.paged_ctxt_len)
            .with_paged_attn_block_size_optional(args.paged_attn_block_size)
            .with_prompt_chunksize_optional(args.prompt_chunksize)
            .with_mcp_config_optional(mcp_config)
            .with_paged_attn_cache_type(args.cache_type.unwrap_or_default())
            .build()
            .await?
    };

    // TODO: refactor this
    let bert_model = get_bert_model(args.enable_search, args.search_bert_model);

    if args.interactive_mode {
        interactive_mode(
            mistralrs,
            bert_model.is_some(),
            args.enable_thinking.then_some(true),
        )
        .await;
        return Ok(());
    }

    if !args.interactive_mode && args.port.is_none() && args.mcp_port.is_none() {
        anyhow::bail!("Interactive mode was not specified, so expected port to be specified. Perhaps you forgot `-i` or `--port` or `--mcp-port`?")
    }

    let mcp_port = if let Some(port) = args.mcp_port {
        let host = args
            .serve_ip
            .clone()
            .unwrap_or_else(|| "0.0.0.0".to_string());
        info!("MCP server listening on http://{host}:{port}/mcp.");
        info!("MCP protocol version is {}.", LATEST_PROTOCOL_VERSION);
        let mcp_server = mcp_server::create_http_mcp_server(mistralrs.clone(), host, port);

        tokio::spawn(async move {
            if let Err(e) = mcp_server.await {
                eprintln!("MCP server error: {e}");
            }
        })
    } else {
        tokio::spawn(async {})
    };

    let oai_port = if let Some(port) = args.port {
        let ip = args
            .serve_ip
            .clone()
            .unwrap_or_else(|| "0.0.0.0".to_string());

        // Create listener early to validate address before model loading
        let listener = tokio::net::TcpListener::bind(format!("{ip}:{port}")).await?;

        let app = MistralRsServerRouterBuilder::new()
            .with_mistralrs(mistralrs)
            .build()
            .await?;

        info!("OpenAI-compatible server listening on http://{ip}:{port}.");

        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                eprintln!("OpenAI server error: {e}");
            }
        })
    } else {
        tokio::spawn(async {})
    };

    let (_, _) = join!(oai_port, mcp_port);

    Ok(())
}
