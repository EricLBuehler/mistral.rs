//! ## mistral.rs instance for server builder.

use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::Device;
use mistralrs_core::{
    get_auto_device_map_params, get_model_dtype, get_tgt_non_granular_index, paged_attn_supported,
    parse_isq_value, AutoDeviceMapParams, DefaultSchedulerMethod, DeviceLayerMapMetadata,
    DeviceMapMetadata, DeviceMapSetting, Loader, LoaderBuilder, McpClientConfig, MemoryGpuConfig,
    MistralRsBuilder, ModelLoaderConfig, ModelSelected, PagedAttentionConfig, PagedCacheType,
    SchedulerConfig, SearchCallback, SearchEmbeddingModel, TokenSource,
};
use tracing::{info, warn};

use crate::types::{LoadedPipeline, SharedMistralRsState};
use std::collections::{HashMap, HashSet};

/// Configuration for a single model in a multi-model setup
#[derive(Clone, serde::Deserialize)]
pub struct ModelConfig {
    /// Configuration key for this model (human-friendly label)
    pub model_id: String,
    /// Optional alias used as the API model ID
    pub alias: Option<String>,
    /// Model selector
    pub model: ModelSelected,
    /// Model-specific chat template
    pub chat_template: Option<String>,
    /// Model-specific JINJA template
    pub jinja_explicit: Option<String>,
    /// Model-specific device layers
    pub num_device_layers: Option<Vec<String>>,
    /// Model-specific in-situ quantization
    pub in_situ_quant: Option<String>,
}

impl ModelConfig {
    pub fn new(model_id: String, model: ModelSelected) -> Self {
        Self {
            model_id,
            alias: None,
            model,
            chat_template: None,
            jinja_explicit: None,
            num_device_layers: None,
            in_situ_quant: None,
        }
    }

    pub fn with_chat_template(mut self, chat_template: String) -> Self {
        self.chat_template = Some(chat_template);
        self
    }

    pub fn with_alias(mut self, alias: String) -> Self {
        self.alias = Some(alias);
        self
    }

    pub fn with_jinja_explicit(mut self, jinja_explicit: String) -> Self {
        self.jinja_explicit = Some(jinja_explicit);
        self
    }

    pub fn with_num_device_layers(mut self, num_device_layers: Vec<String>) -> Self {
        self.num_device_layers = Some(num_device_layers);
        self
    }

    pub fn with_in_situ_quant(mut self, in_situ_quant: String) -> Self {
        self.in_situ_quant = Some(in_situ_quant);
        self
    }
}

pub mod defaults {
    use super::SearchEmbeddingModel;
    // Provides the default values used for the mistral.rs instance for server.
    // These defaults can be used for CLI argument fallbacks, config loading, or general initialization.

    use std::sync::Arc;

    use mistralrs_core::PagedCacheType;

    pub const DEVICE: Option<candle_core::Device> = None;
    pub const SEED: Option<u64> = None;
    pub const LOG: Option<String> = None;
    pub const MODEL: Option<mistralrs_core::ModelSelected> = None;
    pub const MAX_SEQS: usize = 16;
    pub const NO_KV_CACHE: bool = false;
    pub const CHAT_TEMPLATE: Option<String> = None;
    pub const JINJA_EXPLICIT: Option<String> = None;
    pub const INTERACTIVE_MODE: bool = false;
    pub const PREFIX_CACHE_N: usize = 16;
    pub const NUM_DEVICE_LAYERS: Option<Vec<String>> = None;
    pub const IN_SITU_QUANT: Option<String> = None;
    pub const PAGED_ATTN_GPU_MEM: Option<usize> = None;
    pub const PAGED_ATTN_GPU_MEM_USAGE: Option<f32> = None;
    pub const PAGED_CTXT_LEN: Option<usize> = None;
    pub const PAGED_ATTN_BLOCK_SIZE: Option<usize> = None;
    pub const PAGED_ATTN: Option<bool> = None;
    pub const PAGED_ATTN_CPU: bool = false;
    pub const PAGED_ATTN_CUDA: bool = true;
    pub const PAGED_ATTN_METAL: bool = false;
    pub const CPU: bool = false;
    pub const ENABLE_SEARCH: bool = false;
    pub const SEARCH_EMBEDDING_MODEL: Option<SearchEmbeddingModel> = None;
    pub const TOKEN_SOURCE: mistralrs_core::TokenSource = mistralrs_core::TokenSource::CacheToken;
    pub const SEARCH_CALLBACK: Option<Arc<mistralrs_core::SearchCallback>> = None;
    pub const PAGED_CACHE_TYPE: PagedCacheType = PagedCacheType::Auto;
}

/// A builder for creating a mistral.rs instance with configured options for the mistral.rs server.
///
/// ### Examples
///
/// Basic usage:
/// ```ignore
/// use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;
///
/// let args = Args::parse();
///
/// let mistralrs = MistralRsForServerBuilder::new()
///        .with_model(args.model)
///        .with_max_seqs(args.max_seqs)
///        .with_no_kv_cache(args.no_kv_cache)
///        .with_token_source(args.token_source)
///        .with_interactive_mode(args.interactive_mode)
///        .with_prefix_cache_n(args.prefix_cache_n)
///        .with_paged_attn(args.paged_attn)
///        .with_cpu(args.cpu)
///        .with_enable_search(args.enable_search)
///        .with_seed_optional(args.seed)
///        .with_log_optional(args.log)
///        .with_chat_template_optional(args.chat_template)
///        .with_jinja_explicit_optional(args.jinja_explicit)
///        .with_num_device_layers_optional(args.num_device_layers)
///        .with_in_situ_quant_optional(args.in_situ_quant)
///        .with_paged_attn_gpu_mem_optional(args.paged_attn_gpu_mem)
///        .with_paged_attn_gpu_mem_usage_optional(args.paged_attn_gpu_mem_usage)
///        .with_paged_ctxt_len_optional(args.paged_ctxt_len)
///        .with_paged_attn_block_size_optional(args.paged_attn_block_size)
///        .build()
///        .await?;
/// ```
pub struct MistralRsForServerBuilder {
    /// The Candle device to use for model execution (CPU, CUDA, Metal, etc.).
    device: Option<Device>,

    /// Integer seed to ensure reproducible random number generation.
    seed: Option<u64>,

    /// Log all responses and requests to this file
    log: Option<String>,

    /// Model selector (for single-model mode, deprecated in favor of models)
    model: Option<ModelSelected>,

    /// Multiple model configurations (for multi-model mode)
    models: Vec<ModelConfig>,

    /// Default model ID to use when none is specified in requests
    default_model_id: Option<String>,

    /// Maximum running sequences at any time. If the `tgt_non_granular_index` flag is set for X-LoRA models, this will be set to 1.
    max_seqs: usize,

    /// Use no KV cache.
    no_kv_cache: bool,

    /// Chat template file with a JINJA file with `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.
    /// Used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded.
    chat_template: Option<String>,

    /// Explicit JINJA chat template file (.jinja) to be used. If specified, this overrides all other chat templates.
    jinja_explicit: Option<String>,

    /// Source of the token for authentication.
    /// Can be in the formats: `literal:<value>`, `env:<value>`, `path:<value>`, `cache` to use a cached token, or `none` to use no token.
    /// Defaults to `cache`.
    token_source: TokenSource,

    /// Enter interactive mode instead of serving a chat server.
    interactive_mode: bool,

    /// Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy.
    prefix_cache_n: usize,

    /// NOTE: This can be omitted to use automatic device mapping!
    /// Number of device layers to load and run on GPU(s). All others will be on the CPU.
    /// If one GPU is used, then this value should be an integer. Otherwise, it follows the following pattern:
    /// ORD:NUM;... Where ORD is a unique device ordinal and NUM is the number of layers for that device.
    num_device_layers: Option<Vec<String>>,

    /// In-situ quantization to apply.
    in_situ_quant: Option<String>,

    /// GPU memory to allocate for KV cache with PagedAttention in MBs.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    paged_attn_gpu_mem: Option<usize>,

    /// Percentage of GPU memory to utilize after allocation of KV cache with PagedAttention, from 0 to 1.
    /// If this is not set and the device is CUDA, it will default to `0.9`.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    paged_attn_gpu_mem_usage: Option<f32>,

    /// Total context length to allocate the KV cache for (total number of tokens which the KV cache can hold).
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    /// This is the default setting, and it defaults to the `max-seq-len` specified in after the model type.
    paged_ctxt_len: Option<usize>,

    /// Block size (number of tokens per block) for PagedAttention. If this is not set and the device is CUDA, it will default to 32.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    paged_attn_block_size: Option<usize>,

    /// Enables or disables PagedAttention. By default, PagedAttention will be enabled for CUDA and disabled for Metal (and is not supported for CPU). Use this to override the default behavior.
    paged_attn: Option<bool>,

    /// Use CPU only
    cpu: bool,

    /// Enable searching compatible with the OpenAI `web_search_options` setting. This loads the selected search embedding reranker (EmbeddingGemma by default).
    enable_search: bool,

    /// Specify which built-in search embedding model to load.
    search_embedding_model: Option<SearchEmbeddingModel>,

    /// Optional override search callback
    search_callback: Option<Arc<SearchCallback>>,

    /// Optional MCP client configuration
    mcp_client_config: Option<McpClientConfig>,

    /// PagedAttention KV cache type
    paged_cache_type: PagedCacheType,
}

impl Default for MistralRsForServerBuilder {
    /// Creates a new builder with default configuration.
    fn default() -> Self {
        Self {
            device: defaults::DEVICE,
            seed: defaults::SEED,
            log: defaults::LOG,
            model: defaults::MODEL,
            models: Vec::new(),
            default_model_id: None,
            max_seqs: defaults::MAX_SEQS,
            no_kv_cache: defaults::NO_KV_CACHE,
            chat_template: defaults::CHAT_TEMPLATE,
            jinja_explicit: defaults::JINJA_EXPLICIT,
            token_source: defaults::TOKEN_SOURCE,
            interactive_mode: defaults::INTERACTIVE_MODE,
            prefix_cache_n: defaults::PREFIX_CACHE_N,
            num_device_layers: defaults::NUM_DEVICE_LAYERS,
            in_situ_quant: defaults::IN_SITU_QUANT,
            paged_attn_gpu_mem: defaults::PAGED_ATTN_GPU_MEM,
            paged_attn_gpu_mem_usage: defaults::PAGED_ATTN_GPU_MEM_USAGE,
            paged_ctxt_len: defaults::PAGED_CTXT_LEN,
            paged_attn_block_size: defaults::PAGED_ATTN_BLOCK_SIZE,
            paged_attn: defaults::PAGED_ATTN,
            cpu: defaults::CPU,
            enable_search: defaults::ENABLE_SEARCH,
            search_embedding_model: defaults::SEARCH_EMBEDDING_MODEL,
            search_callback: defaults::SEARCH_CALLBACK,
            mcp_client_config: None,
            paged_cache_type: defaults::PAGED_CACHE_TYPE,
        }
    }
}

impl MistralRsForServerBuilder {
    /// Creates a new `MistralRsForServerBuilder` with default settings.
    ///
    /// This is equivalent to calling `Default::default()`.
    ///
    /// ### Examples
    ///
    /// ```ignore
    /// use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;
    ///
    /// let builder = mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder::new();
    /// ```
    pub fn new() -> Self {
        Default::default()
    }

    /// Sets the Candle device to use for model execution.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Sets the random seed for deterministic model behavior.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the random seed if provided.
    pub fn with_seed_optional(mut self, seed: Option<u64>) -> Self {
        if let Some(seed) = seed {
            self = self.with_seed(seed);
        }
        self
    }

    /// Sets the logging configuration.
    pub fn with_log(mut self, log: String) -> Self {
        self.log = Some(log);
        self
    }

    /// Sets the logging configuration if provided.
    pub fn with_log_optional(mut self, log: Option<String>) -> Self {
        if let Some(log) = log {
            self = self.with_log(log);
        }
        self
    }

    /// Sets the model to be used.
    pub fn with_model(mut self, model: ModelSelected) -> Self {
        self.model = Some(model);
        self
    }

    /// Add a model to the multi-model configuration.
    pub fn with_model_config(mut self, model_config: ModelConfig) -> Self {
        self.models.push(model_config);
        self
    }

    /// Add multiple models to the multi-model configuration.
    pub fn with_model_configs(mut self, model_configs: Vec<ModelConfig>) -> Self {
        self.models.extend(model_configs);
        self
    }

    /// Set the default model ID to use when none is specified in requests.
    pub fn with_default_model_id(mut self, default_model_id: String) -> Self {
        self.default_model_id = Some(default_model_id);
        self
    }

    /// Add a model configuration.
    pub fn add_model_config(mut self, config: ModelConfig) -> Self {
        self.models.push(config);
        self
    }

    /// Add a model with just an ID and ModelSelected (convenience method).
    pub fn add_model(mut self, model_id: String, model: ModelSelected) -> Self {
        self.models.push(ModelConfig::new(model_id, model));
        self
    }

    /// Add a model with a custom alias used for API requests.
    pub fn add_model_with_alias(
        mut self,
        model_id: String,
        alias: String,
        model: ModelSelected,
    ) -> Self {
        self.models
            .push(ModelConfig::new(model_id, model).with_alias(alias));
        self
    }

    /// Sets the maximum number of concurrent sequences.
    pub fn with_max_seqs(mut self, max_seqs: usize) -> Self {
        self.max_seqs = max_seqs;
        self
    }

    /// Sets whether to disable the key-value cache.
    pub fn with_no_kv_cache(mut self, no_kv_cache: bool) -> Self {
        self.no_kv_cache = no_kv_cache;
        self
    }

    /// Sets the chat template configuration.
    pub fn with_chat_template(mut self, chat_template: String) -> Self {
        self.chat_template = Some(chat_template);
        self
    }

    /// Sets the chat template configuration if provided.
    pub fn with_chat_template_optional(mut self, chat_template: Option<String>) -> Self {
        if let Some(chat_template) = chat_template {
            self = self.with_chat_template(chat_template);
        }
        self
    }

    /// Sets an explicit JINJA chat template file.
    pub fn with_jinja_explicit(mut self, jinja_explicit: String) -> Self {
        self.jinja_explicit = Some(jinja_explicit);
        self
    }

    /// Sets an explicit JINJA chat template file if provided.
    pub fn with_jinja_explicit_optional(mut self, jinja_explicit: Option<String>) -> Self {
        if let Some(jinja_explicit) = jinja_explicit {
            self = self.with_jinja_explicit(jinja_explicit);
        }
        self
    }

    /// Sets the token source for authentication.
    pub fn with_token_source(mut self, token_source: TokenSource) -> Self {
        self.token_source = token_source;
        self
    }

    /// Sets whether to run in interactive mode.
    pub fn with_interactive_mode(mut self, interactive_mode: bool) -> Self {
        self.interactive_mode = interactive_mode;
        self
    }

    /// Sets the number of prefix caches to hold on the device.
    pub fn with_prefix_cache_n(mut self, prefix_cache_n: usize) -> Self {
        self.prefix_cache_n = prefix_cache_n;
        self
    }

    /// Sets the device layer mapping
    pub fn with_num_device_layers(mut self, num_device_layers: Vec<String>) -> Self {
        self.num_device_layers = Some(num_device_layers);
        self
    }

    /// Sets the device layer mapping if provided.
    pub fn with_num_device_layers_optional(
        mut self,
        num_device_layers: Option<Vec<String>>,
    ) -> Self {
        if let Some(num_device_layers) = num_device_layers {
            self = self.with_num_device_layers(num_device_layers);
        }
        self
    }

    /// Sets the in-situ quantization method.
    pub fn with_in_situ_quant(mut self, in_situ_quant: String) -> Self {
        self.in_situ_quant = Some(in_situ_quant);
        self
    }

    /// Sets the in-situ quantization method if provided.
    pub fn with_in_situ_quant_optional(mut self, in_situ_quant: Option<String>) -> Self {
        if let Some(in_situ_quant) = in_situ_quant {
            self = self.with_in_situ_quant(in_situ_quant);
        }
        self
    }

    /// Sets PagedAttention.
    ///
    /// Unlike other `with_PROP` or `with_PROP_optional` methods, this method
    /// sets the value to whatever `Option<bool>` is passed in as `None`, `Some(true)`
    /// and `Some(false)` have different implications.
    ///
    /// `None`: default behavior for target device (e.g. enable for CUDA, disable for Metal)
    /// `Some(true)`: enable (if supported by target device)
    /// `Some(false)`: disable
    pub fn set_paged_attn(mut self, paged_attn: Option<bool>) -> Self {
        self.paged_attn = paged_attn;
        self
    }

    /// Sets the GPU memory allocation for PagedAttention KV cache.
    pub fn with_paged_attn_gpu_mem(mut self, paged_attn_gpu_mem: usize) -> Self {
        self.paged_attn_gpu_mem = Some(paged_attn_gpu_mem);
        self
    }

    /// Sets the GPU memory allocation for PagedAttention KV cache if provided.
    pub fn with_paged_attn_gpu_mem_optional(mut self, paged_attn_gpu_mem: Option<usize>) -> Self {
        if let Some(paged_attn_gpu_mem) = paged_attn_gpu_mem {
            self = self.with_paged_attn_gpu_mem(paged_attn_gpu_mem);
        }
        self
    }

    /// Sets the percentage of GPU memory to utilize for PagedAttention.
    pub fn with_paged_attn_gpu_mem_usage(mut self, paged_attn_gpu_mem_usage: f32) -> Self {
        self.paged_attn_gpu_mem_usage = Some(paged_attn_gpu_mem_usage);
        self
    }

    /// Sets the percentage of GPU memory to utilize for PagedAttention if provided.
    pub fn with_paged_attn_gpu_mem_usage_optional(
        mut self,
        paged_attn_gpu_mem_usage: Option<f32>,
    ) -> Self {
        if let Some(paged_attn_gpu_mem_usage) = paged_attn_gpu_mem_usage {
            self = self.with_paged_attn_gpu_mem_usage(paged_attn_gpu_mem_usage);
        }
        self
    }

    /// Sets the total context length for KV cache allocation.
    pub fn with_paged_ctxt_len(mut self, paged_ctxt_len: usize) -> Self {
        self.paged_ctxt_len = Some(paged_ctxt_len);
        self
    }

    /// Sets the total context length for KV cache allocation if provided.
    pub fn with_paged_ctxt_len_optional(mut self, paged_ctxt_len: Option<usize>) -> Self {
        if let Some(paged_ctxt_len) = paged_ctxt_len {
            self = self.with_paged_ctxt_len(paged_ctxt_len);
        }
        self
    }

    /// Sets the block size for PagedAttention.
    pub fn with_paged_attn_block_size(mut self, paged_attn_block_size: usize) -> Self {
        self.paged_attn_block_size = Some(paged_attn_block_size);
        self
    }

    /// Sets the block size for PagedAttention.
    pub fn with_paged_attn_cache_type(mut self, cache_type: PagedCacheType) -> Self {
        self.paged_cache_type = cache_type;
        self
    }

    /// Sets the block size for PagedAttention if provided.
    pub fn with_paged_attn_block_size_optional(
        mut self,
        paged_attn_block_size: Option<usize>,
    ) -> Self {
        if let Some(paged_attn_block_size) = paged_attn_block_size {
            self = self.with_paged_attn_block_size(paged_attn_block_size);
        }
        self
    }

    /// Sets whether to force CPU-only execution.
    pub fn with_cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    /// Sets whether to enable web search functionality.
    pub fn with_enable_search(mut self, enable_search: bool) -> Self {
        self.enable_search = enable_search;
        self
    }

    /// Sets the embedding model used for web search assistance.
    pub fn with_search_embedding_model(
        mut self,
        search_embedding_model: SearchEmbeddingModel,
    ) -> Self {
        self.search_embedding_model = Some(search_embedding_model);
        self
    }

    /// Override the search function used when `web_search_options` is enabled.
    pub fn with_search_callback(mut self, callback: Arc<SearchCallback>) -> Self {
        self.search_callback = Some(callback);
        self
    }

    /// Sets the MCP client configuration.
    pub fn with_mcp_config(mut self, mcp_config: McpClientConfig) -> Self {
        self.mcp_client_config = Some(mcp_config);
        self
    }

    /// Sets the MCP client configuration if provided.
    pub fn with_mcp_config_optional(mut self, mcp_config: Option<McpClientConfig>) -> Self {
        if let Some(mcp_config) = mcp_config {
            self = self.with_mcp_config(mcp_config);
        }
        self
    }

    /// Builds the configured mistral.rs instance.
    ///
    /// ### Examples
    ///
    /// ```ignore
    /// use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;
    ///
    /// let shared_mistralrs = MistralRsForServerBuilder::new()
    ///     .with_model(model)
    ///     .with_in_situ_quant("8".to_string())
    ///     .set_paged_attn(Some(true))
    ///     .build()
    ///     .await?;
    /// ```
    pub async fn build(self) -> Result<SharedMistralRsState> {
        // Determine if we're in single-model or multi-model mode
        if !self.models.is_empty() {
            self.build_multi_model().await
        } else {
            self.build_single_model().await
        }
    }

    /// Build a single-model instance (legacy mode)
    async fn build_single_model(mut self) -> Result<SharedMistralRsState> {
        let model = self.model.context("Model was None")?;

        let tgt_non_granular_index = get_tgt_non_granular_index(&model);
        let dtype = get_model_dtype(&model)?;
        let auto_device_map_params = get_auto_device_map_params(&model)?;

        if tgt_non_granular_index.is_some() {
            self.max_seqs = 1;
        }

        let device = if let Some(device) = self.device {
            device
        } else {
            init_device(self.cpu, self.seed)?
        };

        let mapper = init_mapper(&self.num_device_layers, &auto_device_map_params);
        let paged_attn = configure_paged_attn(&device, self.paged_attn);

        let cache_config = init_cache_config(
            self.paged_attn_block_size,
            self.paged_attn_gpu_mem,
            self.paged_attn_gpu_mem_usage,
            self.paged_ctxt_len,
            self.paged_cache_type,
            !paged_attn,
        )?;

        // Clone values needed for loader config before they're moved
        let model_for_config = model.clone();
        let token_source_for_config = self.token_source.clone();
        let mapper_for_config = mapper.clone();
        let chat_template_for_config = self.chat_template.clone();
        let jinja_explicit_for_config = self.jinja_explicit.clone();

        // Configure this last to prevent arg moves
        let loader: Box<dyn Loader> = LoaderBuilder::new(model)
            .with_no_kv_cache(self.no_kv_cache)
            .with_chat_template(self.chat_template)
            .with_jinja_explicit(self.jinja_explicit)
            .build()?;

        mistralrs_instance_info(&*loader);

        let isq = self
            .in_situ_quant
            .as_ref()
            .and_then(|isq| parse_isq_value(isq, Some(&device)).ok());

        let pipeline: LoadedPipeline = loader.load_model_from_hf(
            None,
            self.token_source,
            &dtype,
            &device,
            false,
            mapper,
            isq,
            cache_config,
        )?;
        info!("Model loaded.");

        let scheduler_config = init_scheduler_config(&cache_config, &pipeline, self.max_seqs).await;

        let search_embedding_model =
            get_search_embedding_model(self.enable_search, self.search_embedding_model);

        // Create loader config for unload/reload support
        let loader_config = ModelLoaderConfig {
            model_selected: model_for_config,
            token_source: token_source_for_config,
            hf_revision: None,
            dtype,
            device: device.clone(),
            device_map_setting: mapper_for_config,
            isq,
            paged_attn_config: cache_config,
            silent: false,
            chat_template: chat_template_for_config,
            jinja_explicit: jinja_explicit_for_config,
        };

        let mut builder = MistralRsBuilder::new(
            pipeline,
            scheduler_config,
            !self.interactive_mode,
            search_embedding_model,
        )
        .with_opt_log(self.log)
        .with_no_kv_cache(self.no_kv_cache)
        .with_prefix_cache_n(self.prefix_cache_n)
        .with_loader_config(loader_config);

        // Add MCP client configuration if provided
        if let Some(mcp_config) = self.mcp_client_config {
            builder = builder.with_mcp_client(mcp_config);
        }

        let mistralrs = builder.build().await;

        Ok(mistralrs)
    }

    /// Build a multi-model instance
    pub async fn build_multi_model(mut self) -> Result<SharedMistralRsState> {
        if self.models.is_empty() {
            anyhow::bail!("No models configured for multi-model mode");
        }

        // Use the first model as the base configuration
        let first_model = &self.models[0];
        let model = first_model.model.clone();

        let tgt_non_granular_index = get_tgt_non_granular_index(&model);
        let dtype = get_model_dtype(&model)?;
        let auto_device_map_params = get_auto_device_map_params(&model)?;

        if tgt_non_granular_index.is_some() {
            self.max_seqs = 1;
        }

        let device = if let Some(device) = self.device {
            device
        } else {
            init_device(self.cpu, self.seed)?
        };

        // Create the first model's pipeline
        let loader: Box<dyn Loader> = LoaderBuilder::new(model)
            .with_no_kv_cache(self.no_kv_cache)
            .with_chat_template(
                first_model
                    .chat_template
                    .clone()
                    .or(self.chat_template.clone()),
            )
            .with_jinja_explicit(
                first_model
                    .jinja_explicit
                    .clone()
                    .or(self.jinja_explicit.clone()),
            )
            .build()?;

        mistralrs_instance_info(&*loader);

        let mapper = init_mapper(
            &first_model
                .num_device_layers
                .clone()
                .or(self.num_device_layers.clone()),
            &auto_device_map_params,
        );
        let paged_attn = configure_paged_attn(&device, self.paged_attn);

        let cache_config = init_cache_config(
            self.paged_attn_block_size,
            self.paged_attn_gpu_mem,
            self.paged_attn_gpu_mem_usage,
            self.paged_ctxt_len,
            self.paged_cache_type,
            !paged_attn,
        )?;

        let isq = first_model
            .in_situ_quant
            .as_ref()
            .or(self.in_situ_quant.as_ref())
            .and_then(|isq| parse_isq_value(isq, Some(&device)).ok());

        let mut loaded_model_ids = Vec::new();
        let mut registered_ids = HashSet::new();

        let pipeline: LoadedPipeline = loader.load_model_from_hf(
            None,
            self.token_source.clone(),
            &dtype,
            &device,
            false,
            mapper,
            isq,
            cache_config,
        )?;
        let first_pipeline_name = pipeline.lock().await.name();
        let first_primary_id = first_model
            .alias
            .clone()
            .unwrap_or_else(|| first_pipeline_name.clone());

        if !registered_ids.insert(first_primary_id.clone()) {
            anyhow::bail!(
                "Model ID conflict: '{}' is already registered (config key: {}).",
                first_primary_id,
                first_model.model_id
            );
        }

        if first_primary_id == first_pipeline_name {
            info!(
                "First model loaded: `{}` (from config key: {})",
                first_primary_id, first_model.model_id
            );
        } else {
            info!(
                "First model loaded: `{}` (pipeline: `{}`; config key: {})",
                first_primary_id, first_pipeline_name, first_model.model_id
            );
        }
        loaded_model_ids.push(first_primary_id.clone());

        let scheduler_config = init_scheduler_config(&cache_config, &pipeline, self.max_seqs).await;
        let search_embedding_model =
            get_search_embedding_model(self.enable_search, self.search_embedding_model);

        // Create the first MistralRs instance with the first model
        let mut builder = MistralRsBuilder::new(
            pipeline,
            scheduler_config.clone(),
            !self.interactive_mode,
            search_embedding_model,
        )
        .with_opt_log(self.log.clone())
        .with_no_kv_cache(self.no_kv_cache)
        .with_prefix_cache_n(self.prefix_cache_n);
        if first_primary_id != first_pipeline_name {
            builder = builder.with_model_id(first_primary_id.clone());
        }

        // Add MCP client configuration if provided
        if let Some(mcp_config) = self.mcp_client_config.clone() {
            builder = builder.with_mcp_client(mcp_config);
        }

        let mistralrs = builder.build().await;

        if let Some(alias) = first_model.alias.as_ref() {
            if alias != &first_pipeline_name {
                mistralrs
                    .register_model_alias(first_pipeline_name.clone(), &first_primary_id)
                    .map_err(|e| anyhow::anyhow!(e))?;
            }
        }

        // Load additional models
        for model_config in self.models.iter().skip(1) {
            info!(
                "Loading additional model from config key: {}",
                model_config.model_id
            );

            let model = model_config.model.clone();
            let dtype = get_model_dtype(&model)?;
            let auto_device_map_params = get_auto_device_map_params(&model)?;

            let loader: Box<dyn Loader> = LoaderBuilder::new(model)
                .with_no_kv_cache(self.no_kv_cache)
                .with_chat_template(
                    model_config
                        .chat_template
                        .clone()
                        .or(self.chat_template.clone()),
                )
                .with_jinja_explicit(
                    model_config
                        .jinja_explicit
                        .clone()
                        .or(self.jinja_explicit.clone()),
                )
                .build()?;

            let mapper = init_mapper(
                &model_config
                    .num_device_layers
                    .clone()
                    .or(self.num_device_layers.clone()),
                &auto_device_map_params,
            );

            let isq = model_config
                .in_situ_quant
                .as_ref()
                .or(self.in_situ_quant.as_ref())
                .and_then(|isq| parse_isq_value(isq, Some(&device)).ok());

            let pipeline: LoadedPipeline = loader.load_model_from_hf(
                None,
                self.token_source.clone(),
                &dtype,
                &device,
                false,
                mapper,
                isq,
                cache_config,
            )?;

            // Use the pipeline's name() as the canonical ID, but allow an alias.
            let pipeline_name = pipeline.lock().await.name();
            let primary_id = model_config
                .alias
                .clone()
                .unwrap_or_else(|| pipeline_name.clone());

            if !registered_ids.insert(primary_id.clone()) {
                anyhow::bail!(
                    "Model ID conflict: '{}' is already registered (config key: {}).",
                    primary_id,
                    model_config.model_id
                );
            }

            // Add the model to the MistralRs instance
            let engine_config = mistralrs_core::EngineConfig {
                no_kv_cache: self.no_kv_cache,
                no_prefix_cache: false,
                prefix_cache_n: self.prefix_cache_n,
                disable_eos_stop: false,
                throughput_logging_enabled: !self.interactive_mode,
                search_embedding_model,
                search_callback: self.search_callback.clone(),
                tool_callbacks: HashMap::new(),
                tool_callbacks_with_tools: HashMap::new(),
            };

            let mut add_model_config = mistralrs_core::AddModelConfig::new(engine_config);
            if let Some(mcp_config) = self.mcp_client_config.clone() {
                add_model_config = add_model_config.with_mcp_config(mcp_config);
            }

            mistralrs
                .add_model(
                    primary_id.clone(),
                    pipeline,
                    scheduler_config.clone(),
                    add_model_config,
                )
                .await
                .map_err(|e| anyhow::anyhow!("Failed to add model {}: {}", primary_id, e))?;

            if let Some(alias) = model_config.alias.as_ref() {
                if alias != &pipeline_name {
                    mistralrs
                        .register_model_alias(pipeline_name.clone(), &primary_id)
                        .map_err(|e| anyhow::anyhow!(e))?;
                }
            }

            if primary_id == pipeline_name {
                info!(
                    "Model `{}` registered successfully (from config key: {})",
                    primary_id, model_config.model_id
                );
            } else {
                info!(
                    "Model `{}` registered successfully (pipeline: `{}`; config key: {})",
                    primary_id, pipeline_name, model_config.model_id
                );
            }
            loaded_model_ids.push(primary_id);
        }

        // Set the default model if specified
        if let Some(ref default_model_id) = self.default_model_id {
            mistralrs
                .set_default_model_id(default_model_id)
                .map_err(|e| anyhow::anyhow!("Failed to set default model: {}", e))?;
        }

        // Log all models loaded
        info!("All models loaded: `{}`", loaded_model_ids.join("`, `"));

        // Log default model
        if let Some(ref default_id) = self.default_model_id {
            info!("Default model: {}", default_id);
        } else {
            info!(
                "Default model: {} (first model, from config key: {})",
                loaded_model_ids[0], self.models[0].model_id
            );
        }
        Ok(mistralrs)
    }
}

// TODO: replace with best device?
/// Initializes the device to be used for computation, optionally forcing CPU usage and setting a seed.
fn init_device(force_cpu: bool, seed: Option<u64>) -> Result<candle_core::Device> {
    #[cfg(feature = "metal")]
    let device = if force_cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    #[cfg(not(feature = "metal"))]
    #[allow(clippy::if_same_then_else)]
    let device = if force_cpu {
        Device::Cpu
    } else if mistralrs_core::distributed::use_nccl() {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    if let Some(seed) = seed {
        device.set_seed(seed)?;
    }

    Ok(device)
}

/// Initializes the device mapping configuration for distributing model layers.
fn init_mapper(
    num_device_layers: &Option<Vec<String>>,
    auto_device_map_params: &AutoDeviceMapParams,
) -> DeviceMapSetting {
    // Parse device mapper
    if let Some(device_layers) = num_device_layers {
        if device_layers.len() == 1 && device_layers[0].parse::<usize>().is_ok() {
            let layers = device_layers[0].parse::<usize>().unwrap();
            DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(vec![
                DeviceLayerMapMetadata { ordinal: 0, layers },
            ]))
        } else {
            let mut mapping = Vec::new();
            for layer in device_layers {
                let split = layer.splitn(2, ':').collect::<Vec<_>>();
                if split.len() < 2 {
                    panic!("Expected layer to be of format ORD:NUM, got {layer}");
                }
                let ord = split[0]
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("Failed to parse {} as integer.", split[0]));
                let num = split[1]
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("Failed to parse {} as integer.", split[1]));
                for DeviceLayerMapMetadata { ordinal, layers: _ } in &mapping {
                    if *ordinal == ord {
                        panic!("Duplicate ordinal {ord}");
                    }
                }
                mapping.push(DeviceLayerMapMetadata {
                    ordinal: ord,
                    layers: num,
                });
            }
            DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(mapping))
        }
    } else {
        DeviceMapSetting::Auto(auto_device_map_params.clone())
    }
}

/// Logs hardware feature information and the model's sampling strategy and kind.
fn mistralrs_instance_info(loader: &dyn Loader) {
    info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    info!("Sampling method: penalties -> temperature -> topk -> topp -> minp -> multinomial");
    info!("Model kind is: {}", loader.get_kind().to_string());
}

/// Determines whether paged attention should be enabled based on device type and preferences.
fn configure_paged_attn(device: &Device, paged_attn: Option<bool>) -> bool {
    if device.is_cpu() {
        if paged_attn == Some(true) {
            warn!("Paged attention is not supported on CPU.");
        }

        defaults::PAGED_ATTN_CPU
    } else if device.is_cuda() || mistralrs_core::distributed::use_nccl() {
        paged_attn.unwrap_or(defaults::PAGED_ATTN_CUDA)
    } else if device.is_metal() {
        paged_attn.unwrap_or(defaults::PAGED_ATTN_METAL)
    } else {
        false
    }
}

/// Initializes the cache configuration for paged attention based on provided parameters.
fn init_cache_config(
    paged_attn_block_size: Option<usize>,
    paged_attn_gpu_mem: Option<usize>,
    paged_attn_gpu_mem_usage: Option<f32>,
    paged_ctxt_len: Option<usize>,
    cache_type: PagedCacheType,
    no_paged_attn: bool,
) -> Result<Option<PagedAttentionConfig>> {
    match (
        paged_attn_block_size,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_supported(),
        no_paged_attn,
    ) {
        (block_size, None, None, None, true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::Utilization(0.9),
            cache_type,
        )?)),
        (block_size, None, None, Some(ctxt), true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::ContextSize(ctxt),
            cache_type,
        )?)),
        (block_size, None, Some(f), None, true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::Utilization(f),
            cache_type,
        )?)),
        (block_size, Some(m), None, None, true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::MbAmount(m),
            cache_type,
        )?)),
        (block_size, Some(_m), Some(f), None, true, false) => {
            info!("Both memory size, and usage were specified, defaulting to the usage value.");
            Ok(Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::Utilization(f),
                cache_type,
            )?))
        }
        (block_size, Some(_m), None, Some(ctxt), true, false) => {
            info!("All memory size and ctxt len, defaulting to the context len value.");
            Ok(Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::ContextSize(ctxt),
                cache_type,
            )?))
        }
        (block_size, None, Some(f), Some(_ctxt), true, false) => {
            info!("Both ctxt len and usage were specified, defaulting to the usage value.");
            Ok(Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::Utilization(f),
                cache_type,
            )?))
        }
        (_, _, _, _, _, _) => Ok(None),
    }
}

/// Initializes the scheduler configuration based on cache settings and pipeline metadata.
async fn init_scheduler_config(
    cache_config: &Option<PagedAttentionConfig>,
    pipeline: &LoadedPipeline,
    args_max_seqs: usize,
) -> SchedulerConfig {
    if cache_config.is_some() {
        // Handle case where we may have device mapping
        if let Some(ref cache_config) = pipeline.lock().await.get_metadata().cache_config {
            SchedulerConfig::PagedAttentionMeta {
                max_num_seqs: args_max_seqs,
                config: cache_config.clone(),
            }
        } else {
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(args_max_seqs.try_into().unwrap()),
            }
        }
    } else {
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(args_max_seqs.try_into().unwrap()),
        }
    }
}

/// Configures PagedAttention based on two flags.
///
/// This function resolves the tri-state PagedAttention configuration from
/// the mutually exclusive `paged_attn` and `no_paged_attn` flags.
pub fn configure_paged_attn_from_flags(
    paged_attn: bool,
    no_paged_attn: bool,
) -> Result<Option<bool>> {
    match (paged_attn, no_paged_attn) {
        (true, true) => {
            anyhow::bail!("Error: `--paged-attn` and `--no-paged-attn` cannot be used together.");
        }
        (true, false) => Ok(Some(true)),
        (false, true) => Ok(Some(false)),
        (false, false) => Ok(None),
    }
}

/// Creates a search embedding model configuration for agentic search reranking.
pub fn get_search_embedding_model(
    enable_search: bool,
    search_embedding_model: Option<SearchEmbeddingModel>,
) -> Option<SearchEmbeddingModel> {
    if enable_search {
        Some(search_embedding_model.unwrap_or_default())
    } else {
        None
    }
}
