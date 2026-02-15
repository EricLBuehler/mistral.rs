#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use candle_core::Device;
use engine::Engine;
pub use engine::{
    get_engine_terminate_flag, reset_engine_terminate_flag, should_terminate_engine_sequences,
    EngineInstruction, IntervalLogger, SearchEmbeddingModel, ENGINE_INSTRUCTIONS,
    TERMINATE_ALL_NEXT_STEP,
};
use hf_hub::Cache;
pub use lora::Ordering;
pub use pipeline::ModelCategory;
pub use pipeline::Pipeline;
#[cfg(feature = "pyo3_macros")]
use pyo3::exceptions::PyValueError;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use std::{
    cell::RefCell,
    error::Error,
    fs::OpenOptions,
    io::Write,
    sync::{atomic::AtomicBool, Arc, Mutex, RwLock},
    thread::{self, JoinHandle},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc::{channel, Sender};
use tracing::info;
use tracing::warn;

pub const MISTRALRS_GIT_REVISION: &str = match option_env!("MISTRALRS_GIT_REVISION") {
    Some(value) => value,
    None => "unknown",
};

mod cuda;
mod device_map;
mod engine;
mod lora;
mod model_loader;
mod moe;
mod ops;
pub use model_loader::{
    get_auto_device_map_params, get_model_dtype, get_tgt_non_granular_index, LoaderBuilder,
};
mod embedding_models;
mod kv_cache;
mod search;

mod model_selected;
pub use model_selected::ModelSelected;
pub use toml_selector::{get_toml_selected_model_device_map_params, get_toml_selected_model_dtype};

mod amoe;
mod attention;
mod diagnostics;
mod diffusion_models;
pub mod distributed;
mod gguf;
pub mod harmony;
pub mod layers;
mod layers_masker;
mod layers_utils;
pub mod matformer;
mod mla;
mod models;
mod paged_attention;
mod pipeline;
mod prefix_cacher;
mod request;
mod response;
mod sampler;
mod scheduler;
mod sequence;
mod speech_models;
pub mod think_tags;
mod toml_selector;
mod tools;
mod topology;
mod utils;
mod vision_models;
mod xlora_models;

pub use diagnostics::{
    check_hf_gated_access, collect_system_info, run_doctor, BuildInfo, CpuInfo, DeviceInfo,
    DoctorCheck, DoctorReport, DoctorStatus, HfConnectivityInfo, MemoryInfo, SystemInfo,
};
mod tuning;
pub use tuning::{
    auto_tune, AutoTuneRequest, AutoTuneResult, FitStatus, QualityTier, TuneCandidate, TuneProfile,
};

pub use amoe::{AnyMoeConfig, AnyMoeExpertType};
pub use device_map::{
    DeviceLayerMapMetadata, DeviceMapMetadata, DeviceMapSetting, LayerDeviceMapper,
};
pub use gguf::{GGUFArchitecture, GGUF_MULTI_FILE_DELIMITER};
pub use mistralrs_audio::AudioInput;
pub use mistralrs_mcp::{
    CalledFunction, Function, Tool, ToolCallback, ToolCallbackWithTool, ToolType,
};
pub use mistralrs_mcp::{
    McpClient, McpClientConfig, McpServerConfig, McpServerSource, McpToolInfo,
};
pub use mistralrs_quant::{IsqType, MULTI_LORA_DELIMITER};
pub use paged_attention::{MemoryGpuConfig, PagedAttentionConfig, PagedCacheType};
pub use pipeline::hf::{hf_home_dir, hf_hub_cache_dir, hf_token_path};
pub use pipeline::{
    chat_template::ChatTemplate, parse_isq_value, AdapterPaths, AnyMoeLoader, AnyMoePipeline,
    AutoDeviceMapParams, AutoLoader, AutoLoaderBuilder, DiffusionGenerationParams, DiffusionLoader,
    DiffusionLoaderBuilder, DiffusionLoaderType, EmbeddingLoader, EmbeddingLoaderBuilder,
    EmbeddingLoaderType, EmbeddingModelPaths, EmbeddingSpecificConfig, GGMLLoader,
    GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoader, GGUFLoaderBuilder, GGUFSpecificConfig,
    GemmaLoader, Idefics2Loader, IsqOrganization, LLaVALoader, LLaVANextLoader, LlamaLoader,
    Loader, LocalModelPaths, LoraAdapterPaths, MistralLoader, MixtralLoader, Modalities, ModelKind,
    ModelPaths, MultimodalPromptPrefixer, NormalLoader, NormalLoaderBuilder, NormalLoaderType,
    NormalSpecificConfig, Phi2Loader, Phi3Loader, Phi3VLoader, Qwen2Loader, SpeculativeConfig,
    SpeculativeLoader, SpeculativePipeline, SpeechLoader, SpeechPipeline, Starcoder2Loader,
    SupportedModality, TokenSource, VisionLoader, VisionLoaderBuilder, VisionLoaderType,
    VisionSpecificConfig, UQFF_MULTI_FILE_DELIMITER,
};
pub use request::{
    ApproximateUserLocation, Constraint, DetokenizationRequest, ImageGenerationResponseFormat,
    LlguidanceGrammar, MessageContent, NormalRequest, ReasoningEffort, Request, RequestMessage,
    SearchContextSize, TokenizationRequest, WebSearchOptions, WebSearchUserLocation,
};
pub use response::*;
pub use sampler::{
    CustomLogitsProcessor, DrySamplingParams, SamplingParams, StopTokens, TopLogprob,
};
pub use scheduler::{DefaultSchedulerMethod, SchedulerConfig};
pub use search::{SearchCallback, SearchFunctionParameters, SearchResult};
use serde::Serialize;
pub use speech_models::{utils as speech_utils, SpeechGenerationConfig, SpeechLoaderType};
use tokio::runtime::Runtime;
use toml_selector::{TomlLoaderArgs, TomlSelector};
pub use tools::{ToolCallResponse, ToolCallType, ToolCallbacks, ToolChoice};
pub use topology::{LayerTopology, Topology};
pub use utils::debug::initialize_logging;
pub use utils::memory_usage::MemoryUsage;
pub use utils::normal::{ModelDType, TryIntoDType};
pub use utils::{paged_attn_supported, using_flash_attn};

// re-export llguidance for easier LlguidanceGrammar construction
pub use llguidance;

/// `true` if `MISTRALRS_DEBUG=1`
pub(crate) static DEBUG: AtomicBool = AtomicBool::new(false);
pub static GLOBAL_HF_CACHE: OnceLock<Cache> = OnceLock::new();

/// Configuration for creating an engine instance
#[derive(Clone)]
pub struct EngineConfig {
    pub no_kv_cache: bool,
    pub no_prefix_cache: bool,
    pub prefix_cache_n: usize,
    pub disable_eos_stop: bool,
    pub throughput_logging_enabled: bool,
    pub search_embedding_model: Option<SearchEmbeddingModel>,
    pub search_callback: Option<Arc<SearchCallback>>,
    pub tool_callbacks: tools::ToolCallbacks,
    pub tool_callbacks_with_tools: tools::ToolCallbacksWithTools,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            no_kv_cache: false,
            no_prefix_cache: false,
            prefix_cache_n: 16,
            disable_eos_stop: false,
            throughput_logging_enabled: true,
            search_embedding_model: None,
            search_callback: None,
            tool_callbacks: HashMap::new(),
            tool_callbacks_with_tools: HashMap::new(),
        }
    }
}

/// Configuration for adding a model to MistralRs
#[derive(Clone)]
pub struct AddModelConfig {
    pub engine_config: EngineConfig,
    pub mcp_client_config: Option<McpClientConfig>,
    /// Optional loader config for enabling model unload/reload support.
    /// Without this, models cannot be unloaded and reloaded.
    pub loader_config: Option<ModelLoaderConfig>,
}

impl AddModelConfig {
    pub fn new(engine_config: EngineConfig) -> Self {
        Self {
            engine_config,
            mcp_client_config: None,
            loader_config: None,
        }
    }

    pub fn with_mcp_config(mut self, mcp_config: McpClientConfig) -> Self {
        self.mcp_client_config = Some(mcp_config);
        self
    }

    /// Set the loader config for enabling model unload/reload support.
    /// Without this, models cannot be unloaded and reloaded.
    pub fn with_loader_config(mut self, loader_config: ModelLoaderConfig) -> Self {
        self.loader_config = Some(loader_config);
        self
    }
}

#[derive(Clone)]
pub struct MistralRsConfig {
    pub kind: ModelKind,
    pub device: Device,
    pub category: ModelCategory,
    pub modalities: Modalities,
    pub max_seq_len: Option<usize>,
}

/// Configuration for recreating a model loader when reloading an unloaded model.
/// This captures the essential parameters needed to reconstruct a loader.
#[derive(Clone)]
pub struct ModelLoaderConfig {
    /// The model selection configuration (Plain, GGUF, Vision, etc.)
    pub model_selected: ModelSelected,
    /// Source of the HF token
    pub token_source: TokenSource,
    /// Optional HF revision
    pub hf_revision: Option<String>,
    /// Model data type
    pub dtype: ModelDType,
    /// Device to load the model on
    pub device: Device,
    /// Device mapping setting
    pub device_map_setting: DeviceMapSetting,
    /// In-situ quantization type
    pub isq: Option<IsqType>,
    /// Paged attention configuration
    pub paged_attn_config: Option<PagedAttentionConfig>,
    /// Whether to suppress logging during loading
    pub silent: bool,
    /// Chat template override
    pub chat_template: Option<String>,
    /// Explicit Jinja template path
    pub jinja_explicit: Option<String>,
}

/// State preserved when a model is unloaded.
/// This contains all the information needed to reload the model on demand.
#[derive(Clone)]
pub struct UnloadedModelState {
    /// Configuration to recreate the loader
    pub loader_config: ModelLoaderConfig,
    /// Scheduler configuration
    pub scheduler_config: SchedulerConfig,
    /// Engine configuration
    pub engine_config: EngineConfig,
    /// MCP client configuration
    pub mcp_client_config: Option<McpClientConfig>,
    /// Model category (Text, Vision, etc.)
    pub category: ModelCategory,
    /// Model metadata configuration
    pub mistralrs_config: MistralRsConfig,
}

/// Internal structure to hold per-engine state
struct EngineInstance {
    sender: Sender<Request>,
    engine_handler: JoinHandle<()>,
    reboot_state: RebootState,
    config: MistralRsConfig,
    category: ModelCategory,
    logger: Arc<IntervalLogger>,
}

/// The MistralRs struct handles sending requests to multiple engines.
/// It is the core multi-threaded component of mistral.rs, and uses `mpsc`
/// `Sender` and `Receiver` primitives to send and receive requests to the
/// appropriate engine based on model ID.
///
/// ## Lock Ordering Convention
///
/// This struct uses multiple `RwLock`s. To prevent deadlocks, locks must be
/// acquired in this order:
/// 1. `reloading_models`
/// 2. `engines`
/// 3. `unloaded_models`
/// 4. `default_engine_id`
/// 5. `model_aliases`
///
/// Use scope-based lock management and explicit `drop()` calls.
pub struct MistralRs {
    engines: RwLock<HashMap<String, EngineInstance>>,
    /// Models that have been unloaded but can be reloaded on demand
    unloaded_models: RwLock<HashMap<String, UnloadedModelState>>,
    /// Models currently being reloaded (to prevent concurrent reloads)
    reloading_models: RwLock<HashSet<String>>,
    default_engine_id: RwLock<Option<String>>,
    /// Alternate IDs that resolve to primary model IDs.
    model_aliases: RwLock<HashMap<String, String>>,
    log: Option<String>,
    id: String,
    creation_time: u64,
    next_request_id: Mutex<RefCell<usize>>,
}

#[derive(Clone)]
struct RebootState {
    pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
    method: SchedulerConfig,
    no_kv_cache: bool,
    no_prefix_cache: bool,
    prefix_cache_n: usize,
    disable_eos_stop: bool,
    throughput_logging_enabled: bool,
    search_embedding_model: Option<SearchEmbeddingModel>,
    search_callback: Option<Arc<search::SearchCallback>>,
    tool_callbacks: tools::ToolCallbacks,
    tool_callbacks_with_tools: tools::ToolCallbacksWithTools,
    mcp_client_config: Option<McpClientConfig>,
    /// Optional loader config for reloading after unload
    loader_config: Option<ModelLoaderConfig>,
}

/// Model status for loaded/unloaded state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    Loaded,
    Unloaded,
    Reloading,
}

impl std::fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelStatus::Loaded => write!(f, "loaded"),
            ModelStatus::Unloaded => write!(f, "unloaded"),
            ModelStatus::Reloading => write!(f, "reloading"),
        }
    }
}

#[derive(Debug)]
pub enum MistralRsError {
    EnginePoisoned,
    SenderPoisoned,
    /// The requested model was not found (neither loaded nor unloaded)
    ModelNotFound(String),
    /// The model is currently being reloaded
    ModelReloading(String),
    /// Failed to reload the model
    ReloadFailed(String),
    /// Model does not have loader config for reloading
    NoLoaderConfig(String),
    /// Model is already loaded
    ModelAlreadyLoaded(String),
    /// Model is already unloaded
    ModelAlreadyUnloaded(String),
}

impl std::fmt::Display for MistralRsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self)
    }
}

impl std::error::Error for MistralRsError {}

#[cfg(feature = "pyo3_macros")]
impl From<MistralRsError> for pyo3::PyErr {
    fn from(value: MistralRsError) -> Self {
        PyValueError::new_err(format!("{value:?}"))
    }
}

/// The MistralRsBuilder takes the pipeline and a scheduler method and constructs
/// an Engine and a MistralRs instance. The Engine runs on a separate thread, and the MistralRs
/// instance stays on the calling thread.
pub struct MistralRsBuilder {
    pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
    method: SchedulerConfig,
    model_id_override: Option<String>,
    log: Option<String>,
    no_kv_cache: Option<bool>,
    no_prefix_cache: Option<bool>,
    prefix_cache_n: Option<usize>,
    disable_eos_stop: Option<bool>,
    throughput_logging_enabled: bool,
    search_embedding_model: Option<SearchEmbeddingModel>,
    search_callback: Option<Arc<SearchCallback>>,
    tool_callbacks: tools::ToolCallbacks,
    tool_callbacks_with_tools: tools::ToolCallbacksWithTools,
    mcp_client_config: Option<McpClientConfig>,
    loader_config: Option<ModelLoaderConfig>,
}

impl MistralRsBuilder {
    /// Creates a new builder with the given pipeline, scheduler method, logging flag,
    /// and optional embedding model for web search. To override the search callback,
    /// use `.with_search_callback(...)` on the builder.
    pub fn new(
        pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        method: SchedulerConfig,
        throughput_logging: bool,
        search_embedding_model: Option<SearchEmbeddingModel>,
    ) -> Self {
        Self {
            pipeline,
            method,
            model_id_override: None,
            log: None,
            no_kv_cache: None,
            no_prefix_cache: None,
            prefix_cache_n: None,
            disable_eos_stop: None,
            throughput_logging_enabled: throughput_logging,
            search_embedding_model,
            search_callback: None,
            tool_callbacks: HashMap::new(),
            tool_callbacks_with_tools: HashMap::new(),
            mcp_client_config: None,
            loader_config: None,
        }
    }

    /// Override the model ID used by MistralRs. Defaults to the pipeline name.
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id_override = Some(model_id.into());
        self
    }

    /// Set the loader config for enabling model unload/reload support.
    /// Without this, models cannot be unloaded and reloaded.
    pub fn with_loader_config(mut self, loader_config: ModelLoaderConfig) -> Self {
        self.loader_config = Some(loader_config);
        self
    }
    pub fn with_log(mut self, log: String) -> Self {
        self.log = Some(log);
        self
    }
    pub fn with_opt_log(mut self, log: Option<String>) -> Self {
        self.log = log;
        self
    }
    pub fn with_no_kv_cache(mut self, no_kv_cache: bool) -> Self {
        self.no_kv_cache = Some(no_kv_cache);
        self
    }
    pub fn with_no_prefix_cache(mut self, no_prefix_cache: bool) -> Self {
        self.no_prefix_cache = Some(no_prefix_cache);
        self
    }
    pub fn with_prefix_cache_n(mut self, prefix_cache_n: usize) -> Self {
        self.prefix_cache_n = Some(prefix_cache_n);
        self
    }
    pub fn with_disable_eos_stop(mut self, disable_eos_stop: bool) -> Self {
        self.disable_eos_stop = Some(disable_eos_stop);
        self
    }

    /// Use a custom callback to gather search results.
    pub fn with_search_callback(mut self, search_callback: Arc<SearchCallback>) -> Self {
        self.search_callback = Some(search_callback);
        self
    }

    /// Register a custom callback for the specified tool name.
    pub fn with_tool_callback(
        mut self,
        name: impl Into<String>,
        tool_callback: Arc<ToolCallback>,
    ) -> Self {
        self.tool_callbacks.insert(name.into(), tool_callback);
        self
    }

    /// Register a custom callback with its associated Tool definition. The Tool will be
    /// automatically added to requests when tool callbacks are active.
    pub fn with_tool_callback_and_tool(
        mut self,
        name: impl Into<String>,
        tool_callback: Arc<ToolCallback>,
        tool: Tool,
    ) -> Self {
        let name = name.into();
        self.tool_callbacks_with_tools.insert(
            name,
            ToolCallbackWithTool {
                callback: tool_callback,
                tool,
            },
        );
        self
    }

    /// Configure MCP client to connect to external MCP servers.
    pub fn with_mcp_client(mut self, config: McpClientConfig) -> Self {
        self.mcp_client_config = Some(config);
        self
    }

    pub async fn build(self) -> Arc<MistralRs> {
        MistralRs::new(self).await
    }
}

impl Drop for MistralRs {
    fn drop(&mut self) {
        // Terminate all engines
        if let Ok(engines) = self.engines.read() {
            for (_, engine) in engines.iter() {
                // Use try_send instead of blocking_send to avoid runtime panics
                let _ = engine.sender.try_send(Request::Terminate);
            }
        }
    }
}

impl MistralRs {
    /// Create an engine instance with the given configuration
    fn create_engine_instance(
        pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        method: SchedulerConfig,
        config: EngineConfig,
        reboot_state: RebootState,
    ) -> Result<EngineInstance, String> {
        let (tx, rx) = channel(10_000);

        let pipeline_guard = pipeline.try_lock().unwrap();
        let category = pipeline_guard.category();
        let metadata = pipeline_guard.get_metadata();
        let kind = metadata.kind.clone();
        let device = pipeline_guard.device();
        let modalities = metadata.modalities.clone();
        let max_seq_len = match &category {
            ModelCategory::Diffusion | ModelCategory::Speech => None,
            _ => Some(metadata.max_seq_len),
        };
        let encoder_cache_counters = pipeline_guard.encoder_cache_counters();
        drop(pipeline_guard);

        let logger = Arc::new(IntervalLogger::new(
            Duration::from_secs(5),
            encoder_cache_counters,
        ));
        let logger_for_engine = logger.clone();

        info!("Pipeline input modalities are {:?}", &modalities.input);
        info!("Pipeline output modalities are {:?}", &modalities.output);

        let mistralrs_config = MistralRsConfig {
            kind,
            device,
            category: category.clone(),
            modalities,
            max_seq_len,
        };

        let tx_for_engine = tx.clone();
        let engine_handler = thread::spawn(move || {
            #[cfg(feature = "metal")]
            objc::rc::autoreleasepool(move || {
                let rt = Runtime::new().unwrap();
                rt.block_on(async move {
                    let engine = Engine::new(
                        tx_for_engine,
                        rx,
                        pipeline,
                        method,
                        config.no_kv_cache,
                        config.no_prefix_cache,
                        config.prefix_cache_n,
                        config.disable_eos_stop,
                        config.throughput_logging_enabled,
                        config.search_embedding_model,
                        config.search_callback.clone(),
                        config.tool_callbacks.clone(),
                        config.tool_callbacks_with_tools.clone(),
                        logger_for_engine,
                    )
                    .expect("Engine creation failed.");
                    Arc::new(engine).run().await;
                })
            });

            #[cfg(not(feature = "metal"))]
            {
                let rt = Runtime::new().unwrap();
                rt.block_on(async move {
                    let engine = Engine::new(
                        tx_for_engine,
                        rx,
                        pipeline,
                        method,
                        config.no_kv_cache,
                        config.no_prefix_cache,
                        config.prefix_cache_n,
                        config.disable_eos_stop,
                        config.throughput_logging_enabled,
                        config.search_embedding_model,
                        config.search_callback.clone(),
                        config.tool_callbacks.clone(),
                        config.tool_callbacks_with_tools.clone(),
                        logger_for_engine,
                    )
                    .expect("Engine creation failed.");
                    Arc::new(engine).run().await;
                })
            }
        });

        Ok(EngineInstance {
            sender: tx,
            engine_handler,
            reboot_state,
            config: mistralrs_config,
            category,
            logger,
        })
    }

    async fn new(config: MistralRsBuilder) -> Arc<Self> {
        info!("git revision: {MISTRALRS_GIT_REVISION}");
        let MistralRsBuilder {
            pipeline,
            method,
            model_id_override,
            log,
            no_kv_cache,
            no_prefix_cache,
            prefix_cache_n,
            disable_eos_stop,
            throughput_logging_enabled,
            search_embedding_model,
            search_callback,
            tool_callbacks,
            mut tool_callbacks_with_tools,
            mcp_client_config,
            loader_config,
        } = config;

        mistralrs_quant::cublaslt::maybe_init_cublas_lt_wrapper(
            get_mut_arcmutex!(pipeline).device(),
        );

        // For hybrid models (Mamba-Attention), force batch_size=1 to prevent state bleeding
        // Mamba's stateful nature makes batched inference complex; this ensures correctness
        let method = if !get_mut_arcmutex!(pipeline).get_metadata().no_kv_cache
            && get_mut_arcmutex!(pipeline).cache().is_hybrid()
        {
            info!(
                "Hybrid model detected (Mamba-Attention), enforcing batch_size=1 for correctness"
            );
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(NonZeroUsize::new(1).unwrap()),
            }
        } else {
            method
        };

        let no_kv_cache = no_kv_cache.unwrap_or(false);
        let no_prefix_cache = no_prefix_cache.unwrap_or(false);
        let prefix_cache_n = prefix_cache_n.unwrap_or(16);
        let disable_eos_stop = disable_eos_stop.unwrap_or(false);

        // Initialize MCP client if configured
        if let Some(config) = &mcp_client_config {
            let mut mcp_client = McpClient::new(config.clone());
            let total_servers = config.servers.len();

            match mcp_client.initialize().await {
                Ok(()) => {
                    let mcp_callbacks_with_tools = mcp_client.get_tool_callbacks_with_tools();
                    let tools_count = mcp_callbacks_with_tools.len();

                    // Merge MCP tool callbacks with tools into the new collection
                    for (name, callback_with_tool) in mcp_callbacks_with_tools {
                        tool_callbacks_with_tools.insert(name.clone(), callback_with_tool.clone());
                    }

                    if tools_count == 0 {
                        warn!(
                            "MCP client initialized but no tools were registered from {} servers",
                            total_servers
                        );
                    } else {
                        info!(
                            "MCP client initialized successfully with {} tools from {} servers",
                            tools_count, total_servers
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to initialize MCP client with {} configured servers: {}",
                        total_servers, e
                    );
                    warn!("Continuing without MCP functionality. Check your MCP configuration and server availability.");
                }
            }
        }

        let reboot_state = RebootState {
            pipeline: pipeline.clone(),
            method: method.clone(),
            no_kv_cache,
            no_prefix_cache,
            prefix_cache_n,
            disable_eos_stop,
            throughput_logging_enabled,
            search_embedding_model,
            search_callback: search_callback.clone(),
            tool_callbacks: tool_callbacks.clone(),
            tool_callbacks_with_tools: tool_callbacks_with_tools.clone(),
            mcp_client_config: mcp_client_config.clone(),
            loader_config,
        };

        // Create the engine configuration
        let engine_config = EngineConfig {
            no_kv_cache,
            no_prefix_cache,
            prefix_cache_n,
            disable_eos_stop,
            throughput_logging_enabled,
            search_embedding_model,
            search_callback,
            tool_callbacks,
            tool_callbacks_with_tools,
        };

        // Create the engine instance
        let engine_instance =
            Self::create_engine_instance(pipeline.clone(), method, engine_config, reboot_state)
                .expect("Failed to create engine instance");

        let pipeline_name = pipeline.try_lock().unwrap().name();
        let (id, alias_map) = match model_id_override {
            Some(override_id) => {
                let mut alias_map = HashMap::new();
                if override_id != pipeline_name {
                    alias_map.insert(pipeline_name.clone(), override_id.clone());
                }
                (override_id, alias_map)
            }
            None => (pipeline_name.clone(), HashMap::new()),
        };

        if distributed::is_daemon() {
            let request_sender = engine_instance.sender.clone();

            if cfg!(feature = "ring") {
                // Ring daemon replicator
                distributed::ring_daemon_replicator(request_sender);
            } else {
                // NCCL daemon replicator
                distributed::nccl_daemon_replicator(request_sender);
            }

            #[allow(clippy::empty_loop)]
            loop {}
        }

        // Determine if the current runtime is multi-threaded, as blocking operations are not allowed in single-threaded mode
        let is_multi_threaded = tokio::runtime::Handle::try_current()
            .is_ok_and(|h| h.runtime_flavor() != tokio::runtime::RuntimeFlavor::CurrentThread);

        // Do a dummy run
        if !distributed::is_daemon()
            && is_multi_threaded
            && matches!(
                engine_instance.category,
                ModelCategory::Text | ModelCategory::Vision { .. }
            )
        {
            let clone_sender = engine_instance.sender.clone();
            tokio::task::block_in_place(|| {
                let (tx, mut rx) = channel(1);
                let req = Request::Normal(Box::new(NormalRequest {
                    id: 0,
                    messages: RequestMessage::Completion {
                        text: "hello".to_string(),
                        echo_prompt: false,
                        best_of: None,
                    },
                    sampling_params: SamplingParams {
                        max_len: Some(1),
                        ..SamplingParams::deterministic()
                    },
                    response: tx,
                    return_logprobs: false,
                    is_streaming: false,
                    constraint: Constraint::None,
                    suffix: None,
                    tool_choice: None,
                    tools: None,
                    logits_processors: None,
                    return_raw_logits: false,
                    web_search_options: None,
                    model_id: None,
                    truncate_sequence: false,
                }));
                info!("Beginning dummy run.");
                let start = Instant::now();
                clone_sender.blocking_send(req).unwrap();

                // Drain all responses from the channel until it's closed
                let mut received_any = false;
                while let Some(_resp) = rx.blocking_recv() {
                    received_any = true;
                }

                if received_any {
                    let end = Instant::now();
                    info!(
                        "Dummy run completed in {}s.",
                        end.duration_since(start).as_secs_f64()
                    );
                } else {
                    warn!("Dummy run failed!");
                }
            });

            // Reset logger counters so the dummy run doesn't pollute stats
            engine_instance.logger.reset();
        }

        // Create engines map with the first engine
        let mut engines = HashMap::new();
        engines.insert(id.clone(), engine_instance);

        Arc::new(Self {
            engines: RwLock::new(engines),
            unloaded_models: RwLock::new(HashMap::new()),
            reloading_models: RwLock::new(HashSet::new()),
            default_engine_id: RwLock::new(Some(id.clone())),
            model_aliases: RwLock::new(alias_map),
            log,
            id,
            creation_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time travel has occurred!")
                .as_secs(),
            next_request_id: Mutex::new(RefCell::new(1)),
        })
    }

    /// Attempts to reboot a specific engine by model_id
    fn reboot_engine(&self, model_id: &str) -> Result<(), MistralRsError> {
        let mut engines = self.engines.write().map_err(|_| {
            tracing::warn!("Couldn't get write lock on engines during reboot attempt");
            MistralRsError::EnginePoisoned
        })?;

        if let Some(engine_instance) = engines.get(model_id) {
            if !engine_instance.engine_handler.is_finished() {
                tracing::info!("Engine {} already running, returning ok", model_id);
                return Ok(());
            }

            let reboot_state = engine_instance.reboot_state.clone();
            let engine_config = EngineConfig {
                no_kv_cache: reboot_state.no_kv_cache,
                no_prefix_cache: reboot_state.no_prefix_cache,
                prefix_cache_n: reboot_state.prefix_cache_n,
                disable_eos_stop: reboot_state.disable_eos_stop,
                throughput_logging_enabled: reboot_state.throughput_logging_enabled,
                search_embedding_model: reboot_state.search_embedding_model,
                search_callback: reboot_state.search_callback.clone(),
                tool_callbacks: reboot_state.tool_callbacks.clone(),
                tool_callbacks_with_tools: reboot_state.tool_callbacks_with_tools.clone(),
            };
            let new_engine_instance = Self::create_engine_instance(
                reboot_state.pipeline.clone(),
                reboot_state.method.clone(),
                engine_config,
                reboot_state,
            )
            .map_err(|e| {
                tracing::error!("Failed to create new engine instance: {}", e);
                MistralRsError::EnginePoisoned
            })?;

            engines.insert(model_id.to_string(), new_engine_instance);
            tracing::info!("Successfully rebooted engine {}", model_id);
            Ok(())
        } else {
            Err(MistralRsError::EnginePoisoned)
        }
    }

    fn engine_dead(&self, model_id: &str) -> Result<bool, MistralRsError> {
        let engines = self.engines.read().map_err(|_| {
            tracing::warn!("Couldn't get read lock on engines!");
            MistralRsError::EnginePoisoned
        })?;

        if let Some(engine_instance) = engines.get(model_id) {
            Ok(engine_instance.engine_handler.is_finished())
        } else {
            Err(MistralRsError::EnginePoisoned)
        }
    }

    /// Get sender for a specific model. If model_id is None, uses default engine.
    /// If the model is unloaded, it will be automatically reloaded before returning the sender.
    pub fn get_sender(&self, model_id: Option<&str>) -> Result<Sender<Request>, MistralRsError> {
        let resolved_model_id = self.resolve_alias_or_default(model_id)?;

        // Check if model is loaded
        let is_loaded = {
            let engines = self
                .engines
                .read()
                .map_err(|_| MistralRsError::SenderPoisoned)?;
            engines.contains_key(&resolved_model_id)
        };

        if is_loaded {
            // Check if engine is dead and needs reboot
            if self.engine_dead(&resolved_model_id)? {
                tracing::warn!("Engine {} is dead, rebooting", resolved_model_id);
                self.reboot_engine(&resolved_model_id)?
            }

            let engines = self
                .engines
                .read()
                .map_err(|_| MistralRsError::SenderPoisoned)?;
            if let Some(engine_instance) = engines.get(&resolved_model_id) {
                return Ok(engine_instance.sender.clone());
            }
        }

        // Check if model is unloaded - trigger auto-reload
        let is_unloaded = {
            let unloaded = self
                .unloaded_models
                .read()
                .map_err(|_| MistralRsError::EnginePoisoned)?;
            unloaded.contains_key(&resolved_model_id)
        };

        if is_unloaded {
            tracing::info!(
                "Model {} is unloaded, triggering auto-reload",
                resolved_model_id
            );
            self.reload_model_blocking(&resolved_model_id)?;

            // After reload, get the sender
            let engines = self
                .engines
                .read()
                .map_err(|_| MistralRsError::SenderPoisoned)?;
            if let Some(engine_instance) = engines.get(&resolved_model_id) {
                return Ok(engine_instance.sender.clone());
            }
        }

        Err(MistralRsError::ModelNotFound(resolved_model_id))
    }

    pub fn get_id(&self) -> String {
        self.id.clone()
    }

    pub fn get_creation_time(&self) -> u64 {
        self.creation_time
    }

    fn resolve_alias(&self, model_id: &str) -> Result<String, MistralRsError> {
        let aliases = self
            .model_aliases
            .read()
            .map_err(|_| MistralRsError::SenderPoisoned)?;
        if let Some(primary_id) = aliases.get(model_id) {
            Ok(primary_id.clone())
        } else {
            Ok(model_id.to_string())
        }
    }

    fn resolve_alias_or_default(&self, model_id: Option<&str>) -> Result<String, MistralRsError> {
        match model_id {
            Some(id) => self.resolve_alias(id),
            None => {
                let default_lock = self
                    .default_engine_id
                    .read()
                    .map_err(|_| MistralRsError::SenderPoisoned)?;
                Ok(default_lock
                    .as_ref()
                    .ok_or(MistralRsError::EnginePoisoned)?
                    .clone())
            }
        }
    }

    /// Register an alternate model ID that resolves to an existing model.
    pub fn register_model_alias(
        &self,
        alias: impl Into<String>,
        model_id: &str,
    ) -> Result<(), String> {
        let alias = alias.into();
        let resolved_model_id = self.resolve_alias(model_id).map_err(|e| e.to_string())?;

        if alias == resolved_model_id {
            return Ok(());
        }

        let reloading = self
            .reloading_models
            .read()
            .map_err(|_| "Failed to acquire read lock on reloading_models")?;
        let model_reloading = reloading.contains(&resolved_model_id);
        let alias_conflict = reloading.contains(&alias);
        drop(reloading);

        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        let model_loaded = engines.contains_key(&resolved_model_id);
        let alias_conflict = alias_conflict || engines.contains_key(&alias);
        drop(engines);

        let unloaded = self
            .unloaded_models
            .read()
            .map_err(|_| "Failed to acquire read lock on unloaded_models")?;
        let model_unloaded = unloaded.contains_key(&resolved_model_id);
        let alias_conflict = alias_conflict || unloaded.contains_key(&alias);
        drop(unloaded);

        if !(model_loaded || model_unloaded || model_reloading) {
            return Err(format!("Model {resolved_model_id} not found"));
        }

        if alias_conflict {
            return Err(format!(
                "Alias '{}' conflicts with an existing model ID",
                alias
            ));
        }

        let mut aliases = self
            .model_aliases
            .write()
            .map_err(|_| "Failed to acquire write lock on model_aliases")?;
        if let Some(existing) = aliases.get(&alias) {
            if existing == &resolved_model_id {
                return Ok(());
            }
            return Err(format!(
                "Alias '{}' is already assigned to model '{}'",
                alias, existing
            ));
        }
        aliases.insert(alias, resolved_model_id);
        Ok(())
    }

    /// Check if a model is known (loaded, unloaded, or reloading), resolving aliases if needed.
    pub fn model_exists(&self, model_id: &str) -> Result<bool, MistralRsError> {
        let resolved_model_id = self.resolve_alias(model_id)?;

        let reloading = self
            .reloading_models
            .read()
            .map_err(|_| MistralRsError::EnginePoisoned)?;
        if reloading.contains(&resolved_model_id) {
            return Ok(true);
        }
        drop(reloading);

        let engines = self
            .engines
            .read()
            .map_err(|_| MistralRsError::EnginePoisoned)?;
        if engines.contains_key(&resolved_model_id) {
            return Ok(true);
        }
        drop(engines);

        let unloaded = self
            .unloaded_models
            .read()
            .map_err(|_| MistralRsError::EnginePoisoned)?;
        if unloaded.contains_key(&resolved_model_id) {
            return Ok(true);
        }

        Ok(false)
    }

    /// Get the interval logger for a specific model. If model_id is None, uses default engine.
    pub fn get_logger(
        &self,
        model_id: Option<&str>,
    ) -> Result<Arc<IntervalLogger>, MistralRsError> {
        let resolved_model_id = self.resolve_alias_or_default(model_id)?;

        let engines = self
            .engines
            .read()
            .map_err(|_| MistralRsError::SenderPoisoned)?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.logger.clone())
        } else {
            Err(MistralRsError::EnginePoisoned)
        }
    }

    /// Get model category for a specific model. If model_id is None, uses default engine.
    pub fn get_model_category(
        &self,
        model_id: Option<&str>,
    ) -> Result<ModelCategory, MistralRsError> {
        let resolved_model_id = self.resolve_alias_or_default(model_id)?;

        let engines = self
            .engines
            .read()
            .map_err(|_| MistralRsError::SenderPoisoned)?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.category.clone())
        } else {
            Err(MistralRsError::EnginePoisoned)
        }
    }

    /// Get the maximum supported sequence length for a model, if applicable.
    pub fn max_sequence_length(
        &self,
        model_id: Option<&str>,
    ) -> Result<Option<usize>, MistralRsError> {
        let resolved_model_id = self.resolve_alias_or_default(model_id)?;

        let engines = self
            .engines
            .read()
            .map_err(|_| MistralRsError::SenderPoisoned)?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.config.max_seq_len)
        } else {
            Err(MistralRsError::EnginePoisoned)
        }
    }

    pub fn next_request_id(&self) -> usize {
        let l = self.next_request_id.lock().unwrap();
        let last = &mut *l.borrow_mut();
        let last_v = *last;
        *last += 1;
        last_v
    }

    /// Add a new model engine to the MistralRs instance
    pub async fn add_model(
        &self,
        model_id: String,
        pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        method: SchedulerConfig,
        config: AddModelConfig,
    ) -> Result<(), String> {
        {
            let reloading = self
                .reloading_models
                .read()
                .map_err(|_| "Failed to acquire read lock on reloading_models")?;
            if reloading.contains(&model_id) {
                return Err(format!("Model {model_id} is currently reloading"));
            }
        }
        {
            let engines = self
                .engines
                .read()
                .map_err(|_| "Failed to acquire read lock on engines")?;
            if engines.contains_key(&model_id) {
                return Err(format!("Model {model_id} already exists"));
            }
        }
        {
            let unloaded = self
                .unloaded_models
                .read()
                .map_err(|_| "Failed to acquire read lock on unloaded_models")?;
            if unloaded.contains_key(&model_id) {
                return Err(format!("Model {model_id} already exists (unloaded)"));
            }
        }
        {
            let aliases = self
                .model_aliases
                .read()
                .map_err(|_| "Failed to acquire read lock on model_aliases")?;
            if aliases.contains_key(&model_id) {
                return Err(format!(
                    "Model ID '{}' conflicts with an existing alias",
                    model_id
                ));
            }
        }

        // For hybrid models (Mamba-Attention), force batch_size=1 to prevent state bleeding
        let method = {
            let pipeline_guard = pipeline.try_lock().unwrap();
            if !pipeline_guard.get_metadata().no_kv_cache && pipeline_guard.cache().is_hybrid() {
                info!(
                    "Hybrid model detected (Mamba-Attention), enforcing batch_size=1 for correctness"
                );
                SchedulerConfig::DefaultScheduler {
                    method: DefaultSchedulerMethod::Fixed(NonZeroUsize::new(1).unwrap()),
                }
            } else {
                method
            }
        };

        let reboot_state = RebootState {
            pipeline: pipeline.clone(),
            method: method.clone(),
            no_kv_cache: config.engine_config.no_kv_cache,
            no_prefix_cache: config.engine_config.no_prefix_cache,
            prefix_cache_n: config.engine_config.prefix_cache_n,
            disable_eos_stop: config.engine_config.disable_eos_stop,
            throughput_logging_enabled: config.engine_config.throughput_logging_enabled,
            search_embedding_model: config.engine_config.search_embedding_model,
            search_callback: config.engine_config.search_callback.clone(),
            tool_callbacks: config.engine_config.tool_callbacks.clone(),
            tool_callbacks_with_tools: config.engine_config.tool_callbacks_with_tools.clone(),
            mcp_client_config: config.mcp_client_config.clone(),
            loader_config: config.loader_config.clone(),
        };

        let engine_instance =
            Self::create_engine_instance(pipeline, method, config.engine_config, reboot_state)?;

        let mut engines = self
            .engines
            .write()
            .map_err(|_| "Failed to acquire write lock on engines")?;
        engines.insert(model_id.clone(), engine_instance);

        // If this is the first model, set it as default
        if engines.len() == 1 {
            let mut default_lock = self
                .default_engine_id
                .write()
                .map_err(|_| "Failed to acquire write lock on default_engine_id")?;
            *default_lock = Some(model_id.clone());
            info!("First model added, setting '{}' as default", model_id);
        }

        Ok(())
    }

    /// Remove a model engine from the MistralRs instance
    pub fn remove_model(&self, model_id: &str) -> Result<(), String> {
        let resolved_model_id = self.resolve_alias(model_id).map_err(|e| e.to_string())?;
        let mut engines = self
            .engines
            .write()
            .map_err(|_| "Failed to acquire write lock on engines")?;

        if engines.len() <= 1 {
            return Err("Cannot remove the last model from MistralRs".to_string());
        }

        if let Some(engine_instance) = engines.remove(&resolved_model_id) {
            // Send terminate signal to the engine
            let _ = engine_instance.sender.blocking_send(Request::Terminate);

            // If this was the default engine, set a new default
            let mut default_lock = self
                .default_engine_id
                .write()
                .map_err(|_| "Failed to acquire write lock on default_engine_id")?;
            if let Some(ref default_id) = *default_lock {
                if default_id == &resolved_model_id {
                    // Set the first available engine as the new default
                    *default_lock = engines.keys().next().cloned();
                }
            }
            drop(default_lock);
            drop(engines);

            // Remove any aliases pointing to the removed model
            let mut aliases = self
                .model_aliases
                .write()
                .map_err(|_| "Failed to acquire write lock on model_aliases")?;
            aliases.retain(|_, target| target != &resolved_model_id);

            Ok(())
        } else {
            Err(format!("Model {resolved_model_id} not found"))
        }
    }

    /// List all available model IDs
    pub fn list_models(&self) -> Result<Vec<String>, String> {
        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        Ok(engines.keys().cloned().collect())
    }

    /// Get the current default model ID
    pub fn get_default_model_id(&self) -> Result<Option<String>, String> {
        let default_lock = self
            .default_engine_id
            .read()
            .map_err(|_| "Failed to acquire read lock on default_engine_id")?;
        Ok(default_lock.clone())
    }

    /// Set the default model ID
    pub fn set_default_model_id(&self, model_id: &str) -> Result<(), String> {
        let resolved_model_id = self.resolve_alias(model_id).map_err(|e| e.to_string())?;
        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        if !engines.contains_key(&resolved_model_id) {
            return Err(format!("Model {resolved_model_id} not found"));
        }
        drop(engines);

        let mut default_lock = self
            .default_engine_id
            .write()
            .map_err(|_| "Failed to acquire write lock on default_engine_id")?;
        let old_default = default_lock.clone();
        *default_lock = Some(resolved_model_id.clone());

        // Log the change
        info!(
            "Default model changed: {:?} -> {:?}",
            old_default, resolved_model_id
        );

        Ok(())
    }

    /// Dispatch a request to the appropriate engine based on the model_id in the request
    pub fn send_request(&self, mut request: Request) -> Result<(), MistralRsError> {
        let model_id = match &mut request {
            Request::Normal(normal_req) => normal_req.model_id.as_deref(),
            _ => None, // Other request types don't specify model_id
        };

        let sender = self.get_sender(model_id)?;
        sender
            .blocking_send(request)
            .map_err(|_| MistralRsError::SenderPoisoned)
    }

    pub fn maybe_log_request(this: Arc<Self>, repr: String) {
        if let Some(file) = &this.log {
            let mut f = OpenOptions::new()
                .append(true)
                .create(true) // Optionally create the file if it doesn't already exist
                .open(file)
                .expect("Unable to open file");
            let time = chrono::offset::Local::now();
            f.write_all(format!("Request at {time}: {repr}\n\n").as_bytes())
                .expect("Unable to write data");
        }
    }

    pub fn maybe_log_response<T: Serialize>(this: Arc<Self>, resp: &T) {
        if let Some(file) = &this.log {
            let mut f = OpenOptions::new()
                .append(true)
                .create(true) // Optionally create the file if it doesn't already exist
                .open(file)
                .expect("Unable to open file");
            let time = chrono::offset::Local::now();
            let repr = serde_json::to_string(resp).expect("Serialization of response failed.");
            f.write_all(format!("Response at {time}: {repr}\n\n").as_bytes())
                .expect("Unable to write data");
        }
    }

    pub fn maybe_log_error(this: Arc<Self>, err: &dyn Error) {
        if let Some(file) = &this.log {
            let mut f = OpenOptions::new()
                .append(true)
                .create(true) // Optionally create the file if it doesn't already exist
                .open(file)
                .expect("Unable to open file");
            let time = chrono::offset::Local::now();
            f.write_all(format!("Error response at {time}: {err}\n\n").as_bytes())
                .expect("Unable to write data");
        }
    }

    /// Get the number of tools available for a specific model (including MCP tools)
    pub fn get_tools_count(&self, model_id: Option<&str>) -> Result<usize, String> {
        let resolved_model_id = self
            .resolve_alias_or_default(model_id)
            .map_err(|e| e.to_string())?;

        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.reboot_state.tool_callbacks_with_tools.len())
        } else {
            Err(format!("Model {resolved_model_id} not found"))
        }
    }

    /// Check if MCP client is configured for a specific model
    pub fn has_mcp_client(&self, model_id: Option<&str>) -> Result<bool, String> {
        let resolved_model_id = self
            .resolve_alias_or_default(model_id)
            .map_err(|e| e.to_string())?;

        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.reboot_state.mcp_client_config.is_some())
        } else {
            Err(format!("Model {resolved_model_id} not found"))
        }
    }

    /// Get config for a specific model
    pub fn config(&self, model_id: Option<&str>) -> Result<MistralRsConfig, String> {
        let resolved_model_id = self
            .resolve_alias_or_default(model_id)
            .map_err(|e| e.to_string())?;

        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.config.clone())
        } else {
            Err(format!("Model {resolved_model_id} not found"))
        }
    }

    /// Unload a model from memory while preserving its configuration for later reload.
    /// The model can be reloaded automatically when a request is sent to it, or manually
    /// using `reload_model()`.
    ///
    /// Note: The model must have been added with a `ModelLoaderConfig` for auto-reload to work.
    /// Models added via `MistralRsBuilder` without explicit loader config cannot be reloaded.
    pub fn unload_model(&self, model_id: &str) -> Result<(), MistralRsError> {
        let resolved_model_id = self.resolve_alias(model_id)?;
        // Check if already unloaded
        {
            let unloaded = self
                .unloaded_models
                .read()
                .map_err(|_| MistralRsError::EnginePoisoned)?;
            if unloaded.contains_key(&resolved_model_id) {
                return Err(MistralRsError::ModelAlreadyUnloaded(
                    resolved_model_id.clone(),
                ));
            }
        }

        // Get the engine instance and create UnloadedModelState
        let mut engines = self
            .engines
            .write()
            .map_err(|_| MistralRsError::EnginePoisoned)?;

        let engine_instance = engines
            .remove(&resolved_model_id)
            .ok_or_else(|| MistralRsError::ModelNotFound(resolved_model_id.clone()))?;

        // Check if we have loader config for reloading
        let loader_config = engine_instance
            .reboot_state
            .loader_config
            .clone()
            .ok_or_else(|| MistralRsError::NoLoaderConfig(resolved_model_id.clone()))?;

        // Create the unloaded state
        let unloaded_state = UnloadedModelState {
            loader_config,
            scheduler_config: engine_instance.reboot_state.method.clone(),
            engine_config: EngineConfig {
                no_kv_cache: engine_instance.reboot_state.no_kv_cache,
                no_prefix_cache: engine_instance.reboot_state.no_prefix_cache,
                prefix_cache_n: engine_instance.reboot_state.prefix_cache_n,
                disable_eos_stop: engine_instance.reboot_state.disable_eos_stop,
                throughput_logging_enabled: engine_instance.reboot_state.throughput_logging_enabled,
                search_embedding_model: engine_instance.reboot_state.search_embedding_model,
                search_callback: engine_instance.reboot_state.search_callback.clone(),
                tool_callbacks: engine_instance.reboot_state.tool_callbacks.clone(),
                tool_callbacks_with_tools: engine_instance
                    .reboot_state
                    .tool_callbacks_with_tools
                    .clone(),
            },
            mcp_client_config: engine_instance.reboot_state.mcp_client_config.clone(),
            category: engine_instance.category.clone(),
            mistralrs_config: engine_instance.config.clone(),
        };

        // Send terminate signal to the engine
        let _ = engine_instance.sender.try_send(Request::Terminate);

        drop(engines);

        // Store the unloaded state
        let mut unloaded = self
            .unloaded_models
            .write()
            .map_err(|_| MistralRsError::EnginePoisoned)?;
        unloaded.insert(resolved_model_id.to_string(), unloaded_state);

        // Update default if needed
        let mut default_lock = self
            .default_engine_id
            .write()
            .map_err(|_| MistralRsError::EnginePoisoned)?;
        if let Some(ref default_id) = *default_lock {
            if default_id == &resolved_model_id {
                // Set the first available engine as the new default
                let engines = self
                    .engines
                    .read()
                    .map_err(|_| MistralRsError::EnginePoisoned)?;
                *default_lock = engines.keys().next().cloned();
            }
        }

        info!("Model {} unloaded successfully", resolved_model_id);
        Ok(())
    }

    /// Manually reload a previously unloaded model.
    /// This is also called automatically by `get_sender()` when a request targets an unloaded model.
    pub async fn reload_model(&self, model_id: &str) -> Result<(), MistralRsError> {
        let resolved_model_id = self.resolve_alias(model_id)?;
        // Check if already reloading
        {
            let reloading = self
                .reloading_models
                .read()
                .map_err(|_| MistralRsError::EnginePoisoned)?;
            if reloading.contains(&resolved_model_id) {
                return Err(MistralRsError::ModelReloading(resolved_model_id.clone()));
            }
        }

        // Mark as reloading
        {
            let mut reloading = self
                .reloading_models
                .write()
                .map_err(|_| MistralRsError::EnginePoisoned)?;
            reloading.insert(resolved_model_id.clone());
        }

        // Get the unloaded state
        let unloaded_state = {
            let unloaded = self
                .unloaded_models
                .read()
                .map_err(|_| MistralRsError::EnginePoisoned)?;
            unloaded
                .get(&resolved_model_id)
                .cloned()
                .ok_or_else(|| MistralRsError::ModelNotFound(resolved_model_id.clone()))?
        };

        // Attempt to reload
        let result = self
            .do_reload_model(&resolved_model_id, unloaded_state)
            .await;

        // Remove from reloading set
        {
            let mut reloading = self
                .reloading_models
                .write()
                .map_err(|_| MistralRsError::EnginePoisoned)?;
            reloading.remove(&resolved_model_id);
        }

        result
    }

    /// Internal method to perform the actual model reload
    async fn do_reload_model(
        &self,
        model_id: &str,
        unloaded_state: UnloadedModelState,
    ) -> Result<(), MistralRsError> {
        use crate::model_loader::LoaderBuilder;

        info!("Reloading model: {}", model_id);

        let loader_config = &unloaded_state.loader_config;

        // Build the loader from the stored config
        let loader = LoaderBuilder::new(loader_config.model_selected.clone())
            .with_chat_template(loader_config.chat_template.clone())
            .with_jinja_explicit(loader_config.jinja_explicit.clone())
            .build()
            .map_err(|e| MistralRsError::ReloadFailed(format!("Failed to build loader: {e}")))?;

        // Load the model
        let pipeline = loader
            .load_model_from_hf(
                None,
                loader_config.token_source.clone(),
                &loader_config.dtype,
                &loader_config.device,
                loader_config.silent,
                loader_config.device_map_setting.clone(),
                loader_config.isq,
                loader_config.paged_attn_config,
            )
            .map_err(|e| MistralRsError::ReloadFailed(format!("Failed to load model: {e}")))?;

        // Create the reboot state
        let reboot_state = RebootState {
            pipeline: pipeline.clone(),
            method: unloaded_state.scheduler_config.clone(),
            no_kv_cache: unloaded_state.engine_config.no_kv_cache,
            no_prefix_cache: unloaded_state.engine_config.no_prefix_cache,
            prefix_cache_n: unloaded_state.engine_config.prefix_cache_n,
            disable_eos_stop: unloaded_state.engine_config.disable_eos_stop,
            throughput_logging_enabled: unloaded_state.engine_config.throughput_logging_enabled,
            search_embedding_model: unloaded_state.engine_config.search_embedding_model,
            search_callback: unloaded_state.engine_config.search_callback.clone(),
            tool_callbacks: unloaded_state.engine_config.tool_callbacks.clone(),
            tool_callbacks_with_tools: unloaded_state
                .engine_config
                .tool_callbacks_with_tools
                .clone(),
            mcp_client_config: unloaded_state.mcp_client_config.clone(),
            loader_config: Some(unloaded_state.loader_config.clone()),
        };

        // Create the engine instance
        let engine_instance = Self::create_engine_instance(
            pipeline,
            unloaded_state.scheduler_config,
            unloaded_state.engine_config,
            reboot_state,
        )
        .map_err(|e| MistralRsError::ReloadFailed(format!("Failed to create engine: {e}")))?;

        // Add to engines map
        {
            let mut engines = self
                .engines
                .write()
                .map_err(|_| MistralRsError::EnginePoisoned)?;
            engines.insert(model_id.to_string(), engine_instance);
        }

        // Remove from unloaded map
        {
            let mut unloaded = self
                .unloaded_models
                .write()
                .map_err(|_| MistralRsError::EnginePoisoned)?;
            unloaded.remove(model_id);
        }

        info!("Model {} reloaded successfully", model_id);
        Ok(())
    }

    /// Synchronous version of reload_model for use in non-async contexts.
    ///
    /// This method handles different runtime contexts appropriately:
    /// - If called from a multi-threaded tokio runtime, uses `block_in_place`
    /// - If called from a single-threaded runtime, returns an error (use `reload_model()` instead)
    /// - If called outside any runtime, creates a temporary runtime
    pub fn reload_model_blocking(&self, model_id: &str) -> Result<(), MistralRsError> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::CurrentThread {
                    Err(MistralRsError::ReloadFailed(
                        "Cannot reload model blocking from single-threaded runtime. Use reload_model() instead.".to_string()
                    ))
                } else {
                    tokio::task::block_in_place(|| handle.block_on(self.reload_model(model_id)))
                }
            }
            Err(_) => {
                let rt = tokio::runtime::Runtime::new().map_err(|e| {
                    MistralRsError::ReloadFailed(format!("Failed to create runtime: {e}"))
                })?;
                rt.block_on(self.reload_model(model_id))
            }
        }
    }

    /// List all unloaded model IDs
    pub fn list_unloaded_models(&self) -> Result<Vec<String>, MistralRsError> {
        let unloaded = self
            .unloaded_models
            .read()
            .map_err(|_| MistralRsError::EnginePoisoned)?;
        Ok(unloaded.keys().cloned().collect())
    }

    /// Check if a model is currently loaded (as opposed to unloaded)
    pub fn is_model_loaded(&self, model_id: &str) -> Result<bool, MistralRsError> {
        let resolved_model_id = self.resolve_alias(model_id)?;
        let engines = self
            .engines
            .read()
            .map_err(|_| MistralRsError::EnginePoisoned)?;
        Ok(engines.contains_key(&resolved_model_id))
    }

    /// Get the status of a model, or None if not found
    pub fn get_model_status(&self, model_id: &str) -> Result<Option<ModelStatus>, MistralRsError> {
        let resolved_model_id = self.resolve_alias(model_id)?;
        // Check if reloading
        {
            let reloading = self
                .reloading_models
                .read()
                .map_err(|_| MistralRsError::EnginePoisoned)?;
            if reloading.contains(&resolved_model_id) {
                return Ok(Some(ModelStatus::Reloading));
            }
        }

        // Check if loaded
        {
            let engines = self
                .engines
                .read()
                .map_err(|_| MistralRsError::EnginePoisoned)?;
            if engines.contains_key(&resolved_model_id) {
                return Ok(Some(ModelStatus::Loaded));
            }
        }

        // Check if unloaded
        {
            let unloaded = self
                .unloaded_models
                .read()
                .map_err(|_| MistralRsError::EnginePoisoned)?;
            if unloaded.contains_key(&resolved_model_id) {
                return Ok(Some(ModelStatus::Unloaded));
            }
        }

        Ok(None)
    }

    /// List all models with their status
    pub fn list_models_with_status(&self) -> Result<Vec<(String, ModelStatus)>, MistralRsError> {
        let mut result = Vec::new();

        // Get reloading models
        let reloading = self
            .reloading_models
            .read()
            .map_err(|_| MistralRsError::EnginePoisoned)?;
        for model_id in reloading.iter() {
            result.push((model_id.clone(), ModelStatus::Reloading));
        }
        drop(reloading);

        // Get loaded models
        let engines = self
            .engines
            .read()
            .map_err(|_| MistralRsError::EnginePoisoned)?;
        for model_id in engines.keys() {
            result.push((model_id.clone(), ModelStatus::Loaded));
        }
        drop(engines);

        // Get unloaded models
        let unloaded = self
            .unloaded_models
            .read()
            .map_err(|_| MistralRsError::EnginePoisoned)?;
        for model_id in unloaded.keys() {
            // Skip if already in reloading
            if !result.iter().any(|(id, _)| id == model_id) {
                result.push((model_id.clone(), ModelStatus::Unloaded));
            }
        }

        Ok(result)
    }
}
