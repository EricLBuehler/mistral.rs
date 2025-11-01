#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use candle_core::Device;
use engine::Engine;
pub use engine::{
    get_engine_terminate_flag, reset_engine_terminate_flag, should_terminate_engine_sequences,
    BertEmbeddingModel, EngineInstruction, ENGINE_INSTRUCTIONS, TERMINATE_ALL_NEXT_STEP,
};
use hf_hub::Cache;
pub use lora::Ordering;
pub use pipeline::ModelCategory;
pub use pipeline::Pipeline;
#[cfg(feature = "pyo3_macros")]
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;
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

mod cuda;
mod device_map;
mod engine;
mod lora;
mod model_loader;
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
mod diffusion_models;
pub mod distributed;
mod embedding;
mod gguf;
pub mod layers;
mod layers_masker;
mod layers_utils;
pub mod matformer;
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
mod toml_selector;
mod tools;
mod topology;
mod utils;
mod vision_models;
mod xlora_models;

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
    LlguidanceGrammar, MessageContent, NormalRequest, Request, RequestMessage, SearchContextSize,
    TokenizationRequest, WebSearchOptions, WebSearchUserLocation,
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
    pub truncate_sequence: bool,
    pub no_kv_cache: bool,
    pub no_prefix_cache: bool,
    pub prefix_cache_n: usize,
    pub disable_eos_stop: bool,
    pub throughput_logging_enabled: bool,
    pub search_embedding_model: Option<BertEmbeddingModel>,
    pub search_callback: Option<Arc<SearchCallback>>,
    pub tool_callbacks: tools::ToolCallbacks,
    pub tool_callbacks_with_tools: tools::ToolCallbacksWithTools,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            truncate_sequence: false,
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
}

impl AddModelConfig {
    pub fn new(engine_config: EngineConfig) -> Self {
        Self {
            engine_config,
            mcp_client_config: None,
        }
    }

    pub fn with_mcp_config(mut self, mcp_config: McpClientConfig) -> Self {
        self.mcp_client_config = Some(mcp_config);
        self
    }
}

#[derive(Clone)]
pub struct MistralRsConfig {
    pub kind: ModelKind,
    pub device: Device,
    pub category: ModelCategory,
    pub modalities: Modalities,
}

/// Internal structure to hold per-engine state
struct EngineInstance {
    sender: Sender<Request>,
    engine_handler: JoinHandle<()>,
    reboot_state: RebootState,
    config: MistralRsConfig,
    category: ModelCategory,
}

/// The MistralRs struct handles sending requests to multiple engines.
/// It is the core multi-threaded component of mistral.rs, and uses `mpsc`
/// `Sender` and `Receiver` primitives to send and receive requests to the
/// appropriate engine based on model ID.
pub struct MistralRs {
    engines: RwLock<HashMap<String, EngineInstance>>,
    default_engine_id: RwLock<Option<String>>,
    log: Option<String>,
    id: String,
    creation_time: u64,
    next_request_id: Mutex<RefCell<usize>>,
}

#[derive(Clone)]
struct RebootState {
    pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
    method: SchedulerConfig,
    truncate_sequence: bool,
    no_kv_cache: bool,
    no_prefix_cache: bool,
    prefix_cache_n: usize,
    disable_eos_stop: bool,
    throughput_logging_enabled: bool,
    search_embedding_model: Option<BertEmbeddingModel>,
    search_callback: Option<Arc<search::SearchCallback>>,
    tool_callbacks: tools::ToolCallbacks,
    tool_callbacks_with_tools: tools::ToolCallbacksWithTools,
    mcp_client_config: Option<McpClientConfig>,
}

#[derive(Debug)]
pub enum MistralRsError {
    EnginePoisoned,
    SenderPoisoned,
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
    log: Option<String>,
    truncate_sequence: Option<bool>,
    no_kv_cache: Option<bool>,
    no_prefix_cache: Option<bool>,
    prefix_cache_n: Option<usize>,
    disable_eos_stop: Option<bool>,
    throughput_logging_enabled: bool,
    search_embedding_model: Option<BertEmbeddingModel>,
    search_callback: Option<Arc<SearchCallback>>,
    tool_callbacks: tools::ToolCallbacks,
    tool_callbacks_with_tools: tools::ToolCallbacksWithTools,
    mcp_client_config: Option<McpClientConfig>,
}

impl MistralRsBuilder {
    /// Creates a new builder with the given pipeline, scheduler method, logging flag,
    /// and optional embedding model for web search. To override the search callback,
    /// use `.with_search_callback(...)` on the builder.
    pub fn new(
        pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        method: SchedulerConfig,
        throughput_logging: bool,
        search_embedding_model: Option<BertEmbeddingModel>,
    ) -> Self {
        Self {
            pipeline,
            method,
            log: None,
            truncate_sequence: None,
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
        }
    }
    pub fn with_log(mut self, log: String) -> Self {
        self.log = Some(log);
        self
    }
    pub fn with_opt_log(mut self, log: Option<String>) -> Self {
        self.log = log;
        self
    }
    pub fn with_truncate_sequence(mut self, truncate_sequence: bool) -> Self {
        self.truncate_sequence = Some(truncate_sequence);
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

        let category = pipeline.try_lock().unwrap().category();
        let kind = pipeline.try_lock().unwrap().get_metadata().kind.clone();
        let device = pipeline.try_lock().unwrap().device();
        let modalities = pipeline
            .try_lock()
            .unwrap()
            .get_metadata()
            .modalities
            .clone();

        info!("Pipeline input modalities are {:?}", &modalities.input);
        info!("Pipeline output modalities are {:?}", &modalities.output);

        let mistralrs_config = MistralRsConfig {
            kind,
            device,
            category: category.clone(),
            modalities,
        };

        let engine_handler = thread::spawn(move || {
            #[cfg(feature = "metal")]
            objc::rc::autoreleasepool(move || {
                let rt = Runtime::new().unwrap();
                rt.block_on(async move {
                    let engine = Engine::new(
                        rx,
                        pipeline,
                        method,
                        config.truncate_sequence,
                        config.no_kv_cache,
                        config.no_prefix_cache,
                        config.prefix_cache_n,
                        config.disable_eos_stop,
                        config.throughput_logging_enabled,
                        config.search_embedding_model,
                        config.search_callback.clone(),
                        config.tool_callbacks.clone(),
                        config.tool_callbacks_with_tools.clone(),
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
                        rx,
                        pipeline,
                        method,
                        config.truncate_sequence,
                        config.no_kv_cache,
                        config.no_prefix_cache,
                        config.prefix_cache_n,
                        config.disable_eos_stop,
                        config.throughput_logging_enabled,
                        config.search_embedding_model,
                        config.search_callback.clone(),
                        config.tool_callbacks.clone(),
                        config.tool_callbacks_with_tools.clone(),
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
        })
    }

    async fn new(config: MistralRsBuilder) -> Arc<Self> {
        let MistralRsBuilder {
            pipeline,
            method,
            log,
            truncate_sequence,
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
        } = config;

        mistralrs_quant::cublaslt::maybe_init_cublas_lt_wrapper(
            get_mut_arcmutex!(pipeline).device(),
        );

        let truncate_sequence = truncate_sequence.unwrap_or(false);
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
            truncate_sequence,
            no_kv_cache,
            no_prefix_cache,
            prefix_cache_n,
            disable_eos_stop,
            throughput_logging_enabled,
            search_embedding_model: search_embedding_model.clone(),
            search_callback: search_callback.clone(),
            tool_callbacks: tool_callbacks.clone(),
            tool_callbacks_with_tools: tool_callbacks_with_tools.clone(),
            mcp_client_config: mcp_client_config.clone(),
        };

        // Create the engine configuration
        let engine_config = EngineConfig {
            truncate_sequence,
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

        let id = pipeline.try_lock().unwrap().name();

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
        }

        // Create engines map with the first engine
        let mut engines = HashMap::new();
        engines.insert(id.clone(), engine_instance);

        Arc::new(Self {
            engines: RwLock::new(engines),
            default_engine_id: RwLock::new(Some(id.clone())),
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
                truncate_sequence: reboot_state.truncate_sequence,
                no_kv_cache: reboot_state.no_kv_cache,
                no_prefix_cache: reboot_state.no_prefix_cache,
                prefix_cache_n: reboot_state.prefix_cache_n,
                disable_eos_stop: reboot_state.disable_eos_stop,
                throughput_logging_enabled: reboot_state.throughput_logging_enabled,
                search_embedding_model: reboot_state.search_embedding_model.clone(),
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
    pub fn get_sender(&self, model_id: Option<&str>) -> Result<Sender<Request>, MistralRsError> {
        let resolved_model_id = match model_id {
            Some(id) => id.to_string(),
            None => {
                let default_lock = self
                    .default_engine_id
                    .read()
                    .map_err(|_| MistralRsError::SenderPoisoned)?;
                default_lock
                    .as_ref()
                    .ok_or(MistralRsError::EnginePoisoned)?
                    .clone()
            }
        };

        if self.engine_dead(&resolved_model_id)? {
            tracing::warn!("Engine {} is dead, rebooting", resolved_model_id);
            self.reboot_engine(&resolved_model_id)?
        }

        let engines = self
            .engines
            .read()
            .map_err(|_| MistralRsError::SenderPoisoned)?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.sender.clone())
        } else {
            Err(MistralRsError::EnginePoisoned)
        }
    }

    pub fn get_id(&self) -> String {
        self.id.clone()
    }

    pub fn get_creation_time(&self) -> u64 {
        self.creation_time
    }

    /// Get model category for a specific model. If model_id is None, uses default engine.
    pub fn get_model_category(
        &self,
        model_id: Option<&str>,
    ) -> Result<ModelCategory, MistralRsError> {
        let resolved_model_id = match model_id {
            Some(id) => id.to_string(),
            None => {
                let default_lock = self
                    .default_engine_id
                    .read()
                    .map_err(|_| MistralRsError::SenderPoisoned)?;
                default_lock
                    .as_ref()
                    .ok_or(MistralRsError::EnginePoisoned)?
                    .clone()
            }
        };

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
        let reboot_state = RebootState {
            pipeline: pipeline.clone(),
            method: method.clone(),
            truncate_sequence: config.engine_config.truncate_sequence,
            no_kv_cache: config.engine_config.no_kv_cache,
            no_prefix_cache: config.engine_config.no_prefix_cache,
            prefix_cache_n: config.engine_config.prefix_cache_n,
            disable_eos_stop: config.engine_config.disable_eos_stop,
            throughput_logging_enabled: config.engine_config.throughput_logging_enabled,
            search_embedding_model: config.engine_config.search_embedding_model.clone(),
            search_callback: config.engine_config.search_callback.clone(),
            tool_callbacks: config.engine_config.tool_callbacks.clone(),
            tool_callbacks_with_tools: config.engine_config.tool_callbacks_with_tools.clone(),
            mcp_client_config: config.mcp_client_config.clone(),
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
        }

        Ok(())
    }

    /// Remove a model engine from the MistralRs instance
    pub fn remove_model(&self, model_id: &str) -> Result<(), String> {
        let mut engines = self
            .engines
            .write()
            .map_err(|_| "Failed to acquire write lock on engines")?;

        if engines.len() <= 1 {
            return Err("Cannot remove the last model from MistralRs".to_string());
        }

        if let Some(engine_instance) = engines.remove(model_id) {
            // Send terminate signal to the engine
            let _ = engine_instance.sender.blocking_send(Request::Terminate);

            // If this was the default engine, set a new default
            let mut default_lock = self
                .default_engine_id
                .write()
                .map_err(|_| "Failed to acquire write lock on default_engine_id")?;
            if let Some(ref default_id) = *default_lock {
                if default_id == model_id {
                    // Set the first available engine as the new default
                    *default_lock = engines.keys().next().cloned();
                }
            }

            Ok(())
        } else {
            Err(format!("Model {model_id} not found"))
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
        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        if !engines.contains_key(model_id) {
            return Err(format!("Model {model_id} not found"));
        }
        drop(engines);

        let mut default_lock = self
            .default_engine_id
            .write()
            .map_err(|_| "Failed to acquire write lock on default_engine_id")?;
        *default_lock = Some(model_id.to_string());

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
        let resolved_model_id = match model_id {
            Some(id) => id.to_string(),
            None => {
                let default_lock = self
                    .default_engine_id
                    .read()
                    .map_err(|_| "Failed to acquire read lock")?;
                default_lock
                    .as_ref()
                    .ok_or("No default engine set")?
                    .clone()
            }
        };

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
        let resolved_model_id = match model_id {
            Some(id) => id.to_string(),
            None => {
                let default_lock = self
                    .default_engine_id
                    .read()
                    .map_err(|_| "Failed to acquire read lock")?;
                default_lock
                    .as_ref()
                    .ok_or("No default engine set")?
                    .clone()
            }
        };

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
        let resolved_model_id = match model_id {
            Some(id) => id.to_string(),
            None => {
                let default_lock = self
                    .default_engine_id
                    .read()
                    .map_err(|_| "Failed to acquire read lock")?;
                default_lock
                    .as_ref()
                    .ok_or("No default engine set")?
                    .clone()
            }
        };

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
}
