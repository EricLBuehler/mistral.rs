#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use candle_core::Device;
use engine::Engine;
pub use engine::{
    BertEmbeddingModel, EngineInstruction, ENGINE_INSTRUCTIONS, TERMINATE_ALL_NEXT_STEP,
};
use hf_hub::Cache;
pub use lora::Ordering;
pub use pipeline::ModelCategory;
pub use pipeline::Pipeline;
#[cfg(feature = "pyo3_macros")]
use pyo3::exceptions::PyValueError;
use std::io::BufRead;
use std::io::BufReader;
use std::sync::OnceLock;
use std::time::Instant;
use std::{
    cell::RefCell,
    error::Error,
    fs::OpenOptions,
    io::Write,
    sync::{
        atomic::{self, AtomicBool, AtomicUsize},
        Arc, Mutex, RwLock,
    },
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
mod search;

mod model_selected;
pub use model_selected::ModelSelected;
pub use toml_selector::{get_toml_selected_model_device_map_params, get_toml_selected_model_dtype};

mod amoe;
#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
mod dummy_paged_attention;
mod embedding;
mod gguf;
pub mod layers;
mod layers_masker;
mod layers_utils;
mod models;
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
mod paged_attention;
#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
use dummy_paged_attention as paged_attention;
mod attention;
mod diffusion_models;
pub mod distributed;
mod pipeline;
mod prefix_cacher;
mod request;
mod response;
mod sampler;
mod scheduler;
mod sequence;
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
pub use mistralrs_quant::IsqType;
pub use paged_attention::{MemoryGpuConfig, PagedAttentionConfig};
pub use pipeline::{
    chat_template::ChatTemplate, parse_isq_value, AnyMoeLoader, AnyMoePipeline,
    AutoDeviceMapParams, DiffusionGenerationParams, DiffusionLoader, DiffusionLoaderBuilder,
    DiffusionLoaderType, DiffusionSpecificConfig, GGMLLoader, GGMLLoaderBuilder,
    GGMLSpecificConfig, GGUFLoader, GGUFLoaderBuilder, GGUFSpecificConfig, GemmaLoader,
    Idefics2Loader, IsqOrganization, LLaVALoader, LLaVANextLoader, LlamaLoader, Loader,
    LocalModelPaths, MistralLoader, MixtralLoader, ModelKind, ModelPaths, NormalLoader,
    NormalLoaderBuilder, NormalLoaderType, NormalSpecificConfig, Phi2Loader, Phi3Loader,
    Phi3VLoader, Qwen2Loader, SpeculativeConfig, SpeculativeLoader, SpeculativePipeline,
    Starcoder2Loader, TokenSource, VisionLoader, VisionLoaderBuilder, VisionLoaderType,
    VisionPromptPrefixer, VisionSpecificConfig,
};
pub use request::{
    ApproximateUserLocation, Constraint, DetokenizationRequest, ImageGenerationResponseFormat,
    LlguidanceGrammar, MessageContent, NormalRequest, Request, RequestMessage, TokenizationRequest,
    WebSearchOptions, WebSearchUserLocation,
};
pub use response::*;
pub use sampler::{
    CustomLogitsProcessor, DrySamplingParams, SamplingParams, StopTokens, TopLogprob,
};
pub use scheduler::{DefaultSchedulerMethod, SchedulerConfig};
use serde::Serialize;
use tokio::runtime::Runtime;
use toml_selector::{TomlLoaderArgs, TomlSelector};
pub use tools::{
    CalledFunction, Function, Tool, ToolCallResponse, ToolCallType, ToolChoice, ToolType,
};
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
static ENGINE_ID: AtomicUsize = AtomicUsize::new(0);

pub struct MistralRsConfig {
    pub kind: ModelKind,
    pub device: Device,
    pub category: ModelCategory,
}

/// The MistralRs struct handles sending requests to the engine.
/// It is the core multi-threaded component of mistral.rs, and uses `mpsc`
/// `Sender` and `Receiver` primitives to send and receive requests to the
/// engine.
pub struct MistralRs {
    sender: RwLock<Sender<Request>>,
    log: Option<String>,
    id: String,
    creation_time: u64,
    next_request_id: Mutex<RefCell<usize>>,
    reboot_state: RebootState,
    engine_handler: RwLock<JoinHandle<()>>,
    engine_id: usize,
    category: ModelCategory,
    config: MistralRsConfig,
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
        PyValueError::new_err(format!("{:?}", value))
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
}

impl MistralRsBuilder {
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

    pub fn build(self) -> Arc<MistralRs> {
        MistralRs::new(self)
    }
}

impl Drop for MistralRs {
    fn drop(&mut self) {
        ENGINE_INSTRUCTIONS
            .lock()
            .expect("`ENGINE_INSTRUCTIONS` was poisioned")
            .insert(self.engine_id, Some(EngineInstruction::Terminate));
    }
}

impl MistralRs {
    fn new(config: MistralRsBuilder) -> Arc<Self> {
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
        } = config;

        let category = pipeline.try_lock().unwrap().category();
        mistralrs_quant::cublaslt::maybe_init_cublas_lt_wrapper(
            get_mut_arcmutex!(pipeline).device(),
        );

        let truncate_sequence = truncate_sequence.unwrap_or(false);
        let no_kv_cache = no_kv_cache.unwrap_or(false);
        let no_prefix_cache = no_prefix_cache.unwrap_or(false);
        let prefix_cache_n = prefix_cache_n.unwrap_or(16);
        let disable_eos_stop = disable_eos_stop.unwrap_or(false);

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
        };

        let (tx, rx) = channel(10_000);

        let sender = RwLock::new(tx);
        let id = pipeline.try_lock().unwrap().name();

        let kind = pipeline.try_lock().unwrap().get_metadata().kind.clone();
        let device = pipeline.try_lock().unwrap().device();
        let config = MistralRsConfig {
            kind,
            device,
            category: category.clone(),
        };

        let engine_handler = thread::spawn(move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async move {
                let engine = Engine::new(
                    rx,
                    pipeline,
                    method,
                    truncate_sequence,
                    no_kv_cache,
                    no_prefix_cache,
                    prefix_cache_n,
                    disable_eos_stop,
                    throughput_logging_enabled,
                    search_embedding_model,
                )
                .expect("Engine creation failed.");
                Arc::new(engine).run().await;
            });
        });

        let engine_id = ENGINE_ID.fetch_add(1, atomic::Ordering::SeqCst);

        if distributed::is_daemon() {
            let request_sender = sender.write().unwrap().clone();
            thread::spawn(move || {
                let rt = Runtime::new().unwrap();
                rt.block_on(async move {
                    use interprocess::local_socket::traits::Stream;
                    use interprocess::local_socket::Stream as LocalStream;

                    loop {
                        let name = distributed::ipc_name().unwrap();
                        if let Ok(stream) = LocalStream::connect(name) {
                            let mut reader = BufReader::new(stream);
                            let mut buf = String::new();
                            reader.read_line(&mut buf).unwrap();
                            let mut req: Request = serde_json::from_str(&buf).unwrap();

                            req = match req {
                                Request::ActivateAdapters(x) => Request::ActivateAdapters(x),
                                Request::ReIsq(x) => Request::ReIsq(x),
                                Request::Terminate => Request::Terminate,
                                Request::Detokenize(mut x) => {
                                    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                                    x.response = sender;
                                    let req = Request::Detokenize(x);

                                    request_sender.send(req).await.unwrap();
                                    let resp = receiver.recv().await.unwrap();
                                    resp.unwrap();
                                    continue;
                                }
                                Request::Tokenize(mut x) => {
                                    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                                    x.response = sender;
                                    let req = Request::Tokenize(x);

                                    request_sender.send(req).await.unwrap();
                                    let resp = receiver.recv().await.unwrap();
                                    resp.unwrap();
                                    continue;
                                }
                                Request::Normal(mut x) => {
                                    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                                    x.is_streaming = false;
                                    x.response = sender;
                                    let req = Request::Normal(x);

                                    request_sender.send(req).await.unwrap();
                                    let resp = receiver.recv().await.unwrap();
                                    resp.as_result().unwrap();
                                    continue;
                                }
                                Request::TerminateAllSeqsNextStep => {
                                    Request::TerminateAllSeqsNextStep
                                }
                            };

                            request_sender.send(req).await.unwrap();
                        }
                    }
                });
            });

            #[allow(clippy::empty_loop)]
            loop {}
        }

        // Determine if the current runtime is multi-threaded, as blocking operations are not allowed in single-threaded mode
        let is_multi_threaded = tokio::runtime::Handle::try_current()
            .is_ok_and(|h| h.runtime_flavor() != tokio::runtime::RuntimeFlavor::CurrentThread);

        // Do a dummy run
        if !distributed::is_daemon()
            && is_multi_threaded
            && matches!(category, ModelCategory::Text | ModelCategory::Vision { .. })
        {
            let clone_sender = sender.read().unwrap().clone();
            tokio::task::block_in_place(|| {
                let (tx, mut rx) = channel(1);
                let req = Request::Normal(NormalRequest {
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
                    adapters: None,
                    tool_choice: None,
                    tools: None,
                    logits_processors: None,
                    return_raw_logits: false,
                    web_search_options: None,
                });
                info!("Beginning dummy run.");
                let start = Instant::now();
                clone_sender.blocking_send(req).unwrap();

                if let Some(_resp) = rx.blocking_recv() {
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

        Arc::new(Self {
            engine_id,
            sender,
            log,
            id,
            creation_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time travel has occurred!")
                .as_secs(),
            next_request_id: Mutex::new(RefCell::new(1)),
            reboot_state,
            engine_handler: RwLock::new(engine_handler),
            category,
            config,
        })
    }

    /// attempts to reboot the engine, if the sender (only way to communicate with
    /// the engine) is closed
    fn reboot_engine(&self) -> Result<(), MistralRsError> {
        let (new_sender, rx) = channel(10_000);
        let reboot_state = self.reboot_state.clone();
        let mut sender_lock = self.sender.write().map_err(|_| {
            tracing::warn!("Couldn't get write lock on the sender during reboot attempt");
            MistralRsError::SenderPoisoned
        })?;
        let mut engine_lock = self.engine_handler.write().map_err(|_| {
            tracing::warn!("Couldn't get write lock on the engine during reboot attempt");
            MistralRsError::EnginePoisoned
        })?;

        if !engine_lock.is_finished() {
            tracing::info!("Engine already running, returning ok");
            Ok(())
        } else {
            // critical section. A panic here could lead to poisoned locks
            let new_engine_handler = thread::spawn(move || {
                let rt = Runtime::new().unwrap();
                rt.block_on(async move {
                    let engine = Engine::new(
                        rx,
                        reboot_state.pipeline.clone(),
                        reboot_state.method,
                        reboot_state.truncate_sequence,
                        reboot_state.no_kv_cache,
                        reboot_state.no_prefix_cache,
                        reboot_state.prefix_cache_n,
                        reboot_state.disable_eos_stop,
                        reboot_state.throughput_logging_enabled,
                        reboot_state.search_embedding_model,
                    )
                    .expect("Engine creation failed");
                    Arc::new(engine).run().await;
                });
            });
            *sender_lock = new_sender;
            *engine_lock = new_engine_handler;
            tracing::info!("Successfully rebooted engine and updated sender + engine handler");
            Ok(())
        }
    }

    fn engine_dead(&self) -> Result<bool, MistralRsError> {
        match self.engine_handler.read() {
            Ok(handler) => Ok(handler.is_finished()),
            Err(_) => {
                tracing::warn!("Couldn't get read lock on engine!");
                Err(MistralRsError::EnginePoisoned)
            }
        }
    }

    pub fn get_sender(&self) -> Result<Sender<Request>, MistralRsError> {
        if self.engine_dead()? {
            tracing::warn!("Engine is dead, rebooting");
            self.reboot_engine()?
        }
        match self.sender.read() {
            Ok(sender) => Ok(sender.clone()),
            Err(_) => Err(MistralRsError::SenderPoisoned),
        }
    }

    pub fn get_id(&self) -> String {
        self.id.clone()
    }

    pub fn get_creation_time(&self) -> u64 {
        self.creation_time
    }

    pub fn get_model_category(&self) -> ModelCategory {
        self.category.clone()
    }

    pub fn next_request_id(&self) -> usize {
        let l = self.next_request_id.lock().unwrap();
        let last = &mut *l.borrow_mut();
        let last_v = *last;
        *last += 1;
        last_v
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

    pub fn config(&self) -> &MistralRsConfig {
        &self.config
    }
}
