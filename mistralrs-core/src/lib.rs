#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use cublaslt::setup_cublas_lt_wrapper;
use engine::Engine;
pub use engine::TERMINATE_ALL_NEXT_STEP;
pub use lora::Ordering;
use pipeline::ModelCategory;
pub use pipeline::Pipeline;
#[cfg(feature = "pyo3_macros")]
use pyo3::exceptions::PyValueError;
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

mod aici;
mod cuda;
mod device_map;
mod engine;
mod lora;
mod model_loader;
mod ops;
pub use model_loader::{get_model_dtype, get_tgt_non_granular_index, LoaderBuilder};

mod model_selected;
pub use model_selected::ModelSelected;
pub use toml_selector::get_toml_selected_model_dtype;

mod amoe;
mod cublaslt;
#[cfg(not(all(feature = "cuda", target_family = "unix")))]
mod dummy_paged_attention;
mod gguf;
pub mod layers;
mod layers_masker;
mod layers_utils;
mod models;
#[cfg(all(feature = "cuda", target_family = "unix"))]
mod paged_attention;
#[cfg(not(all(feature = "cuda", target_family = "unix")))]
use dummy_paged_attention as paged_attention;
mod pipeline;
mod prefix_cacher;
mod request;
mod response;
mod sampler;
mod scheduler;
mod sequence;
mod toml_selector;
mod utils;
mod vision_models;
mod xlora_models;

pub use amoe::{AnyMoeConfig, AnyMoeExpertType};
pub use device_map::{DeviceLayerMapMetadata, DeviceMapMetadata, LayerDeviceMapper};
pub use paged_attention::PagedAttentionConfig;
pub use pipeline::{
    chat_template::ChatTemplate, AnyMoeLoader, AnyMoePipeline, GGMLLoader, GGMLLoaderBuilder,
    GGMLSpecificConfig, GGUFArchitecture, GGUFLoader, GGUFLoaderBuilder, GGUFSpecificConfig,
    GemmaLoader, Idefics2Loader, LLaVALoader, LLaVANextLoader, LlamaLoader, Loader,
    LocalModelPaths, MistralLoader, MixtralLoader, ModelKind, ModelPaths, NormalLoader,
    NormalLoaderBuilder, NormalLoaderType, NormalSpecificConfig, Phi2Loader, Phi3Loader,
    Phi3VLoader, Qwen2Loader, SpeculativeConfig, SpeculativeLoader, SpeculativePipeline,
    Starcoder2Loader, TokenSource, VisionLoader, VisionLoaderBuilder, VisionLoaderType,
    VisionModelLoader, VisionSpecificConfig,
};
pub use request::{Constraint, MessageContent, NormalRequest, Request, RequestMessage};
pub use response::Response;
pub use response::*;
pub use sampler::{SamplingParams, StopTokens, TopLogprob};
pub use scheduler::{DefaultSchedulerMethod, SchedulerConfig};
use serde::Serialize;
use tokio::runtime::Runtime;
use toml_selector::{TomlLoaderArgs, TomlSelector};
pub use utils::debug::initialize_logging;
pub use utils::memory_usage::MemoryUsage;
pub use utils::normal::{ModelDType, TryIntoDType};
pub use utils::paged_attn_supported;

/// `true` if `MISTRALRS_DEBUG=1`
pub(crate) static DEBUG: AtomicBool = AtomicBool::new(false);

/// The MistralRs struct handles sending requests to the engine.
/// It is the core multi-threaded component of mistral.rs, and uses `mspc`
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
    gemm_full_precision_f16: Option<bool>,
}

impl MistralRsBuilder {
    pub fn new(pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>, method: SchedulerConfig) -> Self {
        Self {
            pipeline,
            method,
            log: None,
            truncate_sequence: None,
            no_kv_cache: None,
            no_prefix_cache: None,
            prefix_cache_n: None,
            disable_eos_stop: None,
            gemm_full_precision_f16: None,
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
    pub fn with_gemm_full_precision_f16(mut self, gemm_full_precision: bool) -> Self {
        self.gemm_full_precision_f16 = Some(gemm_full_precision);
        self
    }

    pub fn build(self) -> Arc<MistralRs> {
        MistralRs::new(self)
    }
}

pub(crate) static INHIBIT_GEMM_F16: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "cuda")]
fn set_gemm_reduced_precision_f16() {
    use candle_core::{DType, Device, Tensor};

    // NOTE(EricLBuehler): When we support multi-GPU inference, we should check for each gpu here
    let a = Tensor::zeros((2, 2), DType::BF16, &Device::new_cuda(0).unwrap()).unwrap();
    candle_core::cuda::set_gemm_reduced_precision_bf16(true);
    match a.matmul(&a) {
        Ok(_) => tracing::info!("Enabling GEMM reduced precision in BF16."),
        Err(e) => {
            if format!("{e:?}").contains("CUBLAS_STATUS_NOT_SUPPORTED") {
                tracing::info!("GEMM reduced precision in BF16 not supported.");
                candle_core::cuda::set_gemm_reduced_precision_bf16(false);
                INHIBIT_GEMM_F16.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    let a = Tensor::zeros((2, 2), DType::F16, &Device::new_cuda(0).unwrap()).unwrap();
    candle_core::cuda::set_gemm_reduced_precision_f16(true);
    match a.matmul(&a) {
        Ok(_) => tracing::info!("Enabling GEMM reduced precision in F16."),
        Err(e) => {
            if format!("{e:?}").contains("CUBLAS_STATUS_NOT_SUPPORTED") {
                tracing::info!("GEMM reduced precision in F16 not supported.");
                candle_core::cuda::set_gemm_reduced_precision_f16(false);
                INHIBIT_GEMM_F16.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn set_gemm_reduced_precision_f16() {}

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
            gemm_full_precision_f16,
        } = config;

        let model_supports_reduced_gemm = match pipeline.try_lock().unwrap().category() {
            ModelCategory::Text => true,
            ModelCategory::Vision { has_conv2d } => !has_conv2d,
        };
        if !gemm_full_precision_f16.unwrap_or(false) && model_supports_reduced_gemm {
            set_gemm_reduced_precision_f16();
        }
        setup_cublas_lt_wrapper();

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
        };

        let (tx, rx) = channel(10_000);

        let sender = RwLock::new(tx);
        let id = pipeline.try_lock().unwrap().name();

        let engine_handler = thread::spawn(move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async move {
                let mut engine = Engine::new(
                    rx,
                    pipeline,
                    method,
                    truncate_sequence,
                    no_kv_cache,
                    no_prefix_cache,
                    prefix_cache_n,
                    disable_eos_stop,
                );
                engine.run().await;
            });
        });

        Arc::new(Self {
            sender,
            log,
            id,
            creation_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time travel has occurred!")
                .as_secs(),
            next_request_id: Mutex::new(RefCell::new(0)),
            reboot_state,
            engine_handler: RwLock::new(engine_handler),
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
                    let mut engine = Engine::new(
                        rx,
                        reboot_state.pipeline.clone(),
                        reboot_state.method,
                        reboot_state.truncate_sequence,
                        reboot_state.no_kv_cache,
                        reboot_state.no_prefix_cache,
                        reboot_state.prefix_cache_n,
                        reboot_state.disable_eos_stop,
                    );
                    engine.run().await;
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
}
