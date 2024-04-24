#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    cell::RefCell,
    error::Error,
    fs::OpenOptions,
    io::Write,
    sync::{
        mpsc::{channel, Sender},
        Arc, Mutex,
    },
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use engine::Engine;
pub use mistralrs_lora::Ordering;
pub use pipeline::Pipeline;

mod aici;
mod engine;
mod model_loader;
pub use model_loader::{get_tgt_non_granular_index, LoaderBuilder};
mod model_selected;
pub use model_selected::ModelSelected;

mod models;
mod pipeline;
mod prefix_cacher;
mod request;
mod response;
mod sampler;
mod scheduler;
mod sequence;
mod utils;
mod xlora_models;

pub use pipeline::{
    GGMLLoader, GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoader, GGUFLoaderBuilder,
    GGUFSpecificConfig, GemmaLoader, LlamaLoader, Loader, MistralLoader, MixtralLoader, ModelKind,
    NormalLoader, NormalLoaderBuilder, NormalLoaderType, NormalSpecificConfig, Phi2Loader,
    TokenSource,
};
pub use request::{Constraint, Request, RequestMessage};
pub use response::Response;
pub use response::*;
pub use sampler::{SamplingParams, StopTokens, TopLogprob};
pub use scheduler::SchedulerMethod;
use serde::Serialize;

/// The MistralRs struct handles sending requests to the engine.
/// It is the core multi-threaded component of mistral.rs, and uses `mspc`
/// `Sender` and `Receiver` primitives to send and receive requests to the
/// engine.
pub struct MistralRs {
    sender: Sender<Request>,
    log: Option<String>,
    id: String,
    creation_time: u64,
    next_request_id: Mutex<RefCell<usize>>,
}

/// The MistralRsBuilder takes the pipeline and a scheduler method and constructs
/// an Engine and a MistralRs instance. The Engine runs on a separate thread, and the MistralRs
/// instance stays on the calling thread.
pub struct MistralRsBuilder {
    pipeline: Box<Mutex<dyn Pipeline>>,
    method: SchedulerMethod,
    log: Option<String>,
    truncate_sequence: Option<bool>,
    no_kv_cache: Option<bool>,
    no_prefix_cache: Option<bool>,
    prefix_cache_n: Option<usize>,
    disable_eos_stop: Option<bool>,
}

impl MistralRsBuilder {
    pub fn new(pipeline: Box<Mutex<dyn Pipeline>>, method: SchedulerMethod) -> Self {
        Self {
            pipeline,
            method,
            log: None,
            truncate_sequence: None,
            no_kv_cache: None,
            no_prefix_cache: None,
            prefix_cache_n: None,
            disable_eos_stop: None,
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
        } = config;

        let truncate_sequence = truncate_sequence.unwrap_or(false);
        let no_kv_cache = no_kv_cache.unwrap_or(false);
        let no_prefix_cache = no_prefix_cache.unwrap_or(false);
        let prefix_cache_n = prefix_cache_n.unwrap_or(16);
        let disable_eos_stop = disable_eos_stop.unwrap_or(false);

        let (tx, rx) = channel();

        let this = Arc::new(Self {
            sender: tx,
            log,
            id: pipeline.lock().unwrap().name(),
            creation_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time travel has occurred!")
                .as_secs(),
            next_request_id: Mutex::new(RefCell::new(0)),
        });

        thread::spawn(move || {
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
            engine.run();
        });

        this
    }

    pub fn get_sender(&self) -> Sender<Request> {
        self.sender.clone()
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
