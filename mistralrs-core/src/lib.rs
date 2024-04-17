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
    GemmaLoader, GemmaSpecificConfig, LlamaLoader, LlamaSpecificConfig, Loader, MistralLoader,
    MistralSpecificConfig, MixtralLoader, MixtralSpecificConfig, ModelKind, Phi2Loader,
    Phi2SpecificConfig, TokenSource,
};
pub use request::{Constraint, Request, RequestMessage};
pub use response::Response;
pub use response::{ChatCompletionResponse, CompletionResponse, Usage};
pub use sampler::{SamplingParams, StopTokens};
pub use scheduler::SchedulerMethod;
use serde::Serialize;

pub struct MistralRs {
    sender: Sender<Request>,
    log: Option<String>,
    id: String,
    creation_time: u64,
    next_request_id: Mutex<RefCell<usize>>,
}

impl MistralRs {
    pub fn new(
        pipeline: Box<Mutex<dyn Pipeline>>,
        method: SchedulerMethod,
        log: Option<String>,
        truncate_sequence: bool,
        no_kv_cache: bool,
        no_prefix_cache: bool,
        prefix_cache_n: usize,
    ) -> Arc<Self> {
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
            let repr = serde_json::to_string(resp).unwrap();
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
