#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    fs::OpenOptions,
    io::Write,
    sync::{
        mpsc::{channel, Sender},
        Arc, Mutex,
    },
    thread,
};

use engine::Engine;
pub use mistralrs_lora::Ordering;
pub use pipeline::Pipeline;

mod engine;
mod models;
mod pipeline;
mod request;
mod response;
mod sampling;
mod scheduler;
mod sequence;
mod utils;
mod xlora_models;

pub use pipeline::{
    Conversation, GemmaLoader, GemmaSpecificConfig, LlamaLoader, LlamaSpecificConfig, Loader,
    MistralLoader, MistralSpecificConfig, MixtralLoader, MixtralSpecificConfig, ModelKind,
    TokenSource,
};
pub use request::Request;
pub use response::ChatCompletionResponse;
pub use response::Response;
pub use sampling::{SamplingParams, StopTokens};
pub use scheduler::SchedulerMethod;

pub struct MistralRs {
    sender: Sender<Request>,
    log: Option<String>,
}

impl MistralRs {
    pub fn new(
        pipeline: Box<Mutex<dyn Pipeline>>,
        method: SchedulerMethod,
        log: Option<String>,
        truncate_sequence: bool,
        no_kv_cache: bool,
    ) -> Arc<Self> {
        let (tx, rx) = channel();

        let this = Arc::new(Self { sender: tx, log });

        thread::spawn(move || {
            let mut engine = Engine::new(rx, pipeline, method, truncate_sequence, no_kv_cache);
            engine.run();
        });

        this
    }

    pub fn get_sender(&self) -> Sender<Request> {
        self.sender.clone()
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

    pub fn maybe_log_response(this: Arc<Self>, resp: &ChatCompletionResponse) {
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
}
