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
use pipeline::Pipeline;

mod engine;
mod models;
mod pipeline;
mod request;
mod response;
mod sampling;
mod scheduler;
mod sequence;
mod utils;

pub use pipeline::{Loader, MistralLoader, MistralSpecificConfig, TokenSource};
pub use request::Request;
pub use response::ChatCompletionResponse;
pub use response::Response;
pub use sampling::{SamplingParams, StopTokens};
pub use scheduler::SchedulerMethod;

pub struct MistralRs {
    sender: Sender<Request>,
    log: bool,
}

impl MistralRs {
    pub fn new(
        pipeline: Box<Mutex<dyn Pipeline>>,
        method: SchedulerMethod,
        log: bool,
    ) -> Arc<Self> {
        let (tx, rx) = channel();

        let this = Arc::new(Self { sender: tx, log });

        thread::spawn(move || {
            let mut engine = Engine::new(rx, pipeline, method);
            engine.run();
        });

        this
    }

    pub fn get_sender(&self) -> Sender<Request> {
        self.sender.clone()
    }

    pub fn maybe_log_request(this: Arc<Self>, repr: String) {
        if this.log {
            let mut f = OpenOptions::new()
                .append(true)
                .create(true) // Optionally create the file if it doesn't already exist
                .open("output.log")
                .expect("Unable to open file");
            let time = chrono::offset::Local::now();
            f.write_all(format!("Request at {time}: {repr}\n\n").as_bytes())
                .expect("Unable to write data");
        }
    }

    pub fn maybe_log_response(this: Arc<Self>, resp: &ChatCompletionResponse) {
        if this.log {
            let mut f = OpenOptions::new()
                .append(true)
                .create(true) // Optionally create the file if it doesn't already exist
                .open("output.log")
                .expect("Unable to open file");
            let time = chrono::offset::Local::now();
            let repr = serde_json::to_string(resp).unwrap();
            f.write_all(format!("Response at {time}: {repr}\n\n").as_bytes())
                .expect("Unable to write data");
        }
    }
}
