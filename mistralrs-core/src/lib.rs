use std::{
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
use request::Request;
pub use scheduler::SchedulerMethod;

pub struct MistralRs {
    sender: Sender<Request>,
}

impl MistralRs {
    pub fn new(pipeline: Box<Mutex<dyn Pipeline>>, method: SchedulerMethod) -> Arc<Self> {
        let (tx, rx) = channel();

        let this = Arc::new(Self { sender: tx });

        thread::spawn(move || {
            let mut engine = Engine::new(rx, pipeline, method);
            engine.run();
        });

        this
    }

    pub fn get_sender(&self) -> Sender<Request> {
        self.sender.clone()
    }
}
