use std::{
    collections::VecDeque,
    iter::repeat,
    sync::{mpsc::Receiver, Mutex},
};

use candle_core::{Device, Tensor};

use crate::{
    pipeline::Pipeline,
    request::{Request, Sequence},
    response::Response,
    scheduler::Scheduler,
};

pub struct Engine {
    rx: Receiver<Request>,
    pipeline: Box<Mutex<dyn Pipeline>>,
    requests: VecDeque<Request>,
    scheduler: Scheduler<VecDeque<Sequence>>,
    id: usize,
}

macro_rules! get_mut_pipeline {
    ($pipeline:expr) => {
        loop {
            if let Ok(inner) = $pipeline.lock() {
                break inner;
            }
        }
    };
}

impl Engine {
    pub fn new(rx: Receiver<Request>, pipeline: Box<Mutex<dyn Pipeline>>) -> Self {
        Self {
            rx,
            pipeline,
            requests: VecDeque::new(),
            scheduler: Scheduler::new(),
            id: 0,
        }
    }

    pub fn run(&mut self) {
        loop {
            if let Ok(request) = self.rx.recv() {
                self.add_request(request);
            }
            let mut scheduled = self.scheduler.schedule();
            let logits = get_mut_pipeline!(self.pipeline)
                .forward(scheduled.seqs.iter_mut().collect::<Vec<_>>());
        }
    }

    fn add_request(&mut self, request: Request) {
        let prompt = match get_mut_pipeline!(self.pipeline).tokenize_prompt(&request.prompt) {
            Ok(prompt) => prompt,
            Err(e) => {
                // NOTE(EricLBuehler): Unwrap reasoning: The reciever should really be there, otherwise it is their fault.
                request.response.send(Response::Error(e.into())).unwrap();
                return;
            }
        };
        let seq = Sequence::new_waiting(prompt, self.id);
        self.id += 1;

        self.requests.push_back(request);
        self.scheduler.add_seq(seq);
    }
}
