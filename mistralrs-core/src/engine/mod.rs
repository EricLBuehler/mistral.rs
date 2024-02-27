use std::{
    collections::VecDeque,
    sync::{mpsc::Receiver, Arc},
};

use crate::{
    pipeline::Pipeline,
    request::{Request, Sequence},
    response::Response,
    scheduler::Scheduler,
};

pub struct Engine {
    rx: Receiver<Request>,
    pipeline: Arc<dyn Pipeline>,
    requests: VecDeque<Request>,
    scheduler: Scheduler<VecDeque<Sequence>>,
    id: usize,
}

impl Engine {
    pub fn new(rx: Receiver<Request>, pipeline: Arc<dyn Pipeline>) -> Self {
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
            let scheduled = self.scheduler.schedule();
        }
    }

    fn add_request(&mut self, request: Request) {
        let prompt = match self.pipeline.tokenize_prompt(&request.prompt) {
            Ok(prompt) => prompt,
            Err(e) => {
                // Unwrap reasoning: The reciever should really be there, otherwise it is their fault.
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
