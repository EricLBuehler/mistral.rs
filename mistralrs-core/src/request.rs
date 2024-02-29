use crate::{response::Response, sampling::SamplingParams};
use std::{fmt::Debug, sync::mpsc::Sender};

pub struct Request {
    pub prompt: String,
    pub sampling_params: SamplingParams,
    pub response: Sender<Response>,
}

impl Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Request {{ prompt: `{}`, sampling_params: {:?}}}",
            self.prompt, self.sampling_params
        )
    }
}
