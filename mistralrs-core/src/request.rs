use crate::{response::Response, sampling::SamplingParams};
use std::{collections::HashMap, fmt::Debug, sync::mpsc::Sender};

pub struct Request {
    pub messages: Vec<HashMap<String, String>>,
    pub sampling_params: SamplingParams,
    pub response: Sender<Response>,
    pub return_logprobs: bool,
}

impl Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Request {{ messages: `{:?}`, sampling_params: {:?}}}",
            self.messages, self.sampling_params
        )
    }
}
