use either::Either;
use indexmap::IndexMap;

use crate::{response::Response, sampler::SamplingParams};
use std::{fmt::Debug, sync::mpsc::Sender};

pub enum Constraint {
    Regex(String),
    Yacc(String),
    None,
}

#[derive(Debug, PartialEq)]
pub enum RequestType {
    Chat,
    Completion,
}

pub struct Request {
    pub messages: Either<Vec<IndexMap<String, String>>, String>,
    pub sampling_params: SamplingParams,
    pub response: Sender<Response>,
    pub return_logprobs: bool,
    pub is_streaming: bool,
    pub id: usize,
    pub constraint: Constraint,
    pub request_type: RequestType,
    pub suffix: Option<String>,
}

impl Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Request {} ({:?}) {{ messages: `{:?}`, sampling_params: {:?}}}",
            self.id, self.request_type, self.messages, self.sampling_params
        )
    }
}
