use indexmap::IndexMap;

use crate::{response::Response, sampler::SamplingParams};
use std::fmt::Debug;
use tokio::sync::mpsc::Sender;

#[derive(Clone)]
/// Control the constraint with Regex or Yacc.
pub enum Constraint {
    Regex(String),
    Yacc(String),
    None,
}

#[derive(Clone, Debug)]
/// Message or messages for a [`Request`].
pub enum RequestMessage {
    Chat(Vec<IndexMap<String, String>>),
    Completion {
        text: String,
        echo_prompt: bool,
        best_of: usize,
    },
    CompletionTokens(Vec<u32>),
}

#[derive(Clone)]
/// A request to the Engine, encapsulating the various parameters as well as
/// the `mspc` response `Sender` used to return the [`Response`].
pub struct Request {
    pub messages: RequestMessage,
    pub sampling_params: SamplingParams,
    pub response: Sender<Response>,
    pub return_logprobs: bool,
    pub is_streaming: bool,
    pub id: usize,
    pub constraint: Constraint,
    pub suffix: Option<String>,
}

impl Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Request {} {{ messages: `{:?}`, sampling_params: {:?}}}",
            self.id, self.messages, self.sampling_params
        )
    }
}
