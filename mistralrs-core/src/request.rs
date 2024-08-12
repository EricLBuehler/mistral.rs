use either::Either;
use indexmap::IndexMap;
use mistralrs_quant::IsqType;

use crate::{
    response::Response,
    sampler::SamplingParams,
    tools::{Tool, ToolChoice},
};
use std::fmt::Debug;
use tokio::sync::mpsc::Sender;

#[derive(Clone)]
/// Control the constraint with Regex or Yacc.
pub enum Constraint {
    Regex(String),
    Yacc(String),
    None,
}

pub type MessageContent = Either<String, Vec<IndexMap<String, String>>>;

#[derive(Clone, Debug)]
/// Message or messages for a [`Request`].
pub enum RequestMessage {
    Chat(Vec<IndexMap<String, MessageContent>>),
    Completion {
        text: String,
        echo_prompt: bool,
        best_of: usize,
    },
    CompletionTokens(Vec<u32>),
    VisionChat {
        images: Vec<image::DynamicImage>,
        messages: Vec<IndexMap<String, MessageContent>>,
    },
}

#[derive(Clone)]
/// A normal request request to the `MistralRs`
pub struct NormalRequest {
    pub messages: RequestMessage,
    pub sampling_params: SamplingParams,
    pub response: Sender<Response>,
    pub return_logprobs: bool,
    pub is_streaming: bool,
    pub id: usize,
    pub constraint: Constraint,
    pub suffix: Option<String>,
    pub adapters: Option<Vec<String>>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
}

impl NormalRequest {
    pub fn new_simple(
        messages: RequestMessage,
        sampling_params: SamplingParams,
        response: Sender<Response>,
        id: usize,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
    ) -> Self {
        Self {
            messages,
            sampling_params,
            response,
            id,
            tools,
            tool_choice,
            return_logprobs: false,
            is_streaming: false,
            constraint: Constraint::None,
            suffix: None,
            adapters: None,
        }
    }
}

#[derive(Clone)]
/// A request to the Engine, encapsulating the various parameters as well as
/// the `mspc` response `Sender` used to return the [`Response`].
pub enum Request {
    Normal(NormalRequest),
    ReIsq(IsqType),
    ActivateAdapters(Vec<String>),
}

impl Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Request::Normal(NormalRequest {
                messages,
                sampling_params,
                response: _,
                return_logprobs: _,
                is_streaming,
                id,
                constraint: _,
                suffix: _,
                adapters,
                tool_choice: _,
                tools: _,
            }) => {
                write!(
                    f,
                    "Request {id} {{ messages: `{messages:?}`, sampling_params: {sampling_params:?}, is_streaming: {is_streaming}, adapters: {adapters:?}}}",
                )
            }
            Request::ActivateAdapters(adapters) => {
                write!(f, "Activate Adapters Request {adapters:?}",)
            }
            Request::ReIsq(tp) => {
                write!(f, "Re ISQ Request {tp:?}",)
            }
        }
    }
}
