use either::Either;
use indexmap::IndexMap;
use mistralrs_quant::IsqType;
use serde::{Deserialize, Serialize};

use crate::{
    response::Response,
    sampler::SamplingParams,
    tools::{Tool, ToolChoice},
    CustomLogitsProcessor, DiffusionGenerationParams,
};
use std::{fmt::Debug, sync::Arc};
use tokio::sync::mpsc::Sender;

#[derive(Clone)]
/// Control the constraint with Regex or Yacc.
pub enum Constraint {
    Regex(String),
    Yacc(String),
    None,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
/// Image generation response format
pub enum ImageGenerationResponseFormat {
    Url,
    B64Json,
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
    ImageGeneration {
        prompt: String,
        format: ImageGenerationResponseFormat,
        generation_params: DiffusionGenerationParams,
    },
}

#[derive(Clone)]
/// A normal request request to the `MistralRs`.
/// - `messages`: Messages for the request
/// - `sampling_params`: Sampling parameters for generation
/// - `response`: Object to send the result through
/// - `return_logprobs`: Whether to return logprobs
/// - `is_streaming`: Control whether the request is streaming, if so chunk responses will be sent
/// - `id`: Request ID
/// - `constraint`: Constraint to use during generation
/// - `suffix`: Suffix to add
/// - `adapters`: Adapters to use in this request
/// - `tools`: Tools available in this request
/// - `tool_choice`: Choice of tools
/// - `logits_processors`: Custom logits processors. Order of application:
///     1) Apply penalties from `sampling_params`
///     2) Apply these custom logits processors sequentially
///     3) Apply temperature and softmax
///     4) Sample the next token (topk, topp, minp, etc)
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
    pub logits_processors: Option<Vec<Arc<dyn CustomLogitsProcessor>>>,
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
            logits_processors: None,
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
    // Sending a terminate request causes the `run` function to return to the thread created in `MistralRs::new`,
    // and then Engine will be dropped.
    Terminate,
}

impl Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Request::Normal(NormalRequest {
                messages,
                sampling_params,
                is_streaming,
                adapters,
                id,
                ..
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
            Request::Terminate => write!(f, "Termination Request"),
        }
    }
}
