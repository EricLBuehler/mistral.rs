use either::Either;
use indexmap::IndexMap;
use mistralrs_audio::AudioInput;
use mistralrs_quant::IsqType;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    response::Response, sampler::SamplingParams, tools::ToolChoice, CustomLogitsProcessor,
    DiffusionGenerationParams, Tool,
};
use std::{fmt::Debug, path::PathBuf, sync::Arc};
use tokio::sync::mpsc::Sender;

pub type LlguidanceGrammar = llguidance::api::TopLevelGrammar;

#[derive(Clone, Serialize, Deserialize)]
/// Control the constraint with llguidance.
pub enum Constraint {
    Regex(String),
    Lark(String),
    JsonSchema(serde_json::Value),
    Llguidance(LlguidanceGrammar),
    None,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
/// Image generation response format
pub enum ImageGenerationResponseFormat {
    Url,
    B64Json,
}

pub type MessageContent = Either<String, Vec<IndexMap<String, Value>>>;

/// Reasoning effort level for models that support it (e.g., GPT-OSS with Harmony format).
/// Controls the depth of reasoning/analysis in the model's response.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Minimal reasoning, faster responses
    Low,
    /// Balanced reasoning depth
    #[default]
    Medium,
    /// Deep reasoning, more thorough analysis
    High,
}

impl ReasoningEffort {
    /// Convert to string representation for chat template
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Message or messages for a [`Request`].
pub enum RequestMessage {
    Chat {
        messages: Vec<IndexMap<String, MessageContent>>,
        enable_thinking: Option<bool>,
        /// Reasoning effort level for Harmony-format models
        reasoning_effort: Option<ReasoningEffort>,
    },
    Completion {
        text: String,
        echo_prompt: bool,
        best_of: Option<usize>,
    },
    CompletionTokens(Vec<u32>),
    VisionChat {
        #[serde(skip)] // TODO
        images: Vec<image::DynamicImage>,
        #[serde(skip)] // TODO
        audios: Vec<AudioInput>,
        messages: Vec<IndexMap<String, MessageContent>>,
        enable_thinking: Option<bool>,
        /// Reasoning effort level for Harmony-format models
        reasoning_effort: Option<ReasoningEffort>,
    },
    ImageGeneration {
        prompt: String,
        format: ImageGenerationResponseFormat,
        generation_params: DiffusionGenerationParams,
        save_file: Option<PathBuf>,
    },
    SpeechGeneration {
        prompt: String,
    },
    Embedding {
        prompt: String,
    },
    EmbeddingTokens {
        prompt: Vec<u32>,
    },
}

fn default_responder<T>() -> Sender<T> {
    let (sender, _) = tokio::sync::mpsc::channel(1);
    sender
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum SearchContextSize {
    #[serde(rename = "low")]
    Low,
    #[default]
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ApproximateUserLocation {
    pub city: String,
    pub country: String,
    pub region: String,
    pub timezone: String,
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum WebSearchUserLocation {
    #[serde(rename = "approximate")]
    Approximate {
        approximate: ApproximateUserLocation,
    },
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct WebSearchOptions {
    pub search_context_size: Option<SearchContextSize>,
    pub user_location: Option<WebSearchUserLocation>,
    /// Override the description for the search tool.
    pub search_description: Option<String>,
    /// Override the description for the extraction tool.
    pub extract_description: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
/// A normal request request to the `MistralRs`.
/// - `messages`: Messages for the request
/// - `sampling_params`: Sampling parameters for generation
/// - `response`: Object to send the result through
/// - `return_logprobs`: Whether to return logprobs
/// - `is_streaming`: Control whether the request is streaming, if so chunk responses will be sent
/// - `id`: Request ID
/// - `constraint`: Constraint to use during generation
/// - `suffix`: Suffix to add
/// - `tools`: Tools available in this request
/// - `tool_choice`: Choice of tools
/// - `logits_processors`: Custom logits processors. Order of application:
///     1) Apply penalties from `sampling_params`
///     2) Apply these custom logits processors sequentially
///     3) Apply temperature and softmax
///     4) Sample the next token (topk, topp, minp, etc)
/// - `return_raw_logits`: Return raw logits.
/// - `truncate_sequence`: Whether to truncate the prompt if it exceeds the model's maximum context length.
pub struct NormalRequest {
    pub messages: RequestMessage,
    pub sampling_params: SamplingParams,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<Response>,
    pub return_logprobs: bool,
    pub is_streaming: bool,
    pub id: usize,
    pub constraint: Constraint,
    pub suffix: Option<String>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip)]
    pub logits_processors: Option<Vec<Arc<dyn CustomLogitsProcessor>>>,
    pub return_raw_logits: bool,
    pub web_search_options: Option<WebSearchOptions>,
    pub model_id: Option<String>,
    #[serde(default)]
    pub truncate_sequence: bool,
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
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
            model_id: None,
            truncate_sequence: false,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
/// Request to tokenize some messages or some text.
/// - `add_generation_prompt` is only applicable if chat messages are provided and not a raw string.
pub struct TokenizationRequest {
    pub text: Either<Vec<IndexMap<String, MessageContent>>, String>,
    pub tools: Option<Vec<Tool>>,
    pub add_generation_prompt: bool,
    pub add_special_tokens: bool,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<anyhow::Result<Vec<u32>>>,
}

#[derive(Clone, Serialize, Deserialize)]
/// Request to detokenize some text.
pub struct DetokenizationRequest {
    pub tokens: Vec<u32>,
    pub skip_special_tokens: bool,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<anyhow::Result<String>>,
}

#[derive(Clone, Serialize, Deserialize)]
/// A request to the Engine, encapsulating the various parameters as well as
/// the `mpsc` response `Sender` used to return the [`Response`].
pub enum Request {
    Normal(Box<NormalRequest>),
    ReIsq(IsqType),
    Tokenize(TokenizationRequest),
    Detokenize(DetokenizationRequest),
    // Sending a terminate request causes the `run` function to return to the thread created in `MistralRs::new`,
    // and then Engine will be dropped.
    Terminate,
    TerminateAllSeqsNextStep,
}

impl Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Request::Normal(boxed_req) => {
                let NormalRequest {
                    messages,
                    sampling_params,
                    is_streaming,
                    id,
                    ..
                } = &**boxed_req;
                write!(
                    f,
                    "Request {id} {{ messages: `{messages:?}`, sampling_params: {sampling_params:?}, is_streaming: {is_streaming}}}",
                )
            }
            Request::ReIsq(tp) => {
                write!(f, "Re ISQ Request {tp:?}",)
            }
            Request::Tokenize(req) => {
                write!(f, "Tokenization Request {:?}", req.text)
            }
            Request::Detokenize(req) => {
                write!(f, "Tokenization Request {:?}", req.tokens)
            }
            Request::Terminate => write!(f, "Termination Request"),
            Request::TerminateAllSeqsNextStep => write!(f, "Terminate All Seqs Next Step"),
        }
    }
}
