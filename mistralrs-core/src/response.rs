use std::error::Error;

#[cfg(feature = "pyo3_macros")]
use pyo3::{pyclass, pymethods};
use serde::Serialize;

use crate::sampler::TopLogprob;

pub const SYSTEM_FINGERPRINT: &str = "local";

macro_rules! generate_repr {
    ($t:ident) => {
        #[cfg(feature = "pyo3_macros")]
        #[pymethods]
        impl $t {
            fn __repr__(&self) -> String {
                format!("{self:#?}")
            }
        }
    };
}

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Chat completion response message.
pub struct ResponseMessage {
    pub content: String,
    pub role: String,
}

generate_repr!(ResponseMessage);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Delta in content for streaming response.
pub struct Delta {
    pub content: String,
    pub role: String,
}

generate_repr!(Delta);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// A logprob with the top logprobs for this token.
pub struct ResponseLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}

generate_repr!(ResponseLogprob);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Logprobs per token.
pub struct Logprobs {
    pub content: Option<Vec<ResponseLogprob>>,
}

generate_repr!(Logprobs);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Chat completion choice.
pub struct Choice {
    pub finish_reason: String,
    pub index: usize,
    pub message: ResponseMessage,
    pub logprobs: Option<Logprobs>,
}

generate_repr!(Choice);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Completion streaming chunk choice.
pub struct ChunkChoice {
    pub finish_reason: Option<String>,
    pub index: usize,
    pub delta: Delta,
    pub logprobs: Option<ResponseLogprob>,
}

generate_repr!(ChunkChoice);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// OpenAI compatible (superset) usage during a request.
pub struct Usage {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub avg_tok_per_sec: f32,
    pub avg_prompt_tok_per_sec: f32,
    pub avg_compl_tok_per_sec: f32,
    pub total_time_sec: f32,
    pub total_prompt_time_sec: f32,
    pub total_completion_time_sec: f32,
}

generate_repr!(Usage);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// An OpenAI compatible chat completion response.
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Usage,
}

generate_repr!(ChatCompletionResponse);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Chat completion streaming request chunk.
pub struct ChatCompletionChunkResponse {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
}

generate_repr!(ChatCompletionChunkResponse);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Completion request choice.
pub struct CompletionChoice {
    pub finish_reason: String,
    pub index: usize,
    pub text: String,
    pub logprobs: Option<()>,
}

generate_repr!(CompletionChoice);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// An OpenAI compatible completion response.
pub struct CompletionResponse {
    pub id: String,
    pub choices: Vec<CompletionChoice>,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Usage,
}

generate_repr!(CompletionResponse);

/// The response enum contains 3 types of variants:
/// - Error (-Error suffix)
/// - Chat (no suffix or prefix)
/// - Completion (Completion- prefix)
pub enum Response {
    InternalError(Box<dyn Error + Send + Sync>),
    ValidationError(Box<dyn Error + Send + Sync>),
    ModelError(String, ChatCompletionResponse),
    // Chat
    Done(ChatCompletionResponse),
    Chunk(ChatCompletionChunkResponse),
    // Completion
    CompletionModelError(String, CompletionResponse),
    CompletionDone(CompletionResponse),
}
