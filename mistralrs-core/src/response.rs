use std::{
    error::Error,
    fmt::{Debug, Display},
    sync::Arc,
};

use candle_core::Tensor;
#[cfg(feature = "pyo3_macros")]
use pyo3::{pyclass, pymethods};
use serde::Serialize;

use crate::{sampler::TopLogprob, tools::ToolCallResponse};

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
    pub content: Option<String>,
    pub role: String,
    pub tool_calls: Option<Vec<ToolCallResponse>>,
    /// Reasoning/analysis content from Harmony format (separate from final content).
    /// This contains chain-of-thought reasoning that is not intended for end users.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

generate_repr!(ResponseMessage);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Delta in content for streaming response.
pub struct Delta {
    pub content: Option<String>,
    pub role: String,
    pub tool_calls: Option<Vec<ToolCallResponse>>,
    /// Reasoning/analysis content delta from Harmony format.
    /// This contains incremental chain-of-thought reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

generate_repr!(Delta);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// A logprob with the top logprobs for this token.
pub struct ResponseLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
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
/// Chat completion streaming chunk choice.
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
/// Chat completion streaming chunk choice.
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<ResponseLogprob>,
    pub finish_reason: Option<String>,
}

generate_repr!(CompletionChunkChoice);

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
    pub usage: Option<Usage>,
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
    pub logprobs: Option<Logprobs>,
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

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
/// Completion request choice.
pub struct CompletionChunkResponse {
    pub id: String,
    pub choices: Vec<CompletionChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
}

generate_repr!(CompletionChunkResponse);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
pub struct ImageChoice {
    pub url: Option<String>,
    pub b64_json: Option<String>,
}

generate_repr!(ImageChoice);

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationResponse {
    pub created: u128,
    pub data: Vec<ImageChoice>,
}

generate_repr!(ImageGenerationResponse);

/// The response enum contains 3 types of variants:
/// - Error (-Error suffix)
/// - Chat (no prefix)
/// - Completion (Completion- prefix)
pub enum Response {
    InternalError(Box<dyn Error + Send + Sync>),
    ValidationError(Box<dyn Error + Send + Sync>),
    // Chat
    ModelError(String, ChatCompletionResponse),
    Done(ChatCompletionResponse),
    Chunk(ChatCompletionChunkResponse),
    // Completion
    CompletionModelError(String, CompletionResponse),
    CompletionDone(CompletionResponse),
    CompletionChunk(CompletionChunkResponse),
    // Image generation
    ImageGeneration(ImageGenerationResponse),
    // Speech generation
    Speech {
        pcm: Arc<Vec<f32>>,
        rate: usize,
        channels: usize,
    },
    // Raw
    Raw {
        logits_chunks: Vec<Tensor>,
        tokens: Vec<u32>,
    },
    Embeddings {
        embeddings: Vec<f32>,
        prompt_tokens: usize,
        total_tokens: usize,
    },
}

#[derive(Debug, Clone)]
pub enum ResponseOk {
    // Chat
    Done(ChatCompletionResponse),
    Chunk(ChatCompletionChunkResponse),
    // Completion
    CompletionDone(CompletionResponse),
    CompletionChunk(CompletionChunkResponse),
    // Image generation
    ImageGeneration(ImageGenerationResponse),
    // Speech generation
    Speech {
        pcm: Arc<Vec<f32>>,
        rate: usize,
        channels: usize,
    },
    // Raw
    Raw {
        logits_chunks: Vec<Tensor>,
        tokens: Vec<u32>,
    },
    // Embeddings
    Embeddings {
        embeddings: Vec<f32>,
        prompt_tokens: usize,
        total_tokens: usize,
    },
}

pub enum ResponseErr {
    InternalError(Box<dyn Error + Send + Sync>),
    ValidationError(Box<dyn Error + Send + Sync>),
    ModelError(String, ChatCompletionResponse),
    CompletionModelError(String, CompletionResponse),
}

impl Display for ResponseErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternalError(e) | Self::ValidationError(e) => Display::fmt(e, f),
            Self::ModelError(e, x) => f
                .debug_struct("ChatModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
            Self::CompletionModelError(e, x) => f
                .debug_struct("CompletionModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
        }
    }
}

impl Debug for ResponseErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternalError(e) | Self::ValidationError(e) => Debug::fmt(e, f),
            Self::ModelError(e, x) => f
                .debug_struct("ChatModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
            Self::CompletionModelError(e, x) => f
                .debug_struct("CompletionModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
        }
    }
}

impl std::error::Error for ResponseErr {}

impl Response {
    /// Convert the response into a result form.
    pub fn as_result(self) -> Result<ResponseOk, Box<ResponseErr>> {
        match self {
            Self::Done(x) => Ok(ResponseOk::Done(x)),
            Self::Chunk(x) => Ok(ResponseOk::Chunk(x)),
            Self::CompletionDone(x) => Ok(ResponseOk::CompletionDone(x)),
            Self::CompletionChunk(x) => Ok(ResponseOk::CompletionChunk(x)),
            Self::InternalError(e) => Err(Box::new(ResponseErr::InternalError(e))),
            Self::ValidationError(e) => Err(Box::new(ResponseErr::ValidationError(e))),
            Self::ModelError(e, x) => Err(Box::new(ResponseErr::ModelError(e, x))),
            Self::CompletionModelError(e, x) => {
                Err(Box::new(ResponseErr::CompletionModelError(e, x)))
            }
            Self::ImageGeneration(x) => Ok(ResponseOk::ImageGeneration(x)),
            Self::Speech {
                pcm,
                rate,
                channels,
            } => Ok(ResponseOk::Speech {
                pcm,
                rate,
                channels,
            }),
            Self::Raw {
                logits_chunks,
                tokens,
            } => Ok(ResponseOk::Raw {
                logits_chunks,
                tokens,
            }),
            Self::Embeddings {
                embeddings,
                prompt_tokens,
                total_tokens,
            } => Ok(ResponseOk::Embeddings {
                embeddings,
                prompt_tokens,
                total_tokens,
            }),
        }
    }
}
