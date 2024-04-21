use std::error::Error;

use pyo3::{pyclass, pymethods};
use serde::Serialize;

use crate::sampler::TopLogprob;

pub const SYSTEM_FINGERPRINT: &str = "local";

macro_rules! generate_repr {
    ($t:ident) => {
        #[pymethods]
        impl $t {
            fn __repr__(&self) -> String {
                format!("{self:#?}")
            }
        }
    };
}

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    pub content: String,
    pub role: String,
}

generate_repr!(ResponseMessage);

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    pub content: String,
    pub role: String,
}

generate_repr!(Delta);

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, Serialize)]
pub struct ResponseLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}

generate_repr!(ResponseLogprob);

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, Serialize)]
pub struct Logprobs {
    pub content: Option<Vec<ResponseLogprob>>,
}

generate_repr!(Logprobs);

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub finish_reason: String,
    pub index: usize,
    pub message: ResponseMessage,
    pub logprobs: Option<Logprobs>,
}

generate_repr!(Choice);

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub finish_reason: Option<String>,
    pub index: usize,
    pub delta: Delta,
    pub logprobs: Option<ResponseLogprob>,
}

generate_repr!(ChunkChoice);

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, Serialize)]
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

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, Serialize)]
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

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunkResponse {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
}

generate_repr!(ChatCompletionChunkResponse);

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub finish_reason: String,
    pub index: usize,
    pub text: String,
    pub logprobs: Option<()>,
}

generate_repr!(CompletionChoice);

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, Serialize)]
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
}
