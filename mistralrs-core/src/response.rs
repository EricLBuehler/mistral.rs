use std::error::Error;

use pyo3::pyclass;
use serde::Serialize;

use crate::sampler::TopLogprob;

pub const SYSTEM_FINGERPRINT: &str = "local";

#[pyclass]
#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    pub content: String,
    pub role: String,
}

#[pyclass]
#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    pub content: String,
    pub role: String,
}

#[pyclass]
#[derive(Debug, Clone, Serialize)]
pub struct ResponseLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[pyclass]
#[derive(Debug, Clone, Serialize)]
pub struct Logprobs {
    pub content: Option<Vec<ResponseLogprob>>,
}

#[pyclass]
#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub finish_reason: String,
    pub index: usize,
    pub message: ResponseMessage,
    pub logprobs: Option<Logprobs>,
}

#[pyclass]
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub finish_reason: Option<String>,
    pub index: usize,
    pub delta: Delta,
    pub logprobs: Option<ResponseLogprob>,
}

#[pyclass]
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

#[pyclass]
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

#[pyclass]
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunkResponse {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
}

#[pyclass]
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub finish_reason: String,
    pub index: usize,
    pub text: String,
    pub logprobs: Option<()>,
}

#[pyclass]
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
