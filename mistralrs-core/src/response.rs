use std::error::Error;

use serde::Serialize;

use crate::sampler::TopLogprob;

pub const SYSTEM_FINGERPRINT: &str = "local";

#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    pub content: String,
    pub role: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    pub content: String,
    pub role: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Logprobs {
    pub content: Option<Vec<ResponseLogprob>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    #[serde(rename = "finish_reason")]
    pub stopreason: String,
    pub index: usize,
    pub message: ResponseMessage,
    pub logprobs: Option<Logprobs>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    #[serde(rename = "finish_reason")]
    pub stopreason: Option<String>,
    pub index: usize,
    pub delta: Delta,
    pub logprobs: Option<ResponseLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionUsage {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub avg_tok_per_sec: f32,
    pub avg_prompt_tok_per_sec: f32,
    pub avg_compl_tok_per_sec: f32,
    pub avg_sample_tok_per_sec: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: ChatCompletionUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunkResponse {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
}

pub enum Response {
    Error(Box<dyn Error + Send + Sync>),
    Done(ChatCompletionResponse),
    Chunk(ChatCompletionChunkResponse),
}
