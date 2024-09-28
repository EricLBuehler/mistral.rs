//! This is just a hack to make `finish_reason` a string! Normally it is an enum

use mistralrs_core::{Delta, Logprobs, ResponseLogprob, ResponseMessage, Usage};

#[pyo3::pyclass]
#[pyo3(get_all)]
#[derive(Clone, Debug)]
pub struct Choice {
    pub finish_reason: String,
    pub index: usize,
    pub message: ResponseMessage,
    pub logprobs: Option<Logprobs>,
}

#[pyo3::pyclass]
#[pyo3(get_all)]
#[derive(Clone, Debug)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Usage,
}

#[pyo3::pyclass]
#[pyo3(get_all)]
#[derive(Clone, Debug)]
pub struct CompletionChoice {
    pub finish_reason: String,
    pub index: usize,
    pub text: String,
    pub logprobs: Option<()>,
}

#[pyo3::pyclass]
#[pyo3(get_all)]
#[derive(Clone, Debug)]
pub struct CompletionResponse {
    pub id: String,
    pub choices: Vec<CompletionChoice>,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Usage,
}

#[pyo3::pyclass]
#[pyo3(get_all)]
#[derive(Clone, Debug)]
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<ResponseLogprob>,
    pub finish_reason: Option<String>,
}

#[pyo3::pyclass]
#[pyo3(get_all)]
#[derive(Clone, Debug)]
pub struct CompletionChunkResponse {
    pub id: String,
    pub choices: Vec<CompletionChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
}

#[pyo3::pyclass]
#[pyo3(get_all)]
#[derive(Clone, Debug)]
pub struct ChunkChoice {
    pub finish_reason: Option<String>,
    pub index: usize,
    pub delta: Delta,
    pub logprobs: Option<ResponseLogprob>,
}

#[pyo3::pyclass]
#[pyo3(get_all)]
#[derive(Clone, Debug)]
pub struct ChatCompletionChunkResponse {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
}
