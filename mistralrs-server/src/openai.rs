use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub content: String,
    pub role: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StopTokens {
    Multi(Vec<String>),
    Single(String),
    MultiId(Vec<u32>),
    SingleId(u32),
}

fn default_false() -> bool {
    false
}

fn default_1usize() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<Message>,
    pub model: String,
    pub logit_bias: Option<HashMap<u32, f32>>,
    #[serde(default = "default_false")]
    pub logprobs: bool,
    pub top_logprobs: Option<usize>,
    pub max_tokens: Option<usize>,
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    pub n_choices: usize,
    pub presence_penalty: Option<f32>,
    #[serde(rename = "frequency_penalty")]
    pub repetition_penalty: Option<f32>,
    #[serde(rename = "stop")]
    pub stop_seqs: Option<StopTokens>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stream: Option<bool>,

    // mistral.rs additional
    pub top_k: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelObjects {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}
