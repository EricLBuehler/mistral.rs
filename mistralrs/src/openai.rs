use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub content: String,
    pub role: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopTokens {
    Multi(Vec<String>),
    Single(String),
    MultiId(Vec<u32>),
    SingleId(u32),
}

fn default_0f64() -> f64 {
    0.0
}

fn default_false() -> bool {
    false
}

fn default_1usize() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<Message>,
    pub model: String,
    pub logit_bias: Option<HashMap<u32, f64>>,
    #[serde(default = "default_false")]
    pub logprobs: bool,
    pub top_logprobs: Option<usize>,
    pub max_tokens: Option<usize>,
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    pub n_choices: usize,
    #[serde(default = "default_0f64")]
    pub presence_penalty: f64,
    #[serde(rename = "stop")]
    pub stop_seqs: Option<StopTokens>, // TODO(EricLBuehler): We only support single tokens.
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,

    // mistral.rs additional
    pub repeat_penalty: Option<f32>,
    pub top_k: Option<usize>,
}
