use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct Message {
    pub content: String,
    pub role: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
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

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ChatCompletionRequest {
    #[schema(example = json!(vec![Message{content:"Why did the crab cross the road?".to_string(), role:"user".to_string(), name: None}]))]
    pub messages: Vec<Message>,
    #[schema(example = "mistral")]
    pub model: String,
    #[schema(example = json!(Option::None::<HashMap<u32, f32>>))]
    pub logit_bias: Option<HashMap<u32, f32>>,
    #[serde(default = "default_false")]
    #[schema(example = false)]
    pub logprobs: bool,
    #[schema(example = json!(Option::None::<usize>))]
    pub top_logprobs: Option<usize>,
    #[schema(example = 256)]
    pub max_tokens: Option<usize>,
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    #[schema(example = 1)]
    pub n_choices: usize,
    #[schema(example = json!(Option::None::<f32>))]
    pub presence_penalty: Option<f32>,
    #[serde(rename = "frequency_penalty")]
    #[schema(example = json!(Option::None::<f32>))]
    pub repetition_penalty: Option<f32>,
    #[serde(rename = "stop")]
    #[schema(example = json!(Option::None::<StopTokens>))]
    pub stop_seqs: Option<StopTokens>,
    #[schema(example = 0.7)]
    pub temperature: Option<f64>,
    #[schema(example = json!(Option::None::<f64>))]
    pub top_p: Option<f64>,
    #[schema(example = true)]
    pub stream: Option<bool>,

    // mistral.rs additional
    #[schema(example = json!(Option::None::<usize>))]
    pub top_k: Option<usize>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelObjects {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}
