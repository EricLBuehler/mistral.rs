use either::Either;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, ops::Deref};
use utoipa::ToSchema;

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct MessageInnerContent(
    #[serde(with = "either::serde_untagged")] Either<String, HashMap<String, String>>,
);

impl Deref for MessageInnerContent {
    type Target = Either<String, HashMap<String, String>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct MessageContent(
    #[serde(with = "either::serde_untagged")]
    Either<String, Vec<HashMap<String, MessageInnerContent>>>,
);

impl Deref for MessageContent {
    type Target = Either<String, Vec<HashMap<String, MessageInnerContent>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct Message {
    pub content: MessageContent,
    pub role: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum StopTokens {
    Multi(Vec<String>),
    Single(String),
}

fn default_false() -> bool {
    false
}

fn default_1usize() -> usize {
    1
}

fn default_model() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type", content = "value")]
pub enum Grammar {
    #[serde(rename = "regex")]
    Regex(String),
    #[serde(rename = "yacc")]
    Yacc(String),
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ChatCompletionRequest {
    #[schema(example = json!(vec![Message{content:"Why did the crab cross the road?".to_string(), role:"user".to_string(), name: None}]))]
    #[serde(with = "either::serde_untagged")]
    pub messages: Either<Vec<Message>, String>,
    #[schema(example = "mistral")]
    #[serde(default = "default_model")]
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
    #[schema(example = json!(Option::None::<f32>))]
    pub frequency_penalty: Option<f32>,
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
    #[schema(example = json!(Option::None::<Grammar>))]
    pub grammar: Option<Grammar>,
    #[schema(example = json!(Option::None::<Vec<String>>))]
    pub adapters: Option<Vec<String>>,
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

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct CompletionRequest {
    #[schema(example = "mistral")]
    pub model: String,
    #[schema(example = "Say this is a test.")]
    pub prompt: String,
    #[serde(default = "default_1usize")]
    #[schema(example = 1)]
    pub best_of: usize,
    #[serde(rename = "echo")]
    #[serde(default = "default_false")]
    #[schema(example = false)]
    pub echo_prompt: bool,
    #[schema(example = json!(Option::None::<f32>))]
    pub presence_penalty: Option<f32>,
    #[schema(example = json!(Option::None::<f32>))]
    pub frequency_penalty: Option<f32>,
    #[schema(example = json!(Option::None::<HashMap<u32, f32>>))]
    pub logit_bias: Option<HashMap<u32, f32>>,
    #[schema(example = json!(Option::None::<usize>))]
    pub logprobs: Option<usize>,
    #[schema(example = 16)]
    pub max_tokens: Option<usize>,
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    #[schema(example = 1)]
    pub n_choices: usize,
    #[serde(rename = "stop")]
    #[schema(example = json!(Option::None::<StopTokens>))]
    pub stop_seqs: Option<StopTokens>,
    #[serde(rename = "stream")]
    pub _stream: Option<bool>,
    #[schema(example = 0.7)]
    pub temperature: Option<f64>,
    #[schema(example = json!(Option::None::<f64>))]
    pub top_p: Option<f64>,
    #[schema(example = json!(Option::None::<String>))]
    pub suffix: Option<String>,
    #[serde(rename = "user")]
    pub _user: Option<String>,

    // mistral.rs additional
    #[schema(example = json!(Option::None::<usize>))]
    pub top_k: Option<usize>,
    #[schema(example = json!(Option::None::<Grammar>))]
    pub grammar: Option<Grammar>,
    #[schema(example = json!(Option::None::<Vec<String>>))]
    pub adapters: Option<Vec<String>>,
}
