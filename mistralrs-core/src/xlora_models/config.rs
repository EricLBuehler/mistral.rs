use std::collections::HashMap;

use either::Either;
use serde::Deserialize;

fn true_default() -> bool {
    true
}

fn false_default() -> bool {
    false
}
fn default_1() -> usize {
    1
}

fn default_2048() -> usize {
    1
}
fn default_dropout() -> f32 {
    0.2
}
fn default_1f64() -> f64 {
    1.0
}
fn default_0f64() -> f64 {
    0.0
}

#[derive(Clone, Debug, Deserialize)]
pub struct XLoraConfig {
    pub hidden_size: usize,
    pub base_model_id: String,
    #[serde(rename = "adapters")]
    #[serde(with = "either::serde_untagged")]
    pub _adapters: Either<Vec<String>, HashMap<String, String>>,
    #[serde(default = "false_default")]
    pub layerwise_scalings: bool,
    #[serde(default = "false_default")]
    pub enable_relu_and_dropout: bool,
    #[serde(default = "default_1")]
    pub xlora_depth: usize,
    #[serde(default = "default_2048")]
    pub xlora_size: usize,
    #[serde(default = "default_dropout")]
    pub xlora_dropout_p: f32,
    #[serde(default = "true_default")]
    pub enable_softmax: bool,
    #[serde(default = "default_1f64")]
    pub softmax_temperature: f64,
    #[serde(default = "default_0f64")]
    pub scaling_pass_value: f64,
    #[serde(default = "false_default", rename = "use_trainable_adapters")]
    pub _use_trainable_adapters: bool,
    #[serde(default = "true_default")]
    pub use_bias: bool,
    #[serde(default = "default_1f64")]
    pub global_scaling_weight: f64,
    pub top_k_lora: Option<usize>,
    #[serde(default = "false_default")]
    pub enable_softmax_topk: bool,
}
