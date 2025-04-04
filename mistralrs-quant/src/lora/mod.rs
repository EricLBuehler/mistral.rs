mod static_lora;

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
pub use static_lora::linear_no_bias_static_lora;

use crate::ShardedSafeTensors;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StaticLoraConfig {
    pub layer: String,
    pub lora_alpha: f64,
    pub r: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoraConfig {
    #[serde(rename = "r")]
    pub rank: usize,
    #[serde(rename = "lora_alpha")]
    pub alpha: f64,
    pub target_modules: HashSet<String>,
}

pub struct LoraAdapter {
    pub config: LoraConfig,
    pub weights: ShardedSafeTensors,
}
