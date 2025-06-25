mod merged_lora;
mod static_lora;
mod runtime_lora;

use std::{collections::HashSet, sync::{Arc, LazyLock, Mutex}};

use candle_core::Tensor;
pub use merged_lora::merge_lora_weights;
use serde::{Deserialize, Serialize};
pub use static_lora::{linear_no_bias_static_lora, StaticLoraConfig};

use crate::ShardedVarBuilder;

pub static APPLIED_LORAS: LazyLock<Arc<Mutex<Vec<LoraAdapter>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(Vec::new())));

pub const MULTI_LORA_DELIMITER: &str = ";";

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
    pub weights: ShardedVarBuilder,
}

pub struct InstantiatedLoraAdapter {
    pub a: Tensor,
    pub b: Tensor,
    pub scale: f64,
}
