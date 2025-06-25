mod merged_lora;
mod static_lora;
mod runtime_lora;

use std::{cell::RefCell, collections::HashSet};

use candle_core::Tensor;
pub use merged_lora::merge_lora_weights;
use serde::{Deserialize, Serialize};
pub use static_lora::{linear_no_bias_static_lora, StaticLoraConfig};

use crate::{Shard, ShardedVarBuilder};

thread_local! {
    static ENGINE_APPLIED_LORAS: RefCell<Vec<LoraAdapter>> = const { RefCell::new(Vec::new()) };
}

/// Get the LoRA adapters for the current engine thread
pub fn get_applied_loras() -> Vec<LoraAdapter> {
    ENGINE_APPLIED_LORAS.with(|loras| loras.borrow().clone())
}

/// Push a LoRA adapter for the current engine thread
pub fn push_applied_lora(adapter: LoraAdapter) {
    ENGINE_APPLIED_LORAS.with(|loras| loras.borrow_mut().push(adapter));
}

/// Clear all LoRA adapters for the current engine thread
pub fn clear_applied_loras() {
    ENGINE_APPLIED_LORAS.with(|loras| loras.borrow_mut().clear());
}

pub const MULTI_LORA_DELIMITER: &str = ";";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoraConfig {
    #[serde(rename = "r")]
    pub rank: usize,
    #[serde(rename = "lora_alpha")]
    pub alpha: f64,
    pub target_modules: HashSet<String>,
}

#[derive(Clone)]
pub struct LoraAdapter {
    pub config: LoraConfig,
    pub weights: ShardedVarBuilder,
}

pub struct InstantiatedLoraAdapter {
    pub a: Tensor,
    pub b: Tensor,
    pub scale: f64,
}
