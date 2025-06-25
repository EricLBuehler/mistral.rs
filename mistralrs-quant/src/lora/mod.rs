mod merged_lora;
mod static_lora;

use std::sync::{Arc, LazyLock, Mutex};

pub use merged_lora::{merge_lora_weights, LoraAdapter, LoraConfig};
pub use static_lora::{linear_no_bias_static_lora, StaticLoraConfig};

pub static APPLIED_LORAS: LazyLock<Arc<Mutex<Vec<LoraAdapter>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(Vec::new())));

pub const MULTI_LORA_DELIMITER: &str = ";";
