use std::collections::HashMap;

use candle_core::Device;
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
fn default_dropout() -> f64 {
    0.2
}
fn default_1_temp() -> f64 {
    1.0
}
fn default_0f64() -> f64 {
    0.0
}

#[derive(Debug, Deserialize)]
pub struct XLoraConfig {
    /*
    -hidden_size: int
    X device: torch.device
    X adapters: Dict[str, str]
    -enable_softmax: bool = True
    enable_softmax_topk: bool = False
    -layerwise_scalings: bool = False
    -xlora_depth: int = 1
    -xlora_size: int = 2048
    enable_relu_and_dropout: bool = False
    use_bias: bool = True
    -xlora_dropout_p: float = 0.2
    stop_token_id: Optional[int] = None
    use_trainable_adapters: bool = False
    -softmax_temperature: float = 1.0
    top_k_lora: Optional[int] = None
    -scaling_pass_value: float = 0.0
    global_scaling_weight: float = 1.0 */
    pub hidden_size: usize,
    #[serde(default = "false_default")]
    pub layerwise_scalings: bool,
    #[serde(default = "default_1")]
    pub xlora_depth: usize,
    #[serde(default = "default_2048")]
    pub xlora_size: usize,
    #[serde(default = "default_dropout")]
    pub xlora_dropout_p: f64,
    #[serde(default = "true_default")]
    pub enable_softmax: bool,
    #[serde(default = "default_1_temp")]
    pub softmax_temperature: f64,
    #[serde(default = "default_0f64")]
    pub scaling_pass_value: f64,
}
