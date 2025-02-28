use mistralrs_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{
    layers::{Activation, Phi4MMRopeScalingConfig},
    serde_default_fn,
};

serde_default_fn!(bool, d_flash_attn, false);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMLoraConfig {
    pub layer: String,
    pub lora_alpha: f64,
    pub r: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub resid_pdrop: f64,
    pub embd_pdrop: f64,
    pub attention_dropout: f64,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rope_scaling: Option<Phi4MMRopeScalingConfig>,
    pub partial_rotary_factor: f64,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    pub sliding_window: Option<usize>,
    pub embd_layer: String,
    pub img_processor: Option<String>,
    pub audio_processor: Option<String>,
    pub vision_lora: Option<Phi4MMLoraConfig>,
    pub speech_lora: Option<Phi4MMLoraConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "d_flash_attn")]
    pub use_flash_attn: bool,
}

impl Phi4MMConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
