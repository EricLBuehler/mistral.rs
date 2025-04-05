use mistralrs_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{
    layers::{Activation, Llama3RopeConfig},
    serde_default_fn,
};

serde_default_fn!(bool, word_emb_default, false);
serde_default_fn!(bool, use_flash_attn_default, false);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct TextConfig {
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "use_flash_attn_default")]
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
    pub floor_scale: Option<f32>,
    pub attn_scale: Option<f32>,
    pub attn_temperature_tuning: Option<f32>,
    pub use_qk_norm: bool,
    pub moe_layers: Option<Vec<usize>>,
    pub interleave_moe_layer_step: usize,
    pub intermediate_size_mlp: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub attention_chunk_size: usize,
}

impl TextConfig {
    pub fn moe_layers(&self) -> Vec<usize> {
        self.moe_layers.clone().unwrap_or(
            (self.interleave_moe_layer_step - 1..self.num_hidden_layers)
                .step_by(self.interleave_moe_layer_step)
                .collect(),
        )
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Llama4Config {
    pub text_config: TextConfig,
}
