use mistralrs_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{
    layers::{Activation, Llama3RopeConfig},
    serde_default_fn,
};

serde_default_fn!(bool, word_emb_default, false);
serde_default_fn!(Option<f32>, attn_temperature_tuning, Some(4.));
serde_default_fn!(Option<f32>, floor_scale, Some(8192.));
serde_default_fn!(Option<f32>, attn_scale, Some(0.1));

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct TextConfig {
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
    #[serde(default = "floor_scale")]
    pub floor_scale: Option<f32>,
    #[serde(default = "attn_scale")]
    pub attn_scale: Option<f32>,
    #[serde(default = "attn_temperature_tuning")]
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

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub hidden_act: Activation,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub intermediate_size: usize,
    pub vision_output_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub norm_eps: f64,
    pub pixel_shuffle_ratio: f32,
    pub projector_input_dim: usize,
    pub projector_output_dim: usize,
    pub vision_feature_layer: isize,
    pub rope_theta: f32,
}

impl VisionConfig {
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2) + 1
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Llama4Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_index: usize,
    /// Top-level quantization config that should be applied to text_config
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl Llama4Config {
    /// Propagate top-level quantization_config to text_config if text_config doesn't have one
    pub fn propagate_quantization_config(&mut self) {
        if self.text_config.quantization_config.is_none() && self.quantization_config.is_some() {
            self.text_config.quantization_config = self.quantization_config.clone();
        }
    }
}
