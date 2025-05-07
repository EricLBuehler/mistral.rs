use mistralrs_quant::QuantizedConfig;

use crate::{
    layers::{Activation, Gemma3RopeScalingConfig},
    serde_default_fn,
    vision_models::siglip::SiglipVisionConfig,
};

serde_default_fn!(bool, attention_bias, false);
serde_default_fn!(usize, head_dim, 256);
serde_default_fn!(Activation, hidden_activation, Activation::GeluPytorchTanh);
serde_default_fn!(f64, rms_norm_eps, 1e-6);
serde_default_fn!(f64, rope_theta, 1000000.);
serde_default_fn!(usize, vocab_size, 262208);
serde_default_fn!(bool, tie_word_embeddings, true);
serde_default_fn!(usize, query_pre_attn_scalar, 256);
serde_default_fn!(usize, max_position_embeddings, 131072);
serde_default_fn!(f64, rope_local_base_freq, 10000.);
serde_default_fn!(usize, sliding_window_pattern, 6);
serde_default_fn!(usize, num_attention_heads, 8);
serde_default_fn!(usize, num_key_value_heads, 4);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3TextConfig {
    #[serde(default = "attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "head_dim")]
    pub head_dim: usize,
    #[serde(default = "hidden_activation")]
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default = "num_attention_heads")]
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "vocab_size")]
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    #[serde(default = "query_pre_attn_scalar")]
    pub query_pre_attn_scalar: usize,
    #[serde(default = "max_position_embeddings")]
    pub max_position_embeddings: usize,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "rope_local_base_freq")]
    pub rope_local_base_freq: f64,
    #[serde(default = "sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    pub rope_scaling: Option<Gemma3RopeScalingConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub enum Gemma3Config {
    #[serde(untagged)]
    WithVision {
        text_config: Gemma3TextConfig,
        vision_config: SiglipVisionConfig,
        image_token_index: usize,
        mm_tokens_per_image: usize,
    },
    #[serde(untagged)]
    Text(Gemma3TextConfig),
}
