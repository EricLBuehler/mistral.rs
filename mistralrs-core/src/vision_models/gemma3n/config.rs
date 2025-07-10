use either::Either;
use mistralrs_quant::QuantizedConfig;

use crate::{
    layers::{Activation, Gemma3RopeScalingConfig},
    serde_default_fn,
};

serde_default_fn!(bool, attention_bias, false);
serde_default_fn!(usize, head_dim, 256);
serde_default_fn!(Activation, hidden_activation, Activation::GeluPytorchTanh);
serde_default_fn!(f64, rms_norm_eps, 1e-6);
serde_default_fn!(f64, rope_theta, 1000000.);
serde_default_fn!(usize, vocab_size, 262208);
serde_default_fn!(bool, tie_word_embeddings, true);
serde_default_fn!(usize, max_position_embeddings, 131072);
serde_default_fn!(f64, rope_local_base_freq, 10000.);
serde_default_fn!(usize, sliding_window_pattern, 6);
serde_default_fn!(usize, num_attention_heads, 8);
serde_default_fn!(usize, num_key_value_heads, 4);

/// Left is normal, Right is (per layer, orig)
#[derive(Debug, Clone, serde::Deserialize)]
pub struct IntermediateSize(
    #[serde(with = "either::serde_untagged")] pub Either<usize, (Vec<usize>, usize)>,
);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3nTextConfig {
    #[serde(default = "attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "head_dim")]
    pub head_dim: usize,
    #[serde(default = "hidden_activation")]
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub intermediate_size: IntermediateSize,
    #[serde(default = "num_attention_heads")]
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_kv_shared_layers: usize,
    #[serde(default = "num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "vocab_size")]
    pub vocab_size: usize,
    pub sliding_window: usize,
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
    pub vocab_size_per_layer_input: usize,
    pub hidden_size_per_layer_input: usize,
    pub altup_num_inputs: usize,
    pub layer_types: Vec<String>,
    pub altup_active_idx: usize,
    pub altup_coef_clip: Option<f64>,
    pub laurel_rank: usize,
    pub altup_correct_scale: bool,
    pub activation_sparsity_pattern: Vec<f64>,
    pub final_logit_softcapping: Option<f64>,
}

serde_default_fn!(usize, vision_hidden_size, 2048);
serde_default_fn!(i64, vision_vocab_offset, 262144);

serde_default_fn!(usize, vision_vocab_size, 262400);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3nVisionConfig {
    #[serde(default = "vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "vision_vocab_offset")]
    pub vocab_offset: i64,
    #[serde(default = "vision_vocab_size")]
    pub vocab_size: usize,
}

// Audio config defaults
serde_default_fn!(usize, audio_input_feat_size, 80);
serde_default_fn!(usize, audio_hidden_size, 1536);
serde_default_fn!(usize, conf_attention_chunk_size, 12);
serde_default_fn!(usize, conf_attention_context_left, 13);
serde_default_fn!(usize, conf_attention_context_right, 0);
serde_default_fn!(f64, conf_attention_invalid_logits_value, -1e9);
serde_default_fn!(f64, conf_attention_logit_cap, 50.0);
serde_default_fn!(usize, conf_num_attention_heads, 8);
serde_default_fn!(usize, conf_num_hidden_layers, 12);
serde_default_fn!(usize, conf_conv_kernel_size, 5);
serde_default_fn!(usize, conf_reduction_factor, 4);
serde_default_fn!(f64, conf_residual_weight, 0.5);
serde_default_fn!(Vec<usize>, sscp_conv_channel_size, vec![128, 32]);
serde_default_fn!(
    Vec<Vec<usize>>,
    sscp_conv_kernel_size,
    vec![vec![3, 3], vec![3, 3]]
);
serde_default_fn!(
    Vec<Vec<usize>>,
    sscp_conv_stride_size,
    vec![vec![2, 2], vec![2, 2]]
);
serde_default_fn!(usize, audio_vocab_size, 128);
serde_default_fn!(f64, sscp_conv_eps, 1e-3);
serde_default_fn!(f64, audio_rms_norm_eps, 1e-6);
serde_default_fn!(i64, audio_vocab_offset, 262272); // text vocab size (262144) + vision vocab size (128)

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3nAudioConfig {
    #[serde(default = "audio_input_feat_size")]
    pub input_feat_size: usize,
    #[serde(default = "audio_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "conf_attention_chunk_size")]
    pub conf_attention_chunk_size: usize,
    #[serde(default = "conf_attention_context_left")]
    pub conf_attention_context_left: usize,
    #[serde(default = "conf_attention_context_right")]
    pub conf_attention_context_right: usize,
    #[serde(default = "conf_attention_invalid_logits_value")]
    pub conf_attention_invalid_logits_value: f64,
    #[serde(default = "conf_attention_logit_cap")]
    pub conf_attention_logit_cap: f64,
    #[serde(default = "conf_num_attention_heads")]
    pub conf_num_attention_heads: usize,
    #[serde(default = "conf_num_hidden_layers")]
    pub conf_num_hidden_layers: usize,
    #[serde(default = "conf_conv_kernel_size")]
    pub conf_conv_kernel_size: usize,
    #[serde(default = "conf_reduction_factor")]
    pub conf_reduction_factor: usize,
    #[serde(default = "conf_residual_weight")]
    pub conf_residual_weight: f64,
    #[serde(default = "sscp_conv_channel_size")]
    pub sscp_conv_channel_size: Vec<usize>,
    #[serde(default = "sscp_conv_kernel_size")]
    pub sscp_conv_kernel_size: Vec<Vec<usize>>,
    #[serde(default = "sscp_conv_stride_size")]
    pub sscp_conv_stride_size: Vec<Vec<usize>>,
    #[serde(default = "audio_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "sscp_conv_eps")]
    pub sscp_conv_eps: f64,
    #[serde(default = "audio_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "audio_vocab_offset")]
    pub vocab_offset: i64,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3nConfig {
    pub text_config: Gemma3nTextConfig,
    pub vision_config: Gemma3nVisionConfig,
    pub audio_config: Gemma3nAudioConfig,
    pub audio_soft_tokens_per_image: usize,
}
