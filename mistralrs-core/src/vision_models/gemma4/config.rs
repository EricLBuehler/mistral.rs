use mistralrs_quant::QuantizedConfig;

use crate::{layers::Activation, serde_default_fn};

// ── Rope parameter structs ──────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub struct Gemma4RopeLayerParams {
    pub rope_theta: Option<f64>,
    pub rope_type: Option<String>,
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub struct Gemma4RopeParameters {
    pub full_attention: Option<Gemma4RopeLayerParams>,
    pub sliding_attention: Option<Gemma4RopeLayerParams>,
    // Flat-dict fields: vision config uses `{"rope_theta": ..., "rope_type": ...}`
    // instead of the nested `full_attention`/`sliding_attention` structure.
    pub rope_theta: Option<f64>,
    pub rope_type: Option<String>,
    pub partial_rotary_factor: Option<f64>,
}

// ── Text config defaults ────────────────────────────────────────────────────

serde_default_fn!(bool, attention_bias, false);
serde_default_fn!(usize, head_dim, 256);
serde_default_fn!(Activation, hidden_activation, Activation::GeluPytorchTanh);
serde_default_fn!(usize, num_attention_heads, 8);
serde_default_fn!(usize, num_key_value_heads, 4);
serde_default_fn!(f64, rms_norm_eps, 1e-6);
serde_default_fn!(f64, rope_theta, 1000000.);
serde_default_fn!(usize, vocab_size, 262144);
serde_default_fn!(usize, query_pre_attn_scalar, 256);
serde_default_fn!(usize, max_position_embeddings, 131072);
serde_default_fn!(bool, tie_word_embeddings, true);
serde_default_fn!(usize, sliding_window_pattern, 6);
serde_default_fn!(usize, global_head_dim, 512);
serde_default_fn!(bool, attention_k_eq_v, false);
serde_default_fn!(bool, enable_moe_block, false);
serde_default_fn!(usize, num_kv_shared_layers, 0);
serde_default_fn!(bool, use_double_wide_mlp, false);

// ── Gemma4TextConfig ────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub struct Gemma4TextConfig {
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
    pub final_logit_softcapping: Option<f64>,
    #[serde(default = "query_pre_attn_scalar")]
    pub query_pre_attn_scalar: usize,
    #[serde(default = "max_position_embeddings")]
    pub max_position_embeddings: usize,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "sliding_window_pattern", alias = "_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    pub layer_types: Vec<String>,
    #[serde(default = "global_head_dim")]
    pub global_head_dim: usize,
    #[serde(default = "attention_k_eq_v")]
    pub attention_k_eq_v: bool,
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default = "enable_moe_block")]
    pub enable_moe_block: bool,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    #[serde(alias = "moe_intermediate_size")]
    pub expert_intermediate_size: Option<usize>,
    #[serde(default = "num_kv_shared_layers")]
    pub num_kv_shared_layers: usize,
    pub hidden_size_per_layer_input: Option<usize>,
    pub vocab_size_per_layer_input: Option<usize>,
    #[serde(default = "use_double_wide_mlp")]
    pub use_double_wide_mlp: bool,
    pub rope_parameters: Option<Gemma4RopeParameters>,
    pub use_bidirectional_attention: Option<String>,
}

impl Gemma4TextConfig {
    /// Effective sliding window size, adjusted for bidirectional attention.
    /// `self.sliding_window = (self.sliding_window // 2) + 1` only when `use_bidirectional_attention == "all"`.
    pub fn effective_sliding_window(&self) -> usize {
        if self.use_bidirectional_attention.as_deref() == Some("all") {
            (self.sliding_window / 2) + 1
        } else {
            self.sliding_window
        }
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.full_attention.as_ref())
            .and_then(|fa| fa.partial_rotary_factor)
            .unwrap_or(0.25)
    }

    pub fn rope_local_base_freq(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.sliding_attention.as_ref())
            .and_then(|sa| sa.rope_theta)
            .unwrap_or(10000.0)
    }
}

// ── Vision config defaults ──────────────────────────────────────────────────

serde_default_fn!(usize, vision_hidden_size, 768);
serde_default_fn!(usize, vision_intermediate_size, 3072);
serde_default_fn!(usize, vision_num_hidden_layers, 16);
serde_default_fn!(usize, vision_num_attention_heads, 12);
serde_default_fn!(usize, vision_num_key_value_heads, 12);
serde_default_fn!(usize, vision_head_dim, 64);
serde_default_fn!(
    Activation,
    vision_hidden_activation,
    Activation::GeluPytorchTanh
);
serde_default_fn!(f64, vision_rms_norm_eps, 1e-6);
serde_default_fn!(usize, vision_patch_size, 16);
serde_default_fn!(usize, vision_position_embedding_size, 10240);
serde_default_fn!(usize, vision_pooling_kernel_size, 3);
serde_default_fn!(usize, vision_default_output_length, 280);
serde_default_fn!(bool, vision_use_clipped_linears, false);
serde_default_fn!(bool, vision_standardize, false);

// ── Gemma4VisionConfig ──────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub struct Gemma4VisionConfig {
    #[serde(default = "vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "vision_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "vision_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "vision_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "vision_head_dim")]
    pub head_dim: usize,
    #[serde(default = "vision_hidden_activation")]
    pub hidden_activation: Activation,
    #[serde(default = "vision_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "vision_position_embedding_size")]
    pub position_embedding_size: usize,
    #[serde(default = "vision_pooling_kernel_size")]
    pub pooling_kernel_size: usize,
    #[serde(default = "vision_default_output_length")]
    pub default_output_length: usize,
    #[serde(default = "vision_use_clipped_linears")]
    pub use_clipped_linears: bool,
    #[serde(default = "vision_standardize")]
    pub standardize: bool,
    pub rope_parameters: Option<Gemma4RopeParameters>,
}

impl Gemma4VisionConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| {
                // Try nested full_attention first, then flat rope_theta
                rp.full_attention
                    .as_ref()
                    .and_then(|fa| fa.rope_theta)
                    .or(rp.rope_theta)
            })
            .unwrap_or(100.0)
    }
}

// ── Audio config defaults ───────────────────────────────────────────────────

serde_default_fn!(usize, audio_input_feat_size, 128);
serde_default_fn!(usize, audio_hidden_size, 1024);
serde_default_fn!(usize, conf_attention_chunk_size, 12);
serde_default_fn!(usize, conf_attention_context_left, 13);
serde_default_fn!(usize, conf_attention_context_right, 0);
serde_default_fn!(f64, conf_attention_invalid_logits_value, -1e9);
serde_default_fn!(f64, conf_attention_logit_cap, 50.0);
serde_default_fn!(usize, conf_num_attention_heads, 8);
serde_default_fn!(usize, conf_num_hidden_layers, 12);
serde_default_fn!(usize, conf_conv_kernel_size, 5);
serde_default_fn!(usize, conf_reduction_factor, 1);
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
serde_default_fn!(f64, sscp_conv_group_norm_eps, 1e-6);
serde_default_fn!(f64, audio_rms_norm_eps, 1e-6);
serde_default_fn!(i64, audio_vocab_offset, 262272);
serde_default_fn!(f64, gradient_clipping, 1e10);
serde_default_fn!(f64, embedding_norm_eps, 1e-6);
serde_default_fn!(f64, sscp_conv_eps, 1e-3);
serde_default_fn!(Option<usize>, output_proj_dims_default, Some(1536));
serde_default_fn!(String, sscp_conv_norm_type, "layer_norm".to_string());
serde_default_fn!(String, sscp_conv_padding_type, "semicausal".to_string());
serde_default_fn!(bool, streaming, false);
serde_default_fn!(bool, use_clipped_linears, true);

// ── Gemma4AudioConfig ───────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub struct Gemma4AudioConfig {
    #[serde(default = "audio_input_feat_size")]
    pub input_feat_size: usize,
    #[serde(default = "audio_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "output_proj_dims_default")]
    pub output_proj_dims: Option<usize>,
    pub conf_hidden_size: Option<usize>,
    pub conf_positional_bias_size: Option<usize>,
    #[serde(default = "conf_attention_chunk_size", alias = "attention_chunk_size")]
    pub conf_attention_chunk_size: usize,
    #[serde(
        default = "conf_attention_context_left",
        alias = "attention_context_left"
    )]
    pub conf_attention_context_left: usize,
    #[serde(
        default = "conf_attention_context_right",
        alias = "attention_context_right"
    )]
    pub conf_attention_context_right: usize,
    #[serde(
        default = "conf_attention_invalid_logits_value",
        alias = "attention_invalid_logits_value"
    )]
    pub conf_attention_invalid_logits_value: f64,
    #[serde(default = "conf_attention_logit_cap", alias = "attention_logit_cap")]
    pub conf_attention_logit_cap: f64,
    #[serde(default = "conf_num_attention_heads", alias = "num_attention_heads")]
    pub conf_num_attention_heads: usize,
    #[serde(default = "conf_num_hidden_layers", alias = "num_hidden_layers")]
    pub conf_num_hidden_layers: usize,
    #[serde(default = "conf_conv_kernel_size", alias = "conv_kernel_size")]
    pub conf_conv_kernel_size: usize,
    #[serde(default = "conf_reduction_factor")]
    pub conf_reduction_factor: usize,
    #[serde(default = "conf_residual_weight", alias = "residual_weight")]
    pub conf_residual_weight: f64,
    #[serde(
        default = "sscp_conv_channel_size",
        alias = "subsampling_conv_channels"
    )]
    pub sscp_conv_channel_size: Vec<usize>,
    #[serde(default = "sscp_conv_kernel_size")]
    pub sscp_conv_kernel_size: Vec<Vec<usize>>,
    #[serde(default = "sscp_conv_stride_size")]
    pub sscp_conv_stride_size: Vec<Vec<usize>>,
    #[serde(default = "audio_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "sscp_conv_group_norm_eps")]
    pub sscp_conv_group_norm_eps: f64,
    #[serde(default = "sscp_conv_eps")]
    pub sscp_conv_eps: f64,
    #[serde(default = "sscp_conv_norm_type")]
    pub sscp_conv_norm_type: String,
    #[serde(default = "sscp_conv_padding_type")]
    pub sscp_conv_padding_type: String,
    pub sscp_conv_time_pad_top: Option<usize>,
    pub sscp_conv_time_pad_bottom: Option<usize>,
    #[serde(default = "streaming")]
    pub streaming: bool,
    #[serde(default = "audio_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "audio_vocab_offset")]
    pub vocab_offset: i64,
    #[serde(default = "gradient_clipping")]
    pub gradient_clipping: f64,
    #[serde(default = "embedding_norm_eps")]
    pub embedding_norm_eps: f64,
    #[serde(default = "use_clipped_linears")]
    pub use_clipped_linears: bool,
}

// ── Top-level config defaults ───────────────────────────────────────────────

serde_default_fn!(usize, image_token_id, 258880);
serde_default_fn!(usize, audio_token_id, 258881);
serde_default_fn!(usize, video_token_id, 258884);
serde_default_fn!(usize, boi_token_id, 255999);
serde_default_fn!(usize, eoi_token_id, 258882);
serde_default_fn!(usize, boa_token_id, 256000);
serde_default_fn!(usize, eoa_token_id, 258883);

// ── Gemma4Config ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub struct Gemma4Config {
    pub text_config: Gemma4TextConfig,
    pub vision_config: Gemma4VisionConfig,
    pub audio_config: Option<Gemma4AudioConfig>,
    #[serde(default = "image_token_id")]
    pub image_token_id: usize,
    #[serde(default = "audio_token_id")]
    pub audio_token_id: usize,
    #[serde(default = "video_token_id")]
    pub video_token_id: usize,
    #[serde(default = "boi_token_id")]
    pub boi_token_id: usize,
    #[serde(default = "eoi_token_id")]
    pub eoi_token_id: usize,
    #[serde(default = "boa_token_id")]
    pub boa_token_id: usize,
    #[serde(default = "eoa_token_id")]
    pub eoa_token_id: usize,
    /// Ignored duplicate of `eoa_token_id` present in some config files.
    #[serde(default, rename = "eoa_token_index")]
    _eoa_token_index: Option<usize>,
    pub audio_ms_per_token: Option<usize>,
    pub vision_soft_tokens_per_image: Option<usize>,
}
