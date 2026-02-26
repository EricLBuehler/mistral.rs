use mistralrs_quant::QuantizedConfig;

use crate::layers::Activation;
use crate::serde_default_fn;

// Re-export vision config from qwen3_vl
pub use crate::vision_models::qwen3_vl::config::VisionConfig;

serde_default_fn!(Vec<usize>, default_mlp_only_layers, Vec::new());
serde_default_fn!(usize, default_full_attn_interval, 4);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(f64, default_partial_rotary_factor, 0.25);
serde_default_fn!(bool, default_norm_topk_prob, true);
serde_default_fn!(f64, default_rope_theta, 10_000_000.0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

/// Nested rope_parameters from the config JSON.
/// Contains rope_theta, mrope_section, partial_rotary_factor, etc.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    // MoE fields
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    // Hybrid attention fields
    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: usize,
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    // Other
    #[serde(default = "default_mlp_only_layers")]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl TextConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters.partial_rotary_factor
    }

    pub fn mrope_section(&self) -> &[usize] {
        &self.rope_parameters.mrope_section
    }

    pub fn layer_types(&self) -> Vec<LayerType> {
        (0..self.num_hidden_layers)
            .map(|i| {
                if (i + 1) % self.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect()
    }

    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn linear_conv_dim(&self) -> usize {
        2 * self.linear_key_dim() + self.linear_value_dim()
    }

    pub fn rot_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor()) as usize
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub tie_word_embeddings: bool,
    /// Top-level quantization_config takes precedence
    pub quantization_config: Option<QuantizedConfig>,
}
