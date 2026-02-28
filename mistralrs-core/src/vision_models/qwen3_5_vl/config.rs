use mistralrs_quant::QuantizedConfig;

use crate::layers::Activation;
use crate::serde_default_fn;

// Re-export vision config from qwen3_vl
pub use crate::vision_models::qwen3_vl::config::VisionConfig;

serde_default_fn!(Vec<usize>, default_mlp_only_layers, Vec::new());
serde_default_fn!(bool, default_norm_topk_prob, true);
serde_default_fn!(f64, default_partial_rotary_factor, 0.25);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MRopeParameters {
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
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
    #[serde(default)]
    pub hidden_act: Option<Activation>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: MRopeParameters,
    // GDN fields
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    // MoE fields
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_mlp_only_layers")]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    // Optional fields
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

impl TextConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta.unwrap_or(10_000_000.0)
    }

    pub fn activation(&self) -> Activation {
        self.hidden_act.unwrap_or(Activation::Silu)
    }

    /// Dense MLP intermediate size (for mlp_only_layers).
    /// Falls back to shared_expert_intermediate_size if not explicitly set.
    pub fn dense_intermediate_size(&self) -> usize {
        self.intermediate_size
            .unwrap_or(self.shared_expert_intermediate_size)
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
    pub architectures: Option<Vec<String>>,
}
