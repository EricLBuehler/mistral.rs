use mistralrs_quant::QuantizedConfig;

use crate::layers::Activation;
use crate::serde_default_fn;

// Re-export vision config and MRopeScaling from qwen3_vl
pub use crate::vision_models::qwen3_vl::config::{MRopeScaling, VisionConfig};

serde_default_fn!(Vec<usize>, default_mlp_only_layers, Vec::new());
serde_default_fn!(usize, default_decoder_sparse_step, 1);
serde_default_fn!(bool, default_norm_topk_prob, true);
serde_default_fn!(bool, default_use_sliding_window, false);
serde_default_fn!(usize, default_max_window_layers, 0);

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub rope_scaling: MRopeScaling,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
    // MoE specific fields
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_mlp_only_layers")]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_use_sliding_window")]
    pub use_sliding_window: bool,
    #[serde(default = "default_max_window_layers")]
    pub max_window_layers: usize,
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
