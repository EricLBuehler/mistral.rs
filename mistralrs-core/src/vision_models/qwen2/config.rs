// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py

use candle_nn::Activation;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub embed_dim: usize,
    pub hidden_size: usize,
    pub hidden_act: Activation,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MRopeScaling {
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub vision_config: VisionConfig,
    pub rope_scaling: MRopeScaling,
}