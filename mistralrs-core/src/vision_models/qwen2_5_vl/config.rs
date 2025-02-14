// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py

use mistralrs_quant::QuantizedConfig;

use crate::layers::Activation;

use crate::serde_default_fn;

serde_default_fn!(Activation, default_vision_hidden_act, Activation::QuickGelu);
serde_default_fn!(usize, default_in_channels, 3);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub out_hidden_size: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,
    pub intermediate_size: usize,
    pub num_heads: usize,
    #[serde(default = "default_in_channels")]
    pub in_chans: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub window_size: usize,
    pub fullatt_block_indexes: Vec<usize>,
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
    pub vision_config: VisionConfig,
    pub rope_scaling: MRopeScaling,
    pub quantization_config: Option<QuantizedConfig>,
    pub image_token_id: u32,
    pub video_token_id: u32,
    // pub vision_start_token_id: usize,
    // pub max_window_layers: usize,
}
