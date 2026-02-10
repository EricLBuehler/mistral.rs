use mistralrs_quant::QuantizedConfig;

use crate::layers::Activation;

use crate::serde_default_fn;

serde_default_fn!(
    Activation,
    default_vision_hidden_act,
    Activation::GeluPytorchTanh
);
serde_default_fn!(usize, default_in_channels, 3);
serde_default_fn!(usize, default_depth, 27);
serde_default_fn!(usize, default_hidden_size, 1152);
serde_default_fn!(usize, default_out_hidden_size, 3584);
serde_default_fn!(usize, default_intermediate_size, 4304);
serde_default_fn!(usize, default_num_heads, 16);
serde_default_fn!(usize, default_patch_size, 16);
serde_default_fn!(usize, default_spatial_merge_size, 2);
serde_default_fn!(usize, default_temporal_patch_size, 2);
serde_default_fn!(usize, default_num_position_embeddings, 2304);
serde_default_fn!(
    Vec<usize>,
    default_deepstack_visual_indexes,
    vec![8, 16, 24]
);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_in_channels")]
    pub in_chans: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_num_position_embeddings")]
    pub num_position_embeddings: usize,
    #[serde(default = "default_deepstack_visual_indexes")]
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MRopeScaling {
    pub mrope_section: Vec<usize>,
}

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
    #[serde(default)]
    #[allow(dead_code)]
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
