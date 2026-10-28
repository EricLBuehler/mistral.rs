use serde::{Deserialize, Serialize};

use crate::{layers::Activation, serde_default_fn};

serde_default_fn!(f64, default_projector_layer_norm_eps, 1e-6);
serde_default_fn!(bool, default_true, true);

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub enum LayerType {
    #[serde(rename = "conv")]
    Conv,
    #[serde(rename = "full_attention")]
    Attention,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub conv_kernel_size: usize,
    pub norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub layer_types: Vec<LayerType>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub text_config: crate::models::lfm3::Config,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub projector_hidden_act: Activation,
    pub projector_hidden_size: usize,
    pub projector_bias: bool,
    #[serde(default = "default_projector_layer_norm_eps")]
    pub projector_layer_norm_eps: f64,
    pub downsample_factor: usize,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
}
