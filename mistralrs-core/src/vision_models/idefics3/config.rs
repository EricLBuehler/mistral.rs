use serde::Deserialize;

use crate::{layers::Activation, models};

#[derive(Debug, Clone, Deserialize)]
pub struct Idefics3VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_act: Activation,
    pub layer_norm_eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Idefics3Config {
    pub image_token_id: usize,
    pub vision_config: Idefics3VisionConfig,
    pub text_config: models::llama::Config,
    pub scale_factor: usize,
}
