use serde::{Deserialize, Serialize};

use crate::{layers::Activation, serde_default_fn};

serde_default_fn!(usize, default_image_token_id, 396);
serde_default_fn!(usize, default_projector_hidden_size, 2560);
serde_default_fn!(usize, default_downsample_factor, 2);
serde_default_fn!(bool, default_true, true);
serde_default_fn!(bool, default_projector_bias, true);
serde_default_fn!(bool, default_projector_use_layernorm, true);
serde_default_fn!(Activation, default_projector_hidden_act, Activation::Gelu);
serde_default_fn!(usize, default_hidden_size, 768);
serde_default_fn!(usize, default_intermediate_size, 3072);
serde_default_fn!(usize, default_num_hidden_layers, 12);
serde_default_fn!(usize, default_num_attention_heads, 12);
serde_default_fn!(usize, default_num_channels, 3);
serde_default_fn!(usize, default_num_patches, 256);
serde_default_fn!(usize, default_patch_size, 16);
serde_default_fn!(
    Activation,
    default_vision_hidden_act,
    Activation::GeluPytorchTanh
);
serde_default_fn!(f64, default_layer_norm_eps, 1e-6);
serde_default_fn!(usize, default_min_tiles, 2);
serde_default_fn!(usize, default_max_tiles, 10);
serde_default_fn!(usize, default_min_image_tokens, 64);
serde_default_fn!(usize, default_max_image_tokens, 256);
serde_default_fn!(usize, default_encoder_patch_size, 16);
serde_default_fn!(usize, default_tile_size, 512);
serde_default_fn!(f64, default_max_pixels_tolerance, 2.0);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VisionConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_num_patches")]
    pub num_patches: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub text_config: crate::models::lfm2::Config,
    pub vision_config: VisionConfig,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: usize,
    #[serde(default = "default_projector_hidden_act")]
    pub projector_hidden_act: Activation,
    #[serde(default = "default_projector_hidden_size")]
    pub projector_hidden_size: usize,
    #[serde(default = "default_projector_bias")]
    pub projector_bias: bool,
    #[serde(default = "default_projector_use_layernorm")]
    pub projector_use_layernorm: bool,
    #[serde(default = "default_downsample_factor")]
    pub downsample_factor: usize,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_true")]
    pub do_image_splitting: bool,
    #[serde(default = "default_min_tiles")]
    pub min_tiles: usize,
    #[serde(default = "default_max_tiles")]
    pub max_tiles: usize,
    #[serde(default = "default_true")]
    pub use_thumbnail: bool,
    #[serde(default = "default_min_image_tokens")]
    pub min_image_tokens: usize,
    #[serde(default = "default_max_image_tokens")]
    pub max_image_tokens: usize,
    #[serde(default = "default_encoder_patch_size")]
    pub encoder_patch_size: usize,
    #[serde(default = "default_tile_size")]
    pub tile_size: usize,
    #[serde(default = "default_max_pixels_tolerance")]
    pub max_pixels_tolerance: f64,
}
