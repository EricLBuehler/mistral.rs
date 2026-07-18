use mistralrs_quant::QuantizedConfig;

use crate::{layers::Activation, serde_default_fn};

serde_default_fn!(bool, default_attention_bias, false);
serde_default_fn!(bool, default_tie_word_embeddings, false);
serde_default_fn!(usize, default_head_dim, 128);
serde_default_fn!(f64, default_rope_theta, 10_000.0);
serde_default_fn!(usize, default_image_start_token_id, 120_118);
serde_default_fn!(usize, default_image_end_token_id, 120_119);
serde_default_fn!(usize, default_image_token_id, 120_120);
serde_default_fn!(usize, default_image_newline_token_id, 120_121);
serde_default_fn!(usize, default_cat_extra_token, 1);

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub alpha: Option<f64>,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub mscale: Option<f64>,
    #[serde(default)]
    pub mscale_all_dim: Option<f64>,
    #[serde(rename = "type", alias = "rope_type")]
    pub rope_type: String,
    #[serde(default)]
    pub xdrope_section: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub interpolate_mode: Option<String>,
    #[serde(default)]
    pub max_image_size: Option<usize>,
    #[serde(default = "default_cat_extra_token")]
    pub cat_extra_token: usize,
    #[serde(default)]
    pub img_max_token_num: Option<usize>,
    #[serde(default)]
    pub max_vit_seq_len: Option<usize>,
}

#[allow(dead_code)]
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
    #[serde(default)]
    pub sliding_window: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_image_start_token_id")]
    pub image_start_token_id: usize,
    #[serde(default = "default_image_end_token_id")]
    pub image_end_token_id: usize,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: usize,
    #[serde(default = "default_image_newline_token_id")]
    pub image_newline_token_id: usize,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
    pub vision_config: VisionConfig,
}
