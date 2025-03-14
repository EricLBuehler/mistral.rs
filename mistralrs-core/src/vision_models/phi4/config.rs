use std::collections::HashMap;

use mistralrs_quant::{QuantizedConfig, StaticLoraConfig};
use serde::{Deserialize, Serialize};

use crate::{
    layers::{Activation, Phi4MMRopeScalingConfig},
    serde_default_fn,
};

serde_default_fn!(bool, d_flash_attn, false);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMLoraConfig {
    pub layer: String,
    pub lora_alpha: f64,
    pub r: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMImageEmbedConfig {
    pub n_embd: Option<usize>,
    pub crop_size: Option<usize>,
    pub embedding_cls: String,
    pub enable_gradient_checkpointing: bool,
    pub hd_transform_order: Option<String>,
    pub image_token_compression_cls: Option<String>,
    pub projection_cls: Option<String>,
    pub use_hd_transform: Option<bool>,
    pub with_learnable_separator: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMEmbdLayerConfig {
    pub image_embd_layer: Option<Phi4MMImageEmbedConfig>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMImgProcessorConfig {
    pub layer_idx: Option<isize>,
    pub type_feature: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub resid_pdrop: f64,
    pub embd_pdrop: f64,
    pub attention_dropout: f64,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rope_scaling: Option<Phi4MMRopeScalingConfig>,
    pub partial_rotary_factor: f64,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    pub image_input_id: Option<f64>,
    pub sliding_window: Option<usize>,
    pub embd_layer: Phi4MMEmbdLayerConfig,
    pub img_processor: Option<Phi4MMImgProcessorConfig>,
    // pub audio_processor: Option<String>,
    pub vision_lora: StaticLoraConfig,
    pub speech_lora: StaticLoraConfig,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "d_flash_attn")]
    pub use_flash_attn: bool,
}

impl Phi4MMConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn loras(&self) -> HashMap<String, StaticLoraConfig> {
        let mut accum = HashMap::new();
        // Add all the loras
        accum.insert("speech".to_string(), self.speech_lora.clone());
        accum.insert("vision".to_string(), self.vision_lora.clone());
        accum
    }
}
