use serde::Deserialize;

use crate::layers::{Activation, Llama3RopeConfig};
use crate::serde_default_fn;

use crate::models::llama::Config as LLaMAConfig;
use crate::models::mistral::Config as MistralConfig;
use crate::vision_models::clip::{Activation as ClipActivation, ClipConfig};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub image_grid_pinpoints: Option<Vec<(u32, u32)>>,
    pub projector_hidden_act: String,
    pub text_config: LLaVATextConfig,
    pub vision_config: LLaVAVisionConfig,
    pub vision_feature_layer: isize,
    pub vision_feature_select_strategy: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct LLaVATextConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    pub sliding_window: Option<usize>,
    pub rope_scaling: Option<Llama3RopeConfig>,
}

serde_default_fn!(usize, default_num_hidden_layers, 32);
serde_default_fn!(usize, default_hidden_size, 4096);
serde_default_fn!(usize, default_intermediate_size, 11008);
serde_default_fn!(usize, default_max_length, 4096);
serde_default_fn!(usize, default_num_attention_heads, 32);
serde_default_fn!(usize, default_num_key_value_heads, 32);
serde_default_fn!(f32, default_rope_theta, 10000.0);
serde_default_fn!(usize, default_vocab_size, 32064);

#[derive(Deserialize, Debug, Clone)]
pub struct LLaVAVisionConfig {
    pub hidden_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
}

impl Config {
    pub fn to_llama_config(&self) -> LLaMAConfig {
        LLaMAConfig {
            hidden_size: self.text_config.hidden_size,
            intermediate_size: self.text_config.intermediate_size,
            vocab_size: self.text_config.vocab_size,
            num_hidden_layers: self.text_config.num_hidden_layers,
            num_attention_heads: self.text_config.num_attention_heads,
            num_key_value_heads: self.text_config.num_key_value_heads,
            rms_norm_eps: self.text_config.rms_norm_eps,
            rope_theta: self.text_config.rope_theta,
            max_position_embeddings: self.text_config.max_position_embeddings,
            rope_scaling: self.text_config.rope_scaling.clone(),
            quantization_config: None,
            tie_word_embeddings: false,
            hidden_act: Activation::Silu,
        }
    }

    pub fn to_mistral_config(&self) -> MistralConfig {
        MistralConfig {
            vocab_size: self.text_config.vocab_size,
            hidden_size: self.text_config.hidden_size,
            intermediate_size: self.text_config.intermediate_size,
            num_hidden_layers: self.text_config.num_hidden_layers,
            num_attention_heads: self.text_config.num_attention_heads,
            num_key_value_heads: self.text_config.num_key_value_heads,
            hidden_act: Activation::Silu, // as it is in mistralai/Mistral-7B-Instruct-v0.2
            max_position_embeddings: self.text_config.max_position_embeddings,
            rms_norm_eps: self.text_config.rms_norm_eps,
            rope_theta: self.text_config.rope_theta as f64,
            rope_parameters: None,
            sliding_window: self.text_config.sliding_window,
            head_dim: None,
            quantization_config: None,
            tie_word_embeddings: false,
        }
    }

    pub fn to_clip_config(&self) -> ClipConfig {
        ClipConfig {
            hidden_size: self.vision_config.hidden_size,
            intermediate_size: self.vision_config.intermediate_size,
            num_hidden_layers: self.vision_config.num_hidden_layers,
            num_attention_heads: self.vision_config.num_attention_heads,
            num_channels: 3,
            image_size: self.vision_config.image_size,
            patch_size: self.vision_config.patch_size,
            hidden_act: ClipActivation::QuickGelu,
        }
    }
}
