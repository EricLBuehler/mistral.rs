use candle_core::{Result, Tensor};
use candle_nn::Module;
use mistralrs_quant::QuantizedConfig;

use crate::serde_default_fn;

#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub(crate) enum VisionActivation {
    QuickGelu,
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Silu,
}

impl Module for VisionActivation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QuickGelu => xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?),
            Self::Gelu => xs.gelu_erf(),
            // https://github.com/huggingface/transformers/blob/12f043eaeaabfef6f6efea411d98e6f6d3c094b7/src/transformers/activations.py#L49-L78
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Silu => xs.silu(),
        }
    }
}

serde_default_fn!(usize, d_attn_heads, 16);

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct MLlamaVisionConfig {
    pub(crate) hidden_size: usize,
    pub(crate) hidden_act: VisionActivation,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_global_layers: usize,
    #[serde(default = "d_attn_heads")]
    pub(crate) num_attention_heads: usize,
    pub(crate) num_channels: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) vision_output_dim: usize,
    pub(crate) image_size: usize,
    pub(crate) patch_size: usize,
    pub(crate) norm_eps: f64,
    pub(crate) max_num_tiles: usize,
    pub(crate) intermediate_layers_indices: Vec<usize>,
    pub(crate) supported_aspect_ratios: Vec<(usize, usize)>,
}

impl MLlamaVisionConfig {
    pub(crate) fn max_aspect_ratio_id(&self) -> usize {
        self.supported_aspect_ratios.len()
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) enum MLlamaRopeType {
    #[serde(rename = "default")]
    Default,
    #[serde(rename = "linear")]
    Linear,
    #[serde(rename = "dynamic")]
    Dynamic,
    #[serde(rename = "yarn")]
    Yarn,
    #[serde(rename = "longrope")]
    Longrope,
    #[serde(rename = "llama3")]
    Llama3,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct MLlamaRopeScaling {
    pub(crate) rope_type: MLlamaRopeType,
    pub(crate) factor: Option<f32>,
    pub(crate) original_max_position_embeddings: usize,
    pub(crate) attention_factor: Option<f32>,
    pub(crate) beta_fast: Option<f32>,
    pub(crate) beta_slow: Option<f32>,
    pub(crate) short_factor: Option<Vec<f64>>,
    pub(crate) long_factor: Option<Vec<f64>>,
    pub(crate) low_freq_factor: Option<f32>,
    pub(crate) high_freq_factor: Option<f32>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MLlamaTextConfig {
    pub(crate) rope_scaling: Option<MLlamaRopeScaling>,
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) hidden_act: candle_nn::Activation,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) rope_theta: f32,
    pub(crate) rms_norm_eps: f64,
    pub(crate) max_position_embeddings: usize,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) cross_attention_layers: Vec<usize>,
    pub(crate) quantization_config: Option<QuantizedConfig>,
}

impl MLlamaTextConfig {
    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct MLlamaConfig {
    pub(crate) vision_config: MLlamaVisionConfig,
    pub(crate) text_config: MLlamaTextConfig,
}
