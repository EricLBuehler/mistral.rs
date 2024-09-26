use candle_core::{Result, Tensor};
use candle_nn::Module;

#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub(super) enum VisionActivation {
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

#[derive(Debug, Clone, serde::Deserialize)]
pub(super) struct MLlamaVisionConfig {
    pub(super) hidden_size: usize,
    pub(super) hidden_act: VisionActivation,
    pub(super) num_hidden_layers: usize,
    pub(super) num_global_layers: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_channels: usize,
    pub(super) intermediate_size: usize,
    pub(super) vision_output_dim: usize,
    pub(super) image_size: usize,
    pub(super) patch_size: usize,
    pub(super) norm_eps: f64,
    pub(super) max_num_tiles: usize,
    pub(super) intermediate_layers_indices: Vec<usize>,
    pub(super) supported_aspect_ratios: Vec<(usize, usize)>,
}

impl MLlamaVisionConfig {
    pub(super) fn max_aspect_ratio_id(&self) -> usize {
        self.supported_aspect_ratios.len()
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) enum MLlamaRopeType {
    Default,
    Linear,
    Dynamic,
    Yarn,
    Longrope,
    Llama3,
}

#[derive(Debug, Clone, serde::Deserialize)]
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
pub(crate) struct MLlamaTextConfig {
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
    pub(crate) tie_word_embeddings: usize,
    pub(crate) cross_attention_layers: Vec<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct MLlamaConfig {
    pub(crate) vision_config: MLlamaVisionConfig,
    pub(crate) text_config: MLlamaTextConfig,
    pub(crate) image_token_index: usize,
}
