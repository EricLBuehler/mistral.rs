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
pub(super) enum MLlamaRopeType {
    Default,
    Linear,
    Dynamic,
    Yarn,
    Longrope,
    Llama3,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct MLlamaRopeScaling {
    pub(super) rope_type: MLlamaRopeType,
    pub(super) factor: Option<f64>,
    pub(super) original_max_position_embeddings: usize,
    pub(super) attention_factor: Option<f64>,
    pub(super) beta_fast: Option<f64>,
    pub(super) beta_slow: Option<f64>,
    pub(super) short_factor: Option<Vec<f64>>,
    pub(super) long_factor: Option<Vec<f64>>,
    pub(super) low_freq_factor: Option<f64>,
    pub(super) high_freq_factor: Option<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(super) struct MLlamaTextConfig {
    pub(super) rope_scaling: Option<MLlamaRopeScaling>,
    pub(super) vocab_size: usize,
    pub(super) hidden_size: usize,
    pub(super) hidden_act: candle_nn::Activation,
    pub(super) num_hidden_layers: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_key_value_heads: usize,
    pub(super) intermediate_size: usize,
    pub(super) rope_theta: f64,
    pub(super) rms_norm_eps: f64,
    pub(super) max_position_embeddings: usize,
    pub(super) tie_word_embeddings: usize,
    pub(super) cross_attention_layers: Vec<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(super) struct MLlamaConfig {
    pub(super) vision_config: MLlamaVisionConfig,
    pub(super) text_config: MLlamaTextConfig,
    pub(super) image_token_index: usize,
}
