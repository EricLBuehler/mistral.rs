use mistralrs_quant::QuantizedConfig;
use serde::Deserialize;

use crate::{layers::Activation, models::mistral};

use super::vision;

#[derive(Deserialize, Debug, Clone)]
pub struct Mistral3Config {
    pub image_token_index: usize,
    pub multimodal_projector_bias: bool,
    pub projector_hidden_act: Activation,
    pub spatial_merge_size: usize,
    pub vision_feature_layer: isize,
    pub text_config: mistral::Config,
    pub vision_config: vision::Mistral3VisionConfig,
    /// Top-level quantization config that should be applied to text_config
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl Mistral3Config {
    /// Propagate top-level quantization_config to text_config if text_config doesn't have one
    pub fn propagate_quantization_config(&mut self) {
        if self.text_config.quantization_config.is_none() && self.quantization_config.is_some() {
            self.text_config.quantization_config = self.quantization_config.clone();
        }
    }
}
