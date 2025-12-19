use serde::Deserialize;

use crate::{layers::Activation, models::mistral};

use super::vision;

#[derive(Debug, Clone)]
pub struct Mistral3Config {
    pub image_token_index: usize,
    pub multimodal_projector_bias: bool,
    pub projector_hidden_act: Activation,
    pub spatial_merge_size: usize,
    pub vision_feature_layer: isize,
    pub text_config: mistral::Config,
    pub vision_config: vision::Mistral3VisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct RawMistral3Config {
    image_token_index: usize,
    multimodal_projector_bias: bool,
    projector_hidden_act: Activation,
    spatial_merge_size: usize,
    vision_feature_layer: isize,
    text_config: mistral::Config,
    vision_config: vision::Mistral3VisionConfig,
    #[serde(default)]
    quantization_config: Option<mistralrs_quant::QuantizedConfig>,
}

impl<'de> Deserialize<'de> for Mistral3Config {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut raw = RawMistral3Config::deserialize(deserializer)?;
        if raw.text_config.quantization_config.is_none() {
            raw.text_config.quantization_config = raw.quantization_config;
        }
        Ok(Self {
            image_token_index: raw.image_token_index,
            multimodal_projector_bias: raw.multimodal_projector_bias,
            projector_hidden_act: raw.projector_hidden_act,
            spatial_merge_size: raw.spatial_merge_size,
            vision_feature_layer: raw.vision_feature_layer,
            text_config: raw.text_config,
            vision_config: raw.vision_config,
        })
    }
}
