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
}
