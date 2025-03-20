use either::Either;
use serde::Deserialize;

use crate::{layers::Activation, models::mistral, serde_default_fn};

use super::vision;

serde_default_fn!(bool, d_flash_attn, false);

#[derive(Deserialize, Debug, Clone)]
pub struct Mistral3Config {
    pub image_token_index: usize,
    pub multimodal_projector_bias: bool,
    pub projector_hidden_act: Activation,
    pub spatial_merge_size: usize,
    pub vision_feature_layer: Either<isize, Vec<isize>>,
    #[serde(default = "d_flash_attn")]
    pub use_flash_attn: bool,
    pub text_config: mistral::Config,
    pub vision_config: vision::Mistral3VisionConfig,
}
