use crate::{models::qwen2, vision_models::siglip};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MiniCpmOConfig {
    #[serde(flatten)]
    pub text_config: qwen2::Config,
    pub vision_config: siglip::SiglipVisionConfig,
}
