use crate::{
    models::qwen2,
    vision_models::common::{siglip, whisper},
};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MiniCpmOConfig {
    #[serde(flatten)]
    pub text_config: qwen2::Config,
    pub vision_config: siglip::SiglipVisionConfig,
    pub audio_config: whisper::WhisperEncoderConfig,
    pub vision_batch_size: usize,
    pub query_num: usize,
    pub audio_pool_step: usize,
}
