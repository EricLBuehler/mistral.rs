use serde::Deserialize;

use crate::serde_default_fn;
use crate::vision_models::gemma4::config::{Gemma4TextConfig, Gemma4VisionConfig};

pub const DEFAULT_CANVAS_LENGTH: usize = 256;

serde_default_fn!(usize, canvas_length, DEFAULT_CANVAS_LENGTH);
serde_default_fn!(usize, image_token_id, 258880);
serde_default_fn!(usize, boi_token_id, 255999);
serde_default_fn!(usize, eoi_token_id, 258882);

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DiffusionGemmaConfig {
    pub text_config: Gemma4TextConfig,
    pub vision_config: Option<Gemma4VisionConfig>,
    pub canvas_length: usize,
    pub image_token_id: usize,
    pub boi_token_id: usize,
    pub eoi_token_id: usize,
    pub vision_soft_tokens_per_image: Option<usize>,
}

#[derive(Deserialize)]
struct DiffusionGemmaRawConfig {
    text_config: Gemma4TextConfig,
    vision_config: Option<Gemma4VisionConfig>,
    #[serde(default = "canvas_length")]
    canvas_length: usize,
    #[serde(default = "image_token_id")]
    image_token_id: usize,
    #[serde(default = "boi_token_id")]
    boi_token_id: usize,
    #[serde(default = "eoi_token_id")]
    eoi_token_id: usize,
    vision_soft_tokens_per_image: Option<usize>,
}

impl<'de> Deserialize<'de> for DiffusionGemmaConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = DiffusionGemmaRawConfig::deserialize(deserializer)?;
        let mut text_config = raw.text_config;
        // The HF DiffusionGemma text config removed these fields: the model always
        // uses MoE blocks and k==v projections on full-attention layers.
        text_config.enable_moe_block = text_config.num_experts.is_some();
        text_config.attention_k_eq_v = true;

        Ok(Self {
            text_config,
            vision_config: raw.vision_config,
            canvas_length: raw.canvas_length,
            image_token_id: raw.image_token_id,
            boi_token_id: raw.boi_token_id,
            eoi_token_id: raw.eoi_token_id,
            vision_soft_tokens_per_image: raw.vision_soft_tokens_per_image,
        })
    }
}
