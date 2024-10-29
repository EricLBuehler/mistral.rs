use candle_core::Result;
use candle_nn::{Embedding, VarBuilder};
use config::Config;
use text::Qwen2VLTextModel;
use vision::Qwen2VLVisionModel;

use crate::{
    layers::{RmsNorm, RotaryEmbedding},
    paged_attention::AttentionImplementation,
    pipeline::NormalLoadingMetadata,
};

mod config;
mod text;
mod vision;

pub struct Qwen2VLModel {
    model: Qwen2VLTextModel,
    vision: Qwen2VLVisionModel,
}

impl Qwen2VLModel {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let model = Qwen2VLTextModel::new(
            cfg,
            vb.clone(),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let vision = Qwen2VLVisionModel::new(&cfg.vision_config, vb.pp("vision"))?;
        Ok(Self { model, vision })
    }
}
