use candle_core::Result;
use candle_nn::{Embedding, VarBuilder};
use config::Config;
use text::Qwen2VLTextModel;

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
            vb.pp("model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self { model })
    }
}
