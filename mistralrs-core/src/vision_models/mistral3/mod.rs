use crate::{models, paged_attention::AttentionImplementation, pipeline::NormalLoadingMetadata};
use candle_core::Result;
use config::Mistral3Config;
use mistralrs_quant::ShardedVarBuilder;
use models::mistral::Model as Mistral;
use vision::VisionModel;

mod config;
mod vision;

pub struct Mistral3Model {
    text_model: Mistral,
    vision_model: VisionModel,
}

impl Mistral3Model {
    pub fn new(
        cfg: &Mistral3Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vision_model = VisionModel::new(&cfg.vision_config, vb.pp("vision_tower"))?;
        let text_model = Mistral::new(
            &cfg.text_config,
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        todo!()
    }
}
