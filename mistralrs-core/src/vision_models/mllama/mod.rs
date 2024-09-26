mod config;
mod text;
mod vision;

pub(crate) use config::{MLlamaConfig, MLlamaRopeScaling, MLlamaRopeType, MLlamaTextConfig};
use config::{MLlamaVisionConfig, VisionActivation};
use text::MLlamaTextModel;
use vision::MLlamaVisionModel;

use candle_core::Result;
use candle_nn::{linear, Linear, VarBuilder};

struct MLlamaModel {
    vision_model: MLlamaVisionModel,
    language_model: MLlamaTextModel,
    multi_modal_projector: Linear,
}

impl MLlamaModel {
    fn new(cfg: &MLlamaConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            vision_model: MLlamaVisionModel::new(&cfg.vision_config, vb.pp("vision_model"))?,
            language_model: MLlamaTextModel::new(&cfg.text_config, vb.pp("language_model"))?,
            multi_modal_projector: linear(
                cfg.vision_config.vision_output_dim,
                cfg.text_config.hidden_size,
                vb.pp("multi_model_projector"),
            )?,
        })
    }
}
