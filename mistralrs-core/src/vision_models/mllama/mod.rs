mod config;
mod vision;

use config::{
    MLlamaConfig, MLlamaRopeType, MLlamaTextConfig, MLlamaVisionConfig, VisionActivation,
};
use vision::MLlamaVisionModel;

use candle_core::Result;
use candle_nn::{linear, Linear, VarBuilder};

struct MLlamaModel {
    vision_model: MLlamaVisionModel,
    multi_modal_projector: Linear,
}

impl MLlamaModel {
    fn new(cfg: &MLlamaConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            vision_model: MLlamaVisionModel::new(&cfg.vision_config, vb.pp("vision_model"))?,
            multi_modal_projector: linear(
                cfg.vision_config.vision_output_dim,
                cfg.text_config.hidden_size,
                vb.pp("multi_model_projector"),
            )?,
        })
    }
}
