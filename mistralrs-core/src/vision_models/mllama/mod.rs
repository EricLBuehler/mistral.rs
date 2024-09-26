mod config;
mod inputs_processor;
mod text;
mod vision;

pub(crate) use config::{MLlamaConfig, MLlamaRopeScaling, MLlamaRopeType, MLlamaTextConfig};
use config::{MLlamaVisionConfig, VisionActivation};
use text::MLlamaTextModel;
use vision::MLlamaVisionModel;

use candle_core::{Result, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};

struct MLlamaModel {
    vision_model: MLlamaVisionModel,
    language_model: MLlamaTextModel,
    multi_modal_projector: Linear,
    hidden_size: usize,
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
            hidden_size: cfg.text_config.hidden_size,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        aspect_ratio_mask: Option<&Tensor>,
        aspect_ratio_ids: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let cross_attn_states = if let Some(pixel_values) = pixel_values {
            let Some(aspect_ratio_mask) = aspect_ratio_mask else {
                candle_core::bail!("`aspect_ratio_mask` must be specified if `pixel_values` is.");
            };
            let Some(aspect_ratio_ids) = aspect_ratio_ids else {
                candle_core::bail!("`aspect_ratio_ids` must be specified if `pixel_values` is.");
            };
            let vision_outputs =
                self.vision_model
                    .forward(pixel_values, aspect_ratio_ids, aspect_ratio_mask)?;
            let cross_attention_states = self
                .multi_modal_projector
                .forward(&vision_outputs)?
                .reshape(((), vision_outputs.dim(D::Minus2)?, self.hidden_size))?;
            Some(cross_attention_states)
        } else {
            None
        };

        self.language_model.forward(
            input_ids,
            cross_attn_states.as_ref(),
            None,
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
        )
    }
}
