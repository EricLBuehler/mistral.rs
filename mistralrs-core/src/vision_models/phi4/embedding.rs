use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::Module;
use mistralrs_quant::ShardedVarBuilder;

use crate::{ops::BitWiseOp, vision_models::phi4::image_embedding::IMAGE_SPECIAL_TOKEN_ID};

use super::{image_embedding::ImageEmbedding, Phi4MMConfig};

const MAX_INPUT_ID: f64 = 1e9;
const COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE: (f64, f64) = (-9999., -1.);

pub struct Phi4MMImageAudioEmbedding {
    image_embed: Option<ImageEmbedding>,
    image_input_id: f64,
    wte: candle_nn::Embedding,
}

impl Phi4MMImageAudioEmbedding {
    pub fn new(
        cfg: &Phi4MMConfig,
        wte: candle_nn::Embedding,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let image_embed = if let Some(img_embd_config) = &cfg.embd_layer.image_embd_layer {
            Some(ImageEmbedding::new(
                cfg,
                img_embd_config,
                wte.clone(),
                vb.pp("image_embed"),
            )?)
        } else {
            None
        };

        Ok(Self {
            image_embed,
            image_input_id: cfg.image_input_id.unwrap_or(-1.),
            wte,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_image_embeds: &Tensor,
        image_attention_mask: Option<&Tensor>,
        image_sizes: Option<Vec<(usize, usize)>>,
    ) -> Result<Tensor> {
        assert!(-MAX_INPUT_ID < self.image_input_id);

        let mut input_ids = input_ids.reshape(((), input_ids.dim(D::Minus1)?))?;

        let image_position_mask = input_ids
            .ge(COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE.0)?
            .bitwise_and(&input_ids.le(COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE.1)?)?;

        // Slice assign for IMAGE_SPECIAL_TOKEN_ID
        {
            // Get the equiv 0th and 1th rows of the positions_tuple
            let positions_transposed = image_position_mask.t()?;
            let positions_transposed_0 = positions_transposed.i(0)?;
            let positions_transposed_1 = positions_transposed.i(1)?;

            let linear_index = ((positions_transposed_0 * input_ids.dim(D::Minus1)? as f64)?
                + positions_transposed_1)?;

            // Zero it out
            input_ids = input_ids.flatten_all()?.scatter_add(
                &linear_index,
                &Tensor::zeros(
                    linear_index.elem_count(),
                    input_ids.dtype(),
                    input_ids.device(),
                )?,
                0,
            )?;

            input_ids = input_ids.flatten_all()?.scatter_add(
                &linear_index,
                &Tensor::full(
                    IMAGE_SPECIAL_TOKEN_ID,
                    linear_index.elem_count(),
                    input_ids.device(),
                )?,
                0,
            )?;
        }

        let image_hidden_states = if let Some(image_embed) = &self.image_embed {
            Some(image_embed.forward(
                &input_ids,
                input_image_embeds,
                image_attention_mask,
                image_sizes,
            )?)
        } else {
            None
        };

        match image_hidden_states {
            Some(image_hidden_states) => Ok(image_hidden_states),

            None => self.wte.forward(&input_ids),
        }
    }
}
