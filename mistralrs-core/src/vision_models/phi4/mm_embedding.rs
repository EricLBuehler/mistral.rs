use candle_core::{Result, Tensor, D};
use candle_nn::Module;
use mistralrs_quant::ShardedVarBuilder;

use crate::utils::unvarbuilder::UnVarBuilder;

use super::{image_embedding::ImageEmbedding, Phi4MMConfig};

const MAX_INPUT_ID: f64 = 1e9;

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
        image_sizes: Option<Vec<(u32, u32)>>,
    ) -> Result<Tensor> {
        assert!(-MAX_INPUT_ID < self.image_input_id);

        let input_ids = input_ids.reshape(((), input_ids.dim(D::Minus1)?))?;

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

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        if let Some(image_embed) = &self.image_embed {
            uvb.pp("image_embed").extend(image_embed.residual_tensors());
        }

        uvb.to_safetensors()
    }
}
