use std::sync::Mutex;

use candle_core::{Result, Tensor, D};
use candle_nn::Module;
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    paged_attention::encoder_cache::EncoderCacheManager, utils::unvarbuilder::UnVarBuilder,
    vision_models::phi4::image_embedding::IMAGE_SPECIAL_TOKEN_ID,
};

use super::{audio_embedding::AudioEmbedding, image_embedding::ImageEmbedding, Phi4MMConfig};

const MAX_INPUT_ID: f64 = 1e9;

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum InputMode {
    /// If only speech
    Speech,
    /// If vision + speech or only vision (not sure why that is necessary though)
    Vision,
}

pub struct Phi4MMImageAudioEmbedding {
    audio_embed: Option<AudioEmbedding>,
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
        let audio_embed = if let Some(audio_embd_config) = &cfg.embd_layer.audio_embd_layer {
            Some(AudioEmbedding::new(
                cfg,
                wte.clone(),
                audio_embd_config,
                vb.pp("audio_embed"),
            )?)
        } else {
            None
        };

        Ok(Self {
            image_embed,
            audio_embed,
            image_input_id: cfg.image_input_id.unwrap_or(-1.),
            wte,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_image_embeds: Option<&Tensor>,
        image_attention_mask: Option<&Tensor>,
        image_sizes: Option<Vec<(u32, u32)>>,
        input_audio_embeds: Option<&Tensor>,
        audio_embed_sizes: Option<Vec<usize>>,
        audio_attention_mask: Option<&Tensor>,
        input_mode: InputMode,
        image_hashes: &[u64],
        encoder_cache: &Mutex<EncoderCacheManager>,
    ) -> Result<Tensor> {
        assert!(-MAX_INPUT_ID < self.image_input_id);

        let input_ids = input_ids.reshape(((), input_ids.dim(D::Minus1)?))?;

        let image_hidden_states = match &self.image_embed {
            Some(image_embed) if input_image_embeds.is_some() => Some(image_embed.forward(
                &input_ids,
                input_image_embeds.expect("input_image_embeds"),
                image_attention_mask,
                image_sizes,
                image_hashes,
                encoder_cache,
            )?),
            _ => None,
        };

        let audio_hidden_states = match &self.audio_embed {
            Some(audio_embed) if input_audio_embeds.is_some() => Some(audio_embed.forward(
                &input_ids,
                input_audio_embeds.expect("input_audio_embeds"),
                audio_embed_sizes.expect("audio_embed_sizes"),
                audio_attention_mask,
                &input_mode,
            )?),
            _ => None,
        };

        let image_position_mask = input_ids.eq(IMAGE_SPECIAL_TOKEN_ID)?;
        let non_image_position_mask = input_ids.ne(IMAGE_SPECIAL_TOKEN_ID)?;

        match (image_hidden_states, audio_hidden_states) {
            (Some(image_hidden_states), Some(audio_hidden_states)) => {
                // Merge
                image_hidden_states.broadcast_mul(
                    &image_position_mask
                        .to_dtype(image_hidden_states.dtype())?
                        .unsqueeze(D::Minus1)?,
                )? + audio_hidden_states.broadcast_mul(
                    &non_image_position_mask
                        .to_dtype(audio_hidden_states.dtype())?
                        .unsqueeze(D::Minus1)?,
                )?
            }
            (Some(image_hidden_states), None) => Ok(image_hidden_states),
            (None, Some(audio_hidden_states)) => Ok(audio_hidden_states),

            (None, None) => self.wte.forward(&input_ids),
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
