use crate::attention::AttentionMask;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Result, Tensor, D};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    paged_attention::encoder_cache::EncoderCacheManager,
    utils::unvarbuilder::UnVarBuilder,
    vision_models::{
        multimodal_layout::{
            MultimodalEncoderKey, MultimodalEncoderOutputs, PackedMultimodalLayout,
        },
        phi4::image_embedding::IMAGE_SPECIAL_TOKEN_ID,
    },
};

use super::{audio_embedding::AudioEmbedding, image_embedding::ImageEmbedding, Phi4MMConfig};
use crate::paged_attention::block_hash::MultimodalKind;

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
    wte: Arc<dyn QuantMethod>,
    dtype: DType,
}

pub(super) struct Phi4MMPackedInputs<'a> {
    pub image_embeds: Option<&'a Tensor>,
    pub image_attention_mask: Option<&'a Tensor>,
    pub image_sizes: Option<&'a [(u32, u32)]>,
    pub image_hashes: &'a [u64],
    pub audio_embeds: Option<&'a Tensor>,
    pub audio_feature_lens: Option<&'a [usize]>,
    pub audio_embed_sizes: Option<&'a [usize]>,
    pub audio_hashes: &'a [u64],
    pub layout: &'a PackedMultimodalLayout,
}

impl Phi4MMImageAudioEmbedding {
    pub fn new(
        cfg: &Phi4MMConfig,
        wte: Arc<dyn QuantMethod>,
        dtype: DType,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let image_embed = if let Some(img_embd_config) = &cfg.embd_layer.image_embd_layer {
            Some(ImageEmbedding::new(
                cfg,
                img_embd_config,
                wte.clone(),
                dtype,
                vb.pp("image_embed"),
            )?)
        } else {
            None
        };
        let audio_embed = if let Some(audio_embd_config) = &cfg.embd_layer.audio_embd_layer {
            Some(AudioEmbedding::new(
                cfg,
                wte.clone(),
                dtype,
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
            dtype,
        })
    }

    pub fn forward_packed(
        &self,
        input_ids: &Tensor,
        inputs: Phi4MMPackedInputs<'_>,
        encoder_cache: &Mutex<EncoderCacheManager>,
    ) -> Result<Tensor> {
        let input_ids = input_ids.reshape(((), input_ids.dim(D::Minus1)?))?;
        let text_embeddings = self.wte.embedding_forward(&input_ids, self.dtype)?;
        let mut encoder_outputs: MultimodalEncoderOutputs = HashMap::new();

        match (
            &self.image_embed,
            inputs.image_embeds,
            inputs.image_attention_mask,
            inputs.image_sizes,
        ) {
            (Some(image_embed), Some(image_embeds), Some(attention_mask), Some(image_sizes)) => {
                let outputs = image_embed.encode(
                    image_embeds,
                    attention_mask,
                    image_sizes,
                    inputs.image_hashes,
                    encoder_cache,
                )?;
                if outputs.len() != inputs.image_hashes.len() {
                    candle_core::bail!("Phi4MM packed image encoder output count mismatch");
                }
                for (&hash, output) in inputs.image_hashes.iter().zip(outputs) {
                    encoder_outputs.insert(
                        MultimodalEncoderKey {
                            kind: MultimodalKind::Image,
                            hash,
                        },
                        vec![output],
                    );
                }
            }
            (_, None, None, None) if inputs.image_hashes.is_empty() => {}
            (None, Some(_), _, _) => {
                candle_core::bail!("Phi4MM model has no image encoder")
            }
            _ => candle_core::bail!("Phi4MM packed image inputs are incomplete"),
        }

        match (
            &self.audio_embed,
            inputs.audio_embeds,
            inputs.audio_feature_lens,
            inputs.audio_embed_sizes,
        ) {
            (Some(audio_embed), Some(audio_embeds), Some(feature_lens), Some(embed_sizes)) => {
                let outputs = audio_embed.encode(
                    audio_embeds,
                    feature_lens,
                    embed_sizes,
                    inputs.audio_hashes,
                    encoder_cache,
                )?;
                if outputs.len() != inputs.audio_hashes.len() {
                    candle_core::bail!("Phi4MM packed audio encoder output count mismatch");
                }
                for (&hash, output) in inputs.audio_hashes.iter().zip(outputs) {
                    encoder_outputs.insert(
                        MultimodalEncoderKey {
                            kind: MultimodalKind::Audio,
                            hash,
                        },
                        output,
                    );
                }
            }
            (_, None, None, None) if inputs.audio_hashes.is_empty() => {}
            (None, Some(_), _, _) => {
                candle_core::bail!("Phi4MM model has no audio encoder")
            }
            _ => candle_core::bail!("Phi4MM packed audio inputs are incomplete"),
        }

        inputs
            .layout
            .splice_embeddings(&text_embeddings, &encoder_outputs)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_image_embeds: Option<&Tensor>,
        image_attention_mask: &AttentionMask,
        image_sizes: Option<Vec<(u32, u32)>>,
        input_audio_embeds: Option<&Tensor>,
        audio_embed_sizes: Option<Vec<usize>>,
        audio_vision_modes: Option<&[bool]>,
        audio_attention_mask: &AttentionMask,
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
                audio_vision_modes,
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

            (None, None) => self.wte.embedding_forward(&input_ids, self.dtype),
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
