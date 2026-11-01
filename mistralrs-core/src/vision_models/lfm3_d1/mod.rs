pub(crate) mod config;

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    paged_attention::{
        encoder_cache::{CacheModality, EncoderCacheManager},
        AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
    },
    pipeline::{
        EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata,
    },
    utils::unvarbuilder::UnVarBuilder,
    vision_models::{
        grid_processor::GridVisionSpecificArgs,
        lfm3_omni::{
            audio::{AudioEncoder, AudioProjector},
            inputs_processor::Lfm3OmniSpecificArgs,
            memory::AudioMemoryEstimate,
        },
        lfm3_vl::Lfm3VlModel,
    },
};

#[derive(Clone, Debug)]
pub struct DecisionAttention {
    pub sequence_lengths: Vec<usize>,
    pub state_lengths: Option<Vec<usize>>,
    pub question_ids: Option<Vec<Vec<i32>>>,
}

impl DecisionAttention {
    fn validate(&self, batch: usize, length: usize) -> Result<()> {
        if self.sequence_lengths.len() != batch
            || self
                .sequence_lengths
                .iter()
                .any(|&len| len == 0 || len > length)
            || self.state_lengths.as_ref().is_some_and(|states| {
                states.len() != batch
                    || states
                        .iter()
                        .zip(&self.sequence_lengths)
                        .any(|(state, len)| state > len)
            })
            || self.question_ids.as_ref().is_some_and(|groups| {
                self.state_lengths.is_none()
                    || groups.len() != batch
                    || groups.iter().any(|row| row.len() != length)
            })
        {
            candle_core::bail!("Invalid d1 attention layout for {batch} rows of {length} tokens");
        }
        Ok(())
    }

    fn rows(
        &self,
        start: usize,
        width: usize,
        length: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mut values = Vec::with_capacity(self.sequence_lengths.len() * width * length);
        for (row, &real) in self.sequence_lengths.iter().enumerate() {
            let state = self.state_lengths.as_ref().map(|states| states[row]);
            let groups = self.question_ids.as_ref().map(|groups| &groups[row]);
            for query in start..start + width {
                for key in 0..length {
                    let visible = match (state, groups) {
                        (Some(state), Some(groups)) => {
                            query < state || key < state || groups[query] == groups[key]
                        }
                        (Some(state), None) => key <= query || query < state && key < state,
                        (None, _) => key <= query,
                    };
                    values.push(if key < real && visible {
                        0f32
                    } else {
                        f32::NEG_INFINITY
                    });
                }
            }
        }
        Tensor::from_vec(
            values,
            (self.sequence_lengths.len(), 1, width, length),
            device,
        )?
        .to_dtype(dtype)
    }
}

struct AudioInputEncoder {
    encoder: AudioEncoder,
    projector: AudioProjector,
    memory: AudioMemoryEstimate,
}

pub(crate) struct Lfm3D1Model {
    vision: Lfm3VlModel,
    audio: Option<AudioInputEncoder>,
}

impl Lfm3D1Model {
    pub(crate) fn new(
        config: &config::Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        metadata: NormalLoadingMetadata,
        attention: AttentionImplementation,
    ) -> Result<Self> {
        let audio_vb = vb
            .clone()
            .without_lora_registry()
            .pp("model")
            .set_device(metadata.real_device.clone());
        let audio = config
            .audio_encoder_config
            .as_ref()
            .map(|audio| {
                Ok::<_, candle_core::Error>(AudioInputEncoder {
                    encoder: AudioEncoder::new(audio, audio_vb.pp("audio_encoder"))?,
                    projector: AudioProjector::new(
                        audio.hidden_size,
                        config.vision.text_config.hidden_size,
                        audio_vb.pp("audio_projector"),
                    )?,
                    memory: AudioMemoryEstimate::new(
                        audio,
                        config.vision.text_config.hidden_size,
                        audio_vb.dtype(),
                        audio_vb.device(),
                    ),
                })
            })
            .transpose()?;
        Ok(Self {
            vision: Lfm3VlModel::new(&config.vision, vb, is_gptx, metadata, attention)?,
            audio,
        })
    }

    fn input_embeddings(
        &self,
        ids: &Tensor,
        pixels: Option<Tensor>,
        args: Box<dyn Any>,
    ) -> Result<Tensor> {
        let (vision, audios) = match args.downcast::<Lfm3OmniSpecificArgs>() {
            Ok(args) => (
                args.vision
                    .ok_or_else(|| candle_core::Error::msg("Missing d1 vision inputs"))?,
                args.audio,
            ),
            Err(args) => (
                *args
                    .downcast::<GridVisionSpecificArgs>()
                    .map_err(|_| candle_core::Error::msg("Invalid d1 model inputs"))?,
                Vec::new(),
            ),
        };
        let mut input = self.vision.input_embeddings(ids, pixels, vision)?;
        let hidden = input.dim(2)?;
        for item in audios {
            let audio = self.audio.as_ref().ok_or_else(|| {
                candle_core::Error::msg("This d1 checkpoint does not accept audio")
            })?;
            let cache = self.vision.encoder_cache().expect("LFM3-VL encoder cache");
            let cached = cache
                .lock()
                .expect("encoder cache poisoned")
                .get(CacheModality::Audio, item.hash);
            let features = if let Some(cached) = cached {
                cached[0].clone()
            } else {
                let frames = item.features.dim(2)?;
                let features = audio
                    .projector
                    .forward(&audio.encoder.forward(&item.features, &[frames])?)?
                    .squeeze(0)?;
                cache.lock().expect("encoder cache poisoned").insert(
                    CacheModality::Audio,
                    item.hash,
                    vec![features.clone()],
                );
                features
            };
            if features.dim(0)? != item.token_range.len() {
                candle_core::bail!("d1 audio features do not match their prompt placeholders");
            }
            input = input.slice_assign(
                &[
                    item.batch_index..item.batch_index + 1,
                    item.token_range,
                    0..hidden,
                ],
                &features
                    .to_device(input.device())?
                    .to_dtype(input.dtype())?
                    .unsqueeze(0)?,
            )?;
        }
        Ok(input)
    }
}

impl IsqModel for Lfm3D1Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(self.vision.residual_tensors());
        if let Some(audio) = &self.audio {
            uvb.pp("model.audio_encoder")
                .extend(audio.encoder.residual_tensors());
            uvb.pp("model.audio_projector")
                .extend(audio.projector.residual_tensors());
        }
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Lfm3D1Model {}
impl crate::block_diffusion::BlockDiffusionMixin for Lfm3D1Model {}
impl AnyMoeBaseModelMixin for Lfm3D1Model {}

impl MultimodalModel for Lfm3D1Model {
    fn forward(
        &self,
        _input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        _args: Box<dyn Any>,
        _ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        candle_core::bail!(
            "d1 is a decision model; use a decision request instead of token generation"
        )
    }

    fn forward_decision(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        attention: &DecisionAttention,
        readout: &crate::decision::DecisionReadout,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, length) = input_ids.dims2()?;
        if batch != 1 {
            candle_core::bail!("A d1 decision forward expects one packed question set");
        }
        attention.validate(batch, length)?;
        let input = self.input_embeddings(input_ids, pixel_values, model_specific_args)?;
        let hidden = self
            .vision
            .language_model
            .forward_decision_hidden(input, |start, width, dtype, device| {
                attention.rows(start, width, length, dtype, device)
            })?
            .squeeze(0)?;
        let logits = self
            .vision
            .language_model
            .project_decision_slots(&hidden, readout)?;
        Ok((hidden, logits))
    }

    fn device(&self) -> &Device {
        self.vision.device()
    }
    fn cache(&self) -> &EitherCache {
        self.vision.cache()
    }
    fn max_seq_len(&self) -> usize {
        self.vision.max_seq_len()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.vision.config()
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(config::ModelConfig(self.vision.model_config()))
    }
    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
    fn encoder_activation_memory_bytes(
        &self,
        seq: &crate::sequence::Sequence,
    ) -> Result<Option<usize>> {
        let vision = self
            .vision
            .encoder_activation_memory_bytes(seq)?
            .unwrap_or(0);
        let audio = self
            .audio
            .as_ref()
            .map(|audio| audio.memory.for_sequence(seq))
            .transpose()?
            .unwrap_or(0);
        Ok(Some(vision + audio))
    }
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        self.vision.default_model_specific_args(input_ids)
    }
    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        self.vision.encoder_cache()
    }
}
