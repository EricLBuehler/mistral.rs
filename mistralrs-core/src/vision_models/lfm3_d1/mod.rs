mod calibration;
pub(crate) mod config;
mod inputs;
mod parsing;
mod prompt;
mod readout;

pub(crate) use inputs::DecisionProcessor;

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use anyhow::Context;
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    decision::{DecisionResponse, DecisionUsage},
    paged_attention::{
        encoder_cache::{CacheModality, EncoderCacheManager},
        AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
    },
    pipeline::{
        DecisionForwardContext, EitherCache, IsqModel, ModelForwardContext, MultimodalModel,
        NormalLoadingMetadata,
    },
    sequence::{MultimodalData, Sequence},
    utils::unvarbuilder::UnVarBuilder,
    vision_models::{
        grid_processor::GridVisionSpecificArgs,
        lfm3_omni::{
            audio::{AudioEncoder, AudioProjector},
            inputs_processor::Lfm3OmniSpecificArgs,
        },
        lfm3_vl::Lfm3VlModel,
    },
};

use prompt::DecisionPlan;

struct AudioInputEncoder {
    encoder: AudioEncoder,
    projector: AudioProjector,
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
                })
            })
            .transpose()?;
        Ok(Self {
            vision: Lfm3VlModel::new(&config.vision, vb, is_gptx, metadata, attention)?,
            audio,
        })
    }

    fn run_decision(
        &self,
        sequence: &mut Sequence,
        context: &DecisionForwardContext<'_>,
    ) -> anyhow::Result<DecisionResponse> {
        let processor = context
            .processor
            .as_any()
            .downcast_ref::<DecisionProcessor>()
            .context("Invalid d1 processor")?;
        let request = sequence
            .decision
            .clone()
            .context("Missing decision request")?;
        let mut answers = indexmap::IndexMap::new();
        let mut input_tokens = 0;
        for prompt in processor.prompts(&request, context.tokenizer)? {
            if sequence.responder().is_closed() {
                break;
            }
            let row = processor.prepare_row(sequence, prompt?, context)?;
            sequence.multimodal = MultimodalData::new(None, None, None, None, None, None);
            let input = self.input_embeddings(
                &row.inputs.input_ids,
                row.inputs.pixel_values,
                row.inputs.model_specific_args,
            )?;
            let hidden = self
                .vision
                .language_model
                .forward_decision_hidden(input, |start, width, dtype, device| {
                    attention_mask(&row.plan, start, width, dtype, device)
                })?
                .squeeze(0)?;
            let readout = row.plan.code_readout();
            let logits = self.vision.language_model.project_decision_slots(
                &hidden,
                &readout.positions,
                &readout.token_ids,
            )?;
            input_tokens += row.plan.ids.len();
            answers.extend(
                row.plan
                    .decode(&hidden, &logits, &processor.config.d1_calibration)?,
            );
        }
        Ok(DecisionResponse {
            answers,
            usage: DecisionUsage {
                input_tokens,
                output_tokens: 0,
            },
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
        sequence: &mut Sequence,
        context: DecisionForwardContext<'_>,
    ) -> anyhow::Result<DecisionResponse> {
        let result = self.run_decision(sequence, &context);
        sequence.multimodal = MultimodalData::new(None, None, None, None, None, None);
        result
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
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        self.vision.default_model_specific_args(input_ids)
    }
    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        self.vision.encoder_cache()
    }
}

fn attention_mask(
    plan: &DecisionPlan,
    start: usize,
    width: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let length = plan.ids.len();
    let values = (start..start + width)
        .flat_map(|query| {
            (0..length).map(move |key| {
                if query < plan.state_len
                    || key < plan.state_len
                    || plan.question_ids[query] == plan.question_ids[key]
                {
                    0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, width, length), device)?.to_dtype(dtype)
}
