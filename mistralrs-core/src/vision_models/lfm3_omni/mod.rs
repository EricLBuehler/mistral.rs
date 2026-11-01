pub(crate) mod audio;
pub(crate) mod audio_processing;
pub(crate) mod codec;
pub(crate) mod config;
pub(crate) mod generation;
pub(crate) mod inputs_processor;
pub(crate) mod memory;

#[cfg(test)]
mod tests;

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
};

use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    paged_attention::{
        encoder_cache::{CacheModality, EncoderCacheManager},
        AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
    },
    pipeline::{
        EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata,
        NormalModel,
    },
    utils::unvarbuilder::UnVarBuilder,
    vision_models::lfm3_vl::Lfm3VlModel,
};

use audio::{AudioEncoder, AudioProjector};
use codec::{CodecStream, LiquidCodec};
pub(crate) use config::Config;
use config::SpeechConfig;
use generation::{AudioSampling, DepthDecoder, GenerationMarkers, GenerationMode, GenerationState};
pub(crate) use inputs_processor::Lfm3OmniProcessor;
use inputs_processor::Lfm3OmniSpecificArgs;

const ENCODER_CACHE_CAPACITY: usize = 32;

struct SequenceGeneration {
    state: GenerationState,
    frames: Vec<Vec<u32>>,
    code_history: BTreeMap<usize, Vec<u32>>,
    audio_id: String,
    stream: Option<CodecStream>,
}

impl SequenceGeneration {
    fn codes(&self) -> Result<Tensor> {
        let frames = self.frames.len();
        let books = self.frames[0].len();
        Tensor::from_vec(
            self.frames.iter().flatten().copied().collect::<Vec<_>>(),
            (1, frames, books),
            &Device::Cpu,
        )?
        .transpose(1, 2)?
        .contiguous()
    }
}

#[derive(Debug, PartialEq)]
struct GeneratedAudioInput {
    batch_index: usize,
    token_index: usize,
    codes: Vec<u32>,
}

#[derive(Default)]
struct GenerationCache {
    hidden: HashMap<usize, Tensor>,
    sequences: HashMap<usize, SequenceGeneration>,
}

impl GenerationCache {
    fn audio_inputs(
        &self,
        sequence_ids: &[usize],
        offsets: &[usize],
        query_lens: &[usize],
        packed: bool,
    ) -> Vec<GeneratedAudioInput> {
        let mut inputs = Vec::new();
        let mut packed_offset = 0;
        for (row, ((id, &offset), &length)) in
            sequence_ids.iter().zip(offsets).zip(query_lens).enumerate()
        {
            if let Some(generation) = self.sequences.get(id) {
                for (&position, codes) in generation.code_history.range(offset..offset + length) {
                    inputs.push(GeneratedAudioInput {
                        batch_index: if packed { 0 } else { row },
                        token_index: position - offset + if packed { packed_offset } else { 0 },
                        codes: codes.clone(),
                    });
                }
            }
            packed_offset += length;
        }
        inputs
    }

    fn release_sequences(&mut self, ids: &[usize]) {
        for id in ids {
            self.hidden.remove(id);
            self.sequences.remove(id);
        }
    }
}
struct AudioBackbone {
    language_model: crate::models::lfm3::Model,
    encoder_cache: Mutex<EncoderCacheManager>,
}

enum Backbone {
    Vision(Box<Lfm3VlModel>),
    Audio(Box<AudioBackbone>),
}

impl Backbone {
    fn language_model(&self) -> &crate::models::lfm3::Model {
        match self {
            Self::Vision(model) => &model.language_model,
            Self::Audio(model) => &model.language_model,
        }
    }

    fn encoder_cache(&self) -> &Mutex<EncoderCacheManager> {
        match self {
            Self::Vision(model) => model.encoder_cache().expect("LFM3-VL encoder cache"),
            Self::Audio(model) => &model.encoder_cache,
        }
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        match self {
            Self::Vision(model) => model.residual_tensors(),
            Self::Audio(model) => model
                .language_model
                .residual_tensors_m(UnVarBuilder::new().pp("model").pp("language_model")),
        }
    }
}

struct SpeechModelConfig<'a> {
    speech: &'a SpeechConfig,
    text: &'a crate::models::lfm3::Config,
    visual_tokens: &'a [u32],
    sampling: AudioSampling,
}

pub(crate) struct Lfm3OmniModel {
    backbone: Backbone,
    audio_sampling: AudioSampling,
    audio_encoder: AudioEncoder,
    audio_projector: AudioProjector,
    logits_mask: Tensor,
    speech_logits_mask: Tensor,
    depth_decoder: DepthDecoder,
    codebook_embedding: Embedding,
    codec: LiquidCodec,
    generation: Mutex<GenerationCache>,
    markers: GenerationMarkers,
    audio_out_token_id: u32,
    codebook_size: usize,
    maskgit_schedule: Vec<usize>,
    audio_memory: memory::AudioMemoryEstimate,
}

impl Lfm3OmniModel {
    pub(crate) fn new(
        config: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        metadata: NormalLoadingMetadata,
        attention: AttentionImplementation,
    ) -> Result<Self> {
        let device = metadata.real_device.clone();
        let backbone = Backbone::Vision(Box::new(Lfm3VlModel::new(
            &config.vision,
            vb.clone(),
            is_gptx,
            metadata,
            attention,
        )?));
        Self::new_inner(
            SpeechModelConfig {
                speech: &config.speech,
                text: &config.vision.text_config,
                visual_tokens: &[config.vision.image_token_id, config.vision.video_token_id],
                sampling: AudioSampling::default(),
            },
            backbone,
            vb,
            device,
        )
    }

    pub(crate) fn new_audio(
        config: &crate::vision_models::lfm3_audio::config::AudioConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        metadata: NormalLoadingMetadata,
        attention: AttentionImplementation,
    ) -> Result<Self> {
        let device = metadata.real_device.clone();
        let mut text_config = config.text_config.clone();
        text_config.tie_word_embeddings = config.tie_word_embeddings;
        let language_model = crate::models::lfm3::Model::new_inner(
            &text_config,
            vb.pp("model").pp("language_model"),
            vb.pp("lm_head"),
            is_gptx,
            metadata,
            attention,
        )?;
        let backbone = Backbone::Audio(Box::new(AudioBackbone {
            language_model,
            encoder_cache: Mutex::new(EncoderCacheManager::new(ENCODER_CACHE_CAPACITY)),
        }));
        Self::new_inner(
            SpeechModelConfig {
                speech: &config.speech,
                text: &text_config,
                visual_tokens: &[],
                sampling: AudioSampling::standalone(),
            },
            backbone,
            vb,
            device,
        )
    }

    fn new_inner(
        model_config: SpeechModelConfig<'_>,
        backbone: Backbone,
        vb: ShardedVarBuilder,
        device: Device,
    ) -> Result<Self> {
        let config = model_config.speech;
        let audio_vb = vb
            .clone()
            .without_lora_registry()
            .set_device(device.clone())
            .pp("model");
        let mut logits_mask = vec![0f32; model_config.text.vocab_size];
        for token in model_config
            .visual_tokens
            .iter()
            .copied()
            .chain([config.audio_token_id, config.audio_out_token_id])
        {
            logits_mask[token as usize] = f32::NEG_INFINITY;
        }
        let speech_logits_mask =
            Tensor::from_vec(logits_mask.clone(), model_config.text.vocab_size, &device)?
                .to_dtype(vb.dtype())?;
        logits_mask[config.audio_start_token_id as usize] = f32::NEG_INFINITY;
        logits_mask[config.mixed_start_token_id as usize] = f32::NEG_INFINITY;
        let logits_mask = Tensor::from_vec(logits_mask, model_config.text.vocab_size, &device)?
            .to_dtype(vb.dtype())?;
        Ok(Self {
            depth_decoder: DepthDecoder::new(
                &config.depth_config,
                model_config.text.hidden_size,
                config.num_codebooks,
                config.codebook_size,
                audio_vb.pp("depth_decoder"),
            )?,
            codebook_embedding: Embedding::new(
                audio_vb.pp("codebook_embedding").get(
                    (
                        config.num_codebooks * config.codebook_size,
                        model_config.text.hidden_size,
                    ),
                    "weight",
                )?,
                model_config.text.hidden_size,
            ),
            codec: LiquidCodec::new(
                config.codec_config.clone(),
                vb.clone()
                    .without_lora_registry()
                    .set_device(device.clone()),
            )?,
            generation: Mutex::new(GenerationCache::default()),
            markers: GenerationMarkers {
                audio_start: config.audio_start_token_id,
                mixed_start: config.mixed_start_token_id,
                text_end: config.text_end_token_id,
                audio_eos: config.audio_eos_code,
                text_slots: config.interleaved_n_text,
                audio_slots: config.interleaved_n_audio,
            },
            audio_out_token_id: config.audio_out_token_id,
            codebook_size: config.codebook_size,
            maskgit_schedule: config.maskgit_schedule.clone(),
            speech_logits_mask,
            audio_memory: memory::AudioMemoryEstimate::new(
                &config.audio_encoder_config,
                model_config.text.hidden_size,
                vb.dtype(),
                &device,
            ),
            audio_encoder: AudioEncoder::new(
                &config.audio_encoder_config,
                audio_vb.pp("audio_encoder"),
            )?,
            audio_projector: AudioProjector::new(
                config.audio_encoder_config.hidden_size,
                model_config.text.hidden_size,
                audio_vb.pp("audio_projector"),
            )?,
            backbone,
            audio_sampling: config
                .audio_sampling
                .clone()
                .unwrap_or(model_config.sampling),
            logits_mask,
        })
    }
}

impl IsqModel for Lfm3OmniModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(self.backbone.residual_tensors());
        uvb.pp("model")
            .pp("audio_encoder")
            .extend(self.audio_encoder.residual_tensors());
        uvb.pp("model")
            .pp("audio_projector")
            .extend(self.audio_projector.residual_tensors());
        uvb.pp("model")
            .pp("depth_decoder")
            .extend(self.depth_decoder.residual_tensors());
        uvb.pp("model")
            .pp("codebook_embedding")
            .add_tensor("weight", self.codebook_embedding.embeddings().clone());
        uvb.extend(self.codec.residual_tensors());
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Lfm3OmniModel {}
impl crate::block_diffusion::BlockDiffusionMixin for Lfm3OmniModel {}
impl AnyMoeBaseModelMixin for Lfm3OmniModel {}

impl MultimodalModel for Lfm3OmniModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<Lfm3OmniSpecificArgs>()
            .expect("Cannot downcast LFM3-Omni inputs");
        let mut input = match &self.backbone {
            Backbone::Vision(model) => model.input_embeddings(
                input_ids,
                pixel_values,
                args.vision.expect("LFM3-Omni vision arguments"),
            )?,
            Backbone::Audio(model) => model.language_model.embed(input_ids)?,
        };
        let hidden = input.dim(2)?;
        for audio in args.audio {
            let cache = self.backbone.encoder_cache();
            let cached = cache
                .lock()
                .expect("encoder cache poisoned")
                .get(CacheModality::Audio, audio.hash);
            let features = if let Some(cached) = cached {
                cached[0].clone()
            } else {
                let frames = audio.features.dim(2)?;
                let features = self
                    .audio_projector
                    .forward(&self.audio_encoder.forward(&audio.features, &[frames])?)?
                    .squeeze(0)?;
                cache.lock().expect("encoder cache poisoned").insert(
                    CacheModality::Audio,
                    audio.hash,
                    vec![features.clone()],
                );
                features
            };
            let count = audio.token_range.end - audio.token_range.start;
            if features.dim(0)? != count {
                candle_core::bail!(
                    "LFM3-Omni produced {} audio tokens for {count} placeholders",
                    features.dim(0)?
                );
            }
            input = input.slice_assign(
                &[
                    audio.batch_index..audio.batch_index + 1,
                    audio.token_range,
                    0..hidden,
                ],
                &features
                    .to_device(input.device())?
                    .to_dtype(input.dtype())?
                    .unsqueeze(0)?,
            )?;
        }
        let audio_inputs = {
            let generation = self
                .generation
                .lock()
                .expect("Omni generation cache poisoned");
            let query_lens = if ctx.flash_params().packed {
                ctx.paged_input_metadata()
                    .and_then(|metadata| metadata.query_lens.as_ref())
                    .ok_or_else(|| {
                        candle_core::Error::msg("Packed Omni input requires query lengths")
                    })?
                    .clone()
            } else {
                vec![input.dim(1)?; args.sequence_ids.len()]
            };
            generation.audio_inputs(
                &args.sequence_ids,
                ctx.seqlen_offsets(),
                &query_lens,
                ctx.flash_params().packed,
            )
        };
        for audio in audio_inputs {
            let codes = audio
                .codes
                .iter()
                .enumerate()
                .map(|(book, &code)| {
                    u32::try_from(book * self.codebook_size)
                        .map(|offset| code + offset)
                        .map_err(candle_core::Error::wrap)
                })
                .collect::<Result<Vec<_>>>()?;
            let ids = Tensor::from_slice(&codes, (1, codes.len()), self.device())?;
            let embedding = self
                .codebook_embedding
                .forward(&ids)?
                .sum(1)?
                .unsqueeze(1)?;
            input = input.slice_assign(
                &[
                    audio.batch_index..audio.batch_index + 1,
                    audio.token_index..audio.token_index + 1,
                    0..hidden,
                ],
                &embedding,
            )?;
        }
        let hidden = self
            .backbone
            .language_model()
            .forward_hidden(input_ids, input, ctx)?;
        if ctx.needs_logits() {
            let mut generation = self
                .generation
                .lock()
                .expect("Omni generation cache poisoned");
            for (row, &id) in args.sequence_ids.iter().enumerate() {
                // A narrow view would keep the full prefill storage resident.
                generation.hidden.insert(
                    id,
                    hidden
                        .get(row)?
                        .narrow(0, hidden.dim(1)? - 1, 1)?
                        .force_contiguous()?
                        .detach(),
                );
            }
        }
        self.backbone
            .language_model()
            .logits_from_hidden(&hidden, ctx)
    }

    fn has_custom_sampling(&self) -> bool {
        true
    }

    fn response_token_is_hidden(&self, token: u32) -> bool {
        [
            self.audio_out_token_id,
            self.markers.audio_start,
            self.markers.mixed_start,
            self.markers.text_end,
        ]
        .contains(&token)
    }

    fn prepare_sampling(
        &self,
        seq: &crate::sequence::Sequence,
        logits: Tensor,
        rng: &mut rand_isaac::Isaac64Rng,
    ) -> Result<(Tensor, Option<u32>)> {
        let Some(output) = seq.audio_output() else {
            return Ok((
                logits.broadcast_add(
                    &self
                        .logits_mask
                        .to_device(logits.device())?
                        .to_dtype(logits.dtype())?,
                )?,
                None,
            ));
        };
        let mut cache = self
            .generation
            .lock()
            .expect("Omni generation cache poisoned");
        let hidden = cache
            .hidden
            .get(seq.id())
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing Omni generation hidden state"))?;
        let generation = cache
            .sequences
            .entry(*seq.id())
            .or_insert_with(|| SequenceGeneration {
                state: GenerationState::new(
                    match output.mode {
                        crate::ChatAudioMode::Audio => GenerationMode::Audio,
                        crate::ChatAudioMode::TextAudio => GenerationMode::Interleaved,
                    },
                    self.markers,
                ),
                frames: Vec::new(),
                code_history: BTreeMap::new(),
                audio_id: format!("audio_{}", uuid::Uuid::new_v4().simple()),
                stream: None,
            });
        if generation.state.is_audio() {
            let mut codes = self.depth_decoder.sample_frame(
                &hidden,
                &self.maskgit_schedule,
                &self.audio_sampling,
                &generation.frames,
                rng,
            )?;
            if !generation.state.accept_audio(&mut codes) {
                generation.frames.push(codes.clone());
            }
            generation
                .code_history
                .insert(seq.prompt_position_source_toks().len(), codes);
            Ok((logits, Some(self.audio_out_token_id)))
        } else {
            Ok((
                logits.broadcast_add(
                    &self
                        .speech_logits_mask
                        .to_device(logits.device())?
                        .to_dtype(logits.dtype())?,
                )?,
                None,
            ))
        }
    }

    fn accept_sample(&self, seq: &crate::sequence::Sequence, token: u32) -> Result<()> {
        if token != self.audio_out_token_id {
            if let Some(generation) = self
                .generation
                .lock()
                .expect("Omni generation cache poisoned")
                .sequences
                .get_mut(seq.id())
            {
                generation.state.accept_text(token);
            }
        }
        Ok(())
    }

    fn sampling_eos_tokens(
        &self,
        seq: &crate::sequence::Sequence,
        tokens: Option<&[u32]>,
    ) -> Option<Vec<u32>> {
        let interleaved = self
            .generation
            .lock()
            .expect("Omni generation cache poisoned")
            .sequences
            .get(seq.id())
            .is_some_and(|generation| generation.state.is_interleaved());
        tokens.map(|tokens| {
            tokens
                .iter()
                .copied()
                .filter(|&token| !interleaved || token != self.markers.text_end)
                .collect()
        })
    }

    fn finalize_response(&self, seq: &mut crate::sequence::Sequence) -> Result<()> {
        let generation = {
            let mut cache = self
                .generation
                .lock()
                .expect("Omni generation cache poisoned");
            cache.hidden.remove(seq.id());
            cache.sequences.remove(seq.id())
        };
        if let (Some(output), Some(mut generation)) = (seq.audio_output(), generation) {
            let samples = if generation.frames.is_empty() {
                Vec::new()
            } else {
                let codes = generation.codes()?;
                let samples = if let Some(stream) = &mut generation.stream {
                    self.codec.decode_stream(&codes, stream, true)?
                } else {
                    self.codec.decode(&codes)?
                };
                samples.flatten_all()?.to_vec1::<f32>()?
            };
            let mut audio = crate::ChatAudioResponse::from_pcm(
                &samples,
                self.codec.sample_rate(),
                output.format,
            )
            .map_err(candle_core::Error::msg)?;
            audio.id = Some(generation.audio_id);
            seq.set_generated_audio(audio);
        }
        Ok(())
    }

    fn prepare_streaming_response(&self, seq: &mut crate::sequence::Sequence) -> Result<()> {
        if !seq
            .audio_output()
            .is_some_and(|output| output.format == crate::ChatAudioFormat::Pcm16)
        {
            return Ok(());
        }
        let mut cache = self
            .generation
            .lock()
            .expect("Omni generation cache poisoned");
        let Some(generation) = cache.sequences.get_mut(seq.id()) else {
            return Ok(());
        };
        let stream = generation.stream.get_or_insert_with(CodecStream::default);
        if !stream.is_ready(generation.frames.len()) {
            return Ok(());
        }
        let codes = generation.codes()?;
        let samples = self
            .codec
            .decode_stream(&codes, generation.stream.as_mut().unwrap(), false)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut audio = crate::ChatAudioResponse::from_pcm(
            &samples,
            self.codec.sample_rate(),
            crate::ChatAudioFormat::Pcm16,
        )
        .map_err(candle_core::Error::msg)?;
        audio.id = Some(generation.audio_id.clone());
        seq.set_generated_audio(audio);
        Ok(())
    }

    fn release_generation_sequences(&self, ids: &[usize]) {
        self.generation
            .lock()
            .expect("Omni generation cache poisoned")
            .release_sequences(ids);
    }

    fn device(&self) -> &Device {
        self.backbone.language_model().device()
    }
    fn cache(&self) -> &EitherCache {
        self.backbone.language_model().cache()
    }
    fn max_seq_len(&self) -> usize {
        self.backbone.language_model().max_seq_len()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.backbone.language_model().config()
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        self.backbone.language_model().model_config()
    }
    fn supports_packed_prefill(&self) -> bool {
        true
    }
    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
    fn encoder_activation_memory_bytes(
        &self,
        seq: &crate::sequence::Sequence,
    ) -> Result<Option<usize>> {
        let vision = match &self.backbone {
            Backbone::Vision(model) => model.encoder_activation_memory_bytes(seq)?,
            Backbone::Audio(_) => Some(0),
        };
        if vision.is_none() && (seq.has_images() || seq.has_videos()) {
            return Ok(None);
        }
        Ok(Some(
            vision.unwrap_or(0) + self.audio_memory.for_sequence(seq)?,
        ))
    }
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(Lfm3OmniSpecificArgs {
            vision: match &self.backbone {
                Backbone::Vision(model) => Some(
                    *model
                        .default_model_specific_args(input_ids)
                        .downcast()
                        .expect("LFM3-VL inputs"),
                ),
                Backbone::Audio(_) => None,
            },
            audio: vec![],
            sequence_ids: vec![],
        })
    }
    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        Some(self.backbone.encoder_cache())
    }
}
