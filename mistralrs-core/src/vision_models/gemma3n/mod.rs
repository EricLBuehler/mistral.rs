#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Result, Tensor, D};
use config::Gemma3nConfig;
use mistralrs_quant::{NonZeroOp, ShardedVarBuilder};
use text::TextModel;

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
    vision_models::multimodal_layout::{
        MultimodalEncoderKey, MultimodalEncoderOutputs, PackedMultimodalLayout,
    },
};

use self::multimodal_embedding::Gemma3nMultimodalEmbedder;

pub(crate) mod audio;
pub(crate) mod audio_processing;
pub mod config;
mod inputs_processor;
mod multimodal_embedding;
pub(crate) mod text;
pub mod vision;
pub(crate) use inputs_processor::Gemma3nProcessor;

fn select_encoder_rows(outputs: &[Tensor], ranges: &[Range<usize>]) -> Result<Tensor> {
    if outputs.len() != ranges.len() || outputs.is_empty() {
        candle_core::bail!("Gemma 3n encoder output metadata length mismatch");
    }
    let mut selected = Vec::with_capacity(outputs.len());
    for (output, range) in outputs.iter().zip(ranges) {
        if range.start > range.end || range.end > output.dim(0)? {
            candle_core::bail!("Gemma 3n encoder source range exceeds its output");
        }
        selected.push(output.narrow(0, range.start, range.len())?);
    }
    Tensor::cat(&selected, 0)
}

fn scatter_soft_embeddings(
    input_ids: &Tensor,
    input_embeds: &Tensor,
    token_id: u32,
    outputs: &[Tensor],
    ranges: &[Range<usize>],
) -> Result<Tensor> {
    let positions = input_ids
        .eq(token_id)?
        .flatten_all()?
        .nonzero()?
        .squeeze(1)?;
    let token_count = positions.dim(0)?;
    if token_count == 0 {
        if outputs.is_empty() && ranges.is_empty() {
            return Ok(input_embeds.clone());
        }
        candle_core::bail!("Gemma 3n has encoder outputs without active placeholder tokens");
    }
    let hidden_size = input_embeds.dim(D::Minus1)?;
    let source = select_encoder_rows(outputs, ranges)?
        .to_device(input_embeds.device())?
        .to_dtype(input_embeds.dtype())?
        .reshape(((), hidden_size))?;
    if source.dim(0)? != token_count {
        candle_core::bail!(
            "Gemma 3n has {token_count} active placeholder tokens but {} encoder rows",
            source.dim(0)?
        );
    }
    let shape = input_embeds.shape().clone();
    let mut flat = input_embeds.reshape(((), hidden_size))?;
    let current = flat.index_select(&positions, 0)?;
    let positions = positions.unsqueeze(1)?.repeat((1, hidden_size))?;
    flat = flat.scatter_add(&positions, &(source - current)?, 0)?;
    flat.reshape(shape)
}

pub struct Gemma3nModel {
    language_model: TextModel,
    vision_tower: vision::VisionTower,
    audio_tower: audio::AudioModel,
    embed_vision: Gemma3nMultimodalEmbedder,
    embed_audio: Gemma3nMultimodalEmbedder,
    cfg: config::Gemma3nConfig,
    vision_dtype: DType,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Gemma3nModel {
    pub fn new(
        cfg: &Gemma3nConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb = vb.pp("model");
        let non_text_vb = vb.clone().without_lora_registry();

        // Initialize vision tower
        let vision_dtype = if vb.dtype() == DType::F16 {
            // f16 -> f32 for vision model in particular.
            DType::F32
        } else {
            vb.dtype()
        };
        let vision_tower = vision::VisionTower::new(
            normal_loading_metadata
                .mapper
                .set_nm_device(non_text_vb.pp("vision_tower").pp("timm_model"), false)
                .set_dtype(vision_dtype),
        )?;

        // Initialize audio tower and embedder
        let audio_cfg = &cfg.audio_config;
        let audio_tower = audio::AudioModel::new(
            audio_cfg,
            normal_loading_metadata
                .mapper
                .set_nm_device(non_text_vb.pp("audio_tower"), false),
        )?;
        let embed_audio = Gemma3nMultimodalEmbedder::new(
            &cfg.text_config,
            audio_cfg.vocab_size,
            audio_cfg.hidden_size,
            audio_cfg.vocab_offset,
            normal_loading_metadata
                .mapper
                .set_nm_device(non_text_vb.pp("embed_audio"), false),
        )?;

        // Initialize vision tower and embedder
        let multimodal_cfg = &cfg.vision_config;
        let embed_vision = Gemma3nMultimodalEmbedder::new(
            &cfg.text_config,
            multimodal_cfg.vocab_size,
            multimodal_cfg.hidden_size,
            multimodal_cfg.vocab_offset,
            normal_loading_metadata
                .mapper
                .set_nm_device(non_text_vb.pp("embed_vision"), false),
        )?;

        Ok(Self {
            language_model: TextModel::new(
                &cfg.text_config,
                vb.pp("language_model"),
                is_gptx,
                normal_loading_metadata,
                attention_mechanism,
            )?,
            vision_tower,
            audio_tower,
            embed_vision,
            embed_audio,
            cfg: cfg.clone(),
            vision_dtype,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn replace_hard_embeddings(
        input_ids: &Tensor,
        input_embeds: &Tensor,
        mask: &Tensor,
        embedder: &Gemma3nMultimodalEmbedder,
    ) -> Result<Tensor> {
        let positions = mask.flatten_all()?.nonzero()?.squeeze(1)?;
        if positions.dim(0)? == 0 {
            return Ok(input_embeds.clone());
        }
        let hidden_size = input_embeds.dim(D::Minus1)?;
        let ids = input_ids.flatten_all()?.index_select(&positions, 0)?;
        let source = embedder
            .forward_text(&ids)?
            .to_device(input_embeds.device())?
            .to_dtype(input_embeds.dtype())?
            .reshape(((), hidden_size))?;
        let shape = input_embeds.shape().clone();
        let mut flat = input_embeds.reshape(((), hidden_size))?;
        let current = flat.index_select(&positions, 0)?;
        let positions = positions.unsqueeze(1)?.repeat((1, hidden_size))?;
        flat = flat.scatter_add(&positions, &(source - current)?, 0)?;
        flat.reshape(shape)
    }

    fn encode_images(
        &self,
        pixel_values: &Tensor,
        image_hashes: &[u64],
        output_dtype: DType,
    ) -> Result<Vec<Tensor>> {
        let count = pixel_values.dim(0)?;
        if image_hashes.len() != count {
            candle_core::bail!(
                "Gemma 3n has {count} image inputs but {} image hashes",
                image_hashes.len()
            );
        }
        let mut outputs = Vec::with_capacity(count);
        for (index, &hash) in image_hashes.iter().enumerate() {
            let cached = {
                let mut cache = self
                    .encoder_cache
                    .lock()
                    .expect("encoder cache lock poisoned");
                cache
                    .get(CacheModality::Image, hash)
                    .map(|outputs| outputs.to_vec())
            };
            if let Some(cached) = cached {
                if cached.len() != 1 {
                    candle_core::bail!("Gemma 3n cached image output cardinality is invalid");
                }
                outputs.push(cached[0].clone());
                continue;
            }

            let pixels = pixel_values.get(index)?.unsqueeze(0)?;
            let features = self
                .vision_tower
                .forward(&pixels.to_dtype(self.vision_dtype)?)?
                .to_dtype(output_dtype)?;
            let (_, channels, height, width) = features.dims4()?;
            let features =
                features
                    .permute((0, 2, 3, 1))?
                    .reshape((1, height * width, channels))?;
            let output = self.embed_vision.forward_vision(&features)?.squeeze(0)?;
            self.encoder_cache
                .lock()
                .expect("encoder cache lock poisoned")
                .insert(CacheModality::Image, hash, vec![output.clone()]);
            outputs.push(output);
        }
        Ok(outputs)
    }

    fn audio_padding_embedding(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let token_id = Tensor::new(&[(self.cfg.text_config.vocab_size - 1) as u32], device)?;
        self.embed_audio.forward_text(&token_id)?.to_dtype(dtype)
    }

    fn encode_audio_item(
        &self,
        audio_mel: &Tensor,
        audio_mel_mask: &Tensor,
        output_dtype: DType,
    ) -> Result<Tensor> {
        let (features, output_mask) = self
            .audio_tower
            .forward(&audio_mel.to_dtype(output_dtype)?, audio_mel_mask)?;
        let mut output = self.embed_audio.forward_vision(&features)?;
        let padding = self
            .audio_padding_embedding(output.device(), output.dtype())?
            .unsqueeze(0)?
            .broadcast_as(output.shape())?;
        output = output_mask
            .ne(0.)?
            .unsqueeze(D::Minus1)?
            .broadcast_as(output.shape())?
            .where_cond(&padding, &output)?;

        let expected = self.cfg.audio_soft_tokens_per_image;
        let length = output.dim(1)?;
        if length < expected {
            let padding = self
                .audio_padding_embedding(output.device(), output.dtype())?
                .unsqueeze(0)?
                .repeat((1, expected - length, 1))?;
            output = Tensor::cat(&[output, padding], 1)?;
        } else if length > expected {
            output = output.narrow(1, 0, expected)?;
        }
        output.squeeze(0)
    }

    fn encode_audios(
        &self,
        audio_mel: &Tensor,
        audio_mel_mask: &Tensor,
        audio_hashes: &[u64],
        output_dtype: DType,
    ) -> Result<Vec<Tensor>> {
        let count = audio_mel.dim(0)?;
        if audio_mel_mask.dim(0)? != count || audio_hashes.len() != count {
            candle_core::bail!("Gemma 3n active audio metadata length mismatch");
        }
        let mut outputs = Vec::with_capacity(count);
        for (index, &hash) in audio_hashes.iter().enumerate() {
            let cached = {
                let mut cache = self
                    .encoder_cache
                    .lock()
                    .expect("encoder cache lock poisoned");
                cache
                    .get(CacheModality::Audio, hash)
                    .map(|outputs| outputs.to_vec())
            };
            if let Some(cached) = cached {
                if cached.len() != 1 {
                    candle_core::bail!("Gemma 3n cached audio output cardinality is invalid");
                }
                outputs.push(cached[0].clone());
                continue;
            }

            let mel = audio_mel.get(index)?.unsqueeze(0)?;
            let mask = audio_mel_mask.get(index)?.unsqueeze(0)?;
            let output = self.encode_audio_item(&mel, &mask, output_dtype)?;
            self.encoder_cache
                .lock()
                .expect("encoder cache lock poisoned")
                .insert(CacheModality::Audio, hash, vec![output.clone()]);
            outputs.push(output);
        }
        Ok(outputs)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        ctx: &mut ModelForwardContext<'_>,
        audio_mel: Option<&Tensor>,
        audio_mel_mask: Option<&Tensor>,
        image_hashes: &[u64],
        image_source_ranges: &[Range<usize>],
        audio_hashes: &[u64],
        audio_source_ranges: &[Range<usize>],
        packed_layout: Option<&PackedMultimodalLayout>,
    ) -> Result<Tensor> {
        let vision_vocab_offset = self.cfg.vision_config.vocab_offset as f64;
        let audio_vocab_offset = self.cfg.audio_config.vocab_offset as f64;

        let mut input_embeds = self.language_model.embed_tokens(input_ids)?;
        let vision_mask = input_ids
            .to_dtype(DType::F32)?
            .ge(vision_vocab_offset)?
            .mul(&input_ids.to_dtype(DType::F32)?.lt(audio_vocab_offset)?)?;
        input_embeds = Self::replace_hard_embeddings(
            input_ids,
            &input_embeds,
            &vision_mask,
            &self.embed_vision,
        )?;
        let audio_mask = input_ids.to_dtype(DType::F32)?.ge(audio_vocab_offset)?;
        input_embeds = Self::replace_hard_embeddings(
            input_ids,
            &input_embeds,
            &audio_mask,
            &self.embed_audio,
        )?;
        let mut encoder_outputs = MultimodalEncoderOutputs::new();

        if let Some(pixel_values) = pixel_values {
            let outputs = self.encode_images(&pixel_values, image_hashes, input_embeds.dtype())?;
            if outputs.len() != image_source_ranges.len() {
                candle_core::bail!("Gemma 3n active image metadata length mismatch");
            }
            if packed_layout.is_some() {
                for ((&hash, output), source) in
                    image_hashes.iter().zip(outputs).zip(image_source_ranges)
                {
                    if source.start != 0 || source.end != output.dim(0)? {
                        candle_core::bail!(
                            "Gemma 3n packed image prefill requires complete encoder outputs"
                        );
                    }
                    encoder_outputs.insert(
                        MultimodalEncoderKey {
                            kind: crate::paged_attention::block_hash::MultimodalKind::Image,
                            hash,
                        },
                        vec![output],
                    );
                }
            } else {
                input_embeds = scatter_soft_embeddings(
                    input_ids,
                    &input_embeds,
                    inputs_processor::IMAGE_TOKEN_ID,
                    &outputs,
                    image_source_ranges,
                )?;
            }
        } else if !image_hashes.is_empty() || !image_source_ranges.is_empty() {
            candle_core::bail!("Gemma 3n image inputs are incomplete");
        }

        match (audio_mel, audio_mel_mask) {
            (Some(audio_mel), Some(audio_mel_mask)) => {
                let outputs = self.encode_audios(
                    audio_mel,
                    audio_mel_mask,
                    audio_hashes,
                    input_embeds.dtype(),
                )?;
                if outputs.len() != audio_source_ranges.len() {
                    candle_core::bail!("Gemma 3n active audio metadata length mismatch");
                }
                if packed_layout.is_some() {
                    for ((&hash, output), source) in
                        audio_hashes.iter().zip(outputs).zip(audio_source_ranges)
                    {
                        if source.start != 0 || source.end != output.dim(0)? {
                            candle_core::bail!(
                                "Gemma 3n packed audio prefill requires complete encoder outputs"
                            );
                        }
                        encoder_outputs.insert(
                            MultimodalEncoderKey {
                                kind: crate::paged_attention::block_hash::MultimodalKind::Audio,
                                hash,
                            },
                            vec![output],
                        );
                    }
                } else {
                    input_embeds = scatter_soft_embeddings(
                        input_ids,
                        &input_embeds,
                        inputs_processor::AUDIO_TOKEN_ID,
                        &outputs,
                        audio_source_ranges,
                    )?;
                }
            }
            (None, None) if audio_hashes.is_empty() && audio_source_ranges.is_empty() => {}
            _ => candle_core::bail!("Gemma 3n audio inputs are incomplete"),
        }

        if let Some(layout) = packed_layout {
            input_embeds = layout.splice_embeddings(&input_embeds, &encoder_outputs)?;
        }

        let ple_inputs_mask =
            input_ids.lt(self.cfg.text_config.vocab_size_per_layer_input as f64)?;
        let ple_input_ids = ple_inputs_mask.where_cond(input_ids, &input_ids.zeros_like()?)?;

        let res =
            self.language_model
                .forward_embeds(input_ids, &ple_input_ids, input_embeds, ctx)?;
        Ok(res)
    }
}

impl IsqModel for Gemma3nModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_model = uvb.pp("model");

        // Add language model residual tensors
        let uvb_language = uvb_model.pp("language_model");
        uvb_language.extend(self.language_model.residual_tensors());

        // Add vision tower residual tensors (conv layers, norms, etc.)
        // Vision tower uses Conv2d layers which are not quantized
        let uvb_vision = uvb_model.pp("vision_tower").pp("timm_model");
        uvb_vision.extend(self.vision_tower.residual_tensors());

        // Add audio tower residual tensors (norms, conv layers, etc.)
        let uvb_audio = uvb_model.pp("audio_tower");
        uvb_audio.extend(self.audio_tower.residual_tensors());

        // Add multimodal embedder residual tensors (embeddings, norms)
        let uvb_embed_vision = uvb_model.pp("embed_vision");
        uvb_embed_vision
            .pp("embedding")
            .add(&self.embed_vision.embedding);
        uvb_embed_vision
            .pp("hard_embedding_norm")
            .add(&self.embed_vision.hard_embedding_norm);
        uvb_embed_vision
            .pp("soft_embedding_norm")
            .add(&self.embed_vision.soft_embedding_norm);
        uvb_embed_vision
            .pp("embedding_post_projection_norm")
            .add(&self.embed_vision.embedding_post_projection_norm);

        let uvb_embed_audio = uvb_model.pp("embed_audio");
        uvb_embed_audio
            .pp("embedding")
            .add(&self.embed_audio.embedding);
        uvb_embed_audio
            .pp("hard_embedding_norm")
            .add(&self.embed_audio.hard_embedding_norm);
        uvb_embed_audio
            .pp("soft_embedding_norm")
            .add(&self.embed_audio.soft_embedding_norm);
        uvb_embed_audio
            .pp("embedding_post_projection_norm")
            .add(&self.embed_audio.embedding_post_projection_norm);

        uvb.to_safetensors()
    }
}

#[derive(Default)]
pub struct Gemma3nSpecificArgs {
    pub audio_mel: Option<Tensor>,
    pub audio_mel_mask: Option<Tensor>,
    pub image_hashes: Vec<u64>,
    pub image_source_ranges: Vec<Range<usize>>,
    pub audio_hashes: Vec<u64>,
    pub audio_source_ranges: Vec<Range<usize>>,
    pub packed_layout: Option<PackedMultimodalLayout>,
}

impl crate::speculative::SpeculativeTargetMixin for Gemma3nModel {}

impl crate::block_diffusion::BlockDiffusionMixin for Gemma3nModel {}

impl MultimodalModel for Gemma3nModel {
    fn supports_packed_prefill(&self) -> bool {
        true
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor> {
        let args = model_specific_args
            .downcast::<Gemma3nSpecificArgs>()
            .expect("Downcast to Gemma3nSpecificArgs failed");

        self.forward(
            input_ids,
            pixel_values,
            ctx,
            args.audio_mel.as_ref(),
            args.audio_mel_mask.as_ref(),
            &args.image_hashes,
            &args.image_source_ranges,
            &args.audio_hashes,
            &args.audio_source_ranges,
            args.packed_layout.as_ref(),
        )
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(Gemma3nSpecificArgs::default())
    }
    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }
    fn device(&self) -> &Device {
        self.language_model.device()
    }
    fn max_seq_len(&self) -> usize {
        self.language_model.max_seq_len()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.language_model.config()
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        self.language_model.model_config_like()
    }
    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        Some(&self.encoder_cache)
    }
    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    )> {
        Some(
            self.encoder_cache
                .lock()
                .expect("encoder cache poisoned")
                .counters(),
        )
    }
}

impl AnyMoeBaseModelMixin for Gemma3nModel {}
