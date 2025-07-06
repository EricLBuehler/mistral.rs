#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};
use config::Gemma3nConfig;
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};
use text::TextModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, VisionModel,
    },
    utils::unvarbuilder::UnVarBuilder,
};

use self::multimodal_embedding::Gemma3nMultimodalEmbedder;

mod audio;
mod audio_processing;
pub mod config;
mod inputs_processor;
mod multimodal_embedding;
mod text;
mod vision;
pub(crate) use inputs_processor::Gemma3nProcessor;

pub struct Gemma3nModel {
    language_model: TextModel,
    vision_tower: vision::VisionTower,
    audio_tower: audio::AudioModel,
    embed_vision: Gemma3nMultimodalEmbedder,
    embed_audio: Gemma3nMultimodalEmbedder,
    cfg: config::Gemma3nConfig,
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

        let mapper = &normal_loading_metadata.mapper;

        // Initialize vision tower
        let vision_tower = vision::VisionTower::new(
            mapper.set_nm_device(vb.pp("vision_tower").pp("timm_model"), false),
        )?;

        // Initialize audio tower and embedder
        let audio_cfg = &cfg.audio_config;
        let audio_tower =
            audio::AudioModel::new(audio_cfg, mapper.set_nm_device(vb.pp("audio_tower"), false))?;
        let embed_audio = Gemma3nMultimodalEmbedder::new(
            &cfg.text_config,
            audio_cfg.vocab_size,
            audio_cfg.hidden_size,
            audio_cfg.vocab_offset,
            mapper.set_nm_device(vb.pp("embed_audio"), false),
        )?;

        // Initialize vision tower and embedder
        let vision_cfg = &cfg.vision_config;
        let embed_vision = Gemma3nMultimodalEmbedder::new(
            &cfg.text_config,
            vision_cfg.vocab_size,
            vision_cfg.hidden_size,
            vision_cfg.vocab_offset,
            mapper.set_nm_device(vb.pp("embed_vision"), false),
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
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
        audio_mel: Option<&Tensor>,
        audio_mel_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let vision_vocab_offset = self.cfg.vision_config.vocab_offset as f64;
        let audio_vocab_offset = self.cfg.audio_config.vocab_offset as f64;

        let mut input_embeds = self.language_model.embed_tokens(input_ids)?;

        if let Some(pixel_values) = pixel_values {
            let vision_mask = input_ids
                .to_dtype(DType::F32)?
                .ge(vision_vocab_offset)?
                .mul(&input_ids.to_dtype(DType::F32)?.lt(audio_vocab_offset)?)?;

            let vision_mask_idx = vision_mask.flatten_all()?.nonzero()?.squeeze(1)?;
            let vision_mask_embed_idx = vision_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .flatten_all()?
                .nonzero()?
                .squeeze(1)?;

            let vision_token_embeds = self
                .embed_vision
                .forward_text(&input_ids.flatten_all()?.index_select(&vision_mask_idx, 0)?)?
                .reshape((input_ids.dim(0)?, (), input_embeds.dim(D::Minus1)?))?;

            let mut x_flat = input_embeds.flatten_all()?;
            let src_flat = vision_token_embeds.flatten_all()?;

            // Replace image tokens with actual vision embeddings
            let current_vals = x_flat.gather(&vision_mask_embed_idx, 0)?;
            let diff = (src_flat - current_vals)?;
            x_flat = x_flat.scatter_add(&vision_mask_embed_idx, &diff, 0)?;

            input_embeds = x_flat.reshape(input_embeds.shape())?;

            // Process vision inputs through vision tower
            // TODO: this is a hack necessary because the weights for Gemma 3n are broken and require the image to be rotated.
            let pixel_values = pixel_values.t()?;
            let vision_features = self
                .vision_tower
                .forward(&pixel_values.to_dtype(input_embeds.dtype())?)?;

            // Reshape vision features to (batch_size * num_images, soft_tokens_per_image, hidden_size)
            let (batch_size, channels, h, w) = vision_features.dims4()?;
            let vision_features = vision_features
                .permute((0, 2, 3, 1))? // NCHW -> NHWC
                .reshape((batch_size, h * w, channels))?;

            // Convert vision features to embeddings using multimodal embedder
            let image_embeds = self.embed_vision.forward_vision(&vision_features)?;

            // Create mask specifically for the image soft tokens (not BOI/EOI)
            let image_token_mask = input_ids
                .to_dtype(DType::F32)?
                .eq(inputs_processor::IMAGE_TOKEN_ID as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(DType::U32)?;

            // Flatten tensors for scatter operation
            let mask_flat = image_token_mask.flatten_all()?;
            let indices = mask_flat.nonzero()?.squeeze(1)?;

            // Only do the replacement if we have image tokens to replace
            if indices.dim(0)? > 0 {
                let mut x_flat = input_embeds.flatten_all()?;
                let src_flat = image_embeds.flatten_all()?;

                // Replace image tokens with actual vision embeddings
                let current_vals = x_flat.gather(&indices, 0)?;
                let diff = (src_flat - current_vals)?;
                x_flat = x_flat.scatter_add(&indices, &diff, 0)?;

                input_embeds = x_flat.reshape(input_embeds.shape())?;
            }
        }

        if let (Some(audio_mel), Some(audio_mel_mask)) = (audio_mel, audio_mel_mask) {
            let audio_mask = input_ids.to_dtype(DType::F32)?.ge(audio_vocab_offset)?;

            let audio_mask_idx = audio_mask.flatten_all()?.nonzero()?.squeeze(1)?;
            let audio_mask_embed_idx = audio_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .flatten_all()?
                .nonzero()?
                .squeeze(1)?;

            let audio_token_embeds = self
                .embed_audio
                .forward_text(&input_ids.flatten_all()?.index_select(&audio_mask_idx, 0)?)?
                .reshape((input_ids.dim(0)?, (), input_embeds.dim(D::Minus1)?))?;

            let mut x_flat = input_embeds.flatten_all()?;
            let src_flat = audio_token_embeds.flatten_all()?;

            // Replace image tokens with actual audio embeddings
            let current_vals = x_flat.gather(&audio_mask_embed_idx, 0)?;
            let diff = (src_flat - current_vals)?;
            x_flat = x_flat.scatter_add(&audio_mask_embed_idx, &diff, 0)?;

            input_embeds = x_flat.reshape(input_embeds.shape())?;

            // Process audio through audio tower
            let (audio_features, _) = self.audio_tower.forward(audio_mel, audio_mel_mask)?;

            // Convert audio features to embeddings
            let mut audio_embeds = self.embed_audio.forward_vision(&audio_features)?;

            // Pad audio embeddings to expected length (188) if needed
            // This matches the transformers implementation
            let expected_audio_tokens = self.cfg.audio_soft_tokens_per_image;
            let num_audio_embeddings = audio_embeds.dim(1)?;

            if num_audio_embeddings < expected_audio_tokens {
                // Get the padding embedding (last token in audio vocabulary)
                let audio_vocab_size = self.cfg.audio_config.vocab_size;
                let padding_token_id =
                    Tensor::new(&[(audio_vocab_size - 1) as u32], audio_embeds.device())?;

                // Get the padding embedding
                let padding_embed = self.embed_audio.forward_text(&padding_token_id)?;

                // Calculate how many padding embeddings we need
                let num_padding = expected_audio_tokens - num_audio_embeddings;

                // Repeat the padding embedding
                let padding_embeds = padding_embed
                    .unsqueeze(0)? // Add batch dimension
                    .repeat(&[1, num_padding, 1])?; // [1, num_padding, embed_dim]

                // Concatenate original embeddings with padding
                audio_embeds = Tensor::cat(&[&audio_embeds, &padding_embeds], 1)?;
            }

            // Create mask for audio soft tokens
            let audio_token_mask = input_ids
                .to_dtype(DType::F32)?
                .eq(inputs_processor::AUDIO_TOKEN_ID as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(DType::U32)?;

            // Flatten tensors for scatter operation
            let mask_flat = audio_token_mask.flatten_all()?;
            let indices = mask_flat.nonzero()?.squeeze(1)?;

            // Only do the replacement if we have audio tokens to replace
            if indices.dim(0)? > 0 {
                let mut x_flat = input_embeds.flatten_all()?;
                let src_flat = audio_embeds.flatten_all()?;

                // Replace audio tokens with actual audio embeddings
                let current_vals = x_flat.gather(&indices, 0)?;
                let diff = (src_flat - current_vals)?;
                x_flat = x_flat.scatter_add(&indices, &diff, 0)?;

                input_embeds = x_flat.reshape(input_embeds.shape())?;
            }
        }

        let res = self.language_model.forward_embeds(
            input_ids,
            input_embeds,
            seqlen_offsets,
            context_lens,
            flash_params,
        )?;
        Ok(res)
    }
}

impl IsqModel for Gemma3nModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        self.language_model.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("model")
            .extend(self.language_model.residual_tensors());

        uvb.to_safetensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        self.language_model.imatrix_names()
    }
}

#[derive(Default)]
pub struct Gemma3nSpecificArgs {
    pub audio_mel: Option<Tensor>,
    pub audio_mel_mask: Option<Tensor>,
}

impl VisionModel for Gemma3nModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn std::any::Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        let args = model_specific_args
            .downcast::<Gemma3nSpecificArgs>()
            .expect("Downcast to Gemma3nSpecificArgs failed");

        self.forward(
            input_ids,
            pixel_values,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
            args.audio_mel.as_ref(),
            args.audio_mel_mask.as_ref(),
        )
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(Gemma3nSpecificArgs::default())
    }
    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.language_model.cache_mut()
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
}

impl AnyMoeBaseModelMixin for Gemma3nModel {}
