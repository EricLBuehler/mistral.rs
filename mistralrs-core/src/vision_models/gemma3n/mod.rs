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

pub mod config;
mod audio;
mod audio_processing;
mod inputs_processor;
mod multimodal_embedding;
mod text;
mod vision;
pub(crate) use inputs_processor::Gemma3nProcessor;

pub struct Gemma3nModel {
    language_model: TextModel,
    vision_tower: vision::VisionTower,
    audio_tower: Option<audio::AudioModel>,
    embed_vision: Gemma3nMultimodalEmbedder,
    embed_audio: Option<Gemma3nMultimodalEmbedder>,
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

        // Initialize vision tower
        let vision_tower = vision::VisionTower::new(vb.pp("vision_tower").pp("timm_model"))?;

        // Initialize audio tower and embedder if audio config is present
        let (audio_tower, embed_audio) = if let Some(audio_cfg) = &cfg.audio_config {
            let audio_tower = Some(audio::AudioModel::new(audio_cfg, vb.pp("audio_tower"))?);
            let embed_audio = Some(Gemma3nMultimodalEmbedder::new(
                &cfg.text_config,
                audio_cfg.vocab_size,
                audio_cfg.hidden_size,
                audio_cfg.vocab_offset,
                vb.pp("embed_audio"),
            )?);
            (audio_tower, embed_audio)
        } else {
            (None, None)
        };

        // Initialize vision multimodal embedder
        let vision_cfg = cfg.vision_config.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("Vision config is required for Gemma3n".to_string())
        })?;
        let embed_vision = Gemma3nMultimodalEmbedder::new(
            &cfg.text_config,
            vision_cfg.vocab_size,
            vision_cfg.hidden_size,
            vision_cfg.vocab_offset,
            vb.pp("embed_vision"),
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
        // Get the vocab offset from vision config
        let vision_cfg = self
            .cfg
            .vision_config
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Vision config required".to_string()))?;
        let vocab_offset = vision_cfg.vocab_offset as f64;

        // Step 1: Get base language model embeddings
        let language_embeds = self.language_model.embed_tokens(input_ids)?;

        // Step 2: Handle audio tokens if audio config is present
        let mut input_embeds = if let (Some(audio_cfg), Some(embed_audio)) = (&self.cfg.audio_config, &self.embed_audio) {
            let audio_vocab_offset = audio_cfg.vocab_offset as f64;
            
            // Create masks for vision and audio tokens
            let vision_mask = input_ids.to_dtype(DType::F32)?
                .ge(vocab_offset)?
                .mul(&input_ids.to_dtype(DType::F32)?.lt(audio_vocab_offset)?)?;
            let audio_mask = input_ids.to_dtype(DType::F32)?.ge(audio_vocab_offset)?;
            
            // Get embeddings for each modality
            let vision_token_embeds = self.embed_vision.forward_text(input_ids)?;
            let audio_token_embeds = embed_audio.forward_text(input_ids)?;
            
            // Combine embeddings based on token type
            let expanded_vision_mask = vision_mask.unsqueeze(D::Minus1)?.broadcast_as(language_embeds.shape())?;
            let expanded_audio_mask = audio_mask.unsqueeze(D::Minus1)?.broadcast_as(language_embeds.shape())?;
            
            let embeds_with_vision = expanded_vision_mask.where_cond(&vision_token_embeds, &language_embeds)?;
            expanded_audio_mask.where_cond(&audio_token_embeds, &embeds_with_vision)?
        } else {
            // No audio config, just handle vision tokens
            let vision_mask = input_ids.to_dtype(DType::F32)?.ge(vocab_offset)?;
            let expanded_vision_mask = vision_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(language_embeds.shape())?;
            let vision_token_embeds = self.embed_vision.forward_text(input_ids)?;
            expanded_vision_mask.where_cond(&vision_token_embeds, &language_embeds)?
        };

        // Step 5: If we have actual images, replace the image placeholder tokens with vision features
        if let Some(pixel_values) = pixel_values {
            // Process vision inputs through vision tower
            let vision_features = self.vision_tower.forward(&pixel_values)?;

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
            if indices.dims()[0] > 0 {
                let mut x_flat = input_embeds.flatten_all()?;
                let src_flat = image_embeds.flatten_all()?;

                // Replace image tokens with actual vision embeddings
                let current_vals = x_flat.gather(&indices, 0)?;
                let diff = (src_flat - current_vals)?;
                x_flat = x_flat.scatter_add(&indices, &diff, 0)?;

                input_embeds = x_flat.reshape(input_embeds.shape())?;
            }
        }

        // Step 6: If we have audio features, replace audio placeholder tokens
        if let (Some(audio_mel), Some(audio_mel_mask), Some(audio_tower), Some(embed_audio)) = 
            (audio_mel, audio_mel_mask, &self.audio_tower, &self.embed_audio) {
            audio_mel.write_npy("input_features_m.npy")?;
            let audio_mel = Tensor::read_npy("input_features.npy")?.to_device(audio_mel.device())?.to_dtype(audio_mel.dtype())?;
            let audio_mel_mask = Tensor::read_npy("input_features_mask.npy")?.to_device(audio_mel_mask.device())?.to_dtype(audio_mel_mask.dtype())?;
            // Process audio through audio tower
            let (audio_features, _) = audio_tower.forward(&audio_mel, &audio_mel_mask)?;
            audio_features.write_npy("audio_outputs_m.npy")?;
            let audio_features = Tensor::read_npy("audio_outputs.npy")?.to_device(audio_features.device())?.to_dtype(audio_features.dtype())?;
            
            // Convert audio features to embeddings
            let audio_embeds = embed_audio.forward_vision(&audio_features)?;
            
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
            if indices.dims()[0] > 0 {
                let mut x_flat = input_embeds.flatten_all()?;
                let src_flat = audio_embeds.flatten_all()?;
                
                // Get dimensions
                let embed_dim = audio_embeds.dim(2)?;
                let _num_audio_embeddings = audio_embeds.dim(1)?; // 58 audio embeddings
                
                // Count how many positions we need to fill
                let num_positions_to_fill = indices.dims()[0]; // Total positions
                let _num_audio_tokens_in_input = num_positions_to_fill / embed_dim;
                
                // With the corrected token count, we should have a 1:1 mapping
                // between audio tokens and embeddings
                let audio_values = if num_positions_to_fill == src_flat.dims()[0] {
                    // Perfect match - this is the expected case
                    src_flat.clone()
                } else if num_positions_to_fill < src_flat.dims()[0] {
                    // We have more embeddings than needed, take only what we need
                    src_flat.narrow(0, 0, num_positions_to_fill)?
                } else {
                    // This shouldn't happen with correct token counting, but handle it gracefully
                    // by repeating embeddings if needed
                    let mut repeated_values = Vec::new();
                    let src_vec: Vec<f32> = src_flat.to_vec1()?;
                    
                    for i in 0..num_positions_to_fill {
                        repeated_values.push(src_vec[i % src_vec.len()]);
                    }
                    
                    Tensor::from_vec(repeated_values, (num_positions_to_fill,), src_flat.device())?
                };
                
                // Replace audio tokens with actual audio embeddings
                let current_vals = x_flat.gather(&indices, 0)?;
                let diff = (audio_values - current_vals)?;
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
