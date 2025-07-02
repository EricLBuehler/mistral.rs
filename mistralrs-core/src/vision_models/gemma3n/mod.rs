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
    utils::unvarbuilder::UnVarBuilder, vision_models::timm_models,
};

use self::multimodal_embedding::Gemma3nMultimodalEmbedder;

pub mod config;
mod inputs_processor;
mod multimodal_embedding;
mod text;
pub(crate) use inputs_processor::Gemma3nProcessor;

pub struct Gemma3nModel {
    language_model: TextModel,
    vision_tower: timm_models::VisionTower,
    embed_vision: Gemma3nMultimodalEmbedder,
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
        let vision_tower = timm_models::VisionTower::new(vb.pp("vision_tower").pp("timm_model"))?;
        
        // Initialize multimodal embedder
        let vision_cfg = cfg.vision_config.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Vision config is required for Gemma3n".to_string()))?;
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
            embed_vision,
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
    ) -> Result<Tensor> {
        // Get the vocab offset from vision config
        let vision_cfg = self.cfg.vision_config.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Vision config required".to_string()))?;
        let vocab_offset = vision_cfg.vocab_offset as f64;
        
        // Step 1: Get base language model embeddings
        let language_embeds = self.language_model.embed_tokens(input_ids)?;
        
        // Step 2: Create mask for vision tokens (tokens >= vocab_offset)
        // In Gemma3n, vision tokens are those with IDs >= vocab_offset
        let vision_mask = input_ids.to_dtype(DType::F32)?.ge(vocab_offset)?;
        let expanded_vision_mask = vision_mask
            .unsqueeze(D::Minus1)?
            .broadcast_as(language_embeds.shape())?;
        
        // Step 3: Get vision token embeddings using the multimodal embedder
        // These handle BOI, EOI, and placeholder image tokens
        let vision_token_embeds = self.embed_vision.forward_text(input_ids)?;
        
        // Step 4: Combine language and vision token embeddings
        // Use vision embeddings where we have vision tokens, language embeddings elsewhere
        let mut input_embeds = expanded_vision_mask
            .where_cond(&vision_token_embeds, &language_embeds)?;
        
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

pub struct Gemma3nSpecificArgs;

impl VisionModel for Gemma3nModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _model_specific_args: Box<dyn std::any::Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        self.forward(
            input_ids,
            pixel_values,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(Gemma3nSpecificArgs)
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
