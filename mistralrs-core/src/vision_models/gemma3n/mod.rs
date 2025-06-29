#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{Device, Result, Tensor, D};
use config::Gemma3nConfig;
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};
use embed_vision::Gemma3nMultimodalEmbedder;
use text::TextModel;
use vision::Gemma3nVisionTower;

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

pub mod config;
mod embed_vision;
mod inputs_processor;
mod text;
mod vision;
pub(crate) use inputs_processor::Gemma3nProcessor;

pub struct Gemma3nModel {
    language_model: TextModel,
    embed_vision: Gemma3nMultimodalEmbedder,
    vision_tower: Gemma3nVisionTower,
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
        Ok(Self {
            embed_vision: Gemma3nMultimodalEmbedder::new(
                cfg,
                vb.pp("embed_vision")
                    .set_device(normal_loading_metadata.real_device.clone()),
            )?,
            vision_tower: Gemma3nVisionTower::new(
                &cfg.vision_config,
                vb.pp("vision_tower")
                    .pp("timm_model")
                    .set_device(normal_loading_metadata.real_device.clone()),
            )?,
            language_model: TextModel::new(
                &cfg.text_config,
                vb.pp("language_model"),
                is_gptx,
                normal_loading_metadata,
                attention_mechanism,
            )?,
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
        let mut input_embeds = self.language_model.embed_tokens(input_ids)?;
        
        if let Some(pixel_values) = pixel_values {
            // Get vision token mask - tokens >= embed_vision.vocab_offset
            let vision_offset = self.embed_vision.vocab_offset() as f64;
            let vision_mask = input_ids.ge(vision_offset)?;
            
            if vision_mask.sum_all()?.to_scalar::<u8>()? > 0 {
                // Process vision features through vision tower
                let dtype = self.vision_tower.dtype();
                let vision_outputs = self.vision_tower.forward(&pixel_values.to_dtype(dtype)?)?;
                
                // Get vision embeddings using soft embedding path
                let vision_embeds = self.embed_vision.forward(None, Some(&vision_outputs))?;
                
                // Flatten vision embeddings to match sequence dimension
                let (b, h, w, c) = vision_embeds.dims4()?;
                let num_vision_patches = h * w;
                let vision_embeds_flat = vision_embeds.reshape((b * num_vision_patches, c))?;
                
                // Count vision tokens in the input
                let num_vision_tokens = vision_mask.sum_all()?.to_scalar::<u8>()? as usize;
                
                // Vision embeddings should match the number of vision tokens
                // We only take the first num_vision_tokens embeddings
                let vision_embeds_to_use = vision_embeds_flat.narrow(0, 0, num_vision_tokens)?;
                
                // Create expanded vision mask for embedding dimension
                let vision_mask_expanded = vision_mask
                    .unsqueeze(D::Minus1)?
                    .broadcast_as(input_embeds.shape())?;
                
                // Get indices of vision tokens
                let mask_flat = vision_mask_expanded.flatten_all()?;
                let indices = mask_flat.nonzero()?.squeeze(1)?;
                
                // Scatter vision embeddings into input embeddings
                let mut x_flat = input_embeds.flatten_all()?;
                let current_vals = x_flat.gather(&indices, 0)?;
                let diff = (vision_embeds_to_use.flatten_all()? - current_vals)?;
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
        uvb.pp("model.embed_vision")
            .extend(self.embed_vision.residual_tensors());
        uvb.pp("model")
            .extend(self.language_model.residual_tensors());
        uvb.pp("model.vision_tower")
            .pp("timm_model")
            .extend(self.vision_tower.residual_tensors());
        
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
