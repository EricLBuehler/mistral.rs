#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use config::Gemma3nConfig;
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
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
        let input_embeds = if let Some(pixel_values) = pixel_values {
            // Process vision inputs through vision tower
            let vision_features = self.vision_tower.forward(&pixel_values)?;
            
            // Convert vision features to embeddings using multimodal embedder
            self.embed_vision.forward_vision(&vision_features)?
        } else {
            // For text-only inputs, use the multimodal embedder's text path
            self.embed_vision.forward_text(input_ids)?
        };

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
