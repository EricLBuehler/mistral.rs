#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{Device, Result, Tensor, D};
use config::Gemma3Config;
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use mmproj::Gemma3MultiModalProjector;
use text::TextModel;

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    device_map::DeviceMapper,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, VisionModel,
    },
    AnyMoeConfig, AnyMoeExpertType,
};

pub mod config;
mod inputs_processor;
mod mmproj;
mod text;
pub(crate) use inputs_processor::Gemma3Processor;

use super::siglip::SiglipVisionTransformer;

pub struct Gemma3Model {
    language_model: TextModel,
    multi_modal_projector: Gemma3MultiModalProjector,
    vision_tower: SiglipVisionTransformer,
    cfg: Gemma3Config,
}

impl Gemma3Model {
    pub fn new(
        cfg: &Gemma3Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        assert!(cfg.image_token_index < cfg.text_config.vocab_size);
        Ok(Self {
            language_model: TextModel::new(
                &cfg.text_config,
                vb.pp("language_model"),
                is_gptx,
                normal_loading_metadata,
                attention_mechanism,
            )?,
            multi_modal_projector: Gemma3MultiModalProjector::new(
                cfg,
                vb.pp("multi_modal_projector"),
            )?,
            vision_tower: SiglipVisionTransformer::new(&cfg.vision_config, vb.pp("vision_tower"))?,
            cfg: cfg.clone(),
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut input_embeds = self.language_model.embed_tokens(input_ids)?;
        if let Some(pixel_values) = pixel_values {
            let vision_outputs = self.vision_tower.forward(&pixel_values, None, None)?;
            let image_features = self.multi_modal_projector.forward(&vision_outputs)?;

            let special_image_mask = input_ids
                .eq(self.cfg.image_token_index as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?;

            let current_vals = input_embeds.gather(&special_image_mask, 1)?;
            let delta = image_features.broadcast_sub(&current_vals)?;

            input_embeds = input_embeds.scatter_add(&special_image_mask, &delta, 1)?;
        };
        self.language_model.forward_embeds(
            input_ids,
            input_embeds,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
}

impl IsqModel for Gemma3Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        self.language_model.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.language_model.residual_tensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        self.language_model.imatrix_names()
    }
}

pub struct Gemma3SpecificArgs;

impl VisionModel for Gemma3Model {
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
        Box::new(Gemma3SpecificArgs)
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
    fn has_conv2d(&self) -> bool {
        // TODO
        false
    }
}

impl AnyMoeBaseModelMixin for Gemma3Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.language_model.get_mlps()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.language_model.get_mlps_mut()
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        self.language_model.create_anymoe_layers(
            additional_vbs,
            config,
            (prefix, mlp),
            layers,
            expert_type,
            gate_vb,
        )
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}
