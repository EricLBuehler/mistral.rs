mod text;

use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use text::TextModel;
use vision::Llama4VisionModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::linear_no_bias,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel, VisionModel,
    },
};

mod config;
mod vision;

pub use config::{Llama4Config, TextConfig};

struct Llama4MultiModalProjector {
    linear_1: Linear,
}

impl Llama4MultiModalProjector {
    fn new(cfg: &Llama4Config, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear_no_bias(
                cfg.vision_config.vision_output_dim,
                cfg.text_config.hidden_size,
                vb.pp("linear_1"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear_1.forward(xs)
    }
}
pub struct Llama4Model {
    language_model: TextModel,
    vision_model: Llama4VisionModel,
    multi_modal_projector: Llama4MultiModalProjector,
}

impl Llama4Model {
    pub fn new(
        cfg: &Llama4Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        // TODO: evaluate
        assert_eq!(attention_mechanism, AttentionImplementation::Eager);
        let vision_model = Llama4VisionModel::new(
            &cfg.vision_config,
            vb.pp("vision_model"),
            &normal_loading_metadata.real_device,
            &normal_loading_metadata.mapper.get_comm_for(0)?,
        )?;
        let language_model = TextModel::new(
            &cfg.text_config,
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let multi_modal_projector =
            Llama4MultiModalProjector::new(cfg, vb.pp("multi_modal_projector"))?;

        Ok(Self {
            language_model,
            vision_model,
            multi_modal_projector,
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
        self.language_model.forward(
            input_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
}

impl IsqModel for Llama4Model {
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
}

pub struct Llama4ModelSpecificArgs;

impl NormalModel for Llama4Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        self.forward(
            input_ids,
            None,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.language_model.cache_mut()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.language_model.config()
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn device(&self) -> &Device {
        self.language_model.device()
    }
    fn max_seq_len(&self) -> usize {
        self.language_model.max_seq_len()
    }
}

impl VisionModel for Llama4Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        model_specific_args: Box<dyn std::any::Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        let Llama4ModelSpecificArgs = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Llama4ModelSpecificArgs`");
        self.forward(
            input_ids,
            pixel_values,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.language_model.cache_mut()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.language_model.config()
    }
    fn has_conv2d(&self) -> bool {
        false
    }
    fn device(&self) -> &Device {
        self.language_model.device()
    }
    fn max_seq_len(&self) -> usize {
        self.language_model.max_seq_len()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(Llama4ModelSpecificArgs)
    }
}

impl AnyMoeBaseModelMixin for Llama4Model {}
