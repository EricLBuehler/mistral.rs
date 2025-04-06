mod text;

use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use text::TextModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel, VisionModel,
    },
};

mod config;
mod inputs_processor;

pub use config::{Llama4Config, TextConfig};
pub use inputs_processor::Llama4Processor;

pub struct Llama4Model {
    language_model: TextModel,
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
        let language_model = TextModel::new(
            &cfg.text_config,
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;

        Ok(Self { language_model })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
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
        todo!()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        todo!()
    }
}

pub struct Llama4ModelSpecificArgs;

impl VisionModel for Llama4Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        self.forward(
            input_ids,
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
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(Llama4ModelSpecificArgs)
    }
    fn device(&self) -> &Device {
        self.language_model.device()
    }
    fn has_conv2d(&self) -> bool {
        true
    }
    fn max_seq_len(&self) -> usize {
        self.language_model.max_seq_len()
    }
}

impl AnyMoeBaseModelMixin for Llama4Model {}
