mod text;

use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use text::TextModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel,
    },
};

mod config;

pub use config::{Llama4Config, TextConfig};

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
        self.language_model.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.language_model.residual_tensors()
    }
}

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

impl AnyMoeBaseModelMixin for Llama4Model {}
