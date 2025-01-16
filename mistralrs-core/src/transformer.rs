#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use mistralrs_quant::QuantMethod;

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    device_map::DeviceMapper,
    layers::{CausalMasker, MatMul, RmsNorm},
    layers_masker::PastKvLenCache,
    paged_attention::ModelConfigMetadata,
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalModel,
    },
    AnyMoeConfig, AnyMoeExpertType,
};

pub(crate) trait Attention: Send + Sync {
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Option<Tensor>,
        position_ids: Option<&[usize]>,
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor>;

    fn get_tensors(&mut self) -> Vec<&mut Arc<dyn QuantMethod>>;
}

pub(crate) struct DecoderLayer {
    pub self_attn: Box<dyn Attention>,
    pub mlp: Box<dyn MlpLayer>,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: Option<RmsNorm>,
    pub pre_feedforward_layernorm: Option<RmsNorm>,
    pub post_feedforward_layernorm: Option<RmsNorm>,
}

impl DecoderLayer {
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Option<Tensor>,
        position_ids: Option<&[usize]>,
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let mut xs = xs.apply(&self.input_layernorm)?;
        xs = self.self_attn.forward(
            &xs,
            attention_mask,
            sliding_attention_mask,
            seqlen_offsets,
            start_offsets_kernel,
            position_ids,
            kv_cache,
            metadata,
            flash_params,
        )?;
        // Note that the "post-attention layer norm" in some models is applied
        // after residual summation, and as such, must be placed in the field
        // `pre_feedforward_layernorm`.
        if let Some(post_attention_layernorm) = &self.post_attention_layernorm {
            xs = xs.apply(post_attention_layernorm)?;
        }
        xs = (xs + residual)?;
        let residual = xs.clone();
        if let Some(pre_feedforward_layernorm) = &self.pre_feedforward_layernorm {
            xs = xs.apply(pre_feedforward_layernorm)?;
        }
        xs = self.mlp.forward(&xs)?;
        if let Some(post_feedforward_layernorm) = &self.post_feedforward_layernorm {
            xs = xs.apply(post_feedforward_layernorm)?;
        }
        residual + xs
    }
}

pub(crate) struct Model {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub lm_head: Arc<dyn QuantMethod>,
    pub hidden_size: Option<usize>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub use_two_attention_masks: bool,
    pub use_sliding_window_attention_mask: bool,
    pub sliding_window: Option<usize>,
    pub final_logit_softcapping: Option<f64>,
    pub cfg: ModelConfigMetadata,
}

impl Model {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Option<Tensor>,
        context_lens: Vec<(usize, usize)>,
        position_ids: Option<&[usize]>,
        mut metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        if let Some(hidden_size) = self.hidden_size {
            xs = (xs * (hidden_size as f64).sqrt())?;
        }
        let cache = &mut self.cache.normal().0;

        let (attention_mask, sliding_attention_mask) = if self.use_two_attention_masks {
            (
                CausalMasker.make_causal_mask_matrix(
                    input_ids,
                    &*cache,
                    xs.dtype(),
                    self.cfg.num_attn_heads,
                )?,
                CausalMasker.make_sliding_window_causal_mask_matrix(
                    input_ids,
                    &*cache,
                    self.sliding_window,
                    xs.dtype(),
                    self.cfg.num_attn_heads,
                )?,
            )
        } else {
            (
                if self.use_sliding_window_attention_mask {
                    CausalMasker.make_sliding_window_causal_mask_matrix(
                        input_ids,
                        metadata
                            .as_ref()
                            .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                            .unwrap_or(cache as &dyn PastKvLenCache),
                        self.sliding_window,
                        xs.dtype(),
                        self.cfg.num_attn_heads,
                    )?
                } else {
                    CausalMasker.make_causal_mask_matrix(
                        input_ids,
                        metadata
                            .as_ref()
                            .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                            .unwrap_or(cache as &dyn PastKvLenCache),
                        xs.dtype(),
                        self.cfg.num_attn_heads,
                    )?
                },
                None,
            )
        };

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                sliding_attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                start_offsets_kernel.clone(),
                position_ids,
                &mut cache[i],
                metadata
                    .as_mut()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), &mut **metadata)),
                flash_params,
            )?;
        }

        xs = xs.to_device(&self.device)?;
        xs = xs.apply(&self.norm)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }

        xs = MatMul.qmethod_matmul(&xs, &*self.lm_head)?;

        if let Some(final_logit_softcapping) = self.final_logit_softcapping {
            xs = (xs / final_logit_softcapping)?;
            xs = xs.tanh()?;
            xs = (xs * final_logit_softcapping)?;
        }

        extract_logits(&xs, context_lens)
    }
}

pub(crate) trait ModelWrapper {
    fn get_model(&self) -> &Model;
    fn get_model_mut(&mut self) -> &mut Model;
}

pub(crate) trait AutoIsqModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)>;
    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>>;
}

impl<T> IsqModel for T
where
    T: ModelWrapper + AutoIsqModel,
{
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        let model = self.get_model_mut();
        tensors.push((&mut model.lm_head, None));
        for (i, layer) in model.layers.iter_mut().enumerate() {
            for tensor in layer.self_attn.get_tensors() {
                tensors.push((tensor, Some(i)));
            }
            tensors.extend(
                layer
                    .mlp
                    .get_isq_layers()
                    .into_iter()
                    .map(|m| (m, Some(i)))
                    .collect::<Vec<_>>(),
            );
        }
        (tensors, &*model.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.residual_tensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        self.imatrix_names()
    }
}

pub(crate) trait AutoAnyMoeBaseModelMixin {
    fn create_anymoe_layers(
        &mut self,
        _additional_vbs: Vec<VarBuilder>,
        _config: AnyMoeConfig,
        (_prefix, _mlp): (String, String),
        _layers: Vec<usize>,
        _expert_type: AnyMoeExpertType,
        _gate_vb: Option<VarBuilder>,
    ) -> Result<()> {
        candle_core::bail!("Model does not support AnyMoE layers");
    }
}

impl<T> AnyMoeBaseModelMixin for T
where
    T: ModelWrapper + AutoAnyMoeBaseModelMixin,
{
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        let mut mlps = Vec::new();
        for layer in &self.get_model().layers {
            mlps.push(&*layer.mlp);
        }
        mlps
    }

    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        let mut mlps = Vec::new();
        for layer in &mut self.get_model_mut().layers {
            mlps.push(&mut layer.mlp);
        }
        mlps
    }

    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<VarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<VarBuilder>,
    ) -> Result<()> {
        self.create_anymoe_layers(
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

impl<T> NormalModel for T
where
    T: ModelWrapper + IsqModel + AnyMoeBaseModelMixin,
{
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.get_model().forward(
            input_ids,
            seqlen_offsets,
            Some(start_offsets_kernel),
            context_lens,
            Some(&position_ids),
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
        _start_offsets_kernel: Tensor,
        _start_offsets_kernel_full: Tensor,
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
        &self.get_model().cache
    }

    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.get_model_mut().cache
    }

    fn device(&self) -> &Device {
        &self.get_model().device
    }

    fn is_xlora(&self) -> bool {
        false
    }

    fn max_seq_len(&self) -> usize {
        self.get_model().max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.get_model().cfg
    }
}
