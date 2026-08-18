use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Module, Result, Tensor, D};
use mistralrs_quant::{
    softcap, ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};

use super::config::{TextAttentionType, TextConfig};
use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, MlpLayer, MoeMlp},
    attention::{flash_backend_supports, AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    get_delta_from_lora_ab,
    layers::{
        embedding_with_legacy_tied_uqff, CausalMaskConfig, CausalMasker, MatMul, Mlp,
        RotaryEmbedding, Sdpa,
    },
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalCacheType,
        NormalLoadingMetadata,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

fn rms_norm_f32(xs: &Tensor, weight: Option<(&Tensor, bool)>, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs = xs.to_dtype(DType::F32)?;
    let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
    let inv_rms = (&variance + eps)?.recip()?.sqrt()?;
    let mut xs = xs.broadcast_mul(&inv_rms)?;
    if let Some((weight, centered)) = weight {
        let weight = weight.to_dtype(DType::F32)?;
        xs = if centered {
            xs.broadcast_mul(&(&weight + 1.0)?)?
        } else {
            xs.broadcast_mul(&weight)?
        };
    }
    xs.to_dtype(dtype)
}

fn normalize_input_embeddings(xs: &Tensor, eps: f64) -> Result<Tensor> {
    rms_norm_f32(xs, None, eps)
}

fn transform_output_logits(logits: &Tensor, multiplier: f64, softcap_value: f64) -> Result<Tensor> {
    let dtype = logits.dtype();
    softcap(&(logits * multiplier)?, softcap_value as f32)?.to_dtype(dtype)
}

#[derive(Debug, Clone)]
struct MuseRmsNorm {
    weight: Tensor,
    eps: f64,
    centered: bool,
}

impl MuseRmsNorm {
    fn new(size: usize, eps: f64, centered: bool, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
            centered,
        })
    }

    fn original_weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for MuseRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        rms_norm_f32(xs, Some((&self.weight, self.centered)), self.eps)
    }
}

fn cache_types(
    layer_types: &[TextAttentionType],
    sliding_window: usize,
    max_position_embeddings: usize,
) -> Vec<NormalCacheType> {
    layer_types
        .iter()
        .map(|layer_type| match layer_type {
            TextAttentionType::SlidingAttention => NormalCacheType::SlidingWindow {
                window: sliding_window,
            },
            TextAttentionType::FullAttention => NormalCacheType::Normal {
                max_seq_len: max_position_embeddings,
            },
        })
        .collect()
}

fn model_config_metadata(cfg: &TextConfig, world_size: usize) -> ModelConfigMetadata {
    ModelConfigMetadata {
        max_seq_len: cfg.max_position_embeddings,
        num_layers: cfg.num_hidden_layers,
        hidden_size: cfg.hidden_size,
        num_attn_heads: cfg.num_attention_heads / world_size,
        num_kv_heads: (cfg.num_key_value_heads / world_size).max(1),
        sliding_window: None,
        k_head_dim: cfg.head_dim,
        v_head_dim: cfg.head_dim,
        kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    gate_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    qk_scale_factor: f64,
    qk_norm_eps: f64,
    rotary_emb: Option<Arc<RotaryEmbedding>>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

struct AttentionForward<'a> {
    attention_mask: &'a AttentionMask,
    sliding_attention_mask: &'a AttentionMask,
    metadata: Option<((Tensor, Tensor), &'a PagedAttentionInputMetadata)>,
    flash_params: &'a FlashParams,
    positions: Option<Tensor>,
}

impl Attention {
    fn new(
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        cfg: &TextConfig,
        layer_type: TextAttentionType,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let query_size = cfg.num_attention_heads * cfg.head_dim;
        let kv_size = cfg.num_key_value_heads * cfg.head_dim;
        let q_proj = ColumnParallelLayer::new(
            hidden_size,
            query_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard =
            mistralrs_quant::compute_kv_shard(cfg.num_key_value_heads, cfg.head_dim, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_size,
            kv_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_size,
            kv_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            query_size,
            hidden_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            vb.pp("o_proj"),
        )?;
        let gate_proj = ColumnParallelLayer::new(
            hidden_size,
            query_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("gate_proj"),
        )?;
        let sliding_window =
            (layer_type == TextAttentionType::SlidingAttention).then_some(cfg.sliding_window);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            num_heads: cfg.num_attention_heads / comm.world_size(),
            num_kv_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
            head_dim: cfg.head_dim,
            qk_scale_factor: cfg.qk_scale_factor,
            qk_norm_eps: cfg.rms_norm_eps,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                )?,
                softcap: None,
                softmax_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                sliding_window,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        kv_cache: &mut KvCache,
        args: AttentionForward<'_>,
    ) -> Result<Tensor> {
        let AttentionForward {
            attention_mask,
            sliding_attention_mask,
            metadata,
            flash_params,
            positions,
        } = args;
        let (batch_size, query_len, _) = xs.dims3()?;
        let (q, k, v) =
            crate::ops::qkv_projections(xs, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        let (mut q, mut k, v) = if query_len == 1 {
            (
                q.reshape((batch_size, self.num_heads, query_len, self.head_dim))?,
                k.reshape((batch_size, self.num_kv_heads, query_len, self.head_dim))?,
                v.reshape((batch_size, self.num_kv_heads, query_len, self.head_dim))?,
            )
        } else {
            (
                q.reshape((batch_size, query_len, self.num_heads, self.head_dim))?
                    .transpose(1, 2)?,
                k.reshape((batch_size, query_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?,
                v.reshape((batch_size, query_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?,
            )
        };

        q = (rms_norm_f32(&q, None, self.qk_norm_eps)? * self.qk_scale_factor)?;
        k = rms_norm_f32(&k, None, self.qk_norm_eps)?;
        if let Some(rotary_emb) = &self.rotary_emb {
            let positions = positions
                .as_ref()
                .ok_or_else(|| candle_core::Error::msg("missing Muse-Glimmer RoPE positions"))?;
            (q, k) = rotary_emb.forward(&q, &k, positions)?;
        }

        let mask = if self.sdpa_params.sliding_window.is_some() {
            sliding_attention_mask
        } else {
            attention_mask
        };
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let mut attention_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(!matches!(mask, AttentionMask::None));
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    mask,
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        attention_output = if !matches!(mask, AttentionMask::None) {
            attention_output
                .transpose(1, 2)?
                .reshape((batch_size, query_len, ()))?
        } else {
            attention_output.reshape((batch_size, query_len, ()))?
        };
        let gate = candle_nn::ops::sigmoid(&self.gate_proj.forward(xs)?)?;
        self.o_proj.forward(&attention_output.broadcast_mul(&gate)?)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Box<dyn MlpLayer>,
    input_layernorm: MuseRmsNorm,
    post_attention_layernorm: MuseRmsNorm,
    pre_feedforward_layernorm: MuseRmsNorm,
    post_feedforward_layernorm: MuseRmsNorm,
}

struct DecoderLayerLoad<'a> {
    cfg: &'a TextConfig,
    mapper: &'a dyn DeviceMapper,
    layer_idx: usize,
    loading_isq: bool,
    paged_attn: Option<PagedAttention>,
    layer_type: TextAttentionType,
    comm: &'a Arc<mistralrs_quant::Comm>,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        vb: ShardedVarBuilder,
        args: DecoderLayerLoad<'_>,
    ) -> Result<Self> {
        let DecoderLayerLoad {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            layer_type,
            comm,
        } = args;
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            layer_type,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
            comm,
        )?;
        let mlp = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_activation,
            comm,
        )?;
        let norm = |name: &str, eps| {
            MuseRmsNorm::new(
                cfg.hidden_size,
                eps,
                true,
                mapper.set_device(layer_idx, vb.pp(name), false),
            )
        };
        Ok(Self {
            self_attn,
            mlp: Box::new(mlp),
            input_layernorm: norm("input_layernorm", cfg.rms_norm_eps)?,
            post_attention_layernorm: norm("post_attention_layernorm", cfg.post_norm_eps)?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm", cfg.rms_norm_eps)?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm", cfg.post_norm_eps)?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        kv_cache: &mut KvCache,
        args: AttentionForward<'_>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, kv_cache, args)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let xs = self.pre_feedforward_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_feedforward_layernorm.forward(&xs)?;
        residual + xs
    }
}

pub(super) struct TextModel {
    embed_tokens: Arc<dyn QuantMethod>,
    norm: MuseRmsNorm,
    layers: Vec<DecoderLayer>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    lm_head: Arc<dyn QuantMethod>,
    pub(super) cache: EitherCache,
    pub(super) cfg: ModelConfigMetadata,
    pub(super) device: Device,
    dtype: DType,
    pub(super) max_seq_len: usize,
    sliding_window: usize,
    output_multiplier: f64,
    final_logit_softcapping: f64,
}

impl TextModel {
    pub(super) fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        cfg.validate()?;
        let mapper = normal_loading_metadata.mapper;
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;

        let layer_types = cfg.layer_types()?;
        let layer_rope_theta = cfg.layer_rope_theta()?;
        let mut ropes = HashMap::new();
        for (layer_idx, theta) in layer_rope_theta.iter().copied().enumerate() {
            if theta == 0.0 {
                continue;
            }
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes
                .entry((device.location(), theta.to_bits()))
                .or_insert(Arc::new(RotaryEmbedding::new(
                    theta as f32,
                    cfg.head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb.dtype(),
                )?));
        }

        let vb_layers = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = (layer_rope_theta[layer_idx] != 0.0).then(|| {
                ropes
                    .get(&(device.location(), layer_rope_theta[layer_idx].to_bits()))
                    .expect("missing Muse-Glimmer RoPE")
                    .clone()
            });
            let paged_attn = match attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                rotary_emb,
                vb_layers.pp(layer_idx),
                DecoderLayerLoad {
                    cfg,
                    mapper: &*mapper,
                    layer_idx,
                    loading_isq: normal_loading_metadata.loading_isq,
                    paged_attn,
                    layer_type: layer_types[layer_idx],
                    comm: &comm,
                },
            )
        })?;
        let norm = MuseRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            false,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        };
        let cache_types = cache_types(
            &layer_types,
            cfg.sliding_window,
            cfg.max_position_embeddings,
        );
        let world_size = mapper.get_comm_for(0)?.world_size();
        Ok(Self {
            embed_tokens,
            norm,
            layers,
            lm_head,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            cfg: model_config_metadata(cfg, world_size),
            device: normal_loading_metadata.real_device,
            dtype: vb.dtype(),
            mapper,
            max_seq_len: cfg.max_position_embeddings,
            sliding_window: cfg.sliding_window,
            output_multiplier: cfg.output_multiplier,
            final_logit_softcapping: cfg.final_logit_softcapping,
        })
    }

    pub(super) fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        normalize_input_embeddings(
            &self.embed_tokens.embedding_forward(input_ids, self.dtype)?,
            self.norm.eps,
        )
    }

    pub(super) fn supports_packed_prefill(&self) -> bool {
        self.layers
            .iter()
            .all(|layer| flash_backend_supports(layer.self_attn.head_dim, false))
    }

    pub(super) fn forward_embeds(
        &self,
        input_ids: &Tensor,
        mut xs: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;
        let flash_params = ctx.flash_params().clone();
        let mask_cache = ctx.mask_cache(cache);
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig::default(),
        )?;
        let sliding_attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig {
                sliding_window: Some(self.sliding_window),
                ..Default::default()
            },
        )?;
        let keep_mask = ctx.is_first_prompt_chunk();
        let attention_mask = if keep_mask {
            attention_mask
        } else {
            AttentionMask::None
        };
        let sliding_attention_mask = if keep_mask {
            sliding_attention_mask
        } else {
            AttentionMask::None
        };
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        let sliding_attention_mask = DeviceMappedMask::new(sliding_attention_mask, &*self.mapper)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, layer_idx)?;
            let positions = if layer.self_attn.rotary_emb.is_some() {
                ctx.text_positions(xs.device(), xs.dim(1)?)?.cloned()
            } else {
                None
            };
            xs = layer.forward(
                &xs,
                &mut cache[layer_idx],
                AttentionForward {
                    attention_mask: &attention_mask.get(xs.device()),
                    sliding_attention_mask: &sliding_attention_mask.get(xs.device()),
                    metadata: ctx.paged_layer(layer_idx),
                    flash_params: &flash_params,
                    positions,
                },
            )?;
        }
        let xs = xs.to_device(&self.device)?.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        transform_output_logits(
            &self.lm_head.forward(&xs)?,
            self.output_multiplier,
            self.final_logit_softcapping,
        )
    }

    pub(super) fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let model = uvb.pp("model").pp("language_model");
        model.pp("embed_tokens").add(&self.embed_tokens);
        model
            .pp("norm")
            .add_tensor("weight", self.norm.original_weight().clone());
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_uvb = model.pp("layers").pp(layer_idx);
            for (name, norm) in [
                ("input_layernorm", &layer.input_layernorm),
                ("post_attention_layernorm", &layer.post_attention_layernorm),
                (
                    "pre_feedforward_layernorm",
                    &layer.pre_feedforward_layernorm,
                ),
                (
                    "post_feedforward_layernorm",
                    &layer.post_feedforward_layernorm,
                ),
            ] {
                layer_uvb
                    .pp(name)
                    .add_tensor("weight", norm.original_weight().clone());
            }
        }
        uvb.to_safetensors()
    }
}

impl IsqModel for TextModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.residual_tensors()
    }
}

impl AnyMoeBaseModelMixin for TextModel {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.layers.iter().map(|layer| &*layer.mlp).collect()
    }

    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.layers.iter_mut().map(|layer| &mut layer.mlp).collect()
    }

    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        mut layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        if layers.is_empty() {
            layers = (0..self.layers.len()).collect();
        }
        let mut experts = (0..layers.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for vb in additional_vbs {
            let vb = vb.pp(&prefix);
            for (expert_row, &layer_idx) in experts.iter_mut().zip(&layers) {
                let base = &self.layers[layer_idx].mlp;
                let hidden_size = base.get_params()[0];
                let intermediate_size = base.get_params()[1];
                match expert_type {
                    AnyMoeExpertType::FineTuned => {
                        let (dtype, device) = base.dtype_device();
                        expert_row.push(Box::new(Mlp::replicate(
                            base.get_params(),
                            vb.pp(layer_idx)
                                .pp(&mlp)
                                .set_dtype(dtype)
                                .set_device(device),
                            base.hidden_act(),
                            &self.mapper.get_comm_for(layer_idx)?,
                        )?) as Box<dyn MlpLayer>);
                    }
                    AnyMoeExpertType::LoraAdapter {
                        rank,
                        alpha,
                        ref target_modules,
                    } => {
                        let vb_mlp = vb.pp(layer_idx).pp(&mlp);
                        let gate_proj_delta = if target_modules.contains(&"gate_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "gate_proj"
                            ))
                        } else {
                            None
                        };
                        let up_proj_delta = if target_modules.contains(&"up_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "up_proj"
                            ))
                        } else {
                            None
                        };
                        let down_proj_delta = if target_modules.contains(&"down_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (intermediate_size, hidden_size),
                                "down_proj"
                            ))
                        } else {
                            None
                        };
                        expert_row.push(base.new_added_delta(vec![
                            gate_proj_delta,
                            up_proj_delta,
                            down_proj_delta,
                        ])?);
                    }
                }
            }
        }
        for (layer_idx, added) in layers.into_iter().zip(experts) {
            let mut all = vec![self.layers[layer_idx].mlp.clone()];
            all.extend(added);
            let (dtype, device) = self.layers[layer_idx].mlp.dtype_device();
            self.layers[layer_idx].mlp = Box::new(MoeMlp::new(
                all,
                config.clone(),
                dtype,
                &device,
                layer_idx,
                gate_vb.as_ref(),
            )?);
        }
        Ok(())
    }

    fn amoe_supported(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_fixtures_match_transformers_ordering() -> Result<()> {
        let xs = Tensor::new(&[[1f32, -2., 3., -4.]], &Device::Cpu)?;
        let weight = Tensor::new(&[0.25f32, -0.5, 1., -0.75], &Device::Cpu)?;
        let scaleless = rms_norm_f32(&xs, None, 1e-5)?.to_vec2::<f32>()?;
        let centered = rms_norm_f32(&xs, Some((&weight, true)), 1e-5)?.to_vec2::<f32>()?;
        let scaled = rms_norm_f32(&xs, Some((&weight, false)), 1e-5)?.to_vec2::<f32>()?;
        let expected_scaleless = [0.365_148_13, -0.730_296_25, 1.095_444_4, -1.460_592_5];
        let expected_centered = [0.456_435_17, -0.365_148_13, 2.190_889, -0.365_148_13];
        let expected_scaled = [0.091_287_03, 0.365_148_13, 1.095_444_4, 1.095_444_4];
        for (actual, expected) in [
            (&scaleless[0], expected_scaleless),
            (&centered[0], expected_centered),
            (&scaled[0], expected_scaled),
        ] {
            assert!(actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1e-6));
        }
        Ok(())
    }

    #[test]
    fn cache_layout_tracks_full_and_sliding_layers() {
        let types = cache_types(
            &[
                TextAttentionType::SlidingAttention,
                TextAttentionType::FullAttention,
            ],
            2048,
            131_072,
        );
        assert!(matches!(
            types[0],
            NormalCacheType::SlidingWindow { window: 2048 }
        ));
        assert!(matches!(
            types[1],
            NormalCacheType::Normal {
                max_seq_len: 131_072
            }
        ));
    }

    #[test]
    fn mixed_attention_metadata_preserves_full_layer_context() {
        let config: TextConfig = serde_json::from_str("{}").unwrap();
        let metadata = model_config_metadata(&config, 1);
        assert_eq!(metadata.sliding_window, None);
        assert_eq!(metadata.max_seq_len, 131_072);
    }
}
