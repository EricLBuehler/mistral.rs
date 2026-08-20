#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

use candle_core::{DType, Device, Module, Result, Tensor, D};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};

use super::{
    config::{LayerType, TextConfig},
    mtp::Qwen3_5MtpHead,
    packed_gdn::{forward_packed_gdn, packed_gdn_query_lens},
};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    gdn::{
        GatedDeltaNet, GdnConfig, GdnForwardStash, GdnInputProjectionKind, GdnLayerCache,
        GdnVHeadLayout,
    },
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    },
    layers::{self, CausalMasker, GemmaRmsNorm, Mlp, Qwen3VLRotaryEmbedding, Sdpa},
    layers_masker::{CausalMaskConfig, PastKvLenCache},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, ForwardMaskCache, IsqModel, KvCache, ModelForwardContext,
        NormalLoadingMetadata, NormalModel, RecurrentBatchKind,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

impl GdnConfig for TextConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim
    }
    fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim
    }
    fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim
    }
    fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads
    }
    fn quantization_config(&self) -> &Option<QuantizedConfig> {
        &self.quantization_config
    }
    fn v_head_layout(&self) -> GdnVHeadLayout {
        self.gdn_v_head_layout
    }
}

// ====================== Full Attention layer with MRoPE ======================

#[allow(dead_code)]
pub(super) struct FullAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    pub(super) q_norm: GemmaRmsNorm,
    pub(super) k_norm: GemmaRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pub(super) rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl FullAttention {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let vb_sa = mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq);
        let vb_sa_norms = mapper.set_device(layer_idx, vb.pp("self_attn"), false);
        Self::load_with(vb_sa, vb_sa_norms, cfg, rotary_emb, paged_attn, comm)
    }

    /// `vb_sa` and `vb_sa_norms` are already placed on the layer's device.
    pub(super) fn load_with(
        vb_sa: ShardedVarBuilder,
        vb_sa_norms: ShardedVarBuilder,
        cfg: &TextConfig,
        rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // q_proj outputs num_heads * head_dim * 2 (doubled for gate)
        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            num_heads * head_dim * 2,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("o_proj"),
        )?;

        let q_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_sa_norms.pp("q_norm"))?;
        let k_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_sa_norms.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(num_kv_heads, num_heads, comm)?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: Option<&mut KvCache>,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let (q_gate, k, v) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        // Split q_gate into q and gate
        let q_gate = q_gate.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let gate = gate.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        // Reshape to (batch, heads, seq, head_dim)
        let (mut q, mut k, v) = if seq_len != 1 {
            let q = q.transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let cos_sin = &(
            cos_sin.0.to_device(q.device())?,
            cos_sin.1.to_device(q.device())?,
        );
        (q, k) = self.rotary_emb.forward_qk_norm(
            cos_sin,
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
        )?;

        // Standard attention
        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(!matches!(attention_mask, AttentionMask::None));
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let kv_cache = kv_cache.ok_or_else(|| {
                    candle_core::Error::msg("full attention without paged cache needs a KV cache")
                })?;
                let (cache_k, cache_v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q,
                    &cache_k,
                    &cache_v,
                    attention_mask,
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        y = if !matches!(attention_mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        // Apply output gate: y = y * sigmoid(gate)
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        y = y.broadcast_mul(&gate)?;

        let res = self.o_proj.forward(&y)?;
        Ok(res)
    }
}

// ====================== Decoder Layer ======================

pub(super) enum LayerImpl {
    FullAttention(FullAttention),
    LinearAttention(GatedDeltaNet),
}

pub(super) struct DecoderLayer {
    pub(super) layer_impl: LayerImpl,
    pub(super) input_layernorm: GemmaRmsNorm,
    pub(super) post_attention_layernorm: GemmaRmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    /// A full-attention block; `vb_quant` places quantizable projections (ISQ-aware) and `vb_plain`
    /// places norms, both already on the layer's device.
    pub(super) fn load_full_attention(
        vb_quant: ShardedVarBuilder,
        vb_plain: ShardedVarBuilder,
        cfg: &TextConfig,
        rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let attn = FullAttention::load_with(
            vb_quant.pp("self_attn"),
            vb_plain.pp("self_attn"),
            cfg,
            rotary_emb,
            paged_attn,
            comm,
        )?;
        let input_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_plain.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_plain.pp("post_attention_layernorm"),
        )?;
        let mlp = Mlp::new(
            vb_quant.pp("mlp"),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )?;
        Ok(Self {
            layer_impl: LayerImpl::FullAttention(attn),
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    pub(super) fn rotary_emb(&self) -> Option<&Arc<Qwen3VLRotaryEmbedding>> {
        match &self.layer_impl {
            LayerImpl::FullAttention(attn) => Some(&attn.rotary_emb),
            LayerImpl::LinearAttention(_) => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_attention(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: Option<&mut KvCache>,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let attn = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => attn,
            _ => candle_core::bail!("Expected full attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let attn_out = attn.forward(
            &x,
            attention_mask,
            cos_sin,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.mlp.forward(&normed)?;
        ffn_out + residual
    }

    fn forward_linear_with_stash(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
        packed_query_lens: Option<&[usize]>,
        stash_out: Option<&mut Option<GdnForwardStash>>,
    ) -> Result<Tensor> {
        let gdn = match &self.layer_impl {
            LayerImpl::LinearAttention(gdn) => gdn,
            _ => candle_core::bail!("Expected linear attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let gdn_out = if let Some(query_lens) = packed_query_lens {
            forward_packed_gdn(gdn, &x, cache, batch_kind, query_lens)?
        } else {
            gdn.forward_with_stash(&x, cache, batch_kind, stash_out)?
        };
        let x = (gdn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.mlp.forward(&normed)?;
        ffn_out + residual
    }
}

// ====================== Text Model ======================

#[derive(Clone, Copy)]
enum TextWeightPrefix {
    LanguageModelModel,
    ModelLanguageModel,
    Model,
}

/// Target activations captured for the MTP proposer: hidden states after the final norm and the
/// MRoPE position ids they were computed at, `[b, rows, hidden]` / `[3, b, rows]`.
#[derive(Clone)]
pub(super) struct SpecCapture {
    pub(super) hidden: Tensor,
    pub(super) positions: Tensor,
}

/// Per-GDN-layer inputs and pre-forward states of the last multi-token decode, so a rejected tail
/// can be undone by replaying only the accepted prefix.
#[derive(Clone)]
pub(super) struct GdnReplayStash {
    pub(super) slots: Vec<u32>,
    pub(super) layers: Vec<GdnLayerStash>,
}

#[derive(Clone)]
pub(super) struct GdnLayerStash {
    pub(super) layer_idx: usize,
    pub(super) projected: GdnForwardStash,
    pub(super) conv_state: Tensor,
    pub(super) recurrent_state: Tensor,
}

/// Snapshot of the proposer-facing outputs of one target forward (see `SpeculativeGraphState`).
#[derive(Clone)]
pub(super) struct SpecGraphState {
    spec_capture: Option<SpecCapture>,
    full_capture: Option<SpecCapture>,
    gdn_stash: Option<GdnReplayStash>,
}

impl crate::speculative::SpeculativeGraphState for SpecGraphState {
    fn tensors(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for capture in [&self.spec_capture, &self.full_capture]
            .into_iter()
            .flatten()
        {
            out.push(capture.hidden.clone());
            out.push(capture.positions.clone());
        }
        if let Some(stash) = &self.gdn_stash {
            for layer in &stash.layers {
                out.push(layer.projected.mixed_qkv.clone());
                out.push(layer.projected.b.clone());
                out.push(layer.projected.a.clone());
                out.push(layer.conv_state.clone());
                out.push(layer.recurrent_state.clone());
            }
        }
        out
    }

    fn with_tensors(
        &self,
        tensors: Vec<Tensor>,
    ) -> Result<Box<dyn crate::speculative::SpeculativeGraphState>> {
        let mut tensors = tensors.into_iter();
        let mut next = || {
            tensors.next().ok_or_else(|| {
                candle_core::Error::msg("speculative graph state tensor list is short")
            })
        };
        let mut state = self.clone();
        for capture in [&mut state.spec_capture, &mut state.full_capture]
            .into_iter()
            .flatten()
        {
            capture.hidden = next()?;
            capture.positions = next()?;
        }
        if let Some(stash) = state.gdn_stash.as_mut() {
            for layer in stash.layers.iter_mut() {
                layer.projected.mixed_qkv = next()?;
                layer.projected.b = next()?;
                layer.projected.a = next()?;
                layer.conv_state = next()?;
                layer.recurrent_state = next()?;
            }
        }
        Ok(Box::new(state))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct Qwen3_5TextModel {
    embed_tokens: Arc<dyn QuantMethod>,
    pub(super) norm: GemmaRmsNorm,
    layers: Vec<DecoderLayer>,
    pub(super) layer_types: Vec<LayerType>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    lm_head: Arc<dyn QuantMethod>,
    pub(super) cache: EitherCache,
    pub(super) cfg: ModelConfigMetadata,
    pub(super) device: Device,
    pub(super) dtype: DType,
    pub(super) max_seq_len: usize,
    weight_prefix: TextWeightPrefix,
    pub(super) mtp: Option<Qwen3_5MtpHead>,
    store_spec_hidden: AtomicBool,
    // Rows the logits were reduced to (decode: all rows, prompt: the last one)
    last_spec_capture: Mutex<Option<SpecCapture>>,
    // Every row of the last forward, so a proposer can catch up over a prompt chunk
    last_full_capture: Mutex<Option<SpecCapture>>,
    gdn_replay_stash: Mutex<Option<GdnReplayStash>>,
}

impl Qwen3_5TextModel {
    pub fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        tie: bool,
        mtp: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        cfg.validate()?;
        let mtp = mtp
            .then(|| {
                Qwen3_5MtpHead::load(
                    vb.clone(),
                    cfg,
                    &*normal_loading_metadata.mapper,
                    &normal_loading_metadata,
                    &attention_mechanism,
                )
            })
            .transpose()?;
        let mapper = normal_loading_metadata.mapper;
        let (vb_m, weight_prefix) =
            if layers::contains_tensor_or_uqff(&vb, "language_model.model.embed_tokens.weight") {
                (
                    vb.pp("language_model").pp("model"),
                    TextWeightPrefix::LanguageModelModel,
                )
            } else if layers::contains_tensor_or_uqff(
                &vb,
                "model.language_model.embed_tokens.weight",
            ) {
                (
                    vb.pp("model").pp("language_model"),
                    TextWeightPrefix::ModelLanguageModel,
                )
            } else {
                (vb.pp("model"), TextWeightPrefix::Model)
            };

        let embed_tokens = layers::embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            tie.then(|| {
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;

        let layer_types = cfg.layer_types();

        // Create MRoPE embeddings (one per device, using rot_dim not head_dim)
        let rot_dim = cfg.rot_dim();
        let mut ropes = HashMap::new();
        for (layer_idx, layer_type) in layer_types.iter().enumerate() {
            if *layer_type != LayerType::FullAttention {
                continue;
            }
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.entry(device.location()).or_insert_with(|| {
                Arc::new(
                    Qwen3VLRotaryEmbedding::new(
                        cfg.rope_theta() as f32,
                        rot_dim,
                        device,
                        cfg.mrope_section().to_vec(),
                    )
                    .expect("Failed to create rotary embedding"),
                )
            });
        }

        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let comm = mapper.get_comm_for(layer_idx)?;

            let layer_impl = match layer_types[layer_idx] {
                LayerType::FullAttention => {
                    let device = mapper
                        .device_for(layer_idx, false)
                        .unwrap_or(&normal_loading_metadata.real_device);
                    let rotary_emb = ropes
                        .get(&device.location())
                        .expect("No RoPE for device location!")
                        .clone();
                    let paged_attn = match &attention_mechanism {
                        AttentionImplementation::Eager => None,
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(cfg.head_dim, device, None)?)
                        }
                    };
                    LayerImpl::FullAttention(FullAttention::load(
                        vb_l.pp(layer_idx),
                        cfg,
                        &*mapper,
                        layer_idx,
                        normal_loading_metadata.loading_isq,
                        rotary_emb,
                        paged_attn,
                        &comm,
                    )?)
                }
                LayerType::LinearAttention => LayerImpl::LinearAttention(GatedDeltaNet::load(
                    vb_l.pp(layer_idx),
                    cfg as &dyn GdnConfig,
                    &*mapper,
                    layer_idx,
                    normal_loading_metadata.loading_isq,
                    &comm,
                    GdnInputProjectionKind::Split,
                )?),
            };

            let input_layernorm = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb_l.pp(layer_idx).pp("input_layernorm"), false),
            )?;
            let post_attention_layernorm = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(
                    layer_idx,
                    vb_l.pp(layer_idx).pp("post_attention_layernorm"),
                    false,
                ),
            )?;

            let mlp = Mlp::new(
                mapper.set_device(
                    layer_idx,
                    vb_l.pp(layer_idx).pp("mlp"),
                    normal_loading_metadata.loading_isq,
                ),
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                &comm,
            )?;

            Ok(DecoderLayer {
                layer_impl,
                input_layernorm,
                post_attention_layernorm,
                mlp,
            })
        })?;

        let norm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = if !tie {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            embed_tokens.clone()
        };

        // Create pipeline hybrid cache
        let pipeline_layer_types: Vec<HybridLayerType> = layer_types
            .iter()
            .map(|lt| match lt {
                LayerType::FullAttention => HybridLayerType::Attention,
                LayerType::LinearAttention => HybridLayerType::Recurrent,
            })
            .collect();

        let hybrid_cache_config = HybridCacheConfig {
            layer_types: pipeline_layer_types,
            max_seq_len: cfg.max_position_embeddings,
            recurrent: RecurrentLayerConfig {
                conv_dim: cfg.linear_conv_dim(),
                conv_width: cfg.linear_conv_kernel_dim,
                state_dims: vec![
                    cfg.linear_num_value_heads,
                    cfg.linear_key_head_dim,
                    cfg.linear_value_head_dim,
                ],
                recurrent_dtype: Some(DType::F32),
            },
        };
        let layer_devices = (0..hybrid_cache_config.layer_types.len())
            .map(|layer_idx| {
                mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device)
                    .clone()
            })
            .collect::<Vec<_>>();

        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new(hybrid_cache_config, vb_m.dtype(), &layer_devices).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to create hybrid cache: {}", e))
            })?,
        ));

        Ok(Self {
            embed_tokens,
            norm,
            layers,
            layer_types: layer_types.clone(),
            lm_head,
            cache: EitherCache::Hybrid(pipeline_cache),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers + cfg.mtp_layers(mtp.is_some()),
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            device: normal_loading_metadata.real_device.clone(),
            dtype: vb.dtype(),
            mapper,
            weight_prefix,
            mtp,
            store_spec_hidden: AtomicBool::new(false),
            last_spec_capture: Mutex::new(None),
            last_full_capture: Mutex::new(None),
            gdn_replay_stash: Mutex::new(None),
        })
    }

    pub(super) fn set_store_spec_hidden(&self, store: bool) {
        self.store_spec_hidden.store(store, Ordering::Relaxed);
        if !store {
            *self
                .last_spec_capture
                .lock()
                .expect("spec capture poisoned") = None;
            *self
                .last_full_capture
                .lock()
                .expect("spec capture poisoned") = None;
            *self.gdn_replay_stash.lock().expect("gdn stash poisoned") = None;
        }
    }

    /// Roll the recurrent state of forward-batch row `batch_idx` back to what it was after the first
    /// `keep_rows` tokens of the last multi-token decode step.
    pub(super) fn replay_recurrent_prefix(&self, batch_idx: usize, keep_rows: usize) -> Result<()> {
        let stash_guard = self.gdn_replay_stash.lock().expect("gdn stash poisoned");
        let Some(stash) = stash_guard.as_ref() else {
            candle_core::bail!("no GDN replay stash for speculative rollback");
        };
        let slot = *stash.slots.get(batch_idx).ok_or_else(|| {
            candle_core::Error::msg(format!("GDN replay stash has no batch row {batch_idx}"))
        })?;
        let mut snapshots = Vec::with_capacity(stash.layers.len());
        for layer in &stash.layers {
            let gdn = match &self.layers[layer.layer_idx].layer_impl {
                LayerImpl::LinearAttention(gdn) => gdn,
                LayerImpl::FullAttention(_) => {
                    candle_core::bail!("GDN replay stash points at a full-attention layer")
                }
            };
            let mut cache = GdnLayerCache {
                conv_state: layer.conv_state.narrow(0, batch_idx, 1)?.contiguous()?,
                recurrent_state: layer
                    .recurrent_state
                    .narrow(0, batch_idx, 1)?
                    .contiguous()?,
                slots: None,
            };
            gdn.advance_state_from_stash(&layer.projected, batch_idx, keep_rows, &mut cache)?;
            snapshots.push(crate::kv_cache::RecurrentStateSnapshot {
                conv_state: cache.conv_state,
                recurrent_state: cache.recurrent_state,
            });
        }
        drop(stash_guard);
        self.cache
            .hybrid()
            .restore_recurrent_state(slot as usize, &snapshots)
    }

    pub(super) fn clear_gdn_replay_stash(&self) {
        *self.gdn_replay_stash.lock().expect("gdn stash poisoned") = None;
    }

    pub(super) fn take_spec_graph_state(&self) -> Option<SpecGraphState> {
        if !self.store_spec_hidden.load(Ordering::Relaxed) {
            return None;
        }
        Some(SpecGraphState {
            spec_capture: self
                .last_spec_capture
                .lock()
                .expect("spec capture poisoned")
                .take(),
            full_capture: self
                .last_full_capture
                .lock()
                .expect("spec capture poisoned")
                .take(),
            gdn_stash: self
                .gdn_replay_stash
                .lock()
                .expect("gdn stash poisoned")
                .take(),
        })
    }

    pub(super) fn install_spec_graph_state(&self, state: &SpecGraphState) {
        *self
            .last_spec_capture
            .lock()
            .expect("spec capture poisoned") = state.spec_capture.clone();
        *self
            .last_full_capture
            .lock()
            .expect("spec capture poisoned") = state.full_capture.clone();
        let mut gdn_stash = state.gdn_stash.clone();
        if let Some(stash) = gdn_stash.as_mut() {
            // A replayed graph carries the slots of the step it was captured on; the rows belong to
            // whatever sequences occupy the batch now
            if let Some(slots) = self.cache.hybrid().state_indices_host() {
                stash.slots = slots.to_vec();
            }
        }
        *self.gdn_replay_stash.lock().expect("gdn stash poisoned") = gdn_stash;
    }

    pub(super) fn last_spec_capture(&self) -> Option<SpecCapture> {
        self.last_spec_capture
            .lock()
            .expect("spec capture poisoned")
            .clone()
    }

    pub(super) fn last_full_capture(&self) -> Option<SpecCapture> {
        self.last_full_capture
            .lock()
            .expect("spec capture poisoned")
            .clone()
    }

    pub(super) fn lm_head(&self) -> &Arc<dyn QuantMethod> {
        &self.lm_head
    }

    /// Paged KV of the MTP head's own attention layer.
    pub(super) fn paged_kv_layers(&self) -> Vec<bool> {
        let mut layers = self
            .layer_types
            .iter()
            .map(|ty| *ty == LayerType::FullAttention)
            .collect::<Vec<_>>();
        layers.extend(std::iter::repeat_n(true, usize::from(self.mtp.is_some())));
        layers
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.embedding_forward(input_ids, self.dtype)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        mut xs: Tensor,
        attention_mask: &AttentionMask,
        position_ids: &Tensor,
        _seqlen_offsets: &[usize],
        ctx: &ModelForwardContext<'_>,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let mut hybrid_cache = self.cache.hybrid();
        let recurrent_metadata = ctx.recurrent_metadata().cloned();
        let has_linear_attention = self
            .layer_types
            .iter()
            .any(|lt| matches!(lt, LayerType::LinearAttention));
        if has_linear_attention && recurrent_metadata.is_none() {
            candle_core::bail!(
                "Hybrid recurrent metadata is required for linear-attention layers."
            );
        }
        let packed_query_lens = if has_linear_attention {
            packed_gdn_query_lens(&xs, ctx)?
        } else {
            None
        };

        // Compute MRoPE cos/sin using first full-attention layer's rotary embedding
        let cos_sin = {
            let first_attn_idx = self
                .layer_types
                .iter()
                .position(|lt| *lt == LayerType::FullAttention)
                .expect("No full attention layer found");
            match &self.layers[first_attn_idx].layer_impl {
                LayerImpl::FullAttention(attn) => {
                    attn.rotary_emb.compute_cos_sin(position_ids, xs.dtype())?
                }
                _ => unreachable!(),
            }
        };

        let attention_mask = DeviceMappedMask::new(attention_mask.clone(), &*self.mapper)?;

        // A multi-token decode under an attached proposer is a speculative verify; keep what a
        // rejected tail needs to be replayed away.
        let stash_gdn = self.store_spec_hidden.load(Ordering::Relaxed)
            && xs.dim(1)? > 1
            && ctx.paged_input_metadata().is_some_and(|meta| {
                !meta.is_first_prompt_chunk && meta.num_cached_tokens.is_none()
            });
        let mut gdn_stash = stash_gdn.then(|| GdnReplayStash {
            slots: recurrent_metadata
                .as_ref()
                .and_then(|meta| meta.state_indices_host())
                .map(|slots| slots.to_vec())
                .unwrap_or_default(),
            layers: Vec::new(),
        });

        // Precompute deepstack index tensors once to avoid repeated CPU-GPU syncs
        let deepstack_indices = if let Some(visual_pos_masks) = visual_pos_masks {
            let mask_flat: Vec<f32> = visual_pos_masks
                .to_device(&self.device)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let indices: Vec<u32> = mask_flat
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 0.0)
                .map(|(i, _)| i as u32)
                .collect();
            if indices.is_empty() {
                None
            } else {
                let hidden = xs.dim(candle_core::D::Minus1)?;
                let n = indices.len();
                let idx = Tensor::from_vec(indices, (n,), &self.device)?;
                let idx_expanded = idx.unsqueeze(1)?.repeat((1, hidden))?;
                Some((idx, idx_expanded))
            }
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;

            match &self.layer_types[i] {
                LayerType::FullAttention => {
                    if let Some(HybridLayerCache::Attention(kv_cache)) = hybrid_cache.get_mut(i) {
                        xs = layer.forward_attention(
                            &xs,
                            &attention_mask.get(xs.device()),
                            &cos_sin,
                            Some(kv_cache),
                            ctx.paged_layer(i),
                            ctx.flash_params(),
                        )?;
                    }
                }
                LayerType::LinearAttention => {
                    let recurrent_metadata = recurrent_metadata.as_ref().expect(
                        "checked above: linear-attention layers require recurrent metadata",
                    );
                    let indices = hybrid_cache.state_indices_for_layer(i)?.ok_or_else(|| {
                        candle_core::Error::msg(format!(
                            "Hybrid cache layer {i} is missing recurrent state indices"
                        ))
                    })?;
                    if let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(i) {
                        // Gathers are fresh copies, untouched by the in-place state kernels
                        let stash_states = gdn_stash
                            .as_ref()
                            .map(|_| {
                                candle_core::Result::Ok((
                                    pool.gather_conv_state(&indices)?,
                                    pool.gather_recurrent_state(&indices)?,
                                ))
                            })
                            .transpose()?;

                        // Packed prefill slices the gathered rows per logical sequence
                        let mut gdn_cache = if packed_query_lens.is_some() {
                            GdnLayerCache::gathered(
                                pool.gather_conv_state(&indices)?,
                                pool.gather_recurrent_state(&indices)?,
                            )
                        } else {
                            GdnLayerCache::checkout(pool, &indices)?
                        };

                        let mut projected_stash = None;
                        xs = layer.forward_linear_with_stash(
                            &xs,
                            &mut gdn_cache,
                            recurrent_metadata.batch_kind(),
                            packed_query_lens.as_deref(),
                            stash_states.is_some().then_some(&mut projected_stash),
                        )?;
                        if let (Some(stash), Some((conv_state, recurrent_state))) =
                            (gdn_stash.as_mut(), stash_states)
                        {
                            let projected = projected_stash.ok_or_else(|| {
                                candle_core::Error::msg("GDN forward returned no stash")
                            })?;
                            stash.layers.push(GdnLayerStash {
                                layer_idx: i,
                                projected,
                                conv_state,
                                recurrent_state,
                            });
                        }

                        gdn_cache.commit(
                            pool,
                            &indices,
                            recurrent_metadata.state_indices_host(),
                        )?;
                    } else {
                        candle_core::bail!(
                            "Hybrid cache layer {i} is not recurrent for a linear-attention layer."
                        );
                    }
                }
            }

            // Integrate DeepStack visual features when provided
            if let (Some((idx, idx_expanded)), Some(deepstack)) =
                (&deepstack_indices, deepstack_visual_embeds)
            {
                if i < deepstack.len() {
                    xs = self.deepstack_process(xs, idx, idx_expanded, &deepstack[i])?;
                }
            }
        }
        if self.store_spec_hidden.load(Ordering::Relaxed) {
            *self.gdn_replay_stash.lock().expect("gdn stash poisoned") = gdn_stash;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let store_spec = self.store_spec_hidden.load(Ordering::Relaxed);
        if store_spec {
            *self
                .last_full_capture
                .lock()
                .expect("spec capture poisoned") = Some(SpecCapture {
                hidden: xs.clone(),
                positions: position_ids.to_device(&self.device)?,
            });
        }
        let xs = ctx.logits(&xs)?;
        if store_spec {
            // Reduce the position ids exactly like the hidden rows so they stay aligned
            let positions = position_ids
                .to_device(&self.device)?
                .permute((1, 2, 0))?
                .contiguous()?;
            let positions = ctx.logits(&positions)?.permute((2, 0, 1))?.contiguous()?;
            *self
                .last_spec_capture
                .lock()
                .expect("spec capture poisoned") = Some(SpecCapture {
                hidden: xs.clone(),
                positions,
            });
        }
        self.lm_head.forward(&xs)
    }

    fn deepstack_process(
        &self,
        hidden_states: Tensor,
        idx: &Tensor,
        idx_expanded: &Tensor,
        visual_embeds: &Tensor,
    ) -> Result<Tensor> {
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        let visual_embeds = visual_embeds.to_device(device)?.to_dtype(dtype)?;

        let (batch, seq, hidden) = hidden_states.dims3()?;
        let total = batch * seq;
        let hidden_flat = hidden_states.reshape((total, hidden))?;

        if idx.dim(0)? != visual_embeds.dim(0)? {
            candle_core::bail!(
                "Mismatch between DeepStack visual embeds ({}) and mask positions ({})",
                visual_embeds.dim(0)?,
                idx.dim(0)?
            );
        }

        let result = hidden_flat.scatter_add(idx_expanded, &visual_embeds, 0)?;
        result.reshape((batch, seq, hidden))
    }
}

impl IsqModel for Qwen3_5TextModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_lm = match self.weight_prefix {
            TextWeightPrefix::LanguageModelModel => uvb.pp("language_model").pp("model"),
            TextWeightPrefix::ModelLanguageModel => uvb.pp("model").pp("language_model"),
            TextWeightPrefix::Model => uvb.pp("model"),
        };
        uvb_lm.pp("embed_tokens").add(&self.embed_tokens);
        uvb_lm.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_lm.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);

            match &layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    uvb_l.pp("self_attn").pp("q_norm").add(&attn.q_norm);
                    uvb_l.pp("self_attn").pp("k_norm").add(&attn.k_norm);
                }
                LayerImpl::LinearAttention(gdn) => {
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("conv1d.weight", gdn.conv1d_weight.clone());
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("dt_bias", gdn.dt_bias.clone());
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("A_log", gdn.a_log.clone());
                    uvb_l
                        .pp("linear_attn")
                        .pp("norm")
                        .add_tensor("weight", gdn.norm.weight.clone());
                }
            }
        }
        if let Some(mtp) = &self.mtp {
            mtp.residual_tensors(&uvb);
        }

        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Qwen3_5TextModel {}

impl NormalModel for Qwen3_5TextModel {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let input_embeds = self.embed_tokens(input_ids)?;
        let attention_mask = if ctx.is_paged() {
            CausalMasker.make_causal_mask(
                input_ids,
                &ForwardMaskCache::Paged(ctx.seqlen_offsets()),
                self.dtype,
                &CausalMaskConfig::default(),
            )?
        } else {
            let hybrid_cache = self.cache.hybrid();
            CausalMasker.make_causal_mask(
                input_ids,
                &*hybrid_cache as &dyn PastKvLenCache,
                self.dtype,
                &CausalMaskConfig::default(),
            )?
        };
        let attention_mask = if ctx.is_first_prompt_chunk() {
            attention_mask
        } else {
            AttentionMask::None
        };
        let (batch_size, seq_len) = input_ids.dims2()?;
        let text_positions = ctx
            .text_positions(input_ids.device(), seq_len)?
            .ok_or_else(|| candle_core::Error::msg("Qwen3.5 is missing text positions"))?
            .clone();
        let position_ids = text_positions
            .reshape((batch_size, seq_len))?
            .unsqueeze(0)?
            .repeat((3, 1, 1))?;
        self.forward_embeds(
            input_embeds,
            &attention_mask,
            &position_ids,
            ctx.seqlen_offsets(),
            ctx,
            None,
            None,
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
        candle_core::bail!("Qwen3.5 does not support X-LoRA forward")
    }

    fn is_xlora(&self) -> bool {
        false
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn cache(&self) -> &EitherCache {
        &self.cache
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }

    fn supports_packed_prefill(&self) -> bool {
        true
    }
}

impl AnyMoeBaseModelMixin for Qwen3_5TextModel {}
