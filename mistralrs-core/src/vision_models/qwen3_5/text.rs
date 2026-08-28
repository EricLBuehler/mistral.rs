#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

use candle_core::{DType, Device, Module, Result, Tensor, D};
use mistralrs_quant::{
    ActivationQuantizationScheme, ActivationScaleLayout, ColumnParallelLayer, PackedOutputLayout,
    QuantMethod, QuantizedActivation, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};

use super::{
    config::{LayerType, TextConfig},
    mtp::Qwen3_5MtpHead,
    packed_gdn::{forward_packed_gdn, packed_gdn_layout},
};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    gdn::{
        GatedDeltaNet, GdnConfig, GdnForwardContext, GdnForwardStash, GdnInputProjectionKind,
        GdnLayerCache, GdnSpeculativeStash, GdnTransitionCommitConfig, GdnTransitionStash,
        GdnVHeadLayout, PackedGdnLayout,
    },
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    },
    layers::{self, CausalMasker, GemmaRmsNorm, Mlp, Qwen3VLRotaryEmbedding, Sdpa, YarnRopeConfig},
    layers_masker::{CausalMaskConfig, PastKvLenCache},
    paged_attention::{
        load_fp8_attention_scales, AttentionImplementation, ModelConfigMetadata, PagedAttention,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, ForwardMaskCache, IsqModel, KvCache, ModelForwardContext,
        NormalLoadingMetadata, NormalModel, RecurrentBatchKind,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

const GDN_PENDING_APPLY_MAX_LAYERS: usize = 32;

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
    merged_qkv: Option<crate::ops::MergedDenseProjection>,
    o_proj: Arc<dyn QuantMethod>,
    pub(super) q_norm: GemmaRmsNorm,
    pub(super) k_norm: GemmaRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_gate_grouped: bool,
    pub(super) rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

enum AttentionInput<'a> {
    Dense(&'a Tensor),
    Quantized(&'a QuantizedActivation),
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
        use_paged_attention: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let vb_sa = mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq);
        let vb_sa_norms = mapper.set_device(layer_idx, vb.pp("self_attn"), false);
        let paged_attn = if use_paged_attention {
            Some(PagedAttention::new_with_fp8_attention_scales(
                cfg.head_dim,
                vb_sa_norms.device(),
                None,
                load_fp8_attention_scales(&vb_sa_norms)?,
            )?)
        } else {
            None
        };
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

        let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm)?;
        let q_shard = mistralrs_quant::Shard::Simple {
            dim: 0,
            rank: comm.rank(),
            world_size: comm.world_size(),
        };
        let q_dim = num_heads * head_dim * 2;
        let kv_dim = num_kv_heads * head_dim;
        let q_layout = PackedOutputLayout::rank_local_interleaved_to_grouped(
            num_heads,
            &[head_dim, head_dim],
            comm.world_size(),
        )?;
        let output_layouts = [
            q_layout,
            PackedOutputLayout::identity(kv_dim),
            PackedOutputLayout::identity(kv_dim),
        ];
        let packed = ColumnParallelLayer::new_packed_with_output_layouts(
            cfg.hidden_size,
            &[q_dim, kv_dim, kv_dim],
            &["q_proj", "k_proj", "v_proj"],
            &output_layouts,
            &cfg.quantization_config,
            false,
            comm,
            Some(&[q_shard, kv_shard, kv_shard]),
            vb_sa.clone(),
        )?;
        let (q_proj, k_proj, v_proj, merged_qkv) = match &packed {
            Some(group) => (
                group.constituents[0].clone(),
                group.constituents[1].clone(),
                group.constituents[2].clone(),
                Some(crate::ops::MergedDenseProjection::from_packed(group)),
            ),
            None => (
                ColumnParallelLayer::new(
                    cfg.hidden_size,
                    q_dim,
                    &cfg.quantization_config,
                    false,
                    comm,
                    vb_sa.pp("q_proj"),
                )?,
                ColumnParallelLayer::new_with_shard(
                    cfg.hidden_size,
                    kv_dim,
                    &cfg.quantization_config,
                    false,
                    comm,
                    kv_shard,
                    vb_sa.pp("k_proj"),
                )?,
                ColumnParallelLayer::new_with_shard(
                    cfg.hidden_size,
                    kv_dim,
                    &cfg.quantization_config,
                    false,
                    comm,
                    kv_shard,
                    vb_sa.pp("v_proj"),
                )?,
                None,
            ),
        };
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
            merged_qkv,
            o_proj,
            q_norm,
            k_norm,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            q_gate_grouped: packed.is_some(),
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

    fn input_activation_quantization_scheme_for(
        &self,
        input: &Tensor,
    ) -> Option<ActivationQuantizationScheme> {
        self.merged_qkv
            .as_ref()?
            .activation_quantization_scheme_for(input)
    }

    fn preferred_input_activation_scale_layout_for(
        &self,
        input: &Tensor,
    ) -> Option<ActivationScaleLayout> {
        self.merged_qkv
            .as_ref()?
            .preferred_activation_scale_layout_for(input)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input: AttentionInput<'_>,
        attention_mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: Option<&mut KvCache>,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = match input {
            AttentionInput::Dense(x) => x.dims3()?,
            AttentionInput::Quantized(activation) => {
                let [batch, seq_len, width]: [usize; 3] = activation
                    .source_shape()
                    .try_into()
                    .map_err(|_| candle_core::Error::msg("quantized QKV input must have rank 3"))?;
                (batch, seq_len, width)
            }
        };
        let (q_gate, k, v) = match input {
            AttentionInput::Dense(x) => {
                if let Some(merged_qkv) = &self.merged_qkv {
                    let [q_gate, k, v]: [Tensor; 3] =
                        merged_qkv.forward(x)?.try_into().map_err(|_| {
                            candle_core::Error::msg("packed QKV returned the wrong output count")
                        })?;
                    (q_gate, k, v)
                } else {
                    crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?
                }
            }
            AttentionInput::Quantized(activation) => {
                let merged_qkv = self
                    .merged_qkv
                    .as_ref()
                    .expect("quantized QKV input requires a packed projection");
                let [q_gate, k, v]: [Tensor; 3] = merged_qkv
                    .forward_quantized(activation)?
                    .try_into()
                    .map_err(|_| {
                        candle_core::Error::msg("packed QKV returned the wrong output count")
                    })?;
                (q_gate, k, v)
            }
        };
        let q_width = self.num_heads * self.head_dim;
        let (q, gate) = if self.q_gate_grouped {
            (
                q_gate.narrow(D::Minus1, 0, q_width)?.unfold(
                    D::Minus1,
                    self.head_dim,
                    self.head_dim,
                )?,
                q_gate.narrow(D::Minus1, q_width, q_width)?,
            )
        } else {
            let q_gate = q_gate.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
            let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
            let gate = q_gate
                .narrow(D::Minus1, self.head_dim, self.head_dim)?
                .reshape((b_sz, seq_len, q_width))?;
            (q, gate)
        };

        let mut q = q.transpose(1, 2)?;
        let mut k = k
            .unfold(D::Minus1, self.head_dim, self.head_dim)?
            .transpose(1, 2)?;
        let v = v
            .unfold(D::Minus1, self.head_dim, self.head_dim)?
            .transpose(1, 2)?;

        let cos_sin = &(
            cos_sin.0.to_device(q.device())?,
            cos_sin.1.to_device(q.device())?,
        );
        let tokens_first = metadata
            .as_ref()
            .is_some_and(|(_, input_metadata)| input_metadata.is_decode_step());
        (q, k) = self.rotary_emb.forward_qk_norm_layout(
            cos_sin,
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            tokens_first,
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
        if let Some(res) = crate::ops::try_fused_gated_projection(
            &gate,
            &y,
            layers::Activation::Sigmoid,
            &*self.o_proj,
        )? {
            return Ok(res);
        }
        y = crate::ops::mul_and_act(&gate.to_dtype(y.dtype())?, &y, layers::Activation::Sigmoid)?;

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

struct DecoderLayerOutput {
    branch: Tensor,
    residual: Tensor,
}

struct LayerInputQuantizationPlan {
    scheme: ActivationQuantizationScheme,
    scale_layout: ActivationScaleLayout,
    needs_dense: bool,
}

enum PreparedLayerInput {
    Dense(Tensor),
    Quantized {
        normalized: Option<Tensor>,
        activation: QuantizedActivation,
    },
}

impl PreparedLayerInput {
    fn device(&self) -> &Device {
        match self {
            Self::Dense(tensor) => tensor.device(),
            Self::Quantized { activation, .. } => activation.quantized().device(),
        }
    }

    fn dense(&self) -> Option<&Tensor> {
        match self {
            Self::Dense(tensor) => Some(tensor),
            Self::Quantized { normalized, .. } => normalized.as_ref(),
        }
    }
}

struct LinearForwardContext<'a> {
    x: &'a Tensor,
    normalized_x: Option<&'a PreparedLayerInput>,
    cache: &'a mut GdnLayerCache,
    batch_kind: RecurrentBatchKind,
    checkpoint_lanes: usize,
    transition_checkpoints: bool,
    packed_layout: Option<&'a PackedGdnLayout>,
    stash_out: Option<&'a mut Option<GdnSpeculativeStash>>,
}

impl DecoderLayerOutput {
    fn add(self) -> Result<Tensor> {
        self.branch + self.residual
    }

    fn add_and_norm(self, norm: &GemmaRmsNorm) -> Result<(Tensor, Tensor)> {
        norm.forward_add_rms_norm(&self.branch, &self.residual)
    }

    fn add_and_prepare(
        self,
        norm: &GemmaRmsNorm,
        plan: Option<LayerInputQuantizationPlan>,
    ) -> Result<(Tensor, PreparedLayerInput)> {
        let Some(plan) = plan else {
            let (residual, normalized) = self.add_and_norm(norm)?;
            return Ok((residual, PreparedLayerInput::Dense(normalized)));
        };
        if plan.needs_dense {
            let (fused, normalized) =
                mistralrs_quant::fused_add_rms_norm_quantized_with_normalized(
                    &self.branch,
                    &self.residual,
                    norm.weight(),
                    norm.eps() as f32,
                    plan.scheme,
                    plan.scale_layout,
                )?;
            let (residual, activation) = fused.into_parts();
            Ok((
                residual,
                PreparedLayerInput::Quantized {
                    normalized: Some(normalized),
                    activation,
                },
            ))
        } else {
            let fused = mistralrs_quant::fused_add_rms_norm_quantized(
                &self.branch,
                &self.residual,
                norm.weight(),
                norm.eps() as f32,
                plan.scheme,
                plan.scale_layout,
            )?;
            let (residual, activation) = fused.into_parts();
            Ok((
                residual,
                PreparedLayerInput::Quantized {
                    normalized: None,
                    activation,
                },
            ))
        }
    }
}

impl DecoderLayer {
    fn input_quantization_plan(&self, input: &Tensor) -> Option<LayerInputQuantizationPlan> {
        if input.dtype() != DType::BF16 || !input.device().is_cuda() {
            return None;
        }
        let (scheme, scale_layout, needs_dense) = match &self.layer_impl {
            LayerImpl::FullAttention(attention) => (
                attention.input_activation_quantization_scheme_for(input)?,
                attention.preferred_input_activation_scale_layout_for(input)?,
                false,
            ),
            LayerImpl::LinearAttention(gdn) => (
                gdn.input_activation_quantization_scheme_for(input)?,
                gdn.preferred_input_activation_scale_layout_for(input)?,
                true,
            ),
        };
        matches!(scale_layout, ActivationScaleLayout::GroupMajor { .. }).then_some(
            LayerInputQuantizationPlan {
                scheme,
                scale_layout,
                needs_dense,
            },
        )
    }

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
        self.forward_attention_output(
            x,
            None,
            attention_mask,
            cos_sin,
            kv_cache,
            metadata,
            flash_params,
        )?
        .add()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_output(
        &self,
        x: &Tensor,
        normalized_x: Option<&PreparedLayerInput>,
        attention_mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: Option<&mut KvCache>,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<DecoderLayerOutput> {
        let attn = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => attn,
            _ => candle_core::bail!("Expected full attention layer"),
        };
        let residual = x;
        let normalized_storage;
        let input = match normalized_x {
            Some(PreparedLayerInput::Dense(normalized)) => AttentionInput::Dense(normalized),
            Some(PreparedLayerInput::Quantized { activation, .. }) => {
                AttentionInput::Quantized(activation)
            }
            None => {
                normalized_storage = self.input_layernorm.forward(x)?;
                AttentionInput::Dense(&normalized_storage)
            }
        };
        let attn_out = attn.forward(
            input,
            attention_mask,
            cos_sin,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let (x, ffn_out) = self.mlp.forward_with_add_rms_norm(
            &attn_out,
            residual,
            &self.post_attention_layernorm,
        )?;
        Ok(DecoderLayerOutput {
            branch: ffn_out,
            residual: x,
        })
    }

    fn forward_linear_with_stash(
        &self,
        context: LinearForwardContext<'_>,
    ) -> Result<DecoderLayerOutput> {
        let LinearForwardContext {
            x,
            normalized_x,
            cache,
            batch_kind,
            checkpoint_lanes,
            transition_checkpoints,
            packed_layout,
            stash_out,
        } = context;
        let gdn = match &self.layer_impl {
            LayerImpl::LinearAttention(gdn) => gdn,
            _ => candle_core::bail!("Expected linear attention layer"),
        };
        let residual = x;
        let normalized_storage;
        let (normalized_x, quantized_x) = match normalized_x {
            Some(PreparedLayerInput::Dense(normalized)) => (normalized, None),
            Some(PreparedLayerInput::Quantized {
                normalized,
                activation,
            }) => (
                normalized
                    .as_ref()
                    .expect("quantized GDN input requires dense normalized values"),
                Some(activation),
            ),
            None => {
                normalized_storage = self.input_layernorm.forward(x)?;
                (&normalized_storage, None)
            }
        };
        let gdn_out = if let Some(layout) = packed_layout {
            forward_packed_gdn(gdn, normalized_x, cache, batch_kind, layout)?
        } else if let Some(quantized_x) = quantized_x {
            gdn.forward_quantized_with_context(
                normalized_x,
                quantized_x,
                cache,
                GdnForwardContext {
                    batch_kind,
                    checkpoint_lanes,
                    transition_checkpoints,
                    stash_out,
                },
            )?
            .expect("prepared quantized GDN input must match its projection")
        } else {
            gdn.forward_with_context(
                normalized_x,
                cache,
                GdnForwardContext {
                    batch_kind,
                    checkpoint_lanes,
                    transition_checkpoints,
                    stash_out,
                },
            )?
        };
        let (x, ffn_out) = self.mlp.forward_with_add_rms_norm(
            &gdn_out,
            residual,
            &self.post_attention_layernorm,
        )?;
        Ok(DecoderLayerOutput {
            branch: ffn_out,
            residual: x,
        })
    }
}

// ====================== Text Model ======================

#[derive(Clone, Copy)]
enum TextWeightPrefix {
    LanguageModelModel,
    ModelLanguageModel,
    Model,
}

/// Target activations captured for the MTP proposer: hidden states after the final norm and their
/// text RoPE or MRoPE position ids, `[b, rows, hidden]` / `[b, rows]` or `[3, b, rows]`.
#[derive(Clone)]
pub(super) struct SpecCapture {
    pub(super) hidden: Tensor,
    pub(super) positions: Tensor,
    // Hidden states after each DFlash tap layer, row-aligned with `hidden`; empty unless attached
    pub(super) taps: Vec<Tensor>,
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
    pub(super) state_layout: crate::kv_cache::RecurrentStateLayout,
    pub(super) rollback: GdnLayerRollback,
}

#[derive(Clone)]
pub(super) enum GdnLayerRollback {
    Replay {
        projected: GdnForwardStash,
        conv_state: Tensor,
        recurrent_state: Tensor,
    },
    Transition(GdnTransitionStash),
}

struct GdnPendingApplyGroup {
    device: Device,
    config: GdnTransitionCommitConfig,
    layers: Vec<usize>,
}

#[derive(Debug, PartialEq, Eq)]
struct GdnReplayBatch {
    keep_rows: usize,
    batch_indices: Vec<u32>,
    slots: Vec<u32>,
}

struct GdnReplayIndices {
    batch_indices: Tensor,
    slots: Tensor,
}

struct GdnCommitIndices {
    keep_rows: Tensor,
    slots: Tensor,
}

fn index_select_replay_rows(source: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if source.is_contiguous() {
        source.index_select(indices, 0)
    } else {
        source.contiguous()?.index_select(indices, 0)
    }
}

fn recurrent_checkpoint_devices_supported(devices: &[Device]) -> bool {
    cfg!(feature = "cuda") && !devices.is_empty() && devices.iter().all(Device::is_cuda)
}

fn should_stash_gdn_replay(
    native_speculative_commit: bool,
    store_spec_hidden: bool,
    query_len: usize,
    batch_kind: Option<RecurrentBatchKind>,
    continuation_without_cache: bool,
) -> bool {
    !native_speculative_commit
        && store_spec_hidden
        && query_len > 1
        && batch_kind == Some(RecurrentBatchKind::SpeculativeDecode)
        && continuation_without_cache
}

fn narrow_spec_graph_tensor(
    tensor: &Tensor,
    batch_dim: usize,
    captured_batch: usize,
    real_batch: usize,
    name: &str,
) -> Result<Tensor> {
    let tensor_batch = tensor.dim(batch_dim)?;
    if tensor_batch != captured_batch {
        candle_core::bail!(
            "speculative graph {name} has batch {tensor_batch}, expected {captured_batch}"
        );
    }
    if real_batch == captured_batch {
        Ok(tensor.clone())
    } else {
        tensor.narrow(batch_dim, 0, real_batch)
    }
}

fn narrow_spec_capture(capture: &mut SpecCapture, real_batch: usize) -> Result<()> {
    let captured_batch = capture.hidden.dim(0)?;
    if real_batch > captured_batch {
        candle_core::bail!(
            "speculative graph batch {real_batch} exceeds captured batch {captured_batch}"
        );
    }
    capture.hidden = narrow_spec_graph_tensor(
        &capture.hidden,
        0,
        captured_batch,
        real_batch,
        "hidden state",
    )?;
    let position_batch_dim = match capture.positions.rank() {
        2 => 0,
        3 => 1,
        rank => candle_core::bail!("unexpected speculative position rank {rank}"),
    };
    capture.positions = narrow_spec_graph_tensor(
        &capture.positions,
        position_batch_dim,
        captured_batch,
        real_batch,
        "positions",
    )?;
    for tap in &mut capture.taps {
        *tap = narrow_spec_graph_tensor(tap, 0, captured_batch, real_batch, "tap")?;
    }
    Ok(())
}

fn narrow_gdn_replay_stash(stash: &mut GdnReplayStash, real_batch: usize) -> Result<()> {
    let captured_batch = stash.slots.len();
    if real_batch > captured_batch {
        candle_core::bail!("GDN replay batch {real_batch} exceeds captured batch {captured_batch}");
    }
    for layer in &mut stash.layers {
        match &mut layer.rollback {
            GdnLayerRollback::Replay {
                projected,
                conv_state,
                recurrent_state,
            } => {
                projected.mixed_qkv = narrow_spec_graph_tensor(
                    &projected.mixed_qkv,
                    0,
                    captured_batch,
                    real_batch,
                    "mixed_qkv",
                )?;
                projected.convolved_qkv = narrow_spec_graph_tensor(
                    &projected.convolved_qkv,
                    0,
                    captured_batch,
                    real_batch,
                    "convolved_qkv",
                )?;
                projected.b =
                    narrow_spec_graph_tensor(&projected.b, 0, captured_batch, real_batch, "b")?;
                projected.a =
                    narrow_spec_graph_tensor(&projected.a, 0, captured_batch, real_batch, "a")?;
                *conv_state = narrow_spec_graph_tensor(
                    conv_state,
                    0,
                    captured_batch,
                    real_batch,
                    "conv_state",
                )?;
                *recurrent_state = narrow_spec_graph_tensor(
                    recurrent_state,
                    0,
                    captured_batch,
                    real_batch,
                    "recurrent_state",
                )?;
            }
            GdnLayerRollback::Transition(_) => {}
        }
    }
    stash.slots.truncate(real_batch);
    Ok(())
}

fn group_gdn_replay_batches(rows: &[(usize, usize)], slots: &[u32]) -> Result<Vec<GdnReplayBatch>> {
    let mut grouped = BTreeMap::<usize, Vec<(u32, u32)>>::new();
    for &(batch_idx, keep_rows) in rows {
        let tensor_idx = u32::try_from(batch_idx).map_err(|_| {
            candle_core::Error::msg(format!("GDN replay batch row {batch_idx} exceeds u32"))
        })?;
        let slot = *slots.get(batch_idx).ok_or_else(|| {
            candle_core::Error::msg(format!("GDN replay stash has no batch row {batch_idx}"))
        })?;
        grouped
            .entry(keep_rows)
            .or_default()
            .push((tensor_idx, slot));
    }
    Ok(grouped
        .into_iter()
        .map(|(keep_rows, rows)| GdnReplayBatch {
            keep_rows,
            batch_indices: rows.iter().map(|(batch_idx, _)| *batch_idx).collect(),
            slots: rows.into_iter().map(|(_, slot)| slot).collect(),
        })
        .collect())
}

fn refresh_gdn_stash_slots(stash: &mut GdnReplayStash, slots: &[u32]) -> Result<()> {
    let batch_size = stash.slots.len();
    if slots.len() < batch_size {
        candle_core::bail!(
            "GDN graph state has {batch_size} rows, but the live slot table has {}",
            slots.len()
        );
    }
    stash.slots.clear();
    stash.slots.extend_from_slice(&slots[..batch_size]);
    Ok(())
}

fn terminal_gdn_transition_slots(
    rows: &[crate::speculative::SpeculativeCommitRow],
    slots: &[u32],
) -> Result<Vec<u32>> {
    rows.iter()
        .filter(|row| row.terminal)
        .map(|row| {
            slots.get(row.batch_idx).copied().ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "GDN transition stash has no terminal batch row {}",
                    row.batch_idx
                ))
            })
        })
        .collect()
}

fn gdn_transition_keep_rows(
    rows: &[crate::speculative::SpeculativeCommitRow],
    batch_size: usize,
    max_rows: usize,
) -> Result<Vec<u32>> {
    if rows.len() != batch_size {
        candle_core::bail!(
            "GDN transition commit has {} rows for a {batch_size}-row stash",
            rows.len()
        );
    }
    let mut keep_rows = vec![None; batch_size];
    for row in rows {
        if row.keep_rows == 0 || row.keep_rows > max_rows {
            candle_core::bail!(
                "GDN transition commit row {} keeps {}, expected 1..={max_rows}",
                row.batch_idx,
                row.keep_rows
            );
        }
        let destination = keep_rows.get_mut(row.batch_idx).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "GDN transition stash has no batch row {}",
                row.batch_idx
            ))
        })?;
        if destination.is_some() {
            candle_core::bail!(
                "GDN transition commit contains batch row {} more than once",
                row.batch_idx
            );
        }
        *destination = Some(u32::try_from(row.keep_rows).map_err(|_| {
            candle_core::Error::msg(format!(
                "GDN transition row count {} exceeds u32",
                row.keep_rows
            ))
        })?);
    }
    keep_rows
        .into_iter()
        .enumerate()
        .map(|(batch_idx, rows)| {
            rows.ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "GDN transition commit is missing batch row {batch_idx}"
                ))
            })
        })
        .collect()
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
            out.extend(capture.taps.iter().cloned());
        }
        if let Some(stash) = &self.gdn_stash {
            for layer in &stash.layers {
                match &layer.rollback {
                    GdnLayerRollback::Replay {
                        projected,
                        conv_state,
                        recurrent_state,
                    } => {
                        out.push(projected.mixed_qkv.clone());
                        out.push(projected.convolved_qkv.clone());
                        out.push(projected.b.clone());
                        out.push(projected.a.clone());
                        out.push(conv_state.clone());
                        out.push(recurrent_state.clone());
                    }
                    GdnLayerRollback::Transition(_) => {}
                }
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
            for tap in capture.taps.iter_mut() {
                *tap = next()?;
            }
        }
        if let Some(stash) = state.gdn_stash.as_mut() {
            for layer in stash.layers.iter_mut() {
                match &mut layer.rollback {
                    GdnLayerRollback::Replay {
                        projected,
                        conv_state,
                        recurrent_state,
                    } => {
                        projected.mixed_qkv = next()?;
                        projected.convolved_qkv = next()?;
                        projected.b = next()?;
                        projected.a = next()?;
                        *conv_state = next()?;
                        *recurrent_state = next()?;
                    }
                    GdnLayerRollback::Transition(_) => {}
                }
            }
        }
        Ok(Box::new(state))
    }

    fn for_real_batch(
        &self,
        real_batch: usize,
    ) -> Result<Box<dyn crate::speculative::SpeculativeGraphState>> {
        let mut state = self.clone();
        for capture in [&mut state.spec_capture, &mut state.full_capture]
            .into_iter()
            .flatten()
        {
            narrow_spec_capture(capture, real_batch)?;
        }
        if let Some(stash) = state.gdn_stash.as_mut() {
            narrow_gdn_replay_stash(stash, real_batch)?;
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
    // Target layers whose outputs a DFlash drafter consumes; empty when none is attached
    dflash_tap_layers: Mutex<Vec<usize>>,
    pub(super) yarn_rope_config: Option<YarnRopeConfig>,
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
        let yarn_rope_config = cfg.yarn_rope_config()?;
        if let Some(yarn) = &yarn_rope_config {
            tracing::info!(
                factor = yarn.factor,
                original_max_position_embeddings = yarn.original_max_position_embeddings,
                max_position_embeddings = yarn.max_position_embeddings,
                "Using Qwen3.5 YaRN rotary embeddings"
            );
        }
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
                    match yarn_rope_config.as_ref() {
                        Some(yarn) => Qwen3VLRotaryEmbedding::new_yarn(
                            yarn,
                            device,
                            cfg.mrope_section().to_vec(),
                        ),
                        None => Qwen3VLRotaryEmbedding::new(
                            cfg.rope_theta() as f32,
                            rot_dim,
                            device,
                            cfg.mrope_section().to_vec(),
                        ),
                    }
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
                    LayerImpl::FullAttention(FullAttention::load(
                        vb_l.pp(layer_idx),
                        cfg,
                        &*mapper,
                        layer_idx,
                        normal_loading_metadata.loading_isq,
                        rotary_emb,
                        matches!(attention_mechanism, AttentionImplementation::PagedAttention),
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
                state: crate::kv_cache::RecurrentStateSpec::Gdn {
                    heads: cfg.linear_num_value_heads,
                    key_dim: cfg.linear_key_head_dim,
                    value_dim: cfg.linear_value_head_dim,
                },
                recurrent_dtype: Some(cfg.mamba_ssm_dtype.dtype()),
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
            dflash_tap_layers: Mutex::new(Vec::new()),
            yarn_rope_config,
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

    pub(super) fn reserve_recurrent_transition_storage(&self) -> Result<bool> {
        let mut cache = self.cache.hybrid();
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }
        let max_rows = cache.checkpoint_lanes();
        let mut spec = None;
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if *layer_type != LayerType::LinearAttention {
                continue;
            }
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                candle_core::bail!("Qwen3.5 GDN layer has no recurrent state pool");
            };
            if !gdn.speculative_transitions_supported(pool, self.dtype) {
                return Ok(false);
            }
            let layer_spec = gdn.pending_transition_spec(max_rows);
            if spec
                .replace(layer_spec)
                .is_some_and(|spec| spec != layer_spec)
            {
                candle_core::bail!("Qwen3.5 GDN transition dimensions diverge across layers");
            }
        }
        let Some(spec) = spec else {
            return Ok(false);
        };
        cache.reserve_gdn_pending_transitions(spec)
    }

    pub(super) fn reserve_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        let mut cache = self.cache.hybrid();
        let mut spec = None;
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if *layer_type != LayerType::LinearAttention {
                continue;
            }
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                candle_core::bail!("Qwen3.5 GDN layer has no recurrent state pool");
            };
            if !gdn.deferred_decode_supported(pool, self.dtype) {
                return Ok(false);
            }
            let layer_spec = gdn.deferred_state_spec();
            if spec
                .replace(layer_spec)
                .is_some_and(|spec| spec != layer_spec)
            {
                candle_core::bail!("Qwen3.5 GDN deferred-state dimensions diverge across layers");
            }
        }
        let Some(spec) = spec else {
            return Ok(false);
        };
        cache.reserve_gdn_deferred_state(spec)
    }

    pub(super) fn disable_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        self.cache.hybrid().disable_gdn_deferred_state()
    }

    fn apply_pending_recurrent_transitions_with_cache(
        &self,
        cache: &HybridCache,
        slots: &[u32],
    ) -> Result<bool> {
        let mut slots = slots
            .iter()
            .copied()
            .filter(|slot| *slot != crate::cuda::gdn::GDN_PAD_SLOT)
            .collect::<Vec<_>>();
        slots.sort_unstable();
        slots.dedup();
        if slots.is_empty() {
            return Ok(true);
        }
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }
        let active_slots = Tensor::from_vec(slots.clone(), (slots.len(),), &self.device)?;
        self.apply_pending_recurrent_transitions(cache, &active_slots, false)
    }

    fn apply_pending_recurrent_transitions_for_current_batch(
        &self,
        cache: &HybridCache,
    ) -> Result<bool> {
        let Some(active_slots) = cache.state_indices() else {
            return Ok(false);
        };
        self.apply_pending_recurrent_transitions(cache, active_slots, true)
    }

    pub(super) fn apply_current_recurrent_transitions(&self) -> Result<bool> {
        let cache = self.cache.hybrid();
        self.apply_pending_recurrent_transitions_for_current_batch(&cache)
    }

    fn apply_pending_recurrent_transitions(
        &self,
        cache: &HybridCache,
        active_slots: &Tensor,
        use_cached_device_slots: bool,
    ) -> Result<bool> {
        if active_slots.elem_count() == 0 {
            return Ok(true);
        }
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }
        let mut groups = Vec::<GdnPendingApplyGroup>::new();
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if *layer_type != LayerType::LinearAttention {
                continue;
            }
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                return Ok(false);
            };
            if !gdn.speculative_transitions_supported(pool, self.dtype)
                || pool.pending_transitions().is_none()
            {
                return Ok(false);
            }
            let config = gdn.transition_commit_config(pool);
            let device = pool.device();
            if let Some(group) = groups
                .iter_mut()
                .find(|group| group.config == config && group.device.same_device(device))
            {
                group.layers.push(layer_idx);
            } else {
                groups.push(GdnPendingApplyGroup {
                    device: device.clone(),
                    config,
                    layers: vec![layer_idx],
                });
            }
        }
        if groups.is_empty() {
            return Ok(false);
        }

        for group in groups {
            let active_slots = if use_cached_device_slots {
                cache
                    .state_indices_for_device(&group.device)
                    .ok_or_else(|| {
                        candle_core::Error::msg(
                            "GDN transition batch has no device-local state slots",
                        )
                    })?
            } else {
                active_slots.to_device(&group.device)?
            };
            for layer_indices in group.layers.chunks(GDN_PENDING_APPLY_MAX_LAYERS) {
                let mut layers = Vec::with_capacity(layer_indices.len());
                for &layer_idx in layer_indices {
                    let Some(HybridLayerCache::Recurrent(pool)) = cache.get(layer_idx) else {
                        unreachable!("GDN transition pool was validated above")
                    };
                    let pending = pool
                        .pending_transitions()
                        .expect("GDN pending transition pool was validated above");
                    layers.push(crate::cuda::gdn::GdnPendingTransitionApplyLayer {
                        pending_conv_input: &pending.conv_input,
                        pending_key_banks: &pending.key_banks,
                        pending_key_bank: &pending.key_bank,
                        pending_delta: &pending.delta,
                        pending_decay: &pending.decay,
                        pending_keep_rows: &pending.keep_rows,
                        pending_epochs: &pending.pending_epochs,
                        conv_applied_epochs: &pending.conv_applied_epochs,
                        recurrent_applied_epochs: &pending.recurrent_applied_epochs,
                        conv_state: &pool.conv_state,
                        recurrent_state: &pool.recurrent_state,
                    });
                }
                crate::cuda::gdn::pending_transition_apply_batched_cuda(
                    crate::cuda::gdn::GdnPendingTransitionApply {
                        layers: &layers,
                        active_slots: &active_slots,
                        num_k_heads: group.config.num_k_heads,
                        num_v_heads: group.config.num_v_heads,
                        head_k_dim: group.config.head_k_dim,
                        head_v_dim: group.config.head_v_dim,
                        conv_dim: group.config.conv_dim,
                        conv_width: group.config.conv_width,
                        tiled_v_heads: group.config.tiled_v_heads,
                        state_layout: group.config.state_layout,
                    },
                )?;
            }
        }
        Ok(true)
    }

    fn flush_deferred_recurrent_state(
        &self,
        cache: &HybridCache,
        slots: Option<&[u32]>,
    ) -> Result<bool> {
        if !cache.uses_gdn_deferred_state() {
            return Ok(false);
        }
        let host_slots = slots.map(|slots| {
            let mut slots = slots
                .iter()
                .copied()
                .filter(|slot| *slot != crate::cuda::gdn::GDN_PAD_SLOT)
                .collect::<Vec<_>>();
            slots.sort_unstable();
            slots.dedup();
            slots
        });
        if host_slots.as_ref().is_some_and(Vec::is_empty) {
            return Ok(true);
        }
        let mut flushed = false;
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if *layer_type != LayerType::LinearAttention {
                continue;
            }
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                return Ok(false);
            };
            let active_slots = match &host_slots {
                Some(slots) => Tensor::from_vec(slots.clone(), (slots.len(),), pool.device())?,
                None => cache
                    .state_indices_for_device(pool.device())
                    .ok_or_else(|| {
                        candle_core::Error::msg(
                            "GDN deferred-state flush has no device-local state slots",
                        )
                    })?,
            };
            if !gdn.flush_deferred_state(pool, &active_slots, self.dtype)? {
                return Ok(false);
            }
            flushed = true;
        }
        Ok(flushed)
    }

    pub(super) fn flush_current_recurrent_state(&self) -> Result<()> {
        let cache = self.cache.hybrid();
        let has_slots = cache
            .state_indices()
            .is_some_and(|slots| slots.elem_count() != 0);
        if !has_slots {
            return Ok(());
        }
        if cache.uses_recurrent_transition_log()
            && !self.apply_pending_recurrent_transitions_for_current_batch(&cache)?
        {
            candle_core::bail!("Qwen3.5 pending recurrent transitions cannot be applied");
        }
        if cache.uses_gdn_deferred_state() && !self.flush_deferred_recurrent_state(&cache, None)? {
            candle_core::bail!("Qwen3.5 deferred recurrent state cannot be materialized");
        }
        Ok(())
    }

    pub(super) fn flush_recurrent_transitions_for_sequences(
        &self,
        sequence_ids: &[usize],
    ) -> Result<()> {
        let cache = self.cache.hybrid();
        let slots = cache.recurrent_slots_for_sequences(sequence_ids);
        if cache.uses_recurrent_transition_log()
            && !self.apply_pending_recurrent_transitions_with_cache(&cache, &slots)?
            && !slots.is_empty()
        {
            candle_core::bail!("Qwen3.5 pending recurrent transitions cannot be applied");
        }
        if cache.uses_gdn_deferred_state()
            && !self.flush_deferred_recurrent_state(&cache, Some(&slots))?
            && !slots.is_empty()
        {
            candle_core::bail!("Qwen3.5 deferred recurrent state cannot be materialized");
        }
        Ok(())
    }

    pub(super) fn stage_recurrent_prefixes(
        &self,
        rows: &[crate::speculative::SpeculativeCommitRow],
    ) -> Result<bool> {
        if rows.is_empty() {
            return Ok(true);
        }
        let Some(stash) = self
            .gdn_replay_stash
            .lock()
            .expect("gdn stash poisoned")
            .clone()
        else {
            candle_core::bail!("no GDN transition stash for speculative commit");
        };
        if stash.layers.is_empty()
            || stash
                .layers
                .iter()
                .any(|layer| !matches!(layer.rollback, GdnLayerRollback::Transition(_)))
        {
            return Ok(false);
        }

        let max_rows = self.cache.hybrid().checkpoint_lanes();
        let keep_rows_host = gdn_transition_keep_rows(rows, stash.slots.len(), max_rows)?;
        let mut live_slots = stash
            .slots
            .iter()
            .copied()
            .filter(|slot| *slot != crate::cuda::gdn::GDN_PAD_SLOT)
            .collect::<Vec<_>>();
        live_slots.sort_unstable();
        if live_slots.windows(2).any(|slots| slots[0] == slots[1]) {
            candle_core::bail!("GDN transition batch contains duplicate recurrent slots");
        }

        struct PublishGroup {
            device: Device,
            capacity: usize,
            max_rows: usize,
            layers: Vec<usize>,
        }
        let cache = self.cache.hybrid();
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }
        let mut groups = Vec::<PublishGroup>::new();
        for (stash_idx, layer) in stash.layers.iter().enumerate() {
            let GdnLayerRollback::Transition(_) = &layer.rollback else {
                unreachable!("transition stash was validated above")
            };
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) = (
                &self.layers[layer.layer_idx].layer_impl,
                cache.get(layer.layer_idx),
            ) else {
                return Ok(false);
            };
            let Some(pending) = pool.pending_transitions() else {
                return Ok(false);
            };
            if !gdn.speculative_transitions_supported(pool, self.dtype)
                || pool.state_layout() != layer.state_layout
                || pending.capacity() != cache.recurrent_capacity()
                || pending.spec().num_k_heads != gdn.transition_commit_config(pool).num_k_heads
                || pending.spec().max_rows != max_rows
            {
                return Ok(false);
            }
            let device = pool.device();
            if let Some(group) = groups.iter_mut().find(|group| {
                group.capacity == pending.capacity()
                    && group.max_rows == pending.spec().max_rows
                    && group.device.same_device(device)
            }) {
                group.layers.push(stash_idx);
            } else {
                groups.push(PublishGroup {
                    device: device.clone(),
                    capacity: pending.capacity(),
                    max_rows: pending.spec().max_rows,
                    layers: vec![stash_idx],
                });
            }
        }

        for group in groups {
            if live_slots
                .iter()
                .any(|slot| *slot as usize >= group.capacity)
            {
                candle_core::bail!("GDN transition slot exceeds recurrent capacity");
            }
            let keep_rows = Tensor::from_vec(
                keep_rows_host.clone(),
                (keep_rows_host.len(),),
                &group.device,
            )?;
            let slots = Tensor::from_vec(stash.slots.clone(), (stash.slots.len(),), &group.device)?;
            let mut layers = Vec::with_capacity(group.layers.len());
            for stash_idx in group.layers {
                let layer = &stash.layers[stash_idx];
                let GdnLayerRollback::Transition(_) = &layer.rollback else {
                    unreachable!("transition stash was validated above")
                };
                let Some(HybridLayerCache::Recurrent(pool)) = cache.get(layer.layer_idx) else {
                    unreachable!("transition pool was validated above")
                };
                let pending = pool
                    .pending_transitions()
                    .expect("pending transition pool was validated above");
                layers.push(crate::cuda::gdn::GdnPendingTransitionPublishLayer {
                    pending_keep_rows: &pending.keep_rows,
                    pending_epochs: &pending.pending_epochs,
                    pending_key_bank: &pending.key_bank,
                });
            }
            crate::cuda::gdn::pending_transition_publish_batched_cuda(
                crate::cuda::gdn::GdnPendingTransitionPublish {
                    layers: &layers,
                    keep_rows: &keep_rows,
                    destination_slots: &slots,
                    max_rows: group.max_rows,
                    destination_capacity: group.capacity,
                },
            )?;
        }
        let terminal_slots = terminal_gdn_transition_slots(rows, &stash.slots)?;
        if !terminal_slots.is_empty()
            && !self.apply_pending_recurrent_transitions_with_cache(&cache, &terminal_slots)?
        {
            candle_core::bail!("Qwen3.5 terminal recurrent transitions cannot be applied");
        }
        Ok(true)
    }

    pub(super) fn replay_recurrent_prefixes(&self, rows: &[(usize, usize)]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let Some(stash) = self
            .gdn_replay_stash
            .lock()
            .expect("gdn stash poisoned")
            .clone()
        else {
            candle_core::bail!("no GDN replay stash for speculative rollback");
        };
        let transition_layers = stash
            .layers
            .iter()
            .filter(|layer| matches!(layer.rollback, GdnLayerRollback::Transition(_)))
            .count();
        if transition_layers != 0 && transition_layers != stash.layers.len() {
            candle_core::bail!("GDN speculative stash mixes replay and transition layers");
        }
        if transition_layers == stash.layers.len() && !stash.layers.is_empty() {
            candle_core::bail!("GDN direct transitions must be published before replay fallback");
        }

        let devices = stash.layers.iter().fold(Vec::new(), |mut devices, layer| {
            let GdnLayerRollback::Replay { projected, .. } = &layer.rollback else {
                unreachable!("transition layers were handled above")
            };
            let device = projected.mixed_qkv.device();
            if !devices
                .iter()
                .any(|cached: &Device| cached.same_device(device))
            {
                devices.push(device.clone());
            }
            devices
        });
        let fused_commit_supported = !stash.layers.is_empty()
            && stash.layers.iter().all(|layer| match &layer.rollback {
                GdnLayerRollback::Replay { projected, .. } => {
                    projected.mixed_qkv.device().is_cuda()
                }
                GdnLayerRollback::Transition(_) => false,
            })
            && {
                let hybrid_cache = self.cache.hybrid();
                stash.layers.iter().all(|layer| {
                    let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                        (&self.layers[layer.layer_idx].layer_impl, hybrid_cache.get(layer.layer_idx))
                    else {
                        return false;
                    };
                    let GdnLayerRollback::Replay {
                        projected,
                        conv_state,
                        recurrent_state,
                    } = &layer.rollback
                    else {
                        return false;
                    };
                    pool.state_layout() == layer.state_layout
                        && gdn.speculative_state_commit_supported(
                            projected,
                            conv_state,
                            recurrent_state,
                            pool,
                        )
                })
            };
        if fused_commit_supported {
            let mut keep_rows_host = vec![0u32; stash.slots.len()];
            for &(batch_idx, rows) in rows {
                let keep_rows = u32::try_from(rows).map_err(|_| {
                    candle_core::Error::msg(format!("GDN commit row count {rows} exceeds u32"))
                })?;
                *keep_rows_host.get_mut(batch_idx).ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "GDN replay stash has no batch row {batch_idx}"
                    ))
                })? = keep_rows;
            }
            let commit_indices = devices
                .iter()
                .map(|device| {
                    Ok(GdnCommitIndices {
                        keep_rows: Tensor::from_vec(
                            keep_rows_host.clone(),
                            (keep_rows_host.len(),),
                            device,
                        )?,
                        slots: Tensor::from_vec(stash.slots.clone(), (stash.slots.len(),), device)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut hybrid_cache = self.cache.hybrid();
            for layer in &stash.layers {
                let GdnLayerRollback::Replay {
                    projected,
                    conv_state,
                    recurrent_state,
                } = &layer.rollback
                else {
                    unreachable!("transition layers were handled above")
                };
                let gdn = match &self.layers[layer.layer_idx].layer_impl {
                    LayerImpl::LinearAttention(gdn) => gdn,
                    LayerImpl::FullAttention(_) => {
                        candle_core::bail!("GDN replay stash points at a full-attention layer")
                    }
                };
                let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer.layer_idx)
                else {
                    candle_core::bail!(
                        "GDN replay stash layer {} has no recurrent state pool",
                        layer.layer_idx
                    );
                };
                if pool.state_layout() != layer.state_layout {
                    candle_core::bail!(
                        "GDN replay state layout mismatch: stash {:?}, pool {:?}",
                        layer.state_layout,
                        pool.state_layout()
                    );
                }
                let device_idx = devices
                    .iter()
                    .position(|device| device.same_device(projected.mixed_qkv.device()))
                    .expect("stashed GDN layer device was collected above");
                let indices = &commit_indices[device_idx];
                if !gdn.commit_state_batch_from_stash_cuda(
                    projected,
                    conv_state,
                    recurrent_state,
                    &indices.keep_rows,
                    &indices.slots,
                    pool,
                )? {
                    candle_core::bail!("CUDA GDN speculative state commit was unavailable");
                }
            }
            return Ok(());
        }

        let batches = group_gdn_replay_batches(rows, &stash.slots)?;
        let replay_indices = batches
            .iter()
            .map(|batch| {
                devices
                    .iter()
                    .map(|device| {
                        Ok(GdnReplayIndices {
                            batch_indices: Tensor::from_vec(
                                batch.batch_indices.clone(),
                                (batch.batch_indices.len(),),
                                device,
                            )?,
                            slots: Tensor::from_vec(
                                batch.slots.clone(),
                                (batch.slots.len(),),
                                device,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let mut hybrid_cache = self.cache.hybrid();
        for layer in &stash.layers {
            let GdnLayerRollback::Replay {
                projected,
                conv_state,
                recurrent_state,
            } = &layer.rollback
            else {
                unreachable!("transition layers were handled above")
            };
            let gdn = match &self.layers[layer.layer_idx].layer_impl {
                LayerImpl::LinearAttention(gdn) => gdn,
                LayerImpl::FullAttention(_) => {
                    candle_core::bail!("GDN replay stash points at a full-attention layer")
                }
            };
            let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer.layer_idx)
            else {
                candle_core::bail!(
                    "GDN replay stash layer {} has no recurrent state pool",
                    layer.layer_idx
                );
            };
            if pool.state_layout() != layer.state_layout {
                candle_core::bail!(
                    "GDN replay state layout mismatch: stash {:?}, pool {:?}",
                    layer.state_layout,
                    pool.state_layout()
                );
            }
            let device_idx = devices
                .iter()
                .position(|device| device.same_device(projected.mixed_qkv.device()))
                .expect("stashed GDN layer device was collected above");
            for (group_idx, batch) in batches.iter().enumerate() {
                let indices = &replay_indices[group_idx][device_idx];
                let mut cache = GdnLayerCache::gathered(
                    index_select_replay_rows(conv_state, &indices.batch_indices)?,
                    index_select_replay_rows(recurrent_state, &indices.batch_indices)?,
                    layer.state_layout,
                );
                gdn.advance_state_batch_from_stash(
                    projected,
                    &indices.batch_indices,
                    batch.keep_rows,
                    &mut cache,
                )?;
                pool.scatter_conv_state(&indices.slots, &cache.conv_state)?;
                pool.scatter_recurrent_state(&indices.slots, &cache.recurrent_state)?;
            }
        }
        Ok(())
    }

    pub(super) fn clear_gdn_replay_stash(&self) {
        *self.gdn_replay_stash.lock().expect("gdn stash poisoned") = None;
    }

    pub(super) fn set_dflash_tap_layers(&self, layers: Vec<usize>) {
        *self.dflash_tap_layers.lock().expect("dflash taps poisoned") = layers;
    }

    pub(super) fn supports_recurrent_speculative_checkpoints(&self) -> bool {
        self.supports_recurrent_speculative_checkpoints_with_cache(&self.cache.hybrid())
    }

    pub(super) fn supports_recurrent_speculative_transitions(&self) -> bool {
        self.supports_recurrent_speculative_transitions_with_cache(&self.cache.hybrid())
    }

    pub(super) fn supports_recurrent_speculative_transitions_with_cache(
        &self,
        cache: &HybridCache,
    ) -> bool {
        let recurrent_devices = cache.recurrent_devices();
        if !recurrent_checkpoint_devices_supported(&recurrent_devices) {
            return false;
        }
        let mut found_gdn = false;
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if *layer_type != LayerType::LinearAttention {
                continue;
            }
            found_gdn = true;
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                return false;
            };
            if !gdn.speculative_transitions_supported(pool, self.dtype) {
                return false;
            }
        }
        found_gdn
    }

    pub(super) fn supports_recurrent_speculative_checkpoints_with_cache(
        &self,
        cache: &HybridCache,
    ) -> bool {
        let recurrent_devices = cache.recurrent_devices();
        if !recurrent_checkpoint_devices_supported(&recurrent_devices) {
            return false;
        }
        let mut found_gdn = false;
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if *layer_type != LayerType::LinearAttention {
                continue;
            }
            found_gdn = true;
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                return false;
            };
            if !gdn.speculative_checkpoints_supported(pool, self.dtype) {
                return false;
            }
        }
        found_gdn
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

    pub(super) fn install_spec_graph_state(&self, state: &SpecGraphState) -> Result<()> {
        let spec_capture = state.spec_capture.clone();
        let full_capture = state.full_capture.clone();
        let mut gdn_stash = state.gdn_stash.clone();
        let slots = self
            .cache
            .hybrid()
            .state_indices_host()
            .map(ToOwned::to_owned);
        if let (Some(stash), Some(slots)) = (gdn_stash.as_mut(), slots.as_deref()) {
            refresh_gdn_stash_slots(stash, slots)?;
        }
        *self
            .last_spec_capture
            .lock()
            .expect("spec capture poisoned") = spec_capture;
        *self
            .last_full_capture
            .lock()
            .expect("spec capture poisoned") = full_capture;
        *self.gdn_replay_stash.lock().expect("gdn stash poisoned") = gdn_stash;
        Ok(())
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

    pub(super) fn token_embedding(&self) -> &Arc<dyn QuantMethod> {
        &self.embed_tokens
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
        let checkpoint_lanes = hybrid_cache.checkpoint_lanes();
        let batch_size = xs.dim(0)?;
        let query_len = xs.dim(1)?;
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
        let packed_layout = if has_linear_attention {
            packed_gdn_layout(&xs, ctx)?
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
                LayerImpl::FullAttention(attn) => match position_ids.rank() {
                    2 => attn
                        .rotary_emb
                        .compute_text_cos_sin(position_ids, xs.dtype())?,
                    3 => attn.rotary_emb.compute_cos_sin(position_ids, xs.dtype())?,
                    rank => {
                        candle_core::bail!("unexpected Qwen3.5 position rank {rank}")
                    }
                },
                _ => unreachable!(),
            }
        };

        let attention_mask = DeviceMappedMask::new(attention_mask.clone(), &*self.mapper)?;

        let speculative_gdn = checkpoint_lanes > 1
            && (1..=checkpoint_lanes).contains(&query_len)
            && recurrent_metadata.as_ref().is_some_and(|metadata| {
                metadata.batch_kind() == RecurrentBatchKind::SpeculativeDecode
            });
        let transition_gdn = speculative_gdn
            && hybrid_cache.uses_recurrent_transition_log()
            && query_len <= crate::cuda::gdn::GDN_SPEC_FUSED_MAX_TOKENS
            && self.supports_recurrent_speculative_transitions_with_cache(&hybrid_cache);
        if !transition_gdn && hybrid_cache.uses_recurrent_transition_log() {
            let has_slots = hybrid_cache
                .state_indices()
                .is_some_and(|slots| slots.elem_count() != 0);
            if !self.apply_pending_recurrent_transitions_for_current_batch(&hybrid_cache)?
                && has_slots
            {
                candle_core::bail!("Qwen3.5 pending recurrent transitions cannot be applied");
            }
        }
        let deferred_gdn = query_len == 1
            && crate::cuda::gdn::deferred_decode_batch_supported(batch_size)
            && packed_layout.is_none()
            && recurrent_metadata
                .as_ref()
                .is_some_and(|metadata| metadata.batch_kind() == RecurrentBatchKind::Decode)
            && hybrid_cache.uses_gdn_deferred_state();
        if !deferred_gdn && hybrid_cache.uses_gdn_deferred_state() {
            let has_slots = hybrid_cache
                .state_indices()
                .is_some_and(|slots| slots.elem_count() != 0);
            if has_slots && !self.flush_deferred_recurrent_state(&hybrid_cache, None)? {
                candle_core::bail!("Qwen3.5 deferred recurrent state cannot be materialized");
            }
        }
        let checkpoint_gdn = speculative_gdn
            && !transition_gdn
            && self.supports_recurrent_speculative_checkpoints_with_cache(&hybrid_cache);
        let gdn_checkpoint_lanes = if checkpoint_gdn { checkpoint_lanes } else { 1 };
        let store_spec_hidden = self.store_spec_hidden.load(Ordering::Relaxed);
        let stash_replay = should_stash_gdn_replay(
            checkpoint_gdn || transition_gdn,
            store_spec_hidden,
            query_len,
            recurrent_metadata
                .as_ref()
                .map(|metadata| metadata.batch_kind()),
            ctx.paged_input_metadata().is_some_and(|meta| {
                !meta.is_first_prompt_chunk && meta.num_cached_tokens.is_none()
            }),
        );
        let stash_gdn = stash_replay || (transition_gdn && store_spec_hidden);
        let mut gdn_stash = stash_gdn.then(|| GdnReplayStash {
            slots: recurrent_metadata
                .as_ref()
                .and_then(|meta| meta.state_indices_host())
                .map(|slots| slots.to_vec())
                .unwrap_or_default(),
            layers: Vec::new(),
        });

        let tap_layers = self
            .dflash_tap_layers
            .lock()
            .expect("dflash taps poisoned")
            .clone();
        let capture_taps = self.store_spec_hidden.load(Ordering::Relaxed) && !tap_layers.is_empty();
        let mut taps_all: Vec<Tensor> = Vec::with_capacity(tap_layers.len());

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
                .map(|(i, _)| {
                    u32::try_from(i).map_err(|_| {
                        candle_core::Error::msg(format!("visual position index {i} exceeds u32"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
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

        let forward_result = (|| -> Result<Tensor> {
            let mut normalized_x = None;
            for (i, layer) in self.layers.iter().enumerate() {
                xs = self.mapper.map(xs, i)?;
                if normalized_x
                    .as_ref()
                    .is_some_and(|prepared: &PreparedLayerInput| {
                        !prepared.device().same_device(xs.device())
                    })
                {
                    normalized_x = None;
                }

                let layer_output = match &self.layer_types[i] {
                    LayerType::FullAttention => {
                        let Some(HybridLayerCache::Attention(kv_cache)) = hybrid_cache.get_mut(i)
                        else {
                            candle_core::bail!(
                            "Hybrid cache layer {i} is not attention for a full-attention layer."
                        );
                        };
                        layer.forward_attention_output(
                            &xs,
                            normalized_x.as_ref(),
                            &attention_mask.get(xs.device()),
                            &cos_sin,
                            Some(kv_cache),
                            ctx.paged_layer(i),
                            ctx.flash_params(),
                        )?
                    }
                    LayerType::LinearAttention => {
                        let recurrent_metadata = recurrent_metadata.as_ref().expect(
                            "checked above: linear-attention layers require recurrent metadata",
                        );
                        let indices =
                            hybrid_cache.state_indices_for_layer(i)?.ok_or_else(|| {
                                candle_core::Error::msg(format!(
                                    "Hybrid cache layer {i} is missing recurrent state indices"
                                ))
                            })?;
                        let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(i)
                        else {
                            candle_core::bail!(
                            "Hybrid cache layer {i} is not recurrent for a linear-attention layer."
                        );
                        };
                        let stash_states = stash_replay
                            .then_some(())
                            .as_ref()
                            .map(|_| {
                                candle_core::Result::Ok((
                                    pool.gather_conv_state(&indices)?,
                                    pool.gather_recurrent_state(&indices)?,
                                ))
                            })
                            .transpose()?;

                        let mut gdn_cache = if packed_layout.is_some() {
                            GdnLayerCache::gathered(
                                pool.gather_conv_state(&indices)?,
                                pool.gather_recurrent_state(&indices)?,
                                pool.state_layout(),
                            )
                        } else {
                            GdnLayerCache::checkout(pool, &indices)?
                        };

                        let mut projected_stash = None;
                        let output = layer.forward_linear_with_stash(LinearForwardContext {
                            x: &xs,
                            normalized_x: normalized_x.as_ref(),
                            cache: &mut gdn_cache,
                            batch_kind: recurrent_metadata.batch_kind(),
                            checkpoint_lanes: if transition_gdn {
                                checkpoint_lanes
                            } else {
                                gdn_checkpoint_lanes
                            },
                            transition_checkpoints: transition_gdn,
                            packed_layout: packed_layout.as_ref(),
                            stash_out: gdn_stash.as_ref().map(|_| &mut projected_stash),
                        })?;
                        if let Some(stash) = gdn_stash.as_mut() {
                            let captured = projected_stash.ok_or_else(|| {
                                candle_core::Error::msg("GDN forward returned no stash")
                            })?;
                            let rollback = match (captured, stash_states) {
                                (
                                    GdnSpeculativeStash::Replay(projected),
                                    Some((conv_state, recurrent_state)),
                                ) => GdnLayerRollback::Replay {
                                    projected,
                                    conv_state,
                                    recurrent_state,
                                },
                                (GdnSpeculativeStash::Transition(transition), None) => {
                                    GdnLayerRollback::Transition(transition)
                                }
                                _ => candle_core::bail!(
                                    "GDN speculative capture mode does not match cache storage"
                                ),
                            };
                            stash.layers.push(GdnLayerStash {
                                layer_idx: i,
                                state_layout: pool.state_layout(),
                                rollback,
                            });
                        }

                        gdn_cache.commit(
                            pool,
                            &indices,
                            recurrent_metadata.state_indices_host(),
                        )?;
                        output
                    }
                };

                let deepstack = deepstack_indices
                    .as_ref()
                    .zip(deepstack_visual_embeds)
                    .filter(|(_, embeds)| i < embeds.len());
                if let Some(((idx, idx_expanded), embeds)) = deepstack {
                    xs =
                        self.deepstack_process(layer_output.add()?, idx, idx_expanded, &embeds[i])?;
                    normalized_x = None;
                } else {
                    let next_layer = self.layers.get(i + 1);
                    let next_norm = next_layer
                        .map(|next| &next.input_layernorm)
                        .unwrap_or(&self.norm);
                    if next_norm
                        .weight()
                        .device()
                        .same_device(layer_output.residual.device())
                    {
                        let plan = next_layer
                            .filter(|_| packed_layout.is_none())
                            .and_then(|next| next.input_quantization_plan(&layer_output.residual));
                        let (hidden, prepared) = layer_output.add_and_prepare(next_norm, plan)?;
                        xs = hidden;
                        normalized_x = Some(prepared);
                    } else {
                        xs = layer_output.add()?;
                        normalized_x = None;
                    }
                }
                if capture_taps && tap_layers.contains(&i) {
                    taps_all.push(xs.to_device(&self.device)?);
                }
            }
            if self.store_spec_hidden.load(Ordering::Relaxed) {
                *self.gdn_replay_stash.lock().expect("gdn stash poisoned") = gdn_stash;
            }
            let xs = match normalized_x {
                Some(prepared) if prepared.device().same_device(&self.device) => prepared
                    .dense()
                    .expect("final normalized model input must remain dense")
                    .clone(),
                _ => xs.to_device(&self.device)?.apply(&self.norm)?,
            };
            let store_spec = self.store_spec_hidden.load(Ordering::Relaxed);
            if store_spec {
                let full_capture = if recurrent_metadata
                    .as_ref()
                    .is_some_and(|metadata| metadata.batch_kind() == RecurrentBatchKind::Prefill)
                {
                    let positions = position_ids.to_device(&self.device)?;
                    let positions = match positions.rank() {
                        2 | 3 => positions,
                        rank => candle_core::bail!("unexpected Qwen3.5 position rank {rank}"),
                    };
                    Some(SpecCapture {
                        hidden: xs.clone(),
                        positions,
                        taps: taps_all.clone(),
                    })
                } else {
                    None
                };
                *self
                    .last_full_capture
                    .lock()
                    .expect("spec capture poisoned") = full_capture;
            }
            let xs = ctx.logits(&xs)?;
            if store_spec {
                let positions = position_ids.to_device(&self.device)?;
                let positions = match positions.rank() {
                    2 => ctx.logits(&positions.unsqueeze(D::Minus1)?)?.squeeze(2)?,
                    3 => {
                        let positions = positions.permute((1, 2, 0))?.contiguous()?;
                        ctx.logits(&positions)?.permute((2, 0, 1))?.contiguous()?
                    }
                    rank => candle_core::bail!("unexpected Qwen3.5 position rank {rank}"),
                };
                let taps = taps_all
                    .iter()
                    .map(|t| ctx.logits(t))
                    .collect::<Result<Vec<_>>>()?;
                *self
                    .last_spec_capture
                    .lock()
                    .expect("spec capture poisoned") = Some(SpecCapture {
                    hidden: xs.clone(),
                    positions,
                    taps,
                });
            }
            self.lm_head.forward(&xs)
        })();
        forward_result
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

impl crate::speculative::SpeculativeTargetMixin for Qwen3_5TextModel {
    fn supports_recurrent_speculative_checkpoints(&self) -> bool {
        Qwen3_5TextModel::supports_recurrent_speculative_checkpoints(self)
    }

    fn supports_recurrent_speculative_transitions(&self) -> bool {
        Qwen3_5TextModel::supports_recurrent_speculative_transitions(self)
    }

    fn reserve_recurrent_speculative_transition_storage(&self) -> Result<bool> {
        self.reserve_recurrent_transition_storage()
    }

    fn reserve_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        Qwen3_5TextModel::reserve_recurrent_decode_deferred_storage(self)
    }

    fn disable_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        Qwen3_5TextModel::disable_recurrent_decode_deferred_storage(self)
    }

    fn apply_recurrent_speculative_transitions_for_current_batch(&self) -> Result<bool> {
        self.apply_current_recurrent_transitions()
    }

    fn flush_recurrent_state_for_current_batch(&self) -> Result<()> {
        self.flush_current_recurrent_state()
    }

    fn flush_recurrent_speculative_transitions(&self, seq_ids: &[usize]) -> Result<()> {
        self.flush_recurrent_transitions_for_sequences(seq_ids)
    }
}

#[cfg(feature = "cuda")]
const SUPPORTS_CUDA_DECODE_GRAPHS: bool = true;

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
        let position_ids = text_positions.reshape((batch_size, seq_len))?;
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

    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        SUPPORTS_CUDA_DECODE_GRAPHS
    }
}

impl AnyMoeBaseModelMixin for Qwen3_5TextModel {}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::{
        gdn_transition_keep_rows, group_gdn_replay_batches, recurrent_checkpoint_devices_supported,
        refresh_gdn_stash_slots, should_stash_gdn_replay, terminal_gdn_transition_slots,
        GdnLayerRollback, GdnLayerStash, GdnReplayBatch, GdnReplayStash, SpecCapture,
        SpecGraphState,
    };
    use crate::{
        gdn::{GdnForwardStash, GdnTransitionStash},
        kv_cache::RecurrentStateLayout,
        pipeline::RecurrentBatchKind,
        speculative::SpeculativeGraphState,
    };

    #[cfg(feature = "cuda")]
    use super::SUPPORTS_CUDA_DECODE_GRAPHS;

    #[test]
    fn gdn_replay_batches_group_by_prefix_and_preserve_row_order() {
        let batches =
            group_gdn_replay_batches(&[(3, 4), (0, 2), (2, 4), (1, 1)], &[40, 41, 42, 43]).unwrap();
        assert_eq!(
            batches,
            vec![
                GdnReplayBatch {
                    keep_rows: 1,
                    batch_indices: vec![1],
                    slots: vec![41],
                },
                GdnReplayBatch {
                    keep_rows: 2,
                    batch_indices: vec![0],
                    slots: vec![40],
                },
                GdnReplayBatch {
                    keep_rows: 4,
                    batch_indices: vec![3, 2],
                    slots: vec![43, 42],
                },
            ]
        );
    }

    #[test]
    fn gdn_replay_batches_allow_an_all_accepted_empty_set() {
        assert!(group_gdn_replay_batches(&[], &[10, 11]).unwrap().is_empty());
    }

    #[test]
    fn gdn_graph_stash_slot_refresh_preserves_real_batch() {
        let mut stash = GdnReplayStash {
            slots: vec![1, 2, 3],
            layers: Vec::new(),
        };

        refresh_gdn_stash_slots(&mut stash, &[10, 11, 12, u32::MAX, u32::MAX]).unwrap();
        assert_eq!(stash.slots, [10, 11, 12]);
        assert!(refresh_gdn_stash_slots(&mut stash, &[20, 21]).is_err());
        assert_eq!(stash.slots, [10, 11, 12]);
    }

    #[test]
    fn terminal_transition_rows_select_exact_slots() {
        use crate::speculative::SpeculativeCommitRow;

        let rows = [
            SpeculativeCommitRow {
                batch_idx: 2,
                keep_rows: 3,
                accepted_all: true,
                terminal: true,
            },
            SpeculativeCommitRow {
                batch_idx: 0,
                keep_rows: 1,
                accepted_all: false,
                terminal: false,
            },
            SpeculativeCommitRow {
                batch_idx: 1,
                keep_rows: 2,
                accepted_all: false,
                terminal: true,
            },
        ];
        assert_eq!(
            terminal_gdn_transition_slots(&rows, &[10, 11, 12]).unwrap(),
            vec![12, 11]
        );
        assert!(terminal_gdn_transition_slots(&rows, &[10, 11]).is_err());
    }

    #[test]
    fn transition_commit_rows_require_a_unique_exact_cover() {
        use crate::speculative::SpeculativeCommitRow;

        let row = |batch_idx, keep_rows| SpeculativeCommitRow {
            batch_idx,
            keep_rows,
            accepted_all: false,
            terminal: false,
        };
        assert_eq!(
            gdn_transition_keep_rows(&[row(2, 3), row(0, 1), row(1, 2)], 3, 8).unwrap(),
            vec![1, 2, 3]
        );
        assert!(gdn_transition_keep_rows(&[row(0, 1), row(2, 3)], 3, 8).is_err());
        assert!(gdn_transition_keep_rows(&[row(0, 1), row(0, 2), row(2, 3)], 3, 8).is_err());
        assert!(gdn_transition_keep_rows(&[row(0, 0)], 1, 8).is_err());
        assert!(gdn_transition_keep_rows(&[row(0, 9)], 1, 8).is_err());
    }

    #[test]
    fn recurrent_checkpoint_device_gate_rejects_cpu_placement() {
        assert!(!recurrent_checkpoint_devices_supported(&[Device::Cpu]));
    }

    #[test]
    fn gdn_replay_stash_is_only_created_for_fallback_speculative_decode() {
        assert!(should_stash_gdn_replay(
            false,
            true,
            8,
            Some(RecurrentBatchKind::SpeculativeDecode),
            true,
        ));
        for (
            native_speculative_commit,
            store_spec_hidden,
            query_len,
            batch_kind,
            continuation_without_cache,
        ) in [
            (
                true,
                true,
                8,
                Some(RecurrentBatchKind::SpeculativeDecode),
                true,
            ),
            (
                false,
                false,
                8,
                Some(RecurrentBatchKind::SpeculativeDecode),
                true,
            ),
            (
                false,
                true,
                1,
                Some(RecurrentBatchKind::SpeculativeDecode),
                true,
            ),
            (false, true, 512, Some(RecurrentBatchKind::Prefill), true),
            (false, true, 8, Some(RecurrentBatchKind::Decode), true),
            (false, true, 8, None, true),
            (
                false,
                true,
                8,
                Some(RecurrentBatchKind::SpeculativeDecode),
                false,
            ),
        ] {
            assert!(!should_stash_gdn_replay(
                native_speculative_commit,
                store_spec_hidden,
                query_len,
                batch_kind,
                continuation_without_cache,
            ));
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA"]
    fn recurrent_checkpoint_device_gate_rejects_mixed_placement() -> candle_core::Result<()> {
        let cuda = Device::new_cuda(0)?;
        assert!(recurrent_checkpoint_devices_supported(
            std::slice::from_ref(&cuda)
        ));
        assert!(!recurrent_checkpoint_devices_supported(&[
            cuda,
            Device::Cpu
        ]));
        Ok(())
    }

    #[test]
    fn speculative_graph_state_narrows_a_bucket_to_the_live_batch() {
        let device = Device::Cpu;
        let mrope_capture = || SpecCapture {
            hidden: Tensor::zeros((16, 8, 32), DType::F32, &device).unwrap(),
            positions: Tensor::zeros((3, 16, 8), DType::U32, &device).unwrap(),
            taps: vec![Tensor::zeros((16, 8, 32), DType::F32, &device).unwrap()],
        };
        let text_capture = || SpecCapture {
            hidden: Tensor::zeros((16, 8, 32), DType::F32, &device).unwrap(),
            positions: Tensor::zeros((16, 8), DType::U32, &device).unwrap(),
            taps: vec![Tensor::zeros((16, 8, 32), DType::F32, &device).unwrap()],
        };
        let state = SpecGraphState {
            spec_capture: Some(text_capture()),
            full_capture: Some(mrope_capture()),
            gdn_stash: Some(GdnReplayStash {
                slots: (0..16).collect(),
                layers: vec![GdnLayerStash {
                    layer_idx: 2,
                    rollback: GdnLayerRollback::Replay {
                        projected: GdnForwardStash {
                            mixed_qkv: Tensor::zeros((16, 8, 24), DType::F32, &device).unwrap(),
                            convolved_qkv: Tensor::zeros((16, 8, 24), DType::F32, &device).unwrap(),
                            b: Tensor::zeros((16, 8, 4), DType::F32, &device).unwrap(),
                            a: Tensor::zeros((16, 8, 4), DType::F32, &device).unwrap(),
                        },
                        conv_state: Tensor::zeros((16, 24, 4), DType::F32, &device).unwrap(),
                        recurrent_state: Tensor::zeros((16, 2, 3, 4), DType::F32, &device).unwrap(),
                    },
                    state_layout: RecurrentStateLayout::GdnValueMajor,
                }],
            }),
        };

        let state = state.for_real_batch(9).unwrap();
        let state = state.as_any().downcast_ref::<SpecGraphState>().unwrap();

        let text_capture = state.spec_capture.as_ref().unwrap();
        assert_eq!(text_capture.hidden.dims(), &[9, 8, 32]);
        assert_eq!(text_capture.positions.dims(), &[9, 8]);
        assert_eq!(text_capture.taps[0].dims(), &[9, 8, 32]);
        let mrope_capture = state.full_capture.as_ref().unwrap();
        assert_eq!(mrope_capture.hidden.dims(), &[9, 8, 32]);
        assert_eq!(mrope_capture.positions.dims(), &[3, 9, 8]);
        assert_eq!(mrope_capture.taps[0].dims(), &[9, 8, 32]);
        let stash = state.gdn_stash.as_ref().unwrap();
        assert_eq!(stash.slots, (0..9).collect::<Vec<_>>());
        let layer = &stash.layers[0];
        let GdnLayerRollback::Replay {
            projected,
            conv_state,
            recurrent_state,
        } = &layer.rollback
        else {
            panic!("expected replay stash")
        };
        assert_eq!(projected.mixed_qkv.dims(), &[9, 8, 24]);
        assert_eq!(projected.convolved_qkv.dims(), &[9, 8, 24]);
        assert_eq!(projected.b.dims(), &[9, 8, 4]);
        assert_eq!(projected.a.dims(), &[9, 8, 4]);
        assert_eq!(conv_state.dims(), &[9, 24, 4]);
        assert_eq!(recurrent_state.dims(), &[9, 2, 3, 4]);
    }

    #[test]
    fn speculative_graph_state_narrows_direct_transition_slots() {
        let state = SpecGraphState {
            spec_capture: None,
            full_capture: None,
            gdn_stash: Some(GdnReplayStash {
                slots: (0..16).collect(),
                layers: vec![GdnLayerStash {
                    layer_idx: 2,
                    rollback: GdnLayerRollback::Transition(GdnTransitionStash),
                    state_layout: RecurrentStateLayout::GdnValueMajor,
                }],
            }),
        };

        let state = state.for_real_batch(9).unwrap();
        let state = state.as_any().downcast_ref::<SpecGraphState>().unwrap();
        let stash = state.gdn_stash.as_ref().unwrap();
        assert_eq!(stash.slots, (0..9).collect::<Vec<_>>());
        let GdnLayerRollback::Transition(_) = &stash.layers[0].rollback else {
            panic!("expected transition stash")
        };
        assert!(state.tensors().is_empty());
    }

    #[test]
    fn speculative_graph_state_rejects_a_larger_live_batch() {
        let state = SpecGraphState {
            spec_capture: None,
            full_capture: None,
            gdn_stash: Some(GdnReplayStash {
                slots: vec![10],
                layers: Vec::new(),
            }),
        };
        assert!(state.for_real_batch(2).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn text_architecture_supports_cuda_decode_graphs() {
        const _: () = assert!(SUPPORTS_CUDA_DECODE_GRAPHS);
    }
}
