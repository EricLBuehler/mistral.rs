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
    ColumnParallelLayer, PackedOutputLayout, QuantMethod, QuantizedConfig, ReplicatedLayer,
    RowParallelLayer, ShardedVarBuilder,
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
        let (q_gate, k, v) = if let Some(merged_qkv) = &self.merged_qkv {
            let [q_gate, k, v]: [Tensor; 3] = merged_qkv.forward(x)?.try_into().map_err(|_| {
                candle_core::Error::msg("packed QKV returned the wrong output count")
            })?;
            (q_gate, k, v)
        } else {
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?
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

struct LinearForwardContext<'a> {
    x: &'a Tensor,
    normalized_x: Option<&'a Tensor>,
    cache: &'a mut GdnLayerCache,
    batch_kind: RecurrentBatchKind,
    checkpoint_lanes: usize,
    packed_query_lens: Option<&'a [usize]>,
    stash_out: Option<&'a mut Option<GdnForwardStash>>,
}

impl DecoderLayerOutput {
    fn add(self) -> Result<Tensor> {
        self.branch + self.residual
    }

    fn add_and_norm(self, norm: &GemmaRmsNorm) -> Result<(Tensor, Tensor)> {
        norm.forward_add_rms_norm(&self.branch, &self.residual)
    }
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
        normalized_x: Option<&Tensor>,
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
        let normalized_x = match normalized_x {
            Some(normalized_x) => normalized_x.clone(),
            None => self.input_layernorm.forward(x)?,
        };
        let attn_out = attn.forward(
            &normalized_x,
            attention_mask,
            cos_sin,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let (x, normed) = self
            .post_attention_layernorm
            .forward_add_rms_norm(&attn_out, residual)?;
        let ffn_out = self.mlp.forward(&normed)?;
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
            packed_query_lens,
            stash_out,
        } = context;
        let gdn = match &self.layer_impl {
            LayerImpl::LinearAttention(gdn) => gdn,
            _ => candle_core::bail!("Expected linear attention layer"),
        };
        let residual = x;
        let normalized_x = match normalized_x {
            Some(normalized_x) => normalized_x.clone(),
            None => self.input_layernorm.forward(x)?,
        };
        let gdn_out = if let Some(query_lens) = packed_query_lens {
            forward_packed_gdn(gdn, &normalized_x, cache, batch_kind, query_lens)?
        } else {
            gdn.forward_with_stash(
                &normalized_x,
                cache,
                batch_kind,
                checkpoint_lanes,
                stash_out,
            )?
        };
        let (x, normed) = self
            .post_attention_layernorm
            .forward_add_rms_norm(&gdn_out, residual)?;
        let ffn_out = self.mlp.forward(&normed)?;
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

/// Target activations captured for the MTP proposer: hidden states after the final norm and the
/// MRoPE position ids they were computed at, `[b, rows, hidden]` / `[3, b, rows]`.
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
    pub(super) projected: GdnForwardStash,
    pub(super) conv_state: Tensor,
    pub(super) recurrent_state: Tensor,
    pub(super) state_layout: crate::kv_cache::RecurrentStateLayout,
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
    checkpoint_gdn: bool,
    store_spec_hidden: bool,
    query_len: usize,
    batch_kind: Option<RecurrentBatchKind>,
    continuation_without_cache: bool,
) -> bool {
    !checkpoint_gdn
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
    capture.positions = narrow_spec_graph_tensor(
        &capture.positions,
        1,
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
        layer.projected.mixed_qkv = narrow_spec_graph_tensor(
            &layer.projected.mixed_qkv,
            0,
            captured_batch,
            real_batch,
            "mixed_qkv",
        )?;
        layer.projected.convolved_qkv = narrow_spec_graph_tensor(
            &layer.projected.convolved_qkv,
            0,
            captured_batch,
            real_batch,
            "convolved_qkv",
        )?;
        layer.projected.b =
            narrow_spec_graph_tensor(&layer.projected.b, 0, captured_batch, real_batch, "b")?;
        layer.projected.a =
            narrow_spec_graph_tensor(&layer.projected.a, 0, captured_batch, real_batch, "a")?;
        layer.conv_state = narrow_spec_graph_tensor(
            &layer.conv_state,
            0,
            captured_batch,
            real_batch,
            "conv_state",
        )?;
        layer.recurrent_state = narrow_spec_graph_tensor(
            &layer.recurrent_state,
            0,
            captured_batch,
            real_batch,
            "recurrent_state",
        )?;
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
                out.push(layer.projected.mixed_qkv.clone());
                out.push(layer.projected.convolved_qkv.clone());
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
            for tap in capture.taps.iter_mut() {
                *tap = next()?;
            }
        }
        if let Some(stash) = state.gdn_stash.as_mut() {
            for layer in stash.layers.iter_mut() {
                layer.projected.mixed_qkv = next()?;
                layer.projected.convolved_qkv = next()?;
                layer.projected.b = next()?;
                layer.projected.a = next()?;
                layer.conv_state = next()?;
                layer.recurrent_state = next()?;
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
        let devices = stash.layers.iter().fold(Vec::new(), |mut devices, layer| {
            let device = layer.projected.mixed_qkv.device();
            if !devices
                .iter()
                .any(|cached: &Device| cached.same_device(device))
            {
                devices.push(device.clone());
            }
            devices
        });
        let fused_commit_supported = !stash.layers.is_empty()
            && stash
                .layers
                .iter()
                .all(|layer| layer.projected.mixed_qkv.device().is_cuda())
            && {
                let hybrid_cache = self.cache.hybrid();
                stash.layers.iter().all(|layer| {
                    let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                        (&self.layers[layer.layer_idx].layer_impl, hybrid_cache.get(layer.layer_idx))
                    else {
                        return false;
                    };
                    pool.state_layout() == layer.state_layout
                        && gdn.speculative_state_commit_supported(
                            &layer.projected,
                            &layer.conv_state,
                            &layer.recurrent_state,
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
                    .position(|device| device.same_device(layer.projected.mixed_qkv.device()))
                    .expect("stashed GDN layer device was collected above");
                let indices = &commit_indices[device_idx];
                if !gdn.commit_state_batch_from_stash_cuda(
                    &layer.projected,
                    &layer.conv_state,
                    &layer.recurrent_state,
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
                .position(|device| device.same_device(layer.projected.mixed_qkv.device()))
                .expect("stashed GDN layer device was collected above");
            for (group_idx, batch) in batches.iter().enumerate() {
                let indices = &replay_indices[group_idx][device_idx];
                let mut cache = GdnLayerCache::gathered(
                    index_select_replay_rows(&layer.conv_state, &indices.batch_indices)?,
                    index_select_replay_rows(&layer.recurrent_state, &indices.batch_indices)?,
                    layer.state_layout,
                );
                gdn.advance_state_batch_from_stash(
                    &layer.projected,
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

    pub(super) fn install_spec_graph_state(&self, state: &SpecGraphState) {
        let spec_capture = state.spec_capture.clone();
        let full_capture = state.full_capture.clone();
        let mut gdn_stash = state.gdn_stash.clone();
        let slots = self
            .cache
            .hybrid()
            .state_indices_host()
            .map(ToOwned::to_owned);
        if let (Some(stash), Some(slots)) = (gdn_stash.as_mut(), slots.as_deref()) {
            stash.slots = slots.to_vec();
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

        // Checkpointed CUDA verification rolls back by lane; fallback verification retains replay inputs.
        let checkpoint_gdn = checkpoint_lanes > 1
            && (1..=checkpoint_lanes).contains(&query_len)
            && self.supports_recurrent_speculative_checkpoints_with_cache(&hybrid_cache)
            && recurrent_metadata.as_ref().is_some_and(|metadata| {
                metadata.batch_kind() == RecurrentBatchKind::SpeculativeDecode
            });
        let gdn_checkpoint_lanes = if checkpoint_gdn { checkpoint_lanes } else { 1 };
        let stash_gdn = should_stash_gdn_replay(
            checkpoint_gdn,
            self.store_spec_hidden.load(Ordering::Relaxed),
            query_len,
            recurrent_metadata
                .as_ref()
                .map(|metadata| metadata.batch_kind()),
            ctx.paged_input_metadata().is_some_and(|meta| {
                !meta.is_first_prompt_chunk && meta.num_cached_tokens.is_none()
            }),
        );
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

        let mut normalized_x = None;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            if normalized_x
                .as_ref()
                .is_some_and(|normed: &Tensor| !normed.device().same_device(xs.device()))
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
                    let indices = hybrid_cache.state_indices_for_layer(i)?.ok_or_else(|| {
                        candle_core::Error::msg(format!(
                            "Hybrid cache layer {i} is missing recurrent state indices"
                        ))
                    })?;
                    let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(i) else {
                        candle_core::bail!(
                            "Hybrid cache layer {i} is not recurrent for a linear-attention layer."
                        );
                    };
                    let stash_states = gdn_stash
                        .as_ref()
                        .map(|_| {
                            candle_core::Result::Ok((
                                pool.gather_conv_state(&indices)?,
                                pool.gather_recurrent_state(&indices)?,
                            ))
                        })
                        .transpose()?;

                    let mut gdn_cache = if packed_query_lens.is_some() {
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
                        checkpoint_lanes: gdn_checkpoint_lanes,
                        packed_query_lens: packed_query_lens.as_deref(),
                        stash_out: stash_states.is_some().then_some(&mut projected_stash),
                    })?;
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
                            state_layout: pool.state_layout(),
                        });
                    }

                    gdn_cache.commit(pool, &indices, recurrent_metadata.state_indices_host())?;
                    output
                }
            };

            let deepstack = deepstack_indices
                .as_ref()
                .zip(deepstack_visual_embeds)
                .filter(|(_, embeds)| i < embeds.len());
            if let Some(((idx, idx_expanded), embeds)) = deepstack {
                xs = self.deepstack_process(layer_output.add()?, idx, idx_expanded, &embeds[i])?;
                normalized_x = None;
            } else {
                let next_norm = self
                    .layers
                    .get(i + 1)
                    .map(|next| &next.input_layernorm)
                    .unwrap_or(&self.norm);
                if next_norm
                    .weight()
                    .device()
                    .same_device(layer_output.residual.device())
                {
                    let (hidden, normed) = layer_output.add_and_norm(next_norm)?;
                    xs = hidden;
                    normalized_x = Some(normed);
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
            Some(normed) if normed.device().same_device(&self.device) => normed,
            _ => xs.to_device(&self.device)?.apply(&self.norm)?,
        };
        let store_spec = self.store_spec_hidden.load(Ordering::Relaxed);
        if store_spec {
            let full_capture = if recurrent_metadata
                .as_ref()
                .is_some_and(|metadata| metadata.batch_kind() == RecurrentBatchKind::Prefill)
            {
                Some(SpecCapture {
                    hidden: xs.clone(),
                    positions: position_ids.to_device(&self.device)?,
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
            // Reduce the position ids exactly like the hidden rows so they stay aligned
            let positions = position_ids
                .to_device(&self.device)?
                .permute((1, 2, 0))?
                .contiguous()?;
            let positions = ctx.logits(&positions)?.permute((2, 0, 1))?.contiguous()?;
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
        group_gdn_replay_batches, recurrent_checkpoint_devices_supported, should_stash_gdn_replay,
        GdnLayerStash, GdnReplayBatch, GdnReplayStash, SpecCapture, SpecGraphState,
    };
    use crate::{
        gdn::GdnForwardStash, kv_cache::RecurrentStateLayout, pipeline::RecurrentBatchKind,
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
            checkpoint_gdn,
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
                checkpoint_gdn,
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
        let capture = || SpecCapture {
            hidden: Tensor::zeros((16, 8, 32), DType::F32, &device).unwrap(),
            positions: Tensor::zeros((3, 16, 8), DType::U32, &device).unwrap(),
            taps: vec![Tensor::zeros((16, 8, 32), DType::F32, &device).unwrap()],
        };
        let state = SpecGraphState {
            spec_capture: Some(capture()),
            full_capture: Some(capture()),
            gdn_stash: Some(GdnReplayStash {
                slots: (0..16).collect(),
                layers: vec![GdnLayerStash {
                    layer_idx: 2,
                    projected: GdnForwardStash {
                        mixed_qkv: Tensor::zeros((16, 8, 24), DType::F32, &device).unwrap(),
                        convolved_qkv: Tensor::zeros((16, 8, 24), DType::F32, &device).unwrap(),
                        b: Tensor::zeros((16, 8, 4), DType::F32, &device).unwrap(),
                        a: Tensor::zeros((16, 8, 4), DType::F32, &device).unwrap(),
                    },
                    conv_state: Tensor::zeros((16, 24, 4), DType::F32, &device).unwrap(),
                    recurrent_state: Tensor::zeros((16, 2, 3, 4), DType::F32, &device).unwrap(),
                    state_layout: RecurrentStateLayout::GdnValueMajor,
                }],
            }),
        };

        let state = state.for_real_batch(9).unwrap();
        let state = state.as_any().downcast_ref::<SpecGraphState>().unwrap();

        for capture in [
            state.spec_capture.as_ref().unwrap(),
            state.full_capture.as_ref().unwrap(),
        ] {
            assert_eq!(capture.hidden.dims(), &[9, 8, 32]);
            assert_eq!(capture.positions.dims(), &[3, 9, 8]);
            assert_eq!(capture.taps[0].dims(), &[9, 8, 32]);
        }
        let stash = state.gdn_stash.as_ref().unwrap();
        assert_eq!(stash.slots, (0..9).collect::<Vec<_>>());
        let layer = &stash.layers[0];
        assert_eq!(layer.projected.mixed_qkv.dims(), &[9, 8, 24]);
        assert_eq!(layer.projected.convolved_qkv.dims(), &[9, 8, 24]);
        assert_eq!(layer.projected.b.dims(), &[9, 8, 4]);
        assert_eq!(layer.projected.a.dims(), &[9, 8, 4]);
        assert_eq!(layer.conv_state.dims(), &[9, 24, 4]);
        assert_eq!(layer.recurrent_state.dims(), &[9, 2, 3, 4]);
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
        assert!(SUPPORTS_CUDA_DECODE_GRAPHS);
    }
}
