//! Qwen4Exp core decoder (Phase 5 bring-up).
//!
//! Wires the shared sigmoid-gated GDN, the gated-residual hyper-connections, and the
//! Qwen3.5/Qwen3-Next MoE equations into the reference llama.cpp `qwen4exp` decoder order:
//!
//! ```text
//! wide residual (hc identical copies of the embeddings)
//!   -> attention hyper-connection mixer
//!   -> GDN or full-attention branch
//!   -> attention residual injection
//!   -> MoE hyper-connection mixer
//!   -> routed and shared MoE
//!   -> MoE residual injection
//! final hyper-connection mixer -> language-model head
//! ```
//!
//! Full-attention layers run the tested QSA sparse-attention orchestration end to end: the
//! main projection with the converter's per-head interleaved query/gate layout, interleaved
//! MRoPE rotation at each token's position, per-sequence main K/V and raw indexer-key caches,
//! block pooling and scoring, bounded block selection, sparse K/V gathering, and the
//! sigmoid-gated output projection. QSA caches are keyed by the batch's recurrent state
//! slots, so one sequence identity is shared with the GDN layers. Packed prefill routes
//! each logical sequence through the tested packed recurrent, QSA, and PLE plumbing keyed
//! by that same identity. Speculative decoding and paged attention are not wired yet, so
//! this model must not be presented as fully validated support for real Qwen3.8 Flash Next
//! checkpoints.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use crate::amoe::AnyMoeBaseModelMixin;
use crate::device_map::DeviceMapper;
use crate::gdn::{
    try_forward_grouped_packed_gdn, GatedDeltaNet, GdnConfig, GdnInputProjectionKind,
    GdnLayerCache, PackedGdnLayout,
};
use crate::kv_cache::{
    HybridAuxiliarySnapshot, HybridAuxiliaryState, HybridCache, HybridCacheConfig,
    HybridLayerCache, HybridLayerType, RecurrentLayerConfig, RecurrentStateSpec,
};
use crate::layers::{
    contains_tensor_or_weight_source, embedding_with_legacy_tied_uqff, linear_no_bias, Mlp,
};
use crate::moe::{ExpertProjNames, MoEExperts, MoEExpertsConfig};
use crate::paged_attention::{AttentionImplementation, KvCacheLayout, ModelConfigMetadata};
use crate::pipeline::{
    text_models_inputs_processor::FlashParams, EitherCache, IsqModel, ModelForwardContext,
    NormalLoadingMetadata, NormalModel, RecurrentBatchKind, RecurrentMetadata,
};
use crate::utils::unvarbuilder::UnVarBuilder;
use crate::xlora_models::NonGranularState;

use super::config::{Config, LayerType};
use super::hyper_connection::GatedResidual;
use super::ple::{PleEmbedding, PleHasher, PleLayer, PleState};
use super::qsa::{QsaAttention, QsaAttentionSnapshot, QsaQueryGateLayout};

/// Interleaved MRoPE cosine/sine tables for the configured rotary width.
///
/// Produces `[positions.len(), rotary_dim / 2]` tables where pair `j` rotates the adjacent head
/// dimensions `(2j, 2j + 1)` with frequency `theta ** (-2j / rotary_dim)`. Text tokens apply the
/// same offset to every MRoPE section, so the section layout does not change the tables.
pub(crate) fn interleaved_mrope_tables(
    theta: f32,
    rotary_dim: usize,
    positions: &[u32],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        candle_core::bail!("Qwen4Exp interleaved MRoPE width must be positive and even");
    }
    let half = rotary_dim / 2;
    let table_len = positions
        .len()
        .checked_mul(half)
        .ok_or_else(|| candle_core::Error::msg("Qwen4Exp MRoPE table length overflow"))?;
    let mut cos_data = Vec::with_capacity(table_len);
    let mut sin_data = Vec::with_capacity(table_len);
    for &position in positions {
        for pair in 0..half {
            let angle = position as f32 * theta.powf(-(2.0 * pair as f32) / rotary_dim as f32);
            cos_data.push(angle.cos());
            sin_data.push(angle.sin());
        }
    }
    Ok((
        Tensor::from_vec(cos_data, (positions.len(), half), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin_data, (positions.len(), half), device)?.to_dtype(dtype)?,
    ))
}

/// Compute selected interleaved Qwen3-VL MRoPE tables for 3D temporal/height/width positions.
///
/// The frequency pairs are interleaved as `T/H/W` triples. The temporal component supplies the
/// remaining pairs after the configured height and width section counts are exhausted.
pub(crate) fn sectioned_interleaved_mrope_tables(
    theta: f32,
    rotary_dim: usize,
    mrope_section: &[usize],
    positions: &[Vec<Vec<u32>>],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || mrope_section.len() != 3 {
        candle_core::bail!(
            "Qwen4Exp sectioned interleaved MRoPE requires an even rotary width and three sections"
        );
    }
    let half = rotary_dim / 2;
    if mrope_section
        .iter()
        .try_fold(0usize, |sum, width| sum.checked_add(*width))
        != Some(half)
    {
        candle_core::bail!("Qwen4Exp MRoPE section widths must sum to the rotary half width");
    }
    let [temporal, height, width] = positions else {
        candle_core::bail!(
            "Qwen4Exp sectioned MRoPE requires temporal, height, and width positions"
        );
    };
    let batch = temporal.len();
    let tokens = temporal.first().map_or(0, Vec::len);
    if batch == 0
        || tokens == 0
        || height.len() != batch
        || width.len() != batch
        || temporal.iter().any(|row| row.len() != tokens)
        || height.iter().any(|row| row.len() != tokens)
        || width.iter().any(|row| row.len() != tokens)
    {
        candle_core::bail!(
            "Qwen4Exp sectioned MRoPE positions must be non-empty [3, batch, tokens] rows"
        );
    }

    let table_len = batch
        .checked_mul(tokens)
        .and_then(|size| size.checked_mul(half))
        .ok_or_else(|| candle_core::Error::msg("Qwen4Exp MRoPE table length overflow"))?;
    let mut cos_data = Vec::with_capacity(table_len);
    let mut sin_data = Vec::with_capacity(table_len);
    for row in 0..batch {
        for token in 0..tokens {
            for pair in 0..half {
                let component = if pair % 3 == 1 && pair / 3 < mrope_section[1] {
                    height[row][token]
                } else if pair % 3 == 2 && pair / 3 < mrope_section[2] {
                    width[row][token]
                } else {
                    temporal[row][token]
                };
                let angle = component as f32 * theta.powf(-(2.0 * pair as f32) / rotary_dim as f32);
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }
    }
    Ok((
        Tensor::from_vec(cos_data, (batch, tokens, half), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin_data, (batch, tokens, half), device)?.to_dtype(dtype)?,
    ))
}

/// Detect the GDN input-projection layout from tensor presence, matching the shared loader's
/// conventions (fused `qkvz`+`ba`, split `qkv`/`z`/`b`/`a`, or split QKV with grouped B/A).
fn gdn_input_projection_kind(vb: &ShardedVarBuilder) -> GdnInputProjectionKind {
    // GGUF qwen4exp archives split the fused qkvz/ba projections into qkv/z/b/a parts.
    if contains_tensor_or_weight_source(vb, "in_proj_a.weight")
        && (contains_tensor_or_weight_source(vb, "in_proj_qkv.weight")
            || contains_tensor_or_weight_source(vb, "in_proj_z.weight")
            || contains_tensor_or_weight_source(vb, "in_proj_b.weight")
            || contains_tensor_or_weight_source(vb, "in_proj_ba.weight"))
    {
        GdnInputProjectionKind::Split
    } else if contains_tensor_or_weight_source(vb, "in_proj_qkv.weight") {
        GdnInputProjectionKind::SplitQkvzGroupedBa
    } else {
        GdnInputProjectionKind::Grouped
    }
}

/// Sparse MoE block reusing the Qwen3.5/Qwen3-Next equations: softmax router with the configured
/// top-k and renormalization, stacked routed experts, and a shared expert with a sigmoid gate.
#[allow(dead_code)]
struct SparseMoeBlock {
    gate: Linear,
    gate_lora: Option<Arc<mistralrs_quant::LoraSiteHandle>>,
    experts: MoEExperts,
    shared_expert: Mlp,
    shared_expert_gate: Linear,
    shared_expert_gate_lora: Option<Arc<mistralrs_quant::LoraSiteHandle>>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

#[allow(dead_code)]
impl SparseMoeBlock {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(real_device);

        let gate_vb = vb.pp("gate").set_device(layer_device.clone());
        let gate = linear_no_bias(cfg.hidden_size, cfg.num_experts, gate_vb.clone())?;
        let gate_lora = mistralrs_quant::register_dynamic_lora_site(
            &gate_vb,
            mistralrs_quant::LoraLinearSpec::replicated(cfg.hidden_size, cfg.num_experts),
        )?;

        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
            expert_proj_names: ExpertProjNames::DEFAULT,
        };

        let experts = MoEExperts::new(
            &moe_cfg,
            vb.clone(),
            layer_device.clone(),
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        let shared_expert = Mlp::new(
            vb.pp("shared_expert"),
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )?;

        let shared_expert_gate_vb = vb.pp("shared_expert_gate");
        let mut seg_w = shared_expert_gate_vb.get((1, cfg.hidden_size), "weight")?;
        if loading_isq {
            seg_w = seg_w.to_device(&layer_device)?;
        }
        let shared_expert_gate = Linear::new(seg_w, None);
        let shared_expert_gate_lora = mistralrs_quant::register_dynamic_lora_site(
            &shared_expert_gate_vb.set_device(layer_device),
            mistralrs_quant::LoraLinearSpec::replicated(cfg.hidden_size, 1),
        )?;

        Ok(Self {
            gate,
            gate_lora,
            experts,
            shared_expert,
            shared_expert_gate,
            shared_expert_gate_lora,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        let router_logits = self.gate.forward(&xs_flat)?;
        let router_logits = match &self.gate_lora {
            Some(site) => mistralrs_quant::apply_dynamic_lora_delta(site, &xs_flat, router_logits)?,
            None => router_logits,
        };
        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.num_experts_per_tok,
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: self.norm_topk_prob,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )?;

        let mut y = self.experts.forward(xs, topk.values, &topk.indices)?;
        y = y.reshape((b_size, seq_len, hidden_dim))?;

        let shared_out = self.shared_expert.forward(xs)?;

        let shared_gate = self.shared_expert_gate.forward(&xs_flat)?;
        let shared_gate = match &self.shared_expert_gate_lora {
            Some(site) => mistralrs_quant::apply_dynamic_lora_delta(site, &xs_flat, shared_gate)?,
            None => shared_gate,
        };
        let shared_gate = candle_nn::ops::sigmoid(&shared_gate)?;
        let shared_gate = shared_gate.reshape((b_size, seq_len, 1))?;
        let shared_out = shared_out.broadcast_mul(&shared_gate)?;

        y + shared_out
    }
}

#[allow(dead_code, clippy::large_enum_variant)]
enum DecoderBranch {
    Linear(GatedDeltaNet),
    Attention(Arc<Mutex<QsaAttention>>),
}

/// Explicit position representations accepted by the embedding-based decoder path.
#[allow(dead_code)]
pub(crate) enum MropePositionOverride {
    Scalar(Vec<Vec<u32>>),
    Sectioned(Vec<Vec<Vec<u32>>>),
}

enum RopeInputs {
    Scalar {
        positions: Vec<Vec<u32>>,
        cos: Tensor,
        sin: Tensor,
    },
    Sectioned {
        positions: Vec<Vec<u32>>,
        cos: Tensor,
        sin: Tensor,
    },
}

/// One configured PLE layer: the tested projection/convolution component plus its
/// per-sequence hashing and convolution state, keyed by recurrent state slot.
struct PleInjection {
    component: PleLayer,
    state: Arc<Mutex<PleState>>,
}

/// Shared PLE hasher and embedding table plus one transactional injection per
/// configured PLE layer.
struct PleModules {
    hasher: PleHasher,
    embedding: PleEmbedding,
    injections: Vec<(usize, PleInjection)>,
}

impl PleModules {
    fn injection_for(&self, layer_idx: usize) -> Option<&PleInjection> {
        self.injections
            .iter()
            .find(|(layer, _)| *layer == layer_idx)
            .map(|(_, injection)| injection)
    }
}

/// Keeps one QSA attention layer's main K/V and raw indexer-key caches in sync with the
/// hybrid cache's per-slot reset and release lifecycle.
struct QsaAuxiliaryState {
    attention: Arc<Mutex<QsaAttention>>,
}

impl std::fmt::Debug for QsaAuxiliaryState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QsaAuxiliaryState").finish_non_exhaustive()
    }
}

impl HybridAuxiliaryState for QsaAuxiliaryState {
    fn snapshot_slot(&self, slot_idx: usize) -> Result<HybridAuxiliarySnapshot> {
        let attention = self
            .attention
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp QSA layer mutex is poisoned"))?;
        Ok(Arc::new(attention.snapshot_sequence(slot_idx)))
    }

    fn validate_restore_slot(
        &self,
        _slot_idx: usize,
        snapshot: &HybridAuxiliarySnapshot,
    ) -> Result<()> {
        let snapshot = snapshot
            .downcast_ref::<QsaAttentionSnapshot>()
            .ok_or_else(|| {
                candle_core::Error::msg("Qwen4Exp QSA auxiliary snapshot type mismatch")
            })?;
        let attention = self
            .attention
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp QSA layer mutex is poisoned"))?;
        attention.validate_restore_sequence(snapshot)
    }

    fn restore_slot(&self, slot_idx: usize, snapshot: &HybridAuxiliarySnapshot) -> Result<()> {
        let snapshot = snapshot
            .downcast_ref::<QsaAttentionSnapshot>()
            .ok_or_else(|| {
                candle_core::Error::msg("Qwen4Exp QSA auxiliary snapshot type mismatch")
            })?;
        let mut attention = self
            .attention
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp QSA layer mutex is poisoned"))?;
        attention.restore_sequence(slot_idx, snapshot)
    }

    fn reset_slot(&self, slot_idx: usize) -> Result<()> {
        let mut attention = self
            .attention
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp QSA layer mutex is poisoned"))?;
        attention.reset_sequence(slot_idx);
        Ok(())
    }

    fn release_slot(&self, slot_idx: usize) -> Result<()> {
        let mut attention = self
            .attention
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp QSA layer mutex is poisoned"))?;
        attention.release_sequence(slot_idx);
        Ok(())
    }

    fn validate_truncate_slot(&self, slot_idx: usize, len: usize) -> Result<()> {
        let attention = self
            .attention
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp QSA layer mutex is poisoned"))?;
        attention.validate_truncate_sequence(slot_idx, len)
    }

    fn truncate_slot(&self, slot_idx: usize, len: usize) -> Result<()> {
        let mut attention = self
            .attention
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp QSA layer mutex is poisoned"))?;
        attention.truncate_sequence(slot_idx, len)
    }

    fn clear(&self) -> Result<()> {
        let mut attention = self
            .attention
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp QSA layer mutex is poisoned"))?;
        attention.clear_sequences();
        Ok(())
    }
}

/// Keeps one PLE layer's predecessor and dilated-convolution histories in sync with the
/// hybrid cache's per-slot reset and release lifecycle.
struct PleAuxiliaryState {
    state: Arc<Mutex<PleState>>,
}

impl std::fmt::Debug for PleAuxiliaryState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PleAuxiliaryState").finish_non_exhaustive()
    }
}

impl HybridAuxiliaryState for PleAuxiliaryState {
    fn snapshot_slot(&self, slot_idx: usize) -> Result<HybridAuxiliarySnapshot> {
        let state = self
            .state
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp PLE layer mutex is poisoned"))?;
        Ok(Arc::new(state.snapshot(slot_idx)))
    }

    fn validate_restore_slot(
        &self,
        _slot_idx: usize,
        snapshot: &HybridAuxiliarySnapshot,
    ) -> Result<()> {
        let snapshot = snapshot
            .downcast_ref::<super::ple::PleStateSnapshot>()
            .ok_or_else(|| {
                candle_core::Error::msg("Qwen4Exp PLE auxiliary snapshot type mismatch")
            })?;
        let state = self
            .state
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp PLE layer mutex is poisoned"))?;
        state.validate_snapshot(snapshot)
    }

    fn restore_slot(&self, slot_idx: usize, snapshot: &HybridAuxiliarySnapshot) -> Result<()> {
        let snapshot = snapshot
            .downcast_ref::<super::ple::PleStateSnapshot>()
            .ok_or_else(|| {
                candle_core::Error::msg("Qwen4Exp PLE auxiliary snapshot type mismatch")
            })?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp PLE layer mutex is poisoned"))?;
        state.restore(slot_idx, snapshot)
    }

    fn reset_slot(&self, slot_idx: usize) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp PLE layer mutex is poisoned"))?;
        state.reset(slot_idx);
        Ok(())
    }

    fn release_slot(&self, slot_idx: usize) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp PLE layer mutex is poisoned"))?;
        state.release(slot_idx);
        Ok(())
    }

    fn clear(&self) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp PLE layer mutex is poisoned"))?;
        state.clear();
        Ok(())
    }
}

#[allow(dead_code)]
struct DecoderLayer {
    attn_hc: GatedResidual,
    ffn_hc: GatedResidual,
    moe: SparseMoeBlock,
    branch: DecoderBranch,
}

/// Qwen4Exp core decoder (bring-up). Not registered with any loader or GGUF adapter yet.
#[allow(dead_code)]
pub(crate) struct Model {
    embed_tokens: Arc<dyn QuantMethod>,
    layers: Vec<DecoderLayer>,
    final_hc: GatedResidual,
    lm_head: Arc<dyn QuantMethod>,
    dtype: DType,
    kv_cache: EitherCache,
    device: Device,
    cfg: ModelConfigMetadata,
    hc_count: usize,
    rope_theta: f32,
    rotary_dim: usize,
    mrope_section: Vec<usize>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    max_seq_len: usize,
    ple: Option<PleModules>,
}

#[allow(dead_code)]
impl Model {
    pub(crate) fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        cfg.validate()?;
        if matches!(attention_mechanism, AttentionImplementation::PagedAttention) {
            candle_core::bail!(
                "Qwen4Exp paged attention requires indexed sparse K/V gathering from paged blocks, which is not wired yet"
            );
        }

        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");

        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }

        let mapper = normal_loading_metadata.mapper;
        let dtype = vb_m.dtype();
        let loading_isq = normal_loading_metadata.loading_isq;
        let world_size = mapper.get_comm_for(0)?.world_size();

        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), loading_isq),
            cfg.tie_word_embeddings
                .then(|| mapper.set_nm_device(vb_lm_head.clone(), loading_isq)),
            &cfg.quantization_config,
        )?;

        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, loading_isq),
            )?
        } else {
            embed_tokens.clone()
        };

        // PLE is constructed per checkpoint configuration. The table stays a QuantMethod so
        // the GGUF loader can keep it quantized/mapped; only gathered rows ever cross to the
        // layer's compute device and the table is never dequantized or materialized densely.
        let mut ple = if cfg.ple_layer_ids.is_empty() {
            None
        } else {
            let hasher = PleHasher::new(cfg)?;
            let head_count = hasher.head_count();
            let head_dim = cfg.ple_embed_dim / head_count;
            let mut table_rows = 0usize;
            for (offset, vocab_size) in cfg.ple_head_offsets.iter().zip(&cfg.ple_head_vocab_sizes) {
                let rows = offset.checked_add(*vocab_size).ok_or_else(|| {
                    candle_core::Error::msg("Qwen4Exp PLE head row range overflow")
                })?;
                let rows = usize::try_from(rows).map_err(|_| {
                    candle_core::Error::msg("Qwen4Exp PLE table rows do not fit usize")
                })?;
                table_rows = table_rows.max(rows);
            }
            if table_rows == 0 {
                candle_core::bail!("Qwen4Exp PLE head ranges produce no embedding table rows");
            }
            // ReplicatedLayer stores weights as (out, in), so the table loads with
            // `table_rows` rows of width `head_dim` for the embedding row gather.
            let table = ReplicatedLayer::new(
                head_dim,
                table_rows,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_m.pp("per_layer_token_embd"), loading_isq),
            )?;
            Some(PleModules {
                hasher,
                embedding: PleEmbedding::new(table, head_count, head_dim)?,
                injections: Vec::new(),
            })
        };

        let final_hc = GatedResidual::new(
            cfg,
            mapper.set_nm_device(vb_m.pp("hc_head"), loading_isq),
            false,
        )?;

        let rotary_dim = (cfg.head_dim as f64 * cfg.rope_parameters.partial_rotary_factor) as usize;
        let rope_theta = cfg.rope_parameters.rope_theta as f32;
        let num_linear = cfg
            .layer_types
            .iter()
            .filter(|layer_type| matches!(layer_type, LayerType::LinearAttention))
            .count();
        let num_full = cfg.layer_types.len() - num_linear;
        let ple_note = if cfg.ple_layer_ids.is_empty() {
            String::new()
        } else {
            format!(", PLE injection at layers {:?}", cfg.ple_layer_ids)
        };
        tracing::info!(
            "Qwen4Exp bring-up: {num_linear} GDN and {num_full} QSA sparse-attention layers{ple_note}"
        );

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (layer_idx, layer_type) in cfg.layer_types.iter().enumerate() {
            let comm = mapper.get_comm_for(layer_idx)?;
            let vb_layer = vb_l.pp(layer_idx);

            let branch = match layer_type {
                LayerType::LinearAttention => {
                    let vb_linear_attn = vb_layer.pp("linear_attn");
                    let projection_kind = gdn_input_projection_kind(&vb_linear_attn);
                    DecoderBranch::Linear(GatedDeltaNet::load(
                        vb_layer.clone(),
                        cfg as &dyn GdnConfig,
                        &*mapper,
                        layer_idx,
                        loading_isq,
                        &comm,
                        projection_kind,
                    )?)
                }
                LayerType::FullAttention => {
                    let vb_attn =
                        mapper.set_device(layer_idx, vb_layer.pp("self_attn"), loading_isq);
                    DecoderBranch::Attention(Arc::new(Mutex::new(QsaAttention::new(
                        cfg,
                        rotary_dim,
                        QsaQueryGateLayout::InterleavedPerHead,
                        vb_attn,
                    )?)))
                }
            };

            let attn_hc = GatedResidual::new(
                cfg,
                mapper.set_device(layer_idx, vb_layer.pp("hc_attn"), loading_isq),
                true,
            )?;
            let ffn_hc = GatedResidual::new(
                cfg,
                mapper.set_device(layer_idx, vb_layer.pp("hc_ffn"), loading_isq),
                true,
            )?;
            let moe = SparseMoeBlock::new(
                cfg,
                mapper.set_device(layer_idx, vb_layer.pp("mlp"), loading_isq),
                &*mapper,
                layer_idx,
                loading_isq,
                &comm,
                normal_loading_metadata.real_device.clone(),
            )?;

            if let Some(ple) = ple.as_mut() {
                if cfg.ple_layer_ids.contains(&layer_idx) {
                    let component = PleLayer::new(
                        cfg,
                        mapper.set_device(layer_idx, vb_layer.pp("ple"), loading_isq),
                    )?;
                    let state = PleState::new(
                        &ple.hasher,
                        cfg.ple_conv_kernel_size,
                        cfg.ngram_size,
                        cfg.hc_count * cfg.hidden_size,
                    )?;
                    ple.injections.push((
                        layer_idx,
                        PleInjection {
                            component,
                            state: Arc::new(Mutex::new(state)),
                        },
                    ));
                }
            }

            layers.push(DecoderLayer {
                attn_hc,
                ffn_hc,
                moe,
                branch,
            });
        }

        let hybrid_cache_config = HybridCacheConfig {
            layer_types: cfg
                .layer_types
                .iter()
                .map(|layer_type| match layer_type {
                    LayerType::FullAttention => HybridLayerType::Attention,
                    LayerType::LinearAttention => HybridLayerType::Recurrent,
                })
                .collect(),
            max_seq_len: cfg.max_position_embeddings,
            recurrent: RecurrentLayerConfig {
                conv_dim: cfg.linear_conv_dim(),
                conv_width: cfg.linear_conv_kernel_dim,
                state: RecurrentStateSpec::Gdn {
                    heads: cfg.linear_num_value_heads,
                    key_dim: cfg.linear_key_head_dim,
                    value_dim: cfg.linear_value_head_dim,
                },
                recurrent_dtype: Some(cfg.mamba_ssm_dtype.dtype()),
            },
        };
        let layer_devices = (0..cfg.num_hidden_layers)
            .map(|layer_idx| {
                mapper
                    .device_for(layer_idx, false)
                    .cloned()
                    .unwrap_or(normal_loading_metadata.real_device.clone())
            })
            .collect::<Vec<_>>();
        let mut hybrid_cache = HybridCache::new(hybrid_cache_config, dtype, &layer_devices)
            .map_err(|e| {
                candle_core::Error::Msg(format!("Failed to create Qwen4Exp hybrid cache: {e}"))
            })?;
        // QSA caches and PLE hashing/convolution state are keyed by the same recurrent state
        // slots as the cache's pools. Registering them as auxiliary state keeps the cache's
        // reset and release lifecycle from leaving stale per-sequence state behind.
        for layer in &layers {
            if let DecoderBranch::Attention(attention) = &layer.branch {
                hybrid_cache.register_auxiliary_state(Arc::new(QsaAuxiliaryState {
                    attention: Arc::clone(attention),
                }));
            }
        }
        if let Some(ple) = &ple {
            for (_, injection) in &ple.injections {
                hybrid_cache.register_auxiliary_state(Arc::new(PleAuxiliaryState {
                    state: Arc::clone(&injection.state),
                }));
            }
        }
        let pipeline_cache = Arc::new(Mutex::new(hybrid_cache));

        Ok(Self {
            embed_tokens,
            layers,
            final_hc,
            lm_head,
            dtype,
            kv_cache: EitherCache::Hybrid(pipeline_cache),
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / world_size).max(1),
                num_attn_heads: cfg.num_attention_heads / world_size,
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: KvCacheLayout::Standard,
            },
            hc_count: cfg.hc_count,
            rope_theta,
            rotary_dim,
            mrope_section: cfg.rope_parameters.mrope_section.clone(),
            mapper,
            max_seq_len: cfg.max_position_embeddings,
            ple,
        })
    }

    /// Derive stable per-row QSA and PLE sequence identities from the recurrent state slots.
    ///
    /// The pipeline stores the current batch's physical state indices on the hybrid cache and
    /// mirrors them into the recurrent metadata; either source identifies the same sequence
    /// slots the GDN layers commit to, so QSA caches and PLE hashing/convolution state share
    /// one identity per sequence.
    fn sequence_state_ids(
        hybrid_cache: &HybridCache,
        recurrent_metadata: Option<&RecurrentMetadata>,
        batch: usize,
    ) -> Result<Vec<usize>> {
        let host = hybrid_cache
            .state_indices_host()
            .map(<[u32]>::to_vec)
            .or_else(|| {
                recurrent_metadata
                    .and_then(|metadata| metadata.state_indices_host())
                    .map(<[u32]>::to_vec)
            })
            .ok_or_else(|| {
                candle_core::Error::msg(
                    "Qwen4Exp QSA and PLE layers require per-sequence recurrent state indices to key their caches; provide sequence state indices before forward",
                )
            })?;
        if host.len() != batch {
            candle_core::bail!(
                "Qwen4Exp QSA/PLE sequence state indices cover {} slots for a batch of {batch}",
                host.len()
            );
        }
        let mut seen = std::collections::HashSet::with_capacity(batch);
        let mut sequence_ids = Vec::with_capacity(batch);
        for slot in host {
            if !seen.insert(slot) {
                candle_core::bail!(
                    "Qwen4Exp QSA/PLE batch reuses sequence slot {slot}; each row needs its own state slot"
                );
            }
            sequence_ids.push(slot as usize);
        }
        Ok(sequence_ids)
    }

    pub(crate) fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let x = self.embed_tokens.embedding_forward(input_ids, self.dtype)?;
        self.forward_embeds(x, input_ids, None, ctx)
    }

    /// Run the decoder on precomputed embeddings.
    ///
    /// Multimodal inputs arrive as embeddings after vision substitution, so the original
    /// token IDs (with image placeholder tokens preserved) are passed separately for PLE
    /// hashing, matching the reference which hashes the image token id on embedding-only
    /// batches. `positions_override` supplies explicit per-row positions for multimodal
    /// MRoPE; text-only callers pass `None` and positions come from the forward context.
    pub(crate) fn forward_embeds(
        &self,
        x: Tensor,
        token_ids: &Tensor,
        positions_override: Option<MropePositionOverride>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let input_ids = token_ids;
        let _ = &input_ids;

        let recurrent_metadata = ctx.recurrent_metadata().cloned();
        let has_linear_attention = self
            .layers
            .iter()
            .any(|layer| matches!(layer.branch, DecoderBranch::Linear(_)));
        if has_linear_attention && recurrent_metadata.is_none() {
            candle_core::bail!("Qwen4Exp requires recurrent metadata for GDN layers");
        }
        if let Some(metadata) = &recurrent_metadata {
            if metadata.batch_kind() == RecurrentBatchKind::SpeculativeDecode {
                candle_core::bail!(
                    "Qwen4Exp speculative decoding requires QSA and PLE rollback, which is not wired yet"
                );
            }
        }
        let packed_layout = if ctx.flash_params().packed {
            Some(Self::packed_layout(
                input_ids,
                ctx,
                recurrent_metadata.as_ref(),
                has_linear_attention,
            )?)
        } else {
            None
        };

        // The wide residual starts as hc identical copies of the embedding, as in the reference.
        let mut residual = x.unsqueeze(2)?.repeat((1, 1, self.hc_count, 1))?;

        let mut hybrid_cache = self.kv_cache.hybrid();

        // QSA layers key their per-sequence caches and PLE layers key their hashing and
        // convolution state by the recurrent state slots, so the batch's sequence identity
        // must be resolved before any layer runs. QSA enforces causal visibility internally
        // through selection and gathering, and the GDN branch needs no causal mask.
        let has_full_attention = self
            .layers
            .iter()
            .any(|layer| matches!(layer.branch, DecoderBranch::Attention(_)));
        // Packed prefill has one physical row but one logical sequence per query length, so
        // sequence identity is resolved for the logical batch.
        let logical_batch = match &packed_layout {
            Some(layout) => layout.batch_size(),
            None => input_ids.dim(0)?,
        };
        let sequence_ids = if has_full_attention || self.ple.is_some() {
            Some(Self::sequence_state_ids(
                &hybrid_cache,
                recurrent_metadata.as_ref(),
                logical_batch,
            )?)
        } else {
            None
        };

        // Scalar text positions use a shared absolute-position cache. Multimodal callers pass
        // selected section-aware tables so QSA main and indexer rotation retain THW semantics.
        let rope_inputs = if has_full_attention {
            match positions_override {
                Some(MropePositionOverride::Scalar(positions)) => {
                    if packed_layout.is_some() {
                        candle_core::bail!(
                            "Qwen4Exp multimodal packed prefill is not supported yet"
                        );
                    }
                    let max_position = positions.iter().flatten().copied().max().unwrap_or(0);
                    let cache_positions = (0..=max_position).collect::<Vec<u32>>();
                    let (cos, sin) = interleaved_mrope_tables(
                        self.rope_theta,
                        self.rotary_dim,
                        &cache_positions,
                        residual.device(),
                        residual.dtype(),
                    )?;
                    Some(RopeInputs::Scalar {
                        positions,
                        cos,
                        sin,
                    })
                }
                Some(MropePositionOverride::Sectioned(sectioned_positions)) => {
                    if packed_layout.is_some() {
                        candle_core::bail!(
                            "Qwen4Exp multimodal packed prefill is not supported yet"
                        );
                    }
                    let positions = sectioned_positions
                        .first()
                        .ok_or_else(|| {
                            candle_core::Error::msg("Qwen4Exp MRoPE has no temporal positions")
                        })?
                        .clone();
                    let (cos, sin) = sectioned_interleaved_mrope_tables(
                        self.rope_theta,
                        self.rotary_dim,
                        &self.mrope_section,
                        &sectioned_positions,
                        residual.device(),
                        residual.dtype(),
                    )?;
                    Some(RopeInputs::Sectioned {
                        positions,
                        cos,
                        sin,
                    })
                }
                None => {
                    let positions = match &packed_layout {
                        Some(layout) => {
                            let packed_positions = ctx
                                .text_positions(residual.device(), layout.token_count())?
                                .ok_or_else(|| {
                                    candle_core::Error::msg(
                                        "Qwen4Exp full attention is missing RoPE positions",
                                    )
                                })?
                                .to_vec1::<u32>()?;
                            if packed_positions.len() != layout.token_count() {
                                candle_core::bail!("Qwen4Exp packed prefill has {} packed RoPE positions but {} logical tokens", packed_positions.len(), layout.token_count());
                            }
                            layout
                                .token_ranges()
                                .map(|range| packed_positions[range].to_vec())
                                .collect()
                        }
                        None => {
                            let (batch, seq_len) = input_ids.dims2()?;
                            ctx.text_positions(residual.device(), seq_len)?
                                .ok_or_else(|| {
                                    candle_core::Error::msg(
                                        "Qwen4Exp full attention is missing RoPE positions",
                                    )
                                })?
                                .reshape((batch, seq_len))?
                                .to_vec2::<u32>()?
                        }
                    };
                    let max_position = positions.iter().flatten().copied().max().unwrap_or(0);
                    let cache_positions = (0..=max_position).collect::<Vec<u32>>();
                    let (cos, sin) = interleaved_mrope_tables(
                        self.rope_theta,
                        self.rotary_dim,
                        &cache_positions,
                        residual.device(),
                        residual.dtype(),
                    )?;
                    Some(RopeInputs::Scalar {
                        positions,
                        cos,
                        sin,
                    })
                }
            }
        } else {
            None
        };

        // PLE hashing consumes the original token IDs. The text path forwards the raw input
        // IDs; multimodal inputs must preserve placeholder token IDs through embedding
        // substitution before PLE injection becomes usable there.
        let ple_token_rows = if self.ple.is_some() {
            Some(match &packed_layout {
                Some(layout) => {
                    let rows = input_ids.to_vec2::<u32>()?;
                    let packed_tokens = rows.into_iter().next().ok_or_else(|| {
                        candle_core::Error::msg("Qwen4Exp packed prefill has no packed token row")
                    })?;
                    if packed_tokens.len() != layout.token_count() {
                        candle_core::bail!(
                            "Qwen4Exp packed prefill has {} packed tokens but {} logical tokens",
                            packed_tokens.len(),
                            layout.token_count()
                        );
                    }
                    layout
                        .token_ranges()
                        .map(|range| packed_tokens[range].to_vec())
                        .collect::<Vec<_>>()
                }
                None => input_ids.to_vec2::<u32>()?,
            })
        } else {
            None
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            residual = self.mapper.map(residual, layer_idx)?;

            // The reference injects PLE into the wide residual before the layer's attention
            // hyper-connection mixer.
            if let Some(modules) = self.ple.as_ref() {
                if let Some(injection) = modules.injection_for(layer_idx) {
                    let token_rows = ple_token_rows
                        .as_deref()
                        .expect("checked above: PLE requires token rows");
                    let sequence_ids = sequence_ids
                        .as_deref()
                        .expect("checked above: PLE requires sequence ids");
                    residual = match &packed_layout {
                        Some(layout) => self.ple_inject_packed(
                            injection,
                            modules,
                            &residual,
                            token_rows,
                            sequence_ids,
                            layout,
                        )?,
                        None => self.ple_inject(
                            injection,
                            modules,
                            &residual,
                            token_rows,
                            sequence_ids,
                        )?,
                    };
                }
            }

            let (branch_input, attn_injection) = layer.attn_hc.mix(&residual)?;
            let attn_injection = attn_injection
                .expect("Qwen4Exp attention hyper-connection has an injection projection");
            let branch_output = match &layer.branch {
                DecoderBranch::Attention(qsa) => {
                    let mut qsa = qsa.lock().map_err(|_| {
                        candle_core::Error::msg("Qwen4Exp QSA layer mutex is poisoned")
                    })?;
                    let rope_inputs = rope_inputs
                        .as_ref()
                        .expect("checked above: full-attention layers require position tables");
                    let (rope_positions, cos_cache, sin_cache, sectioned) = match rope_inputs {
                        RopeInputs::Scalar {
                            positions,
                            cos,
                            sin,
                        } => (positions, cos, sin, false),
                        RopeInputs::Sectioned {
                            positions,
                            cos,
                            sin,
                        } => (positions, cos, sin, true),
                    };
                    // The frequency tables live on the residual device; share them across
                    // layers on the same device location and move them only when needed.
                    let (cos_cache, sin_cache) =
                        if cos_cache.device().location() == branch_input.device().location() {
                            (cos_cache.clone(), sin_cache.clone())
                        } else {
                            (
                                cos_cache.to_device(branch_input.device())?,
                                sin_cache.to_device(branch_input.device())?,
                            )
                        };
                    let sequence_ids = sequence_ids
                        .as_deref()
                        .expect("checked above: full-attention layers require sequence ids");
                    match &packed_layout {
                        Some(layout) => {
                            // Packed prefill runs one transactional chunk per logical
                            // sequence, each a `[1, tokens, hidden]` slice of the packed
                            // batch row; the rotaries select their own rows from the
                            // position-indexed frequency cache.
                            debug_assert_eq!(rope_positions.len(), layout.batch_size());
                            let chunk_hidden = layout
                                .token_ranges()
                                .map(|range| branch_input.narrow(1, range.start, range.len()))
                                .collect::<Result<Vec<_>>>()?;
                            let chunks = sequence_ids
                                .iter()
                                .copied()
                                .zip(chunk_hidden.iter())
                                .zip(rope_positions.iter())
                                .map(|((sequence_id, hidden), positions)| {
                                    (sequence_id, hidden, positions.as_slice())
                                })
                                .collect::<Vec<_>>();
                            let outputs =
                                qsa.forward_packed_chunks(&chunks, &cos_cache, &sin_cache)?;
                            Tensor::cat(&outputs, 1)?
                        }
                        None => {
                            let (batch, tokens, _) = branch_input.dims3()?;
                            debug_assert_eq!(rope_positions.len(), batch);
                            // Each non-packed sequence row carries its own position offset, so
                            // run the per-sequence QSA orchestration per row; the rotaries
                            // select their own rows from the position-indexed frequency cache.
                            let mut row_outputs = Vec::with_capacity(batch);
                            for (row, positions) in rope_positions.iter().enumerate() {
                                debug_assert_eq!(positions.len(), tokens);
                                let row_hidden = branch_input.narrow(0, row, 1)?;
                                row_outputs.push(if sectioned {
                                    qsa.forward_chunk_with_position_tables(
                                        sequence_ids[row],
                                        &row_hidden,
                                        positions,
                                        &cos_cache.narrow(0, row, 1)?.squeeze(0)?,
                                        &sin_cache.narrow(0, row, 1)?.squeeze(0)?,
                                    )?
                                } else {
                                    qsa.forward_chunk(
                                        sequence_ids[row],
                                        &row_hidden,
                                        positions,
                                        &cos_cache,
                                        &sin_cache,
                                    )?
                                });
                            }
                            if row_outputs.len() == 1 {
                                row_outputs.swap_remove(0)
                            } else {
                                Tensor::cat(&row_outputs, 0)?
                            }
                        }
                    }
                }
                DecoderBranch::Linear(gdn) => {
                    let metadata = recurrent_metadata
                        .as_ref()
                        .expect("checked above: GDN layers require recurrent metadata");
                    let indices = hybrid_cache
                        .state_indices_for_layer(layer_idx)?
                        .ok_or_else(|| {
                            candle_core::Error::msg(format!(
                                "Qwen4Exp hybrid cache layer {layer_idx} is missing recurrent state indices"
                            ))
                        })?;
                    let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer_idx)
                    else {
                        candle_core::bail!(
                            "Qwen4Exp hybrid cache layer {layer_idx} is not recurrent for a GDN layer"
                        );
                    };
                    // Packed prefill needs gathered logical state rows: the grouped packed GDN
                    // path rejects slot-table caches, and commit must scatter the updated rows.
                    let mut gdn_cache = if packed_layout.is_some() {
                        GdnLayerCache::gathered(
                            pool.gather_conv_state(&indices)?,
                            pool.gather_recurrent_state(&indices)?,
                            pool.state_layout(),
                        )
                    } else {
                        GdnLayerCache::checkout(pool, &indices)?
                    };
                    let output = match &packed_layout {
                        Some(layout) => {
                            forward_packed_gdn_segments(gdn, &branch_input, &mut gdn_cache, layout)?
                        }
                        None => {
                            gdn.forward(&branch_input, &mut gdn_cache, metadata.batch_kind())?
                        }
                    };
                    gdn_cache.commit(pool, &indices, metadata.state_indices_host())?;
                    output
                }
            };
            residual = layer
                .attn_hc
                .combine(&residual, &branch_output, &attn_injection)?;

            let (ffn_input, ffn_injection) = layer.ffn_hc.mix(&residual)?;
            let ffn_injection =
                ffn_injection.expect("Qwen4Exp MoE hyper-connection has an injection projection");
            let moe_output = layer.moe.forward(&ffn_input)?;
            residual = layer
                .ffn_hc
                .combine(&residual, &moe_output, &ffn_injection)?;
        }

        let residual = residual.to_device(&self.device)?;
        let (mixed, _) = self.final_hc.mix(&residual)?;
        let mixed = ctx.logits(&mixed)?;
        ctx.lm_head(&*self.lm_head, &mixed)
    }

    /// The token embedding table, exposed for multimodal wrappers that substitute
    /// vision outputs into the embedding stream before the decoder runs.
    pub(crate) fn embed_tokens(&self) -> &Arc<dyn QuantMethod> {
        &self.embed_tokens
    }

    /// The activation dtype used for embedding lookups.
    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    /// Apply the configured PLE layer's residual update per batch row.
    ///
    /// Each row runs one transactional PLE chunk: predecessor hashing, bounded embedding
    /// row gathering, projection/gating, the stateful dilated convolution, and the
    /// `hidden + gated + convolved` residual update, keyed by the row's sequence slot.
    fn ple_inject(
        &self,
        injection: &PleInjection,
        modules: &PleModules,
        residual: &Tensor,
        token_rows: &[Vec<u32>],
        sequence_ids: &[usize],
    ) -> Result<Tensor> {
        let (batch, tokens, streams, hidden) = residual.dims4()?;
        if streams != self.hc_count || hidden != self.cfg.hidden_size {
            candle_core::bail!(
                "Qwen4Exp PLE injection expects residual [{batch}, {tokens}, {}, {}], got {:?}",
                self.hc_count,
                self.cfg.hidden_size,
                residual.dims()
            );
        }
        if token_rows.len() != batch || sequence_ids.len() != batch {
            candle_core::bail!(
                "Qwen4Exp PLE injection needs one token row and sequence id per batch row: batch {batch}, token rows {}, sequence ids {}",
                token_rows.len(),
                sequence_ids.len()
            );
        }

        let mut row_outputs = Vec::with_capacity(batch);
        for (row, row_tokens) in token_rows.iter().enumerate() {
            if row_tokens.len() != tokens {
                candle_core::bail!(
                    "Qwen4Exp PLE row {row} has {} tokens but the residual covers {tokens}",
                    row_tokens.len()
                );
            }
            let row_residual = residual.narrow(0, row, 1)?;
            let sequence_id = sequence_ids[row];
            let updated = {
                let mut state = injection
                    .state
                    .lock()
                    .map_err(|_| candle_core::Error::msg("Qwen4Exp PLE layer mutex is poisoned"))?;
                state.forward_packed_tensor_chunks(
                    &modules.hasher,
                    &modules.embedding,
                    &injection.component,
                    &[(sequence_id, row_tokens.as_slice())],
                    &[(sequence_id, &row_residual)],
                )?
            };
            row_outputs.push(updated);
        }
        if row_outputs.len() == 1 {
            Ok(row_outputs.swap_remove(0))
        } else {
            Tensor::cat(&row_outputs, 0)
        }
    }

    /// Build the packed prefill plan from the pipeline's logical query lengths.
    ///
    /// A packed batch is one physical row covering `sum(query_lens)` logical tokens. Every
    /// logical sequence owns one recurrent state slot, and the batch must be a first prefill
    /// prompt chunk so no sequence carries partially processed history.
    fn packed_layout(
        input_ids: &Tensor,
        ctx: &ModelForwardContext<'_>,
        recurrent_metadata: Option<&RecurrentMetadata>,
        has_linear_attention: bool,
    ) -> Result<PackedGdnLayout> {
        if !ctx.is_first_prompt_chunk() {
            candle_core::bail!("Qwen4Exp packed prefill requires the first prompt chunk");
        }
        let query_lens = ctx
            .paged_input_metadata()
            .and_then(|metadata| metadata.query_lens.clone())
            .ok_or_else(|| {
                candle_core::Error::msg(
                    "Qwen4Exp packed prefill requires logical query lengths from the paged input metadata",
                )
            })?;
        if has_linear_attention {
            let metadata = recurrent_metadata.ok_or_else(|| {
                candle_core::Error::msg(
                    "Qwen4Exp packed prefill requires recurrent metadata for GDN layers",
                )
            })?;
            if metadata.batch_kind() != RecurrentBatchKind::Prefill {
                candle_core::bail!(
                    "Qwen4Exp packed prefill cannot run a non-prefill recurrent batch"
                );
            }
            let index_count = metadata.state_indices().dims1()?;
            if index_count != query_lens.len() {
                candle_core::bail!(
                    "Qwen4Exp packed prefill has {index_count} recurrent state indices but {} logical sequences",
                    query_lens.len()
                );
            }
        }
        let (physical_batch, physical_tokens) = input_ids.dims2()?;
        if physical_batch != 1 {
            candle_core::bail!(
                "Qwen4Exp packed prefill expects one packed batch row, got {physical_batch}"
            );
        }
        if query_lens.is_empty() || query_lens.contains(&0) {
            candle_core::bail!(
                "Qwen4Exp packed prefill requires a positive query length for every logical sequence"
            );
        }
        let logical_tokens = query_lens.iter().try_fold(0usize, |total, &len| {
            total.checked_add(len).ok_or_else(|| {
                candle_core::Error::msg("Qwen4Exp packed prefill logical token count overflows")
            })
        })?;
        if logical_tokens != physical_tokens {
            candle_core::bail!(
                "Qwen4Exp packed prefill has {logical_tokens} logical tokens for {physical_tokens} packed input tokens"
            );
        }
        PackedGdnLayout::new(query_lens, ctx.flash_params().cumulative_seqlens_q.clone())
    }

    /// Apply the configured PLE layer's residual update to every packed logical sequence.
    ///
    /// All sequences run as one transactional packed PLE chunk — predecessor hashing, bounded
    /// embedding row gathering, projection/gating, the stateful dilated convolution, and the
    /// `hidden + gated + convolved` residual update — and the packed PLE state emits outputs
    /// in packed token order, so the updated residual keeps the packed layout.
    fn ple_inject_packed(
        &self,
        injection: &PleInjection,
        modules: &PleModules,
        residual: &Tensor,
        token_rows: &[Vec<u32>],
        sequence_ids: &[usize],
        layout: &PackedGdnLayout,
    ) -> Result<Tensor> {
        let (batch, tokens, streams, hidden) = residual.dims4()?;
        if batch != 1
            || tokens != layout.token_count()
            || streams != self.hc_count
            || hidden != self.cfg.hidden_size
        {
            candle_core::bail!(
                "Qwen4Exp packed PLE injection expects residual [1, {}, {}, {}], got {:?}",
                layout.token_count(),
                self.hc_count,
                self.cfg.hidden_size,
                residual.dims()
            );
        }
        if token_rows.len() != layout.batch_size() || sequence_ids.len() != layout.batch_size() {
            candle_core::bail!(
                "Qwen4Exp packed PLE injection needs one token row and sequence id per logical sequence: {} sequences, token rows {}, sequence ids {}",
                layout.batch_size(),
                token_rows.len(),
                sequence_ids.len()
            );
        }

        let chunk_hidden = layout
            .token_ranges()
            .map(|range| residual.narrow(1, range.start, range.len()))
            .collect::<Result<Vec<_>>>()?;
        let hidden_chunks = sequence_ids
            .iter()
            .copied()
            .zip(chunk_hidden.iter())
            .collect::<Vec<_>>();
        let token_chunks = sequence_ids
            .iter()
            .copied()
            .zip(token_rows.iter())
            .map(|(sequence_id, tokens)| (sequence_id, tokens.as_slice()))
            .collect::<Vec<_>>();

        let mut state = injection
            .state
            .lock()
            .map_err(|_| candle_core::Error::msg("Qwen4Exp PLE layer mutex is poisoned"))?;
        state.forward_packed_tensor_chunks(
            &modules.hasher,
            &modules.embedding,
            &injection.component,
            &token_chunks,
            &hidden_chunks,
        )
    }
}

/// Run one packed GDN forward across the layout's logical sequences.
///
/// Tries the shared grouped packed GDN path first, then falls back to per-sequence segments.
/// Each segment updates one logical sequence's gathered state rows, and the segment outputs
/// and updated states are reassembled in packed order.
fn forward_packed_gdn_segments(
    gdn: &GatedDeltaNet,
    input: &Tensor,
    cache: &mut GdnLayerCache,
    layout: &PackedGdnLayout,
) -> Result<Tensor> {
    let (physical_batch, physical_tokens, _) = input.dims3()?;
    let (conv_state_batch, _, _) = cache.conv_state.dims3()?;
    let (recurrent_state_batch, _, _, _) = cache.recurrent_state.dims4()?;
    if physical_batch != 1 || physical_tokens != layout.token_count() {
        candle_core::bail!("Qwen4Exp packed GDN token dimensions are incompatible");
    }
    if conv_state_batch != layout.batch_size() {
        candle_core::bail!(
            "Qwen4Exp packed GDN has {conv_state_batch} convolution state rows but {} logical sequences",
            layout.batch_size()
        );
    }
    if recurrent_state_batch != layout.batch_size() {
        candle_core::bail!(
            "Qwen4Exp packed GDN has {recurrent_state_batch} recurrent state rows but {} logical sequences",
            layout.batch_size()
        );
    }
    if input.dtype() != cache.conv_state.dtype() {
        candle_core::bail!(
            "Qwen4Exp packed GDN dtype mismatch: tokens are {:?}, convolution state is {:?}",
            input.dtype(),
            cache.conv_state.dtype()
        );
    }
    if !input.device().same_device(cache.conv_state.device())
        || !input.device().same_device(cache.recurrent_state.device())
    {
        candle_core::bail!(
            "Qwen4Exp packed GDN tokens and recurrent states are on different devices"
        );
    }

    if let Some(output) = try_forward_grouped_packed_gdn(gdn, input, cache, layout)? {
        return Ok(output);
    }

    let mut outputs = Vec::with_capacity(layout.batch_size());
    let mut next_conv_states = Vec::with_capacity(layout.batch_size());
    let mut next_recurrent_states = Vec::with_capacity(layout.batch_size());
    for (state_index, range) in layout.token_ranges().enumerate() {
        let segment_input = input.narrow(1, range.start, range.len())?;
        let mut segment_cache = GdnLayerCache {
            conv_state: cache.conv_state.narrow(0, state_index, 1)?,
            recurrent_state: cache.recurrent_state.narrow(0, state_index, 1)?,
            state_layout: cache.state_layout,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
        };
        outputs.push(mistralrs_quant::with_lora_execution_row_range(
            range.clone(),
            || {
                gdn.forward(
                    &segment_input,
                    &mut segment_cache,
                    RecurrentBatchKind::Prefill,
                )
            },
        )?);
        next_conv_states.push(segment_cache.conv_state);
        next_recurrent_states.push(segment_cache.recurrent_state);
    }
    cache.conv_state = Tensor::cat(&next_conv_states, 0)?;
    cache.recurrent_state = Tensor::cat(&next_recurrent_states, 0)?;
    Tensor::cat(&outputs, 1)
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m
            .pp("hc_head")
            .pp("norm")
            .add_tensor("weight", self.final_hc.norm_weight().clone());

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l
                .pp("hc_attn")
                .pp("norm")
                .add_tensor("weight", layer.attn_hc.norm_weight().clone());
            uvb_l
                .pp("hc_ffn")
                .pp("norm")
                .add_tensor("weight", layer.ffn_hc.norm_weight().clone());

            match &layer.branch {
                DecoderBranch::Linear(gdn) => {
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
                DecoderBranch::Attention(attention) => {
                    let attention = attention
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let (q_norm, k_norm) = attention.residual_norms();
                    uvb_l
                        .pp("self_attn")
                        .pp("attn")
                        .pp("q_norm")
                        .add_tensor("weight", q_norm.clone());
                    uvb_l
                        .pp("self_attn")
                        .pp("attn")
                        .pp("k_norm")
                        .add_tensor("weight", k_norm.clone());
                }
            }

            uvb_l
                .pp("mlp")
                .pp("gate")
                .add_tensor("weight", layer.moe.gate.weight().clone());
            uvb_l
                .pp("mlp")
                .pp("shared_expert_gate")
                .add_tensor("weight", layer.moe.shared_expert_gate.weight().clone());
        }

        if let Some(ple) = &self.ple {
            for (layer_idx, injection) in &ple.injections {
                let [norm_key, norm_query, norm_conv] = injection.component.residual_norms();
                let uvb_ple = uvb_m.pp("layers").pp(*layer_idx).pp("ple");
                uvb_ple.add_tensor("norm_key.weight", norm_key.clone());
                uvb_ple.add_tensor("norm_query.weight", norm_query.clone());
                uvb_ple.add_tensor("norm_conv.weight", norm_conv.clone());
            }
        }

        uvb.to_safetensors()
    }
}

impl NormalModel for Model {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        // The inherent `Model::forward` takes precedence over this trait method in
        // method resolution, so this delegates to the decoder's own forward.
        self.forward(input_ids, ctx)
    }

    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        candle_core::bail!("Qwen4Exp does not support X-LoRA forward")
    }

    fn cache(&self) -> &EitherCache {
        &self.kv_cache
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn is_xlora(&self) -> bool {
        false
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

impl AnyMoeBaseModelMixin for Model {}

impl crate::speculative::SpeculativeTargetMixin for Model {}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use candle_core::DType;
    use indicatif::MultiProgress;
    use mistralrs_quant::{ShardedSafeTensors, ShardedVarBuilder};

    use super::*;
    use crate::device_map::DeviceMapSetting;
    use crate::pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        RecurrentMetadata,
    };

    const TINY_TEXT_CONFIG: &str = r#"{
        "head_dim": 6,
        "vocab_size": 16,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [1, 1, 1],
            "partial_rotary_factor": 1.0,
            "mrope_interleaved": true
        },
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "norm_topk_prob": true,
        "full_attention_interval": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 3,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "mamba_ssm_dtype": "float32",
        "hc_count": 2,
        "hc_lowrank": 4,
        "indexer_n_heads": 1,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 6,
        "indexer_budget": 8,
        "indexer_compress_ratio": 4,
        "output_gate_type": "sigmoid",
        "ple_layer_ids": [],
        "ple_embed_dim": 0,
        "ple_conv_kernel_size": 0,
        "ngram_size": 2,
        "heads_per_ngram": 0,
        "ple_layer_multipliers": [],
        "ple_head_offsets": [],
        "ple_head_vocab_sizes": [],
        "eos_token_id": 15,
        "image_token_id": null,
        "tie_word_embeddings": false
    }"#;

    pub(crate) fn tiny_config() -> Config {
        serde_json::from_str(TINY_TEXT_CONFIG).unwrap()
    }

    /// A PLE-capable variant injecting at layer 0, the fixture's GDN layer.
    pub(crate) fn ple_config() -> Config {
        let mut config = tiny_config();
        config.ple_layer_ids = vec![0];
        config.ple_embed_dim = 2;
        config.ple_conv_kernel_size = 4;
        config.ngram_size = 2;
        config.heads_per_ngram = 1;
        config.ple_layer_multipliers = vec![1, 1];
        config.ple_head_offsets = vec![0];
        config.ple_head_vocab_sizes = vec![2];
        config.eos_token_id = 15;
        config
    }

    fn weight(shape: (usize, usize), seed: usize) -> Result<Tensor> {
        let (rows, cols) = shape;
        let mut data = Vec::with_capacity(rows * cols);
        for index in 0..rows * cols {
            // Small deterministic values keep the tiny F32 reference well-conditioned.
            data.push((((index * 2_654_435_761 + seed) % 2_000) as f32 / 1_000.0) - 1.0);
        }
        Tensor::from_vec(data, (rows, cols), &Device::Cpu)
    }

    fn weight1(length: usize, seed: usize) -> Result<Tensor> {
        weight((1, length), seed)?.reshape((length,))
    }

    fn weight3(shape: (usize, usize, usize), seed: usize) -> Result<Tensor> {
        let (rows, cols, depth) = shape;
        let mut data = Vec::with_capacity(rows * cols * depth);
        for index in 0..rows * cols * depth {
            data.push((((index * 2_654_435_761 + seed) % 2_000) as f32 / 1_000.0) - 1.0);
        }
        Tensor::from_vec(data, (rows, cols, depth), &Device::Cpu)
    }

    /// The raw tensor map behind [`fixture_builder`], so sibling test modules can merge in
    /// additional tensors (for example the multimodal vision tower) before wrapping.
    pub(crate) fn fixture_tensors(cfg: &Config) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        build_fixture_tensors(cfg, &mut tensors)?;
        Ok(tensors)
    }

    /// Build every tensor the bring-up decoder reads, mirroring the planned GGUF binding names.
    pub(crate) fn fixture_builder(cfg: &Config) -> Result<ShardedVarBuilder> {
        Ok(ShardedSafeTensors::wrap(
            fixture_tensors(cfg)?,
            DType::F32,
            Device::Cpu,
        ))
    }

    fn build_fixture_tensors(cfg: &Config, tensors: &mut HashMap<String, Tensor>) -> Result<()> {
        let hidden = cfg.hidden_size;
        let wide = cfg.hc_count * hidden;
        let lowrank = cfg.hc_lowrank;
        let head_dim = cfg.head_dim;
        let query_heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let inter = cfg.moe_intermediate_size;
        let shared_inter = cfg.shared_expert_intermediate_size;
        let experts = cfg.num_experts;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let conv_dim = value_dim + key_dim * 2;
        let qkvz_out = key_dim * 2 + value_dim * 2;
        let ba_out = cfg.linear_num_value_heads * 2;

        let mut seed = 0usize;

        macro_rules! put_1d {
            ($name:expr, $length:expr) => {{
                seed += 1;
                tensors.insert($name.to_string(), weight1($length, seed)?);
            }};
        }
        macro_rules! put_2d {
            ($name:expr, $rows:expr, $cols:expr) => {{
                seed += 1;
                tensors.insert($name.to_string(), weight(($rows, $cols), seed)?);
            }};
        }
        macro_rules! put_3d {
            ($name:expr, $rows:expr, $cols:expr, $depth:expr) => {{
                seed += 1;
                tensors.insert($name.to_string(), weight3(($rows, $cols, $depth), seed)?);
            }};
        }

        put_2d!("model.embed_tokens.weight", cfg.vocab_size, hidden);
        put_2d!("lm_head.weight", cfg.vocab_size, hidden);
        put_1d!("model.hc_head.norm.weight", wide);
        put_2d!("model.hc_head.down.weight", lowrank, wide);
        put_2d!("model.hc_head.up.weight", wide, lowrank);

        for layer_idx in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}");
            for hc_name in ["hc_attn", "hc_ffn"] {
                put_1d!(format!("{prefix}.{hc_name}.norm.weight"), wide);
                put_2d!(format!("{prefix}.{hc_name}.down.weight"), lowrank, wide);
                put_2d!(format!("{prefix}.{hc_name}.up.weight"), wide, lowrank);
                put_2d!(
                    format!("{prefix}.{hc_name}.inject.weight"),
                    cfg.hc_count,
                    wide
                );
            }
            match cfg.layer_types[layer_idx] {
                LayerType::LinearAttention => {
                    let gdn = format!("{prefix}.linear_attn");
                    put_2d!(format!("{gdn}.in_proj_qkvz.weight"), qkvz_out, hidden);
                    put_2d!(format!("{gdn}.in_proj_ba.weight"), ba_out, hidden);
                    put_3d!(
                        format!("{gdn}.conv1d.weight"),
                        conv_dim,
                        1,
                        cfg.linear_conv_kernel_dim
                    );
                    put_1d!(format!("{gdn}.dt_bias"), cfg.linear_num_value_heads);
                    put_1d!(format!("{gdn}.A_log"), cfg.linear_num_value_heads);
                    put_1d!(format!("{gdn}.norm.weight"), cfg.linear_value_head_dim);
                    put_2d!(format!("{gdn}.out_proj.weight"), hidden, value_dim);
                }
                LayerType::FullAttention => {
                    let attn = format!("{prefix}.self_attn.attn");
                    put_2d!(
                        format!("{attn}.q_proj.weight"),
                        query_heads * head_dim * 2,
                        hidden
                    );
                    put_2d!(format!("{attn}.k_proj.weight"), kv_heads * head_dim, hidden);
                    put_2d!(format!("{attn}.v_proj.weight"), kv_heads * head_dim, hidden);
                    put_2d!(
                        format!("{attn}.o_proj.weight"),
                        hidden,
                        query_heads * head_dim
                    );
                    put_1d!(format!("{attn}.q_norm.weight"), head_dim);
                    put_1d!(format!("{attn}.k_norm.weight"), head_dim);
                    let indexer_head_dim = cfg.indexer_head_dim;
                    let indexer = format!("{prefix}.self_attn.indexer");
                    put_2d!(
                        format!("{indexer}.q_proj.weight"),
                        cfg.indexer_n_heads * indexer_head_dim,
                        hidden
                    );
                    put_2d!(format!("{indexer}.k_proj.weight"), indexer_head_dim, hidden);
                    put_1d!(format!("{indexer}.q_norm.weight"), indexer_head_dim);
                    put_1d!(format!("{indexer}.k_norm.weight"), indexer_head_dim);
                }
            }
            let mlp = format!("{prefix}.mlp");
            put_2d!(format!("{mlp}.gate.weight"), cfg.num_experts, hidden);
            put_3d!(
                format!("{mlp}.experts.gate_up_proj"),
                experts,
                inter * 2,
                hidden
            );
            put_3d!(format!("{mlp}.experts.down_proj"), experts, hidden, inter);
            put_2d!(
                format!("{mlp}.shared_expert.gate_proj.weight"),
                shared_inter,
                hidden
            );
            put_2d!(
                format!("{mlp}.shared_expert.up_proj.weight"),
                shared_inter,
                hidden
            );
            put_2d!(
                format!("{mlp}.shared_expert.down_proj.weight"),
                hidden,
                shared_inter
            );
            put_2d!(format!("{mlp}.shared_expert_gate.weight"), 1, hidden);
        }

        // PLE tensors come last so every non-PLE tensor keeps the same seed sequence
        // whether the fixture configures PLE, keeping weight comparisons valid.
        if !cfg.ple_layer_ids.is_empty() {
            let head_count = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
            let head_dim = cfg.ple_embed_dim / head_count;
            let table_rows = cfg
                .ple_head_offsets
                .iter()
                .zip(&cfg.ple_head_vocab_sizes)
                .map(|(offset, vocab)| usize::try_from(offset + vocab).unwrap())
                .max()
                .unwrap_or(0);
            put_2d!("model.per_layer_token_embd.weight", table_rows, head_dim);
            for &ple_layer in &cfg.ple_layer_ids {
                let ple = format!("model.layers.{ple_layer}.ple");
                put_2d!(format!("{ple}.key.weight"), wide, cfg.ple_embed_dim);
                put_2d!(
                    format!("{ple}.value.weight"),
                    cfg.hidden_size,
                    cfg.ple_embed_dim
                );
                put_1d!(format!("{ple}.norm_key.weight"), wide);
                put_1d!(format!("{ple}.norm_query.weight"), wide);
                put_1d!(format!("{ple}.norm_conv.weight"), wide);
                // GGUF presentation: candle row-major [channels, kernel].
                put_2d!(
                    format!("{ple}.conv1d.weight"),
                    wide,
                    cfg.ple_conv_kernel_size
                );
            }
        }

        Ok(())
    }

    pub(crate) fn loading_metadata(layer_count: usize) -> Result<NormalLoadingMetadata> {
        let mapper = DeviceMapSetting::dummy().into_mapper(layer_count, &Device::Cpu, None, &[])?;
        Ok(NormalLoadingMetadata {
            mapper,
            loading_isq: false,
            real_device: Device::Cpu,
            multi_progress: Arc::new(MultiProgress::new()),
            matformer_slicing_config: None,
            rope_pairing: None,
        })
    }

    fn build_model(cfg: &Config, vb: ShardedVarBuilder) -> Result<Model> {
        build_model_mechanism(cfg, vb, AttentionImplementation::Eager)
    }

    fn build_model_mechanism(
        cfg: &Config,
        vb: ShardedVarBuilder,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Model> {
        Model::new(
            cfg,
            vb,
            true,
            loading_metadata(cfg.num_hidden_layers)?,
            attention_mechanism,
        )
    }

    fn run_forward(
        model: &Model,
        tokens: &[u32],
        batch: usize,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        position_ids: &[usize],
        kind: RecurrentBatchKind,
    ) -> Result<Tensor> {
        let flash = FlashParams::empty(true);
        let host: Vec<u32> = (0..batch as u32).collect();
        let indices = Tensor::from_vec(host.clone(), (batch,), &Device::Cpu)?;
        // The pipeline normally populates the cache's batch state indices in clone_in_cache;
        // tests set them directly before forward.
        model
            .kv_cache
            .hybrid()
            .set_physical_state_indices_with_host(Some(indices.clone()), Some(host));
        let mut ctx =
            ModelForwardContext::new(seqlen_offsets, context_lens, position_ids, None, &flash)
                .with_recurrent_metadata(Some(RecurrentMetadata::new(kind, indices, None)));
        let seq_len = tokens.len() / batch;
        let input = Tensor::from_vec(tokens.to_vec(), (batch, seq_len), &Device::Cpu)?;
        model.forward(&input, &mut ctx)
    }

    /// Run one forward with the given recurrent state slots installed as the batch's
    /// physical state indices, mirroring how the engine keys QSA and PLE state by slot.
    #[allow(clippy::too_many_arguments)]
    fn run_forward_with_slots(
        model: &Model,
        tokens: &[u32],
        batch: usize,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        position_ids: &[usize],
        kind: RecurrentBatchKind,
        slots: &[u32],
    ) -> Result<Tensor> {
        let flash = FlashParams::empty(true);
        let indices = Tensor::from_vec(slots.to_vec(), (batch,), &Device::Cpu)?;
        model
            .kv_cache
            .hybrid()
            .set_physical_state_indices_with_host(Some(indices.clone()), Some(slots.to_vec()));
        let mut ctx =
            ModelForwardContext::new(seqlen_offsets, context_lens, position_ids, None, &flash)
                .with_recurrent_metadata(Some(RecurrentMetadata::new(kind, indices, None)));
        let seq_len = tokens.len() / batch;
        let input = Tensor::from_vec(tokens.to_vec(), (batch, seq_len), &Device::Cpu)?;
        model.forward(&input, &mut ctx)
    }

    /// Whether any QSA layer still holds main K/V or indexer state for a slot.
    fn qsa_has_cached_sequence(model: &Model, slot: usize) -> bool {
        model
            .layers
            .iter()
            .filter_map(|layer| match &layer.branch {
                DecoderBranch::Attention(qsa) => Some(qsa),
                DecoderBranch::Linear(_) => None,
            })
            .any(|qsa| {
                qsa.lock()
                    .expect("Qwen4Exp QSA layer mutex")
                    .has_cached_sequence(slot)
            })
    }

    /// Whether any PLE layer still holds predecessor or convolution state for a slot.
    fn ple_has_sequence(model: &Model, slot: usize) -> bool {
        model
            .ple
            .as_ref()
            .map(|ple| {
                ple.injections.iter().any(|(_, injection)| {
                    injection
                        .state
                        .lock()
                        .expect("Qwen4Exp PLE layer mutex")
                        .has_sequence(slot)
                })
            })
            .unwrap_or(false)
    }

    #[test]
    fn interleaved_mrope_tables_match_reference() -> Result<()> {
        let theta = 100.0f32;
        let rotary_dim = 4usize;
        let positions = [0u32, 1, 3];
        let (cos, sin) =
            interleaved_mrope_tables(theta, rotary_dim, &positions, &Device::Cpu, DType::F32)?;
        assert_eq!(cos.dims(), [positions.len(), rotary_dim / 2]);
        assert_eq!(sin.dims(), [positions.len(), rotary_dim / 2]);
        let cos_rows = cos.to_vec2::<f32>()?;
        let sin_rows = sin.to_vec2::<f32>()?;
        for (row, &position) in positions.iter().enumerate() {
            for pair in 0..rotary_dim / 2 {
                let angle = position as f32 * theta.powf(-(2.0 * pair as f32) / rotary_dim as f32);
                assert!(
                    (cos_rows[row][pair] - angle.cos()).abs() < 1e-5,
                    "cos mismatch at position {position} pair {pair}"
                );
                assert!(
                    (sin_rows[row][pair] - angle.sin()).abs() < 1e-5,
                    "sin mismatch at position {position} pair {pair}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn sectioned_interleaved_mrope_tables_use_designated_thw_pairs() -> Result<()> {
        let theta = 100.0f32;
        let positions = vec![
            vec![vec![2u32, 3]],
            vec![vec![5u32, 7]],
            vec![vec![11u32, 13]],
        ];
        let (cos, sin) = sectioned_interleaved_mrope_tables(
            theta,
            6,
            &[1, 1, 1],
            &positions,
            &Device::Cpu,
            DType::F32,
        )?;
        assert_eq!(cos.dims(), [1, 2, 3]);
        let cos = cos.to_vec3::<f32>()?;
        let sin = sin.to_vec3::<f32>()?;
        for token in 0..2 {
            for (pair, position) in [
                positions[0][0][token],
                positions[1][0][token],
                positions[2][0][token],
            ]
            .into_iter()
            .enumerate()
            {
                let angle = position as f32 * theta.powf(-(2.0 * pair as f32) / 6.0);
                assert!((cos[0][token][pair] - angle.cos()).abs() < 1e-5);
                assert!((sin[0][token][pair] - angle.sin()).abs() < 1e-5);
            }
        }
        Ok(())
    }

    #[test]
    fn prefill_then_decode_matches_one_shot_prefill() -> Result<()> {
        let config = tiny_config();
        config.validate()?;
        let model = build_model(&config, fixture_builder(&config)?)?;

        let prefill = run_forward(
            &model,
            &[3, 5, 7],
            1,
            &[0],
            &[(0, 3)],
            &[3],
            RecurrentBatchKind::Prefill,
        )?;
        assert_eq!(prefill.dims(), [1, 3, config.vocab_size]);

        let decode = run_forward(
            &model,
            &[9],
            1,
            &[3],
            &[(0, 1)],
            &[4],
            RecurrentBatchKind::Decode,
        )?;
        assert_eq!(decode.dims(), [1, 1, config.vocab_size]);

        let fresh = build_model(&config, fixture_builder(&config)?)?;
        let one_shot = run_forward(
            &fresh,
            &[3, 5, 7, 9],
            1,
            &[0],
            &[(0, 4)],
            &[4],
            RecurrentBatchKind::Prefill,
        )?;
        assert_eq!(one_shot.dims(), [1, 4, config.vocab_size]);

        let decode_vec = decode.to_vec3::<f32>()?;
        let one_shot_vec = one_shot.to_vec3::<f32>()?;
        assert_eq!(decode_vec.len(), 1);
        assert_eq!(decode_vec[0].len(), 1);
        let decode_last = &decode_vec[0][0];
        let one_shot_last = &one_shot_vec[0][3];
        assert_eq!(decode_last.len(), config.vocab_size);
        for (decode_logit, one_shot_logit) in decode_last.iter().zip(one_shot_last.iter()) {
            assert!(
                (decode_logit - one_shot_logit).abs() < 1e-4,
                "decode logit {decode_logit} does not match one-shot {one_shot_logit}"
            );
        }
        Ok(())
    }

    #[test]
    fn decoder_forward_is_deterministic() -> Result<()> {
        let config = tiny_config();
        let tokens = [3u32, 5, 7, 9];
        let first = run_forward(
            &build_model(&config, fixture_builder(&config)?)?,
            &tokens,
            1,
            &[0],
            &[(0, 4)],
            &[4],
            RecurrentBatchKind::Prefill,
        )?;
        let second = run_forward(
            &build_model(&config, fixture_builder(&config)?)?,
            &tokens,
            1,
            &[0],
            &[(0, 4)],
            &[4],
            RecurrentBatchKind::Prefill,
        )?;
        assert_eq!(first.to_vec3::<f32>()?, second.to_vec3::<f32>()?);
        Ok(())
    }

    #[test]
    fn batched_sequences_keep_independent_qsa_state() -> Result<()> {
        let config = tiny_config();
        config.validate()?;
        let model = build_model(&config, fixture_builder(&config)?)?;

        // Two rows prefill together while keeping independent QSA caches by state slot.
        let prefill = run_forward(
            &model,
            &[3, 5, 7, 11, 13, 15],
            2,
            &[0, 0],
            &[(0, 3), (0, 3)],
            &[3, 3],
            RecurrentBatchKind::Prefill,
        )?;
        assert_eq!(prefill.dims(), [2, 3, config.vocab_size]);

        let decode = run_forward(
            &model,
            &[9, 4],
            2,
            &[3, 3],
            &[(0, 1), (0, 1)],
            &[4, 4],
            RecurrentBatchKind::Decode,
        )?;
        assert_eq!(decode.dims(), [2, 1, config.vocab_size]);

        // Each row must equal a fresh one-shot prefill of its own tokens.
        let fresh_first = build_model(&config, fixture_builder(&config)?)?;
        let one_shot_first = run_forward(
            &fresh_first,
            &[3, 5, 7, 9],
            1,
            &[0],
            &[(0, 4)],
            &[4],
            RecurrentBatchKind::Prefill,
        )?;
        let fresh_second = build_model(&config, fixture_builder(&config)?)?;
        let one_shot_second = run_forward(
            &fresh_second,
            &[11, 13, 15, 4],
            1,
            &[0],
            &[(0, 4)],
            &[4],
            RecurrentBatchKind::Prefill,
        )?;

        let decode_rows = decode.to_vec3::<f32>()?;
        for (row, one_shot) in [(0usize, &one_shot_first), (1usize, &one_shot_second)] {
            let decode_last = &decode_rows[row][0];
            let one_shot_last = &one_shot.to_vec3::<f32>()?[0][3];
            for (decoded, expected) in decode_last.iter().zip(one_shot_last.iter()) {
                assert!(
                    (decoded - expected).abs() < 1e-4,
                    "row {row} decode logit {decoded} does not match one-shot {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn qsa_layers_require_sequence_state_indices() -> Result<()> {
        let config = tiny_config();
        let model = build_model(&config, fixture_builder(&config)?)?;
        let flash = FlashParams::empty(true);

        // Neither the cache nor the metadata carries host state indices.
        let indices = Tensor::from_vec(vec![0u32], (1,), &Device::Cpu)?;
        let mut ctx = ModelForwardContext::new(&[0], &[(0, 2)], &[2], None, &flash)
            .with_recurrent_metadata(Some(RecurrentMetadata::new(
                RecurrentBatchKind::Prefill,
                indices,
                None,
            )));
        let input = Tensor::from_vec(vec![3u32, 5], (1, 2), &Device::Cpu)?;
        let err = model.forward(&input, &mut ctx).unwrap_err();
        assert!(err.to_string().contains("sequence"), "{err}");

        // A host inventory shorter than the batch must also be rejected.
        let indices = Tensor::from_vec(vec![0u32], (1,), &Device::Cpu)?;
        let mut ctx = ModelForwardContext::new(&[0, 0], &[(0, 1), (0, 1)], &[1, 1], None, &flash)
            .with_recurrent_metadata(Some(RecurrentMetadata::new(
                RecurrentBatchKind::Prefill,
                indices,
                Some(vec![0]),
            )));
        let input = Tensor::from_vec(vec![3u32, 5], (2, 1), &Device::Cpu)?;
        let err = model.forward(&input, &mut ctx).unwrap_err();
        assert!(err.to_string().contains("sequence"), "{err}");
        Ok(())
    }

    #[test]
    fn ple_injection_prefill_decode_matches_one_shot() -> Result<()> {
        let config = ple_config();
        config.validate().expect("PLE config validates");
        let model = build_model(&config, fixture_builder(&config)?)?;

        // Identical non-PLE weights must produce different logits once PLE injects,
        // proving the injection path is live for the configured layer.
        let plain = build_model(&tiny_config(), fixture_builder(&config)?)?;
        let ple_prefill = run_forward(
            &model,
            &[3, 5, 7],
            1,
            &[0],
            &[(0, 3)],
            &[3],
            RecurrentBatchKind::Prefill,
        )?;
        let plain_prefill = run_forward(
            &plain,
            &[3, 5, 7],
            1,
            &[0],
            &[(0, 3)],
            &[3],
            RecurrentBatchKind::Prefill,
        )?;
        assert_ne!(
            ple_prefill.to_vec3::<f32>()?,
            plain_prefill.to_vec3::<f32>()?,
            "PLE injection did not change the logits"
        );

        let decode = run_forward(
            &model,
            &[9],
            1,
            &[3],
            &[(0, 1)],
            &[4],
            RecurrentBatchKind::Decode,
        )?;
        assert_eq!(decode.dims(), [1, 1, config.vocab_size]);

        let fresh = build_model(&config, fixture_builder(&config)?)?;
        let one_shot = run_forward(
            &fresh,
            &[3, 5, 7, 9],
            1,
            &[0],
            &[(0, 4)],
            &[4],
            RecurrentBatchKind::Prefill,
        )?;

        let decode_last = &decode.to_vec3::<f32>()?[0][0];
        let one_shot_last = &one_shot.to_vec3::<f32>()?[0][3];
        for (decoded, expected) in decode_last.iter().zip(one_shot_last.iter()) {
            assert!(
                (decoded - expected).abs() < 1e-4,
                "decode logit {decoded} does not match one-shot {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn ple_eos_reset_continues_across_decode() -> Result<()> {
        let config = ple_config();
        let model = build_model(&config, fixture_builder(&config)?)?;

        // EOS (15) inside the prefill resets the PLE predecessor window; the decode token
        // must hash against the reset history exactly as in a one-shot prefill.
        let prefill = run_forward(
            &model,
            &[3, 5, 15, 7],
            1,
            &[0],
            &[(0, 4)],
            &[4],
            RecurrentBatchKind::Prefill,
        )?;
        assert_eq!(prefill.dims(), [1, 4, config.vocab_size]);

        let decode = run_forward(
            &model,
            &[9],
            1,
            &[4],
            &[(0, 1)],
            &[5],
            RecurrentBatchKind::Decode,
        )?;

        let fresh = build_model(&config, fixture_builder(&config)?)?;
        let one_shot = run_forward(
            &fresh,
            &[3, 5, 15, 7, 9],
            1,
            &[0],
            &[(0, 5)],
            &[5],
            RecurrentBatchKind::Prefill,
        )?;

        let decode_last = &decode.to_vec3::<f32>()?[0][0];
        let one_shot_last = &one_shot.to_vec3::<f32>()?[0][4];
        for (decoded, expected) in decode_last.iter().zip(one_shot_last.iter()) {
            assert!(
                (decoded - expected).abs() < 1e-4,
                "decode logit {decoded} does not match one-shot {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn batched_sequences_keep_independent_ple_state() -> Result<()> {
        let config = ple_config();
        config.validate()?;
        let model = build_model(&config, fixture_builder(&config)?)?;

        // Two rows prefill together while keeping independent PLE hashing and convolution
        // state by state slot.
        let prefill = run_forward(
            &model,
            &[3, 5, 7, 11, 13, 15],
            2,
            &[0, 0],
            &[(0, 3), (0, 3)],
            &[3, 3],
            RecurrentBatchKind::Prefill,
        )?;
        assert_eq!(prefill.dims(), [2, 3, config.vocab_size]);

        let decode = run_forward(
            &model,
            &[9, 4],
            2,
            &[3, 3],
            &[(0, 1), (0, 1)],
            &[4, 4],
            RecurrentBatchKind::Decode,
        )?;
        assert_eq!(decode.dims(), [2, 1, config.vocab_size]);

        // Each row must equal a fresh one-shot prefill of its own tokens.
        let fresh_first = build_model(&config, fixture_builder(&config)?)?;
        let one_shot_first = run_forward(
            &fresh_first,
            &[3, 5, 7, 9],
            1,
            &[0],
            &[(0, 4)],
            &[4],
            RecurrentBatchKind::Prefill,
        )?;
        let fresh_second = build_model(&config, fixture_builder(&config)?)?;
        let one_shot_second = run_forward(
            &fresh_second,
            &[11, 13, 15, 4],
            1,
            &[0],
            &[(0, 4)],
            &[4],
            RecurrentBatchKind::Prefill,
        )?;

        let decode_rows = decode.to_vec3::<f32>()?;
        for (row, one_shot) in [(0usize, &one_shot_first), (1usize, &one_shot_second)] {
            let decode_last = &decode_rows[row][0];
            let one_shot_last = &one_shot.to_vec3::<f32>()?[0][3];
            for (decoded, expected) in decode_last.iter().zip(one_shot_last.iter()) {
                assert!(
                    (decoded - expected).abs() < 1e-4,
                    "row {row} decode logit {decoded} does not match one-shot {expected}"
                );
            }
        }
        Ok(())
    }

    /// Resetting one sequence's slot must clear its QSA and PLE state and rerunning the
    /// sequence must reproduce the original logits instead of reading stale state.
    #[test]
    fn cache_reset_restores_fresh_qsa_and_ple_state() -> Result<()> {
        let config = ple_config();
        config.validate()?;
        let model = build_model(&config, fixture_builder(&config)?)?;
        let slot = {
            let mut hybrid = model.kv_cache.hybrid();
            hybrid.allocate_seq(101)?
        };

        let run = |model: &Model| -> Result<Tensor> {
            let prefill = run_forward_with_slots(
                model,
                &[3, 5, 7],
                1,
                &[0],
                &[(0, 3)],
                &[3],
                RecurrentBatchKind::Prefill,
                &[slot as u32],
            )?;
            assert_eq!(prefill.dims(), [1, 3, config.vocab_size]);
            run_forward_with_slots(
                model,
                &[9],
                1,
                &[3],
                &[(0, 1)],
                &[4],
                RecurrentBatchKind::Decode,
                &[slot as u32],
            )
        };

        let first = run(&model)?;
        assert!(
            qsa_has_cached_sequence(&model, slot),
            "prefill must populate QSA auxiliary state for the slot"
        );
        assert!(
            ple_has_sequence(&model, slot),
            "prefill must populate PLE auxiliary state for the slot"
        );

        model.kv_cache.hybrid().reset_seq(101, slot)?;
        assert!(
            !qsa_has_cached_sequence(&model, slot),
            "reset_seq must clear QSA auxiliary state"
        );
        assert!(
            !ple_has_sequence(&model, slot),
            "reset_seq must clear PLE auxiliary state"
        );

        let second = run(&model)?;
        assert_eq!(first.to_vec3::<f32>()?, second.to_vec3::<f32>()?);
        Ok(())
    }

    /// Resetting one sequence's slot must leave other sequences' QSA and PLE state intact.
    #[test]
    fn cache_reset_keeps_other_sequences_auxiliary_state() -> Result<()> {
        let config = ple_config();
        config.validate()?;
        let model = build_model(&config, fixture_builder(&config)?)?;
        let slots = {
            let mut hybrid = model.kv_cache.hybrid();
            vec![
                hybrid.allocate_seq(201)? as u32,
                hybrid.allocate_seq(202)? as u32,
            ]
        };
        assert_ne!(slots[0], slots[1]);

        run_forward_with_slots(
            &model,
            &[3, 5, 7, 11, 13, 14],
            2,
            &[0, 0],
            &[(0, 3), (0, 3)],
            &[3, 3],
            RecurrentBatchKind::Prefill,
            &slots,
        )?;
        assert!(qsa_has_cached_sequence(&model, slots[0] as usize));
        assert!(ple_has_sequence(&model, slots[0] as usize));
        assert!(qsa_has_cached_sequence(&model, slots[1] as usize));
        assert!(ple_has_sequence(&model, slots[1] as usize));

        model.kv_cache.hybrid().reset_seq(201, slots[0] as usize)?;
        assert!(!qsa_has_cached_sequence(&model, slots[0] as usize));
        assert!(!ple_has_sequence(&model, slots[0] as usize));
        assert!(qsa_has_cached_sequence(&model, slots[1] as usize));
        assert!(ple_has_sequence(&model, slots[1] as usize));
        Ok(())
    }

    /// Releasing a finished sequence's slot drops its QSA and PLE state, and a fresh
    /// sequence reusing the slot must reproduce the original logits.
    #[test]
    fn cache_release_drops_auxiliary_state() -> Result<()> {
        let config = ple_config();
        config.validate()?;
        let model = build_model(&config, fixture_builder(&config)?)?;

        let run = |model: &Model, slot: usize| -> Result<Tensor> {
            let prefill = run_forward_with_slots(
                model,
                &[3, 5, 7],
                1,
                &[0],
                &[(0, 3)],
                &[3],
                RecurrentBatchKind::Prefill,
                &[slot as u32],
            )?;
            assert_eq!(prefill.dims(), [1, 3, config.vocab_size]);
            run_forward_with_slots(
                model,
                &[9],
                1,
                &[3],
                &[(0, 1)],
                &[4],
                RecurrentBatchKind::Decode,
                &[slot as u32],
            )
        };

        let slot = {
            let mut hybrid = model.kv_cache.hybrid();
            hybrid.allocate_seq(301)?
        };
        let first = run(&model, slot)?;
        assert!(qsa_has_cached_sequence(&model, slot));
        assert!(ple_has_sequence(&model, slot));

        assert!(model.kv_cache.hybrid().release_seq(301, slot)?);
        assert!(!qsa_has_cached_sequence(&model, slot));
        assert!(!ple_has_sequence(&model, slot));

        let reused = {
            let mut hybrid = model.kv_cache.hybrid();
            hybrid.allocate_seq(301)?
        };
        assert_eq!(reused, slot);
        let second = run(&model, reused)?;
        assert_eq!(first.to_vec3::<f32>()?, second.to_vec3::<f32>()?);
        Ok(())
    }

    /// Restoring a captured prefix into a different allocated slot must reproduce direct
    /// continuation across GDN recurrent state, QSA caches, and PLE histories together.
    #[test]
    fn prefix_snapshot_restore_matches_direct_continuation() -> Result<()> {
        let config = ple_config();
        config.validate()?;
        let model = build_model(&config, fixture_builder(&config)?)?;
        let (source_slot, restored_slot) = {
            let mut hybrid = model.kv_cache.hybrid();
            (hybrid.allocate_seq(401)?, hybrid.allocate_seq(402)?)
        };

        run_forward_with_slots(
            &model,
            &[3, 5, 7],
            1,
            &[0],
            &[(0, 3)],
            &[3],
            RecurrentBatchKind::Prefill,
            &[source_slot as u32],
        )?;
        let (recurrent, auxiliary) = {
            let hybrid = model.kv_cache.hybrid();
            (
                hybrid.snapshot_recurrent_state(401, source_slot)?,
                hybrid.snapshot_auxiliary_state(source_slot)?,
            )
        };

        let direct = run_forward_with_slots(
            &model,
            &[9],
            1,
            &[3],
            &[(0, 1)],
            &[4],
            RecurrentBatchKind::Decode,
            &[source_slot as u32],
        )?;
        assert!(!qsa_has_cached_sequence(&model, restored_slot));
        assert!(!ple_has_sequence(&model, restored_slot));

        {
            let mut hybrid = model.kv_cache.hybrid();
            hybrid.validate_auxiliary_restore(restored_slot, &auxiliary)?;
            hybrid.restore_recurrent_state(402, restored_slot, &recurrent)?;
            hybrid.restore_auxiliary_state(restored_slot, &auxiliary)?;
        }
        assert!(qsa_has_cached_sequence(&model, restored_slot));
        assert!(ple_has_sequence(&model, restored_slot));

        let restored = run_forward_with_slots(
            &model,
            &[9],
            1,
            &[3],
            &[(0, 1)],
            &[4],
            RecurrentBatchKind::Decode,
            &[restored_slot as u32],
        )?;
        let direct = direct.to_vec3::<f32>()?;
        let restored = restored.to_vec3::<f32>()?;
        for (actual, expected) in restored[0][0].iter().zip(&direct[0][0]) {
            assert!(
                (actual - expected).abs() < 1e-4,
                "restored prefix logit {actual} does not match direct continuation {expected}"
            );
        }
        Ok(())
    }

    /// The checkpoint object used around speculative/CUDA warmup must roll back the same
    /// mixed GDN/QSA/PLE state as an ordinary prefix snapshot.
    #[test]
    fn recurrent_checkpoint_rollback_restores_mixed_state() -> Result<()> {
        let config = ple_config();
        config.validate()?;
        let model = build_model(&config, fixture_builder(&config)?)?;
        let slot = model.kv_cache.hybrid().allocate_seq(501)?;

        run_forward_with_slots(
            &model,
            &[3, 5, 7],
            1,
            &[0],
            &[(0, 3)],
            &[3],
            RecurrentBatchKind::Prefill,
            &[slot as u32],
        )?;
        let checkpoint = model
            .kv_cache
            .hybrid()
            .snapshot_recurrent_checkpoint_state(slot)?;
        let expected = run_forward_with_slots(
            &model,
            &[9],
            1,
            &[3],
            &[(0, 1)],
            &[4],
            RecurrentBatchKind::Decode,
            &[slot as u32],
        )?;

        model
            .kv_cache
            .hybrid()
            .restore_recurrent_checkpoint_state(slot, &checkpoint)?;
        let restored = run_forward_with_slots(
            &model,
            &[9],
            1,
            &[3],
            &[(0, 1)],
            &[4],
            RecurrentBatchKind::Decode,
            &[slot as u32],
        )?;
        let expected = expected.to_vec3::<f32>()?;
        let restored = restored.to_vec3::<f32>()?;
        for (actual, expected) in restored[0][0].iter().zip(&expected[0][0]) {
            assert!(
                (actual - expected).abs() < 1e-4,
                "checkpoint rollback logit {actual} does not match original continuation {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn ple_forward_is_deterministic() -> Result<()> {
        let config = ple_config();
        let tokens = [3u32, 5, 7, 9];
        let first = run_forward(
            &build_model(&config, fixture_builder(&config)?)?,
            &tokens,
            1,
            &[0],
            &[(0, 4)],
            &[4],
            RecurrentBatchKind::Prefill,
        )?;
        let second = run_forward(
            &build_model(&config, fixture_builder(&config)?)?,
            &tokens,
            1,
            &[0],
            &[(0, 4)],
            &[4],
            RecurrentBatchKind::Prefill,
        )?;
        assert_eq!(first.to_vec3::<f32>()?, second.to_vec3::<f32>()?);
        Ok(())
    }

    #[test]
    fn paged_attention_is_fail_closed() -> Result<()> {
        let config = tiny_config();
        let err = build_model_mechanism(
            &config,
            fixture_builder(&config)?,
            AttentionImplementation::PagedAttention,
        )
        .err()
        .expect("expected paged-attention construction failure");
        assert!(err.to_string().contains("paged"), "{err}");
        Ok(())
    }

    /// Run one packed prefill forward with explicit logical query lengths and state slots,
    /// mirroring how the engine builds packed prompt batches.
    fn run_forward_packed(
        model: &Model,
        tokens: &[u32],
        query_lens: &[usize],
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        position_ids: &[usize],
        slots: &[u32],
    ) -> Result<Tensor> {
        let total: usize = query_lens.iter().sum();
        let max_q = *query_lens
            .iter()
            .max()
            .expect("packed prefill has query lengths");
        let mut cumulative = vec![0u32];
        let mut acc = 0u32;
        for &len in query_lens {
            acc += len as u32;
            cumulative.push(acc);
        }
        let cumulative_count = cumulative.len();
        let cumulative_tensor = Tensor::from_vec(cumulative, (cumulative_count,), &Device::Cpu)?;
        let flash = FlashParams {
            packed: true,
            max_q: max_q as u32,
            cumulative_seqlens_q: HashMap::from([(Device::Cpu.location(), cumulative_tensor)]),
            ..FlashParams::empty(true)
        };
        // Packed RoPE positions: each logical sequence starts at its own seqlen offset.
        let mut positions = Vec::with_capacity(total);
        for (&offset, &len) in seqlen_offsets.iter().zip(query_lens) {
            for token in 0..len {
                positions.push((offset + token) as u32);
            }
        }
        let positions_tensor = Tensor::from_vec(positions, (total,), &Device::Cpu)?;
        let mut metadata = PagedAttentionInputMetadata::dummy(&Device::Cpu)?;
        metadata.query_lens = Some(query_lens.to_vec());
        metadata.rope_positions = Some(HashMap::from([(Device::Cpu.location(), positions_tensor)]));
        let kv_placeholder = Tensor::zeros((1, 1, 1, 1), DType::F32, &Device::Cpu)?;
        let kv_cache = [(kv_placeholder.clone(), kv_placeholder)];
        let indices = Tensor::from_vec(slots.to_vec(), (slots.len(),), &Device::Cpu)?;
        model
            .kv_cache
            .hybrid()
            .set_physical_state_indices_with_host(Some(indices.clone()), Some(slots.to_vec()));
        let mut ctx = ModelForwardContext::new(
            seqlen_offsets,
            context_lens,
            position_ids,
            Some((&kv_cache, &metadata)),
            &flash,
        )
        .with_recurrent_metadata(Some(RecurrentMetadata::new(
            RecurrentBatchKind::Prefill,
            indices,
            Some(slots.to_vec()),
        )));
        let input = Tensor::from_vec(tokens.to_vec(), (1, total), &Device::Cpu)?;
        model.forward(&input, &mut ctx)
    }

    #[test]
    fn packed_prefill_matches_independent_rows() -> Result<()> {
        let config = tiny_config();
        config.validate()?;
        let model = build_model(&config, fixture_builder(&config)?)?;

        // Sequence A: five tokens on slot 0, sequence B: three tokens on slot 1, both
        // fresh at position 0. Token IDs stay inside the fixture's 16-entry vocabulary.
        let a_tokens = [3u32, 5, 7, 9, 11];
        let b_tokens = [4u32, 6, 8];
        let independent_a = run_forward_with_slots(
            &model,
            &a_tokens,
            1,
            &[0],
            &[(4, 1)],
            &[5],
            RecurrentBatchKind::Prefill,
            &[0],
        )?;
        let independent_b = run_forward_with_slots(
            &model,
            &b_tokens,
            1,
            &[0],
            &[(2, 1)],
            &[3],
            RecurrentBatchKind::Prefill,
            &[1],
        )?;

        // The packed run needs fresh per-sequence state, so use a freshly built but
        // identically seeded model.
        let packed_model = build_model(&config, fixture_builder(&config)?)?;
        let packed = run_forward_packed(
            &packed_model,
            &[a_tokens.as_slice(), b_tokens.as_slice()].concat(),
            &[5, 3],
            &[0, 0],
            &[(4, 1), (2, 1)],
            &[5, 3],
            &[0, 1],
        )?;
        assert_eq!(packed.dims(), [2, 1, config.vocab_size]);

        let a_rows = independent_a.to_vec3::<f32>()?;
        let b_rows = independent_b.to_vec3::<f32>()?;
        let packed_rows = packed.to_vec3::<f32>()?;
        let expected = [&a_rows[0][0], &b_rows[0][0]];
        for (row, expected_last) in expected.iter().enumerate() {
            for (got, want) in packed_rows[row][0].iter().zip(expected_last.iter()) {
                assert!(
                    (got - want).abs() < 1e-4,
                    "packed row {row} logit {got} does not match independent {want}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn packed_prefill_then_decode_matches_independent() -> Result<()> {
        let config = tiny_config();
        config.validate()?;

        // Packed prefill of A (five tokens, slot 0) and B (three tokens, slot 1), then a
        // decode continuation for slot 0. Token IDs stay inside the fixture vocabulary.
        let packed_model = build_model(&config, fixture_builder(&config)?)?;
        let packed = run_forward_packed(
            &packed_model,
            &[3u32, 5, 7, 9, 11, 4, 6, 8],
            &[5, 3],
            &[0, 0],
            &[(4, 1), (2, 1)],
            &[5, 3],
            &[0, 1],
        )?;
        assert_eq!(packed.dims(), [2, 1, config.vocab_size]);
        let packed_decode = run_forward_with_slots(
            &packed_model,
            &[12u32],
            1,
            &[5],
            &[(0, 1)],
            &[6],
            RecurrentBatchKind::Decode,
            &[0],
        )?;

        // The same pipeline run independently: prefill A alone, then decode one token.
        let independent = build_model(&config, fixture_builder(&config)?)?;
        run_forward_with_slots(
            &independent,
            &[3u32, 5, 7, 9, 11],
            1,
            &[0],
            &[(4, 1)],
            &[5],
            RecurrentBatchKind::Prefill,
            &[0],
        )?;
        let independent_decode = run_forward_with_slots(
            &independent,
            &[12u32],
            1,
            &[5],
            &[(0, 1)],
            &[6],
            RecurrentBatchKind::Decode,
            &[0],
        )?;

        let packed_rows = packed_decode.to_vec3::<f32>()?;
        let independent_rows = independent_decode.to_vec3::<f32>()?;
        for (got, want) in packed_rows[0][0].iter().zip(independent_rows[0][0].iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "decode after packed prefill logit {got} does not match independent {want}"
            );
        }
        Ok(())
    }

    #[test]
    fn packed_prefill_fail_closed_cases() -> Result<()> {
        let config = tiny_config();
        let model = build_model(&config, fixture_builder(&config)?)?;
        let kv_placeholder = Tensor::zeros((1, 1, 1, 1), DType::F32, &Device::Cpu)?;
        let kv_cache = [(kv_placeholder.clone(), kv_placeholder)];
        let flash = FlashParams {
            packed: true,
            ..FlashParams::empty(true)
        };
        let input = Tensor::from_vec(vec![3u32, 5], (1, 2), &Device::Cpu)?;

        // Packed flash parameters without logical query lengths.
        let indices = Tensor::from_vec(vec![0u32], (1,), &Device::Cpu)?;
        let mut ctx = ModelForwardContext::new(&[0], &[(0, 1)], &[1], None, &flash)
            .with_recurrent_metadata(Some(RecurrentMetadata::new(
                RecurrentBatchKind::Prefill,
                indices,
                None,
            )));
        let err = model.forward(&input, &mut ctx).unwrap_err();
        assert!(err.to_string().contains("query lengths"), "{err}");

        // Packed metadata on a non-first prompt chunk.
        let mut metadata = PagedAttentionInputMetadata::dummy(&Device::Cpu)?;
        metadata.query_lens = Some(vec![2]);
        metadata.is_first_prompt_chunk = false;
        let indices = Tensor::from_vec(vec![0u32], (1,), &Device::Cpu)?;
        let mut ctx =
            ModelForwardContext::new(&[0], &[(0, 1)], &[1], Some((&kv_cache, &metadata)), &flash)
                .with_recurrent_metadata(Some(RecurrentMetadata::new(
                    RecurrentBatchKind::Prefill,
                    indices,
                    None,
                )));
        let err = model.forward(&input, &mut ctx).unwrap_err();
        assert!(err.to_string().contains("first prompt chunk"), "{err}");

        // A decode recurrent batch kind.
        metadata.is_first_prompt_chunk = true;
        let indices = Tensor::from_vec(vec![0u32], (1,), &Device::Cpu)?;
        let mut ctx =
            ModelForwardContext::new(&[0], &[(0, 1)], &[1], Some((&kv_cache, &metadata)), &flash)
                .with_recurrent_metadata(Some(RecurrentMetadata::new(
                    RecurrentBatchKind::Decode,
                    indices,
                    None,
                )));
        let err = model.forward(&input, &mut ctx).unwrap_err();
        assert!(err.to_string().contains("non-prefill"), "{err}");

        // Logical and packed token counts disagree.
        let indices = Tensor::from_vec(vec![0u32], (1,), &Device::Cpu)?;
        let mut ctx =
            ModelForwardContext::new(&[0], &[(0, 1)], &[1], Some((&kv_cache, &metadata)), &flash)
                .with_recurrent_metadata(Some(RecurrentMetadata::new(
                    RecurrentBatchKind::Prefill,
                    indices,
                    None,
                )));
        let input = Tensor::from_vec(vec![3u32, 5, 7], (1, 3), &Device::Cpu)?;
        let err = model.forward(&input, &mut ctx).unwrap_err();
        assert!(err.to_string().contains("logical tokens"), "{err}");

        // Recurrent state index cardinality must cover every logical sequence.
        metadata.query_lens = Some(vec![1, 2]);
        let indices = Tensor::from_vec(vec![0u32], (1,), &Device::Cpu)?;
        let mut ctx =
            ModelForwardContext::new(&[0], &[(0, 1)], &[1], Some((&kv_cache, &metadata)), &flash)
                .with_recurrent_metadata(Some(RecurrentMetadata::new(
                    RecurrentBatchKind::Prefill,
                    indices,
                    None,
                )));
        let err = model.forward(&input, &mut ctx).unwrap_err();
        assert!(err.to_string().contains("recurrent state indices"), "{err}");
        Ok(())
    }
}
