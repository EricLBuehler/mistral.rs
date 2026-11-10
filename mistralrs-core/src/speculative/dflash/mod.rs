//! DFlash block-diffusion draft models (<https://arxiv.org/abs/2602.06036>).
//!
//! A small stack of Qwen3-style layers drafts a whole block of tokens in one pass: queries come
//! from the noise block `[anchor, mask, mask, ...]`, keys/values from both the projected target
//! context features (hidden states tapped from intermediate target layers) and the noise itself.
//! Context keys accumulate in a per-sequence cache; only accepted positions are ever appended, so
//! a rejected tail needs no drafter rollback. DFlash 2 adds two-tap grouped dynamic convolutions
//! around each sublayer and a candidate path selector over the top-k logits per position.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::{Arc, Mutex};

#[cfg(cuda_flash_attn)]
use candle_core::{
    cuda_backend::cudarc::driver::{sys, CudaStream},
    Var,
};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use mistralrs_quant::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, ShardedVarBuilder, UnquantLinear,
};
use serde::Deserialize;

use crate::layers::{yarn_inv_freq_and_attention_factor, RmsNorm, YarnRopeConfig};
use crate::prefix_cacher::PagedAuxiliaryPrefixState;
use crate::speculative::{MtpConfig, MtpDraftSamplingMethod, SpeculativePrefixReplay};
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};

#[cfg(cuda_flash_attn)]
use crate::paged_attention::windowed_pool::{
    WindowedKvBatch, WindowedKvBatchTensors, WindowedKvCheckpoint, WindowedKvPool,
    WindowedKvPoolConfig, WindowedKvQuery,
};
#[cfg(cuda_flash_attn)]
use crate::pipeline::cuda_graph::{
    record_cuda_graph_dispatch, record_cuda_graph_evictions, record_cuda_graph_resident_entries,
    take_cuda_graph_capacity_eviction, CudaGraphComponent, CudaGraphDispatchMode,
    CudaGraphDispatchReason, CudaGraphEvent, CudaGraphEventGuard, CudaGraphEvictionReason,
    CudaPhaseTimer,
};

mod config;
#[cfg(any(cuda_flash_attn, test))]
mod graph;
pub(crate) mod host;
mod selector;

pub use config::*;
#[cfg(any(cuda_flash_attn, test))]
use graph::*;
use selector::*;

const DEFAULT_BLOCK_SIZE: usize = 16;
pub const DEFAULT_MAX_DRAFTS: usize = 7;
/// Depths the speculative autotuner chooses between for a block drafter; block drafts accept longer runs.
pub const AUTO_DEPTHS: [usize; 3] = [3, 5, 7];
// Eager forwards use this cache; CUDA graphs derive RoPE from their replayed position inputs.
const ROPE_CACHE_LEN: usize = 65536;
const MASK_CACHE_CAP: usize = 64;
#[cfg(cuda_flash_attn)]
const DFLASH_CUDA_GRAPH_DEFAULT_CACHE_CAPACITY: usize = 40;
const ADAPT_FULL_DEPTH_MAX_BATCH: usize = 8;
const ADAPT_BATCH_DEPTH: usize = 3;
const DFLASH_ADAPTIVE_ENV: &str = "MISTRALRS_DFLASH_ADAPTIVE";

fn select_dflash_depth(adaptive: bool, max_n: usize, batch: usize, live_sequences: usize) -> usize {
    if max_n == 0 || !adaptive || batch.max(live_sequences) <= ADAPT_FULL_DEPTH_MAX_BATCH {
        max_n
    } else {
        ADAPT_BATCH_DEPTH.min(max_n)
    }
}

fn dflash_adaptive_env_value(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true")
}

fn dflash_adaptive_supported(max_n: usize, max_live_sequences: usize) -> bool {
    max_n > ADAPT_BATCH_DEPTH && max_live_sequences > ADAPT_FULL_DEPTH_MAX_BATCH
}

#[cfg(any(cuda_flash_attn, test))]
fn dflash_graph_positions_fit(start_positions: &[usize], block: usize) -> bool {
    block > 0
        && start_positions.iter().all(|start| {
            start
                .checked_add(block - 1)
                .is_some_and(|last| u32::try_from(last).is_ok())
        })
}

#[cfg(any(cuda_flash_attn, test))]
fn dflash_rope_from_positions(
    positions: &Tensor,
    inv_freq: &Tensor,
    dtype: DType,
    attention_factor: f32,
) -> Result<(Tensor, Tensor)> {
    let positions = positions.flatten_all()?;
    #[cfg(feature = "cuda")]
    {
        let rope_dtype = if attention_factor == 1.0 {
            dtype
        } else {
            DType::F32
        };
        if let Some(rope) =
            crate::ops::try_cuda_rope_sincos_positions(&positions, inv_freq, rope_dtype)?
        {
            return finalize_dflash_rope(rope, dtype, attention_factor);
        }
    }
    let positions = positions.to_dtype(DType::F32)?.unsqueeze(1)?;
    let inv_freq = inv_freq.to_dtype(DType::F32)?.unsqueeze(0)?;
    let freqs = positions.broadcast_mul(&inv_freq)?;
    finalize_dflash_rope((freqs.cos()?, freqs.sin()?), dtype, attention_factor)
}

fn finalize_dflash_rope(
    (mut cos, mut sin): (Tensor, Tensor),
    dtype: DType,
    attention_factor: f32,
) -> Result<(Tensor, Tensor)> {
    if attention_factor != 1.0 {
        cos = (cos * attention_factor as f64)?;
        sin = (sin * attention_factor as f64)?;
    }
    Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
}

pub(crate) fn dflash_adaptive_requested() -> bool {
    std::env::var(DFLASH_ADAPTIVE_ENV)
        .ok()
        .as_deref()
        .map(dflash_adaptive_env_value)
        .unwrap_or(true)
}

fn dflash_graph_plans(adaptive: bool, max_n: usize) -> Vec<super::SpeculativeGraphPlan> {
    if max_n == 0 {
        return Vec::new();
    }
    if !adaptive || max_n <= ADAPT_BATCH_DEPTH {
        return vec![super::SpeculativeGraphPlan::new(max_n, None)];
    }
    vec![
        super::SpeculativeGraphPlan::new(max_n, Some(ADAPT_FULL_DEPTH_MAX_BATCH)),
        super::SpeculativeGraphPlan::new(ADAPT_BATCH_DEPTH, None),
    ]
}

#[cfg(any(cuda_flash_attn, test))]
fn dflash_graph_precapture_shapes(
    plans: &[super::SpeculativeGraphPlan],
    batches: impl IntoIterator<Item = usize>,
    sequence_capacity: usize,
) -> Vec<(usize, usize)> {
    let batches = batches.into_iter().collect::<Vec<_>>();
    let mut shapes = Vec::new();
    for plan in plans {
        let max_batch = plan
            .max_batch_size
            .unwrap_or(sequence_capacity)
            .min(sequence_capacity);
        for batch in batches.iter().copied().filter(|batch| *batch <= max_batch) {
            shapes.push((batch, plan.proposal_len + 1));
        }
    }
    shapes.sort_unstable();
    shapes.dedup();
    shapes
}

#[cfg(any(cuda_flash_attn, test))]
fn drain_dflash_lru_entries<T>(entries: &mut Vec<T>, max_entries: usize) -> Vec<T> {
    let count = max_entries.min(entries.len());
    entries.drain(..count).collect()
}

fn dflash_prefix_replay(
    layer_windows: impl IntoIterator<Item = Option<usize>>,
) -> SpeculativePrefixReplay {
    let mut required = 0;
    for window in layer_windows {
        let Some(window) = window else {
            return SpeculativePrefixReplay::Full;
        };
        required = required.max(window);
    }
    if required == 0 {
        SpeculativePrefixReplay::NotRequired
    } else {
        SpeculativePrefixReplay::Suffix(required)
    }
}

struct DFlashLayer {
    qkv_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    gate_up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    attention_conv: Option<DynamicConv>,
    mlp_conv: Option<DynamicConv>,
    is_causal: bool,
    sliding_window: Option<usize>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum DraftAttentionLayout {
    HeadsFirst,
    #[cfg(cuda_flash_attn)]
    TokensFirst,
}

/// Per-sequence accumulated context keys/values stacked over layers,
/// `[layers, kv_heads, len, head_dim]` with rotary already applied to the keys. `next_pos` is the
/// absolute position of the next token to append; `start_pos` the absolute position of the first
/// cached entry (after window trimming).
struct SeqCtxCache {
    k: Tensor,
    v: Tensor,
    start_pos: usize,
    next_pos: usize,
}

/// One sequence's context rows to append at absolute positions `start_pos..start_pos + rows`.
pub struct CtxAppend {
    pub seq_id: usize,
    pub rows: usize,
    pub start_pos: usize,
}

pub struct DFlashPreparedContext {
    k: Tensor,
    v: Tensor,
    rows: usize,
}

fn contiguous_row_range(indices: &[u32]) -> Option<(usize, usize)> {
    let start = usize::try_from(*indices.first()?).ok()?;
    indices
        .iter()
        .enumerate()
        .all(|(offset, index)| usize::try_from(*index).ok() == start.checked_add(offset))
        .then_some((start, indices.len()))
}

fn select_ctx_kv_rows(k: &Tensor, v: &Tensor, row_indices: &[u32]) -> Result<(Tensor, Tensor)> {
    if row_indices.is_empty() {
        return Ok((k.narrow(2, 0, 0)?, v.narrow(2, 0, 0)?));
    }
    if let Some((start, len)) = contiguous_row_range(row_indices) {
        return Ok((k.narrow(2, start, len)?, v.narrow(2, start, len)?));
    }
    let row_indices = Tensor::from_vec(row_indices.to_vec(), (row_indices.len(),), k.device())?;
    Ok((
        k.contiguous()?.index_select(&row_indices, 2)?,
        v.contiguous()?.index_select(&row_indices, 2)?,
    ))
}

fn gather_ctx_taps(taps: &[Tensor], flat_row_indices: Vec<u32>, device: &Device) -> Result<Tensor> {
    let Some(first) = taps.first() else {
        candle_core::bail!("DFlash context append has no taps");
    };
    let (source_batch, source_rows, _) = first.dims3()?;
    let source_len = source_batch
        .checked_mul(source_rows)
        .ok_or_else(|| candle_core::Error::msg("DFlash context tap row count overflow"))?;
    if flat_row_indices
        .iter()
        .any(|index| usize::try_from(*index).map_or(true, |index| index >= source_len))
    {
        candle_core::bail!("DFlash context tap row index is out of range");
    }
    #[cfg(feature = "cuda")]
    if let Some(packed) = crate::cuda::dflash_context::pack_taps(taps, &flat_row_indices)? {
        return packed.unsqueeze(0);
    }
    let contiguous_range = contiguous_row_range(&flat_row_indices);
    let indices = if contiguous_range.is_none() {
        let len = flat_row_indices.len();
        Some(Tensor::from_vec(flat_row_indices, (len,), device)?)
    } else {
        None
    };
    let mut gathered = Vec::with_capacity(taps.len());
    for tap in taps {
        let (batch, rows, hidden) = tap.dims3()?;
        if (batch, rows) != (source_batch, source_rows) {
            candle_core::bail!("DFlash context tap row shapes changed");
        }
        if !tap.device().same_device(device) {
            candle_core::bail!("DFlash context taps must be on the draft device");
        }
        let tap = if tap.is_contiguous() {
            tap.clone()
        } else {
            tap.contiguous()?
        };
        let flat = tap.reshape((source_len, hidden))?;
        gathered.push(match contiguous_range {
            Some((start, len)) => flat.narrow(0, start, len)?,
            None => flat.index_select(indices.as_ref().expect("non-contiguous row indices"), 0)?,
        });
    }
    let packed = if gathered.len() == 1 {
        gathered.pop().expect("one DFlash tap")
    } else {
        Tensor::cat(&gathered.iter().collect::<Vec<_>>(), D::Minus1)?
    };
    packed.unsqueeze(0)
}

struct AdaptiveState {
    max_n: usize,
}

#[derive(Clone, Copy)]
enum DFlashSequenceEviction {
    Dormant,
    Released,
}

#[derive(Clone, Copy)]
pub(crate) struct DFlashSamplingInputs<'a> {
    pub(crate) inverse_temperatures: &'a [f32],
    pub(crate) uniforms: &'a [f32],
}

pub(crate) enum DFlashProposalBatch {
    Tokens(Vec<Vec<u32>>),
    #[cfg(cuda_flash_attn)]
    DeviceTokens(Tensor),
    #[cfg(feature = "cuda")]
    DeviceSparse {
        tokens: Tensor,
        candidate_ids: Tensor,
        candidate_probs: Tensor,
    },
}

fn update_dormant_sequences(
    dormant: &mut HashSet<usize>,
    seq_ids: &[usize],
    eviction: DFlashSequenceEviction,
) {
    match eviction {
        DFlashSequenceEviction::Dormant => dormant.extend(seq_ids.iter().copied()),
        DFlashSequenceEviction::Released => {
            for seq_id in seq_ids {
                dormant.remove(seq_id);
            }
        }
    }
}

#[cfg(cuda_flash_attn)]
struct DFlashWindowedForward<'a> {
    noise_embedding: &'a Tensor,
    q_cos: &'a Tensor,
    q_sin: &'a Tensor,
    batch: usize,
    block: usize,
    attention_batch: &'a WindowedKvBatch,
    metadata: &'a WindowedKvBatchTensors,
}

pub struct DFlashDraftModel {
    #[cfg(cuda_flash_attn)]
    cuda_graphs: Mutex<DFlashCudaGraphState>,
    layers: Vec<DFlashLayer>,
    fc: Arc<dyn QuantMethod>,
    context_kv_proj: Arc<dyn QuantMethod>,
    #[cfg(feature = "cuda")]
    context_k_norm_weights: Option<Tensor>,
    hidden_norm: RmsNorm,
    norm: RmsNorm,
    selector: Option<CandidateSelector>,
    draft_sampling_method: MtpDraftSamplingMethod,
    inv_freq: Tensor,
    rope_attention_factor: f32,
    pub target_layer_ids: Vec<usize>,
    mask_token_id: u32,
    block_size: usize,
    input_embedding_scale: f64,
    output_multiplier: f64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    dtype: DType,
    device: Device,
    ctx_cache: Mutex<HashMap<usize, SeqCtxCache>>,
    #[cfg(cuda_flash_attn)]
    windowed_pool: Option<Mutex<WindowedKvPool>>,
    dormant_seqs: Mutex<HashSet<usize>>,
    // cos/sin for positions 0..ROPE_CACHE_LEN, [len, head_dim/2]
    rope_table: (Tensor, Tensor),
    mask_cache: Mutex<HashMap<MaskKey, Tensor>>,
    adaptive: Mutex<Option<AdaptiveState>>,
}

#[cfg(cuda_flash_attn)]
struct DFlashPagedPrefixState {
    checkpoint: WindowedKvCheckpoint,
}

#[cfg(cuda_flash_attn)]
impl PagedAuxiliaryPrefixState for DFlashPagedPrefixState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn bytes(&self) -> usize {
        self.checkpoint.bytes()
    }
}

fn load_linear(
    vb: &ShardedVarBuilder,
    shape: (usize, usize),
    name: &str,
    isq: Option<IsqType>,
    device: &Device,
) -> Result<Arc<dyn QuantMethod>> {
    let weight = vb.pp(name).get((shape.1, shape.0), "weight")?;
    linear_from_weight(weight, isq, device)
}

fn linear_from_weight(
    weight: Tensor,
    isq: Option<IsqType>,
    device: &Device,
) -> Result<Arc<dyn QuantMethod>> {
    let layer: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
        QuantMethodConfig::Unquantized(candle_nn::Linear::new(weight, None)),
    )?);
    match isq {
        Some(ty) => layer.apply_isq(
            Some(ty),
            device.clone(),
            &std::sync::atomic::AtomicUsize::new(0),
            None,
            QuantizeOntoGuard::new(),
        ),
        None => Ok(layer),
    }
}

/// Reads only the drafter's config when it identifies a DFlash checkpoint.
pub fn peek_config(config: &MtpConfig) -> Result<Option<DFlashConfig>> {
    let path = config.resolve_path()?;
    let raw = fs::read_to_string(path.join("config.json"))
        .map_err(|e| candle_core::Error::Msg(format!("failed to read MTP model config: {e}")))?;
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(candle_core::Error::msg)?;
    let is_dflash = value
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|architectures| {
            architectures.iter().any(|architecture| {
                architecture
                    .as_str()
                    .is_some_and(|name| name.contains("DFlash") || name.contains("Dflash"))
            })
        });
    if !is_dflash {
        return Ok(None);
    }
    serde_json::from_value(value)
        .map(Some)
        .map_err(candle_core::Error::msg)
}

#[cfg(cuda_flash_attn)]
pub(crate) fn windowed_kv_checkpoint_capacity(retained_prefixes: usize) -> Result<usize> {
    if retained_prefixes == 0 {
        return Ok(0);
    }
    retained_prefixes
        .checked_add(1)
        .ok_or_else(|| candle_core::Error::msg("DFlash prefix checkpoint capacity overflow"))
}

#[cfg(cuda_flash_attn)]
pub(crate) fn windowed_kv_cache_size_in_bytes(
    config: &MtpConfig,
    live_sequence_capacity: usize,
    retained_prefixes: usize,
    page_size: usize,
) -> Result<usize> {
    let path = config.resolve_path()?;
    let raw = fs::read_to_string(path.join("config.json"))
        .map_err(|err| candle_core::Error::msg(format!("failed to read MTP config: {err}")))?;
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(candle_core::Error::msg)?;
    let is_dflash = value
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|architectures| {
            architectures.iter().any(|architecture| {
                architecture
                    .as_str()
                    .is_some_and(|name| name.contains("DFlash") || name.contains("Dflash"))
            })
        });
    if !is_dflash {
        return Ok(0);
    }
    let cfg: DFlashConfig = serde_json::from_str(&raw).map_err(candle_core::Error::msg)?;
    let windows = (0..cfg.num_hidden_layers)
        .map(|layer| cfg.layer_attention(layer).1)
        .collect::<Vec<_>>();
    if windows.iter().any(Option::is_none) {
        return Ok(0);
    }
    let max_window = windows
        .into_iter()
        .flatten()
        .max()
        .expect("DFlash has at least one layer");
    let retained_tokens = max_window
        .checked_sub(1)
        .and_then(|value| value.checked_add(cfg.block_size()))
        .and_then(|value| value.checked_add(page_size - 1))
        .ok_or_else(|| candle_core::Error::msg("DFlash windowed KV capacity overflow"))?;
    let pages_per_sequence = retained_tokens.div_ceil(page_size);
    let checkpoint_capacity = windowed_kv_checkpoint_capacity(retained_prefixes)?;
    let slot_capacity = live_sequence_capacity
        .checked_add(checkpoint_capacity)
        .ok_or_else(|| candle_core::Error::msg("DFlash windowed KV slot capacity overflow"))?;
    let elements = cfg
        .num_hidden_layers
        .checked_mul(slot_capacity)
        .and_then(|value| value.checked_mul(pages_per_sequence))
        .and_then(|value| value.checked_mul(cfg.num_key_value_heads))
        .and_then(|value| value.checked_mul(page_size))
        .and_then(|value| value.checked_mul(cfg.head_dim()))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| candle_core::Error::msg("DFlash windowed KV size overflow"))?;
    elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| candle_core::Error::msg("DFlash windowed KV byte size overflow"))
}

pub(crate) struct DFlashGraphProposalInputs<'a> {
    pub seq_ids: &'a [usize],
    pub anchors: &'a [u32],
    pub start_positions: &'a [usize],
    pub n_predict: usize,
    pub sampling: Option<DFlashSamplingInputs<'a>>,
    pub token_embedding: &'a Arc<dyn QuantMethod>,
    pub lm_head: &'a Arc<dyn QuantMethod>,
}

impl DFlashDraftModel {
    pub(crate) fn evict_cuda_graphs_lru(&self, max_entries: usize) -> usize {
        #[cfg(cuda_flash_attn)]
        {
            self.cuda_graphs
                .lock()
                .expect("dflash CUDA graph cache poisoned")
                .evict_lru_for_memory_pressure(max_entries)
        }
        #[cfg(not(cuda_flash_attn))]
        {
            let _ = max_entries;
            0
        }
    }

    /// Loads a DFlash/DFlash2 drafter from a local path or HF repo. Target metadata validates the
    /// checkpoint against the target model and supplies any shared RoPE scaling parameters.
    pub fn load(config: &MtpConfig, target: DFlashLoadTarget<'_>, silent: bool) -> Result<Self> {
        let DFlashLoadTarget {
            num_layers: target_num_layers,
            hidden_size: target_hidden_size,
            yarn_rope_config,
            device,
            dtype,
        } = target;
        let path = config.resolve_path()?;
        let raw = fs::read_to_string(path.join("config.json"))
            .map_err(|e| candle_core::Error::Msg(format!("failed to read DFlash config: {e}")))?;
        let cfg: DFlashConfig = serde_json::from_str(&raw).map_err(candle_core::Error::msg)?;
        if !cfg.is_dflash() {
            candle_core::bail!(
                "`--mtp-model` for this target must be a DFlash draft model; got architectures {:?}",
                cfg.architectures
            );
        }
        cfg.validate_rope_type()?;
        if cfg.hidden_size != target_hidden_size {
            candle_core::bail!(
                "DFlash hidden size {} does not match target hidden size {target_hidden_size}",
                cfg.hidden_size
            );
        }
        let target_layer_ids = cfg.target_layer_ids()?;
        if let Some(max) = target_layer_ids.iter().max() {
            if *max >= target_num_layers {
                candle_core::bail!(
                    "DFlash taps target layer {max} but the target has {target_num_layers} layers"
                );
            }
        }
        let head_dim = cfg.head_dim();
        let (inv_freq, rope_attention_factor) = match cfg.yarn_rope_config(yarn_rope_config)? {
            Some(yarn) => {
                tracing::info!(
                    factor = yarn.factor,
                    original_max_position_embeddings = yarn.original_max_position_embeddings,
                    max_position_embeddings = yarn.max_position_embeddings,
                    "Using target YaRN scaling for DFlash rotary embeddings"
                );
                yarn_inv_freq_and_attention_factor(&yarn, device)?
            }
            None => {
                #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                let inv_freq = (0..head_dim)
                    .step_by(2)
                    .map(|i| 1f32 / (cfg.rope_theta() as f32).powf(i as f32 / head_dim as f32))
                    .collect::<Vec<_>>();
                (Tensor::from_vec(inv_freq, (head_dim / 2,), device)?, 1.0)
            }
        };

        let mut weight_paths = fs::read_dir(&path)
            .map_err(|e| {
                candle_core::Error::Msg(format!("failed to list {}: {e}", path.display()))
            })?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        weight_paths.sort();
        if weight_paths.is_empty() {
            candle_core::bail!(
                "DFlash model directory {} has no safetensors",
                path.display()
            );
        }
        let vb = from_mmaped_safetensors(
            weight_paths,
            Vec::new(),
            Some(dtype),
            device,
            Vec::new(),
            silent,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )?;

        // The drafter conditions on quantized target hiddens already; its own precision is a
        // tunable acceptance/speed trade (bf16 reads ~4 GB per draft block, q4k ~1.1 GB).
        let isq = match std::env::var("MISTRALRS_DFLASH_ISQ").ok().as_deref() {
            Some("none" | "bf16") => None,
            Some(name) => Some(
                crate::pipeline::parse_isq_value(name, Some(device))
                    .map_err(candle_core::Error::Msg)?,
            ),
            None => config.draft_lm_head_isq,
        };
        let hidden = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut context_kv_weights = Vec::with_capacity(cfg.num_hidden_layers * 2);
        for i in 0..cfg.num_hidden_layers {
            let vb_l = vb.pp("layers").pp(i);
            let vb_attn = vb_l.pp("self_attn");
            let (is_causal, sliding_window) = cfg.layer_attention(i);
            let q_size = cfg.num_attention_heads * head_dim;
            let kv_size = cfg.num_key_value_heads * head_dim;
            let q_weight = vb_attn.pp("q_proj").get((q_size, hidden), "weight")?;
            let k_weight = vb_attn.pp("k_proj").get((kv_size, hidden), "weight")?;
            let v_weight = vb_attn.pp("v_proj").get((kv_size, hidden), "weight")?;
            let qkv_weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;
            context_kv_weights.push(k_weight);
            context_kv_weights.push(v_weight);

            let vb_mlp = vb_l.pp("mlp");
            let gate_weight = vb_mlp
                .pp("gate_proj")
                .get((cfg.intermediate_size, hidden), "weight")?;
            let up_weight = vb_mlp
                .pp("up_proj")
                .get((cfg.intermediate_size, hidden), "weight")?;
            let gate_up_weight = Tensor::cat(&[&gate_weight, &up_weight], 0)?;
            layers.push(DFlashLayer {
                qkv_proj: linear_from_weight(qkv_weight, isq, device)?,
                o_proj: load_linear(
                    &vb_attn,
                    (cfg.num_attention_heads * head_dim, hidden),
                    "o_proj",
                    isq,
                    device,
                )?,
                q_norm: RmsNorm::new(head_dim, eps, vb_attn.pp("q_norm"))?,
                k_norm: RmsNorm::new(head_dim, eps, vb_attn.pp("k_norm"))?,
                input_layernorm: RmsNorm::new(hidden, eps, vb_l.pp("input_layernorm"))?,
                post_attention_layernorm: RmsNorm::new(
                    hidden,
                    eps,
                    vb_l.pp("post_attention_layernorm"),
                )?,
                gate_up_proj: linear_from_weight(gate_up_weight, isq, device)?,
                down_proj: load_linear(
                    &vb_mlp,
                    (cfg.intermediate_size, hidden),
                    "down_proj",
                    isq,
                    device,
                )?,
                attention_conv: cfg
                    .is_v2()
                    .then(|| DynamicConv::load(vb_l.pp("attention_conv"), &cfg))
                    .transpose()?,
                mlp_conv: cfg
                    .is_v2()
                    .then(|| DynamicConv::load(vb_l.pp("mlp_conv"), &cfg))
                    .transpose()?,
                is_causal,
                sliding_window,
            });
        }
        let context_kv_weight = Tensor::cat(&context_kv_weights.iter().collect::<Vec<_>>(), 0)?;
        let context_kv_proj = linear_from_weight(context_kv_weight, isq, device)?;
        #[cfg(feature = "cuda")]
        let context_k_norm_weights = if device.is_cuda() {
            let weights = layers
                .iter()
                .map(|layer| layer.k_norm.weight().clone())
                .collect::<Vec<_>>();
            Some(Tensor::stack(&weights, 0)?)
        } else {
            None
        };
        let fc = load_linear(
            &vb,
            (target_layer_ids.len() * hidden, hidden),
            "fc",
            isq,
            device,
        )?;
        let selector = cfg
            .is_v2()
            .then(|| CandidateSelector::load(vb.pp("candidate_selector"), &cfg))
            .transpose()?;
        let selector_capability = match selector.as_ref() {
            None => Err("a DFlash2 checkpoint with a candidate selector is required".to_string()),
            #[cfg(feature = "cuda")]
            Some(selector) => selector.configured_cuda_capability(device.is_cuda(), dtype),
            #[cfg(not(feature = "cuda"))]
            Some(_) => Err("this build has no CUDA candidate selector support".to_string()),
        };
        let draft_sampling_method =
            resolve_dflash_sampling_policy(config.draft_sampling_method, selector_capability)?;

        let rope_table = {
            #[allow(clippy::cast_precision_loss)]
            let pos: Vec<f32> = (0..ROPE_CACHE_LEN).map(|p| p as f32).collect();
            let pos = Tensor::from_vec(pos, (ROPE_CACHE_LEN, 1), device)?;
            let freqs = pos.broadcast_matmul(&inv_freq.reshape((1, ()))?)?;
            finalize_dflash_rope((freqs.cos()?, freqs.sin()?), dtype, rope_attention_factor)?
        };

        Ok(Self {
            #[cfg(cuda_flash_attn)]
            cuda_graphs: Mutex::new(DFlashCudaGraphState::default()),
            layers,
            fc,
            context_kv_proj,
            #[cfg(feature = "cuda")]
            context_k_norm_weights,
            hidden_norm: RmsNorm::new(hidden, eps, vb.pp("hidden_norm"))?,
            norm: RmsNorm::new(hidden, eps, vb.pp("norm"))?,
            selector,
            draft_sampling_method,
            inv_freq,
            rope_attention_factor,
            target_layer_ids,
            mask_token_id: cfg.mask_token_id()?,
            block_size: cfg.block_size(),
            input_embedding_scale: cfg.dflash_config.input_embedding_scale.unwrap_or(1.0),
            output_multiplier: cfg.dflash_config.output_multiplier.unwrap_or(1.0),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            intermediate_size: cfg.intermediate_size,
            dtype,
            device: device.clone(),
            ctx_cache: Mutex::new(HashMap::new()),
            #[cfg(cuda_flash_attn)]
            windowed_pool: None,
            dormant_seqs: Mutex::new(HashSet::new()),
            rope_table,
            mask_cache: Mutex::new(HashMap::new()),
            adaptive: Mutex::new(None),
        })
    }

    pub fn enable_windowed_kv(
        &mut self,
        live_sequence_capacity: usize,
        retained_prefixes: usize,
    ) -> Result<bool> {
        #[cfg(cuda_flash_attn)]
        {
            if self.dtype != DType::BF16 || !self.device.is_cuda() {
                return Ok(false);
            }
            let layer_windows = self
                .layers
                .iter()
                .map(|layer| layer.sliding_window)
                .collect::<Vec<_>>();
            if layer_windows.iter().any(Option::is_none) {
                return Ok(false);
            }
            let checkpoint_capacity = windowed_kv_checkpoint_capacity(retained_prefixes)?;
            let config = WindowedKvPoolConfig::new_with_capacities(
                live_sequence_capacity,
                checkpoint_capacity,
                layer_windows,
                self.block_size,
                crate::paged_attention::DEFAULT_PAGED_ATTENTION_BLOCK_SIZE,
                self.num_kv_heads,
                self.head_dim,
            )?;
            let pages = config.pages_per_sequence();
            self.windowed_pool = Some(Mutex::new(WindowedKvPool::new(
                config,
                &self.device,
                "dflash",
            )?));
            tracing::info!(
                live_sequence_capacity,
                retained_prefixes,
                checkpoint_capacity,
                pages_per_sequence = pages,
                "Using bounded paged FlashAttention KV for DFlash"
            );
            Ok(true)
        }
        #[cfg(not(cuda_flash_attn))]
        {
            let _ = (live_sequence_capacity, retained_prefixes);
            Ok(false)
        }
    }

    #[cfg(cuda_flash_attn)]
    pub(crate) fn precapture_cuda_graphs(
        &self,
        max_n: usize,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
    ) -> Result<()> {
        if !crate::pipeline::cuda_graph::cuda_decode_graphs_enabled() || max_n == 0 {
            return Ok(());
        }
        let Some(pool) = &self.windowed_pool else {
            return Ok(());
        };
        let use_selector = self.selector.is_some();
        if use_selector
            && self
                .selector
                .as_ref()
                .expect("selector checked above")
                .configured_cuda_capability(self.device.is_cuda(), self.dtype)
                .is_err()
        {
            return Ok(());
        }
        let selector_modes = match (use_selector, self.draft_sampling_method) {
            (false, _) => vec![DFlashSelectorMode::Disabled],
            (true, MtpDraftSamplingMethod::Auto) => {
                unreachable!("DFlash draft sampling is resolved during loading")
            }
            (true, MtpDraftSamplingMethod::Greedy) => vec![DFlashSelectorMode::Greedy],
            (true, MtpDraftSamplingMethod::Probabilistic) => {
                vec![DFlashSelectorMode::Sampling, DFlashSelectorMode::Greedy]
            }
        };
        let sequence_capacity = pool
            .lock()
            .expect("dflash windowed pool poisoned")
            .config()
            .sequence_capacity();
        let shapes = dflash_graph_precapture_shapes(
            &self.graph_plans(max_n),
            crate::pipeline::cuda_graph::cuda_graph_precapture_batches(
                CudaGraphComponent::DFlash,
                max_n.saturating_add(1),
            ),
            sequence_capacity,
        );
        let Some(max_batch) = shapes.iter().map(|(batch, _)| *batch).max() else {
            return Ok(());
        };
        let temporary = DFlashGraphTemporarySequences::acquire(pool, max_batch)?;
        let started = std::time::Instant::now();
        let mut captured = 0usize;
        let mut state = self
            .cuda_graphs
            .lock()
            .expect("dflash CUDA graph cache poisoned");
        state.capacity = state
            .capacity
            .max(shapes.len().saturating_mul(selector_modes.len()));
        for selector_mode in selector_modes {
            for &(batch_bucket, block) in &shapes {
                let key = DFlashCudaGraphKey {
                    batch_bucket,
                    block,
                    selector_mode,
                };
                if state.entries.iter().any(|entry| {
                    entry.key == key && entry.matches_dependencies(token_embedding, lm_head)
                }) || state.failed.contains(&key)
                {
                    continue;
                }
                if let Some(position) = state.entries.iter().position(|entry| entry.key == key) {
                    release_dflash_cuda_graph(state.entries.remove(position));
                    record_cuda_graph_resident_entries(
                        CudaGraphComponent::DFlash,
                        state.entries.len(),
                    );
                }
                let attention_batch = temporary.attention_batch(batch_bucket, block)?;
                let anchors = vec![0u32; batch_bucket];
                let start_positions = vec![0usize; batch_bucket];
                let sampling_values = (selector_mode == DFlashSelectorMode::Sampling).then(|| {
                    (
                        vec![1.0f32; batch_bucket],
                        vec![0.5f32; batch_bucket * (block - 1)],
                    )
                });
                let sampling = sampling_values
                    .as_ref()
                    .map(|(inverse_temperatures, uniforms)| DFlashSamplingInputs {
                        inverse_temperatures,
                        uniforms,
                    });
                let rows = dflash_graph_host_rows(DFlashGraphHostInput {
                    anchors: &anchors,
                    start_positions: &start_positions,
                    mask_token_id: self.mask_token_id,
                    block,
                    batch_bucket,
                    sampling,
                })?;
                DFlashCudaGraphState::eager(
                    self,
                    key,
                    &rows,
                    &attention_batch,
                    token_embedding,
                    lm_head,
                    batch_bucket,
                )?;
                state.warmed.insert(key);
                let entry = DFlashCudaGraphState::capture(
                    self,
                    key,
                    rows,
                    &attention_batch,
                    token_embedding,
                    lm_head,
                )?;
                state.store(entry);
                captured += 1;
            }
        }
        if captured > 0 {
            tracing::info!(
                "Captured {captured} DFlash CUDA graphs through batch bucket {max_batch} in {:.2?}",
                started.elapsed()
            );
        }
        Ok(())
    }

    pub(crate) fn proposals_cuda_graph(
        &self,
        inputs: &DFlashGraphProposalInputs<'_>,
    ) -> Result<Option<DFlashProposalBatch>> {
        let DFlashGraphProposalInputs {
            seq_ids,
            anchors,
            start_positions,
            n_predict,
            sampling,
            token_embedding,
            lm_head,
        } = *inputs;
        #[cfg(cuda_flash_attn)]
        {
            if !crate::pipeline::cuda_graph::cuda_decode_graphs_enabled()
                || self.windowed_pool.is_none()
                || seq_ids.is_empty()
                || seq_ids.len() != anchors.len()
                || seq_ids.len() != start_positions.len()
                || n_predict == 0
            {
                return Ok(None);
            }
            let block = n_predict + 1;
            if block > self.block_size || !dflash_graph_positions_fit(start_positions, block) {
                return Ok(None);
            }
            let Some(batch_bucket) = crate::pipeline::cuda_graph::cuda_graph_batch_bucket(
                CudaGraphComponent::DFlash,
                block,
                seq_ids.len(),
            ) else {
                return Ok(None);
            };
            let use_selector = self.selector.is_some();
            if use_selector
                && self
                    .selector
                    .as_ref()
                    .expect("selector checked above")
                    .configured_cuda_capability(self.device.is_cuda(), self.dtype)
                    .is_err()
            {
                return Ok(None);
            }
            let selector_mode = match (use_selector, sampling.is_some()) {
                (false, _) => DFlashSelectorMode::Disabled,
                (true, false) => DFlashSelectorMode::Greedy,
                (true, true) => DFlashSelectorMode::Sampling,
            };
            let key = DFlashCudaGraphKey {
                batch_bucket,
                block,
                selector_mode,
            };
            let sampling = sampling.filter(|_| selector_mode == DFlashSelectorMode::Sampling);
            let attention_batch = {
                let pool = self
                    .windowed_pool
                    .as_ref()
                    .expect("windowed pool checked above")
                    .lock()
                    .expect("dflash windowed pool poisoned");
                for (seq_id, start_position) in seq_ids.iter().zip(start_positions) {
                    let state = pool.sequence(*seq_id).ok_or_else(|| {
                        candle_core::Error::msg(format!(
                            "DFlash draft requested for sequence {seq_id} without paged context"
                        ))
                    })?;
                    if state.next_committed_pos != *start_position {
                        candle_core::bail!(
                            "DFlash draft at position {start_position} but paged context ends at {}",
                            state.next_committed_pos
                        );
                    }
                }
                let queries = seq_ids
                    .iter()
                    .map(|seq_id| WindowedKvQuery {
                        seq_id: *seq_id,
                        query_len: block,
                    })
                    .collect::<Vec<_>>();
                pool.scratch_graph_batch(&queries, batch_bucket)?
            };
            return self
                .cuda_graphs
                .lock()
                .expect("dflash CUDA graph cache poisoned")
                .run(DFlashCudaGraphRun {
                    model: self,
                    key,
                    anchors,
                    start_positions,
                    sampling,
                    attention_batch: &attention_batch,
                    token_embedding,
                    lm_head,
                    real_batch: seq_ids.len(),
                });
        }
        #[cfg(not(cuda_flash_attn))]
        {
            let _ = (
                seq_ids,
                anchors,
                start_positions,
                n_predict,
                sampling,
                token_embedding,
                lm_head,
            );
            Ok(None)
        }
    }

    pub fn enable_adaptive(&self, max_n: usize, max_live_sequences: usize) -> bool {
        if !dflash_adaptive_supported(max_n, max_live_sequences) {
            return false;
        }
        *self.adaptive.lock().expect("dflash adaptive poisoned") = Some(AdaptiveState { max_n });
        true
    }

    pub fn plan_n(&self, max_n: usize, batch: usize) -> usize {
        let adaptive = {
            let guard = self.adaptive.lock().expect("dflash adaptive poisoned");
            if let Some(adaptive) = guard.as_ref() {
                debug_assert_eq!(adaptive.max_n, max_n);
            }
            guard.is_some()
        };
        select_dflash_depth(adaptive, max_n, batch, self.live_sequence_count())
    }

    pub fn graph_plans(&self, max_n: usize) -> Vec<super::SpeculativeGraphPlan> {
        let guard = self.adaptive.lock().expect("dflash adaptive poisoned");
        if let Some(adaptive) = guard.as_ref() {
            debug_assert_eq!(adaptive.max_n, max_n);
        }
        dflash_graph_plans(guard.is_some(), max_n)
    }

    fn live_sequence_count(&self) -> usize {
        #[cfg(cuda_flash_attn)]
        if let Some(pool) = &self.windowed_pool {
            return pool
                .lock()
                .expect("dflash windowed pool poisoned")
                .live_sequence_count();
        }
        self.ctx_cache.lock().expect("dflash cache poisoned").len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn prefix_replay(&self) -> SpeculativePrefixReplay {
        dflash_prefix_replay(self.layers.iter().map(|layer| layer.sliding_window))
    }

    pub fn supports_paged_auxiliary_prefix_state(&self) -> bool {
        #[cfg(cuda_flash_attn)]
        {
            self.windowed_pool.as_ref().is_some_and(|pool| {
                pool.lock()
                    .expect("dflash windowed pool poisoned")
                    .config()
                    .checkpoint_capacity()
                    > 0
            })
        }
        #[cfg(not(cuda_flash_attn))]
        {
            false
        }
    }

    pub fn capture_paged_auxiliary_prefix_state(
        &self,
        sequence_id: usize,
        cached_tokens: usize,
    ) -> Result<Option<Arc<dyn PagedAuxiliaryPrefixState>>> {
        #[cfg(cuda_flash_attn)]
        {
            let Some(pool) = &self.windowed_pool else {
                return Ok(None);
            };
            let mut pool = pool.lock().expect("dflash windowed pool poisoned");
            let state = pool.sequence(sequence_id).ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "DFlash sequence {sequence_id} has no context to checkpoint"
                ))
            })?;
            if state.next_committed_pos != cached_tokens {
                candle_core::bail!(
                    "DFlash sequence {sequence_id} is at position {}, expected checkpoint boundary {cached_tokens}",
                    state.next_committed_pos
                );
            }
            if !pool.sequence_query_ready(sequence_id) {
                candle_core::bail!(
                    "DFlash sequence {sequence_id} is not ready at checkpoint boundary {cached_tokens}"
                );
            }
            let checkpoint = pool.snapshot_sequence(sequence_id)?;
            metrics::counter!("mistralrs_speculative_prefix_cache_captures_total").increment(1);
            Ok(Some(Arc::new(DFlashPagedPrefixState { checkpoint })))
        }
        #[cfg(not(cuda_flash_attn))]
        {
            let _ = (sequence_id, cached_tokens);
            Ok(None)
        }
    }

    pub fn restore_paged_auxiliary_prefix_state(
        &self,
        sequence_id: usize,
        cached_tokens: usize,
        state: &dyn PagedAuxiliaryPrefixState,
    ) -> Result<()> {
        #[cfg(cuda_flash_attn)]
        {
            let state = state
                .as_any()
                .downcast_ref::<DFlashPagedPrefixState>()
                .ok_or_else(|| candle_core::Error::msg("invalid DFlash prefix checkpoint type"))?;
            if state.checkpoint.next_committed_pos() != cached_tokens {
                candle_core::bail!(
                    "DFlash prefix checkpoint is at position {}, expected {cached_tokens}",
                    state.checkpoint.next_committed_pos()
                );
            }
            let pool = self.windowed_pool.as_ref().ok_or_else(|| {
                candle_core::Error::msg("DFlash windowed KV is unavailable during prefix restore")
            })?;
            let started = std::time::Instant::now();
            pool.lock()
                .expect("dflash windowed pool poisoned")
                .restore_sequence(sequence_id, &state.checkpoint)?;
            self.activate_seqs(&[sequence_id]);
            metrics::counter!("mistralrs_speculative_prefix_cache_restore_copies_total")
                .increment(1);
            metrics::histogram!("mistralrs_speculative_prefix_cache_restore_seconds")
                .record(started.elapsed().as_secs_f64());
            Ok(())
        }
        #[cfg(not(cuda_flash_attn))]
        {
            let _ = (sequence_id, cached_tokens, state);
            candle_core::bail!("DFlash auxiliary prefix restore requires CUDA FlashAttention")
        }
    }

    pub fn has_selector(&self) -> bool {
        self.selector.is_some()
    }

    pub fn draft_sampling_method(&self) -> MtpDraftSamplingMethod {
        self.draft_sampling_method
    }

    pub fn mask_token_id(&self) -> u32 {
        self.mask_token_id
    }

    pub fn input_embedding_scale(&self) -> f64 {
        self.input_embedding_scale
    }

    /// Absolute position of the next context token a sequence expects, or None if unseen.
    pub fn ctx_next_pos(&self, seq_id: usize) -> Option<usize> {
        #[cfg(cuda_flash_attn)]
        if let Some(pool) = &self.windowed_pool {
            return pool
                .lock()
                .expect("dflash windowed pool poisoned")
                .sequence(seq_id)
                .map(|state| state.next_committed_pos);
        }
        self.ctx_cache
            .lock()
            .expect("dflash cache poisoned")
            .get(&seq_id)
            .map(|c| c.next_pos)
    }

    pub fn contexts_ready_for_draft(&self, seq_ids: &[usize]) -> bool {
        #[cfg(cuda_flash_attn)]
        if let Some(pool) = &self.windowed_pool {
            let pool = pool.lock().expect("dflash windowed pool poisoned");
            return seq_ids
                .iter()
                .all(|seq_id| pool.sequence_query_ready(*seq_id));
        }
        let cache = self.ctx_cache.lock().expect("dflash cache poisoned");
        seq_ids.iter().all(|seq_id| cache.contains_key(seq_id))
    }

    fn release_seq_storage(&self, seq_ids: &[usize]) {
        #[cfg(cuda_flash_attn)]
        if let Some(pool) = &self.windowed_pool {
            let mut pool = pool.lock().expect("dflash windowed pool poisoned");
            for seq_id in seq_ids {
                pool.release(*seq_id);
            }
        }
        let mut ctx_cache = self.ctx_cache.lock().expect("dflash cache poisoned");
        for seq_id in seq_ids {
            ctx_cache.remove(seq_id);
        }
    }

    pub fn mark_seqs_dormant(&self, seq_ids: &[usize]) {
        self.release_seq_storage(seq_ids);
        let mut dormant = self
            .dormant_seqs
            .lock()
            .expect("dflash dormant set poisoned");
        update_dormant_sequences(&mut dormant, seq_ids, DFlashSequenceEviction::Dormant);
    }

    pub fn release_seqs(&self, seq_ids: &[usize]) {
        self.release_seq_storage(seq_ids);
        let mut dormant = self
            .dormant_seqs
            .lock()
            .expect("dflash dormant set poisoned");
        update_dormant_sequences(&mut dormant, seq_ids, DFlashSequenceEviction::Released);
    }

    pub fn activate_seqs(&self, seq_ids: &[usize]) {
        self.dormant_seqs
            .lock()
            .expect("dflash dormant set poisoned")
            .retain(|id| !seq_ids.contains(id));
    }

    pub fn has_dormant_seq(&self, seq_ids: &[usize]) -> bool {
        let dormant = self
            .dormant_seqs
            .lock()
            .expect("dflash dormant set poisoned");
        seq_ids.iter().any(|id| dormant.contains(id))
    }

    pub fn clear(&self) {
        #[cfg(cuda_flash_attn)]
        if let Some(pool) = &self.windowed_pool {
            pool.lock().expect("dflash windowed pool poisoned").clear();
        }
        self.ctx_cache
            .lock()
            .expect("dflash cache poisoned")
            .clear();
        self.dormant_seqs
            .lock()
            .expect("dflash dormant set poisoned")
            .clear();
    }

    /// cos/sin for the contiguous positions `start..start + len`, from the precomputed table.
    fn cos_sin(&self, start: usize, len: usize) -> Result<(Tensor, Tensor)> {
        if start + len <= ROPE_CACHE_LEN {
            return Ok((
                self.rope_table.0.narrow(0, start, len)?,
                self.rope_table.1.narrow(0, start, len)?,
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let pos: Vec<f32> = (start..start + len).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, (len, 1), &self.device)?;
        let freqs = pos.broadcast_matmul(&self.inv_freq.reshape((1, ()))?)?;
        finalize_dflash_rope(
            (freqs.cos()?, freqs.sin()?),
            self.dtype,
            self.rope_attention_factor,
        )
    }

    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        candle_nn::rotary_emb::rope(&x.contiguous()?, cos, sin)
    }

    /// Projects tap features and appends context keys/values for every entry at once: the fc and
    /// per-layer k/v projections read their weights once over all sequences' rows packed together.
    pub fn append_ctx_batch(
        &self,
        taps: &[Tensor],
        flat_row_indices: Vec<u32>,
        entries: &[CtxAppend],
    ) -> Result<()> {
        let total = entries.iter().map(|entry| entry.rows).sum::<usize>();
        if total == 0 {
            return Ok(());
        }
        let prepared = self.prepare_ctx_batch(taps, flat_row_indices, entries)?;
        let row_indices = (0..total)
            .map(|row| u32::try_from(row).map_err(candle_core::Error::wrap))
            .collect::<Result<Vec<_>>>()?;
        self.commit_prepared_ctx_batch(&prepared, row_indices, entries)
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn prepare_ctx_batch(
        &self,
        taps: &[Tensor],
        flat_row_indices: Vec<u32>,
        entries: &[CtxAppend],
    ) -> Result<DFlashPreparedContext> {
        let rows = entries.iter().map(|entry| entry.rows).collect::<Vec<_>>();
        let total: usize = rows.iter().sum();
        if total == 0 {
            candle_core::bail!("DFlash context preparation requires at least one row");
        }
        if taps.len() != self.target_layer_ids.len() {
            candle_core::bail!("DFlash context append tap count changed");
        }
        if flat_row_indices.len() != total {
            candle_core::bail!("DFlash context append row count changed");
        }
        let packed = gather_ctx_taps(taps, flat_row_indices, &self.device)?;
        let ctx_hidden = self
            .hidden_norm
            .forward(&self.fc.forward(&packed.to_dtype(self.dtype)?)?)?;
        let all_kv = self
            .context_kv_proj
            .forward(&ctx_hidden)?
            .squeeze(0)?
            .reshape((
                total,
                self.layers.len(),
                2,
                self.num_kv_heads,
                self.head_dim,
            ))?
            .permute((2, 1, 3, 0, 4))?
            .contiguous()?;
        let raw_k = all_kv.i(0)?;
        let v_all = all_kv.i(1)?;
        #[cfg(feature = "cuda")]
        let fused_k = if entries.iter().all(|entry| {
            entry
                .start_pos
                .checked_add(entry.rows)
                .is_some_and(|end| end <= ROPE_CACHE_LEN)
        }) {
            if let Some(norm_weights) = &self.context_k_norm_weights {
                let mut positions = Vec::with_capacity(total);
                for entry in entries {
                    for position in entry.start_pos..entry.start_pos + entry.rows {
                        positions.push(u32::try_from(position).map_err(candle_core::Error::wrap)?);
                    }
                }
                let positions = Tensor::from_vec(positions, (total,), &self.device)?;
                crate::cuda::dflash_context::context_keys(
                    &raw_k,
                    norm_weights,
                    &self.rope_table.0,
                    &self.rope_table.1,
                    &positions,
                    self.layers[0].k_norm.eps() as f32,
                )?
            } else {
                None
            }
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let fused_k: Option<Tensor> = None;
        let k_all = if let Some(k) = fused_k {
            k
        } else {
            let mut coss = Vec::with_capacity(entries.len());
            let mut sins = Vec::with_capacity(entries.len());
            for (entry, rows) in entries.iter().zip(&rows) {
                if *rows == 0 {
                    continue;
                }
                let (cos, sin) = self.cos_sin(entry.start_pos, *rows)?;
                coss.push(cos);
                sins.push(sin);
            }
            let (cos, sin) = (Tensor::cat(&coss, 0)?, Tensor::cat(&sins, 0)?);
            let mut ks = Vec::with_capacity(self.layers.len());
            for (i, layer) in self.layers.iter().enumerate() {
                let k = layer.k_norm.forward(&raw_k.narrow(0, i, 1)?)?;
                let k = self.rope(&k, &cos, &sin)?;
                ks.push(k.squeeze(0)?);
            }
            Tensor::stack(&ks, 0)?
        };

        Ok(DFlashPreparedContext {
            k: k_all,
            v: v_all,
            rows: total,
        })
    }

    pub fn commit_prepared_ctx_batch(
        &self,
        prepared: &DFlashPreparedContext,
        row_indices: Vec<u32>,
        entries: &[CtxAppend],
    ) -> Result<()> {
        let rows = entries.iter().map(|entry| entry.rows).collect::<Vec<_>>();
        let total = rows.iter().sum::<usize>();
        if total == 0 {
            return Ok(());
        }
        if row_indices.len() != total
            || row_indices
                .iter()
                .any(|row| usize::try_from(*row).map_or(true, |row| row >= prepared.rows))
        {
            candle_core::bail!("DFlash prepared context row selection is invalid");
        }
        let (k_all, v_all) = select_ctx_kv_rows(&prepared.k, &prepared.v, &row_indices)?;
        self.append_projected_ctx_batch(entries, &rows, &k_all, &v_all)
    }

    fn append_projected_ctx_batch(
        &self,
        entries: &[CtxAppend],
        rows: &[usize],
        k_all: &Tensor,
        v_all: &Tensor,
    ) -> Result<()> {
        #[cfg(cuda_flash_attn)]
        if self.windowed_pool.is_some() {
            return self.append_ctx_windowed(entries, rows, k_all, v_all);
        }

        // Sliding layers never look further back than their window; trim by the widest window plus
        // a block so every layer keeps what it can see (full-attention layers keep everything).
        let keep = self
            .layers
            .iter()
            .map(|l| match l.sliding_window {
                Some(w) => w + self.block_size,
                None => usize::MAX,
            })
            .max()
            .unwrap_or(usize::MAX);

        let mut cache = self.ctx_cache.lock().expect("dflash cache poisoned");
        let mut offset = 0;
        for (e, r) in entries.iter().zip(rows.iter()) {
            if *r == 0 {
                continue;
            }
            let k = k_all.narrow(2, offset, *r)?;
            let v = v_all.narrow(2, offset, *r)?;
            offset += r;
            let entry = match cache.get_mut(&e.seq_id) {
                Some(entry) => {
                    if entry.next_pos != e.start_pos {
                        candle_core::bail!(
                            "DFlash context append at position {} but cache expects {}",
                            e.start_pos,
                            entry.next_pos
                        );
                    }
                    entry.k = Tensor::cat(&[&entry.k, &k], 2)?;
                    entry.v = Tensor::cat(&[&entry.v, &v], 2)?;
                    entry.next_pos = e.start_pos + r;
                    entry
                }
                None => {
                    cache.insert(
                        e.seq_id,
                        SeqCtxCache {
                            k: k.contiguous()?,
                            v: v.contiguous()?,
                            start_pos: e.start_pos,
                            next_pos: e.start_pos + r,
                        },
                    );
                    cache.get_mut(&e.seq_id).expect("just inserted")
                }
            };
            if keep != usize::MAX {
                let len = entry.next_pos - entry.start_pos;
                if len > keep {
                    let drop = len - keep;
                    entry.k = entry.k.narrow(2, drop, keep)?.contiguous()?;
                    entry.v = entry.v.narrow(2, drop, keep)?.contiguous()?;
                    entry.start_pos += drop;
                }
            }
        }
        Ok(())
    }

    #[cfg(cuda_flash_attn)]
    fn append_ctx_windowed(
        &self,
        entries: &[CtxAppend],
        rows: &[usize],
        k_all: &Tensor,
        v_all: &Tensor,
    ) -> Result<()> {
        let mut pool = self
            .windowed_pool
            .as_ref()
            .expect("windowed pool checked by caller")
            .lock()
            .expect("dflash windowed pool poisoned");
        let mut writes = Vec::with_capacity(entries.len());
        for (entry, row_count) in entries.iter().zip(rows) {
            pool.acquire_at(entry.seq_id, entry.start_pos)?;
            writes.push(pool.plan_context_write(entry.seq_id, *row_count)?);
        }

        let retained_total = writes
            .iter()
            .map(|write| write.retained_input_range().len())
            .sum();
        if retained_total == 0 {
            for write in &writes {
                pool.commit_context(write)?;
            }
            return Ok(());
        }
        let mut retained_indices = Vec::with_capacity(retained_total);
        let mut slot_mapping = Vec::with_capacity(retained_total);
        let mut offset = 0;
        for (write, row_count) in writes.iter().zip(rows) {
            for row in write.retained_input_range() {
                retained_indices
                    .push(u32::try_from(offset + row).map_err(candle_core::Error::wrap)?);
            }
            slot_mapping.extend_from_slice(write.slot_mapping());
            offset += *row_count;
        }
        let slot_mapping = Tensor::from_vec(slot_mapping, (retained_total,), &self.device)?;
        let (k_retained, v_retained) = select_ctx_kv_rows(k_all, v_all, &retained_indices)?;
        let k_packed = k_retained.permute((0, 2, 1, 3))?.contiguous()?;
        let v_packed = v_retained.permute((0, 2, 1, 3))?.contiguous()?;
        for layer_idx in 0..self.layers.len() {
            let (key_cache, value_cache) = pool.layer_cache(layer_idx)?;
            mistralrs_paged_attn::reshape_and_cache_flashinfer(
                &k_packed.narrow(0, layer_idx, 1)?,
                &v_packed.narrow(0, layer_idx, 1)?,
                &key_cache,
                &value_cache,
                &slot_mapping,
                mistralrs_paged_attn::DEFAULT_FP8_KV_CACHE_SCALES,
            )?;
        }
        for write in &writes {
            pool.commit_context(write)?;
        }
        Ok(())
    }

    /// One block draft for every sequence at once: noise `[anchor, mask * n_drafts]` rows at each
    /// sequence's `start_positions[i]..`, attending to that sequence's accumulated context left-
    /// padded to the batch maximum (pad columns are masked out). Every weight is read once for the
    /// whole batch. Returns the normed hidden states `[batch, n_drafts, hidden]`.
    pub fn draft_hidden_batch(
        &self,
        seq_ids: &[usize],
        noise_embedding: &Tensor,
        start_positions: &[usize],
    ) -> Result<Tensor> {
        let (b, block, _) = noise_embedding.dims3()?;
        #[cfg(cuda_flash_attn)]
        if self.windowed_pool.is_some() {
            return self.draft_hidden_windowed(seq_ids, noise_embedding, start_positions, b, block);
        }
        let cache = self.ctx_cache.lock().expect("dflash cache poisoned");
        let mut lens = Vec::with_capacity(b);
        for (seq_id, start_pos) in seq_ids.iter().zip(start_positions.iter()) {
            let entry = cache.get(seq_id).ok_or_else(|| {
                candle_core::Error::msg(
                    "DFlash draft requested for a sequence with no context cache",
                )
            })?;
            if entry.next_pos != *start_pos {
                candle_core::bail!(
                    "DFlash draft at position {start_pos} but context ends at {}",
                    entry.next_pos
                );
            }
            lens.push(entry.next_pos - entry.start_pos);
        }
        let max_ctx = *lens.iter().max().expect("non-empty batch");
        let mut padded_k = Vec::with_capacity(b);
        let mut padded_v = Vec::with_capacity(b);
        for (seq_id, len) in seq_ids.iter().zip(lens.iter()) {
            let entry = cache.get(seq_id).expect("validated above");
            if *len == max_ctx {
                padded_k.push(entry.k.clone());
                padded_v.push(entry.v.clone());
            } else {
                let pad = Tensor::zeros(
                    (
                        self.layers.len(),
                        self.num_kv_heads,
                        max_ctx - len,
                        self.head_dim,
                    ),
                    self.dtype,
                    &self.device,
                )?;
                padded_k.push(Tensor::cat(&[&pad, &entry.k], 2)?);
                padded_v.push(Tensor::cat(&[&pad, &entry.v], 2)?);
            }
        }
        drop(cache);
        // [layers, batch, kv_heads, max_ctx, head_dim]
        let ctx_k = Tensor::stack(&padded_k, 1)?;
        let ctx_v = Tensor::stack(&padded_v, 1)?;

        // One mask per attention kind, shared by every layer of that kind
        let mut kind_masks: HashMap<(bool, Option<usize>), Tensor> = HashMap::new();
        for layer in &self.layers {
            let kind = (layer.is_causal, layer.sliding_window);
            if kind_masks.contains_key(&kind) {
                continue;
            }
            let mut rows = Vec::with_capacity(b);
            for len in &lens {
                let geometry = self.geometry_mask(kind.0, kind.1, *len, block)?;
                if *len == max_ctx {
                    rows.push(geometry);
                } else {
                    let pad = Tensor::full(
                        f32::NEG_INFINITY,
                        (1, 1, block, max_ctx - len),
                        &self.device,
                    )?
                    .to_dtype(self.dtype)?;
                    rows.push(Tensor::cat(&[pad, geometry], 3)?);
                }
            }
            kind_masks.insert(kind, Tensor::cat(&rows, 0)?);
        }

        let (q_cos, q_sin) = {
            let mut coss = Vec::with_capacity(b);
            let mut sins = Vec::with_capacity(b);
            for start_pos in start_positions {
                let (c, s) = self.cos_sin(*start_pos, block)?;
                coss.push(c);
                sins.push(s);
            }
            // [batch, block, head_dim/2]
            (Tensor::stack(&coss, 0)?, Tensor::stack(&sins, 0)?)
        };
        #[allow(clippy::cast_precision_loss)]
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let groups = self.num_heads / self.num_kv_heads;
        let hs = self.run_draft_layers(
            noise_embedding,
            &q_cos,
            &q_sin,
            DraftAttentionLayout::HeadsFirst,
            |i, layer, q, k_noise, v_noise| {
                let ctx_k = ctx_k.narrow(0, i, 1)?.squeeze(0)?;
                let ctx_v = ctx_v.narrow(0, i, 1)?.squeeze(0)?;
                let k = Tensor::cat(&[&ctx_k, k_noise], 2)?;
                let v = Tensor::cat(&[&ctx_v, v_noise], 2)?;
                let mask = &kind_masks[&(layer.is_causal, layer.sliding_window)];
                let k = repeat_kv(&k, groups)?;
                let v = repeat_kv(&v, groups)?;
                let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
                let att = att.broadcast_add(mask)?;
                let att = candle_nn::ops::softmax_last_dim(&att)?;
                let out = att.matmul(&v)?;
                out.transpose(1, 2)?
                    .reshape((b, block, self.num_heads * self.head_dim))
            },
        )?;

        let hs = self.norm.forward(&hs)?;
        hs.narrow(1, 1, block - 1)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn run_draft_layers<F>(
        &self,
        noise_embedding: &Tensor,
        q_cos: &Tensor,
        q_sin: &Tensor,
        attention_layout: DraftAttentionLayout,
        mut attention: F,
    ) -> Result<Tensor>
    where
        F: FnMut(usize, &DFlashLayer, &Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    {
        let (batch, block, _) = noise_embedding.dims3()?;
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;
        let mut hs = noise_embedding.to_dtype(self.dtype)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let residual = &hs;
            let mut x = layer.input_layernorm.forward(&hs)?;
            let attn_kernel = match &layer.attention_conv {
                Some(conv) => {
                    let (pre, kernel) = conv.prepare(&x)?;
                    x = pre;
                    Some(kernel)
                }
                None => None,
            };
            let qkv = layer.qkv_proj.forward(&x)?;
            let q_input = qkv
                .narrow(D::Minus1, 0, q_size)?
                .reshape((batch, block, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k_input = qkv
                .narrow(D::Minus1, q_size, kv_size)?
                .reshape((batch, block, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            #[cfg(feature = "cuda")]
            let fused_qk = {
                let cos = q_cos.reshape((batch * block, ()))?;
                let sin = q_sin.reshape((batch * block, ()))?;
                let output_layout = match attention_layout {
                    DraftAttentionLayout::HeadsFirst => crate::ops::QkRopeOutputLayout::HeadsFirst,
                    #[cfg(cuda_flash_attn)]
                    DraftAttentionLayout::TokensFirst => {
                        crate::ops::QkRopeOutputLayout::TokensFirst
                    }
                };
                crate::ops::try_cuda_qk_rms_norm_rope(
                    &q_input,
                    Some(&k_input),
                    layer.q_norm.weight(),
                    Some(layer.k_norm.weight()),
                    layer.q_norm.eps() as f32,
                    layer.k_norm.eps() as f32,
                    &cos,
                    &sin,
                    true,
                    output_layout,
                )?
            };
            #[cfg(not(feature = "cuda"))]
            let fused_qk: Option<(Tensor, Option<Tensor>)> = None;
            let (q, k_noise) = match fused_qk {
                Some((q, Some(k))) => (q, k),
                Some((_, None)) => unreachable!("DFlash fused Q/K omitted K output"),
                None => {
                    let q = candle_nn::rotary_emb::rope(
                        &layer.q_norm.forward(&q_input)?,
                        q_cos,
                        q_sin,
                    )?;
                    let k = candle_nn::rotary_emb::rope(
                        &layer.k_norm.forward(&k_input)?,
                        q_cos,
                        q_sin,
                    )?;
                    match attention_layout {
                        DraftAttentionLayout::HeadsFirst => (q, k),
                        #[cfg(all(
                            feature = "cuda",
                            feature = "flash-attn",
                            target_family = "unix"
                        ))]
                        DraftAttentionLayout::TokensFirst => (
                            q.transpose(1, 2)?.contiguous()?,
                            k.transpose(1, 2)?.contiguous()?,
                        ),
                    }
                }
            };
            let v_tokens = qkv.narrow(D::Minus1, q_size + kv_size, kv_size)?.reshape((
                batch,
                block,
                self.num_kv_heads,
                self.head_dim,
            ))?;
            let v_noise = match attention_layout {
                DraftAttentionLayout::HeadsFirst => v_tokens.transpose(1, 2)?.contiguous()?,
                #[cfg(cuda_flash_attn)]
                DraftAttentionLayout::TokensFirst => v_tokens,
            };
            let out = attention(layer_idx, layer, &q, &k_noise, &v_noise)?;
            let mut out = layer.o_proj.forward(&out)?;
            if let (Some(conv), Some(kernel)) = (&layer.attention_conv, attn_kernel) {
                out = conv.finish(&out, &kernel)?;
            }
            hs = (residual + out)?;

            let residual = &hs;
            let mut x = layer.post_attention_layernorm.forward(&hs)?;
            let mlp_kernel = match &layer.mlp_conv {
                Some(conv) => {
                    let (pre, kernel) = conv.prepare(&x)?;
                    x = pre;
                    Some(kernel)
                }
                None => None,
            };
            let gate_up = layer.gate_up_proj.forward(&x)?;
            let mut out = if let Some(output) = crate::ops::try_fused_split_glu_quantized_forward(
                &gate_up,
                self.intermediate_size,
                crate::layers::Activation::Silu,
                &*layer.down_proj,
            )? {
                output
            } else {
                let inter = crate::ops::split_mul_and_act(
                    &gate_up,
                    self.intermediate_size,
                    crate::layers::Activation::Silu,
                )?;
                layer.down_proj.forward(&inter)?
            };
            if let (Some(conv), Some(kernel)) = (&layer.mlp_conv, mlp_kernel) {
                out = conv.finish(&out, &kernel)?;
            }
            hs = (residual + out)?;
        }
        Ok(hs)
    }

    #[cfg(cuda_flash_attn)]
    fn draft_hidden_windowed(
        &self,
        seq_ids: &[usize],
        noise_embedding: &Tensor,
        start_positions: &[usize],
        batch: usize,
        block: usize,
    ) -> Result<Tensor> {
        let attention_batch = {
            let pool = self
                .windowed_pool
                .as_ref()
                .expect("windowed pool checked by caller")
                .lock()
                .expect("dflash windowed pool poisoned");
            for (seq_id, start_pos) in seq_ids.iter().zip(start_positions) {
                let state = pool.sequence(*seq_id).ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "DFlash draft requested for sequence {seq_id} without paged context"
                    ))
                })?;
                if state.next_committed_pos != *start_pos {
                    candle_core::bail!(
                        "DFlash draft at position {start_pos} but paged context ends at {}",
                        state.next_committed_pos
                    );
                }
            }
            let queries = seq_ids
                .iter()
                .map(|seq_id| WindowedKvQuery {
                    seq_id: *seq_id,
                    query_len: block,
                })
                .collect::<Vec<_>>();
            pool.scratch_batch(&queries)?
        };
        let metadata = attention_batch.to_tensors(&self.device)?;

        let (q_cos, q_sin) = {
            let mut coss = Vec::with_capacity(batch);
            let mut sins = Vec::with_capacity(batch);
            for start_pos in start_positions {
                let (cos, sin) = self.cos_sin(*start_pos, block)?;
                coss.push(cos);
                sins.push(sin);
            }
            (Tensor::stack(&coss, 0)?, Tensor::stack(&sins, 0)?)
        };
        self.draft_hidden_windowed_tensors(DFlashWindowedForward {
            noise_embedding,
            q_cos: &q_cos,
            q_sin: &q_sin,
            batch,
            block,
            attention_batch: &attention_batch,
            metadata: &metadata,
        })
    }

    #[cfg(cuda_flash_attn)]
    fn draft_hidden_windowed_tensors(&self, forward: DFlashWindowedForward<'_>) -> Result<Tensor> {
        let DFlashWindowedForward {
            noise_embedding,
            q_cos,
            q_sin,
            batch,
            block,
            attention_batch,
            metadata,
        } = forward;
        let pool = self
            .windowed_pool
            .as_ref()
            .expect("windowed pool checked by caller")
            .lock()
            .expect("dflash windowed pool poisoned");
        #[allow(clippy::cast_precision_loss)]
        let scale = 1f32 / (self.head_dim as f32).sqrt();
        let hs = self.run_draft_layers(
            noise_embedding,
            q_cos,
            q_sin,
            DraftAttentionLayout::TokensFirst,
            |layer_idx, layer, q, k_noise, v_noise| {
                let (key_cache, value_cache) = pool.layer_cache(layer_idx)?;
                mistralrs_paged_attn::reshape_and_cache_flashinfer(
                    k_noise,
                    v_noise,
                    &key_cache,
                    &value_cache,
                    &metadata.slot_mapping,
                    mistralrs_paged_attn::DEFAULT_FP8_KV_CACHE_SCALES,
                )?;
                let (key_paged, value_paged) = pool.paged_attention_layer_cache(layer_idx)?;
                let window = layer
                    .sliding_window
                    .expect("windowed pool requires finite windows");
                let window_right = if layer.is_causal {
                    Some(0)
                } else {
                    Some(window - 1)
                };
                let q = q.reshape((batch * block, self.num_heads, self.head_dim))?;
                let out = mistralrs_flash_attn::flash_attn_varlen_paged_windowed(
                    &q,
                    &key_paged,
                    &value_paged,
                    &metadata.cumulative_query_lens,
                    &metadata.cumulative_kv_lens,
                    &metadata.block_tables,
                    None,
                    attention_batch.max_query_len(),
                    attention_batch.max_kv_len(),
                    scale,
                    Some(window - 1),
                    window_right,
                    pool.config().page_size(),
                    None,
                )?;
                out.reshape((batch, block, self.num_heads * self.head_dim))
            },
        )?;
        let hs = self.norm.forward(&hs)?;
        hs.narrow(1, 1, block - 1)
    }

    #[cfg(cuda_flash_attn)]
    fn cuda_graph_output(
        &self,
        key: DFlashCudaGraphKey,
        buffers: &DFlashCudaGraphBuffers,
        attention_batch: &WindowedKvBatch,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
    ) -> Result<DFlashCudaGraphOutput> {
        let mut noise = token_embedding
            .embedding_forward(&buffers.token_ids.as_detached_tensor(), self.dtype.into())?;
        if (self.input_embedding_scale - 1.0).abs() > f64::EPSILON {
            noise = (noise * self.input_embedding_scale)?;
        }
        let rope_indices = buffers.rope_indices.as_detached_tensor();
        let (q_cos, q_sin) = dflash_rope_from_positions(
            &rope_indices,
            &self.inv_freq,
            self.dtype,
            self.rope_attention_factor,
        )?;
        let q_cos = q_cos.reshape((key.batch_bucket, key.block, self.head_dim / 2))?;
        let q_sin = q_sin.reshape((key.batch_bucket, key.block, self.head_dim / 2))?;
        let metadata = buffers.metadata();
        let hidden = self.draft_hidden_windowed_tensors(DFlashWindowedForward {
            noise_embedding: &noise,
            q_cos: &q_cos,
            q_sin: &q_sin,
            batch: key.batch_bucket,
            block: key.block,
            attention_batch,
            metadata: &metadata,
        })?;
        let mut logits = lm_head.forward(&hidden)?;
        if (self.output_multiplier - 1.0).abs() > f64::EPSILON {
            logits = (logits * self.output_multiplier)?;
        }
        let anchors = buffers.anchors.as_detached_tensor();
        match key.selector_mode {
            DFlashSelectorMode::Disabled => Ok(DFlashCudaGraphOutput {
                tokens: logits.argmax(D::Minus1)?.contiguous()?,
                candidate_ids: None,
                candidate_probs: None,
            }),
            DFlashSelectorMode::Greedy => Ok(DFlashCudaGraphOutput {
                tokens: self
                    .selector
                    .as_ref()
                    .expect("selector graph key requires a selector")
                    .select_greedy_cuda(&hidden, &logits, &anchors)?
                    .contiguous()?,
                candidate_ids: None,
                candidate_probs: None,
            }),
            DFlashSelectorMode::Sampling => {
                let selected = self
                    .selector
                    .as_ref()
                    .expect("selector graph key requires a selector")
                    .select_sample_cuda(
                        &hidden,
                        &logits,
                        &anchors,
                        &buffers
                            .selector_inverse_temperatures
                            .as_ref()
                            .expect("sampling graph requires inverse temperatures")
                            .as_detached_tensor(),
                        &buffers
                            .selector_uniforms
                            .as_ref()
                            .expect("sampling graph requires uniforms")
                            .as_detached_tensor(),
                    )?;
                Ok(DFlashCudaGraphOutput {
                    tokens: selected.tokens.contiguous()?,
                    candidate_ids: Some(selected.candidate_ids.contiguous()?),
                    candidate_probs: Some(selected.candidate_probs.contiguous()?),
                })
            }
        }
    }

    pub(crate) fn finish_proposals(
        &self,
        hidden: &Tensor,
        anchors: &[u32],
        sampling: Option<DFlashSamplingInputs<'_>>,
        lm_head: &Arc<dyn QuantMethod>,
    ) -> Result<DFlashProposalBatch> {
        let (batch, positions, _) = hidden.dims3()?;
        if anchors.len() != batch {
            candle_core::bail!("DFlash anchors do not match draft rows");
        }
        if let Some(sampling) = sampling {
            if sampling.inverse_temperatures.len() != batch
                || sampling.uniforms.len() != batch * positions
            {
                candle_core::bail!("DFlash selector sampling inputs do not match draft rows");
            }
        }
        let mut logits = lm_head.forward(hidden)?;
        if (self.output_multiplier - 1.0).abs() > f64::EPSILON {
            logits = (logits * self.output_multiplier)?;
        }
        if let Some(sampling) = sampling {
            let selector = self.selector.as_ref().ok_or_else(|| {
                candle_core::Error::msg(
                    "probabilistic DFlash drafting requires a DFlash2 candidate selector",
                )
            })?;
            #[cfg(feature = "cuda")]
            {
                let vocab = logits.dim(D::Minus1)?;
                selector
                    .cuda_capability(logits.device().is_cuda(), logits.dtype(), vocab)
                    .map_err(|reason| {
                        candle_core::Error::msg(format!(
                            "probabilistic DFlash drafting is unavailable: {reason}"
                        ))
                    })?;
                let anchors = Tensor::from_vec(anchors.to_vec(), (batch,), logits.device())?;
                let inverse_temperatures = Tensor::from_vec(
                    sampling.inverse_temperatures.to_vec(),
                    (batch,),
                    logits.device(),
                )?;
                let uniforms = Tensor::from_vec(
                    sampling.uniforms.to_vec(),
                    (batch, positions),
                    logits.device(),
                )?;
                let selected = selector.select_sample_cuda(
                    hidden,
                    &logits,
                    &anchors,
                    &inverse_temperatures,
                    &uniforms,
                )?;
                return Ok(DFlashProposalBatch::DeviceSparse {
                    tokens: selected.tokens,
                    candidate_ids: selected.candidate_ids,
                    candidate_probs: selected.candidate_probs,
                });
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (selector, sampling);
                candle_core::bail!(
                    "probabilistic DFlash drafting is unavailable: this build has no CUDA candidate selector support"
                );
            }
        }

        let tokens_per_seq = match &self.selector {
            Some(selector) => selector.select_greedy_batch(hidden, &logits, anchors)?,
            None => logits.argmax(D::Minus1)?.to_vec2::<u32>()?,
        };
        Ok(DFlashProposalBatch::Tokens(tokens_per_seq))
    }

    fn geometry_mask(
        &self,
        is_causal: bool,
        sliding_window: Option<usize>,
        ctx_len: usize,
        block: usize,
    ) -> Result<Tensor> {
        // Visibility depends only on relative geometry (the context always ends right before the
        // block), so one cached mask per (layer kind, ctx_len, block) covers every step.
        let key = (is_causal, sliding_window, ctx_len, block);
        if let Some(mask) = self
            .mask_cache
            .lock()
            .expect("dflash mask cache poisoned")
            .get(&key)
        {
            return Ok(mask.clone());
        }
        let total = ctx_len + block;
        let mut mask = vec![0f32; block * total];
        for qi in 0..block {
            let qp = ctx_len + qi;
            for kj in 0..total {
                let kp = if kj < ctx_len {
                    kj
                } else {
                    ctx_len + (kj - ctx_len)
                };
                let mut visible = true;
                if is_causal {
                    visible &= kp <= qp;
                }
                if let Some(w) = sliding_window {
                    visible &= qp.saturating_sub(kp) < w;
                    if !is_causal {
                        visible &= kp.saturating_sub(qp) < w;
                    }
                }
                if !visible {
                    mask[qi * total + kj] = f32::NEG_INFINITY;
                }
            }
        }
        let mask =
            Tensor::from_vec(mask, (1, 1, block, total), &self.device)?.to_dtype(self.dtype)?;
        let mut cache = self.mask_cache.lock().expect("dflash mask cache poisoned");
        if cache.len() >= MASK_CACHE_CAP {
            cache.clear();
        }
        cache.insert(key, mask.clone());
        Ok(mask)
    }
}

fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, kv_heads, len, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv_heads, groups, len, hd))?
        .reshape((b, kv_heads * groups, len, hd))
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests;
