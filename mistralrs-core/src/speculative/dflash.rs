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

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
use candle_core::{cuda_backend::cudarc::driver::sys, Var};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use mistralrs_quant::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, ShardedVarBuilder, UnquantLinear,
};
use serde::Deserialize;

use crate::layers::RmsNorm;
use crate::speculative::MtpConfig;
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
use crate::paged_attention::windowed_pool::{
    WindowedKvBatch, WindowedKvBatchTensors, WindowedKvPool, WindowedKvPoolConfig, WindowedKvQuery,
};

const DEFAULT_BLOCK_SIZE: usize = 16;
pub const DEFAULT_MAX_DRAFTS: usize = 7;
// Rotary table length precomputed at load; positions past it fall back to on-the-fly computation.
const ROPE_TABLE_LEN: usize = 65536;
const MASK_CACHE_CAP: usize = 64;
#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
const DFLASH_CUDA_GRAPH_CACHE_CAPACITY: usize = 8;
const ADAPT_FULL_DEPTH_MAX_BATCH: usize = 2;
const ADAPT_MIN_DEPTH: usize = 1;
const DFLASH_ADAPTIVE_ENV: &str = "MISTRALRS_DFLASH_ADAPTIVE";

fn select_dflash_depth(adaptive: bool, max_n: usize, batch: usize) -> usize {
    if max_n == 0 || !adaptive || batch <= ADAPT_FULL_DEPTH_MAX_BATCH {
        max_n
    } else {
        ADAPT_MIN_DEPTH.min(max_n)
    }
}

fn dflash_adaptive_env_value(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true")
}

pub(crate) fn dflash_adaptive_requested() -> bool {
    std::env::var(DFLASH_ADAPTIVE_ENV)
        .ok()
        .as_deref()
        .is_some_and(dflash_adaptive_env_value)
}

fn dflash_graph_plans(adaptive: bool, max_n: usize) -> Vec<super::SpeculativeGraphPlan> {
    if max_n == 0 {
        return Vec::new();
    }
    if !adaptive || max_n == ADAPT_MIN_DEPTH {
        return vec![super::SpeculativeGraphPlan::new(max_n, None)];
    }
    vec![
        super::SpeculativeGraphPlan::new(max_n, Some(ADAPT_FULL_DEPTH_MAX_BATCH)),
        super::SpeculativeGraphPlan::new(ADAPT_MIN_DEPTH, None),
    ]
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct DFlashSpecificConfig {
    pub target_layer_ids: Option<Vec<usize>>,
    pub block_size: Option<usize>,
    pub mask_token_id: Option<u32>,
    pub input_embedding_scale: Option<f64>,
    pub output_multiplier: Option<f64>,
    pub selector_rank: Option<usize>,
    pub selector_top_k: Option<usize>,
    pub conv_kernel_size: Option<usize>,
    pub conv_group_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DFlashConfig {
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub rope_theta: Option<f64>,
    pub rope_parameters: Option<serde_json::Value>,
    pub num_target_layers: Option<usize>,
    pub sliding_window: Option<usize>,
    pub layer_types: Option<Vec<String>>,
    pub is_causal: Option<bool>,
    // v1 checkpoints keep some fields at the top level instead of inside `dflash_config`
    pub block_size: Option<usize>,
    pub mask_token_id: Option<u32>,
    #[serde(default)]
    pub dflash_config: DFlashSpecificConfig,
}

impl DFlashConfig {
    pub fn is_dflash(&self) -> bool {
        self.architectures
            .iter()
            .any(|a| a.contains("DFlash") || a.contains("Dflash"))
    }

    fn is_v2(&self) -> bool {
        self.architectures.iter().any(|a| a.contains("DFlash2"))
    }

    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    fn rope_theta(&self) -> f64 {
        self.rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|p| p.get("rope_theta"))
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1e7)
    }

    pub fn block_size(&self) -> usize {
        self.dflash_config
            .block_size
            .or(self.block_size)
            .unwrap_or(DEFAULT_BLOCK_SIZE)
    }

    fn mask_token_id(&self) -> Result<u32> {
        self.dflash_config
            .mask_token_id
            .or(self.mask_token_id)
            .ok_or_else(|| candle_core::Error::msg("DFlash config has no mask_token_id"))
    }

    pub fn target_layer_ids(&self) -> Result<Vec<usize>> {
        if let Some(ids) = &self.dflash_config.target_layer_ids {
            return Ok(ids.clone());
        }
        let Some(num_target_layers) = self.num_target_layers else {
            candle_core::bail!("DFlash config has neither target_layer_ids nor num_target_layers");
        };
        let n = self.num_hidden_layers;
        if n == 1 {
            return Ok(vec![num_target_layers / 2]);
        }
        // Mirrors the reference build_target_layer_ids: evenly spaced over [1, num_target_layers - 3]
        let (start, end) = (1, num_target_layers - 3);
        Ok((0..n)
            .map(|i| start + (i * (end - start) + (n - 1) / 2) / (n - 1))
            .collect())
    }

    /// Per-layer (is_causal, sliding_window), following the reference: a layer is causal exactly
    /// when its type is `sliding_attention`, unless the config pins `is_causal` explicitly.
    fn layer_attention(&self, layer_idx: usize) -> (bool, Option<usize>) {
        let layer_type = self
            .layer_types
            .as_ref()
            .and_then(|t| t.get(layer_idx))
            .map(String::as_str)
            .unwrap_or("full_attention");
        let sliding = layer_type == "sliding_attention";
        let is_causal = self.is_causal.unwrap_or(sliding);
        (is_causal, sliding.then_some(self.sliding_window).flatten())
    }
}

struct DynamicConv {
    // [2, kernel_size, hidden]
    base_kernel: Tensor,
    kernel_projection: Tensor,
    kernel_size: usize,
    group_size: usize,
}

impl DynamicConv {
    fn load(vb: ShardedVarBuilder, cfg: &DFlashConfig) -> Result<Self> {
        let kernel_size = cfg
            .dflash_config
            .conv_kernel_size
            .ok_or_else(|| candle_core::Error::msg("DFlash2 config has no conv_kernel_size"))?;
        let group_size = cfg
            .dflash_config
            .conv_group_size
            .ok_or_else(|| candle_core::Error::msg("DFlash2 config has no conv_group_size"))?;
        let groups = cfg.hidden_size / group_size;
        Ok(Self {
            base_kernel: vb.get((2, kernel_size, cfg.hidden_size), "base_kernel")?,
            kernel_projection: vb
                .pp("kernel_projection")
                .get((2 * kernel_size * groups, cfg.hidden_size), "weight")?,
            kernel_size,
            group_size,
        })
    }

    // base + dynamic two-tap causal depthwise conv, grouped so one dynamic scalar covers group_size
    // channels: out_t = sum_o (base[o] + dyn[t, o]) * x_{t-o}
    fn convolve(&self, hidden: &Tensor, dynamic: &Tensor, base: &Tensor) -> Result<Tensor> {
        let (b, len, h) = hidden.dims3()?;
        let groups = h / self.group_size;
        let blocks = hidden.reshape((b, len, groups, self.group_size))?;
        // dynamic: [b, len, kernel_size, groups] -> broadcast over group_size
        let dynamic = dynamic.reshape((b, len, self.kernel_size, groups, 1))?;
        let mut output: Option<Tensor> = None;
        for offset in 0..self.kernel_size {
            let values = if offset == 0 {
                blocks.clone()
            } else {
                let kept = blocks.narrow(1, 0, len - offset)?;
                let pad = Tensor::zeros(
                    (b, offset, groups, self.group_size),
                    blocks.dtype(),
                    blocks.device(),
                )?;
                Tensor::cat(&[pad, kept], 1)?
            };
            let kernel = base
                .i(offset)?
                .reshape((1, 1, groups, self.group_size))?
                .to_dtype(hidden.dtype())?;
            let dyn_o = dynamic.i((.., .., offset))?;
            let term = values
                .broadcast_mul(&kernel)?
                .add(&values.broadcast_mul(&dyn_o)?)?;
            output = Some(match output {
                Some(acc) => acc.add(&term)?,
                None => term,
            });
        }
        output
            .ok_or_else(|| candle_core::Error::msg("empty conv kernel"))?
            .reshape((b, len, h))
    }

    fn prepare(&self, hidden: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, len, h) = hidden.dims3()?;
        let groups = h / self.group_size;
        let dynamic = hidden
            .broadcast_matmul(&self.kernel_projection.t()?.to_dtype(hidden.dtype())?)?
            .reshape((b, len, 2, self.kernel_size, groups))?;
        let pre = self.convolve(
            hidden,
            &dynamic.i((.., .., 0))?.contiguous()?,
            &self.base_kernel.i(0)?,
        )?;
        Ok((pre, dynamic.i((.., .., 1))?.contiguous()?))
    }

    fn finish(&self, hidden: &Tensor, dynamic: &Tensor) -> Result<Tensor> {
        self.convolve(hidden, dynamic, &self.base_kernel.i(1)?)
    }
}

struct CandidateSelector {
    // [vocab, rank] raw parameters (no `.weight` suffix in the checkpoint)
    predecessor_codebook: Tensor,
    successor_codebook: Tensor,
    hidden_projection: Tensor,
    top_k: usize,
}

impl CandidateSelector {
    fn load(vb: ShardedVarBuilder, cfg: &DFlashConfig) -> Result<Self> {
        let rank = cfg
            .dflash_config
            .selector_rank
            .ok_or_else(|| candle_core::Error::msg("DFlash2 config has no selector_rank"))?;
        let top_k = cfg
            .dflash_config
            .selector_top_k
            .ok_or_else(|| candle_core::Error::msg("DFlash2 config has no selector_top_k"))?;
        Ok(Self {
            predecessor_codebook: vb.get_unchecked("predecessor_codebook")?,
            successor_codebook: vb.get_unchecked("successor_codebook")?,
            hidden_projection: vb
                .pp("hidden_projection")
                .get((rank, cfg.hidden_size), "weight")?,
            top_k,
        })
    }

    #[cfg(feature = "cuda")]
    fn supports_cuda(&self) -> bool {
        matches!(self.hidden_projection.dtype(), DType::BF16 | DType::F32)
            && matches!(self.predecessor_codebook.dtype(), DType::BF16 | DType::F32)
            && matches!(self.successor_codebook.dtype(), DType::BF16 | DType::F32)
    }

    #[cfg(feature = "cuda")]
    fn select_greedy_cuda(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
        anchors: &Tensor,
    ) -> Result<Tensor> {
        let (batch, positions, vocab) = logits.dims3()?;
        let rows = batch * positions;
        let logits = logits
            .reshape((rows, vocab))?
            .to_dtype(DType::F32)?
            .contiguous()?;
        if !logits.device().is_cuda() || !self.supports_cuda() {
            candle_core::bail!("DFlash CUDA selector does not support these tensors");
        }
        let inverse_temperatures = Tensor::ones((rows,), DType::F32, logits.device())?;
        let topk = crate::ops::cuda_topk_logits_f32_packed_batched(
            &logits,
            self.top_k,
            &inverse_temperatures,
        )?;
        let projection_dtype = self.hidden_projection.dtype();
        let hidden = hidden.reshape((rows, ()))?.to_dtype(projection_dtype)?;
        let projected = hidden
            .broadcast_matmul(&self.hidden_projection.t()?)?
            .contiguous()?;
        crate::ops::cuda_dflash_greedy_select(
            &topk.packed,
            &projected,
            &self.predecessor_codebook.contiguous()?,
            &self.successor_codebook.contiguous()?,
            anchors,
        )
    }

    /// Greedy path walk over the per-position top-k candidates for every sequence at once; scores
    /// couple each candidate to the chosen predecessor through the low-rank codebooks gated by the
    /// position's hidden state. `hidden` is `[batch, n, hidden]` and `logits` is
    /// `[batch, n, vocab]`.
    fn select_greedy_batch(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
        anchors: &[u32],
    ) -> Result<Vec<Vec<u32>>> {
        let (batch, positions, vocab) = logits.dims3()?;
        let k = self.top_k;
        let rows = batch * positions;

        #[cfg(feature = "cuda")]
        if logits.device().is_cuda()
            && matches!(self.hidden_projection.dtype(), DType::BF16 | DType::F32)
            && matches!(self.predecessor_codebook.dtype(), DType::BF16 | DType::F32)
            && matches!(self.successor_codebook.dtype(), DType::BF16 | DType::F32)
        {
            let anchors = Tensor::from_vec(anchors.to_vec(), (batch,), logits.device())?;
            return self
                .select_greedy_cuda(hidden, logits, &anchors)?
                .to_vec2::<u32>();
        }

        let logits = logits
            .reshape((rows, vocab))?
            .to_dtype(DType::F32)?
            .contiguous()?;

        let (unary, candidates) = topk_rows(&logits, k)?;
        let hproj = hidden
            .reshape((rows, ()))?
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.hidden_projection.t()?.to_dtype(DType::F32)?)?;
        let cand_flat = candidates.iter().flatten().copied().collect::<Vec<u32>>();
        let cand_ids = Tensor::from_vec(cand_flat.clone(), (rows * k,), logits.device())?;
        let succ = self
            .successor_codebook
            .to_dtype(DType::F32)?
            .index_select(&cand_ids, 0)?;
        let mut pred_ids = anchors.to_vec();
        pred_ids.extend_from_slice(&cand_flat);
        let pred_ids_t = Tensor::from_vec(pred_ids.clone(), (pred_ids.len(),), logits.device())?;
        let pred = self
            .predecessor_codebook
            .to_dtype(DType::F32)?
            .index_select(&pred_ids_t, 0)?;
        let rank = hproj.dim(1)?;
        let packed = Tensor::cat(
            &[
                hproj.flatten_all()?,
                succ.flatten_all()?,
                pred.flatten_all()?,
            ],
            0,
        )?
        .to_vec1::<f32>()?;
        let (h_len, s_len) = (rows * rank, rows * k * rank);
        let hproj: Vec<&[f32]> = packed[..h_len].chunks(rank).collect();
        let succ: Vec<&[f32]> = packed[h_len..h_len + s_len].chunks(rank).collect();
        let pred: Vec<&[f32]> = packed[h_len + s_len..].chunks(rank).collect();

        let mut paths = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut path = Vec::with_capacity(positions);
            let mut pred_row = b;
            for pos in 0..positions {
                let row = b * positions + pos;
                let mut best = f32::NEG_INFINITY;
                let mut best_idx = 0usize;
                for cand in 0..k {
                    let mut dot = 0f32;
                    let sv = &succ[row * k + cand];
                    let pv = &pred[pred_row];
                    let hv = &hproj[row];
                    for r in 0..rank {
                        dot += pv[r] * hv[r] * sv[r];
                    }
                    let score = unary[row][cand] + dot;
                    if score > best {
                        best = score;
                        best_idx = cand;
                    }
                }
                path.push(candidates[row][best_idx]);
                pred_row = batch + row * k + best_idx;
            }
            paths.push(path);
        }
        Ok(paths)
    }
}

type TopkRows = (Vec<Vec<f32>>, Vec<Vec<u32>>);
// (is_causal, sliding_window, ctx_len, block)
type MaskKey = (bool, Option<usize>, usize, usize);

fn topk_rows(logits: &Tensor, k: usize) -> Result<TopkRows> {
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() {
        let (rows, _) = logits.dims2()?;
        let ones = Tensor::ones((rows,), DType::F32, logits.device())?;
        let packed = crate::ops::cuda_topk_logits_f32_packed_batched(logits, k, &ones)?;
        let packed_rows = packed.packed.to_vec2::<f32>()?;
        let mut values = Vec::with_capacity(rows);
        let mut indices = Vec::with_capacity(rows);
        for row in packed_rows {
            values.push(row[..k].to_vec());
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            indices.push(row[k..2 * k].iter().map(|i| *i as u32).collect());
        }
        return Ok((values, indices));
    }
    let rows = logits.to_vec2::<f32>()?;
    let mut values = Vec::with_capacity(rows.len());
    let mut indices = Vec::with_capacity(rows.len());
    for row in rows {
        let mut order = (0..row.len()).collect::<Vec<_>>();
        order.select_nth_unstable_by(k - 1, |&a, &b| row[b].total_cmp(&row[a]));
        let top = &order[..k];
        values.push(top.iter().map(|&i| row[i]).collect());
        #[allow(clippy::cast_possible_truncation)]
        indices.push(top.iter().map(|&i| i as u32).collect());
    }
    Ok((values, indices))
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

/// One sequence's context rows to append: `taps` `[1, rows, taps*hidden]` at absolute positions
/// `start_pos..start_pos + rows`.
pub struct CtxAppend {
    pub seq_id: usize,
    pub taps: Tensor,
    pub start_pos: usize,
}

struct AdaptiveState {
    max_n: usize,
}

#[derive(Clone, Copy)]
enum DFlashSequenceEviction {
    Dormant,
    Released,
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

#[cfg(any(
    all(feature = "cuda", feature = "flash-attn", target_family = "unix"),
    test
))]
struct DFlashGraphHostRows {
    token_ids: Vec<u32>,
    rope_indices: Vec<u32>,
    anchors: Vec<u32>,
}

#[cfg(any(
    all(feature = "cuda", feature = "flash-attn", target_family = "unix"),
    test
))]
fn dflash_graph_host_rows(
    anchors: &[u32],
    start_positions: &[usize],
    mask_token_id: u32,
    block: usize,
    batch_bucket: usize,
) -> Result<DFlashGraphHostRows> {
    if anchors.is_empty() || anchors.len() != start_positions.len() {
        candle_core::bail!("DFlash graph inputs must contain matching non-empty rows");
    }
    if block == 0 || batch_bucket < anchors.len() {
        candle_core::bail!("DFlash graph input shape is invalid");
    }
    let mut token_ids = Vec::with_capacity(batch_bucket * block);
    let mut rope_indices = Vec::with_capacity(batch_bucket * block);
    let mut padded_anchors = Vec::with_capacity(batch_bucket);
    for row in 0..batch_bucket {
        let source = row.min(anchors.len() - 1);
        let anchor = anchors[source];
        let start = start_positions[source];
        padded_anchors.push(anchor);
        token_ids.push(anchor);
        token_ids.extend(std::iter::repeat_n(mask_token_id, block - 1));
        for offset in 0..block {
            let position = start
                .checked_add(offset)
                .ok_or_else(|| candle_core::Error::msg("DFlash graph position overflow"))?;
            rope_indices.push(u32::try_from(position).map_err(candle_core::Error::wrap)?);
        }
    }
    Ok(DFlashGraphHostRows {
        token_ids,
        rope_indices,
        anchors: padded_anchors,
    })
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DFlashCudaGraphKey {
    batch_bucket: usize,
    block: usize,
    use_selector: bool,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
struct DFlashCudaGraphBuffers {
    token_ids: Var,
    rope_indices: Var,
    anchors: Var,
    block_tables: Var,
    slot_mapping: Var,
    cumulative_kv_lens: Var,
    cumulative_query_lens: Tensor,
    output_tokens: Var,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
struct DFlashCudaGraphEntry {
    key: DFlashCudaGraphKey,
    staging: crate::pipeline::cuda_graph::CudaGraphHostStaging,
    buffers: DFlashCudaGraphBuffers,
    token_embedding: Arc<dyn QuantMethod>,
    lm_head: Arc<dyn QuantMethod>,
    graph: crate::pipeline::cuda_graph::CudaGraphHandle,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
#[derive(Default)]
struct DFlashCudaGraphState {
    entries: Vec<DFlashCudaGraphEntry>,
    warmed: HashSet<DFlashCudaGraphKey>,
    failed: HashSet<DFlashCudaGraphKey>,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
struct DFlashCudaGraphRun<'a> {
    model: &'a DFlashDraftModel,
    key: DFlashCudaGraphKey,
    rows: &'a DFlashGraphHostRows,
    attention_batch: &'a WindowedKvBatch,
    token_embedding: &'a Arc<dyn QuantMethod>,
    lm_head: &'a Arc<dyn QuantMethod>,
    real_batch: usize,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    cuda_graphs: Mutex<DFlashCudaGraphState>,
    layers: Vec<DFlashLayer>,
    fc: Arc<dyn QuantMethod>,
    context_kv_proj: Arc<dyn QuantMethod>,
    hidden_norm: RmsNorm,
    norm: RmsNorm,
    selector: Option<CandidateSelector>,
    inv_freq: Tensor,
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
    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    windowed_pool: Option<Mutex<WindowedKvPool>>,
    dormant_seqs: Mutex<HashSet<usize>>,
    // cos/sin for positions 0..ROPE_TABLE_LEN, [len, head_dim/2]
    rope_table: (Tensor, Tensor),
    mask_cache: Mutex<HashMap<MaskKey, Tensor>>,
    adaptive: Mutex<Option<AdaptiveState>>,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
impl DFlashCudaGraphBuffers {
    fn new(
        key: DFlashCudaGraphKey,
        rows: &DFlashGraphHostRows,
        batch: &WindowedKvBatch,
        device: &Device,
    ) -> Result<Self> {
        let token_ids = Var::from_tensor(&Tensor::from_vec(
            rows.token_ids.clone(),
            (key.batch_bucket, key.block),
            device,
        )?)?;
        let rope_indices = Var::from_tensor(&Tensor::from_vec(
            rows.rope_indices.clone(),
            (key.batch_bucket * key.block,),
            device,
        )?)?;
        let anchors = Var::from_tensor(&Tensor::from_vec(
            rows.anchors.clone(),
            (key.batch_bucket,),
            device,
        )?)?;
        let block_tables = Var::from_tensor(&Tensor::from_vec(
            batch.block_tables_for_graph().to_vec(),
            (key.batch_bucket, batch.block_table_width_for_graph()),
            device,
        )?)?;
        let slot_mapping = Var::from_tensor(&Tensor::from_vec(
            batch.slot_mapping_for_graph().to_vec(),
            (key.batch_bucket * key.block,),
            device,
        )?)?;
        let cumulative_kv_lens = Var::from_tensor(&Tensor::from_vec(
            batch.cumulative_kv_lens_for_graph().to_vec(),
            (key.batch_bucket + 1,),
            device,
        )?)?;
        let cumulative_query_lens = Tensor::from_vec(
            batch.cumulative_query_lens_for_graph().to_vec(),
            (key.batch_bucket + 1,),
            device,
        )?;
        let output_tokens = Var::zeros((key.batch_bucket, key.block - 1), DType::U32, device)?;
        Ok(Self {
            token_ids,
            rope_indices,
            anchors,
            block_tables,
            slot_mapping,
            cumulative_kv_lens,
            cumulative_query_lens,
            output_tokens,
        })
    }

    fn metadata(&self) -> WindowedKvBatchTensors {
        WindowedKvBatchTensors {
            block_tables: self.block_tables.as_detached_tensor(),
            slot_mapping: self.slot_mapping.as_detached_tensor(),
            cumulative_query_lens: self.cumulative_query_lens.clone(),
            cumulative_kv_lens: self.cumulative_kv_lens.as_detached_tensor(),
        }
    }

    fn update(
        &self,
        rows: &DFlashGraphHostRows,
        batch: &WindowedKvBatch,
        staging: &mut crate::pipeline::cuda_graph::CudaGraphHostStaging,
    ) -> Result<()> {
        let location = self.token_ids.device().location();
        staging.update(|staging| {
            staging.copy_from_u32_slice(
                "dflash_token_ids",
                location,
                &rows.token_ids,
                &self.token_ids,
            )?;
            staging.copy_from_u32_slice(
                "dflash_rope_indices",
                location,
                &rows.rope_indices,
                &self.rope_indices,
            )?;
            staging.copy_from_u32_slice(
                "dflash_anchors",
                location,
                &rows.anchors,
                &self.anchors,
            )?;
            staging.copy_from_u32_slice(
                "dflash_block_tables",
                location,
                batch.block_tables_for_graph(),
                &self.block_tables,
            )?;
            staging.copy_from_i64_slice(
                "dflash_slot_mapping",
                location,
                batch.slot_mapping_for_graph(),
                &self.slot_mapping,
            )?;
            staging.copy_from_u32_slice(
                "dflash_cumulative_kv_lens",
                location,
                batch.cumulative_kv_lens_for_graph(),
                &self.cumulative_kv_lens,
            )
        })
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
impl DFlashCudaGraphEntry {
    fn matches_dependencies(
        &self,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
    ) -> bool {
        Arc::ptr_eq(&self.token_embedding, token_embedding) && Arc::ptr_eq(&self.lm_head, lm_head)
    }

    fn launch(&mut self, real_batch: usize) -> Result<Vec<Vec<u32>>> {
        self.graph.launch()?;
        self.staging.record_graph_complete()?;
        let mut tokens = self
            .buffers
            .output_tokens
            .as_detached_tensor()
            .to_vec2::<u32>()?;
        tokens.truncate(real_batch);
        Ok(tokens)
    }

    fn replay(
        &mut self,
        rows: &DFlashGraphHostRows,
        batch: &WindowedKvBatch,
        real_batch: usize,
    ) -> Result<Vec<Vec<u32>>> {
        self.buffers.update(rows, batch, &mut self.staging)?;
        self.staging.order_before_graph()?;
        self.launch(real_batch)
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
fn release_dflash_cuda_graph(entry: DFlashCudaGraphEntry) {
    let stream = entry.graph.stream().clone();
    drop(entry);
    if let Err(err) = crate::pipeline::cuda_graph::trim_cuda_graph_memory(&stream) {
        tracing::warn!("Failed to trim released DFlash CUDA graph memory: {err:?}");
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
impl Drop for DFlashCudaGraphState {
    fn drop(&mut self) {
        for entry in self.entries.drain(..) {
            release_dflash_cuda_graph(entry);
        }
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
impl DFlashCudaGraphState {
    fn retire_failed_entry(
        &mut self,
        entry: DFlashCudaGraphEntry,
        operation: &str,
        failure: candle_core::Error,
    ) -> Result<()> {
        let key = entry.key;
        let synchronize_result = entry
            .graph
            .stream()
            .synchronize()
            .map_err(candle_core::Error::wrap);
        self.failed.insert(key);
        self.warmed.remove(&key);
        release_dflash_cuda_graph(entry);
        match synchronize_result {
            Ok(()) => {
                tracing::warn!(
                    batch_bucket = key.batch_bucket,
                    block = key.block,
                    "DFlash CUDA graph {operation} failed; disabled this shape and retrying eagerly: {failure:?}"
                );
                Ok(())
            }
            Err(synchronize_err) => Err(candle_core::Error::msg(format!(
                "DFlash CUDA graph {operation} failed: {failure}; recovery synchronization failed: {synchronize_err}"
            ))),
        }
    }

    fn eager(
        model: &DFlashDraftModel,
        key: DFlashCudaGraphKey,
        rows: &DFlashGraphHostRows,
        attention_batch: &WindowedKvBatch,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
        real_batch: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let buffers = DFlashCudaGraphBuffers::new(key, rows, attention_batch, &model.device)?;
        let Device::Cuda(cuda_device) = &model.device else {
            candle_core::bail!("DFlash CUDA graph expected a CUDA device");
        };
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
        let mut tokens = model
            .cuda_graph_tokens(key, &buffers, attention_batch, token_embedding, lm_head)?
            .to_vec2::<u32>()?;
        tokens.truncate(real_batch);
        Ok(tokens)
    }

    fn capture(
        model: &DFlashDraftModel,
        key: DFlashCudaGraphKey,
        rows: &DFlashGraphHostRows,
        attention_batch: &WindowedKvBatch,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
    ) -> Result<DFlashCudaGraphEntry> {
        let buffers = DFlashCudaGraphBuffers::new(key, rows, attention_batch, &model.device)?;
        model.device.synchronize()?;
        let Device::Cuda(cuda_device) = &model.device else {
            candle_core::bail!("DFlash CUDA graph expected a CUDA device");
        };
        let stream = cuda_device.cuda_stream();
        let _memory_pool_guard =
            crate::pipeline::cuda_graph::prepare_cuda_graph_memory_pool(&stream)?;
        let restore_event_tracking =
            crate::pipeline::cuda_graph::disable_event_tracking_for_capture(&stream);
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
        if let Err(err) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
                &stream,
                restore_event_tracking,
            );
            return Err(candle_core::Error::msg(err.to_string())
                .context("DFlash CUDA graph begin capture failed"));
        }

        let capture_result =
            model.cuda_graph_tokens(key, &buffers, attention_batch, token_embedding, lm_head);
        match capture_result {
            Ok(tokens) => {
                if let Err(err) = crate::cuda::graph::copy_tensor(
                    &tokens,
                    &buffers.output_tokens.as_detached_tensor(),
                ) {
                    crate::pipeline::cuda_graph::end_cuda_capture_discard(&stream);
                    crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
                        &stream,
                        restore_event_tracking,
                    );
                    return Err(err.context("DFlash CUDA graph output copy capture failed"));
                }
            }
            Err(err) => {
                crate::pipeline::cuda_graph::end_cuda_capture_discard(&stream);
                crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
                    &stream,
                    restore_event_tracking,
                );
                return Err(err.context("DFlash CUDA graph forward capture failed"));
            }
        }

        let graph = match crate::pipeline::cuda_graph::CudaGraphHandle::end_capture(&stream) {
            Ok(Some(graph)) => graph,
            Ok(None) => {
                crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
                    &stream,
                    restore_event_tracking,
                );
                return Err(candle_core::Error::msg(
                    "DFlash CUDA graph capture returned no graph",
                ));
            }
            Err(err) => {
                crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
                    &stream,
                    restore_event_tracking,
                );
                return Err(err);
            }
        };
        crate::pipeline::cuda_graph::restore_event_tracking_after_capture(
            &stream,
            restore_event_tracking,
        );
        graph.upload()?;
        let staging = crate::pipeline::cuda_graph::CudaGraphHostStaging::new(stream)?;
        Ok(DFlashCudaGraphEntry {
            key,
            staging,
            buffers,
            token_embedding: token_embedding.clone(),
            lm_head: lm_head.clone(),
            graph,
        })
    }

    fn run(&mut self, run: DFlashCudaGraphRun<'_>) -> Result<Option<Vec<Vec<u32>>>> {
        let DFlashCudaGraphRun {
            model,
            key,
            rows,
            attention_batch,
            token_embedding,
            lm_head,
            real_batch,
        } = run;
        if let Some(position) = self.entries.iter().position(|entry| {
            entry.key == key && entry.matches_dependencies(token_embedding, lm_head)
        }) {
            let mut entry = self.entries.remove(position);
            match entry.replay(rows, attention_batch, real_batch) {
                Ok(tokens) => {
                    self.entries.push(entry);
                    return Ok(Some(tokens));
                }
                Err(err) => {
                    self.retire_failed_entry(entry, "replay", err)?;
                    return Self::eager(
                        model,
                        key,
                        rows,
                        attention_batch,
                        token_embedding,
                        lm_head,
                        real_batch,
                    )
                    .map(Some)
                    .map_err(|err| err.context("DFlash eager fallback after graph replay failed"));
                }
            }
        }
        let mismatched = self
            .entries
            .iter()
            .position(|entry| entry.key == key)
            .map(|position| self.entries.remove(position));
        if let Some(entry) = mismatched {
            release_dflash_cuda_graph(entry);
        }
        if self.failed.contains(&key) {
            return Ok(None);
        }
        if self.warmed.insert(key) {
            return Self::eager(
                model,
                key,
                rows,
                attention_batch,
                token_embedding,
                lm_head,
                real_batch,
            )
            .map(Some);
        }

        let mut entry =
            match Self::capture(model, key, rows, attention_batch, token_embedding, lm_head) {
                Ok(entry) => entry,
                Err(err) => {
                    self.failed.insert(key);
                    tracing::warn!(
                        batch_bucket = key.batch_bucket,
                        block = key.block,
                        "DFlash CUDA graph capture disabled for this shape: {err:?}"
                    );
                    return Ok(None);
                }
            };
        let tokens = match entry.launch(real_batch) {
            Ok(tokens) => tokens,
            Err(err) => {
                self.retire_failed_entry(entry, "first launch", err)?;
                return Self::eager(
                    model,
                    key,
                    rows,
                    attention_batch,
                    token_embedding,
                    lm_head,
                    real_batch,
                )
                .map(Some)
                .map_err(|err| {
                    err.context("DFlash eager fallback after first graph launch failed")
                });
            }
        };
        if self.entries.len() >= DFLASH_CUDA_GRAPH_CACHE_CAPACITY {
            release_dflash_cuda_graph(self.entries.remove(0));
        }
        tracing::debug!(
            batch_bucket = key.batch_bucket,
            block = key.block,
            "Captured DFlash CUDA graph"
        );
        self.entries.push(entry);
        Ok(Some(tokens))
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

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
pub(crate) fn windowed_kv_cache_size_in_bytes(
    config: &MtpConfig,
    sequence_capacity: usize,
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
    let elements = cfg
        .num_hidden_layers
        .checked_mul(sequence_capacity)
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

impl DFlashDraftModel {
    /// Loads a DFlash/DFlash2 drafter from a local path or HF repo. `target_num_layers` and
    /// `target_hidden_size` validate the checkpoint against the target model; the config's draft
    /// ISQ type requantizes the projection weights so drafting reads match the target's own.
    pub fn load(
        config: &MtpConfig,
        target_num_layers: usize,
        target_hidden_size: usize,
        device: &Device,
        dtype: DType,
        silent: bool,
    ) -> Result<Self> {
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
        let head_dim = cfg.head_dim();
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

        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / (cfg.rope_theta() as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (head_dim / 2,), device)?;
        let rope_table = {
            #[allow(clippy::cast_precision_loss)]
            let pos: Vec<f32> = (0..ROPE_TABLE_LEN).map(|p| p as f32).collect();
            let pos = Tensor::from_vec(pos, (ROPE_TABLE_LEN, 1), device)?;
            let freqs = pos.broadcast_matmul(&inv_freq.reshape((1, ()))?)?;
            (freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?)
        };

        Ok(Self {
            #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
            cuda_graphs: Mutex::new(DFlashCudaGraphState::default()),
            layers,
            fc,
            context_kv_proj,
            hidden_norm: RmsNorm::new(hidden, eps, vb.pp("hidden_norm"))?,
            norm: RmsNorm::new(hidden, eps, vb.pp("norm"))?,
            selector,
            inv_freq,
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
            #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
            windowed_pool: None,
            dormant_seqs: Mutex::new(HashSet::new()),
            rope_table,
            mask_cache: Mutex::new(HashMap::new()),
            adaptive: Mutex::new(None),
        })
    }

    pub fn enable_windowed_kv(&mut self, sequence_capacity: usize) -> Result<bool> {
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
            let config = WindowedKvPoolConfig::new(
                sequence_capacity,
                layer_windows,
                self.block_size,
                crate::paged_attention::DEFAULT_PAGED_ATTENTION_BLOCK_SIZE,
                self.num_kv_heads,
                self.head_dim,
            )?;
            let pages = config.pages_per_sequence();
            self.windowed_pool = Some(Mutex::new(WindowedKvPool::new(config, &self.device)?));
            tracing::info!(
                sequence_capacity,
                pages_per_sequence = pages,
                "Using bounded paged FlashAttention KV for DFlash"
            );
            Ok(true)
        }
        #[cfg(not(all(feature = "cuda", feature = "flash-attn", target_family = "unix")))]
        {
            let _ = sequence_capacity;
            Ok(false)
        }
    }

    pub(crate) fn greedy_proposals_cuda_graph(
        &self,
        seq_ids: &[usize],
        anchors: &[u32],
        start_positions: &[usize],
        n_predict: usize,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
            if block > self.block_size
                || start_positions
                    .iter()
                    .any(|start| start.saturating_add(block) > ROPE_TABLE_LEN)
            {
                return Ok(None);
            }
            let Some(batch_bucket) =
                crate::pipeline::cuda_graph::cuda_graph_batch_bucket(seq_ids.len())
            else {
                return Ok(None);
            };
            let use_selector =
                self.selector.is_some() && std::env::var("MISTRALRS_DFLASH_NO_SELECTOR").is_err();
            if use_selector
                && !self
                    .selector
                    .as_ref()
                    .expect("selector checked above")
                    .supports_cuda()
            {
                return Ok(None);
            }
            let key = DFlashCudaGraphKey {
                batch_bucket,
                block,
                use_selector,
            };
            let rows = dflash_graph_host_rows(
                anchors,
                start_positions,
                self.mask_token_id,
                block,
                batch_bucket,
            )?;
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
                    rows: &rows,
                    attention_batch: &attention_batch,
                    token_embedding,
                    lm_head,
                    real_batch: seq_ids.len(),
                });
        }
        #[cfg(not(all(feature = "cuda", feature = "flash-attn", target_family = "unix")))]
        {
            let _ = (
                seq_ids,
                anchors,
                start_positions,
                n_predict,
                token_embedding,
                lm_head,
            );
            Ok(None)
        }
    }

    pub fn enable_adaptive(&self, max_n: usize) -> bool {
        if max_n == 0 {
            return false;
        }
        *self.adaptive.lock().expect("dflash adaptive poisoned") = Some(AdaptiveState { max_n });
        true
    }

    pub fn plan_n(&self, max_n: usize, batch: usize) -> usize {
        let guard = self.adaptive.lock().expect("dflash adaptive poisoned");
        if let Some(adaptive) = guard.as_ref() {
            debug_assert_eq!(adaptive.max_n, max_n);
        }
        select_dflash_depth(guard.is_some(), max_n, batch)
    }

    pub fn graph_plans(&self, max_n: usize) -> Vec<super::SpeculativeGraphPlan> {
        let guard = self.adaptive.lock().expect("dflash adaptive poisoned");
        if let Some(adaptive) = guard.as_ref() {
            debug_assert_eq!(adaptive.max_n, max_n);
        }
        dflash_graph_plans(guard.is_some(), max_n)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn has_selector(&self) -> bool {
        self.selector.is_some()
    }

    pub fn mask_token_id(&self) -> u32 {
        self.mask_token_id
    }

    pub fn input_embedding_scale(&self) -> f64 {
        self.input_embedding_scale
    }

    /// Absolute position of the next context token a sequence expects, or None if unseen.
    pub fn ctx_next_pos(&self, seq_id: usize) -> Option<usize> {
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
        if start + len <= ROPE_TABLE_LEN {
            return Ok((
                self.rope_table.0.narrow(0, start, len)?,
                self.rope_table.1.narrow(0, start, len)?,
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let pos: Vec<f32> = (start..start + len).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, (len, 1), &self.device)?;
        let freqs = pos.broadcast_matmul(&self.inv_freq.reshape((1, ()))?)?;
        Ok((
            freqs.cos()?.to_dtype(self.dtype)?,
            freqs.sin()?.to_dtype(self.dtype)?,
        ))
    }

    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        candle_nn::rotary_emb::rope(&x.contiguous()?, cos, sin)
    }

    // [1, len, hidden] -> [1, heads, len, head_dim]
    fn split_heads(&self, x: &Tensor, heads: usize) -> Result<Tensor> {
        let (b, len, _) = x.dims3()?;
        x.reshape((b, len, heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    // [1, len, heads, head_dim] -> [1, heads, len, head_dim]
    fn heads_first(x: &Tensor) -> Result<Tensor> {
        x.transpose(1, 2)?.contiguous()
    }

    /// Projects tap features and appends context keys/values for every entry at once: the fc and
    /// per-layer k/v projections read their weights once over all sequences' rows packed together.
    pub fn append_ctx_batch(&self, entries: &[CtxAppend]) -> Result<()> {
        let rows = entries
            .iter()
            .map(|e| e.taps.dim(1))
            .collect::<Result<Vec<_>>>()?;
        let total: usize = rows.iter().sum();
        if total == 0 {
            return Ok(());
        }
        let packed = if entries.len() == 1 {
            entries[0].taps.clone()
        } else {
            Tensor::cat(&entries.iter().map(|e| &e.taps).collect::<Vec<_>>(), 1)?
        };
        let ctx_hidden = self
            .hidden_norm
            .forward(&self.fc.forward(&packed.to_dtype(self.dtype)?)?)?;
        let (cos, sin) = {
            let mut coss = Vec::with_capacity(entries.len());
            let mut sins = Vec::with_capacity(entries.len());
            for (e, r) in entries.iter().zip(rows.iter()) {
                if *r == 0 {
                    continue;
                }
                let (c, s) = self.cos_sin(e.start_pos, *r)?;
                coss.push(c);
                sins.push(s);
            }
            (Tensor::cat(&coss, 0)?, Tensor::cat(&sins, 0)?)
        };

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
        let mut ks = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let k = layer.k_norm.forward(&raw_k.narrow(0, i, 1)?)?;
            let k = self.rope(&k, &cos, &sin)?;
            ks.push(k.squeeze(0)?);
        }
        // [layers, kv_heads, total, head_dim]
        let k_all = Tensor::stack(&ks, 0)?;

        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
        if self.windowed_pool.is_some() {
            return self.append_ctx_windowed(entries, &rows, &k_all, &v_all);
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

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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

        for layer_idx in 0..self.layers.len() {
            let mut k_rows = Vec::with_capacity(entries.len());
            let mut v_rows = Vec::with_capacity(entries.len());
            let mut slot_mapping = Vec::new();
            let mut offset = 0;
            for (write, row_count) in writes.iter().zip(rows) {
                let retained = write.retained_input_range();
                let retained_len = retained.end - retained.start;
                k_rows.push(
                    k_all
                        .i(layer_idx)?
                        .narrow(1, offset + retained.start, retained_len)?
                        .transpose(0, 1)?
                        .force_contiguous()?,
                );
                v_rows.push(
                    v_all
                        .i(layer_idx)?
                        .narrow(1, offset + retained.start, retained_len)?
                        .transpose(0, 1)?
                        .force_contiguous()?,
                );
                slot_mapping.extend_from_slice(write.slot_mapping());
                offset += *row_count;
            }
            let k = if k_rows.len() == 1 {
                k_rows.pop().expect("one context K tensor")
            } else {
                Tensor::cat(&k_rows.iter().collect::<Vec<_>>(), 0)?
            };
            let v = if v_rows.len() == 1 {
                v_rows.pop().expect("one context V tensor")
            } else {
                Tensor::cat(&v_rows.iter().collect::<Vec<_>>(), 0)?
            };
            let slot_count = slot_mapping.len();
            let slots = Tensor::from_vec(slot_mapping, (slot_count,), &self.device)?;
            let (key_cache, value_cache) = pool.layer_cache(layer_idx)?;
            mistralrs_paged_attn::reshape_and_cache_flashinfer(
                &k,
                &v,
                &key_cache,
                &value_cache,
                &slots,
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
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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

    fn run_draft_layers<F>(
        &self,
        noise_embedding: &Tensor,
        q_cos: &Tensor,
        q_sin: &Tensor,
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
            let q = Self::heads_first(&layer.q_norm.forward(
                &qkv.narrow(D::Minus1, 0, q_size)?.reshape((
                    batch,
                    block,
                    self.num_heads,
                    self.head_dim,
                ))?,
            )?)?;
            let k_noise = Self::heads_first(&layer.k_norm.forward(
                &qkv.narrow(D::Minus1, q_size, kv_size)?.reshape((
                    batch,
                    block,
                    self.num_kv_heads,
                    self.head_dim,
                ))?,
            )?)?;
            let q = candle_nn::rotary_emb::rope(&q, q_cos, q_sin)?;
            let k_noise = candle_nn::rotary_emb::rope(&k_noise, q_cos, q_sin)?;
            let v_noise = self.split_heads(
                &qkv.narrow(D::Minus1, q_size + kv_size, kv_size)?,
                self.num_kv_heads,
            )?;
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
            let mut out = layer.down_proj.forward(&crate::ops::split_mul_and_act(
                &gate_up,
                self.intermediate_size,
                crate::layers::Activation::Silu,
            )?)?;
            if let (Some(conv), Some(kernel)) = (&layer.mlp_conv, mlp_kernel) {
                out = conv.finish(&out, &kernel)?;
            }
            hs = (residual + out)?;
        }
        Ok(hs)
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
            |layer_idx, layer, q, k_noise, v_noise| {
                let k_write = k_noise.transpose(1, 2)?.contiguous()?;
                let v_write = v_noise.transpose(1, 2)?.contiguous()?;
                let (key_cache, value_cache) = pool.layer_cache(layer_idx)?;
                mistralrs_paged_attn::reshape_and_cache_flashinfer(
                    &k_write,
                    &v_write,
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
                let q = q.transpose(1, 2)?.contiguous()?.reshape((
                    batch * block,
                    self.num_heads,
                    self.head_dim,
                ))?;
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

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    fn cuda_graph_tokens(
        &self,
        key: DFlashCudaGraphKey,
        buffers: &DFlashCudaGraphBuffers,
        attention_batch: &WindowedKvBatch,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
    ) -> Result<Tensor> {
        let mut noise = token_embedding
            .embedding_forward(&buffers.token_ids.as_detached_tensor(), self.dtype)?;
        if (self.input_embedding_scale - 1.0).abs() > f64::EPSILON {
            noise = (noise * self.input_embedding_scale)?;
        }
        let rope_indices = buffers.rope_indices.as_detached_tensor();
        let q_cos = self.rope_table.0.index_select(&rope_indices, 0)?.reshape((
            key.batch_bucket,
            key.block,
            self.head_dim / 2,
        ))?;
        let q_sin = self.rope_table.1.index_select(&rope_indices, 0)?.reshape((
            key.batch_bucket,
            key.block,
            self.head_dim / 2,
        ))?;
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
        let tokens = if key.use_selector {
            self.selector
                .as_ref()
                .expect("selector graph key requires a selector")
                .select_greedy_cuda(&hidden, &logits, &buffers.anchors.as_detached_tensor())?
        } else {
            logits.argmax(D::Minus1)?
        };
        tokens.contiguous()
    }

    /// One lm_head projection and one D2H finish every sequence's drafts: `hidden` is the
    /// `[batch, n, hidden]` output of `draft_hidden_batch`. Returns per-seq
    /// (tokens, logits `[n, vocab]`).
    pub fn finish_greedy_drafts(
        &self,
        hidden: &Tensor,
        anchors: &[u32],
        lm_head: &Arc<dyn QuantMethod>,
    ) -> Result<Vec<(Vec<u32>, Tensor)>> {
        let mut logits = lm_head.forward(hidden)?; // [batch, n, vocab]
        if (self.output_multiplier - 1.0).abs() > f64::EPSILON {
            logits = (logits * self.output_multiplier)?;
        }
        let use_selector = std::env::var("MISTRALRS_DFLASH_NO_SELECTOR").is_err();
        let tokens_per_seq = match (&self.selector, use_selector) {
            (Some(selector), true) => selector.select_greedy_batch(hidden, &logits, anchors)?,
            _ => logits.argmax(D::Minus1)?.to_vec2::<u32>()?,
        };
        tokens_per_seq
            .into_iter()
            .enumerate()
            .map(|(i, tokens)| Ok((tokens, logits.i(i)?)))
            .collect()
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
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use candle_core::{Device, Result, Tensor, D};
    use mistralrs_quant::QuantMethod;

    use super::{
        dflash_adaptive_env_value, dflash_graph_host_rows, dflash_graph_plans, linear_from_weight,
        select_dflash_depth, update_dormant_sequences, DFlashSequenceEviction,
        ADAPT_FULL_DEPTH_MAX_BATCH,
    };
    use crate::speculative::SpeculativeGraphPlan;

    #[test]
    fn final_sequence_release_does_not_leave_dormant_ids() {
        let mut dormant = HashSet::from([1, 2]);
        update_dormant_sequences(&mut dormant, &[2, 3], DFlashSequenceEviction::Dormant);
        assert_eq!(dormant, HashSet::from([1, 2, 3]));

        update_dormant_sequences(&mut dormant, &[2, 3], DFlashSequenceEviction::Released);
        assert_eq!(dormant, HashSet::from([1]));
    }

    #[test]
    fn packed_projection_preserves_component_layout() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::new(&[[2f32, 3., 5.], [7., 11., 13.]], &device)?;
        let weights = [
            Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?,
            Tensor::new(&[[7f32, 8., 9.]], &device)?,
            Tensor::new(&[[10f32, 11., 12.]], &device)?,
        ];
        let components = weights
            .iter()
            .map(|weight| linear_from_weight(weight.clone(), None, &device))
            .collect::<Result<Vec<Arc<dyn QuantMethod>>>>()?;
        let packed_weight = Tensor::cat(&weights.iter().collect::<Vec<_>>(), 0)?;
        let packed = linear_from_weight(packed_weight, None, &device)?.forward(&input)?;

        let mut offset = 0;
        for component in components {
            let expected = component.forward(&input)?;
            let rows = expected.dim(D::Minus1)?;
            assert_eq!(
                packed.narrow(D::Minus1, offset, rows)?.to_vec2::<f32>()?,
                expected.to_vec2::<f32>()?
            );
            offset += rows;
        }
        Ok(())
    }

    #[test]
    fn adaptive_depth_is_full_for_small_batches() {
        assert_eq!(select_dflash_depth(true, 7, 1), 7);
        assert_eq!(select_dflash_depth(true, 7, ADAPT_FULL_DEPTH_MAX_BATCH), 7);
    }

    #[test]
    fn adaptive_depth_keeps_a_live_draft_at_large_batches() {
        for max_n in 1..=7 {
            for batch in [3, 8, 16] {
                assert!(select_dflash_depth(true, max_n, batch) > 0);
            }
        }
        assert_eq!(select_dflash_depth(true, 7, 16), 1);
    }

    #[test]
    fn fixed_depth_ignores_batch_size() {
        assert_eq!(select_dflash_depth(false, 7, 1), 7);
        assert_eq!(select_dflash_depth(false, 7, 16), 7);
    }

    #[test]
    fn adaptive_mode_requires_an_explicit_true_value() {
        assert!(dflash_adaptive_env_value("1"));
        assert!(dflash_adaptive_env_value("true"));
        assert!(dflash_adaptive_env_value("TRUE"));
        for value in ["", "0", "false", "yes", "on", "7"] {
            assert!(!dflash_adaptive_env_value(value));
        }
    }

    #[test]
    fn graph_plans_cover_every_batch_without_a_zero_depth_gap() {
        assert_eq!(
            dflash_graph_plans(false, 7),
            vec![SpeculativeGraphPlan::new(7, None)]
        );
        assert_eq!(
            dflash_graph_plans(true, 7),
            vec![
                SpeculativeGraphPlan::new(7, Some(ADAPT_FULL_DEPTH_MAX_BATCH)),
                SpeculativeGraphPlan::new(1, None),
            ]
        );
        assert_eq!(
            dflash_graph_plans(true, 1),
            vec![SpeculativeGraphPlan::new(1, None)]
        );
    }

    #[test]
    fn graph_rows_pad_with_a_valid_alias() -> Result<()> {
        let rows = dflash_graph_host_rows(&[5, 7], &[100, 200], 9, 4, 4)?;
        assert_eq!(
            rows.token_ids,
            [5, 9, 9, 9, 7, 9, 9, 9, 7, 9, 9, 9, 7, 9, 9, 9]
        );
        assert_eq!(rows.anchors, [5, 7, 7, 7]);
        assert_eq!(
            rows.rope_indices,
            [100, 101, 102, 103, 200, 201, 202, 203, 200, 201, 202, 203, 200, 201, 202, 203,]
        );
        Ok(())
    }
}
