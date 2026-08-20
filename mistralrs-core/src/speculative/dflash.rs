//! DFlash block-diffusion draft models (<https://arxiv.org/abs/2602.06036>).
//!
//! A small stack of Qwen3-style layers drafts a whole block of tokens in one pass: queries come
//! from the noise block `[anchor, mask, mask, ...]`, keys/values from both the projected target
//! context features (hidden states tapped from intermediate target layers) and the noise itself.
//! Context keys accumulate in a per-sequence cache; only accepted positions are ever appended, so
//! a rejected tail needs no drafter rollback. DFlash 2 adds two-tap grouped dynamic convolutions
//! around each sublayer and a candidate path selector over the top-k logits per position.

use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use mistralrs_quant::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, ShardedVarBuilder, UnquantLinear,
};
use serde::Deserialize;

use crate::layers::RmsNorm;
use crate::speculative::MtpConfig;
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};

const DEFAULT_BLOCK_SIZE: usize = 16;
// Rotary table length precomputed at load; positions past it fall back to on-the-fly computation.
const ROPE_TABLE_LEN: usize = 65536;
const MASK_CACHE_CAP: usize = 64;
// Adaptive draft depth (SGLang-style): EMA of mean accepted drafts per step picks a tier from
// {3, 5, max}; a tier switch only lands after ADAPT_STICKY consecutive agreeing decisions.
const ADAPT_EMA_ALPHA: f32 = 0.2;
// Tier t is worth keeping while the EMA stays above t * ADAPT_TIER_FRAC accepted drafts.
const ADAPT_TIER_FRAC: f32 = 0.65;
const ADAPT_HYSTERESIS: f32 = 0.25;
// Upshift once the EMA saturates the current depth to within this margin.
const ADAPT_UP_MARGIN: f32 = 0.5;
const ADAPT_STICKY: u32 = 4;
const ADAPT_WARMUP: u32 = 8;
// After a switch, hold the new depth for this many steps so borderline content cannot flap between
// tiers every few steps (the bands overlap by construction: an EMA capped at the shallow tier can
// sit inside the deeper tier's downshift range).
const ADAPT_COOLDOWN: u32 = 32;
// Past this batch size the batched drafter makes deep drafts nearly free, so always draft at max.
// Also bounds which graph buckets the non-max draft depths are precaptured for.
pub const ADAPT_MAX_BATCH: usize = 2;
const ADAPT_MID_TIER: usize = 5;
const ADAPT_MIN_TIER: usize = 3;

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

    /// Greedy path walk over the per-position top-k candidates; scores couple each candidate to the
    /// chosen predecessor through the low-rank codebooks gated by the position's hidden state.
    /// Greedy path walk over the per-position top-k candidates for every sequence at once; scores
    /// couple each candidate to the chosen predecessor through the low-rank codebooks gated by the
    /// position's hidden state. `hidden` `[batch, n, hidden]`, `logits` `[batch, n, vocab]`; one
    /// batched top-k kernel and one D2H cover the whole batch, the walks are host-side.
    fn select_greedy_batch(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
        anchors: &[u32],
    ) -> Result<Vec<Vec<u32>>> {
        let (batch, positions, vocab) = logits.dims3()?;
        let k = self.top_k;
        let rows = batch * positions;
        let logits = logits
            .reshape((rows, vocab))?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (unary, candidates) = topk_rows(&logits, k)?;
        // H(h): [rows, rank]
        let hproj = hidden
            .reshape((rows, ()))?
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.hidden_projection.t()?.to_dtype(DType::F32)?)?;
        let cand_flat = candidates.iter().flatten().copied().collect::<Vec<u32>>();
        let cand_ids = Tensor::from_vec(cand_flat.clone(), (rows * k,), logits.device())?;
        let succ = self
            .successor_codebook
            .to_dtype(DType::F32)?
            .index_select(&cand_ids, 0)?; // [rows*k, rank]
                                          // Predecessors can only be an anchor or one of the candidates
        let mut pred_ids = anchors.to_vec();
        pred_ids.extend_from_slice(&cand_flat);
        let pred_ids_t = Tensor::from_vec(pred_ids.clone(), (pred_ids.len(),), logits.device())?;
        let pred = self
            .predecessor_codebook
            .to_dtype(DType::F32)?
            .index_select(&pred_ids_t, 0)?;
        // One D2H for everything; the walks themselves are tiny and host-side
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
            // pred rows: the batch anchors first, then every candidate
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
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
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
    ema: f32,
    current: usize,
    pending: usize,
    streak: u32,
    seen: u32,
    cooldown: u32,
}

pub struct DFlashDraftModel {
    layers: Vec<DFlashLayer>,
    fc: Arc<dyn QuantMethod>,
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
    dtype: DType,
    device: Device,
    ctx_cache: Mutex<HashMap<usize, SeqCtxCache>>,
    // cos/sin for positions 0..ROPE_TABLE_LEN, [len, head_dim/2]
    rope_table: (Tensor, Tensor),
    mask_cache: Mutex<HashMap<MaskKey, Tensor>>,
    adaptive: Mutex<Option<AdaptiveState>>,
}

fn load_linear(
    vb: &ShardedVarBuilder,
    shape: (usize, usize),
    name: &str,
    isq: Option<IsqType>,
    device: &Device,
) -> Result<Arc<dyn QuantMethod>> {
    let weight = vb.pp(name).get((shape.1, shape.0), "weight")?;
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

/// Reads only the drafter's config, to decide whether an `--mtp-model` path is a DFlash checkpoint.
pub fn peek_config(config: &MtpConfig) -> Result<DFlashConfig> {
    let path = config.resolve_path()?;
    let raw = fs::read_to_string(path.join("config.json"))
        .map_err(|e| candle_core::Error::Msg(format!("failed to read MTP model config: {e}")))?;
    serde_json::from_str(&raw).map_err(candle_core::Error::msg)
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
        for i in 0..cfg.num_hidden_layers {
            let vb_l = vb.pp("layers").pp(i);
            let vb_attn = vb_l.pp("self_attn");
            let (is_causal, sliding_window) = cfg.layer_attention(i);
            layers.push(DFlashLayer {
                q_proj: load_linear(
                    &vb_attn,
                    (hidden, cfg.num_attention_heads * head_dim),
                    "q_proj",
                    isq,
                    device,
                )?,
                k_proj: load_linear(
                    &vb_attn,
                    (hidden, cfg.num_key_value_heads * head_dim),
                    "k_proj",
                    isq,
                    device,
                )?,
                v_proj: load_linear(
                    &vb_attn,
                    (hidden, cfg.num_key_value_heads * head_dim),
                    "v_proj",
                    isq,
                    device,
                )?,
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
                gate_proj: load_linear(
                    &vb_l.pp("mlp"),
                    (hidden, cfg.intermediate_size),
                    "gate_proj",
                    isq,
                    device,
                )?,
                up_proj: load_linear(
                    &vb_l.pp("mlp"),
                    (hidden, cfg.intermediate_size),
                    "up_proj",
                    isq,
                    device,
                )?,
                down_proj: load_linear(
                    &vb_l.pp("mlp"),
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
            layers,
            fc,
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
            dtype,
            device: device.clone(),
            ctx_cache: Mutex::new(HashMap::new()),
            rope_table,
            mask_cache: Mutex::new(HashMap::new()),
            adaptive: Mutex::new(None),
        })
    }

    fn adaptive_tiers(max_n: usize) -> Vec<usize> {
        let mut tiers = vec![ADAPT_MIN_TIER, ADAPT_MID_TIER, max_n];
        tiers.retain(|t| *t <= max_n);
        tiers.dedup();
        tiers
    }

    /// Enables acceptance-driven draft depth; drafting starts at `max_n` and settles per workload.
    pub fn enable_adaptive(&self, max_n: usize) -> bool {
        if Self::adaptive_tiers(max_n).len() < 2 {
            return false;
        }
        *self.adaptive.lock().expect("dflash adaptive poisoned") = Some(AdaptiveState {
            ema: 0.0,
            current: max_n,
            pending: max_n,
            streak: 0,
            seen: 0,
            cooldown: 0,
        });
        true
    }

    /// The draft depth to use this step: the adaptive tier if enabled, else `max_n`.
    pub fn current_n(&self, max_n: usize) -> usize {
        self.adaptive
            .lock()
            .expect("dflash adaptive poisoned")
            .as_ref()
            .map_or(max_n, |s| s.current)
    }

    /// The draft depths the adaptive controller can pick, for graph precapture. Empty when fixed.
    pub fn adaptive_depths(&self, max_n: usize) -> Vec<usize> {
        if self
            .adaptive
            .lock()
            .expect("dflash adaptive poisoned")
            .is_none()
        {
            return Vec::new();
        }
        Self::adaptive_tiers(max_n)
    }

    /// Feeds this step's per-sequence accepted draft counts into the depth controller.
    pub fn adaptive_observe(&self, accepted: &[usize], max_n: usize, batch: usize) {
        let mut guard = self.adaptive.lock().expect("dflash adaptive poisoned");
        let Some(state) = guard.as_mut() else {
            return;
        };
        if batch > ADAPT_MAX_BATCH {
            if state.current != max_n {
                tracing::info!(
                    "DFlash adaptive draft depth {} -> {max_n} (batch {batch})",
                    state.current
                );
                state.current = max_n;
                state.cooldown = ADAPT_COOLDOWN;
            }
            state.streak = 0;
            return;
        }
        if accepted.is_empty() {
            return;
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = accepted.iter().sum::<usize>() as f32 / accepted.len() as f32;
        state.seen += 1;
        state.ema = if state.seen == 1 {
            mean
        } else {
            ADAPT_EMA_ALPHA * mean + (1.0 - ADAPT_EMA_ALPHA) * state.ema
        };
        if state.seen < ADAPT_WARMUP {
            return;
        }
        if state.cooldown > 0 {
            state.cooldown -= 1;
            return;
        }
        let tiers = Self::adaptive_tiers(max_n);
        let Some(idx) = tiers.iter().position(|t| *t == state.current) else {
            return;
        };
        #[allow(clippy::cast_precision_loss)]
        let desired = if idx + 1 < tiers.len()
            && state.ema >= state.current as f32 - ADAPT_UP_MARGIN
        {
            tiers[idx + 1]
        } else if idx > 0 && state.ema < state.current as f32 * ADAPT_TIER_FRAC - ADAPT_HYSTERESIS {
            tiers[idx - 1]
        } else {
            state.current
        };
        if desired == state.current {
            state.streak = 0;
            return;
        }
        if state.pending == desired {
            state.streak += 1;
        } else {
            state.pending = desired;
            state.streak = 1;
        }
        if state.streak >= ADAPT_STICKY {
            tracing::info!(
                "DFlash adaptive draft depth {} -> {desired} (accept EMA {:.2})",
                state.current,
                state.ema
            );
            state.current = desired;
            state.streak = 0;
            state.cooldown = ADAPT_COOLDOWN;
        }
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
        self.ctx_cache
            .lock()
            .expect("dflash cache poisoned")
            .get(&seq_id)
            .map(|c| c.next_pos)
    }

    pub fn retain_seqs(&self, seq_ids: &[usize]) {
        self.ctx_cache
            .lock()
            .expect("dflash cache poisoned")
            .retain(|id, _| seq_ids.contains(id));
    }

    pub fn clear(&self) {
        self.ctx_cache
            .lock()
            .expect("dflash cache poisoned")
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

        let mut ks = Vec::with_capacity(self.layers.len());
        let mut vs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let k = Self::heads_first(&layer.k_norm.forward(
                &layer.k_proj.forward(&ctx_hidden)?.reshape((
                    1,
                    total,
                    self.num_kv_heads,
                    self.head_dim,
                ))?,
            )?)?;
            let k = self.rope(&k, &cos, &sin)?;
            let v = self.split_heads(&layer.v_proj.forward(&ctx_hidden)?, self.num_kv_heads)?;
            ks.push(k.squeeze(0)?);
            vs.push(v.squeeze(0)?);
        }
        // [layers, kv_heads, total, head_dim]
        let k_all = Tensor::stack(&ks, 0)?;
        let v_all = Tensor::stack(&vs, 0)?;

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

        let mut hs = noise_embedding.to_dtype(self.dtype)?;
        for (i, layer) in self.layers.iter().enumerate() {
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
            let q = Self::heads_first(
                &layer.q_norm.forward(&layer.q_proj.forward(&x)?.reshape((
                    b,
                    block,
                    self.num_heads,
                    self.head_dim,
                ))?)?,
            )?;
            let k_noise = Self::heads_first(
                &layer.k_norm.forward(&layer.k_proj.forward(&x)?.reshape((
                    b,
                    block,
                    self.num_kv_heads,
                    self.head_dim,
                ))?)?,
            )?;
            // candle's fused rope kernel takes the batched [b, block, d/2] tables directly
            let q = candle_nn::rotary_emb::rope(&q, &q_cos, &q_sin)?;
            let k_noise = candle_nn::rotary_emb::rope(&k_noise, &q_cos, &q_sin)?;
            let v_noise = self.split_heads(&layer.v_proj.forward(&x)?, self.num_kv_heads)?;
            let k = Tensor::cat(&[ctx_k.narrow(0, i, 1)?.squeeze(0)?, k_noise], 2)?;
            let v = Tensor::cat(&[ctx_v.narrow(0, i, 1)?.squeeze(0)?, v_noise], 2)?;
            let mask = &kind_masks[&(layer.is_causal, layer.sliding_window)];
            // GQA via repeat: tiny shapes, plain matmul attention
            let k = repeat_kv(&k, groups)?;
            let v = repeat_kv(&v, groups)?;
            let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let att = att.broadcast_add(mask)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            let out = att.matmul(&v)?;
            let out = out
                .transpose(1, 2)?
                .reshape((b, block, self.num_heads * self.head_dim))?;
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
            let gate = layer.gate_proj.forward(&x)?;
            let up = layer.up_proj.forward(&x)?;
            let mut out = layer.down_proj.forward(&crate::ops::mul_and_act(
                &gate,
                &up,
                crate::layers::Activation::Silu,
            )?)?;
            if let (Some(conv), Some(kernel)) = (&layer.mlp_conv, mlp_kernel) {
                out = conv.finish(&out, &kernel)?;
            }
            hs = (residual + out)?;
        }

        let hs = self.norm.forward(&hs)?;
        // Positions 1.. predict the drafts; position 0 carries the anchor
        hs.narrow(1, 1, block - 1)
    }

    /// One lm_head projection and one D2H finish every sequence's drafts: `hidden` is the
    /// `[batch, n, hidden]` output of `draft_hidden_batch`. Returns per-seq
    /// (tokens, logits `[n, vocab]`).
    pub fn finish_drafts(
        &self,
        hidden: &Tensor,
        anchors: &[u32],
        lm_head: &Arc<dyn QuantMethod>,
        greedy: bool,
    ) -> Result<Vec<(Vec<u32>, Tensor)>> {
        let mut logits = lm_head.forward(hidden)?; // [batch, n, vocab]
        if (self.output_multiplier - 1.0).abs() > f64::EPSILON {
            logits = (logits * self.output_multiplier)?;
        }
        let use_selector = std::env::var("MISTRALRS_DFLASH_NO_SELECTOR").is_err();
        let tokens_per_seq = match (&self.selector, greedy && use_selector) {
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
