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
    fn select_greedy(&self, hidden: &Tensor, logits: &Tensor, anchor_id: u32) -> Result<Vec<u32>> {
        let (positions, _vocab) = logits.dims2()?;
        let k = self.top_k;
        let logits = logits.to_dtype(DType::F32)?.contiguous()?;
        let (unary, candidates) = topk_rows(&logits, k)?;
        // H(h): [positions, rank]
        let hproj = hidden
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.hidden_projection.t()?.to_dtype(DType::F32)?)?;
        let cand_flat = candidates.iter().flatten().copied().collect::<Vec<u32>>();
        let cand_ids = Tensor::from_vec(cand_flat.clone(), (positions * k,), logits.device())?;
        let succ = self
            .successor_codebook
            .to_dtype(DType::F32)?
            .index_select(&cand_ids, 0)?; // [positions*k, rank]
                                          // Predecessors can only be the anchor or one of the candidates
        let mut pred_ids = vec![anchor_id];
        pred_ids.extend_from_slice(&cand_flat);
        let pred_ids_t = Tensor::from_vec(pred_ids.clone(), (pred_ids.len(),), logits.device())?;
        let pred = self
            .predecessor_codebook
            .to_dtype(DType::F32)?
            .index_select(&pred_ids_t, 0)?;
        // One D2H for everything; the walk itself is tiny and host-side
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
        let (h_len, s_len) = (positions * rank, positions * k * rank);
        let hproj: Vec<&[f32]> = packed[..h_len].chunks(rank).collect();
        let succ: Vec<&[f32]> = packed[h_len..h_len + s_len].chunks(rank).collect();
        let pred: Vec<&[f32]> = packed[h_len + s_len..].chunks(rank).collect();

        let mut path = Vec::with_capacity(positions);
        let mut pred_row = 0usize;
        for pos in 0..positions {
            let mut best = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            for cand in 0..k {
                let mut dot = 0f32;
                let s = &succ[pos * k + cand];
                let p = &pred[pred_row];
                let h = &hproj[pos];
                for r in 0..rank {
                    dot += p[r] * h[r] * s[r];
                }
                let score = unary[pos][cand] + dot;
                if score > best {
                    best = score;
                    best_idx = cand;
                }
            }
            path.push(candidates[pos][best_idx]);
            pred_row = 1 + pos * k + best_idx;
        }
        Ok(path)
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

/// Per-sequence accumulated context keys/values, one entry per layer, `[1, kv_heads, len, head_dim]`
/// with rotary already applied to the keys. `next_pos` is the absolute position of the next token
/// to append; `start_pos` the absolute position of the first cached entry (after window trimming).
struct SeqCtxCache {
    layers: Vec<(Tensor, Tensor)>,
    start_pos: usize,
    next_pos: usize,
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
        })
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

    /// Projects tap features and appends per-layer context keys/values for `taps` (`[1, rows,
    /// taps*hidden]`) at absolute positions `start_pos..start_pos + rows`.
    pub fn append_ctx(&self, seq_id: usize, taps: &Tensor, start_pos: usize) -> Result<()> {
        let rows = taps.dim(1)?;
        if rows == 0 {
            return Ok(());
        }
        let ctx_hidden = self
            .hidden_norm
            .forward(&self.fc.forward(&taps.to_dtype(self.dtype)?)?)?;
        let (cos, sin) = self.cos_sin(start_pos, rows)?;

        let mut cache = self.ctx_cache.lock().expect("dflash cache poisoned");
        let entry = cache.entry(seq_id).or_insert_with(|| SeqCtxCache {
            layers: Vec::new(),
            start_pos,
            next_pos: start_pos,
        });
        if entry.next_pos != start_pos {
            candle_core::bail!(
                "DFlash context append at position {start_pos} but cache expects {}",
                entry.next_pos
            );
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let k = Self::heads_first(&layer.k_norm.forward(
                &layer.k_proj.forward(&ctx_hidden)?.reshape((
                    1,
                    rows,
                    self.num_kv_heads,
                    self.head_dim,
                ))?,
            )?)?;
            let k = self.rope(&k, &cos, &sin)?;
            let v = self.split_heads(&layer.v_proj.forward(&ctx_hidden)?, self.num_kv_heads)?;
            if let Some((ck, cv)) = entry.layers.get_mut(i) {
                *ck = Tensor::cat(&[&*ck, &k], 2)?;
                *cv = Tensor::cat(&[&*cv, &v], 2)?;
            } else {
                entry.layers.push((k, v));
            }
        }
        entry.next_pos = start_pos + rows;
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
        if keep != usize::MAX {
            let len = entry.next_pos - entry.start_pos;
            if len > keep {
                let drop = len - keep;
                for (k, v) in entry.layers.iter_mut() {
                    *k = k.narrow(2, drop, keep)?;
                    *v = v.narrow(2, drop, keep)?;
                }
                entry.start_pos += drop;
            }
        }
        Ok(())
    }

    /// One block draft: noise `[anchor, mask * n_drafts]` at absolute positions `start_pos..`,
    /// attending to the accumulated context. Returns the drafted tokens and the full draft logits
    /// (`[n_drafts, vocab]`) for stochastic verification.
    pub fn draft(
        &self,
        seq_id: usize,
        noise_embedding: &Tensor,
        start_pos: usize,
        lm_head: &Arc<dyn QuantMethod>,
        greedy: bool,
        anchor_id: u32,
    ) -> Result<(Vec<u32>, Tensor)> {
        let block = noise_embedding.dim(1)?;
        let cache = self.ctx_cache.lock().expect("dflash cache poisoned");
        let entry = cache.get(&seq_id).ok_or_else(|| {
            candle_core::Error::msg("DFlash draft requested for a sequence with no context cache")
        })?;
        if entry.next_pos != start_pos {
            candle_core::bail!(
                "DFlash draft at position {start_pos} but context ends at {}",
                entry.next_pos
            );
        }
        let ctx_start = entry.start_pos;
        let ctx_len = entry.next_pos - entry.start_pos;

        let (q_cos, q_sin) = self.cos_sin(start_pos, block)?;
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
                    1,
                    block,
                    self.num_heads,
                    self.head_dim,
                ))?)?,
            )?;
            let q = self.rope(&q, &q_cos, &q_sin)?;
            let k_noise = Self::heads_first(
                &layer.k_norm.forward(&layer.k_proj.forward(&x)?.reshape((
                    1,
                    block,
                    self.num_kv_heads,
                    self.head_dim,
                ))?)?,
            )?;
            let k_noise = self.rope(&k_noise, &q_cos, &q_sin)?;
            let v_noise = self.split_heads(&layer.v_proj.forward(&x)?, self.num_kv_heads)?;
            let (ctx_k, ctx_v) = &entry.layers[i];
            let k = Tensor::cat(&[ctx_k, &k_noise], 2)?;
            let v = Tensor::cat(&[ctx_v, &v_noise], 2)?;
            let mask = self.attention_mask(layer, start_pos, block, ctx_start, ctx_len)?;
            // GQA via repeat: tiny shapes, plain matmul attention
            let k = repeat_kv(&k, groups)?;
            let v = repeat_kv(&v, groups)?;
            let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let att = att.broadcast_add(&mask)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            let out = att.matmul(&v)?;
            let out = out
                .transpose(1, 2)?
                .reshape((1, block, self.num_heads * self.head_dim))?;
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
        drop(cache);

        let hs = self.norm.forward(&hs)?;
        // Positions 1.. predict the drafts; position 0 carries the anchor
        let draft_hidden = hs.narrow(1, 1, block - 1)?;
        let mut logits = lm_head.forward(&draft_hidden)?.squeeze(0)?;
        if (self.output_multiplier - 1.0).abs() > f64::EPSILON {
            logits = (logits * self.output_multiplier)?;
        }

        let use_selector = std::env::var("MISTRALRS_DFLASH_NO_SELECTOR").is_err();
        let tokens = match (&self.selector, greedy && use_selector) {
            (Some(selector), true) => {
                selector.select_greedy(&draft_hidden.squeeze(0)?, &logits, anchor_id)?
            }
            _ => logits.argmax(D::Minus1)?.to_vec1::<u32>()?,
        };
        Ok((tokens, logits))
    }

    fn attention_mask(
        &self,
        layer: &DFlashLayer,
        start_pos: usize,
        block: usize,
        ctx_start: usize,
        ctx_len: usize,
    ) -> Result<Tensor> {
        // Visibility depends only on relative geometry (the context always ends right before the
        // block), so one cached mask per (layer kind, ctx_len, block) covers every step.
        debug_assert_eq!(ctx_start + ctx_len, start_pos);
        let key = (layer.is_causal, layer.sliding_window, ctx_len, block);
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
                if layer.is_causal {
                    visible &= kp <= qp;
                }
                if let Some(w) = layer.sliding_window {
                    visible &= qp.saturating_sub(kp) < w;
                    if !layer.is_causal {
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
