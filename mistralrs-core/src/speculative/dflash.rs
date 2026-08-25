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

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
use crate::paged_attention::windowed_pool::{
    WindowedKvBatch, WindowedKvBatchTensors, WindowedKvCheckpoint, WindowedKvPool,
    WindowedKvPoolConfig, WindowedKvQuery,
};
#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
use crate::pipeline::cuda_graph::{
    record_cuda_graph_dispatch, record_cuda_graph_evictions, record_cuda_graph_resident_entries,
    take_cuda_graph_capacity_eviction, CudaGraphComponent, CudaGraphDispatchMode,
    CudaGraphDispatchReason, CudaGraphEvent, CudaGraphEventGuard, CudaGraphEvictionReason,
};

const DEFAULT_BLOCK_SIZE: usize = 16;
pub const DEFAULT_MAX_DRAFTS: usize = 7;
// Eager forwards use this cache; CUDA graphs derive RoPE from their replayed position inputs.
const ROPE_CACHE_LEN: usize = 65536;
const MASK_CACHE_CAP: usize = 64;
#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
const DFLASH_CUDA_GRAPH_CACHE_CAPACITY: usize = 32;
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

#[cfg(any(
    all(feature = "cuda", feature = "flash-attn", target_family = "unix"),
    test
))]
fn dflash_graph_positions_fit(start_positions: &[usize], block: usize) -> bool {
    block > 0
        && start_positions.iter().all(|start| {
            start
                .checked_add(block - 1)
                .is_some_and(|last| u32::try_from(last).is_ok())
        })
}

#[cfg(any(
    all(feature = "cuda", feature = "flash-attn", target_family = "unix"),
    test
))]
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

#[cfg(any(
    all(feature = "cuda", feature = "flash-attn", target_family = "unix"),
    test
))]
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

#[cfg(any(
    all(feature = "cuda", feature = "flash-attn", target_family = "unix"),
    test
))]
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
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
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

pub struct DFlashLoadTarget<'a> {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub yarn_rope_config: Option<&'a YarnRopeConfig>,
    pub device: &'a Device,
    pub dtype: DType,
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

    fn validate_rope_type(&self) -> Result<()> {
        let Some(parameters) = self.rope_parameters.as_ref() else {
            return Ok(());
        };
        let parameters = parameters.as_object().ok_or_else(|| {
            candle_core::Error::msg("DFlash rope_parameters must be a JSON object")
        })?;
        let rope_type = parameters
            .get("rope_type")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    candle_core::Error::msg("DFlash rope_parameters.rope_type must be a string")
                })
            })
            .transpose()?;
        let legacy_type = parameters
            .get("type")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    candle_core::Error::msg("DFlash rope_parameters.type must be a string")
                })
            })
            .transpose()?;
        if let (Some(rope_type), Some(legacy_type)) = (rope_type, legacy_type) {
            if rope_type != legacy_type {
                candle_core::bail!(
                    "DFlash rope_parameters.rope_type `{rope_type}` conflicts with legacy type `{legacy_type}`"
                );
            }
        }
        let rope_type = rope_type.or(legacy_type).unwrap_or("default");
        if rope_type != "default" {
            candle_core::bail!(
                "DFlash draft-side RoPE type `{rope_type}` is unsupported; configure RoPE scaling on the target model"
            );
        }
        Ok(())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn yarn_rope_config(&self, target: Option<&YarnRopeConfig>) -> Result<Option<YarnRopeConfig>> {
        let Some(target) = target else {
            return Ok(None);
        };
        let max_position_embeddings = self.max_position_embeddings.ok_or_else(|| {
            candle_core::Error::msg(
                "DFlash max_position_embeddings is required when the target uses YaRN",
            )
        })?;
        if max_position_embeddings != target.original_max_position_embeddings {
            candle_core::bail!(
                "DFlash native context length {} does not match target YaRN original context length {}",
                max_position_embeddings,
                target.original_max_position_embeddings
            );
        }
        let mut config = target.clone();
        config.base = self.rope_theta() as f32;
        config.head_dim = self.head_dim();
        Ok(Some(config))
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
        #[cfg(feature = "cuda")]
        if hidden.device().is_cuda()
            && self.kernel_size <= crate::cuda::dynamic_conv::MAX_DYNAMIC_CONV_KERNEL_SIZE
        {
            return crate::cuda::dynamic_conv::dynamic_conv(
                hidden,
                dynamic,
                base,
                self.kernel_size,
                self.group_size,
            );
        }
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
        let pre = self.convolve(hidden, &dynamic.i((.., .., 0))?, &self.base_kernel.i(0)?)?;
        Ok((pre, dynamic.i((.., .., 1))?))
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
    #[cfg(feature = "cuda")]
    vocab_size: usize,
}

fn resolve_dflash_sampling_policy(
    method: MtpDraftSamplingMethod,
    capability: std::result::Result<(), String>,
) -> Result<MtpDraftSamplingMethod> {
    match method {
        MtpDraftSamplingMethod::Auto => Ok(if capability.is_ok() {
            MtpDraftSamplingMethod::Probabilistic
        } else {
            MtpDraftSamplingMethod::Greedy
        }),
        MtpDraftSamplingMethod::Greedy => Ok(MtpDraftSamplingMethod::Greedy),
        MtpDraftSamplingMethod::Probabilistic => capability
            .map(|()| MtpDraftSamplingMethod::Probabilistic)
            .map_err(|reason| {
                candle_core::Error::msg(format!(
                    "probabilistic DFlash drafting is unavailable: {reason}"
                ))
            }),
    }
}

#[cfg(feature = "cuda")]
struct CandidateSelectorCudaSpec {
    device_is_cuda: bool,
    logits_dtype: DType,
    hidden_projection_dtype: DType,
    predecessor_dtype: DType,
    successor_dtype: DType,
    top_k: usize,
    vocab_size: usize,
    predecessor_vocab_size: Option<usize>,
    successor_vocab_size: Option<usize>,
}

#[cfg(feature = "cuda")]
fn validate_candidate_selector_cuda(
    spec: CandidateSelectorCudaSpec,
) -> std::result::Result<(), String> {
    if !spec.device_is_cuda {
        return Err("the draft model is not on CUDA".to_string());
    }
    if !matches!(spec.logits_dtype, DType::BF16 | DType::F16 | DType::F32) {
        return Err(format!(
            "CUDA ranked top-k does not support {:?} logits",
            spec.logits_dtype
        ));
    }
    for (name, dtype) in [
        ("hidden projection", spec.hidden_projection_dtype),
        ("predecessor codebook", spec.predecessor_dtype),
        ("successor codebook", spec.successor_dtype),
    ] {
        if !matches!(dtype, DType::BF16 | DType::F32) {
            return Err(format!(
                "CUDA candidate selection does not support {dtype:?} {name}"
            ));
        }
    }
    if spec.vocab_size == 0 {
        return Err("the vocabulary is empty".to_string());
    }
    if spec.vocab_size > crate::ops::CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        return Err(format!(
            "vocabulary size {} cannot be represented exactly by packed F32 indices",
            spec.vocab_size
        ));
    }
    for (name, codebook_vocab_size) in [
        ("predecessor", spec.predecessor_vocab_size),
        ("successor", spec.successor_vocab_size),
    ] {
        if codebook_vocab_size != Some(spec.vocab_size) {
            return Err(format!(
                "{name} codebook vocabulary {codebook_vocab_size:?} does not match logits vocabulary {}",
                spec.vocab_size
            ));
        }
    }
    let max_top_k = crate::ops::cuda_topk_ranked_packed_max_k(spec.vocab_size)
        .expect("nonempty representable vocabulary checked above")
        .min(crate::ops::CUDA_DFLASH_SELECTOR_MAX_K);
    if spec.top_k == 0 || spec.top_k > max_top_k {
        return Err(format!(
            "selector top_k={} must be in [1, {max_top_k}] for vocabulary {}",
            spec.top_k, spec.vocab_size
        ));
    }
    Ok(())
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
            #[cfg(feature = "cuda")]
            vocab_size: cfg.vocab_size,
        })
    }

    #[cfg(feature = "cuda")]
    fn cuda_capability(
        &self,
        device_is_cuda: bool,
        logits_dtype: DType,
        vocab_size: usize,
    ) -> std::result::Result<(), String> {
        if vocab_size != self.vocab_size {
            return Err(format!(
                "logits vocabulary {vocab_size} does not match configured vocabulary {}",
                self.vocab_size
            ));
        }
        validate_candidate_selector_cuda(CandidateSelectorCudaSpec {
            device_is_cuda,
            logits_dtype,
            hidden_projection_dtype: self.hidden_projection.dtype(),
            predecessor_dtype: self.predecessor_codebook.dtype(),
            successor_dtype: self.successor_codebook.dtype(),
            top_k: self.top_k,
            vocab_size,
            predecessor_vocab_size: self.predecessor_codebook.dims().first().copied(),
            successor_vocab_size: self.successor_codebook.dims().first().copied(),
        })
    }

    #[cfg(feature = "cuda")]
    fn configured_cuda_capability(
        &self,
        device_is_cuda: bool,
        logits_dtype: DType,
    ) -> std::result::Result<(), String> {
        self.cuda_capability(device_is_cuda, logits_dtype, self.vocab_size)
    }

    #[cfg(feature = "cuda")]
    fn cuda_candidates(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
    ) -> Result<(crate::ops::RankedTopKPackedOutput, Tensor)> {
        let (batch, positions, vocab) = logits.dims3()?;
        let rows = batch * positions;
        self.cuda_capability(logits.device().is_cuda(), logits.dtype(), vocab)
            .map_err(candle_core::Error::msg)?;
        let logits = logits.reshape((rows, vocab))?.contiguous()?;
        let topk = crate::ops::cuda_topk_ranked_packed_batched(&logits, self.top_k)?;
        let projection_dtype = self.hidden_projection.dtype();
        let projected = hidden
            .reshape((rows, ()))?
            .to_dtype(projection_dtype)?
            .broadcast_matmul(&self.hidden_projection.t()?)?
            .contiguous()?;
        Ok((topk, projected))
    }

    #[cfg(feature = "cuda")]
    fn select_greedy_cuda(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
        anchors: &Tensor,
    ) -> Result<Tensor> {
        let (topk, projected) = self.cuda_candidates(hidden, logits)?;
        crate::ops::cuda_dflash_greedy_select(
            &topk,
            &projected,
            &self.predecessor_codebook.contiguous()?,
            &self.successor_codebook.contiguous()?,
            anchors,
        )
    }

    #[cfg(feature = "cuda")]
    fn select_sample_cuda(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
        anchors: &Tensor,
        inverse_temperatures: &Tensor,
        uniforms: &Tensor,
    ) -> Result<crate::ops::DFlashSelectorSampleOutput> {
        let (topk, projected) = self.cuda_candidates(hidden, logits)?;
        crate::ops::cuda_dflash_sample_select(crate::ops::DFlashSelectorSampleInput {
            topk: &topk,
            projected_hidden: &projected,
            predecessor_codebook: &self.predecessor_codebook.contiguous()?,
            successor_codebook: &self.successor_codebook.contiguous()?,
            anchors,
            inverse_temperatures,
            uniforms,
        })
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
        if self
            .cuda_capability(logits.device().is_cuda(), logits.dtype(), vocab)
            .is_ok()
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

#[derive(Clone, Copy, Eq, PartialEq)]
enum DraftAttentionLayout {
    HeadsFirst,
    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
    #[cfg(feature = "cuda")]
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

#[cfg(any(
    all(feature = "cuda", feature = "flash-attn", target_family = "unix"),
    test
))]
struct DFlashGraphHostRows {
    token_ids: Vec<u32>,
    rope_indices: Vec<u32>,
    anchors: Vec<u32>,
    selector_inverse_temperatures: Option<Vec<f32>>,
    selector_uniforms: Option<Vec<f32>>,
}

#[cfg(any(
    all(feature = "cuda", feature = "flash-attn", target_family = "unix"),
    test
))]
#[derive(Clone, Copy)]
struct DFlashGraphHostInput<'a> {
    anchors: &'a [u32],
    start_positions: &'a [usize],
    mask_token_id: u32,
    block: usize,
    batch_bucket: usize,
    sampling: Option<DFlashSamplingInputs<'a>>,
}

#[cfg(any(
    all(feature = "cuda", feature = "flash-attn", target_family = "unix"),
    test
))]
impl DFlashGraphHostRows {
    fn update(&mut self, input: DFlashGraphHostInput<'_>) -> Result<()> {
        let DFlashGraphHostInput {
            anchors,
            start_positions,
            mask_token_id,
            block,
            batch_bucket,
            sampling,
        } = input;
        if anchors.is_empty() || anchors.len() != start_positions.len() {
            candle_core::bail!("DFlash graph inputs must contain matching non-empty rows");
        }
        if block == 0 || batch_bucket < anchors.len() {
            candle_core::bail!("DFlash graph input shape is invalid");
        }
        if let Some(sampling) = sampling {
            let expected_uniforms = anchors
                .len()
                .checked_mul(block - 1)
                .ok_or_else(|| candle_core::Error::msg("DFlash sampling input size overflow"))?;
            if sampling.inverse_temperatures.len() != anchors.len()
                || sampling.uniforms.len() != expected_uniforms
            {
                candle_core::bail!("DFlash sampling inputs do not match the graph rows");
            }
        }

        self.token_ids.clear();
        self.rope_indices.clear();
        self.anchors.clear();
        self.token_ids.reserve(batch_bucket * block);
        self.rope_indices.reserve(batch_bucket * block);
        self.anchors.reserve(batch_bucket);
        for row in 0..batch_bucket {
            let source = row.min(anchors.len() - 1);
            let anchor = anchors[source];
            let start = start_positions[source];
            self.anchors.push(anchor);
            self.token_ids.push(anchor);
            self.token_ids
                .extend(std::iter::repeat_n(mask_token_id, block - 1));
            for offset in 0..block {
                let position = start
                    .checked_add(offset)
                    .ok_or_else(|| candle_core::Error::msg("DFlash graph position overflow"))?;
                self.rope_indices
                    .push(u32::try_from(position).map_err(candle_core::Error::wrap)?);
            }
        }

        match sampling {
            Some(sampling) => {
                let inverse_temperatures = self
                    .selector_inverse_temperatures
                    .get_or_insert_with(Vec::new);
                inverse_temperatures.clear();
                inverse_temperatures.resize(batch_bucket, 0.0);
                inverse_temperatures[..anchors.len()]
                    .copy_from_slice(sampling.inverse_temperatures);
                let uniforms = self.selector_uniforms.get_or_insert_with(Vec::new);
                uniforms.clear();
                uniforms.resize(batch_bucket * (block - 1), 0.0);
                uniforms[..sampling.uniforms.len()].copy_from_slice(sampling.uniforms);
            }
            None => {
                self.selector_inverse_temperatures = None;
                self.selector_uniforms = None;
            }
        }
        Ok(())
    }
}

#[cfg(any(
    all(feature = "cuda", feature = "flash-attn", target_family = "unix"),
    test
))]
fn dflash_graph_host_rows(input: DFlashGraphHostInput<'_>) -> Result<DFlashGraphHostRows> {
    let DFlashGraphHostInput {
        block,
        batch_bucket,
        sampling,
        ..
    } = input;
    let mut rows = DFlashGraphHostRows {
        token_ids: Vec::with_capacity(batch_bucket * block),
        rope_indices: Vec::with_capacity(batch_bucket * block),
        anchors: Vec::with_capacity(batch_bucket),
        selector_inverse_temperatures: sampling.map(|_| Vec::with_capacity(batch_bucket)),
        selector_uniforms: sampling.map(|_| Vec::with_capacity(batch_bucket * (block - 1))),
    };
    rows.update(input)?;
    Ok(rows)
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DFlashCudaGraphKey {
    batch_bucket: usize,
    block: usize,
    selector_mode: DFlashSelectorMode,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DFlashSelectorMode {
    Disabled,
    Greedy,
    Sampling,
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
    selector_inverse_temperatures: Option<Var>,
    selector_uniforms: Option<Var>,
    output_candidate_ids: Option<Var>,
    output_candidate_probs: Option<Var>,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
struct DFlashCudaGraphEntry {
    key: DFlashCudaGraphKey,
    staging: crate::pipeline::cuda_graph::CudaGraphHostStaging,
    buffers: DFlashCudaGraphBuffers,
    token_embedding: Arc<dyn QuantMethod>,
    lm_head: Arc<dyn QuantMethod>,
    mask_token_id: u32,
    host_rows: DFlashGraphHostRows,
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
    anchors: &'a [u32],
    start_positions: &'a [usize],
    sampling: Option<DFlashSamplingInputs<'a>>,
    attention_batch: &'a WindowedKvBatch,
    token_embedding: &'a Arc<dyn QuantMethod>,
    lm_head: &'a Arc<dyn QuantMethod>,
    real_batch: usize,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
struct DFlashGraphTemporarySequences<'a> {
    pool: &'a Mutex<WindowedKvPool>,
    seq_ids: Vec<usize>,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
impl<'a> DFlashGraphTemporarySequences<'a> {
    fn acquire(pool: &'a Mutex<WindowedKvPool>, count: usize) -> Result<Self> {
        let mut locked = pool.lock().expect("dflash windowed pool poisoned");
        let mut seq_ids = Vec::with_capacity(count);
        let mut candidate = usize::MAX;
        while seq_ids.len() < count {
            if locked.sequence(candidate).is_none() {
                if let Err(err) = locked.acquire_at(candidate, 0) {
                    for seq_id in &seq_ids {
                        locked.release(*seq_id);
                    }
                    return Err(err);
                }
                seq_ids.push(candidate);
            }
            candidate = candidate.checked_sub(1).ok_or_else(|| {
                candle_core::Error::msg("DFlash graph temporary sequence id space exhausted")
            })?;
        }
        Ok(Self { pool, seq_ids })
    }

    fn attention_batch(&self, batch: usize, block: usize) -> Result<WindowedKvBatch> {
        let queries = self.seq_ids[..batch]
            .iter()
            .map(|seq_id| WindowedKvQuery {
                seq_id: *seq_id,
                query_len: block,
            })
            .collect::<Vec<_>>();
        self.pool
            .lock()
            .expect("dflash windowed pool poisoned")
            .scratch_graph_batch(&queries, batch)
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
impl Drop for DFlashGraphTemporarySequences<'_> {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().expect("dflash windowed pool poisoned");
        for seq_id in &self.seq_ids {
            pool.release(*seq_id);
        }
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
struct DFlashCudaGraphOutput {
    tokens: Tensor,
    candidate_ids: Option<Tensor>,
    candidate_probs: Option<Tensor>,
}

#[cfg(any(
    all(feature = "cuda", feature = "flash-attn", target_family = "unix"),
    test
))]
fn copy_dflash_graph_output_rows(output: &Tensor, real_batch: usize) -> Result<Tensor> {
    output.narrow(0, 0, real_batch)?.copy()
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
impl DFlashCudaGraphOutput {
    fn finish(self, real_batch: usize) -> Result<DFlashProposalBatch> {
        let tokens = copy_dflash_graph_output_rows(&self.tokens, real_batch)?;
        match (self.candidate_ids, self.candidate_probs) {
            (Some(candidate_ids), Some(candidate_probs)) => Ok(DFlashProposalBatch::DeviceSparse {
                tokens,
                candidate_ids: copy_dflash_graph_output_rows(&candidate_ids, real_batch)?,
                candidate_probs: copy_dflash_graph_output_rows(&candidate_probs, real_batch)?,
            }),
            (None, None) => Ok(DFlashProposalBatch::DeviceTokens(tokens)),
            _ => candle_core::bail!("DFlash selector returned incomplete sparse probabilities"),
        }
    }
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
    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    windowed_pool: Option<Mutex<WindowedKvPool>>,
    dormant_seqs: Mutex<HashSet<usize>>,
    // cos/sin for positions 0..ROPE_CACHE_LEN, [len, head_dim/2]
    rope_table: (Tensor, Tensor),
    mask_cache: Mutex<HashMap<MaskKey, Tensor>>,
    adaptive: Mutex<Option<AdaptiveState>>,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
struct DFlashPagedPrefixState {
    checkpoint: WindowedKvCheckpoint,
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
impl PagedAuxiliaryPrefixState for DFlashPagedPrefixState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn bytes(&self) -> usize {
        self.checkpoint.bytes()
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
impl DFlashCudaGraphBuffers {
    fn new(
        key: DFlashCudaGraphKey,
        rows: &DFlashGraphHostRows,
        batch: &WindowedKvBatch,
        model: &DFlashDraftModel,
    ) -> Result<Self> {
        let device = &model.device;
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
        let (
            selector_inverse_temperatures,
            selector_uniforms,
            output_candidate_ids,
            output_candidate_probs,
        ) = if key.selector_mode == DFlashSelectorMode::Sampling {
            let selector = model
                .selector
                .as_ref()
                .expect("sampling graph key requires a selector");
            let inverse_temperatures = rows
                .selector_inverse_temperatures
                .as_ref()
                .expect("sampling graph rows require inverse temperatures");
            let uniforms = rows
                .selector_uniforms
                .as_ref()
                .expect("sampling graph rows require uniforms");
            (
                Some(Var::from_tensor(&Tensor::from_vec(
                    inverse_temperatures.clone(),
                    (key.batch_bucket,),
                    device,
                )?)?),
                Some(Var::from_tensor(&Tensor::from_vec(
                    uniforms.clone(),
                    (key.batch_bucket, key.block - 1),
                    device,
                )?)?),
                Some(Var::zeros(
                    (key.batch_bucket, key.block - 1, selector.top_k),
                    DType::U32,
                    device,
                )?),
                Some(Var::zeros(
                    (key.batch_bucket, key.block - 1, selector.top_k),
                    DType::F32,
                    device,
                )?),
            )
        } else {
            (None, None, None, None)
        };
        Ok(Self {
            token_ids,
            rope_indices,
            anchors,
            block_tables,
            slot_mapping,
            cumulative_kv_lens,
            cumulative_query_lens,
            output_tokens,
            selector_inverse_temperatures,
            selector_uniforms,
            output_candidate_ids,
            output_candidate_probs,
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
            )?;
            match (
                &rows.selector_inverse_temperatures,
                &rows.selector_uniforms,
                &self.selector_inverse_temperatures,
                &self.selector_uniforms,
            ) {
                (
                    Some(inverse_temperatures),
                    Some(uniforms),
                    Some(dst_temperatures),
                    Some(dst_uniforms),
                ) => {
                    staging.copy_from_f32_slice(
                        "dflash_selector_inverse_temperatures",
                        location,
                        inverse_temperatures,
                        dst_temperatures,
                    )?;
                    staging.copy_from_f32_slice(
                        "dflash_selector_uniforms",
                        location,
                        uniforms,
                        dst_uniforms,
                    )
                }
                (None, None, None, None) => Ok(()),
                _ => candle_core::bail!("DFlash graph selector sampling state changed"),
            }
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

    fn launch(&mut self, real_batch: usize) -> Result<DFlashProposalBatch> {
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::DFlash, CudaGraphEvent::Replay);
        self.graph.launch()?;
        self.staging.record_graph_complete()?;
        let output = DFlashCudaGraphOutput {
            tokens: self.buffers.output_tokens.as_detached_tensor(),
            candidate_ids: self
                .buffers
                .output_candidate_ids
                .as_ref()
                .map(Var::as_detached_tensor),
            candidate_probs: self
                .buffers
                .output_candidate_probs
                .as_ref()
                .map(Var::as_detached_tensor),
        }
        .finish(real_batch)?;
        graph_event.success();
        record_cuda_graph_dispatch(
            CudaGraphComponent::DFlash,
            CudaGraphDispatchMode::Replay,
            CudaGraphDispatchReason::CacheHit,
        );
        Ok(output)
    }

    fn replay(
        &mut self,
        anchors: &[u32],
        start_positions: &[usize],
        sampling: Option<DFlashSamplingInputs<'_>>,
        batch: &WindowedKvBatch,
        real_batch: usize,
    ) -> Result<DFlashProposalBatch> {
        self.host_rows.update(DFlashGraphHostInput {
            anchors,
            start_positions,
            mask_token_id: self.mask_token_id,
            block: self.key.block,
            batch_bucket: self.key.batch_bucket,
            sampling,
        })?;
        self.buffers
            .update(&self.host_rows, batch, &mut self.staging)?;
        self.staging.order_before_graph()?;
        self.launch(real_batch)
    }

    fn release(self) -> (Arc<CudaStream>, Result<()>) {
        let Self {
            key: _,
            staging,
            buffers,
            token_embedding,
            lm_head,
            mask_token_id: _,
            host_rows,
            graph,
        } = self;
        let release = release_dflash_cuda_graph_resources(graph, (staging, buffers, host_rows));
        drop((token_embedding, lm_head));
        release
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
fn release_dflash_cuda_graph_resources<T>(
    graph: crate::pipeline::cuda_graph::CudaGraphHandle,
    resources: T,
) -> (Arc<CudaStream>, Result<()>) {
    let stream = graph.stream().clone();
    let mut release_result = stream
        .synchronize()
        .map_err(candle_core::Error::wrap)
        .map_err(|err| err.context("DFlash CUDA graph entry release wait failed"));
    drop(resources);
    if let Err(err) = stream.context().check_err() {
        if release_result.is_ok() {
            release_result = Err(candle_core::Error::wrap(err)
                .context("DFlash CUDA graph entry storage release failed"));
        }
    }
    let storage_result = stream
        .synchronize()
        .map_err(candle_core::Error::wrap)
        .map_err(|err| err.context("DFlash CUDA graph entry storage release wait failed"));
    if release_result.is_ok() {
        release_result = storage_result;
    }
    drop(graph);
    (stream, release_result)
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
fn release_dflash_cuda_graphs(entries: Vec<DFlashCudaGraphEntry>) {
    let mut streams = Vec::new();
    for entry in entries {
        let (stream, release_result) = entry.release();
        if let Err(err) = release_result {
            tracing::warn!("Failed to release DFlash CUDA graph entry storage: {err:?}");
        }
        if !streams.iter().any(|known: &Arc<CudaStream>| {
            known.context().cu_device() == stream.context().cu_device()
        }) {
            streams.push(stream);
        }
    }
    for stream in streams {
        if let Err(err) = crate::pipeline::cuda_graph::trim_cuda_graph_memory(&stream) {
            tracing::warn!("Failed to trim released DFlash CUDA graph memory: {err:?}");
        }
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
fn release_dflash_cuda_graph(entry: DFlashCudaGraphEntry) {
    release_dflash_cuda_graphs(vec![entry]);
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
impl Drop for DFlashCudaGraphState {
    fn drop(&mut self) {
        let entries = std::mem::take(&mut self.entries);
        record_cuda_graph_resident_entries(CudaGraphComponent::DFlash, 0);
        release_dflash_cuda_graphs(entries);
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
impl DFlashCudaGraphState {
    fn evict_lru_for_memory_pressure(&mut self, max_entries: usize) -> usize {
        let entries = drain_dflash_lru_entries(&mut self.entries, max_entries);
        let evicted = entries.len();
        if evicted == 0 {
            return 0;
        }
        record_cuda_graph_resident_entries(CudaGraphComponent::DFlash, self.entries.len());
        release_dflash_cuda_graphs(entries);
        record_cuda_graph_evictions(
            CudaGraphComponent::DFlash,
            CudaGraphEvictionReason::MemoryPressure,
            evicted,
        );
        evicted
    }

    fn store(&mut self, entry: DFlashCudaGraphEntry) {
        if let Some(evicted) =
            take_cuda_graph_capacity_eviction(&mut self.entries, DFLASH_CUDA_GRAPH_CACHE_CAPACITY)
        {
            record_cuda_graph_evictions(
                CudaGraphComponent::DFlash,
                CudaGraphEvictionReason::Capacity,
                1,
            );
            release_dflash_cuda_graph(evicted);
        }
        self.warmed.remove(&entry.key);
        self.failed.remove(&entry.key);
        self.entries.push(entry);
        record_cuda_graph_resident_entries(CudaGraphComponent::DFlash, self.entries.len());
    }

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
        record_cuda_graph_resident_entries(CudaGraphComponent::DFlash, self.entries.len());
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
    ) -> Result<DFlashProposalBatch> {
        let buffers = DFlashCudaGraphBuffers::new(key, rows, attention_batch, model)?;
        let Device::Cuda(cuda_device) = &model.device else {
            candle_core::bail!("DFlash CUDA graph expected a CUDA device");
        };
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
        model
            .cuda_graph_output(key, &buffers, attention_batch, token_embedding, lm_head)?
            .finish(real_batch)
    }

    fn capture(
        model: &DFlashDraftModel,
        key: DFlashCudaGraphKey,
        host_rows: DFlashGraphHostRows,
        attention_batch: &WindowedKvBatch,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
    ) -> Result<DFlashCudaGraphEntry> {
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::DFlash, CudaGraphEvent::Capture);
        let buffers = DFlashCudaGraphBuffers::new(key, &host_rows, attention_batch, model)?;
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
            model.cuda_graph_output(key, &buffers, attention_batch, token_embedding, lm_head);
        match capture_result {
            Ok(output) => {
                let copy_result = (|| {
                    crate::cuda::graph::copy_tensor(
                        &output.tokens,
                        &buffers.output_tokens.as_detached_tensor(),
                    )?;
                    match (
                        output.candidate_ids,
                        output.candidate_probs,
                        &buffers.output_candidate_ids,
                        &buffers.output_candidate_probs,
                    ) {
                        (Some(ids), Some(probs), Some(dst_ids), Some(dst_probs)) => {
                            crate::cuda::graph::copy_tensor(&ids, &dst_ids.as_detached_tensor())?;
                            crate::cuda::graph::copy_tensor(&probs, &dst_probs.as_detached_tensor())
                        }
                        (None, None, None, None) => Ok(()),
                        _ => candle_core::bail!("DFlash graph selector output shape changed"),
                    }
                })();
                if let Err(err) = copy_result {
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
        let entry = DFlashCudaGraphEntry {
            key,
            staging,
            buffers,
            token_embedding: token_embedding.clone(),
            lm_head: lm_head.clone(),
            mask_token_id: model.mask_token_id,
            host_rows,
            graph,
        };
        graph_event.success();
        Ok(entry)
    }

    fn run(&mut self, run: DFlashCudaGraphRun<'_>) -> Result<Option<DFlashProposalBatch>> {
        let DFlashCudaGraphRun {
            model,
            key,
            anchors,
            start_positions,
            sampling,
            attention_batch,
            token_embedding,
            lm_head,
            real_batch,
        } = run;
        if let Some(position) = self.entries.iter().position(|entry| {
            entry.key == key && entry.matches_dependencies(token_embedding, lm_head)
        }) {
            let mut entry = self.entries.remove(position);
            match entry.replay(
                anchors,
                start_positions,
                sampling,
                attention_batch,
                real_batch,
            ) {
                Ok(tokens) => {
                    self.entries.push(entry);
                    return Ok(Some(tokens));
                }
                Err(err) => {
                    self.retire_failed_entry(entry, "replay", err)?;
                    let graph_event = CudaGraphEventGuard::new(
                        CudaGraphComponent::DFlash,
                        CudaGraphEvent::EagerFallback,
                    );
                    let result = Self::eager(
                        model,
                        key,
                        &dflash_graph_host_rows(DFlashGraphHostInput {
                            anchors,
                            start_positions,
                            mask_token_id: model.mask_token_id,
                            block: key.block,
                            batch_bucket: key.batch_bucket,
                            sampling,
                        })?,
                        attention_batch,
                        token_embedding,
                        lm_head,
                        real_batch,
                    )
                    .map(Some)
                    .map_err(|err| err.context("DFlash eager fallback after graph replay failed"));
                    if result.is_ok() {
                        graph_event.success();
                    }
                    return result;
                }
            }
        }
        let mismatched = self
            .entries
            .iter()
            .position(|entry| entry.key == key)
            .map(|position| self.entries.remove(position));
        if let Some(entry) = mismatched {
            record_cuda_graph_resident_entries(CudaGraphComponent::DFlash, self.entries.len());
            release_dflash_cuda_graph(entry);
        }
        if self.failed.contains(&key) {
            return Ok(None);
        }
        if self.warmed.insert(key) {
            let graph_event =
                CudaGraphEventGuard::new(CudaGraphComponent::DFlash, CudaGraphEvent::EagerFallback);
            let rows = dflash_graph_host_rows(DFlashGraphHostInput {
                anchors,
                start_positions,
                mask_token_id: model.mask_token_id,
                block: key.block,
                batch_bucket: key.batch_bucket,
                sampling,
            })?;
            let result = Self::eager(
                model,
                key,
                &rows,
                attention_batch,
                token_embedding,
                lm_head,
                real_batch,
            )
            .map(Some);
            if result.is_ok() {
                graph_event.success();
            }
            return result;
        }

        let rows = dflash_graph_host_rows(DFlashGraphHostInput {
            anchors,
            start_positions,
            mask_token_id: model.mask_token_id,
            block: key.block,
            batch_bucket: key.batch_bucket,
            sampling,
        })?;
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
                let graph_event = CudaGraphEventGuard::new(
                    CudaGraphComponent::DFlash,
                    CudaGraphEvent::EagerFallback,
                );
                let result = Self::eager(
                    model,
                    key,
                    &dflash_graph_host_rows(DFlashGraphHostInput {
                        anchors,
                        start_positions,
                        mask_token_id: model.mask_token_id,
                        block: key.block,
                        batch_bucket: key.batch_bucket,
                        sampling,
                    })?,
                    attention_batch,
                    token_embedding,
                    lm_head,
                    real_batch,
                )
                .map(Some)
                .map_err(|err| {
                    err.context("DFlash eager fallback after first graph launch failed")
                });
                if result.is_ok() {
                    graph_event.success();
                }
                return result;
            }
        };
        tracing::debug!(
            batch_bucket = key.batch_bucket,
            block = key.block,
            "Captured DFlash CUDA graph"
        );
        self.store(entry);
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
pub(crate) fn windowed_kv_checkpoint_capacity(retained_prefixes: usize) -> Result<usize> {
    if retained_prefixes == 0 {
        return Ok(0);
    }
    retained_prefixes
        .checked_add(1)
        .ok_or_else(|| candle_core::Error::msg("DFlash prefix checkpoint capacity overflow"))
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
        {
            self.cuda_graphs
                .lock()
                .expect("dflash CUDA graph cache poisoned")
                .evict_lru_for_memory_pressure(max_entries)
        }
        #[cfg(not(all(feature = "cuda", feature = "flash-attn", target_family = "unix")))]
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
            #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
            cuda_graphs: Mutex::new(DFlashCudaGraphState::default()),
            layers,
            fc,
            context_kv_proj,
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
            #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
        #[cfg(not(all(feature = "cuda", feature = "flash-attn", target_family = "unix")))]
        {
            let _ = (live_sequence_capacity, retained_prefixes);
            Ok(false)
        }
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
            crate::pipeline::cuda_graph::cuda_graph_precapture_batches(),
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
            if block > self.block_size || !dflash_graph_positions_fit(start_positions, block) {
                return Ok(None);
            }
            let Some(batch_bucket) =
                crate::pipeline::cuda_graph::cuda_graph_batch_bucket(seq_ids.len())
            else {
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
        #[cfg(not(all(feature = "cuda", feature = "flash-attn", target_family = "unix")))]
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

    pub fn prefix_replay(&self) -> SpeculativePrefixReplay {
        dflash_prefix_replay(self.layers.iter().map(|layer| layer.sliding_window))
    }

    pub fn supports_paged_auxiliary_prefix_state(&self) -> bool {
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
        {
            return self.windowed_pool.as_ref().is_some_and(|pool| {
                pool.lock()
                    .expect("dflash windowed pool poisoned")
                    .config()
                    .checkpoint_capacity()
                    > 0
            });
        }
        #[cfg(not(all(feature = "cuda", feature = "flash-attn", target_family = "unix")))]
        {
            false
        }
    }

    pub fn capture_paged_auxiliary_prefix_state(
        &self,
        sequence_id: usize,
        cached_tokens: usize,
    ) -> Result<Option<Arc<dyn PagedAuxiliaryPrefixState>>> {
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
            return Ok(Some(Arc::new(DFlashPagedPrefixState { checkpoint })));
        }
        #[cfg(not(all(feature = "cuda", feature = "flash-attn", target_family = "unix")))]
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
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
            return Ok(());
        }
        #[cfg(not(all(feature = "cuda", feature = "flash-attn", target_family = "unix")))]
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
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
                    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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
                #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    fn cuda_graph_output(
        &self,
        key: DFlashCudaGraphKey,
        buffers: &DFlashCudaGraphBuffers,
        attention_batch: &WindowedKvBatch,
        token_embedding: &Arc<dyn QuantMethod>,
        lm_head: &Arc<dyn QuantMethod>,
    ) -> Result<DFlashCudaGraphOutput> {
        let mut noise = token_embedding
            .embedding_forward(&buffers.token_ids.as_detached_tensor(), self.dtype)?;
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
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use candle_core::{Device, Result, Tensor, D};
    use mistralrs_quant::QuantMethod;

    use super::{
        contiguous_row_range, copy_dflash_graph_output_rows, dflash_adaptive_env_value,
        dflash_graph_host_rows, dflash_graph_plans, dflash_graph_positions_fit,
        dflash_graph_precapture_shapes, dflash_prefix_replay, dflash_rope_from_positions,
        drain_dflash_lru_entries, gather_ctx_taps, linear_from_weight,
        resolve_dflash_sampling_policy, select_ctx_kv_rows, select_dflash_depth,
        update_dormant_sequences, DFlashConfig, DFlashGraphHostInput, DFlashSamplingInputs,
        DFlashSequenceEviction, ADAPT_FULL_DEPTH_MAX_BATCH,
    };
    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    use super::{release_dflash_cuda_graph_resources, windowed_kv_checkpoint_capacity};
    #[cfg(feature = "cuda")]
    use super::{validate_candidate_selector_cuda, CandidateSelectorCudaSpec};
    use crate::layers::{yarn_inv_freq_and_attention_factor, YarnRopeConfig};
    use crate::speculative::MtpDraftSamplingMethod;
    use crate::speculative::{SpeculativeGraphPlan, SpeculativePrefixReplay};

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    #[test]
    fn prefix_checkpoint_capacity_includes_transactional_staging() -> Result<()> {
        assert_eq!(windowed_kv_checkpoint_capacity(0)?, 0);
        assert_eq!(windowed_kv_checkpoint_capacity(16)?, 17);
        assert!(windowed_kv_checkpoint_capacity(usize::MAX).is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn selector_cuda_spec(vocab_size: usize, top_k: usize) -> CandidateSelectorCudaSpec {
        CandidateSelectorCudaSpec {
            device_is_cuda: true,
            logits_dtype: candle_core::DType::BF16,
            hidden_projection_dtype: candle_core::DType::BF16,
            predecessor_dtype: candle_core::DType::BF16,
            successor_dtype: candle_core::DType::BF16,
            top_k,
            vocab_size,
            predecessor_vocab_size: Some(vocab_size),
            successor_vocab_size: Some(vocab_size),
        }
    }

    #[test]
    fn dflash_draft_sampling_resolves_once_at_load() {
        assert_eq!(
            resolve_dflash_sampling_policy(MtpDraftSamplingMethod::Auto, Ok(())).unwrap(),
            MtpDraftSamplingMethod::Probabilistic
        );
        assert_eq!(
            resolve_dflash_sampling_policy(
                MtpDraftSamplingMethod::Auto,
                Err("missing selector".to_string()),
            )
            .unwrap(),
            MtpDraftSamplingMethod::Greedy
        );
        assert_eq!(
            resolve_dflash_sampling_policy(
                MtpDraftSamplingMethod::Greedy,
                Err("unsupported dtype".to_string()),
            )
            .unwrap(),
            MtpDraftSamplingMethod::Greedy
        );
        assert_eq!(
            resolve_dflash_sampling_policy(MtpDraftSamplingMethod::Probabilistic, Ok(())).unwrap(),
            MtpDraftSamplingMethod::Probabilistic
        );

        let missing = resolve_dflash_sampling_policy(
            MtpDraftSamplingMethod::Probabilistic,
            Err("a DFlash2 checkpoint with a candidate selector is required".to_string()),
        )
        .expect_err("probabilistic drafting must not silently use greedy selection");
        assert!(missing
            .to_string()
            .contains("DFlash2 checkpoint with a candidate selector is required"));

        let unsupported = resolve_dflash_sampling_policy(
            MtpDraftSamplingMethod::Probabilistic,
            Err("unsupported dtype".to_string()),
        )
        .expect_err("probabilistic drafting must reject an unsupported CUDA selector");
        assert!(unsupported.to_string().contains("unsupported dtype"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_selector_top_k_respects_the_ranked_workspace_bound() {
        let vocab_size = 1_000_000;
        let max_top_k = crate::ops::cuda_topk_ranked_packed_max_k(vocab_size)
            .expect("test vocabulary is representable")
            .min(crate::ops::CUDA_DFLASH_SELECTOR_MAX_K);
        assert!(
            validate_candidate_selector_cuda(selector_cuda_spec(vocab_size, max_top_k)).is_ok()
        );
        let error = validate_candidate_selector_cuda(selector_cuda_spec(vocab_size, max_top_k + 1))
            .expect_err("top_k above the ranked workspace bound must be rejected");
        assert!(error.contains(&format!("[1, {max_top_k}]")));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_selector_rejects_unrepresentable_vocabularies() {
        let vocab_size = crate::ops::CUDA_TOPK_MAX_EXACT_PACKED_VOCAB + 1;
        let error = validate_candidate_selector_cuda(selector_cuda_spec(vocab_size, 1))
            .expect_err("packed F32 indices must represent every vocabulary id exactly");
        assert!(error.contains("cannot be represented exactly"));
    }

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

    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn context_tap_gather_preserves_flat_row_order() -> Result<()> {
        let device = Device::Cpu;
        let tap_a = Tensor::from_vec(
            (0..12)
                .flat_map(|row| [row as f32, 100.0 + row as f32])
                .collect::<Vec<_>>(),
            (3, 4, 2),
            &device,
        )?;
        let tap_b = Tensor::from_vec(
            (0..12).map(|row| -(row as f32)).collect::<Vec<_>>(),
            (3, 4, 1),
            &device,
        )?;
        let packed = gather_ctx_taps(&[tap_a, tap_b], vec![9, 10, 1, 6, 7], &device)?;
        assert_eq!(
            packed.to_vec3::<f32>()?,
            [vec![
                vec![9.0, 109.0, -9.0],
                vec![10.0, 110.0, -10.0],
                vec![1.0, 101.0, -1.0],
                vec![6.0, 106.0, -6.0],
                vec![7.0, 107.0, -7.0],
            ]]
        );
        Ok(())
    }

    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn context_tap_gather_handles_contiguous_rows_without_reordering() -> Result<()> {
        let device = Device::Cpu;
        let tap = Tensor::from_vec((0..12).map(|row| row as f32).collect(), (3, 4, 1), &device)?;
        let packed = gather_ctx_taps(&[tap], vec![4, 5, 6, 7], &device)?;
        assert_eq!(
            packed.to_vec3::<f32>()?,
            [vec![vec![4.0], vec![5.0], vec![6.0], vec![7.0]]]
        );
        Ok(())
    }

    #[test]
    fn prepared_context_row_selection_detects_only_ordered_ranges() {
        assert_eq!(contiguous_row_range(&[4, 5, 6, 7]), Some((4, 4)));
        assert_eq!(contiguous_row_range(&[4, 6, 7]), None);
        assert_eq!(contiguous_row_range(&[7, 6, 5]), None);
        assert_eq!(contiguous_row_range(&[]), None);
    }

    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn context_kv_row_selection_handles_narrowed_sources() -> Result<()> {
        let device = Device::Cpu;
        let source = Tensor::from_vec(
            (0..32).map(|value| value as f32).collect(),
            (2, 2, 4, 2),
            &device,
        )?;
        let narrowed = source.narrow(2, 1, 3)?;
        assert!(!narrowed.is_contiguous());
        let (k, v) = select_ctx_kv_rows(&narrowed, &narrowed, &[0, 2])?;
        assert_eq!(
            k.flatten_all()?.to_vec1::<f32>()?,
            v.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            k.flatten_all()?.to_vec1::<f32>()?,
            [
                2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0, 18.0, 19.0, 22.0, 23.0, 26.0, 27.0,
                30.0, 31.0,
            ]
        );
        assert_eq!(k.dims(), &[2, 2, 2, 2]);
        let (empty_k, empty_v) = select_ctx_kv_rows(&narrowed, &narrowed, &[])?;
        assert_eq!(empty_k.dims(), &[2, 2, 0, 2]);
        assert_eq!(empty_v.dims(), &[2, 2, 0, 2]);
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
    fn prefix_replay_covers_every_dflash_attention_window() {
        assert_eq!(
            dflash_prefix_replay([Some(1024), Some(2048), Some(512)]),
            SpeculativePrefixReplay::Suffix(2048)
        );
        assert_eq!(
            dflash_prefix_replay([Some(2048), None, Some(1024)]),
            SpeculativePrefixReplay::Full
        );
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
    fn graph_precapture_shapes_follow_depth_batch_limits_and_pool_capacity() {
        let batches = (1..=8).chain(std::iter::once(16)).collect::<Vec<_>>();
        assert_eq!(
            dflash_graph_precapture_shapes(&dflash_graph_plans(false, 7), batches.clone(), 16),
            vec![
                (1, 8),
                (2, 8),
                (3, 8),
                (4, 8),
                (5, 8),
                (6, 8),
                (7, 8),
                (8, 8),
                (16, 8),
            ]
        );
        assert_eq!(
            dflash_graph_precapture_shapes(&dflash_graph_plans(true, 7), batches, 8),
            vec![
                (1, 2),
                (1, 8),
                (2, 2),
                (2, 8),
                (3, 2),
                (4, 2),
                (5, 2),
                (6, 2),
                (7, 2),
                (8, 2),
            ]
        );
    }

    #[test]
    fn graph_pressure_eviction_drains_lru_entries_in_one_batch() {
        let mut entries = vec![10, 20, 30, 40];
        assert!(drain_dflash_lru_entries(&mut entries, 0).is_empty());
        assert_eq!(drain_dflash_lru_entries(&mut entries, 2), vec![10, 20]);
        assert_eq!(entries, vec![30, 40]);
        assert_eq!(
            drain_dflash_lru_entries(&mut entries, usize::MAX),
            vec![30, 40]
        );
        assert!(entries.is_empty());
    }

    #[test]
    fn graph_outputs_do_not_alias_reclaimable_buffers() -> Result<()> {
        use candle_core::Var;

        let source = Var::from_tensor(&Tensor::from_vec(
            vec![1u32, 2, 3, 4, 5, 6],
            (2, 3),
            &Device::Cpu,
        )?)?;
        let output = copy_dflash_graph_output_rows(&source.as_detached_tensor(), 1)?;
        source.set(&Tensor::from_vec(
            vec![11u32, 12, 13, 14, 15, 16],
            (2, 3),
            &Device::Cpu,
        )?)?;
        assert_eq!(output.to_vec2::<u32>()?, vec![vec![1, 2, 3]]);
        Ok(())
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_release_waits_for_detached_output_copies() -> anyhow::Result<()> {
        use candle_core::{cuda_backend::cudarc::driver::sys, Var};

        use crate::pipeline::cuda_graph::{
            disable_event_tracking_for_capture, prepare_cuda_graph_memory_pool,
            restore_event_tracking_after_capture, CudaGraphHandle, CudaGraphHostStaging,
        };

        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;
        let input = Var::from_tensor(&Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2, 3),
            &device,
        )?)?;
        let output = Var::zeros((2, 3), candle_core::DType::F32, &device)?;
        device.synchronize()?;

        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let captured = input.as_detached_tensor().affine(2.0, 1.0)?;
        crate::cuda::graph::copy_tensor(&captured, &output.as_detached_tensor())?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("DFlash graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;

        let mut staging = CudaGraphHostStaging::new(stream)?;
        graph.launch()?;
        staging.record_graph_complete()?;
        let detached = copy_dflash_graph_output_rows(&output.as_detached_tensor(), 1)?;
        let (_, release_result) =
            release_dflash_cuda_graph_resources(graph, (staging, input, output, captured));
        release_result?;

        assert_eq!(detached.to_vec2::<f32>()?, vec![vec![3.0, 5.0, 7.0]]);
        Ok(())
    }

    #[test]
    fn graph_positions_support_long_contexts_and_preserve_fallback_bounds() {
        assert!(dflash_graph_positions_fit(&[65_529, 100_000, 250_000], 8));
        assert!(dflash_graph_positions_fit(&[u32::MAX as usize - 7], 8));
        assert!(!dflash_graph_positions_fit(&[u32::MAX as usize - 6], 8));
        assert!(!dflash_graph_positions_fit(&[100_000], 0));
    }

    fn qwen35_dflash_config() -> DFlashConfig {
        serde_json::from_str(
            r#"{
                "architectures": ["DFlash2DraftModel"],
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_hidden_layers": 5,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "vocab_size": 248320,
                "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
                "num_target_layers": 64
            }"#,
        )
        .unwrap()
    }

    fn qwen35_target_yarn() -> YarnRopeConfig {
        YarnRopeConfig {
            base: 10_000_000.0,
            head_dim: 64,
            max_position_embeddings: 1_048_576,
            original_max_position_embeddings: 262_144,
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 0.0,
            attention_factor: None,
        }
    }

    #[test]
    fn dflash_yarn_uses_draft_geometry_and_target_scaling() -> Result<()> {
        let cfg = qwen35_dflash_config();
        let yarn = cfg
            .yarn_rope_config(Some(&qwen35_target_yarn()))?
            .expect("target uses YaRN");
        assert_eq!(yarn.base, 10_000_000.0);
        assert_eq!(yarn.head_dim, 128);
        assert_eq!(yarn.factor, 4.0);
        assert_eq!(yarn.original_max_position_embeddings, 262_144);

        let (inv_freq, attention_factor) = yarn_inv_freq_and_attention_factor(&yarn, &Device::Cpu)?;
        let inv_freq = inv_freq.to_vec1::<f32>()?;
        assert_eq!(inv_freq.len(), 64);
        assert!((inv_freq[0] - 1.0).abs() < 1e-6);
        let unscaled_last = 1.0 / yarn.base.powf(126.0 / 128.0);
        assert!((inv_freq[63] - unscaled_last / yarn.factor).abs() < 1e-8);
        assert!((attention_factor - (1.0 + 0.1 * yarn.factor.ln())).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn dflash_yarn_rejects_a_mismatched_native_context() {
        let cfg = qwen35_dflash_config();
        let mut target = qwen35_target_yarn();
        target.original_max_position_embeddings = 131_072;
        let error = cfg
            .yarn_rope_config(Some(&target))
            .expect_err("mismatched native context must not silently change draft RoPE");
        assert!(error.to_string().contains("native context length"));
    }

    #[test]
    fn dflash_native_rope_does_not_require_a_declared_context() -> Result<()> {
        let mut cfg = qwen35_dflash_config();
        cfg.max_position_embeddings = None;
        assert!(cfg.yarn_rope_config(None)?.is_none());
        let error = cfg
            .yarn_rope_config(Some(&qwen35_target_yarn()))
            .expect_err("YaRN must validate the draft's native context");
        assert!(error
            .to_string()
            .contains("max_position_embeddings is required"));
        Ok(())
    }

    #[test]
    fn dflash_rejects_draft_side_rope_scaling() {
        for key in ["rope_type", "type"] {
            let mut cfg = qwen35_dflash_config();
            cfg.rope_parameters = Some(serde_json::json!({
                "rope_theta": 10_000_000,
                (key): "yarn",
            }));
            let error = cfg
                .validate_rope_type()
                .expect_err("draft-side scaling must not be silently ignored");
            assert!(error
                .to_string()
                .contains("configure RoPE scaling on the target"));
        }
    }

    #[test]
    fn dflash_rejects_conflicting_rope_type_aliases() {
        let mut cfg = qwen35_dflash_config();
        cfg.rope_parameters = Some(serde_json::json!({
            "rope_type": "default",
            "type": "yarn",
        }));
        let error = cfg
            .validate_rope_type()
            .expect_err("conflicting aliases must not be accepted");
        assert!(error.to_string().contains("conflicts with legacy type"));
    }

    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn graph_rope_uses_each_replayed_long_position() -> Result<()> {
        let device = Device::Cpu;
        let positions = [65_535u32, 65_536, 100_000, 131_071];
        let frequencies = [1.0f32, 0.5, 0.01];
        let positions_tensor = Tensor::from_vec(positions.to_vec(), (positions.len(),), &device)?;
        let inv_freq = Tensor::from_vec(frequencies.to_vec(), (frequencies.len(),), &device)?;
        let (cos, sin) =
            dflash_rope_from_positions(&positions_tensor, &inv_freq, candle_core::DType::F32, 1.0)?;
        let cos = cos.to_vec2::<f32>()?;
        let sin = sin.to_vec2::<f32>()?;

        for (row, position) in positions.into_iter().enumerate() {
            for (column, frequency) in frequencies.into_iter().enumerate() {
                let angle = position as f32 * frequency;
                assert!((cos[row][column] - angle.cos()).abs() < 1e-5);
                assert!((sin[row][column] - angle.sin()).abs() < 1e-5);
            }
        }
        Ok(())
    }

    #[test]
    fn graph_rope_applies_yarn_attention_scaling() -> Result<()> {
        let positions = Tensor::from_vec(vec![0u32, 1], 2, &Device::Cpu)?;
        let inv_freq = Tensor::from_vec(vec![1.0f32, 0.25], 2, &Device::Cpu)?;
        let attention_factor = 1.25;
        let (cos, sin) = dflash_rope_from_positions(
            &positions,
            &inv_freq,
            candle_core::DType::F32,
            attention_factor,
        )?;
        let cos = cos.to_vec2::<f32>()?;
        let sin = sin.to_vec2::<f32>()?;
        assert_eq!(cos[0], vec![attention_factor, attention_factor]);
        assert_eq!(sin[0], vec![0.0, 0.0]);
        assert!((cos[1][0] - attention_factor * 1.0f32.cos()).abs() < 1e-6);
        assert!((sin[1][1] - attention_factor * 0.25f32.sin()).abs() < 1e-6);
        Ok(())
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_rope_replays_mixed_long_positions_on_cuda() -> anyhow::Result<()> {
        use candle_core::{cuda_backend::cudarc::driver::sys, Var};

        use crate::pipeline::cuda_graph::{
            disable_event_tracking_for_capture, prepare_cuda_graph_memory_pool,
            restore_event_tracking_after_capture, CudaGraphHandle,
        };

        const BATCH: usize = 16;
        const BLOCK: usize = 8;
        const HEAD_DIM: usize = 128;
        const ROPE_THETA: f32 = 10_000_000.0;
        const BENCH_WARMUP: usize = 100;
        const BENCH_ITERATIONS: usize = 10_000;

        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;
        let inv_freq_values = (0..HEAD_DIM)
            .step_by(2)
            .map(|index| 1.0 / ROPE_THETA.powf(index as f32 / HEAD_DIM as f32))
            .collect::<Vec<_>>();
        let replay_positions = |bases: [u32; BATCH]| {
            bases
                .into_iter()
                .flat_map(|base| (0..BLOCK).map(move |offset| base + offset as u32))
                .collect::<Vec<_>>()
        };
        let first_positions = replay_positions([
            65_532, 65_536, 70_000, 80_000, 90_000, 100_000, 110_000, 120_000, 130_000, 140_000,
            150_000, 160_000, 180_000, 200_000, 225_000, 250_000,
        ]);
        let positions = Var::from_tensor(&Tensor::from_vec(
            first_positions.clone(),
            BATCH * BLOCK,
            &device,
        )?)?;
        let inv_freq = Tensor::from_vec(inv_freq_values.clone(), HEAD_DIM / 2, &device)?;
        let warmup = dflash_rope_from_positions(
            &positions.as_detached_tensor(),
            &inv_freq,
            candle_core::DType::BF16,
            1.0,
        )?;
        device.synchronize()?;
        drop(warmup);

        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let (cos, sin) = dflash_rope_from_positions(
            &positions.as_detached_tensor(),
            &inv_freq,
            candle_core::DType::BF16,
            1.0,
        )?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;

        for replayed in [
            first_positions,
            replay_positions([
                100_003, 110_000, 120_000, 130_000, 131_072, 140_000, 150_000, 160_000, 175_000,
                190_000, 200_000, 210_000, 220_000, 230_000, 240_000, 250_000,
            ]),
        ] {
            positions.set(&Tensor::from_vec(replayed.clone(), BATCH * BLOCK, &device)?)?;
            graph.launch()?;
            stream.synchronize()?;
            let cos = cos.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
            let sin = sin.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
            for (row, position) in replayed.into_iter().enumerate() {
                for (column, frequency) in inv_freq_values.iter().copied().enumerate() {
                    let angle = position as f32 * frequency;
                    assert!((cos[row][column] - angle.cos()).abs() < 0.005);
                    assert!((sin[row][column] - angle.sin()).abs() < 0.005);
                }
            }
        }

        for _ in 0..BENCH_WARMUP {
            graph.launch()?;
        }
        stream.synchronize()?;
        let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..BENCH_ITERATIONS {
            graph.launch()?;
        }
        let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        end.synchronize()?;
        let latency_us = f64::from(start.elapsed_ms(&end)?) * 1_000.0 / BENCH_ITERATIONS as f64;
        eprintln!("DFlash C16/block8 RoPE graph latency: {latency_us:.3} us");

        drop(graph);
        drop(cos);
        drop(sin);
        device.synchronize()?;
        Ok(())
    }

    #[test]
    fn graph_rows_keep_mixed_positions_across_the_old_table_boundary() -> Result<()> {
        let rows = dflash_graph_host_rows(DFlashGraphHostInput {
            anchors: &[5, 7],
            start_positions: &[65_532, 100_000],
            mask_token_id: 9,
            block: 8,
            batch_bucket: 2,
            sampling: None,
        })?;
        assert_eq!(
            rows.rope_indices,
            [
                65_532, 65_533, 65_534, 65_535, 65_536, 65_537, 65_538, 65_539, 100_000, 100_001,
                100_002, 100_003, 100_004, 100_005, 100_006, 100_007,
            ]
        );
        Ok(())
    }

    #[test]
    fn graph_rows_pad_with_a_valid_alias() -> Result<()> {
        let rows = dflash_graph_host_rows(DFlashGraphHostInput {
            anchors: &[5, 7],
            start_positions: &[100, 200],
            mask_token_id: 9,
            block: 4,
            batch_bucket: 4,
            sampling: None,
        })?;
        assert_eq!(
            rows.token_ids,
            [5, 9, 9, 9, 7, 9, 9, 9, 7, 9, 9, 9, 7, 9, 9, 9]
        );
        assert_eq!(rows.anchors, [5, 7, 7, 7]);
        assert!(rows.selector_inverse_temperatures.is_none());
        assert!(rows.selector_uniforms.is_none());
        assert_eq!(
            rows.rope_indices,
            [100, 101, 102, 103, 200, 201, 202, 203, 200, 201, 202, 203, 200, 201, 202, 203,]
        );
        Ok(())
    }

    #[test]
    fn graph_rows_pad_selector_sampling_inputs_with_inert_rows() -> Result<()> {
        let rows = dflash_graph_host_rows(DFlashGraphHostInput {
            anchors: &[5, 7],
            start_positions: &[100, 200],
            mask_token_id: 9,
            block: 4,
            batch_bucket: 4,
            sampling: Some(DFlashSamplingInputs {
                inverse_temperatures: &[1.0, 0.5],
                uniforms: &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }),
        })?;
        assert_eq!(
            rows.selector_inverse_temperatures.as_deref(),
            Some(&[1.0, 0.5, 0.0, 0.0][..])
        );
        assert_eq!(
            rows.selector_uniforms.as_deref(),
            Some(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][..])
        );
        Ok(())
    }

    #[test]
    fn graph_rows_reuse_bucket_storage() -> Result<()> {
        let mut rows = dflash_graph_host_rows(DFlashGraphHostInput {
            anchors: &[5, 7],
            start_positions: &[100, 200],
            mask_token_id: 9,
            block: 4,
            batch_bucket: 4,
            sampling: Some(DFlashSamplingInputs {
                inverse_temperatures: &[1.0, 0.5],
                uniforms: &[0.1; 6],
            }),
        })?;
        let capacities = (
            rows.token_ids.capacity(),
            rows.rope_indices.capacity(),
            rows.anchors.capacity(),
            rows.selector_inverse_temperatures
                .as_ref()
                .expect("sampling temperatures")
                .capacity(),
            rows.selector_uniforms
                .as_ref()
                .expect("sampling uniforms")
                .capacity(),
        );
        rows.update(DFlashGraphHostInput {
            anchors: &[11, 13],
            start_positions: &[300, 400],
            mask_token_id: 17,
            block: 4,
            batch_bucket: 4,
            sampling: Some(DFlashSamplingInputs {
                inverse_temperatures: &[0.25, 0.75],
                uniforms: &[0.2; 6],
            }),
        })?;
        assert_eq!(rows.token_ids[0], 11);
        assert_eq!(rows.token_ids[4], 13);
        assert_eq!(rows.rope_indices[0], 300);
        assert_eq!(rows.rope_indices[4], 400);
        assert_eq!(
            capacities,
            (
                rows.token_ids.capacity(),
                rows.rope_indices.capacity(),
                rows.anchors.capacity(),
                rows.selector_inverse_temperatures
                    .as_ref()
                    .expect("sampling temperatures")
                    .capacity(),
                rows.selector_uniforms
                    .as_ref()
                    .expect("sampling uniforms")
                    .capacity(),
            )
        );
        Ok(())
    }
}
