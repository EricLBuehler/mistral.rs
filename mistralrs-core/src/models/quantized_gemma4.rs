#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{
    q_rms_norm_rope, qk_rms_norm_rope, Activation, CausalMaskConfig, CausalMasker, RmsNorm,
    RotaryEmbedding, ScaledEmbedding, Sdpa,
};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache, NormalCacheType};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

// Defaults fall back to upstream Gemma 4 config values when the GGUF
// metadata omits an optional key.
const DEFAULT_MAX_SEQ_LEN: u32 = 131072;
const DEFAULT_ROPE_FREQ_BASE: f32 = 1_000_000.0;
const DEFAULT_ROPE_FREQ_BASE_SWA: f32 = 10_000.0;
const DEFAULT_RMS_NORM_EPS: f32 = 1e-6;
const DEFAULT_HEAD_DIM_SWA: usize = 256;
const DEFAULT_HEAD_DIM_GLOBAL: usize = 512;
const DEFAULT_SLIDING_WINDOW_PATTERN: usize = 6;
const DEFAULT_PARTIAL_ROTARY_FACTOR: f64 = 0.25;
const HIDDEN_ACTIVATION: Activation = Activation::GeluPytorchTanh;

const SWA_LAYER: &str = "sliding_attention";
const FULL_LAYER: &str = "full_attention";

/// Proportional RoPE for Gemma 4 full-attention layers, ported to the GGUF path.
///
/// The full-attention layers rotate only the first
/// `partial_rotary_factor * head_dim` dimensions but use `head_dim` as the
/// denominator in `inv_freq`. The remaining frequencies are zero, which makes
/// the standard rotary formula act as identity on the tail of each head.
#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    is_gpt_neox: bool,
}

impl ProportionalRotaryEmbedding {
    fn new(
        base: f32,
        head_dim: usize,
        partial_rotary_factor: f64,
        max_position_embeddings: usize,
        device: &Device,
        is_gpt_neox: bool,
        dtype: DType,
    ) -> Result<Self> {
        let rope_angles = ((partial_rotary_factor * head_dim as f64) / 2.0) as usize;
        let half_dim = head_dim / 2;

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / base.powf((2 * i) as f32 / head_dim as f32));
        }
        inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            is_gpt_neox,
        })
    }

    fn forward_qk_norm(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        q_eps: f64,
        k_eps: f64,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        qk_rms_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            q_eps,
            k_eps,
            &self.cos,
            &self.sin,
            self.is_gpt_neox,
            seqlen_offsets,
        )
    }

    fn forward_q_norm(
        &self,
        q: &Tensor,
        q_weight: &Tensor,
        q_eps: f64,
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        q_rms_norm_rope(
            q,
            q_weight,
            q_eps,
            &self.cos,
            &self.sin,
            self.is_gpt_neox,
            seqlen_offsets,
        )
    }
}

#[derive(Clone)]
enum LayerRotary {
    Sliding(Arc<RotaryEmbedding>),
    Global(Arc<ProportionalRotaryEmbedding>),
}

struct Mlp {
    feed_forward_w1: Arc<dyn QuantMethod>,
    feed_forward_w2: Arc<dyn QuantMethod>,
    feed_forward_w3: Arc<dyn QuantMethod>,
    act: Activation,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        let y = crate::ops::mul_and_act(&w1, &w3, self.act)?;
        self.feed_forward_w2.forward(&y)
    }
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Option<Arc<dyn QuantMethod>>,
    attention_wv: Option<Arc<dyn QuantMethod>>,
    attention_wo: Arc<dyn QuantMethod>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: LayerRotary,
    is_sliding: bool,
    sliding_window: Option<usize>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
    kv_shared_layer_index: Option<usize>,
    layer_idx: usize,
    /// PLE per-layer gate. GGUF `blk.X.inp_gate.weight`, shape
    /// `[hidden_size, ple_dim]`. None for non-PLE GGUFs (legacy).
    per_layer_input_gate: Option<Arc<dyn QuantMethod>>,
    /// PLE per-layer projection. GGUF `blk.X.proj.weight`, shape
    /// `[ple_dim, hidden_size]`.
    per_layer_projection: Option<Arc<dyn QuantMethod>>,
    /// Norm applied after the PLE residual. GGUF `blk.X.post_norm.weight`,
    /// shape `[hidden_size]`. Loaded F32 to match the F32 activations the
    /// Q4_K projections produce (see commit 60c1e46 for the same
    /// rationale on the other gemma norms).
    post_per_layer_input_norm: Option<RmsNorm>,
    /// Optional per-layer scalar applied to the layer output. GGUF
    /// `blk.X.layer_output_scale.weight`, shape `[1]`. When PLE is
    /// active, the scalar is folded into the PLE norm's
    /// `forward_residual_scaled`; otherwise it's a standalone
    /// `broadcast_mul` at the end of `forward_attn`'s caller.
    layer_scalar: Option<Tensor>,
    /// Activation function for the PLE gate-multiply step. Gemma 4 uses
    /// `gelu_pytorch_tanh`. Cached on the layer to avoid re-parsing per
    /// forward call.
    ple_act: Activation,
}

impl LayerWeights {
    #[allow(clippy::too_many_arguments)]
    fn forward_attn(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        sliding_attention_mask: &AttentionMask,
        seqlen_offsets: &[usize],
        kv_caches: &mut [KvCache],
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let is_shared = self.kv_shared_layer_index.is_some();

        let q = self.attention_wq.forward(x)?;
        let mut q = if seq_len != 1 {
            q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?
        } else {
            q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?
        };
        // NOTE: q/k/v are F32 here (GgufMatMul::forward returns F32 from the
        // Q4_K dequant intermediate). Keep them at F32 through the rotary +
        // per-head norm step because the candle Metal `rms_norm` kernel
        // bails on `(F32, BF16)` and the Gemma 4 q/k_norm weights are
        // explicitly stored at F32 (see `gemma_norm_from_qtensor`). The
        // existing `.to_dtype(self.dtype)?` block right before sdpa
        // reconciles back to the model dtype for the attention kernel.

        let (mut k, v_norm) = if is_shared {
            (None, None)
        } else {
            let wk = self
                .attention_wk
                .as_ref()
                .expect("missing wk on dense layer");
            let wv = self
                .attention_wv
                .as_ref()
                .expect("missing wv on dense layer");
            let k = wk.forward(x)?;
            let v = wv.forward(x)?;
            let (k, v) = if seq_len != 1 {
                (
                    k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                        .transpose(1, 2)?,
                    v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                        .transpose(1, 2)?,
                )
            } else {
                (
                    k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?,
                    v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?,
                )
            };
            let v = v.apply(&self.v_norm)?;
            (Some(k), Some(v))
        };

        match &self.rotary {
            LayerRotary::Sliding(rot) => {
                if let Some(k_val) = k.take() {
                    let (q_rot, k_rot) = rot.forward_qk_norm(
                        &q,
                        &k_val,
                        self.q_norm.weight(),
                        self.k_norm.weight(),
                        self.q_norm.eps(),
                        self.k_norm.eps(),
                        seqlen_offsets,
                    )?;
                    q = q_rot;
                    k = Some(k_rot);
                } else {
                    q = rot.forward_q_norm(
                        &q,
                        self.q_norm.weight(),
                        self.q_norm.eps(),
                        seqlen_offsets,
                    )?;
                }
            }
            LayerRotary::Global(rot) => {
                if let Some(k_val) = k.take() {
                    let (q_rot, k_rot) = rot.forward_qk_norm(
                        &q,
                        &k_val,
                        self.q_norm.weight(),
                        self.k_norm.weight(),
                        self.q_norm.eps(),
                        self.k_norm.eps(),
                        seqlen_offsets,
                    )?;
                    q = q_rot;
                    k = Some(k_rot);
                } else {
                    q = rot.forward_q_norm(
                        &q,
                        self.q_norm.weight(),
                        self.q_norm.eps(),
                        seqlen_offsets,
                    )?;
                }
            }
        }

        let (q, k_typed, v_typed) = (
            q.to_dtype(self.dtype)?,
            k.as_ref().map(|t| t.to_dtype(self.dtype)).transpose()?,
            v_norm
                .as_ref()
                .map(|t| t.to_dtype(self.dtype))
                .transpose()?,
        );

        let mut y = match &self.paged_attn {
            Some(paged_attn) => {
                let mask = if self.is_sliding {
                    sliding_attention_mask
                } else {
                    attention_mask
                };
                match metadata {
                    Some(((key_cache, value_cache), input_metadata)) => {
                        if is_shared {
                            paged_attn.forward_donor_cache(
                                &q,
                                &key_cache,
                                &value_cache,
                                mask,
                                input_metadata,
                                &self.sdpa_params,
                                None,
                            )?
                        } else {
                            paged_attn.forward(
                                &q,
                                k_typed.as_ref().unwrap(),
                                v_typed.as_ref().unwrap(),
                                mask,
                                Some(key_cache),
                                Some(value_cache),
                                input_metadata,
                                &self.sdpa_params,
                                None,
                            )?
                        }
                    }
                    None => {
                        let dummy = PagedAttentionInputMetadata::dummy(q.device())?;
                        paged_attn.forward(
                            &q,
                            k_typed.as_ref().unwrap(),
                            v_typed.as_ref().unwrap(),
                            mask,
                            None,
                            None,
                            &dummy,
                            &self.sdpa_params,
                            None,
                        )?
                    }
                }
            }
            None => {
                let (k_full, v_full) = if let Some(donor_idx) = self.kv_shared_layer_index {
                    let donor = &kv_caches[donor_idx];
                    let dk = donor.appended_k()?.unwrap().to_device(q.device())?;
                    let dv = donor.appended_v()?.unwrap().to_device(q.device())?;
                    (dk, dv)
                } else {
                    kv_caches[self.layer_idx]
                        .append(k_typed.as_ref().unwrap(), v_typed.as_ref().unwrap())?
                };

                let (k_used, v_used) = if let Some((start, len)) = sliding_decode_kv_window(
                    self.is_sliding,
                    seq_len,
                    self.sliding_window,
                    k_full.dim(2)?,
                ) {
                    (k_full.narrow(2, start, len)?, v_full.narrow(2, start, len)?)
                } else {
                    (k_full, v_full)
                };

                let mask = if self.is_sliding {
                    sliding_attention_mask
                } else {
                    attention_mask
                };
                Sdpa.run_attention(&q, &k_used, &v_used, mask, None, &self.sdpa_params)?
            }
        };

        let has_mask = !matches!(attention_mask, AttentionMask::None)
            || !matches!(sliding_attention_mask, AttentionMask::None);
        y = if has_mask {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        self.attention_wo.forward(&y.to_dtype(x.dtype())?)
    }
}

fn sliding_decode_kv_window(
    is_sliding: bool,
    q_len: usize,
    sliding_window: Option<usize>,
    kv_len: usize,
) -> Option<(usize, usize)> {
    let window = sliding_window?;
    if !is_sliding || q_len != 1 || kv_len <= window {
        return None;
    }
    Some((kv_len - window, window))
}

pub struct ModelWeights {
    embed_tokens: ScaledEmbedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    final_logit_softcapping: Option<f64>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    sliding_window: Option<usize>,
    /// PLE token embedding lookup. GGUF `per_layer_token_embd.weight`,
    /// shape `[block_count * ple_dim, vocab_size]`. None for non-PLE
    /// GGUFs.
    embed_tokens_per_layer: Option<Embedding>,
    /// PLE projection of `inputs_embeds`. GGUF
    /// `per_layer_model_proj.weight`, shape
    /// `[hidden_size, block_count * ple_dim]`.
    per_layer_model_projection: Option<Arc<dyn QuantMethod>>,
    /// RMS norm applied to the PLE projection. GGUF
    /// `per_layer_proj_norm.weight`, shape `[ple_dim]`. Loaded F32.
    per_layer_projection_norm: Option<RmsNorm>,
    /// PLE dimension, 0 disables PLE. Mirrors
    /// `embedding_length_per_layer_input` from GGUF metadata.
    hidden_size_per_layer_input: usize,
    /// Cached scalar `hidden_size^-0.5` applied to the PLE projection
    /// before the norm.
    per_layer_projection_scalar: f64,
    /// Cached scalar `2^-0.5` applied to the (projection + embedding)
    /// sum to combine both PLE branches.
    per_layer_input_scale: f64,
}

// `gemma4.*` metadata pulled from
// `ggml-org/llama.cpp:gguf-py/gguf/constants.py:2450-2483` plus the
// per-arch keys collected in `src/llama-arch.cpp` (sliding window, shared
// kv layers, swa head dims, swa rope base, final logit softcapping).
pub(crate) struct PropsGGUF {
    head_count: usize,
    head_count_kv: usize,
    block_count: usize,
    embedding_length: usize,
    rms_norm_eps: f32,
    max_seq_len: usize,
    rope_freq_base: f32,
    rope_freq_base_swa: f32,
    key_length_full: usize,
    value_length_full: usize,
    key_length_swa: usize,
    value_length_swa: usize,
    sliding_window: usize,
    sliding_window_pattern: SlidingWindowPattern,
    num_kv_shared_layers: usize,
    final_logit_softcapping: Option<f64>,
    /// Per-Layer Input dimension. `gemma4.embedding_length_per_layer_input`
    /// in GGUF metadata. `0` (or absent) disables PLE entirely; this is
    /// the legacy code path that produced multilingual garbage on real
    /// Gemma 4 GGUFs because every Gemma 4 release in the wild does set
    /// this key. Kept as `usize` instead of `Option<usize>` so the
    /// downstream code can branch with a single zero-check.
    embedding_length_per_layer_input: usize,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("gemma4")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;

        let head_count = c.get_value::<u32>("attention.head_count")? as usize;
        let embedding_length = c.get_value::<u32>("embedding_length")? as usize;
        let key_length_full = c
            .get_value::<u32>("attention.key_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(DEFAULT_HEAD_DIM_GLOBAL);
        let value_length_full = c
            .get_value::<u32>("attention.value_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(DEFAULT_HEAD_DIM_GLOBAL);
        let key_length_swa = c
            .get_value::<u32>("attention.key_length_swa")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(DEFAULT_HEAD_DIM_SWA);
        let value_length_swa = c
            .get_value::<u32>("attention.value_length_swa")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(DEFAULT_HEAD_DIM_SWA);

        Ok(Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length,
            rms_norm_eps: c
                .get_value("attention.layer_norm_rms_epsilon")
                .unwrap_or(DEFAULT_RMS_NORM_EPS),
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c
                .get_value("rope.freq_base")
                .ok()
                .unwrap_or(DEFAULT_ROPE_FREQ_BASE),
            rope_freq_base_swa: c
                .get_value("rope.freq_base_swa")
                .ok()
                .unwrap_or(DEFAULT_ROPE_FREQ_BASE_SWA),
            key_length_full,
            value_length_full,
            key_length_swa,
            value_length_swa,
            sliding_window: c
                .get_value::<u32>("attention.sliding_window")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            // Gemma 4 GGUFs store `attention.sliding_window_pattern` two
            // different ways depending on the converter version:
            //   - Newer (unsloth, llama.cpp HEAD): a per-layer `Vec<u32>`
            //     mask of length `block_count`, where 1 = sliding (SWA)
            //     and 0 = full/global attention.
            //   - Older: a single `u32` period `n`, where every layer with
            //     `(idx % n) < n - 1` is sliding and the rest are global.
            // Read both and prefer the explicit mask when it is present;
            // fall back to the scalar period when only that is available,
            // and finally to the historical default of 6 when neither is
            // set. Misreading the array case as `u32` silently classified
            // most layers using the default period, producing 8/256 vs
            // 8/512 head-dim mismatches inside qk_rms_norm_rope.
            sliding_window_pattern: SlidingWindowPattern::from_metadata(&c),
            num_kv_shared_layers: c
                .get_value::<u32>("attention.shared_kv_layers")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            final_logit_softcapping: c
                .get_option_value::<f32>("final_logit_softcapping")?
                .map(f64::from),
            embedding_length_per_layer_input: c
                .get_value::<u32>("embedding_length_per_layer_input")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
        })
    }
}

/// How the per-layer sliding/full attention pattern is encoded in the GGUF.
///
/// Different llama.cpp converter revisions write this metadata three
/// different ways for Gemma 4; we accept all of them.
#[derive(Debug, Clone)]
enum SlidingWindowPattern {
    /// Explicit per-layer mask, length == `block_count`, `true` = sliding
    /// (SWA), `false` = full/global attention. This is the actual layout
    /// emitted by unsloth's `gemma-4-E2B-it-GGUF` (and llama.cpp HEAD):
    /// `[true, true, true, true, false, ...]` for period 5.
    BoolMask(Vec<bool>),
    /// Same as `BoolMask` but stored as integers (`1`/`0`). Some older
    /// converter revisions used this form before the bool-array encoding
    /// became standard.
    IntMask(Vec<u32>),
    /// Scalar period `n`: layer `il` is sliding when `il % n < n - 1`,
    /// matching llama.cpp's `set_swa_pattern(n, dense_first=false)`. The
    /// oldest converters wrote this form.
    Period(usize),
}

impl SlidingWindowPattern {
    fn from_metadata(c: &ContentMetadata<'_>) -> Self {
        if let Ok(mask) = c.get_value::<Vec<bool>>("attention.sliding_window_pattern") {
            if !mask.is_empty() {
                return Self::BoolMask(mask);
            }
        }
        if let Ok(mask) = c.get_value::<Vec<u32>>("attention.sliding_window_pattern") {
            if !mask.is_empty() {
                return Self::IntMask(mask);
            }
        }
        let period = c
            .get_value::<u32>("attention.sliding_window_pattern")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(DEFAULT_SLIDING_WINDOW_PATTERN);
        Self::Period(period)
    }
}

// `layer_types` is reconstructed from the sliding-window pattern stored in
// metadata. Either mask form is a direct per-layer flag; the scalar form
// follows llama.cpp's `set_swa_pattern(n, dense_first=false)`, which marks
// layer `il` as SWA when `il % n < n - 1`.
fn build_layer_types(
    block_count: usize,
    sliding_window_pattern: &SlidingWindowPattern,
) -> Vec<&'static str> {
    let mut out = Vec::with_capacity(block_count);
    for il in 0..block_count {
        let is_swa = match sliding_window_pattern {
            SlidingWindowPattern::BoolMask(mask) => mask.get(il).copied().unwrap_or(false),
            SlidingWindowPattern::IntMask(mask) => mask.get(il).copied().unwrap_or(0) != 0,
            SlidingWindowPattern::Period(n) => {
                *n == 0 || (il % *n) < n.saturating_sub(1)
            }
        };
        out.push(if is_swa { SWA_LAYER } else { FULL_LAYER });
    }
    out
}

fn kv_shared_layer_index(
    layer_types: &[&str],
    num_kv_shared_layers: usize,
    layer_idx: usize,
) -> Result<Option<usize>> {
    let first_shared = layer_types.len().saturating_sub(num_kv_shared_layers);
    if first_shared == 0 || layer_idx < first_shared {
        return Ok(None);
    }
    let attention_type = layer_types[layer_idx];
    layer_types[..first_shared]
        .iter()
        .rposition(|ty| *ty == attention_type)
        .map(Some)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Gemma4 layer {layer_idx} shares KV but no prior `{attention_type}` donor layer exists."
            ))
        })
}

fn gemma_norm_from_qtensor(
    qt: candle_core::quantized::QTensor,
    eps: f32,
    device: &Device,
) -> Result<RmsNorm> {
    // Gemma stores raw RmsNorm weights; the model adds 1.0 at runtime, matching
    // `Gemma3Model.norm_shift` in `ggml-org/llama.cpp:conversion/gemma.py:124`.
    //
    // Gemma 4 GGUFs store the per-head `attn_{q,k}_norm.weight` tensors as
    // raw BF16 rather than Q4_K (norm scales are too small to quantize), so
    // `dequantize` returns BF16 directly. Cast to F32 here so:
    //   1. the `+ 1.0` norm shift stays numerically accurate; and
    //   2. the weight dtype matches the F32 activations produced by
    //      `GgufMatMul::forward` (Q4_K dequant intermediate). The candle
    //      Metal `rms_norm` kernel requires `input.dtype() == weight.dtype()`
    //      and bails with `rmsnorm is not implemented for F32 BF16`
    //      otherwise.
    // The forward path already casts back to `self.dtype` after the rotary
    // step (see `LayerWeights::forward_attn`), so keeping the norm at F32
    // does not poison the rest of the model.
    let w = qt.dequantize(device)?.to_dtype(DType::F32)?;
    let w = (w + 1.0)?;
    RmsNorm::from_w(w, eps as f64)
}

fn detect_moe<R: std::io::Seek + std::io::Read>(ct: &mut Content<'_, R>) -> Result<()> {
    if ct.has_tensor("blk.0.ffn_gate_exps.weight") || ct.has_tensor("blk.0.ffn_gate_up_exps.weight")
    {
        candle_core::bail!(
            "Gemma 4 MoE (FFN_*_EXP tensors detected) is not supported by this GGUF loader; tracked separately."
        );
    }
    Ok(())
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        detect_moe(&mut ct)?;

        let meta = ct.get_metadata();
        let metadata = ContentMetadata {
            path_prefix: "gemma4",
            metadata: meta,
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            rope_freq_base_swa,
            key_length_full,
            value_length_full,
            key_length_swa,
            value_length_swa,
            sliding_window,
            sliding_window_pattern,
            num_kv_shared_layers,
            final_logit_softcapping,
            embedding_length_per_layer_input,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        if key_length_full != value_length_full {
            candle_core::bail!(
                "Gemma 4 GGUF requires key_length == value_length for full-attention layers, got {key_length_full} != {value_length_full}"
            );
        }
        if key_length_swa != value_length_swa {
            candle_core::bail!(
                "Gemma 4 GGUF requires key_length_swa == value_length_swa for sliding layers, got {key_length_swa} != {value_length_swa}"
            );
        }

        let layer_types = build_layer_types(block_count, &sliding_window_pattern);

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let output_norm = gemma_norm_from_qtensor(
            ct.tensor("output_norm.weight", device)?,
            rms_norm_eps,
            device,
        )?;
        let output_qt = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };

        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if embedding_length_per_layer_input > 0 {
                let ple_emb_qt = ct.tensor("per_layer_token_embd.weight", device)?;
                let ple_emb_w = ple_emb_qt.dequantize(device)?.to_dtype(dtype)?;
                let ple_emb = Embedding::new(
                    ple_emb_w,
                    block_count * embedding_length_per_layer_input,
                );

                let ple_proj_qt = ct.tensor("per_layer_model_proj.weight", device)?;
                let ple_proj = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(ple_proj_qt),
                    b: None,
                })?) as Arc<dyn QuantMethod>;

                let ple_norm = gemma_norm_from_qtensor(
                    ct.tensor("per_layer_proj_norm.weight", device)?,
                    rms_norm_eps,
                    device,
                )?;

                (Some(ple_emb), Some(ple_proj), Some(ple_norm))
            } else {
                (None, None, None)
            };

        // Sliding RoPE: one per device, head_dim_swa, rope_freq_base_swa.
        // Global RoPE: one per device, head_dim_full, proportional, rope_freq_base.
        let mut sliding_ropes: HashMap<_, Arc<RotaryEmbedding>> = HashMap::new();
        let mut global_ropes: HashMap<_, Arc<ProportionalRotaryEmbedding>> = HashMap::new();
        for layer_idx in 0..block_count {
            let dev = mapper.device_for(layer_idx, false).unwrap_or(device);
            sliding_ropes.entry(dev.location()).or_insert_with(|| {
                Arc::new(
                    RotaryEmbedding::new(
                        rope_freq_base_swa,
                        key_length_swa,
                        max_seq_len,
                        dev,
                        true,
                        DType::F32,
                    )
                    .expect("failed to build sliding RoPE"),
                )
            });
            global_ropes.entry(dev.location()).or_insert_with(|| {
                Arc::new(
                    ProportionalRotaryEmbedding::new(
                        rope_freq_base,
                        key_length_full,
                        DEFAULT_PARTIAL_ROTARY_FACTOR,
                        max_seq_len,
                        dev,
                        true,
                        DType::F32,
                    )
                    .expect("failed to build global RoPE"),
                )
            });
        }

        let sliding_window_opt = if sliding_window == 0 {
            None
        } else {
            Some(sliding_window)
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let dev = mapper.device_for(layer_idx, false).unwrap_or(device);
            let is_sliding = layer_types[layer_idx] == SWA_LAYER;
            let head_dim = if is_sliding {
                key_length_swa
            } else {
                key_length_full
            };

            let donor = kv_shared_layer_index(&layer_types, num_kv_shared_layers, layer_idx)?;

            let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), dev)?;
            let attention_wo = ct.tensor(&format!("{prefix}.attn_output.weight"), dev)?;

            let (attention_wk, attention_wv): (Option<_>, Option<_>) = if donor.is_some() {
                (None, None)
            } else {
                let wk = ct.tensor(&format!("{prefix}.attn_k.weight"), dev)?;
                // `attn_v` is optional: alternative attention reuses K as V.
                let wv = if ct.has_tensor(&format!("{prefix}.attn_v.weight")) {
                    ct.tensor(&format!("{prefix}.attn_v.weight"), dev)?
                } else {
                    ct.tensor(&format!("{prefix}.attn_k.weight"), dev)?
                };
                (
                    Some(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(wk),
                        b: None,
                    })?) as Arc<dyn QuantMethod>),
                    Some(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(wv),
                        b: None,
                    })?) as Arc<dyn QuantMethod>),
                )
            };

            let feed_forward_w1 = ct.tensor(&format!("{prefix}.ffn_gate.weight"), dev)?;
            let feed_forward_w2 = ct.tensor(&format!("{prefix}.ffn_down.weight"), dev)?;
            let feed_forward_w3 = ct.tensor(&format!("{prefix}.ffn_up.weight"), dev)?;

            let q_norm = gemma_norm_from_qtensor(
                ct.tensor(&format!("{prefix}.attn_q_norm.weight"), dev)?,
                rms_norm_eps,
                dev,
            )?;
            let k_norm = if donor.is_some() {
                // The donor layer keeps its own k_norm; we still need a placeholder
                // since RoPE is applied to Q only. Placeholder is F32 to match
                // the dtype contract of the other Gemma norm weights.
                RmsNorm::from_w(
                    Tensor::ones(head_dim, DType::F32, dev)?,
                    rms_norm_eps as f64,
                )?
            } else {
                gemma_norm_from_qtensor(
                    ct.tensor(&format!("{prefix}.attn_k_norm.weight"), dev)?,
                    rms_norm_eps,
                    dev,
                )?
            };
            // V norm is a fused RmsNorm with weight=1.0; there is no
            // `attn_v_norm` tensor in upstream Gemma 4 GGUFs. Use F32 to
            // match the F32 activations produced by the quantized V proj.
            let v_norm = RmsNorm::from_w(
                Tensor::ones(head_dim, DType::F32, dev)?,
                rms_norm_eps as f64,
            )?;

            let input_layernorm = gemma_norm_from_qtensor(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), dev)?,
                rms_norm_eps,
                dev,
            )?;
            let post_attention_layernorm = gemma_norm_from_qtensor(
                ct.tensor(&format!("{prefix}.post_attention_norm.weight"), dev)?,
                rms_norm_eps,
                dev,
            )?;
            let pre_feedforward_layernorm = gemma_norm_from_qtensor(
                ct.tensor(&format!("{prefix}.ffn_norm.weight"), dev)?,
                rms_norm_eps,
                dev,
            )?;
            let post_feedforward_layernorm = gemma_norm_from_qtensor(
                ct.tensor(&format!("{prefix}.post_ffw_norm.weight"), dev)?,
                rms_norm_eps,
                dev,
            )?;

            let ple_dim = embedding_length_per_layer_input;
            let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm, layer_scalar) =
                if ple_dim > 0 {
                    let gate_qt = ct.tensor(&format!("{prefix}.inp_gate.weight"), dev)?;
                    let gate = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(gate_qt),
                        b: None,
                    })?) as Arc<dyn QuantMethod>;

                    let proj_qt = ct.tensor(&format!("{prefix}.proj.weight"), dev)?;
                    let proj = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(proj_qt),
                        b: None,
                    })?) as Arc<dyn QuantMethod>;

                    let post_norm = gemma_norm_from_qtensor(
                        ct.tensor(&format!("{prefix}.post_norm.weight"), dev)?,
                        rms_norm_eps,
                        dev,
                    )?;

                    // layer_output_scale is optional in older converter
                    // revisions but always present on current unsloth Gemma 4
                    // GGUFs.
                    let scalar = ct
                        .tensor(&format!("{prefix}.layer_output_scale.weight"), dev)
                        .ok()
                        .map(|qt| qt.dequantize(dev).and_then(|t| t.to_dtype(DType::F32)))
                        .transpose()?;

                    (Some(gate), Some(proj), Some(post_norm), scalar)
                } else {
                    (None, None, None, None)
                };

            let rotary = if is_sliding {
                LayerRotary::Sliding(
                    sliding_ropes
                        .get(&dev.location())
                        .expect("missing sliding RoPE for device")
                        .clone(),
                )
            } else {
                LayerRotary::Global(
                    global_ropes
                        .get(&dev.location())
                        .expect("missing global RoPE for device")
                        .clone(),
                )
            };

            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, dev, None)?)
                }
            };

            layers.push(LayerWeights {
                attention_wq: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wq),
                    b: None,
                })?),
                attention_wk,
                attention_wv,
                attention_wo: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wo),
                    b: None,
                })?),
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
                q_norm,
                k_norm,
                v_norm,
                mlp: Mlp {
                    feed_forward_w1: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w1),
                        b: None,
                    })?),
                    feed_forward_w2: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w2),
                        b: None,
                    })?),
                    feed_forward_w3: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w3),
                        b: None,
                    })?),
                    act: HIDDEN_ACTIVATION,
                },
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary,
                is_sliding,
                sliding_window: if is_sliding { sliding_window_opt } else { None },
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    softcap: None,
                    softmax_scale: 1.0,
                    sliding_window: if is_sliding { sliding_window_opt } else { None },
                    sinks: None,
                },
                dtype,
                kv_shared_layer_index: donor,
                layer_idx,
                per_layer_input_gate,
                per_layer_projection,
                post_per_layer_input_norm,
                layer_scalar,
                ple_act: Activation::GeluPytorchTanh,
            });
        }

        Ok(Self {
            embed_tokens: ScaledEmbedding::new(
                (embedding_length as f64).sqrt(),
                Embedding::new(tok_embeddings, embedding_length),
            ),
            layers,
            norm: output_norm,
            output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(output_qt),
                b: None,
            })?),
            final_logit_softcapping,
            device: device.clone(),
            // Per-layer cache types. Sliding (SWA) layers use a rotating
            // window cache sized by the GGUF `attention.sliding_window`
            // metadata; global/full-attention layers use the normal
            // append-on-write cache; and shared-KV layers point at their
            // donor's cache instead of holding their own state. This is
            // necessary because Gemma 4 uses different `head_dim` per
            // attention type (256 for SWA, 512 for global on E2B-it). A
            // uniform `NormalCache::new(...)` would still infer head_dim
            // on first append per layer, but the engine's batched
            // `clone_in_cache` step relies on the per-layer cache type to
            // skip shared slots, and the rotating cache flavor is
            // required for SWA layers to bound memory by the window
            // instead of the full context length.
            cache: EitherCache::Normal(NormalCache::from_types(
                layer_types
                    .iter()
                    .enumerate()
                    .map(|(layer_idx, ty)| {
                        if let Some(donor) =
                            kv_shared_layer_index(&layer_types, num_kv_shared_layers, layer_idx)
                                .ok()
                                .flatten()
                        {
                            NormalCacheType::Shared { owner: donor }
                        } else if *ty == SWA_LAYER && sliding_window > 0 {
                            NormalCacheType::SlidingWindow {
                                window: sliding_window,
                            }
                        } else {
                            NormalCacheType::Normal { max_seq_len }
                        }
                    })
                    .collect(),
            )),
            max_seq_len,
            mapper: Some(mapper),
            dtype,
            sliding_window: sliding_window_opt,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            hidden_size_per_layer_input: embedding_length_per_layer_input,
            per_layer_projection_scalar: (embedding_length as f64).powf(-0.5),
            per_layer_input_scale: 2f64.powf(-0.5),
        })
    }
}

impl ModelWeights {
    /// Compute the per-layer PLE inputs once at the start of `forward`.
    /// Mirrors `vision_models/gemma4/text.rs::compute_ple`.
    ///
    /// Returns `Ok(None)` when PLE is disabled (legacy GGUFs without
    /// the `embedding_length_per_layer_input` metadata key), letting the
    /// caller skip the per-layer injection without any branching cost.
    fn compute_ple(
        &self,
        ple_input_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Option<Vec<Tensor>>> {
        if self.hidden_size_per_layer_input == 0 {
            return Ok(None);
        }
        let ple_emb = self
            .embed_tokens_per_layer
            .as_ref()
            .expect("PLE embedding required when hidden_size_per_layer_input > 0");
        let ple_proj = self
            .per_layer_model_projection
            .as_ref()
            .expect("PLE projection required when hidden_size_per_layer_input > 0");
        let ple_norm = self
            .per_layer_projection_norm
            .as_ref()
            .expect("PLE norm required when hidden_size_per_layer_input > 0");

        let ple_dim = self.hidden_size_per_layer_input;
        let num_layers = self.layers.len();
        let (b, seq, _) = inputs_embeds.dims3()?;

        // 1. Token-level per-layer embeddings.
        let embedded = ple_emb.forward(ple_input_ids)?;
        let embedded = (embedded * (ple_dim as f64).sqrt())?;
        let embedded = embedded.reshape((b, seq, num_layers, ple_dim))?;

        // 2. Project the hidden state.
        let projected = ple_proj.forward(inputs_embeds)?;
        let projected = (projected * self.per_layer_projection_scalar)?;
        let projected = projected.reshape((b, seq, num_layers, ple_dim))?;

        // 3. Normalize the projection (per ple_dim slot).
        let projected = ple_norm.forward(&projected)?;

        // 4. Combine projection + embedding and apply the 2^-0.5 scale.
        let combined = ((projected + embedded)? * self.per_layer_input_scale)?;

        // 5. Split into per-layer slices via one contiguous transpose
        // followed by narrow + squeeze (zero-copy after `contiguous`).
        let combined = combined.transpose(1, 2)?.contiguous()?;
        let mut per_layer_inputs = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            per_layer_inputs.push(combined.narrow(1, i, 1)?.squeeze(1)?);
        }
        Ok(Some(per_layer_inputs))
    }

    pub fn forward(
        &self,
        x: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(x)?;
        // Compute PLE per-layer inputs once. None when PLE is disabled
        // (legacy GGUFs without embedding_length_per_layer_input).
        let per_layer_inputs = self.compute_ple(x, &xs)?;
        let cache = &mut self.cache.normal().0;

        let attention_mask = CausalMasker.make_causal_mask(
            x,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let attention_mask = if metadata
            .as_ref()
            .map(|(_, meta)| meta.is_first_prompt_chunk)
            .unwrap_or(true)
        {
            attention_mask
        } else {
            AttentionMask::None
        };
        let attention_mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(attention_mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(attention_mask)
        };

        let sliding_mask_owned = if let Some(window) = self.sliding_window {
            let m = CausalMasker.make_causal_mask(
                x,
                metadata
                    .as_ref()
                    .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                    .unwrap_or(cache as &dyn PastKvLenCache),
                self.dtype,
                &CausalMaskConfig {
                    sliding_window: Some(window),
                    force_custom: false,
                },
            )?;
            let m = if metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
            {
                m
            } else {
                AttentionMask::None
            };
            let dm = if let Some(ref mapper) = self.mapper {
                DeviceMappedMask::new(m, &**mapper)?
            } else {
                DeviceMappedMask::from_single(m)
            };
            Some(dm)
        } else {
            None
        };
        let sliding_mask_ref: &DeviceMappedMask =
            sliding_mask_owned.as_ref().unwrap_or(&attention_mask);

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                xs = mapper.map(xs, i)?;
            }

            let residual = xs.clone();
            let normed = layer.input_layernorm.forward(&xs)?;
            let attn_out = layer.forward_attn(
                &normed,
                &attention_mask.get(xs.device()),
                &sliding_mask_ref.get(xs.device()),
                seqlen_offsets,
                cache.as_mut_slice(),
                metadata
                    .as_ref()
                    .map(|(kv_cache, meta)| (kv_cache[i].clone(), *meta)),
            )?;
            xs = layer
                .post_attention_layernorm
                .forward_residual(&attn_out, &residual)?;

            let residual = xs.clone();
            let normed = xs.apply(&layer.pre_feedforward_layernorm)?;
            let mlp_out = layer.mlp.forward(&normed)?;
            xs = layer
                .post_feedforward_layernorm
                .forward_residual(&mlp_out, &residual)?;

            // PLE: per-layer embedding injection (after feedforward, before
            // the layer's residual return). Mirrors
            // vision_models/gemma4/text.rs:1001-1030.
            let mut layer_scalar_applied = false;
            xs = if let (Some(gate), Some(proj), Some(norm)) = (
                layer.per_layer_input_gate.as_ref(),
                layer.per_layer_projection.as_ref(),
                layer.post_per_layer_input_norm.as_ref(),
            ) {
                let pli = per_layer_inputs.as_ref().and_then(|v| v.get(i));
                if let Some(pli) = pli {
                    let residual_ple = xs.clone();
                    let gated = gate.forward(&xs)?;
                    let gated = crate::ops::mul_and_act(&gated, pli, layer.ple_act)?;
                    let projected = proj.forward(&gated)?;
                    if let Some(scalar) = layer.layer_scalar.as_ref() {
                        layer_scalar_applied = true;
                        norm.forward_residual_scaled(&projected, &residual_ple, scalar)?
                    } else {
                        norm.forward_residual(&projected, &residual_ple)?
                    }
                } else {
                    xs
                }
            } else {
                xs
            };

            // If PLE didn't run but the layer has a standalone scalar,
            // apply it directly. Mirrors safetensors `if
            // !layer_scalar_applied` branch.
            if !layer_scalar_applied {
                if let Some(scalar) = layer.layer_scalar.as_ref() {
                    xs = xs.broadcast_mul(scalar)?;
                }
            }
        }

        let xs = self.norm.forward(&xs)?;
        let xs = extract_logits(&xs, context_lens)?;
        let mut logits = self.output.forward(&xs.contiguous()?)?;
        if let Some(softcap) = self.final_logit_softcapping {
            let scale = softcap as f32;
            logits = ((logits / scale as f64)?.tanh()? * scale as f64)?;
        }
        Ok(logits)
    }
}
