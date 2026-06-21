#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF quantized Qwen3.5 model.
//!
//! Qwen3.5 uses a hybrid architecture that alternates standard transformer
//! attention layers with GatedDeltaNet (GDN) linear-attention layers.

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::{
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    gguf::Content,
    kv_cache::{
        EitherCache, HybridCache, HybridCacheConfig, HybridLayerType, KvCache, RecurrentLayerConfig,
    },
    layers::{
        CausalMaskConfig, CausalMasker, QRmsNorm, Qwen3VLRotaryEmbedding, RotaryEmbedding, Sdpa,
    },
    layers_masker::PastKvLenCache,
    models::gdn::{causal_conv1d_fwd, gated_delta_rule_recurrence, GdnLayerCache},
    paged_attention::{AttentionImplementation, PagedAttention},
    pipeline::{extract_logits, text_models_inputs_processor::PagedAttentionInputMetadata},
    utils::{
        gguf_metadata::ContentMetadata,
        model_config as ModelConfig,
        progress::{new_multi_progress, NiceProgressBar},
    },
};

const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

// ---------------------------------------------------------------------------
// Hybrid metadata parsed from GGUF
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum GgufLayerKind {
    FullAttention,
    LinearAttention,
}

struct HybridMeta {
    layer_kinds: Vec<GgufLayerKind>,
    num_k_heads: usize,
    num_v_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel_size: usize,
    /// key_dim  = num_k_heads * key_head_dim
    key_dim: usize,
    /// value_dim = num_v_heads * value_head_dim
    value_dim: usize,
}

fn parse_hybrid_meta(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
    arch: &str,
    block_count: usize,
) -> HybridMeta {
    let md = |s: &str| metadata.get(s);

    // Determine layer types
    let layer_kinds: Vec<GgufLayerKind> = if let Some(v) = md(&format!("{arch}.layer_types")) {
        if let Ok(arr) = v.to_vec() {
            arr.iter()
                .filter_map(|val| val.to_string().ok())
                .map(|s| {
                    if s == "full_attention" || s == "attention" {
                        GgufLayerKind::FullAttention
                    } else {
                        GgufLayerKind::LinearAttention
                    }
                })
                .collect()
        } else {
            vec![GgufLayerKind::FullAttention; block_count]
        }
    } else if let Some(v) = md(&format!("{arch}.full_attention_interval")) {
        let interval = v.to_u32().unwrap_or(0) as usize;
        if interval > 0 {
            (0..block_count)
                .map(|i| {
                    if (i + 1) % interval == 0 {
                        GgufLayerKind::FullAttention
                    } else {
                        GgufLayerKind::LinearAttention
                    }
                })
                .collect()
        } else {
            vec![GgufLayerKind::FullAttention; block_count]
        }
    } else {
        vec![GgufLayerKind::FullAttention; block_count]
    };

    let conv_kernel_size = md(&format!("{arch}.ssm.conv_kernel"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(4) as usize;
    let num_k_heads = md(&format!("{arch}.ssm.group_count"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;
    let num_v_heads = md(&format!("{arch}.ssm.time_step_rank"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;
    let key_head_dim = md(&format!("{arch}.ssm.state_size"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(128) as usize;
    let inner_size = md(&format!("{arch}.ssm.inner_size"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;
    let value_head_dim = if num_v_heads > 0 && inner_size > 0 && inner_size % num_v_heads == 0 {
        inner_size / num_v_heads
    } else {
        key_head_dim
    };

    let key_dim = num_k_heads * key_head_dim;
    let value_dim = num_v_heads * value_head_dim;

    HybridMeta {
        layer_kinds,
        num_k_heads,
        num_v_heads,
        key_head_dim,
        value_head_dim,
        conv_kernel_size,
        key_dim,
        value_dim,
    }
}

// ---------------------------------------------------------------------------
// Undo GGUF v-head tiling (copied from candle-vllm reference)
// ---------------------------------------------------------------------------

fn undo_tiled_v_heads_first_dim(
    x: &Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if num_k_heads == num_v_heads {
        return Ok(x.clone());
    }
    let num_v_per_k = num_v_heads / num_k_heads;
    let dims = x.dims().to_vec();
    let mut reshaped = vec![num_v_per_k, num_k_heads, head_dim];
    reshaped.extend_from_slice(&dims[1..]);
    x.reshape(reshaped)?
        .transpose(0, 1)?
        .contiguous()?
        .reshape(dims)
}

fn undo_tiled_v_heads_last_dim(
    x: &Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if num_k_heads == num_v_heads {
        return Ok(x.clone());
    }
    let num_v_per_k = num_v_heads / num_k_heads;
    let dims = x.dims().to_vec();
    let split_dim = dims.len() - 1;
    let mut reshaped = dims[..split_dim].to_vec();
    reshaped.extend_from_slice(&[num_v_per_k, num_k_heads, head_dim]);
    x.reshape(reshaped)?
        .transpose(split_dim, split_dim + 1)?
        .contiguous()?
        .reshape(dims)
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

struct Mlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.gate.forward(xs)?;
        let w3 = self.up.forward(xs)?;
        let y = crate::ops::mul_and_act(&w1, &w3, crate::layers::Activation::Silu)?;
        self.down.forward(&y)
    }
}

// ---------------------------------------------------------------------------
// Rotary embedding variant enum for text-only (Plain) vs multimodal (MRope) RoPE
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Rotary {
    Plain(Arc<RotaryEmbedding>),
    MRope(Arc<Qwen3VLRotaryEmbedding>),
}

// Full-attention layer weights
// ---------------------------------------------------------------------------

struct FullAttnWeights {
    wq: Arc<dyn QuantMethod>,
    wk: Arc<dyn QuantMethod>,
    wv: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope_dim: usize,
    attn_output_gate: bool,
    rotary: Rotary,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl FullAttnWeights {
    fn forward(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q_raw = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        // Split Q into (q, gate) if attn_output_gate; gate has same shape as q.
        let (q, gate) = if self.attn_output_gate {
            let q_gate = q_raw.reshape((b_sz, seq_len, self.n_head, self.head_dim * 2))?;
            let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?.contiguous()?;
            let gate = q_gate
                .narrow(D::Minus1, self.head_dim, self.head_dim)?
                .contiguous()?;
            let q = q.reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
            let gate = gate.reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
            (q, Some(gate))
        } else {
            (q_raw, None)
        };

        let (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k, v)
        };

        // Per-head QK RMSNorm (same as Qwen3)
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q =
            self.q_norm
                .forward(&q_flat)?
                .reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
        let k = self.k_norm.forward(&k_flat)?.reshape((
            b_sz,
            self.n_kv_head,
            seq_len,
            self.head_dim,
        ))?;

        // Partial RoPE: rotate first `rope_dim` dims of each head, leave the rest unrotated.
        let positions = crate::pipeline::text_positions_tensor(start_offsets, seq_len, q.device())?;
        let (q, k) = match &self.rotary {
            Rotary::Plain(rope) => {
                if self.rope_dim < self.head_dim {
                    let q_rot = q.narrow(D::Minus1, 0, self.rope_dim)?.contiguous()?;
                    let q_pass = q
                        .narrow(D::Minus1, self.rope_dim, self.head_dim - self.rope_dim)?
                        .contiguous()?;
                    let k_rot = k.narrow(D::Minus1, 0, self.rope_dim)?.contiguous()?;
                    let k_pass = k
                        .narrow(D::Minus1, self.rope_dim, self.head_dim - self.rope_dim)?
                        .contiguous()?;
                    let (q_rot, k_rot) = rope.forward(&q_rot, &k_rot, &positions)?;
                    let q = Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?.contiguous()?;
                    let k = Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?.contiguous()?;
                    (q, k)
                } else {
                    rope.forward(&q, &k, &positions)?
                }
            }
            Rotary::MRope(mrope) => {
                // For text-only inputs, build position_ids with all three dimensions equal
                let pos_ids = Tensor::arange(0u32, seq_len as u32, &q.device())?
                    .unsqueeze(0)?
                    .broadcast_add(&Tensor::from_vec(
                        start_offsets.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                        (b_sz, 1),
                        &q.device(),
                    )?)?;
                let pos_ids = Tensor::stack(&[&pos_ids, &pos_ids, &pos_ids], 0)?; // (3, batch, seq_len)

                // Compute cos/sin for M-RoPE
                let (cos, sin) = mrope.compute_cos_sin(&pos_ids, self.dtype)?;

                if self.rope_dim < self.head_dim {
                    let q_pass = q
                        .narrow(D::Minus1, self.rope_dim, self.head_dim - self.rope_dim)?
                        .contiguous()?;
                    let k_pass = k
                        .narrow(D::Minus1, self.rope_dim, self.head_dim - self.rope_dim)?
                        .contiguous()?;
                    let mut q_rot = q.narrow(D::Minus1, 0, self.rope_dim)?.contiguous()?;
                    let mut k_rot = k.narrow(D::Minus1, 0, self.rope_dim)?.contiguous()?;
                    mrope.forward(&(cos, sin), &mut q_rot, &mut k_rot)?;
                    let q = Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?.contiguous()?;
                    let k = Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?.contiguous()?;
                    (q, k)
                } else {
                    let mut q_rot = q;
                    let mut k_rot = k;
                    mrope.forward(&(cos, sin), &mut q_rot, &mut k_rot)?;
                    (q_rot, k_rot)
                }
            }
        };

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        let y = match &self.paged_attn {
            Some(pa) => {
                let ((kc, vc), im) = metadata.unwrap();
                pa.forward(
                    &q,
                    &k,
                    &v,
                    mask,
                    Some(kc),
                    Some(vc),
                    im,
                    &self.sdpa_params,
                    None,
                )?
            }
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?
            }
        };

        let y = if mask.is_custom() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        // Apply output gate: y = y * sigmoid(gate)
        let y = if let Some(gate) = gate {
            let gate_sig =
                candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(y.dtype())?;
            y.broadcast_mul(&gate_sig)?
        } else {
            y
        };

        self.wo.forward(&y.to_dtype(x.dtype())?)
    }
}

// ---------------------------------------------------------------------------
// GDN layer weights (dequantized for GGUF tiling compatibility)
// ---------------------------------------------------------------------------

struct GdnWeights {
    // Projection weights stored PRE-TRANSPOSED as [in, out] so the per-call
    // matmul can skip `w.t()?.contiguous()?` (which copied the entire weight
    // matrix on every forward — ~140 GB-scale weight copies/token at decode).
    in_proj_qkv: Tensor,   // [hidden, key_dim*2 + value_dim]
    in_proj_z: Tensor,     // [hidden, value_dim]
    in_proj_beta: Tensor,  // [hidden, num_v_heads]
    in_proj_alpha: Tensor, // [hidden, num_v_heads]
    out_proj: Tensor,      // [value_dim, hidden]
    // Conv state
    conv_weight: Tensor, // [conv_dim, 1, kernel_size]
    conv_bias: Option<Tensor>,
    // SSM
    a_log: Tensor,   // [num_v_heads]
    dt_bias: Tensor, // [num_v_heads]
    // Output norm weight [head_v_dim]
    norm_weight: Tensor,
    // Config
    num_k_heads: usize,
    num_v_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    rms_norm_eps: f64,
}

impl GdnWeights {
    /// `x` is (..., in_features); `w` is PRE-TRANSPOSED `(in_features, out_features)`.
    /// Returns (..., out_features) in `w`'s dtype. Casts x to w's dtype because:
    /// (a) QRmsNorm returns F32 weight dtype but model dtype is F16/BF16
    /// (b) downstream conv1d / gating kernels are F16/BF16 only -- keeping intermediates
    ///     in weight (model) dtype avoids per-op casts.
    fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let in_dim = w.dim(0)?;
        let out_dim = w.dim(1)?;
        let x_dims = x.dims().to_vec();
        let n = x.elem_count() / in_dim;
        let x2 = x.reshape((n, in_dim))?.to_dtype(w.dtype())?;
        let y = x2.matmul(w)?;
        let mut out_shape = x_dims;
        let last = out_shape.len() - 1;
        out_shape[last] = out_dim;
        y.reshape(out_shape)
    }

    fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = x.dims3()?;
        let v_per_group = self.num_v_heads / self.num_k_heads;

        // 1. Projections — cast x to weight dtype once and reuse across the four
        // input projections (saves 3 redundant F32→F16 dispatches per layer per
        // call; benefits CUDA decode + Metal prefill which both reach forward()).
        let x_w = x.to_dtype(self.in_proj_qkv.dtype())?;
        let proj_qkv = Self::linear(&x_w, &self.in_proj_qkv)?;
        let q = proj_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = proj_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = proj_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let z = Self::linear(&x_w, &self.in_proj_z)?;
        let beta_raw = Self::linear(&x_w, &self.in_proj_beta)?;
        let alpha_raw = Self::linear(&x_w, &self.in_proj_alpha)?;

        // 2. Causal conv1d on [q, k, v] (causal_conv1d_fwd already applies silu).
        let mixed = Tensor::cat(&[&q, &k, &v], D::Minus1)?;
        let mixed = causal_conv1d_fwd(
            &mixed,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            cache,
            self.conv_kernel_size,
        )?;

        // 3. Split back after conv
        let q_c = mixed.narrow(D::Minus1, 0, self.key_dim)?;
        let k_c = mixed.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v_c = mixed.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        // 4. Compute beta and g
        let beta = candle_nn::ops::sigmoid(&beta_raw.to_dtype(DType::F32)?)?.to_dtype(x.dtype())?;
        let dt_b = self.dt_bias.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, num_v_heads]
        let a_exp = self.a_log.exp()?.neg()?.unsqueeze(0)?.unsqueeze(0)?;
        // softplus(alpha + dt_bias)
        let sp_in = alpha_raw
            .to_dtype(DType::F32)?
            .broadcast_add(&dt_b.to_dtype(DType::F32)?)?;
        let sp = (Tensor::ones_like(&sp_in)? + sp_in.exp()?)?.log()?;
        let g = a_exp
            .to_dtype(DType::F32)?
            .broadcast_mul(&sp)?
            .to_dtype(x.dtype())?;

        // 5. Reshape for recurrence
        let q_h = q_c.reshape((batch, seq_len, self.num_k_heads, self.key_head_dim))?;
        let k_h = k_c.reshape((batch, seq_len, self.num_k_heads, self.key_head_dim))?;
        let v_h = v_c.reshape((batch, seq_len, self.num_v_heads, self.value_head_dim))?;

        // L2-normalise q and k (per-head)
        let q_n = l2_norm_4d(&q_h)?;
        let k_n = l2_norm_4d(&k_h)?;

        // Expand k-heads to v-heads if grouped
        let (q_e, k_e) = if v_per_group > 1 {
            let q_e = q_n
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch, seq_len, self.num_v_heads, self.key_head_dim))?;
            let k_e = k_n
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch, seq_len, self.num_v_heads, self.key_head_dim))?;
            (q_e, k_e)
        } else {
            (q_n, k_n)
        };

        // 6. Recurrence
        let y =
            gated_delta_rule_recurrence(&q_e, &k_e, &v_h, &g, &beta, &mut cache.recurrent_state)?;
        // y: (batch, seq, num_v_heads, value_head_dim)

        // 7. RMSNorm gated by z
        let val_shape = y.shape().clone();
        let y_flat = y.reshape(((), self.value_head_dim))?;
        let z_flat = z.reshape(((), self.value_head_dim))?;
        let y_normed = rms_norm_gated(&y_flat, &z_flat, &self.norm_weight, self.rms_norm_eps)?;
        let y_out = y_normed
            .reshape(val_shape)?
            .reshape((batch, seq_len, self.value_dim))?;

        // 8. Output projection
        cache.seqlen_offset += seq_len;
        Self::linear(&y_out, &self.out_proj.t()?.t()?) // out_proj is [hidden, value_dim], so y @ out_proj.T
    }

    /// Decode-step forward pass that operates on the global state pool directly
    /// via slot indices — no gather/scatter. Metal-only fast path for `seq_len=1`.
    #[cfg(feature = "metal")]
    fn forward_decode_slots(
        &self,
        x: &Tensor,
        conv_state_pool: &mut Tensor,
        recurrent_state_pool: &mut Tensor,
        slots_gpu: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = x.dims3()?;
        debug_assert_eq!(seq_len, 1, "forward_decode_slots requires seq_len == 1");
        let v_per_group = self.num_v_heads / self.num_k_heads;

        // 1. Projections — cast x to weight dtype once, reuse for all 4 input
        // projections (avoids 3 redundant F32→F16 casts per layer).
        let x_w = x.to_dtype(self.in_proj_qkv.dtype())?;
        let proj_qkv = Self::linear(&x_w, &self.in_proj_qkv)?;
        let q = proj_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = proj_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = proj_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;
        let z = Self::linear(&x_w, &self.in_proj_z)?;
        let beta_raw = Self::linear(&x_w, &self.in_proj_beta)?;
        let alpha_raw = Self::linear(&x_w, &self.in_proj_alpha)?;

        // 2. Causal conv1d update via slots (in-place pool write)
        let mixed = Tensor::cat(&[&q, &k, &v], D::Minus1)?; // [batch, 1, conv_dim]
        let conv_dim = mixed.dim(D::Minus1)?;
        // Squeeze seq dim → [batch, conv_dim]
        let mixed_t = mixed.reshape((batch, conv_dim))?.contiguous()?;
        let weight_2d = self
            .conv_weight
            .squeeze(1)?
            .to_dtype(mixed_t.dtype())?
            .contiguous()?;
        let conv_out = crate::metal::gdn::causal_conv1d_update_slots_metal(
            &mixed_t,
            &weight_2d,
            self.conv_bias.as_ref(),
            conv_state_pool,
            slots_gpu,
            self.conv_kernel_size,
        )?; // [batch, conv_dim]
        let mixed_conv = conv_out.unsqueeze(1)?; // [batch, 1, conv_dim]

        // 3. Split after conv
        let q_c = mixed_conv.narrow(D::Minus1, 0, self.key_dim)?;
        let k_c = mixed_conv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v_c = mixed_conv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        // 4. Gating (identical to forward)
        let (beta, g) =
            compute_gdn_gating(&beta_raw, &alpha_raw, &self.a_log, &self.dt_bias, x.dtype())?;

        // 5. Reshape for recurrence
        let q_h = q_c.reshape((batch, seq_len, self.num_k_heads, self.key_head_dim))?;
        let k_h = k_c.reshape((batch, seq_len, self.num_k_heads, self.key_head_dim))?;
        let v_h = v_c.reshape((batch, seq_len, self.num_v_heads, self.value_head_dim))?;

        let q_n = l2_norm_4d(&q_h)?;
        let k_n = l2_norm_4d(&k_h)?;

        // Expand k-heads to v-heads if grouped
        let (q_e, k_e) = if v_per_group > 1 {
            let q_e = q_n
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch, seq_len, self.num_v_heads, self.key_head_dim))?;
            let k_e = k_n
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch, seq_len, self.num_v_heads, self.key_head_dim))?;
            (q_e, k_e)
        } else {
            (q_n, k_n)
        };

        // 6. Recurrence via decode slots kernel (in-place pool write).
        // Pack to [batch*num_v_heads, head_k_dim] and squeeze the seq dim.
        let scale = 1.0 / (self.key_head_dim as f64).sqrt();
        let bh = batch * self.num_v_heads;
        let q_bh = (q_e.transpose(1, 2)?.contiguous()? * scale)?
            .reshape((bh, self.key_head_dim))?
            .contiguous()?;
        let k_bh = k_e
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bh, self.key_head_dim))?
            .contiguous()?;
        let v_bh = v_h
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bh, self.value_head_dim))?
            .contiguous()?;
        let g_bh = g
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bh,))?
            .contiguous()?;
        let beta_bh = beta
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bh,))?
            .contiguous()?;

        let y_bh = crate::metal::gdn::gated_delta_rule_decode_slots_metal(
            &q_bh,
            &k_bh,
            &v_bh,
            &g_bh,
            &beta_bh,
            recurrent_state_pool,
            slots_gpu,
            self.num_v_heads,
        )?; // [bh, value_head_dim]

        // [bh, v_dim] → [batch, num_heads, 1, v_dim] → [batch, 1, num_heads, v_dim]
        let y = y_bh
            .reshape((batch, self.num_v_heads, seq_len, self.value_head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // 7. RMSNorm gated by z
        let val_shape = y.shape().clone();
        let y_flat = y.reshape(((), self.value_head_dim))?;
        let z_flat = z.reshape(((), self.value_head_dim))?;
        let y_normed = rms_norm_gated(&y_flat, &z_flat, &self.norm_weight, self.rms_norm_eps)?;
        let y_out = y_normed
            .reshape(val_shape)?
            .reshape((batch, seq_len, self.value_dim))?;

        // 8. Output projection
        Self::linear(&y_out, &self.out_proj)
    }
}

/// L2-normalise the last dimension (per-head) of a 4-D tensor.
fn l2_norm_4d(x: &Tensor) -> Result<Tensor> {
    let norm_sq = x.sqr()?.sum_keepdim(D::Minus1)?;
    let inv = (norm_sq + 1e-6_f64)?.sqrt()?.recip()?;
    x.broadcast_mul(&inv)
}

/// Per-row RMSNorm then scale by silu(gate), weight per head_v_dim.
fn rms_norm_gated(x: &Tensor, gate: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let gate_f32 = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
    let var = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = x_f32.broadcast_div(&(var + eps)?.sqrt()?)?;
    let out = normed
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .mul(&gate_f32)?;
    out.to_dtype(dtype)
}

// ---------------------------------------------------------------------------
// Layer abstraction
// ---------------------------------------------------------------------------

enum LayerImpl {
    FullAttention(FullAttnWeights),
    LinearAttention(GdnWeights),
}

struct Layer {
    layer_impl: LayerImpl,
    attention_norm: QRmsNorm,
    ffn_norm: QRmsNorm,
    mlp: Mlp,
}

// ---------------------------------------------------------------------------
// GGUF model props
// ---------------------------------------------------------------------------

pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    pub key_length: usize,
    pub value_length: usize,
    pub mrope_sections: Vec<usize>,
}

fn verify_qwen35_arch(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if actual != "qwen35" {
        candle_core::bail!("Expected `qwen35` architecture, got `{actual}`.");
    }
    Ok(actual)
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;
    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let _ = verify_qwen35_arch(c.metadata)?;
        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;
        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;
        let key_length = c
            .get_value::<u32>("attention.key_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(embed_len / head_count);
        // Parse mrope_sections if present (for multimodal RoPE)
        // Values may be stored as u64 or u32 in GGUF
        let mrope_sections: Vec<usize> =
            match c.get_option_value::<Vec<u64>>("rope.dimension_sections") {
                Ok(Some(v)) => v.iter().take(3).map(|x| *x as usize).collect(),
                _ => match c.get_option_value::<Vec<u32>>("rope.dimension_sections") {
                    Ok(Some(v)) => v.iter().take(3).map(|x| *x as usize).collect(),
                    _ => Vec::new(),
                },
            };

        Ok(Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
            rope_dim: c
                .get_value::<u32>("rope.dimension_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(key_length),
            key_length,
            value_length: c
                .get_value::<u32>("attention.value_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            mrope_sections,
        })
    }
}

// ---------------------------------------------------------------------------
// ModelWeights
// ---------------------------------------------------------------------------

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<Layer>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        let meta = ct.get_metadata();
        let actual_arch = verify_qwen35_arch(meta)?;

        let metadata = ContentMetadata {
            path_prefix: &actual_arch,
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
            rope_dim,
            key_length,
            value_length,
            mrope_sections,
        } = PropsGGUF::try_from(metadata).or_else(|e| candle_core::bail!("{e}"))?;

        let head_dim = key_length;
        if key_length != value_length {
            candle_core::bail!(
                "Expected key_length == value_length, got {key_length} != {value_length}"
            );
        }

        let hybrid = parse_hybrid_meta(ct.get_metadata(), &actual_arch, block_count);

        // Build per-device RoPE tables (Plain for text-only, MRope for multimodal)
        let mut ropes: HashMap<candle_core::DeviceLocation, Rotary> = HashMap::new();
        for layer_idx in 0..block_count {
            let dev = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.entry(dev.location()).or_insert_with(|| {
                if mrope_sections.is_empty() {
                    Rotary::Plain(Arc::new(
                        RotaryEmbedding::new_partial(
                            rope_freq_base,
                            rope_dim,
                            max_seq_len,
                            dev,
                            true,
                            DType::F32,
                        )
                        .expect("RoPE init"),
                    ))
                } else {
                    Rotary::MRope(Arc::new(
                        Qwen3VLRotaryEmbedding::new(
                            rope_freq_base,
                            rope_dim,
                            dev,
                            mrope_sections.clone(),
                        )
                        .expect("M-RoPE init"),
                    ))
                }
            });
        }

        // Global embeddings / norm / output
        let tok_embd_q = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = tok_embd_q.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?;
        let output = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };

        // Build per-layer weights
        let mut layers = Vec::with_capacity(block_count);
        let mut hybrid_layer_types = Vec::with_capacity(block_count);
        let mut gdn_layer_count = 0usize;

        let needs_untile = hybrid.num_k_heads != hybrid.num_v_heads;

        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let dev = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes[&dev.location()].clone();

            let attention_norm = ct.tensor(&format!("{prefix}.attn_norm.weight"), dev)?;
            let ffn_norm = ct.tensor(&format!("{prefix}.post_attention_norm.weight"), dev)?;

            let feed_forward_gate = ct.tensor(&format!("{prefix}.ffn_gate.weight"), dev)?;
            let feed_forward_up = ct.tensor(&format!("{prefix}.ffn_up.weight"), dev)?;
            let feed_forward_down = ct.tensor(&format!("{prefix}.ffn_down.weight"), dev)?;
            let mlp = Mlp {
                gate: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(feed_forward_gate),
                    b: None,
                })?),
                up: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(feed_forward_up),
                    b: None,
                })?),
                down: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(feed_forward_down),
                    b: None,
                })?),
            };

            let kind = &hybrid.layer_kinds[layer_idx];
            let layer_impl = match kind {
                GgufLayerKind::FullAttention => {
                    hybrid_layer_types.push(HybridLayerType::Attention);
                    let wq = ct.tensor(&format!("{prefix}.attn_q.weight"), dev)?;
                    let wk = ct.tensor(&format!("{prefix}.attn_k.weight"), dev)?;
                    let wv = ct.tensor(&format!("{prefix}.attn_v.weight"), dev)?;
                    let wo = ct.tensor(&format!("{prefix}.attn_output.weight"), dev)?;
                    // Detect attn_output_gate by checking Q output dim (= 2 * n_head * head_dim if gated)
                    let wq_out_dim = wq.shape().dims()[0];
                    let attn_output_gate = wq_out_dim == head_count * head_dim * 2;
                    let q_norm = QRmsNorm::new(
                        ct.tensor(&format!("{prefix}.attn_q_norm.weight"), dev)?,
                        rms_norm_eps,
                    )?;
                    let k_norm = QRmsNorm::new(
                        ct.tensor(&format!("{prefix}.attn_k_norm.weight"), dev)?,
                        rms_norm_eps,
                    )?;
                    let paged_attn = match &attention_mechanism {
                        AttentionImplementation::Eager => None,
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(head_dim, dev, None)?)
                        }
                    };
                    LayerImpl::FullAttention(FullAttnWeights {
                        wq: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                            q_weight: Arc::new(wq),
                            b: None,
                        })?),
                        wk: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                            q_weight: Arc::new(wk),
                            b: None,
                        })?),
                        wv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                            q_weight: Arc::new(wv),
                            b: None,
                        })?),
                        wo: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                            q_weight: Arc::new(wo),
                            b: None,
                        })?),
                        q_norm,
                        k_norm,
                        n_head: head_count,
                        n_kv_head: head_count_kv,
                        head_dim,
                        rope_dim,
                        attn_output_gate,
                        rotary,
                        paged_attn,
                        sdpa_params: SdpaParams {
                            n_kv_groups: head_count / head_count_kv,
                            softcap: None,
                            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                            sliding_window: None,
                            sinks: None,
                        },
                        dtype,
                    })
                }
                GgufLayerKind::LinearAttention => {
                    hybrid_layer_types.push(HybridLayerType::Recurrent);
                    gdn_layer_count += 1;

                    let mut load_dequant =
                        |name: &str| -> Result<Tensor> { ct.tensor(name, dev)?.dequantize(dev) };

                    // QKV + tiling
                    let qkv_raw = load_dequant(&format!("{prefix}.attn_qkv.weight"))?;
                    let q_w = qkv_raw.narrow(0, 0, hybrid.key_dim)?;
                    let k_w = qkv_raw.narrow(0, hybrid.key_dim, hybrid.key_dim)?;
                    let v_raw = qkv_raw.narrow(0, hybrid.key_dim * 2, hybrid.value_dim)?;
                    let v_w = undo_tiled_v_heads_first_dim(
                        &v_raw,
                        hybrid.num_k_heads,
                        hybrid.num_v_heads,
                        hybrid.value_head_dim,
                    )?;
                    let in_proj_qkv = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;

                    // Gate (Z) + tiling
                    let z_raw = load_dequant(&format!("{prefix}.attn_gate.weight"))?;
                    let in_proj_z = undo_tiled_v_heads_first_dim(
                        &z_raw,
                        hybrid.num_k_heads,
                        hybrid.num_v_heads,
                        hybrid.value_head_dim,
                    )?;

                    // Beta + tiling (dim=1 per-head scalar)
                    let beta_raw = load_dequant(&format!("{prefix}.ssm_beta.weight"))?;
                    let in_proj_beta = undo_tiled_v_heads_first_dim(
                        &beta_raw,
                        hybrid.num_k_heads,
                        hybrid.num_v_heads,
                        1,
                    )?;

                    // Alpha + tiling
                    let alpha_raw = load_dequant(&format!("{prefix}.ssm_alpha.weight"))?;
                    let in_proj_alpha = undo_tiled_v_heads_first_dim(
                        &alpha_raw,
                        hybrid.num_k_heads,
                        hybrid.num_v_heads,
                        1,
                    )?;

                    // Output projection + tiling (last dim)
                    let out_raw = load_dequant(&format!("{prefix}.ssm_out.weight"))?;
                    let out_proj = if needs_untile {
                        undo_tiled_v_heads_last_dim(
                            &out_raw,
                            hybrid.num_k_heads,
                            hybrid.num_v_heads,
                            hybrid.value_head_dim,
                        )?
                    } else {
                        out_raw
                    };

                    // Conv1d weights
                    let conv_raw = ct
                        .tensor(&format!("{prefix}.ssm_conv1d.weight"), dev)?
                        .dequantize(dev)?;
                    let conv_weight = if conv_raw.dims().len() == 2 {
                        conv_raw.unsqueeze(1)?
                    } else {
                        conv_raw
                    };
                    let conv_weight = if needs_untile {
                        let conv_key_dim = hybrid.num_k_heads * hybrid.key_head_dim;
                        let conv_val_dim = hybrid.num_v_heads * hybrid.value_head_dim;
                        let q_c = conv_weight.narrow(0, 0, conv_key_dim)?;
                        let k_c = conv_weight.narrow(0, conv_key_dim, conv_key_dim)?;
                        let v_c = conv_weight.narrow(0, conv_key_dim * 2, conv_val_dim)?;
                        let v_c = undo_tiled_v_heads_first_dim(
                            &v_c,
                            hybrid.num_k_heads,
                            hybrid.num_v_heads,
                            hybrid.value_head_dim,
                        )?;
                        Tensor::cat(&[&q_c, &k_c, &v_c], 0)?
                    } else {
                        conv_weight
                    };

                    let conv_bias = ct
                        .tensor(&format!("{prefix}.ssm_conv1d.bias"), dev)
                        .ok()
                        .map(|qt| qt.dequantize(dev))
                        .transpose()?;
                    let conv_bias = if needs_untile {
                        conv_bias
                            .map(|cb| {
                                let conv_key_dim = hybrid.num_k_heads * hybrid.key_head_dim;
                                let conv_val_dim = hybrid.num_v_heads * hybrid.value_head_dim;
                                let q_b = cb.narrow(0, 0, conv_key_dim)?;
                                let k_b = cb.narrow(0, conv_key_dim, conv_key_dim)?;
                                let v_b = cb.narrow(0, conv_key_dim * 2, conv_val_dim)?;
                                let v_b = undo_tiled_v_heads_first_dim(
                                    &v_b,
                                    hybrid.num_k_heads,
                                    hybrid.num_v_heads,
                                    hybrid.value_head_dim,
                                )?;
                                Tensor::cat(&[&q_b, &k_b, &v_b], 0)
                            })
                            .transpose()?
                    } else {
                        conv_bias
                    };

                    // A parameter (GGUF stores raw a; convert to a_log = (-a).log())
                    let a_raw = ct
                        .tensor(&format!("{prefix}.ssm_a"), dev)?
                        .dequantize(dev)?
                        .to_dtype(DType::F32)?;
                    let a_log = a_raw.neg()?.log()?;
                    let a_log = if needs_untile {
                        undo_tiled_v_heads_first_dim(
                            &a_log,
                            hybrid.num_k_heads,
                            hybrid.num_v_heads,
                            1,
                        )?
                    } else {
                        a_log
                    };

                    let dt_bias = ct
                        .tensor(&format!("{prefix}.ssm_dt.bias"), dev)?
                        .dequantize(dev)?
                        .to_dtype(DType::F32)?;
                    let dt_bias = if needs_untile {
                        undo_tiled_v_heads_first_dim(
                            &dt_bias,
                            hybrid.num_k_heads,
                            hybrid.num_v_heads,
                            1,
                        )?
                    } else {
                        dt_bias
                    };

                    let norm_weight = ct
                        .tensor(&format!("{prefix}.ssm_norm.weight"), dev)?
                        .dequantize(dev)?
                        .to_dtype(DType::F32)?;

                    // Pre-transpose projection weights once at init so per-token
                    // forward avoids re-transposing them on every call.
                    let in_proj_qkv = in_proj_qkv.t()?.contiguous()?;
                    let in_proj_z = in_proj_z.t()?.contiguous()?;
                    let in_proj_beta = in_proj_beta.t()?.contiguous()?;
                    let in_proj_alpha = in_proj_alpha.t()?.contiguous()?;
                    let out_proj = out_proj.t()?.contiguous()?;

                    LayerImpl::LinearAttention(GdnWeights {
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_beta,
                        in_proj_alpha,
                        out_proj,
                        conv_weight,
                        conv_bias,
                        a_log,
                        dt_bias,
                        norm_weight,
                        num_k_heads: hybrid.num_k_heads,
                        num_v_heads: hybrid.num_v_heads,
                        key_head_dim: hybrid.key_head_dim,
                        value_head_dim: hybrid.value_head_dim,
                        key_dim: hybrid.key_dim,
                        value_dim: hybrid.value_dim,
                        conv_kernel_size: hybrid.conv_kernel_size,
                        rms_norm_eps: rms_norm_eps as f64,
                    })
                }
            };

            layers.push(Layer {
                layer_impl,
                attention_norm: QRmsNorm::new(attention_norm, rms_norm_eps)?,
                ffn_norm: QRmsNorm::new(ffn_norm, rms_norm_eps)?,
                mlp,
            });
        }

        // Build HybridCache
        let conv_dim = if hybrid.num_k_heads > 0 {
            hybrid.key_dim * 2 + hybrid.value_dim
        } else {
            1
        };
        let state_dims = if hybrid.num_v_heads > 0 {
            vec![
                hybrid.num_v_heads,
                hybrid.key_head_dim,
                hybrid.value_head_dim,
            ]
        } else {
            vec![1, 1, 1]
        };
        let recurrent_cfg = RecurrentLayerConfig {
            conv_dim,
            conv_width: hybrid.conv_kernel_size,
            state_dims,
            recurrent_dtype: Some(dtype),
        };
        let cache = EitherCache::Hybrid(std::sync::Arc::new(std::sync::Mutex::new(
            HybridCache::new(
                HybridCacheConfig {
                    layer_types: hybrid_layer_types,
                    max_seq_len,
                    recurrent: recurrent_cfg,
                },
                DType::F32,
                device,
            )?,
        )));

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(output),
                b: None,
            })?),
            device: device.clone(),
            cache,
            max_seq_len,
            mapper: Some(mapper),
            dtype,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(x)?;

        let mut hybrid_cache = self.cache.hybrid();
        let state_indices = hybrid_cache.state_indices().cloned();

        let mask = CausalMasker.make_causal_mask(
            x,
            metadata
                .as_ref()
                .map(|_| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*hybrid_cache as &dyn PastKvLenCache),
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let mask = if metadata
            .as_ref()
            .map(|(_, m)| m.is_first_prompt_chunk)
            .unwrap_or(true)
        {
            mask
        } else {
            AttentionMask::None
        };
        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;

            let attn_out = match &layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    if let Some(crate::kv_cache::HybridLayerCache::Attention(kv_cache)) =
                        hybrid_cache.get_mut(i)
                    {
                        attn.forward(
                            &x,
                            &mask.get(x.device()),
                            start_offsets,
                            kv_cache,
                            metadata.as_ref().map(|(kv, m)| (kv[i].clone(), *m)),
                        )?
                    } else {
                        candle_core::bail!(
                            "Hybrid cache layer {i} is not Attention for a full-attention layer"
                        );
                    }
                }
                LayerImpl::LinearAttention(gdn) => {
                    if let Some(crate::kv_cache::HybridLayerCache::Recurrent(pool)) =
                        hybrid_cache.get_mut(i)
                    {
                        let indices = state_indices.as_ref().ok_or_else(|| {
                            candle_core::Error::Msg(
                                "GDN layers require recurrent state indices (paged-attn mode)"
                                    .into(),
                            )
                        })?;
                        let indices_vec: Vec<u32> = indices.to_vec1()?;
                        // Decode when we already have a prefill (start_offsets[0] > 0) and
                        // the current step is a single token.
                        let is_decode =
                            start_offsets.first().copied().unwrap_or(0) > 0 && x.dim(1)? == 1;
                        let seqlen_offset = start_offsets.first().copied().unwrap_or(0);

                        // Metal decode fast path: address pool directly via slots,
                        // skipping per-layer gather/scatter (~112 Metal dispatches/token saved).
                        #[cfg(feature = "metal")]
                        {
                            if x.device().is_metal() && is_decode {
                                let slots_gpu = indices.to_device(x.device())?;
                                let out = gdn.forward_decode_slots(
                                    &x,
                                    &mut pool.conv_state,
                                    &mut pool.recurrent_state,
                                    &slots_gpu,
                                )?;
                                out
                            } else {
                                let conv_state = pool.gather_conv_state(indices)?;
                                let recurrent_state = pool.gather_recurrent_state(indices)?;
                                let mut gdn_cache = GdnLayerCache {
                                    conv_state,
                                    recurrent_state,
                                    seqlen_offset,
                                };
                                let out = gdn.forward(&x, &mut gdn_cache)?;
                                pool.scatter_conv_state(indices, &gdn_cache.conv_state)?;
                                pool.scatter_recurrent_state(indices, &gdn_cache.recurrent_state)?;
                                out
                            }
                        }
                        #[cfg(not(feature = "metal"))]
                        {
                            let conv_state = pool.gather_conv_state(indices)?;
                            let recurrent_state = pool.gather_recurrent_state(indices)?;
                            let mut gdn_cache = GdnLayerCache {
                                conv_state,
                                recurrent_state,
                                seqlen_offset,
                            };
                            let out = gdn.forward(&x, &mut gdn_cache)?;
                            pool.scatter_conv_state(indices, &gdn_cache.conv_state)?;
                            pool.scatter_recurrent_state(indices, &gdn_cache.recurrent_state)?;
                            out
                        }
                    } else {
                        candle_core::bail!(
                            "Hybrid cache layer {i} is not Recurrent for a GDN layer"
                        );
                    }
                }
            };

            let x = (attn_out + residual)?;
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            layer_in = (x + residual)?;
        }

        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.forward(&x.contiguous()?)
    }
}
