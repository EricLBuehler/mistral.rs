use candle_core::{DType, IndexOp, Result, Tensor, D};

use super::cache::GdnLayerCache;
use super::config::GdnDims;
use crate::pipeline::RecurrentBatchKind;

#[cfg(any(feature = "cuda", feature = "metal"))]
const RECURRENCE_CHUNK_THRESHOLD: usize = 64;
const QK_NORM_EPS: f64 = 1e-6;

#[cfg(feature = "cuda")]
fn use_warp_prefill_recurrence(dims: &GdnDims) -> bool {
    matches!(dims.head_k_dim, 64 | 128)
}

pub fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let inv_norm = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .broadcast_add(&Tensor::new(eps as f32, x.device())?.to_dtype(x.dtype())?)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&inv_norm)
}

pub fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let dtype = q.dtype();
    let k_head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (k_head_dim as f64).sqrt();

    let q = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?;
    let k = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let v = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let g = g.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let beta = beta.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

    let seq_len = q.dim(2)?;
    let mut s = state.to_dtype(DType::F32)?;
    let mut outputs = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let q_t = q.i((.., .., i, ..))?;
        let k_t = k.i((.., .., i, ..))?;
        let v_t = v.i((.., .., i, ..))?;
        let g_t = g.i((.., .., i))?;
        let beta_t = beta.i((.., .., i))?;

        let decay = g_t.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        s = s.broadcast_mul(&decay)?;

        let k_exp = k_t.unsqueeze(D::Minus1)?;
        let kv_mem = s.broadcast_mul(&k_exp)?.sum(2)?;
        let beta_exp = beta_t.unsqueeze(D::Minus1)?;
        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_exp)?;

        let outer = k_exp.broadcast_mul(&delta.unsqueeze(2)?)?;
        s = (s + outer)?;

        let q_exp = q_t.unsqueeze(D::Minus1)?;
        let y_t = s.broadcast_mul(&q_exp)?.sum(2)?;
        outputs.push(y_t);
    }

    *state = s.to_dtype(state.dtype())?;

    Tensor::stack(&outputs, 2)?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(dtype)
}

pub fn compute_beta_g(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if b.device().is_cuda() {
        let b_flat = b.contiguous()?.flatten_all()?;
        let a_flat = a.contiguous()?.flatten_all()?;
        let a_log_f32 = a_log.to_dtype(DType::F32)?.contiguous()?;
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?.contiguous()?;
        let (beta_flat, g_flat) =
            crate::cuda::gdn::fused_gdn_gating_cuda(&b_flat, &a_flat, &a_log_f32, &dt_bias_f32)?;
        let shape = b.shape();
        return Ok((beta_flat.reshape(shape)?, g_flat.reshape(shape)?));
    }

    #[cfg(feature = "metal")]
    if b.device().is_metal() {
        let b_flat = b.contiguous()?.flatten_all()?;
        let a_flat = a.contiguous()?.flatten_all()?;
        let a_log_f32 = a_log.to_dtype(DType::F32)?.contiguous()?;
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?.contiguous()?;
        let (beta_flat, g_flat) =
            crate::metal::gdn::fused_gdn_gating_metal(&b_flat, &a_flat, &a_log_f32, &dt_bias_f32)?;
        let shape = b.shape();
        return Ok((beta_flat.reshape(shape)?, g_flat.reshape(shape)?));
    }

    compute_beta_g_cpu(b, a, a_log, dt_bias, dtype)
}

fn compute_beta_g_cpu(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let beta = candle_nn::ops::sigmoid(b)?;
    let a_f = a.to_dtype(DType::F32)?;
    let dt_bias_expanded = dt_bias.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?;
    let g = a_log
        .to_dtype(DType::F32)?
        .exp()?
        .neg()?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .broadcast_mul(&softplus(&a_f.broadcast_add(&dt_bias_expanded)?)?)?
        .to_dtype(dtype)?;
    Ok((beta, g))
}

#[allow(clippy::too_many_arguments)]
pub fn apply_recurrence_from_convolved(
    mixed_qkv: &Tensor,
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if mixed_qkv.device().is_cuda() {
        return recurrence_cuda_from_convolved(
            mixed_qkv, b, a, a_log, dt_bias, dims, batch_size, seq_len, cache, dtype,
        );
    }

    let q = mixed_qkv.narrow(D::Minus1, 0, dims.key_dim)?;
    let k = mixed_qkv.narrow(D::Minus1, dims.key_dim, dims.key_dim)?;
    let v = mixed_qkv.narrow(D::Minus1, dims.key_dim * 2, dims.value_dim)?;
    let q = q.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
    let k = k.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
    let v = v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?;
    let (q, k) = if dims.v_per_group > 1 {
        let q = q
            .unsqueeze(3)?
            .repeat((1, 1, 1, dims.v_per_group, 1))?
            .reshape((batch_size, seq_len, dims.num_v_heads, dims.head_k_dim))?;
        let k = k
            .unsqueeze(3)?
            .repeat((1, 1, 1, dims.v_per_group, 1))?
            .reshape((batch_size, seq_len, dims.num_v_heads, dims.head_k_dim))?;
        (q, k)
    } else {
        (q, k)
    };
    let (beta, g) = compute_beta_g(b, a, a_log, dt_bias, dtype)?;
    let q = l2_norm(&q, QK_NORM_EPS)?;
    let k = l2_norm(&k, QK_NORM_EPS)?;
    apply_recurrence(
        &q, &k, &v, &g, &beta, dims, batch_size, seq_len, cache, dtype,
    )
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn recurrence_cuda_from_convolved(
    mixed_qkv: &Tensor,
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    let mixed_qkv = mixed_qkv.contiguous()?;
    let b = b.contiguous()?;
    let a = a.contiguous()?;
    let a_log = a_log.to_dtype(DType::F32)?.contiguous()?;
    let dt_bias = dt_bias.to_dtype(DType::F32)?.contiguous()?;
    let mut state_flat = prepare_state_for_backend(cache, dims, batch_size)?;

    let out_bh = if seq_len == 1 {
        crate::cuda::gdn::fused_decode_recurrence_cuda(
            &mixed_qkv,
            &b,
            &a,
            &a_log,
            &dt_bias,
            &mut state_flat,
            batch_size,
            dims.num_k_heads,
            dims.num_v_heads,
            dims.head_k_dim,
            dims.head_v_dim,
        )?
    } else {
        let (q_bh, k_bh, v_bh, g_bh, beta_bh) = crate::cuda::gdn::prepare_recurrence_inputs_cuda(
            &mixed_qkv,
            &b,
            &a,
            &a_log,
            &dt_bias,
            batch_size,
            seq_len,
            dims.num_k_heads,
            dims.num_v_heads,
            dims.head_k_dim,
            dims.head_v_dim,
        )?;
        if seq_len >= RECURRENCE_CHUNK_THRESHOLD && use_warp_prefill_recurrence(dims) {
            crate::cuda::gdn::warp_gated_delta_rule_recurrence_cuda(
                &q_bh,
                &k_bh,
                &v_bh,
                &g_bh,
                &beta_bh,
                &mut state_flat,
            )?
        } else if seq_len >= RECURRENCE_CHUNK_THRESHOLD {
            crate::cuda::gdn::chunked_gated_delta_rule_recurrence_cuda(
                &q_bh,
                &k_bh,
                &v_bh,
                &g_bh,
                &beta_bh,
                &mut state_flat,
            )?
        } else {
            crate::cuda::gdn::gated_delta_rule_recurrence_cuda(
                &q_bh,
                &k_bh,
                &v_bh,
                &g_bh,
                &beta_bh,
                &mut state_flat,
            )?
        }
    };

    finish_recurrence(out_bh, state_flat, dims, batch_size, seq_len, cache, dtype)
}

#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
pub fn apply_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        return recurrence_cuda(q, k, v, g, beta, dims, batch_size, seq_len, cache, dtype);
    }

    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        return recurrence_metal(q, k, v, g, beta, dims, batch_size, seq_len, cache, dtype);
    }

    gated_delta_rule_recurrence(q, k, v, g, beta, &mut cache.recurrent_state)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn recurrence_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    let q_bh = prepare_q_for_backend(q, dims, batch_size, seq_len)?;
    let k_bh = prepare_kv_for_backend(k, dims, batch_size, seq_len, dims.head_k_dim)?;
    let v_bh = prepare_kv_for_backend(v, dims, batch_size, seq_len, dims.head_v_dim)?;
    let g_bh = prepare_gate_for_backend(g, dims, batch_size, seq_len)?;
    let beta_bh = prepare_gate_for_backend(beta, dims, batch_size, seq_len)?;
    let mut state_flat = prepare_state_for_backend(cache, dims, batch_size)?;

    let out_bh = if seq_len >= RECURRENCE_CHUNK_THRESHOLD && use_warp_prefill_recurrence(dims) {
        crate::cuda::gdn::warp_gated_delta_rule_recurrence_cuda(
            &q_bh,
            &k_bh,
            &v_bh,
            &g_bh,
            &beta_bh,
            &mut state_flat,
        )?
    } else if seq_len >= RECURRENCE_CHUNK_THRESHOLD {
        crate::cuda::gdn::chunked_gated_delta_rule_recurrence_cuda(
            &q_bh,
            &k_bh,
            &v_bh,
            &g_bh,
            &beta_bh,
            &mut state_flat,
        )?
    } else {
        crate::cuda::gdn::gated_delta_rule_recurrence_cuda(
            &q_bh,
            &k_bh,
            &v_bh,
            &g_bh,
            &beta_bh,
            &mut state_flat,
        )?
    };

    finish_recurrence(out_bh, state_flat, dims, batch_size, seq_len, cache, dtype)
}

#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
fn recurrence_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    let q_bh = prepare_q_for_backend(q, dims, batch_size, seq_len)?;
    let k_bh = prepare_kv_for_backend(k, dims, batch_size, seq_len, dims.head_k_dim)?;
    let v_bh = prepare_kv_for_backend(v, dims, batch_size, seq_len, dims.head_v_dim)?;
    let g_bh = prepare_gate_for_backend(g, dims, batch_size, seq_len)?;
    let beta_bh = prepare_gate_for_backend(beta, dims, batch_size, seq_len)?;
    let mut state_flat = prepare_state_for_backend(cache, dims, batch_size)?;

    let out_bh = if seq_len >= RECURRENCE_CHUNK_THRESHOLD {
        crate::metal::gdn::chunked_gated_delta_rule_recurrence_metal(
            &q_bh,
            &k_bh,
            &v_bh,
            &g_bh,
            &beta_bh,
            &mut state_flat,
        )?
    } else {
        crate::metal::gdn::gated_delta_rule_recurrence_metal(
            &q_bh,
            &k_bh,
            &v_bh,
            &g_bh,
            &beta_bh,
            &mut state_flat,
        )?
    };

    finish_recurrence(out_bh, state_flat, dims, batch_size, seq_len, cache, dtype)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn prepare_q_for_backend(
    q: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
) -> Result<Tensor> {
    let scale = 1.0 / (dims.head_k_dim as f64).sqrt();
    (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?
        .reshape((batch_size * dims.num_v_heads, seq_len, dims.head_k_dim))?
        .contiguous()
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn prepare_kv_for_backend(
    x: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Tensor> {
    x.transpose(1, 2)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .reshape((batch_size * dims.num_v_heads, seq_len, head_dim))?
        .contiguous()
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn prepare_gate_for_backend(
    x: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
) -> Result<Tensor> {
    x.to_dtype(DType::F32)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch_size * dims.num_v_heads, seq_len))?
        .contiguous()
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn prepare_state_for_backend(
    cache: &GdnLayerCache,
    dims: &GdnDims,
    batch_size: usize,
) -> Result<Tensor> {
    cache
        .recurrent_state
        .to_dtype(DType::F32)?
        .reshape((
            batch_size * dims.num_v_heads,
            dims.head_k_dim,
            dims.head_v_dim,
        ))?
        .contiguous()
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn finish_recurrence(
    out_bh: Tensor,
    state_flat: Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    cache.recurrent_state = state_flat
        .reshape((
            batch_size,
            dims.num_v_heads,
            dims.head_k_dim,
            dims.head_v_dim,
        ))?
        .to_dtype(cache.recurrent_state.dtype())?;

    out_bh
        .reshape((batch_size, dims.num_v_heads, seq_len, dims.head_v_dim))?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(dtype)
}

pub fn causal_conv1d(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
    batch_kind: RecurrentBatchKind,
) -> Result<Tensor> {
    let (_, seq_len, _) = x.dims3()?;
    if matches!(batch_kind, RecurrentBatchKind::Decode) {
        if seq_len != 1 {
            candle_core::bail!("GDN decode expects a single-token query.");
        }
        causal_conv1d_update(x, conv1d_weight, dims, cache)
    } else {
        causal_conv1d_full(x, conv1d_weight, dims, cache)
    }
}

fn causal_conv1d_update(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let (_, seq_len, _) = x.dims3()?;
    let x_t = x.transpose(1, 2)?.contiguous()?;

    #[cfg(feature = "cuda")]
    if x_t.device().is_cuda() {
        let weight = conv1d_weight
            .squeeze(1)?
            .to_dtype(x_t.dtype())?
            .contiguous()?;
        let conv_state = cache.conv_state.contiguous()?;
        let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
            &x_t,
            &weight,
            &conv_state,
            dims.conv_kernel_size,
            true,
        )?;
        cache.conv_state = new_conv_state;
        return output.transpose(1, 2);
    }

    #[cfg(feature = "metal")]
    if x_t.device().is_metal() {
        let weight = conv1d_weight
            .squeeze(1)?
            .to_dtype(x_t.dtype())?
            .contiguous()?;
        let conv_state = cache.conv_state.contiguous()?;
        let (output, new_conv_state) = crate::metal::gdn::causal_conv1d_metal(
            &x_t,
            &weight,
            &conv_state,
            true,
            dims.conv_kernel_size,
        )?;
        cache.conv_state = new_conv_state;
        return output.transpose(1, 2);
    }

    let state_len = cache.conv_state.dim(2)?;
    let hidden_new = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
    let new_len = hidden_new.dim(2)?;
    cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

    let weight = conv1d_weight.squeeze(1)?.to_dtype(hidden_new.dtype())?;
    let mut conv_outputs = Vec::with_capacity(seq_len);
    let total_len = hidden_new.dim(2)?;
    for i in (total_len - seq_len)..total_len {
        let window = hidden_new.narrow(2, i + 1 - dims.conv_kernel_size, dims.conv_kernel_size)?;
        let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
        conv_outputs.push(out);
    }
    candle_nn::ops::silu(&Tensor::stack(&conv_outputs, 2)?)?.transpose(1, 2)
}

fn causal_conv1d_full(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let (batch_size, seq_len, conv_dim) = x.dims3()?;
    let x_t = x.transpose(1, 2)?.contiguous()?;

    #[cfg(feature = "cuda")]
    if x_t.device().is_cuda() {
        let weight = conv1d_weight
            .squeeze(1)?
            .to_dtype(x_t.dtype())?
            .contiguous()?;
        let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
            &x_t,
            &weight,
            &cache.conv_state,
            dims.conv_kernel_size,
            false,
        )?;
        cache.conv_state = new_conv_state;
        return output.transpose(1, 2);
    }

    #[cfg(feature = "metal")]
    if x_t.device().is_metal() {
        let weight = conv1d_weight
            .squeeze(1)?
            .to_dtype(x_t.dtype())?
            .contiguous()?;
        let (output, new_conv_state) = crate::metal::gdn::causal_conv1d_metal(
            &x_t,
            &weight,
            &cache.conv_state,
            false,
            dims.conv_kernel_size,
        )?;
        cache.conv_state = new_conv_state;
        return output.transpose(1, 2);
    }

    let pad_width = dims.conv_kernel_size.saturating_sub(seq_len);
    cache.conv_state = if pad_width > 0 {
        let zeros = Tensor::zeros((batch_size, conv_dim, pad_width), x_t.dtype(), x_t.device())?;
        Tensor::cat(&[zeros, x_t.clone()], 2)?
    } else {
        x_t.narrow(2, seq_len - dims.conv_kernel_size, dims.conv_kernel_size)?
    };

    let padded_t = Tensor::cat(
        &[
            Tensor::zeros(
                (batch_size, conv_dim, dims.conv_kernel_size - 1),
                x_t.dtype(),
                x_t.device(),
            )?,
            x_t,
        ],
        2,
    )?;

    let weight = conv1d_weight.squeeze(1)?.to_dtype(padded_t.dtype())?;
    let mut conv_outputs = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let window = padded_t.narrow(2, i, dims.conv_kernel_size)?;
        let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
        conv_outputs.push(out);
    }
    candle_nn::ops::silu(&Tensor::stack(&conv_outputs, 2)?)?.transpose(1, 2)
}
