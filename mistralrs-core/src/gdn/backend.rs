use candle_core::{DType, IndexOp, Result, Storage, Tensor, D};
use rayon::prelude::*;

use super::cache::GdnLayerCache;
use super::config::GdnDims;
use crate::pipeline::RecurrentBatchKind;

#[cfg(any(feature = "cuda", feature = "metal"))]
const RECURRENCE_CHUNK_THRESHOLD: usize = 64;
const QK_NORM_EPS: f64 = 1e-6;
const QK_NORM_EPS_F32: f32 = 1e-6;
const SOFTPLUS_LINEAR_THRESHOLD: f32 = 20.0;
const DECODE_STACK_HEAD_K_DIM: usize = 256;

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

    if seq_len == 1 && mixed_qkv.device().is_cpu() {
        return decode_recurrence_cpu_from_convolved(
            mixed_qkv, b, a, a_log, dt_bias, dims, batch_size, cache, dtype,
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

#[allow(clippy::too_many_arguments)]
fn decode_recurrence_cpu_from_convolved(
    mixed_qkv: &Tensor,
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    let dev = mixed_qkv.device();
    let mixed_f32 = mixed_qkv.to_dtype(DType::F32)?.contiguous()?;
    let b_f32 = b.to_dtype(DType::F32)?.contiguous()?;
    let a_f32 = a.to_dtype(DType::F32)?.contiguous()?;
    let a_log_f32 = a_log.to_dtype(DType::F32)?.contiguous()?;
    let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?.contiguous()?;

    let (mixed_storage, mixed_layout) = mixed_f32.storage_and_layout();
    let mixed = cpu_f32_slice(&mixed_storage, mixed_layout.start_offset(), "mixed_qkv")?;
    let (b_storage, b_layout) = b_f32.storage_and_layout();
    let b = cpu_f32_slice(&b_storage, b_layout.start_offset(), "b")?;
    let (a_storage, a_layout) = a_f32.storage_and_layout();
    let a = cpu_f32_slice(&a_storage, a_layout.start_offset(), "a")?;
    let (a_log_storage, a_log_layout) = a_log_f32.storage_and_layout();
    let a_log = cpu_f32_slice(&a_log_storage, a_log_layout.start_offset(), "a_log")?;
    let (dt_bias_storage, dt_bias_layout) = dt_bias_f32.storage_and_layout();
    let dt_bias = cpu_f32_slice(&dt_bias_storage, dt_bias_layout.start_offset(), "dt_bias")?;
    let mut state = cache
        .recurrent_state
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut output = vec![0.0f32; batch_size * dims.num_v_heads * dims.head_v_dim];
    let q_scale = 1.0f32 / (dims.head_k_dim as f32).sqrt();
    let state_head_len = dims.head_k_dim * dims.head_v_dim;

    state
        .par_chunks_mut(state_head_len)
        .zip(output.par_chunks_mut(dims.head_v_dim))
        .enumerate()
        .for_each(|(gate_idx, (state_head, out_head))| {
            let bidx = gate_idx / dims.num_v_heads;
            let hv = gate_idx % dims.num_v_heads;
            let hk = hv / dims.v_per_group;
            let row = bidx * dims.conv_dim;
            let q_base = row + hk * dims.head_k_dim;
            let k_base = row + dims.key_dim + hk * dims.head_k_dim;
            let v_base = row + 2 * dims.key_dim + hv * dims.head_v_dim;

            let mut q_sum = 0.0f32;
            let mut k_sum = 0.0f32;
            for d in 0..dims.head_k_dim {
                let q = mixed[q_base + d];
                let k = mixed[k_base + d];
                q_sum += q * q;
                k_sum += k * k;
            }

            let q_mul = q_scale / (q_sum + QK_NORM_EPS_F32).sqrt();
            let k_mul = 1.0f32 / (k_sum + QK_NORM_EPS_F32).sqrt();
            let mut q_stack = [0.0f32; DECODE_STACK_HEAD_K_DIM];
            let mut k_stack = [0.0f32; DECODE_STACK_HEAD_K_DIM];
            let mut q_heap;
            let mut k_heap;
            let (q_buf, k_buf) = if dims.head_k_dim <= DECODE_STACK_HEAD_K_DIM {
                (
                    &mut q_stack[..dims.head_k_dim],
                    &mut k_stack[..dims.head_k_dim],
                )
            } else {
                q_heap = vec![0.0f32; dims.head_k_dim];
                k_heap = vec![0.0f32; dims.head_k_dim];
                (q_heap.as_mut_slice(), k_heap.as_mut_slice())
            };
            for d in 0..dims.head_k_dim {
                q_buf[d] = mixed[q_base + d] * q_mul;
                k_buf[d] = mixed[k_base + d] * k_mul;
            }

            let beta = sigmoid_f32(b[gate_idx]);
            let decay = (-a_log[hv].exp() * softplus_f32(a[gate_idx] + dt_bias[hv])).exp();

            for v_idx in 0..dims.head_v_dim {
                let mut kv_mem = 0.0f32;
                for (k_idx, &k) in k_buf.iter().enumerate() {
                    let state_idx = k_idx * dims.head_v_dim + v_idx;
                    let s = state_head[state_idx] * decay;
                    state_head[state_idx] = s;
                    kv_mem += s * k;
                }

                let delta = (mixed[v_base + v_idx] - kv_mem) * beta;
                let mut y = 0.0f32;
                for (k_idx, (&k, &q)) in k_buf.iter().zip(q_buf.iter()).enumerate() {
                    let state_idx = k_idx * dims.head_v_dim + v_idx;
                    let s = state_head[state_idx] + k * delta;
                    state_head[state_idx] = s;
                    y += s * q;
                }
                out_head[v_idx] = y;
            }
        });

    cache.recurrent_state = Tensor::from_vec(
        state,
        (
            batch_size,
            dims.num_v_heads,
            dims.head_k_dim,
            dims.head_v_dim,
        ),
        dev,
    )?
    .to_dtype(cache.recurrent_state.dtype())?;

    Tensor::from_vec(
        output,
        (batch_size, 1, dims.num_v_heads, dims.head_v_dim),
        dev,
    )?
    .to_dtype(dtype)
}

fn cpu_f32_slice<'a>(
    storage: &'a Storage,
    start_offset: usize,
    name: &'static str,
) -> Result<&'a [f32]> {
    let Storage::Cpu(cpu) = storage else {
        candle_core::bail!("Expected CPU storage for {name}");
    };
    let data = cpu.as_slice::<f32>()?;
    Ok(&data[start_offset..])
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus_f32(x: f32) -> f32 {
    if x > SOFTPLUS_LINEAR_THRESHOLD {
        x
    } else if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
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

    if x.device().is_cpu() {
        return causal_conv1d_update_cpu(x, conv1d_weight, dims, cache);
    }

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

fn causal_conv1d_update_cpu(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let (batch_size, seq_len, conv_dim) = x.dims3()?;
    if seq_len != 1 {
        candle_core::bail!("GDN CPU conv decode expects a single-token query.");
    }

    let dev = x.device();
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?.contiguous()?;
    let weight_f32 = conv1d_weight.to_dtype(DType::F32)?.contiguous()?;
    let (x_storage, x_layout) = x_f32.storage_and_layout();
    let x = cpu_f32_slice(&x_storage, x_layout.start_offset(), "x")?;
    let (weight_storage, weight_layout) = weight_f32.storage_and_layout();
    let weight = cpu_f32_slice(
        &weight_storage,
        weight_layout.start_offset(),
        "conv1d_weight",
    )?;
    let mut state = cache
        .conv_state
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut output = vec![0.0f32; batch_size * conv_dim];

    state
        .par_chunks_mut(dims.conv_kernel_size)
        .zip(output.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (state_channel, output))| {
            let cidx = idx % conv_dim;
            for kidx in 1..dims.conv_kernel_size {
                state_channel[kidx - 1] = state_channel[kidx];
            }
            state_channel[dims.conv_kernel_size - 1] = x[idx];

            let weight_base = cidx * dims.conv_kernel_size;
            let mut sum = 0.0f32;
            for kidx in 0..dims.conv_kernel_size {
                sum += state_channel[kidx] * weight[weight_base + kidx];
            }
            *output = silu_f32(sum);
        });

    cache.conv_state = Tensor::from_vec(state, (batch_size, conv_dim, dims.conv_kernel_size), dev)?
        .to_dtype(cache.conv_state.dtype())?;

    Tensor::from_vec(output, (batch_size, 1, conv_dim), dev)?.to_dtype(dtype)
}

fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Result as CandleResult};

    const ASSERT_EPS: f32 = 5e-5;

    fn patterned(len: usize, salt: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i.wrapping_mul(37) + salt.wrapping_mul(17)) % 257) as f32;
                ((x / 128.0) - 1.0) * scale + offset
            })
            .collect()
    }

    fn dims(
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> GdnDims {
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        GdnDims {
            hidden_size: value_dim,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size: 4,
            key_dim,
            value_dim,
            conv_dim: key_dim * 2 + value_dim,
            v_per_group: num_v_heads / num_k_heads,
        }
    }

    fn assert_close(lhs: &Tensor, rhs: &Tensor) -> CandleResult<()> {
        let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
        let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&lhs, &rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= ASSERT_EPS,
                "idx={idx} lhs={lhs} rhs={rhs} diff={diff}"
            );
        }
        Ok(())
    }

    fn run_decode_case(dims: GdnDims, batch_size: usize) -> CandleResult<()> {
        let dev = Device::Cpu;
        let seq_len = 1;
        let mixed = Tensor::from_vec(
            patterned(batch_size * dims.conv_dim, 1, 0.08, 0.01),
            (batch_size, seq_len, dims.conv_dim),
            &dev,
        )?;
        let b = Tensor::from_vec(
            patterned(batch_size * dims.num_v_heads, 2, 0.2, 0.1),
            (batch_size, seq_len, dims.num_v_heads),
            &dev,
        )?;
        let a = Tensor::from_vec(
            patterned(batch_size * dims.num_v_heads, 3, 0.2, -0.05),
            (batch_size, seq_len, dims.num_v_heads),
            &dev,
        )?;
        let a_log = Tensor::from_vec(
            patterned(dims.num_v_heads, 4, 0.05, -0.2),
            (dims.num_v_heads,),
            &dev,
        )?;
        let dt_bias = Tensor::from_vec(
            patterned(dims.num_v_heads, 5, 0.1, 0.3),
            (dims.num_v_heads,),
            &dev,
        )?;
        let initial_state = Tensor::from_vec(
            patterned(
                batch_size * dims.num_v_heads * dims.head_k_dim * dims.head_v_dim,
                6,
                0.02,
                0.0,
            ),
            (
                batch_size,
                dims.num_v_heads,
                dims.head_k_dim,
                dims.head_v_dim,
            ),
            &dev,
        )?;
        let conv_state = Tensor::zeros(
            (batch_size, dims.conv_dim, dims.conv_kernel_size),
            DType::F32,
            &dev,
        )?;
        let mut fast_cache = GdnLayerCache {
            conv_state: conv_state.clone(),
            recurrent_state: initial_state.clone(),
        };
        let fast = decode_recurrence_cpu_from_convolved(
            &mixed,
            &b,
            &a,
            &a_log,
            &dt_bias,
            &dims,
            batch_size,
            &mut fast_cache,
            DType::F32,
        )?;

        let q = mixed.narrow(D::Minus1, 0, dims.key_dim)?;
        let k = mixed.narrow(D::Minus1, dims.key_dim, dims.key_dim)?;
        let v = mixed.narrow(D::Minus1, dims.key_dim * 2, dims.value_dim)?;
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
        let (beta, g) = compute_beta_g(&b, &a, &a_log, &dt_bias, DType::F32)?;
        let q = l2_norm(&q, QK_NORM_EPS)?;
        let k = l2_norm(&k, QK_NORM_EPS)?;
        let mut reference_cache = GdnLayerCache {
            conv_state,
            recurrent_state: initial_state,
        };
        let reference = gated_delta_rule_recurrence(
            &q,
            &k,
            &v,
            &g,
            &beta,
            &mut reference_cache.recurrent_state,
        )?;

        assert_close(&fast, &reference)?;
        assert_close(
            &fast_cache.recurrent_state,
            &reference_cache.recurrent_state,
        )
    }

    fn causal_conv1d_update_reference(
        x: &Tensor,
        conv1d_weight: &Tensor,
        dims: &GdnDims,
        cache: &mut GdnLayerCache,
    ) -> CandleResult<Tensor> {
        let (_, seq_len, _) = x.dims3()?;
        let x_t = x.transpose(1, 2)?.contiguous()?;
        let state_len = cache.conv_state.dim(2)?;
        let hidden_new = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
        let new_len = hidden_new.dim(2)?;
        cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

        let weight = conv1d_weight.squeeze(1)?.to_dtype(hidden_new.dtype())?;
        let mut conv_outputs = Vec::with_capacity(seq_len);
        let total_len = hidden_new.dim(2)?;
        for i in (total_len - seq_len)..total_len {
            let window =
                hidden_new.narrow(2, i + 1 - dims.conv_kernel_size, dims.conv_kernel_size)?;
            let out = window
                .broadcast_mul(&weight.unsqueeze(0)?)?
                .sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        candle_nn::ops::silu(&Tensor::stack(&conv_outputs, 2)?)?.transpose(1, 2)
    }

    #[test]
    fn causal_conv1d_update_cpu_matches_tensor_path() -> CandleResult<()> {
        let dev = Device::Cpu;
        let dims = dims(2, 4, 5, 3);
        let batch_size = 2;
        let x = Tensor::from_vec(
            patterned(batch_size * dims.conv_dim, 7, 0.08, 0.01),
            (batch_size, 1, dims.conv_dim),
            &dev,
        )?;
        let weight = Tensor::from_vec(
            patterned(dims.conv_dim * dims.conv_kernel_size, 8, 0.05, -0.01),
            (dims.conv_dim, 1, dims.conv_kernel_size),
            &dev,
        )?;
        let initial_state = Tensor::from_vec(
            patterned(
                batch_size * dims.conv_dim * dims.conv_kernel_size,
                9,
                0.03,
                0.0,
            ),
            (batch_size, dims.conv_dim, dims.conv_kernel_size),
            &dev,
        )?;
        let recurrent_state = Tensor::zeros(
            (
                batch_size,
                dims.num_v_heads,
                dims.head_k_dim,
                dims.head_v_dim,
            ),
            DType::F32,
            &dev,
        )?;
        let mut fast_cache = GdnLayerCache {
            conv_state: initial_state.clone(),
            recurrent_state: recurrent_state.clone(),
        };
        let mut reference_cache = GdnLayerCache {
            conv_state: initial_state,
            recurrent_state,
        };

        let fast = causal_conv1d_update_cpu(&x, &weight, &dims, &mut fast_cache)?;
        let reference = causal_conv1d_update_reference(&x, &weight, &dims, &mut reference_cache)?;

        assert_close(&fast, &reference)?;
        assert_close(&fast_cache.conv_state, &reference_cache.conv_state)
    }

    #[test]
    fn decode_recurrence_cpu_matches_tensor_path() -> CandleResult<()> {
        run_decode_case(dims(2, 4, 5, 3), 2)?;
        run_decode_case(dims(3, 3, 4, 2), 1)
    }
}
