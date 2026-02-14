use candle_core::{DType, Result, Tensor};
use mistralrs_quant::MatMul;

use crate::{
    attention::{repeat_kv, SdpaParams},
    pipeline::text_models_inputs_processor::FlashParams,
};

/// Fused attention with per-head sinks.
///
/// Dispatches to:
///   CUDA  -> flash_attn_sinks / flash_attn_sinks_varlen
///   Metal -> flash_attn_sinks_metal / flash_attn_sinks_varlen_metal
///   CPU   -> unfused matmul + softmax_with_sinks
///
/// Varlen is used when flash_params contains cu_seqlens_k for this device AND
/// q has batch > 1 AND k is packed (dim 0 == 1).
#[allow(unused_variables, clippy::too_many_arguments)]
pub(crate) fn sinks_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    mask: Option<&Tensor>,
    flash_params: Option<&FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b_sz, _n_heads, _q_len, _head_dim) = q.dims4()?;
    let window_size = sdpa_params.sliding_window.unwrap_or(0);

    // Detect varlen: flash_params has cu_seqlens_k AND k is packed (dim 0 == 1) AND batch > 1
    let is_varlen = b_sz > 1
        && k.dim(0)? == 1
        && flash_params
            .is_some_and(|fp| fp.cumulative_seqlens_k.contains_key(&q.device().location()));

    if is_varlen {
        return sinks_attn_varlen(
            q,
            k,
            v,
            sinks,
            flash_params.unwrap(),
            sdpa_params,
            window_size,
        );
    }

    // Non-varlen path
    sinks_attn_regular(q, k, v, sinks, mask, sdpa_params, window_size)
}

/// Non-varlen sinks attention: Q [B, H, q_len, D], K/V [B, kv_H, kv_len, D]
#[allow(unused_variables)]
fn sinks_attn_regular(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
    window_size: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        return mistralrs_paged_attn::flash_attn_sinks(
            q,
            k,
            v,
            Some(sinks),
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        return mistralrs_quant::flash_attn_sinks_metal(
            q,
            k,
            v,
            Some(sinks),
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    // CPU fallback: unfused matmul + softmax_with_sinks
    sinks_attn_cpu(q, k, v, sinks, mask, sdpa_params)
}

/// Varlen sinks attention: Q [B, H, max_q, D], K/V packed [1, kv_H, total_kv, D]
#[allow(unused_variables)]
fn sinks_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    flash_params: &FlashParams,
    sdpa_params: &SdpaParams,
    window_size: usize,
) -> Result<Tensor> {
    let device = q.device();

    // K/V are [1, kv_H, total_kv, D] -> [total_kv, kv_H, D]
    let k_packed = k.squeeze(0)?.transpose(0, 1)?;
    let v_packed = v.squeeze(0)?.transpose(0, 1)?;

    // Get cu_seqlens from flash_params
    let cu_seqlens_q = &flash_params.cumulative_seqlens_q[&device.location()];
    let cu_seqlens_k = &flash_params.cumulative_seqlens_k[&device.location()];

    // Compute q_lens from cu_seqlens_q: diff of consecutive entries
    let cu_q_vec: Vec<i64> = cu_seqlens_q.to_vec1()?;
    let q_lens_vec: Vec<i32> = cu_q_vec.windows(2).map(|w| (w[1] - w[0]) as i32).collect();
    let q_lens = Tensor::new(q_lens_vec, device)?;

    // cu_seqlens_k needs to be i32 for our kernel
    let cu_k = cu_seqlens_k.to_dtype(DType::I32)?;

    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        return mistralrs_paged_attn::flash_attn_sinks_varlen(
            q,
            &k_packed,
            &v_packed,
            Some(sinks),
            &q_lens,
            &cu_k,
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    #[cfg(feature = "metal")]
    if device.is_metal() {
        return mistralrs_quant::flash_attn_sinks_varlen_metal(
            q,
            &k_packed,
            &v_packed,
            Some(sinks),
            &q_lens,
            &cu_k,
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    // CPU fallback: per-sequence loop
    sinks_attn_cpu_varlen(q, &k_packed, &v_packed, sinks, sdpa_params, &cu_q_vec, &{
        let cu_k_vec: Vec<i64> = cu_seqlens_k.to_vec1()?;
        cu_k_vec
    })
}

/// CPU fallback: unfused matmul + softmax_with_sinks
fn sinks_attn_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
    let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;

    let att = MatMul.matmul_affine_mul(q, &k.t()?, sdpa_params.softmax_scale.into())?;
    let att = mistralrs_quant::softmax_with_sinks(&att, sinks, mask)?;
    MatMul.matmul(&att, &v)
}

/// CPU fallback for varlen: per-sequence unfused loop
fn sinks_attn_cpu_varlen(
    q: &Tensor,
    k_packed: &Tensor,
    v_packed: &Tensor,
    sinks: &Tensor,
    sdpa_params: &SdpaParams,
    cu_q: &[i64],
    cu_k: &[i64],
) -> Result<Tensor> {
    let (b_sz, num_heads, max_q, head_dim) = q.dims4()?;
    let device = q.device();
    let mut outputs = Vec::with_capacity(b_sz);

    for i in 0..b_sz {
        let q_len = (cu_q[i + 1] - cu_q[i]) as usize;
        let kv_start = cu_k[i] as usize;
        let kv_len = (cu_k[i + 1] - cu_k[i]) as usize;

        // Extract this sequence's Q [1, H, q_len, D]
        let qi = q.narrow(0, i, 1)?;
        let qi = qi.narrow(2, 0, q_len)?;

        // Extract this sequence's K/V from packed [total_kv, kv_H, D]
        let ki = k_packed
            .narrow(0, kv_start, kv_len)?
            .transpose(0, 1)?
            .unsqueeze(0)?;
        let vi = v_packed
            .narrow(0, kv_start, kv_len)?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        let oi = sinks_attn_cpu(&qi, &ki, &vi, sinks, None, sdpa_params)?;

        // Pad back to max_q
        if q_len < max_q {
            let pad = Tensor::zeros((1, num_heads, max_q - q_len, head_dim), qi.dtype(), device)?;
            outputs.push(Tensor::cat(&[&oi, &pad], 2)?);
        } else {
            outputs.push(oi);
        }
    }

    Tensor::cat(&outputs, 0)
}
