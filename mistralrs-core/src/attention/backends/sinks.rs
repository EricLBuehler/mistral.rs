use candle_core::{DType, DeviceLocation, Result, Tensor};
use mistralrs_quant::MatMul;

use crate::{
    attention::{repeat_kv, SdpaParams},
    pipeline::text_models_inputs_processor::FlashParams,
};

/// Fused attention with per-head sinks.
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

    // the cpu kv cache may store f16/bf16 below the activation dtype; this path is tensor-op
    // based, so bring k/v back up rather than mixing dtypes
    let (k_conv, v_conv);
    let (k, v) = if k.dtype() != q.dtype() {
        k_conv = k.to_dtype(q.dtype())?;
        v_conv = v.to_dtype(q.dtype())?;
        (&k_conv, &v_conv)
    } else {
        (k, v)
    };

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    if let Some(flash_params) = flash_params.filter(|params| params.packed && q.device().is_cuda())
    {
        return sinks_attn_varlen(q, k, v, sinks, flash_params, sdpa_params, window_size);
    }

    let is_varlen = b_sz > 1
        && kv_layout_is_packed(k.dims())
        && kv_layout_is_packed(v.dims())
        && flash_params.is_some_and(|fp| {
            let location = q.device().location();
            fp.cumulative_seqlens_q.contains_key(&location)
                && fp
                    .k_meta(sdpa_params.sliding_window)
                    .cumulative_seqlens
                    .contains_key(&location)
        });

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

    sinks_attn_regular(q, k, v, sinks, mask, sdpa_params, window_size)
}

pub(crate) fn sinks_backend_is_available(query: &Tensor, head_size: usize) -> bool {
    sinks_backend_supports(query.dtype(), query.device().location(), head_size)
}

pub(crate) fn sinks_backend_supports(
    dtype: DType,
    location: DeviceLocation,
    head_size: usize,
) -> bool {
    if !sinks_kernel_supports(dtype, head_size) {
        return false;
    }

    match location {
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        DeviceLocation::Cuda { .. } => true,
        #[cfg(feature = "metal")]
        DeviceLocation::Metal { .. } => true,
        _ => false,
    }
}

fn sinks_kernel_supports(dtype: DType, head_size: usize) -> bool {
    matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
        && matches!(head_size, 64 | 80 | 96 | 112 | 128 | 192 | 256)
}

fn kv_layout_is_packed(dims: &[usize]) -> bool {
    matches!(dims, [_, _, _] | [1, _, _, _])
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
    if mask.is_some() {
        return sinks_attn_cpu(q, k, v, sinks, mask, sdpa_params);
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
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

/// Sink attention with packed K/V and either padded or packed queries.
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

    // Handle K/V shape: 4D [1, kv_H, total, D] -> 3D [total, kv_H, D], or 3D as-is
    let (k_packed, v_packed) = if k.dims().len() == 4 {
        (
            k.squeeze(0)?.transpose(0, 1)?,
            v.squeeze(0)?.transpose(0, 1)?,
        )
    } else {
        (k.clone(), v.clone())
    };

    let location = device.location();
    let cu_seqlens_q = flash_params
        .cumulative_seqlens_q
        .get(&location)
        .ok_or_else(|| {
            candle_core::Error::msg("sinks varlen attention is missing query metadata")
        })?;
    let cu_seqlens_k = flash_params
        .k_meta(sdpa_params.sliding_window)
        .cumulative_seqlens
        .get(&location)
        .ok_or_else(|| candle_core::Error::msg("sinks varlen attention is missing key metadata"))?;

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    if device.is_cuda() && flash_params.packed && q.dim(0)? == 1 {
        let query = q.squeeze(0)?.transpose(0, 1)?;
        return mistralrs_paged_attn::flash_attn_sinks_varlen_packed(
            &query,
            &k_packed,
            &v_packed,
            Some(sinks),
            cu_seqlens_q,
            cu_seqlens_k,
            flash_params.max_q as usize,
            sdpa_params.softmax_scale,
            window_size,
        )?
        .transpose(0, 1)?
        .unsqueeze(0);
    }

    if cu_seqlens_q.dim(0)? != q.dim(0)? + 1 || cu_seqlens_k.dim(0)? != q.dim(0)? + 1 {
        candle_core::bail!("sinks varlen metadata does not match the query batch");
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    if device.is_cuda() {
        return mistralrs_paged_attn::flash_attn_sinks_varlen(
            q,
            &k_packed,
            &v_packed,
            Some(sinks),
            cu_seqlens_q,
            cu_seqlens_k,
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
            cu_seqlens_q,
            cu_seqlens_k,
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    // CPU fallback: per-sequence loop (to_vec1 is fine on CPU path)
    let cu_q_vec: Vec<u32> = cu_seqlens_q.to_vec1()?;
    let cu_k_vec: Vec<u32> = cu_seqlens_k.to_vec1()?;
    sinks_attn_cpu_varlen(
        q,
        &k_packed,
        &v_packed,
        sinks,
        sdpa_params,
        &cu_q_vec,
        &cu_k_vec,
    )
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
    let q = q.contiguous()?;
    let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?.contiguous()?;
    let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?.contiguous()?;

    let att = MatMul.matmul_affine_mul(&q, &k.t()?, sdpa_params.softmax_scale.into())?;
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
    cu_q: &[u32],
    cu_k: &[u32],
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

#[cfg(test)]
mod tests {
    use super::{
        kv_layout_is_packed, sinks_attn_regular, sinks_backend_supports, sinks_kernel_supports,
    };
    use crate::attention::SdpaParams;
    use candle_core::{DType, Device, DeviceLocation, Result, Tensor};

    fn check_masked_token_major_qkv(device: &Device, dtype: DType) -> Result<()> {
        const BATCH: usize = 2;
        const QUERY_LEN: usize = 3;
        const KEY_LEN: usize = 5;
        const HEADS: usize = 4;
        const HEAD_DIM: usize = 8;
        const TOLERANCE: f32 = 1e-6;

        let tensor = |len: usize, heads: usize| {
            Tensor::arange(0f32, (BATCH * len * heads * HEAD_DIM) as f32, device)?
                .affine(0.003, -0.4)?
                .to_dtype(dtype)?
                .reshape((BATCH, len, heads, HEAD_DIM))?
                .transpose(1, 2)
        };
        let mask = (0..QUERY_LEN)
            .flat_map(|query| {
                (0..KEY_LEN).map(move |key| {
                    if key > KEY_LEN - QUERY_LEN + query {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(mask, (QUERY_LEN, KEY_LEN), device)?.to_dtype(dtype)?;
        let sinks = Tensor::arange(0f32, HEADS as f32, device)?
            .affine(0.1, -0.2)?
            .to_dtype(dtype)?;
        let q = tensor(QUERY_LEN, HEADS)?;
        assert!(!q.is_contiguous());
        for groups in [1, 2] {
            let k = tensor(KEY_LEN, HEADS / groups)?;
            let v = tensor(KEY_LEN, HEADS / groups)?;
            assert!(!k.is_contiguous() && !v.is_contiguous());
            let params = SdpaParams {
                n_kv_groups: groups,
                softcap: None,
                softmax_scale: 1. / (HEAD_DIM as f32).sqrt(),
                sliding_window: None,
                sinks: Some(sinks.clone()),
            };
            let expected = sinks_attn_regular(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                &sinks,
                Some(&mask),
                &params,
                0,
            )?;
            let actual = sinks_attn_regular(&q, &k, &v, &sinks, Some(&mask), &params, 0)?;
            let error = (actual - expected)?
                .to_dtype(DType::F32)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(error < TOLERANCE, "max masked sinks difference: {error}");
        }
        Ok(())
    }

    #[test]
    fn masked_sinks_accepts_token_major_qkv() -> Result<()> {
        check_masked_token_major_qkv(&Device::Cpu, DType::F32)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_masked_sinks_accepts_token_major_qkv() -> Result<()> {
        check_masked_token_major_qkv(&Device::new_cuda(0)?, DType::BF16)
    }

    #[test]
    fn packed_kv_layout_rejects_regular_batched_tensors() {
        assert!(kv_layout_is_packed(&[17, 4, 64]));
        assert!(kv_layout_is_packed(&[1, 4, 17, 64]));
        assert!(!kv_layout_is_packed(&[2, 4, 17, 64]));
        assert!(!kv_layout_is_packed(&[4, 17]));
    }

    #[test]
    fn fused_sinks_accepts_only_compiled_kernel_shapes() {
        assert!(sinks_kernel_supports(DType::F32, 64));
        assert!(sinks_kernel_supports(DType::BF16, 256));
        assert!(!sinks_kernel_supports(DType::F16, 320));
        assert!(!sinks_kernel_supports(DType::U32, 128));
    }

    #[test]
    fn fused_sinks_backend_capability_includes_device_and_build() {
        assert!(!sinks_backend_supports(DType::F16, DeviceLocation::Cpu, 64));
        assert_eq!(
            sinks_backend_supports(DType::BF16, DeviceLocation::Cuda { gpu_id: 0 }, 128),
            cfg!(all(feature = "cuda", target_family = "unix"))
        );
        assert_eq!(
            sinks_backend_supports(DType::F32, DeviceLocation::Metal { gpu_id: 0 }, 256),
            cfg!(feature = "metal")
        );
        assert!(!sinks_backend_supports(
            DType::F16,
            DeviceLocation::Cuda { gpu_id: 0 },
            72
        ));
    }
}

#[cfg(all(test, feature = "cuda", target_family = "unix"))]
mod packed_cuda_tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Result, Tensor};

    use crate::{
        attention::{AttentionMask, Sdpa, SdpaParams},
        pipeline::text_models_inputs_processor::{FlashKMeta, FlashParams},
    };

    const QUERY_HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const QUERY_LENS: [usize; 2] = [2, 5];
    const SLIDING_WINDOW: usize = 3;
    const BF16_TOLERANCE: f32 = 0.01;
    const F16_TOLERANCE: f32 = 0.001;
    const F32_TOLERANCE: f32 = 1e-5;

    #[test]
    fn cuda_cold_packed_sinks_preserves_sequence_boundaries() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let location = device.location();
        let total_q = QUERY_LENS.iter().sum::<usize>();
        let cu_q = Tensor::from_vec(
            vec![0u32, QUERY_LENS[0] as u32, total_q as u32],
            QUERY_LENS.len() + 1,
            &device,
        )?;
        let mut flash = FlashParams::empty(true);
        flash.packed = true;
        flash.max_q = *QUERY_LENS.iter().max().unwrap() as u32;
        flash.cumulative_seqlens_q = HashMap::from([(location, cu_q.clone())]);
        flash.logical_k = FlashKMeta {
            max: flash.max_q,
            cumulative_seqlens: HashMap::from([(location, cu_q)]),
        };
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let key = Tensor::zeros((1, KV_HEADS, total_q, HEAD_DIM), dtype, &device)?;
            let values = QUERY_LENS
                .iter()
                .enumerate()
                .flat_map(|(sequence, &len)| {
                    std::iter::repeat_n((sequence + 1) as f32, len * KV_HEADS * HEAD_DIM)
                })
                .collect::<Vec<_>>();
            let value = Tensor::from_vec(values, (total_q, KV_HEADS, HEAD_DIM), &device)?
                .to_dtype(dtype)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            let sinks =
                Tensor::full(std::f32::consts::LN_2, QUERY_HEADS, &device)?.to_dtype(dtype)?;
            let sink_mass = sinks.to_dtype(DType::F32)?.to_vec1::<f32>()?[0].exp();
            for (batch, query_len) in [(1, total_q), (QUERY_LENS.len(), flash.max_q as usize)] {
                let query =
                    Tensor::zeros((batch, QUERY_HEADS, query_len, HEAD_DIM), dtype, &device)?;
                for sliding_window in [None, Some(SLIDING_WINDOW)] {
                    let params = SdpaParams {
                        n_kv_groups: QUERY_HEADS / KV_HEADS,
                        softmax_scale: 1. / (HEAD_DIM as f32).sqrt(),
                        softcap: None,
                        sliding_window,
                        sinks: Some(sinks.clone()),
                    };
                    let output = Sdpa.run_attention(
                        &query,
                        &key,
                        &value,
                        &AttentionMask::CausalFlash,
                        Some(&flash),
                        &params,
                    )?;
                    assert_eq!(output.dims(), query.dims());
                    let output = output
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let tolerance = match dtype {
                        DType::BF16 => BF16_TOLERANCE,
                        DType::F16 => F16_TOLERANCE,
                        _ => F32_TOLERANCE,
                    };
                    for head in 0..QUERY_HEADS {
                        let mut offset = 0;
                        for (sequence, &len) in QUERY_LENS.iter().enumerate() {
                            for token in 0..if batch == 1 { len } else { query_len } {
                                let visible = sliding_window
                                    .map_or(token + 1, |window| (token + 1).min(window));
                                let expected = if token < len {
                                    (sequence + 1) as f32 * visible as f32
                                        / (visible as f32 + sink_mass)
                                } else {
                                    0.
                                };
                                let row = if batch == 1 {
                                    head * total_q + offset + token
                                } else {
                                    (sequence * QUERY_HEADS + head) * query_len + token
                                };
                                for dim in 0..HEAD_DIM {
                                    let actual = output[row * HEAD_DIM + dim];
                                    assert!(
                                    (actual - expected).abs() < tolerance,
                                    "{dtype:?}, batch {batch}, window {sliding_window:?}, sequence {sequence}, token {token}: {actual} != {expected}"
                                );
                                }
                            }
                            offset += len;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
