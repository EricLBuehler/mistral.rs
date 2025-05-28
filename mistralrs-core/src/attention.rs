#![allow(clippy::cast_precision_loss)]

#[cfg(feature = "metal")]
use std::sync::atomic::AtomicUsize;

use crate::{pipeline::text_models_inputs_processor::FlashParams, MemoryUsage};

use candle_core::{Context, DType, Device, Result, Storage, Tensor, WithDType};
use mistralrs_quant::MatMul;

use rayon::prelude::*;
use std::{f32, iter::Sum};

/// Dot product between two f16 slices, accumulated in f32
#[inline]
fn vec_dot<T: WithDType + Sum>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum::<T>()
}

fn run_flash_attn_cpu<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real,
{
    // Inline CPU slice extraction for q, k, v, and optional mask
    let (q_guard, q_layout) = q.storage_and_layout();
    let q_data: &[T] = if let Storage::Cpu(cpu) = &*q_guard {
        let data = cpu.as_slice::<T>().context("Expected CPU storage for q")?;
        &data[q_layout.start_offset()..]
    } else {
        return Err(candle_core::Error::Msg("Expected CPU storage for q".into()));
    };
    let (k_guard, k_layout) = k.storage_and_layout();
    let k_data: &[T] = if let Storage::Cpu(cpu) = &*k_guard {
        let data = cpu.as_slice::<T>().context("Expected CPU storage for k")?;
        &data[k_layout.start_offset()..]
    } else {
        return Err(candle_core::Error::Msg("Expected CPU storage for k".into()));
    };
    let (v_guard, v_layout) = v.storage_and_layout();
    let v_data: &[T] = if let Storage::Cpu(cpu) = &*v_guard {
        let data = cpu.as_slice::<T>().context("Expected CPU storage for v")?;
        &data[v_layout.start_offset()..]
    } else {
        return Err(candle_core::Error::Msg("Expected CPU storage for v".into()));
    };
    let mask_guard = match mask {
        Some(mask) => Some(mask.storage_and_layout().0),
        None => None,
    };
    let mask_data: Option<&[T]> = if let Some(mask_guard) = &mask_guard {
        let mask = mask.as_ref().unwrap();

        if let Storage::Cpu(cpu) = &**mask_guard {
            let data = cpu
                .as_slice::<T>()
                .context("Expected CPU storage for mask")?;
            Some(&data[mask.layout().start_offset()..])
        } else {
            return Err(candle_core::Error::Msg(
                "Expected CPU storage for mask".into(),
            ));
        }
    } else {
        None
    };
    // q_guard, k_guard, v_guard, and m_guard (if any) are kept in scope to hold storage alive

    let q_stride = q.stride();
    let k_stride = k.stride();
    let v_stride = v.stride();

    flash_attn_cpu(
        q_data,
        k_data,
        v_data,
        mask_data,
        q.shape().dims(),
        k.shape().dims(),
        v.shape().dims(),
        &q_stride,
        &k_stride,
        &v_stride,
        sdpa_params.softmax_scale,
        0.0,
        sdpa_params.softcap.unwrap_or(0.0),
    )
}

/// Main forward flash-attention CPU routine.
/// Shapes follow Candle convention: (B, S, H, D)
pub fn flash_attn_cpu<T: WithDType + Sum + num_traits::real::Real>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    mask_vec: Option<&[T]>,
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    // Shapes: (B, S, H, D)
    let qshape = qshape;
    let kshape = kshape;
    let vshape = vshape;

    let (b, q_len, h, d) = (
        qshape[0] as usize,
        qshape[1] as usize,
        qshape[2] as usize,
        qshape[3] as usize,
    );
    let kv_len = kshape[1] as usize;
    // --- Head broadcasting factors ----------------------------------------------------
    // Allows K and V to have fewer heads than Q (grouped‑KV); the ratio is an
    // integer factor.  rk2 = #Q‑heads / #K‑heads,  rv2 = #Q‑heads / #V‑heads.
    let k_h = kshape[2] as usize;
    let v_h = vshape[2] as usize;
    let rk2 = h / k_h; // must divide exactly; panic otherwise
    let rv2 = h / v_h;
    let dv = d; // value dim = key dim in this kernel

    // Precompute constants for positional bias
    let n_head_log2 = 1u32 << ((h as f32).log2().floor() as u32);
    let m0 = (-(max_bias) / n_head_log2 as f32).exp2();
    let m1 = (-(max_bias / 2.0) / n_head_log2 as f32).exp2();

    let mut out = vec![0f32; b * q_len * h * dv];

    // ------------------------------------------------------------------
    // Rayon‑parallel version: each (b_i, h_i, q_pos) row is independent.
    // ------------------------------------------------------------------

    let _rows = b * h * q_len; // total independent work items

    // SAFETY: `par_chunks_mut` hands out non‑overlapping &mut [f32] slices,
    // so no two threads can write the same output area.
    out.par_chunks_mut(dv)
        .enumerate()
        .for_each(|(row_idx, out_chunk)| {
            // Decode flat index back to (batch, head, q_pos)
            let rows_per_batch = h * q_len;
            let b_i = row_idx / rows_per_batch;
            let rem = row_idx % rows_per_batch;
            let h_i = rem / q_len;
            let q_pos = rem % q_len;

            let slope = if max_bias > 0.0 {
                if (h_i as u32) < n_head_log2 {
                    m0.powi((h_i + 1) as i32)
                } else {
                    m1.powi((2 * (h_i as i32 - n_head_log2 as i32) + 1) as i32)
                }
            } else {
                1.0
            };

            // For grouped‑KV we collapse multiple query heads into the same K/V head.
            let k_head = h_i / rk2;
            let v_head = h_i / rv2;

            // Buffers local to this row
            let mut vkq = vec![0f32; dv];
            let mut s = 0.0f32;
            let mut m = f32::NEG_INFINITY;

            // ------------------- gather Q (strided) --------------------
            let q_base = b_i * qstride[0] + q_pos * qstride[1] + h_i * qstride[2];
            let mut q_row: Vec<T> = Vec::with_capacity(d);
            for di in 0..d {
                q_row.push(q_data[q_base + di * qstride[3]]);
            }

            // ---------------- iterate over keys/values -----------------
            for kv_pos in 0..kv_len {
                // Mask (optional)
                let mv = if let Some(mv_vec) = mask_vec {
                    let mval = mv_vec[((b_i * q_len + q_pos) * kv_len) + kv_pos];
                    slope * mval.to_f64() as f32
                } else {
                    0.0
                };
                if mv == f32::NEG_INFINITY {
                    continue;
                }

                // K row (strided)
                let k_base = b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
                let mut k_row: Vec<T> = Vec::with_capacity(d);
                for di in 0..d {
                    k_row.push(k_data[k_base + di * kstride[3]]);
                }

                // dot(Q, K)
                let mut s_val = vec_dot::<T>(&q_row, &k_row);
                let mut scale_applied = scale;
                if logit_softcap != 0.0 {
                    scale_applied /= logit_softcap;
                }
                s_val *= T::from_f64(scale_applied as f64);
                if logit_softcap != 0.0 {
                    s_val = T::from_f64(logit_softcap as f64 * s_val.to_f64().tanh());
                }
                s_val += T::from_f64(mv as f64);

                // online softmax
                let m_old = m;
                let mut ms = 1.0f32;
                let mut vs = 1.0f32;
                if s_val.to_f64() as f32 > m {
                    m = s_val.to_f64() as f32;
                    ms = (m_old - m).exp();
                    for v in vkq.iter_mut() {
                        *v *= ms;
                    }
                } else {
                    vs = (s_val.to_f64() as f32 - m).exp();
                }

                // V row (strided)
                let v_base = b_i * vstride[0] + kv_pos * vstride[1] + v_head * vstride[2];
                for d_i in 0..dv {
                    vkq[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * vs;
                }

                s = s * ms + vs;
            }

            // ------------------- normalise & write out ------------------
            let inv_s = 1.0 / s;
            for v in vkq.iter_mut() {
                *v *= inv_s;
            }
            out_chunk.copy_from_slice(&vkq);
        });

    // Build output tensor with shape (B, H, S, D) to match standard (permute 0,2,1,3)
    let out_shape = (b, h, q_len, dv);
    Ok(Tensor::from_vec(out, out_shape, &Device::Cpu)?)
}

#[cfg(feature = "metal")]
/// Initial, sentinel value is usize::MAX
static METAL_VERSION_CACHE: AtomicUsize = AtomicUsize::new(usize::MAX);

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (_b_sz, _n_attn_heads, seq_len, _head_dim) = q.dims4()?;
    let causal = seq_len > 1;

    use crate::pipeline::text_models_inputs_processor::FlashParams;

    if let Some(FlashParams {
        max_q,
        max_k,
        cumulative_seqlens_q,
        cumulative_seqlens_k,
    }) = flash_params
    {
        let qshape = q.shape();
        let q = q.flatten_to(1)?;
        let k = k.flatten_to(1)?;
        let v = v.flatten_to(1)?;

        let window_size_left = sdpa_params.sliding_window;
        let window_size_right = if causal { Some(0) } else { None };

        let cumulative_seqlens_q = &cumulative_seqlens_q[&q.device().location()];
        let cumulative_seqlens_k = &cumulative_seqlens_k[&q.device().location()];

        candle_flash_attn::flash_attn_varlen_windowed_softcap(
            &q,
            &k,
            &v,
            cumulative_seqlens_q,
            cumulative_seqlens_k,
            *max_q as usize,
            *max_k as usize,
            sdpa_params.softmax_scale,
            sdpa_params.softcap,
            window_size_left,
            window_size_right,
        )?
        .reshape(qshape)
    } else {
        candle_flash_attn::flash_attn_softcap(
            q,
            k,
            v,
            sdpa_params.softmax_scale,
            sdpa_params.softcap,
            causal,
        )
    }
}

#[cfg(feature = "flash-attn-v3")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (_b_sz, _n_attn_heads, seq_len, _head_dim) = q.dims4()?;
    let causal = seq_len > 1;

    use crate::pipeline::text_models_inputs_processor::FlashParams;

    if let Some(FlashParams {
        max_q,
        max_k,
        cumulative_seqlens_q,
        cumulative_seqlens_k,
    }) = flash_params
    {
        let qshape = q.shape();
        let q = q.flatten_to(1)?;
        let k = k.flatten_to(1)?;
        let v = v.flatten_to(1)?;

        let window_size_left = sdpa_params.sliding_window;
        let window_size_right = if causal { Some(0) } else { None };

        let cumulative_seqlens_q = &cumulative_seqlens_q[&q.device().location()];
        let cumulative_seqlens_k = &cumulative_seqlens_k[&q.device().location()];

        candle_flash_attn_v3::flash_attn_varlen_windowed(
            &q,
            &k,
            &v,
            cumulative_seqlens_q,
            cumulative_seqlens_k,
            *max_q as usize,
            *max_k as usize,
            sdpa_params.softmax_scale,
            window_size_left,
            window_size_right,
            true,
        )?
        .reshape(qshape)
    } else {
        candle_flash_attn_v3::flash_attn(q, k, v, sdpa_params.softmax_scale, causal, true)
    }
}

#[cfg(not(any(feature = "flash-attn", feature = "flash-attn-v3")))]
fn flash_attn(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    _: &SdpaParams,
) -> Result<Tensor> {
    unimplemented!("Compile with `--features flash-attn` or `--features flash-attn-v3`.")
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

fn supports_attn_softmax() -> Result<bool> {
    #[cfg(feature = "metal")]
    {
        use std::sync::atomic::Ordering;
        let cache = METAL_VERSION_CACHE.load(Ordering::Relaxed);

        let version = if cache != usize::MAX {
            cache
        } else {
            // echo "__METAL_VERSION__" | xcrun -sdk macosx metal -E -x metal -P -

            use std::process::{Command, Stdio};

            // Create the `echo` command and pipe its output into `xcrun`
            let mut echo = Command::new("echo")
                .arg("__METAL_VERSION__")
                .stdout(Stdio::piped())
                .spawn()
                .expect("Failed to start echo command");

            echo.wait()?;

            // Run the `xcrun` command, taking input from the `echo` command's output
            let output = Command::new("xcrun")
                .arg("-sdk")
                .arg("macosx")
                .arg("metal")
                .arg("-E")
                .arg("-x")
                .arg("metal")
                .arg("-P")
                .arg("-")
                .stdin(echo.stdout.unwrap())
                .output()
                .expect("Failed to run xcrun command");

            // Handle the output
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout)
                    .split('\n')
                    .nth(1)
                    .unwrap()
                    .trim()
                    .to_string()
                    .parse::<usize>()
                    .unwrap();
                METAL_VERSION_CACHE.store(version, Ordering::Relaxed);
                version
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                panic!("Error:\n{}", stderr);
            }
        };
        // Attn softmax is only supported for metal >= 310
        Ok(version >= 310)
    }

    #[cfg(not(feature = "metal"))]
    Ok(true)
}

/// Not *really* sure why this is necessary but it is.
fn maybe_synchronize(device: &Device) -> Result<()> {
    // If less that 4 GB available, synchronize
    if MemoryUsage.get_memory_available(device)? < 4 * 1024 * (1024 * 1024) {
        device.synchronize()?;
    }
    Ok(())
}

/// Computes softmax(QK^T*sqrt(d_k))V
pub(crate) fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    if q.device().is_cpu() {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        match q.dtype() {
            DType::F32 => run_flash_attn_cpu::<f32>(&q, &k, &v, mask, sdpa_params),
            DType::F16 => run_flash_attn_cpu::<half::f16>(&q, &k, &v, mask, sdpa_params),
            DType::BF16 => run_flash_attn_cpu::<half::bf16>(&q, &k, &v, mask, sdpa_params),
            _ => Err(candle_core::Error::Msg("Unsupported data type".into())),
        }
    } else {
        maybe_synchronize(q.device())?;

        // Use faster softmax if mask is rank 2 or it's rank 3
        if mask.is_some_and(|mask| mask.rank() == 2 || mask.rank() == 3) && supports_attn_softmax()?
        {
            let mask = match mask {
                Some(mask) if mask.rank() == 3 || mask.rank() == 2 => mask.clone(),
                _ => candle_core::bail!("unsupported mask {mask:?}"),
            };

            let mut att = MatMul.matmul(q, &k.t()?)?;

            candle_nn::ops::inplace_attn_softmax_last_dim(
                &mut att,
                &mask.contiguous()?,
                sdpa_params.softmax_scale / sdpa_params.softcap.unwrap_or(1.0),
            )?;

            if let Some(softcap) = sdpa_params.softcap {
                att = (att.tanh()? * softcap as f64)?;
            }

            MatMul.matmul(&att, v)
        } else if let Some(mask) = mask {
            let mut att = MatMul.matmul_affine_mul(q, &k.t()?, sdpa_params.softmax_scale.into())?;
            if let Some(softcap) = sdpa_params.softcap {
                att = (att / softcap as f64)?;
                att = att.tanh()?;
                att = (att * softcap as f64)?;
            }

            att = att.broadcast_add(mask)?;
            candle_nn::ops::inplace_softmax_last_dim(&mut att)?;

            MatMul.matmul(&att, v)
        } else {
            let mut att = MatMul.matmul_affine_mul(q, &k.t()?, sdpa_params.softmax_scale.into())?;
            if let Some(softcap) = sdpa_params.softcap {
                att = (att / softcap as f64)?;
                att = att.tanh()?;
                att = (att * softcap as f64)?;
            }

            candle_nn::ops::inplace_softmax_last_dim(&mut att)?;
            MatMul.matmul(&att, v)
        }
    }
}

pub struct SdpaParams {
    pub n_kv_groups: usize,
    pub softcap: Option<f32>,
    pub softmax_scale: f32,
    pub sliding_window: Option<usize>,
}

pub struct Sdpa;

impl Sdpa {
    /// Computes softmax(QK^T*sqrt(d_k))V
    ///
    /// Inputs:
    /// - q: (b_sz, n_attn_heads, q_len, head_dim)
    /// - k: (b_sz, n_kv_heads, q_len, head_dim)
    /// - v: (b_sz, n_kv_heads, q_len, head_dim)
    ///
    /// The attention implementation is dispatched as follows:
    /// 1) If using flash attn (CUDA), use a flash attention V2/V3 kernel
    /// 2) If decoding and using a Metal device, use a fused kkernel
    /// 2) Otherwise, use the "naive" SDPA implementation (with optimized mask+softmax+scale application)
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        flash_params: Option<&FlashParams>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
        let (_, _, _, k_head_dim) = k.dims4()?;
        let (_, _, _, v_head_dim) = v.dims4()?;
        if crate::using_flash_attn() && q.device().is_cuda() {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            return flash_attn(&q, &k, &v, flash_params, sdpa_params)?.transpose(1, 2);
        }

        // We can use Metal SDPA (vector/full) if the mask is the correct size and head dims match.
        // If the mask is provided, then softcapping isn't allowed - default back to naive SDPA
        // Softcapping is implemented for vector SDPA.
        let all_head_dims_match = head_dim == k_head_dim && k_head_dim == v_head_dim;
        let tgt_mask_shape = vec![b_sz, n_attn_heads, seq_len, k.dim(2)?];
        let can_use_mask = mask.is_none_or(|mask| {
            mask.layout().broadcast_as(tgt_mask_shape.clone()).is_ok()
                && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        });
        let valid_head_dims: &[usize] = if seq_len == 1 {
            &[32, 64, 72, 80, 96, 128, 256]
        } else {
            // Not sure why the full kernel doesn't like 256.
            // [32, 64, 72, 80, 96, 128, 256]
            &[32, 64, 72, 80, 96, 128]
        };
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && all_head_dims_match
            && valid_head_dims.contains(&head_dim)
            && can_use_mask
        {
            let mask = match mask {
                Some(mask) => Some(mask.broadcast_as(tgt_mask_shape)?),
                None => None,
            };
            return candle_nn::ops::sdpa(
                q,
                k,
                v,
                mask.as_ref(),
                false,
                sdpa_params.softmax_scale,
                sdpa_params.softcap.unwrap_or(1.0),
            );
        }

        let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;

        if mask.is_some_and(|x| x.rank() == 2) || mistralrs_quant::distributed::use_nccl() {
            return naive_sdpa(q, &k, &v, mask, sdpa_params);
        }

        // TODO: bench?
        #[allow(unused)]
        if let (Device::Cuda(_), Some(cublaslt)) = (
            q.device(),
            mistralrs_quant::cublaslt::CUBLASLT_CONTROLLER.get(),
        ) {
            #[cfg(feature = "cuda")]
            {
                maybe_synchronize(q.device())?;

                // cuBLASLt batch matmul implementation requires inputs to be dims3
                let k = k.flatten(0, 1)?;
                let q = q.flatten(0, 1)?;
                let v = v.flatten(0, 1)?;
                let attention_bias = match mask {
                    Some(mask) if mask.rank() == 3 && mask.dims()[0] == 1 => {
                        Some(mask.repeat((n_attn_heads, 1, 1))?)
                    }
                    Some(mask) if mask.rank() == 3 => Some(mask.clone()),
                    Some(mask) if mask.rank() == 4 => {
                        Some(mask.broadcast_as(tgt_mask_shape)?.flatten(0, 1)?)
                    }
                    Some(mask) => {
                        candle_core::bail!("cublaslt attn mask: rank must be 3 or 4")
                    }
                    None => None,
                };

                // If attention_bias is set, we fuse the add by giving it as the output matrix
                // and setting beta to 1.0
                let beta = match attention_bias.is_some() {
                    true => Some(1.0),
                    false => None,
                };

                // Batch matrix multiplication
                // Fuse softmax scale and attention_bias add
                let mut attention_scores = cublaslt.batch_matmul(
                    &k,
                    &q,
                    attention_bias.as_ref(),
                    Some(sdpa_params.softmax_scale / sdpa_params.softcap.unwrap_or(1.0)),
                    beta,
                    None,
                    None,
                )?;
                if let Some(softcap) = sdpa_params.softcap {
                    attention_scores = (attention_scores.tanh()? * softcap as f64)?;
                }
                candle_nn::ops::inplace_softmax_last_dim(&mut attention_scores)?;

                let context_layer = cublaslt.batch_matmul(
                    &v.t()?.contiguous()?,
                    &attention_scores,
                    // We save one allocation
                    Some(&q),
                    None,
                    None,
                    None,
                )?;

                // Reshape to dims4
                context_layer.reshape((b_sz, n_attn_heads, seq_len, v_head_dim))
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle_core::bail!("`cuda` feature is not enabled")
            }
        } else {
            naive_sdpa(q, &k, &v, mask, sdpa_params)
        }
    }
}
