#![allow(clippy::cast_precision_loss)]

#[cfg(feature = "metal")]
use std::sync::atomic::AtomicUsize;

use crate::{pipeline::text_models_inputs_processor::FlashParams, MemoryUsage};

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::MatMul;

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
fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    maybe_synchronize(q.device())?;

    // Use faster softmax if mask is rank 2 or it's rank 3
    if mask.is_some_and(|mask| mask.rank() == 2 || mask.rank() == 3) && supports_attn_softmax()? {
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
            *mistralrs_quant::cublaslt::CUBLASLT_HANDLE.lock().unwrap(),
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
