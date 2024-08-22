#![allow(clippy::cast_precision_loss)]

use crate::{
    cublaslt::CUBLASLT_HANDLE,
    layers::{get_use_matmul_via_f16, MatMul},
    pipeline::text_models_inputs_processor::FlashParams,
};

use candle_core::{Device, Result, Tensor};

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

        //dbg!(&qshape);
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

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    _: &SdpaParams,
) -> Result<Tensor> {
    unimplemented!("Compile with '--features flash-attn'")
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

/// Computes softmax(QK^T*sqrt(d_k))V
fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    head_dim: usize,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let mut att = MatMul.matmul_affine_div(
        &q.contiguous()?,
        &k.t()?.contiguous()?,
        (head_dim as f64).sqrt(),
    )?;
    if let Some(softcap) = sdpa_params.softcap {
        att = (att / softcap as f64)?;
        att = att.tanh()?;
        att = (att * softcap as f64)?;
    }

    let att = match mask {
        Some(m) => att.broadcast_add(m)?,
        None => att,
    };
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    MatMul.matmul(&att, &v.contiguous()?)
}

pub struct SdpaParams {
    pub n_kv_groups: usize,
    pub use_flash_attn: bool,
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
    /// 1) If `use_flash_attn == true`, use a flash attention V2 kernel
    /// 2) If using CUDA and the cuBLASLt kernel is initialized, then it will use an optimized version.
    /// 3) Otherwise, use the "naive" SDPA implementation.
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
        if sdpa_params.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            return flash_attn(&q, &k, &v, flash_params, sdpa_params)?.transpose(1, 2);
        }

        let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?.contiguous()?;
        let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?.contiguous()?;
        if let (Device::Cuda(_), Some(cublaslt)) = (q.device(), *CUBLASLT_HANDLE.lock().unwrap()) {
            if !get_use_matmul_via_f16() {
                #[cfg(feature = "cuda")]
                {
                    // cuBLASLt batch matmul implementation requires inputs to be dims3
                    let k = k.flatten(0, 1)?;
                    let q = q.flatten(0, 1)?;
                    let v = v.flatten(0, 1)?;
                    let attention_bias = mask.map(|mask| mask.flatten(0, 1)).transpose()?;

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
                    let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

                    let context_layer = cublaslt.batch_matmul(
                        &v.t()?.contiguous()?,
                        &attention_probs,
                        // We save one allocation
                        Some(&q),
                        None,
                        None,
                        None,
                        None,
                    )?;

                    // Reshape to dims4
                    context_layer.reshape((b_sz, n_attn_heads, seq_len, head_dim))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    candle_core::bail!("`cuda` feature is not enabled")
                }
            } else {
                // Use the f16 kernels here if quantized (ISQ or GGML), and a large enough prompt
                naive_sdpa(q, &k, &v, mask, head_dim, sdpa_params)
            }
        } else {
            naive_sdpa(q, &k, &v, mask, head_dim, sdpa_params)
        }
    }
}
