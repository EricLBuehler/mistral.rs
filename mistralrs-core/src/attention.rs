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
    softmax_scale: f32,
    softcap: Option<f32>,
    causal: bool,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
) -> Result<Tensor> {
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
        //dbg!(&qshape);
        candle_flash_attn::flash_attn_varlen_softcap(
            &q,
            &k,
            &v,
            cumulative_seqlens_q,
            cumulative_seqlens_k,
            *max_q as usize,
            *max_k as usize,
            softmax_scale,
            softcap,
            causal,
        )?
        .reshape(qshape)
    } else {
        candle_flash_attn::flash_attn_softcap(q, k, v, softmax_scale, softcap, causal)
    }
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: f32,
    _: Option<f32>,
    _: bool,
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
    head_dim: usize,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let att = MatMul.matmul_affine_div(
        &q.contiguous()?,
        &k.t()?.contiguous()?,
        (head_dim as f64).sqrt(),
    )?;

    let att = match mask {
        Some(m) => att.broadcast_add(m)?,
        None => att,
    };
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    MatMul.matmul(&att, &v.contiguous()?)
}

pub struct ScaledDotProductAttention;

impl ScaledDotProductAttention {
    /// Computes softmax(QK^T*sqrt(d_k))V
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
        n_attn_heads: usize,
        head_dim: usize,
        mask: Option<&Tensor>,
        use_flash_attn: bool,
        b_sz: usize,
        seq_len: usize,
        softcap: Option<f32>,
        n_kv_groups: usize,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        if use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (head_dim as f32).sqrt();
            return flash_attn(
                &q,
                &k,
                &v,
                softmax_scale,
                softcap,
                seq_len > 1,
                flash_params,
            )?
            .transpose(1, 2);
        }

        let k = repeat_kv(k.clone(), n_kv_groups)?.contiguous()?;
        let v = repeat_kv(v.clone(), n_kv_groups)?.contiguous()?;
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
                    let attention_scores = cublaslt.batch_matmul(
                        &k,
                        &q,
                        attention_bias.as_ref(),
                        Some((1.0 / (head_dim as f64).sqrt()) as f32),
                        beta,
                        None,
                        None,
                    )?;
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
                naive_sdpa(q, &k, &v, head_dim, mask)
            }
        } else {
            naive_sdpa(q, &k, &v, head_dim, mask)
        }
    }
}
