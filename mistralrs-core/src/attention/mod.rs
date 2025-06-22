#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::pipeline::text_models_inputs_processor::FlashParams;

use candle_core::{Device, Result, Tensor};

mod backends;

#[allow(unused)]
pub(crate) use backends::{flash_attn, maybe_synchronize, naive_sdpa};

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
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
            return naive_sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask,
                sdpa_params,
            );
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
