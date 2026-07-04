#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::{attention::backends::cpu, pipeline::text_models_inputs_processor::FlashParams};

use candle_core::{DType, Device, Result, Tensor};

/// Attention mask passed to [`Sdpa::run_attention`].
///
/// Encodes both the mask data and the *intent*, whether the attention layer
/// should use flash attention (causal handled by the kernel), eager attention
/// with an explicit mask tensor, or no masking at all.
#[derive(Clone, Debug)]
pub enum AttentionMask {
    /// No masking. Used for single-token decode or truly unmasked attention.
    None,
    /// Flash attention with `is_causal = true`. No mask tensor is needed;
    /// the flash kernel applies causal masking internally. Also signals
    /// "this is a prefill" to the paged attention layer.
    CausalFlash,
    /// An explicit mask tensor (causal, sliding window, bidirectional, etc).
    /// CPU fused attention can consume it directly; other backends route to eager as needed.
    Custom(Tensor),
}

impl AttentionMask {
    /// Extract the inner tensor as `Option<&Tensor>`.
    ///
    /// Returns `Some(&tensor)` for [`Custom`](Self::Custom), `None` otherwise.
    /// Useful for interfacing with paged-attention and MLA helpers that still
    /// accept `Option<&Tensor>`.
    pub fn as_option_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Custom(t) => Some(t),
            _ => None,
        }
    }

    /// Returns `true` when the mask carries an explicit tensor
    /// ([`Custom`](Self::Custom) variant), mirroring the old
    /// `Option<Tensor>::is_some()` semantics.
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }
}

mod backends;

#[allow(unused)]
pub(crate) use backends::{flash_attn, maybe_synchronize, naive_sdpa, sinks_attn};

/// Chunk size for attention computation to avoid OOM on long sequences
pub(crate) const ATTENTION_CHUNK_SIZE: usize = 1024;
const FLASH_ATTN_NATIVE_MAX_GQA_GROUP: usize = 8;

/// Generic chunked attention computation that can be used by different backends
pub(crate) fn chunked_attention<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attention_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>) -> Result<Tensor>,
{
    chunked_attention_with_offset(q, k, v, mask, |q, k, v, mask, _offset| {
        attention_fn(q, k, v, mask)
    })
}

pub(crate) fn chunked_attention_with_offset<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attention_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>, usize) -> Result<Tensor>,
{
    let seq_len = q.dim(2)?;

    if seq_len <= ATTENTION_CHUNK_SIZE {
        return attention_fn(q, k, v, mask, 0);
    }

    let num_chunks = seq_len.div_ceil(ATTENTION_CHUNK_SIZE);
    let mut attn_chunks = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let offset = chunk_idx * ATTENTION_CHUNK_SIZE;
        let chunk_len = ATTENTION_CHUNK_SIZE.min(seq_len - offset);

        // Extract query chunk
        let q_chunk = q.narrow(2, offset, chunk_len)?;

        // Extract mask chunk if present
        let mask_chunk = mask
            .map(|m| {
                match m.rank() {
                    2 => {
                        // For 2D masks (seq_len, seq_len), narrow along dimension 0
                        m.narrow(0, offset, chunk_len)
                    }
                    3 => {
                        // For 3D masks (batch, seq_len, seq_len), narrow along dimension 1
                        m.narrow(1, offset, chunk_len)
                    }
                    4 => {
                        // For 4D masks (batch, heads, seq_len, seq_len), narrow along dimension 2
                        m.narrow(2, offset, chunk_len)
                    }
                    _ => m.narrow(2, offset, chunk_len), // Default to dimension 2
                }
            })
            .transpose()?;

        let att_chunk = attention_fn(&q_chunk, k, v, mask_chunk.as_ref(), offset)?;

        attn_chunks.push(att_chunk);
    }

    Tensor::cat(&attn_chunks, 2)
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

fn run_flash_attn_cpu_for_dtype(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    match q.dtype() {
        DType::F32 => cpu::run_flash_attn_cpu::<f32>(q, k, v, mask, sdpa_params),
        DType::F16 => cpu::run_flash_attn_cpu::<half::f16>(q, k, v, mask, sdpa_params),
        DType::BF16 => cpu::run_flash_attn_cpu::<half::bf16>(q, k, v, mask, sdpa_params),
        _ => Err(candle_core::Error::Msg("Unsupported data type".into())),
    }
}

pub struct SdpaParams {
    pub n_kv_groups: usize,
    pub softcap: Option<f32>,
    pub softmax_scale: f32,
    pub sliding_window: Option<usize>,
    pub sinks: Option<Tensor>,
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
    /// Dispatch attention based on the `AttentionMask` variant:
    ///
    /// - `AttentionMask::CausalFlash`: flash attention with `is_causal = true`
    /// - `AttentionMask::None`: flash if available (decode), else eager without mask
    /// - `AttentionMask::Custom`: CPU fused attention or eager attention with the explicit mask tensor
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &AttentionMask,
        flash_params: Option<&FlashParams>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        // If sinks are present, dispatch to the sinks backend
        if let Some(sinks) = &sdpa_params.sinks {
            let mask_tensor = match mask {
                AttentionMask::Custom(t) => Some(t),
                _ => None,
            };
            return sinks_attn(q, k, v, sinks, mask_tensor, flash_params, sdpa_params);
        }

        // The mask carries causality already; the kernel-level do_causal
        // early-exit is safe to enable only when the request is known causal.
        let do_causal = flash_params.is_some_and(|p| p.causal);

        if let AttentionMask::Custom(mask_tensor) = mask {
            if q.device().is_cpu() {
                let q = q.transpose(1, 2)?;
                let k = k.transpose(1, 2)?;
                let v = v.transpose(1, 2)?;
                return run_flash_attn_cpu_for_dtype(&q, &k, &v, Some(mask_tensor), sdpa_params);
            }

            return self.run_attention_noflash(q, k, v, Some(mask_tensor), sdpa_params, do_causal);
        }

        // CausalFlash or None: try flash attention, fall back to eager
        let can_use_flash = q.device().is_cpu()
            || q.device().is_cuda() && crate::using_flash_attn() && q.dtype() != DType::F32;

        if can_use_flash {
            let expanded_kv = if q.device().is_cuda()
                && crate::using_flash_attn()
                && q.dtype() != DType::F32
                && sdpa_params.n_kv_groups > FLASH_ATTN_NATIVE_MAX_GQA_GROUP
            {
                Some((
                    repeat_kv(k.clone(), sdpa_params.n_kv_groups)?,
                    repeat_kv(v.clone(), sdpa_params.n_kv_groups)?,
                    SdpaParams {
                        n_kv_groups: 1,
                        softcap: sdpa_params.softcap,
                        softmax_scale: sdpa_params.softmax_scale,
                        sliding_window: sdpa_params.sliding_window,
                        sinks: sdpa_params.sinks.clone(),
                    },
                ))
            } else {
                None
            };
            let (k, v, sdpa_params) = match &expanded_kv {
                Some((k, v, sdpa_params)) => (k, v, sdpa_params),
                None => (k, v, sdpa_params),
            };

            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;

            if q.device().is_cpu() {
                return run_flash_attn_cpu_for_dtype(&q, &k, &v, None, sdpa_params);
            } else {
                // hd512 flash kernels drop softcap/sliding-window at compile time; fail loud, not silent.
                let (_, _, _, head_dim) = q.dims4()?;
                if head_dim == 512
                    && (sdpa_params.softcap.is_some_and(|s| s != 1.0)
                        || sdpa_params.sliding_window.is_some())
                {
                    return Err(candle_core::Error::Msg(
                        "flash-attn head_dim 512 kernels are compiled without softcap/sliding-window; \
                         remove the FLASHATTENTION_DISABLE_* defines in \
                         mistralrs-flash-attn/kernels/*hdim512*.cu to re-enable (slow compile)"
                            .to_string(),
                    ));
                }
                return flash_attn(&q, &k, &v, flash_params, sdpa_params)?.transpose(1, 2);
            }
        }

        self.run_attention_noflash(q, k, v, None, sdpa_params, do_causal)
    }

    /// Same as `run_attention`, but skips the flash-attention dispatch.
    ///
    /// `causal` tells the Metal SDPA-full kernel to enable its upper-triangle skip (`do_causal=true`).
    /// Pass `true` only when the caller's mask is causal-or-stricter.
    /// Pass false` for bidirectional masks (e.g. vision attention).
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention_noflash(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sdpa_params: &SdpaParams,
        causal: bool,
    ) -> Result<Tensor> {
        let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
        let (_, _, _, k_head_dim) = k.dims4()?;
        let (_, _, _, v_head_dim) = v.dims4()?;

        // We can use Metal SDPA (vector/full) if the mask is the correct size and head dims match.
        // If the mask is provided, then softcapping isn't allowed - default back to naive SDPA
        // Softcapping is implemented for vector SDPA.
        let all_head_dims_match = head_dim == k_head_dim && k_head_dim == v_head_dim;
        let tgt_mask_shape = vec![b_sz, n_attn_heads, seq_len, k.dim(2)?];
        let can_use_mask = mask.is_none_or(|mask| {
            mask.layout().broadcast_as(tgt_mask_shape.clone()).is_ok()
                && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        });
        let valid_head_dims: &[usize] = &[32, 64, 72, 80, 96, 128, 256, 512];
        // Metal SDPA full kernel requires q_seq <= k_seq when a mask is present.
        let metal_supports_mask = mask.is_none() || seq_len <= k.dim(2)?;

        // Metal FA path for DK=512 BF16 with a mask. Two specializations:
        // prefill (seq_len > 8) goes through the BlockMMA kernel; decode
        // (seq_len == 1) uses a vector FA kernel ported from llama.cpp.
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && head_dim == 512
            && k_head_dim == 512
            && v_head_dim == 512
            && q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && seq_len == 1
            && mask.is_some()
            && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        {
            if let Some(out) =
                crate::attention::backends::metal_flash_attn::try_flash_attn_ext_vec_bf16_dk512(
                    q,
                    k,
                    v,
                    mask,
                    sdpa_params.softmax_scale,
                )?
            {
                return Ok(out);
            }
        }
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && head_dim == 512
            && k_head_dim == 512
            && v_head_dim == 512
            && q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && seq_len > 8
            && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        {
            if let Some(mask) = mask {
                if let Some(out) =
                    crate::attention::backends::metal_flash_attn::try_flash_attn_ext_bf16_dk512(
                        q,
                        k,
                        v,
                        mask,
                        sdpa_params.softmax_scale,
                    )?
                {
                    return Ok(out);
                }
            }
        }

        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && all_head_dims_match
            && valid_head_dims.contains(&head_dim)
            && can_use_mask
            && metal_supports_mask
            && !(head_dim == 512 && seq_len > 8)
        {
            let mask = match mask {
                Some(mask) => Some(mask.broadcast_as(tgt_mask_shape)?),
                None => None,
            };
            // do_causal lets the steel_attention kernel bound its kb-loop to
            // the per-query position, skipping the upper triangle of Q*K^T
            // entirely (roughly halves matmul cost for prefill).
            let do_causal = seq_len > 1 && causal;
            return candle_nn::ops::sdpa(
                q,
                k,
                v,
                mask.as_ref(),
                do_causal,
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
            mistralrs_quant::cublaslt::CUBLASLT_CONTROLLER.get_for_device(q.device()),
        ) {
            #[cfg(feature = "cuda")]
            {
                maybe_synchronize(q.device())?;

                // Use chunked attention for cuBLASLt path
                let k_flat = k.flatten(0, 1)?;
                let v_flat = v.flatten(0, 1)?;

                let kv_len = k.dim(2)?;
                let prefix_len = kv_len.saturating_sub(seq_len);
                chunked_attention_with_offset(
                    q,
                    &k,
                    &v,
                    mask,
                    |q_chunk, _k, _v, mask_chunk, q_offset| {
                        // cuBLASLt batch matmul implementation requires inputs to be dims3
                        let (chunk_b_sz, chunk_n_heads, chunk_seq_len, chunk_head_dim) =
                            q_chunk.dims4()?;
                        let q_flat = q_chunk.flatten(0, 1)?;

                        let attention_bias = match mask_chunk {
                            Some(mask) if mask.rank() == 3 && mask.dims()[0] == 1 => {
                                Some(mask.repeat((chunk_n_heads, 1, 1))?)
                            }
                            Some(mask) if mask.rank() == 3 => Some(mask.clone()),
                            Some(mask) if mask.rank() == 4 => {
                                let tgt_shape =
                                    vec![chunk_b_sz, chunk_n_heads, chunk_seq_len, k.dim(2)?];
                                Some(mask.broadcast_as(tgt_shape)?.flatten(0, 1)?)
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
                            &k_flat,
                            &q_flat,
                            attention_bias.as_ref(),
                            Some(sdpa_params.softmax_scale / sdpa_params.softcap.unwrap_or(1.0)),
                            beta,
                            None,
                            None,
                        )?;
                        if let Some(softcap) = sdpa_params.softcap {
                            attention_scores = (attention_scores.tanh()? * softcap as f64)?;
                        }
                        // Compute softmax in F32 for precision. BF16's 7 mantissa
                        // bits cause exp() to lose information on long sequences.
                        // Flash attention already computes softmax in F32; this
                        // matches that behaviour for the eager path.
                        let scores_dtype = attention_scores.dtype();
                        if scores_dtype == DType::BF16 || scores_dtype == DType::F16 {
                            attention_scores = attention_scores.to_dtype(DType::F32)?;
                        }
                        if causal && mask_chunk.is_none() {
                            crate::ops::cuda_apply_causal_mask_f32(
                                &attention_scores,
                                q_offset,
                                prefix_len,
                            )?;
                        }
                        attention_scores = candle_nn::ops::softmax_last_dim(&attention_scores)?;
                        if attention_scores.dtype() != scores_dtype {
                            attention_scores = attention_scores.to_dtype(scores_dtype)?;
                        }

                        let context_layer = cublaslt.batch_matmul(
                            &v_flat.t()?.contiguous()?,
                            &attention_scores,
                            // We save one allocation
                            Some(&q_flat),
                            None,
                            None,
                            None,
                            None,
                        )?;

                        // Reshape to dims4
                        context_layer.reshape((
                            chunk_b_sz,
                            chunk_n_heads,
                            chunk_seq_len,
                            v_head_dim,
                        ))
                    },
                )
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Result as CandleResult, D};

    const EPS: f32 = 1e-4;

    fn assert_close(lhs: &Tensor, rhs: &Tensor) -> CandleResult<()> {
        let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
        let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
        for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
            assert!((lhs - rhs).abs() < EPS, "{lhs} != {rhs}");
        }
        Ok(())
    }

    #[test]
    fn test_custom_cpu_mask_uses_attention_dispatch() -> CandleResult<()> {
        let (b, h, q_len, kv_len, d) = (1, 2, 3, 3, 4);
        let q = Tensor::from_vec(
            (0..b * h * q_len * d)
                .map(|x| x as f32 / 31.0)
                .collect::<Vec<_>>(),
            (b, h, q_len, d),
            &Device::Cpu,
        )?;
        let k = Tensor::from_vec(
            (0..b * h * kv_len * d)
                .map(|x| x as f32 / 37.0)
                .collect::<Vec<_>>(),
            (b, h, kv_len, d),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(
            (0..b * h * kv_len * d)
                .map(|x| x as f32 / 41.0)
                .collect::<Vec<_>>(),
            (b, h, kv_len, d),
            &Device::Cpu,
        )?;
        let mask = Tensor::from_vec(
            vec![
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
            ],
            (q_len, kv_len),
            &Device::Cpu,
        )?;
        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0,
            sliding_window: None,
            sinks: None,
        };

        let out = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::Custom(mask.clone()),
            Some(&FlashParams::empty(true)),
            &sdpa_params,
        )?;
        let logits = q.matmul(&k.transpose(2, 3)?)?.broadcast_add(&mask)?;
        let expected = candle_nn::ops::softmax(&logits, D::Minus1)?.matmul(&v)?;

        assert_eq!(out.shape().dims(), &[b, h, q_len, d]);
        assert_close(&out, &expected)
    }
}
