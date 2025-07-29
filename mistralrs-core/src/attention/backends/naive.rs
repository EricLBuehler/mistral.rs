#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::MemoryUsage;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::MatMul;

use crate::attention::SdpaParams;

use super::cpu;

/// Not *really* sure why this is necessary but it is.
pub(crate) fn maybe_synchronize(device: &Device) -> Result<()> {
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
            DType::F32 => cpu::run_flash_attn_cpu::<f32>(&q, &k, &v, mask, sdpa_params),
            DType::F16 => cpu::run_flash_attn_cpu::<half::f16>(&q, &k, &v, mask, sdpa_params),
            DType::BF16 => cpu::run_flash_attn_cpu::<half::bf16>(&q, &k, &v, mask, sdpa_params),
            _ => Err(candle_core::Error::Msg("Unsupported data type".into())),
        }
    } else {
        maybe_synchronize(q.device())?;

        // Check if we need to chunk the attention computation
        let seq_len = q.dim(2)?;
        const CHUNK_SIZE: usize = 1024;

        if seq_len > CHUNK_SIZE {
            // Chunk the query to avoid OOM on long sequences
            let num_chunks = seq_len.div_ceil(CHUNK_SIZE);
            let mut attn_chunks = Vec::with_capacity(num_chunks);

            for chunk_idx in 0..num_chunks {
                let offset = chunk_idx * CHUNK_SIZE;
                let chunk_len = CHUNK_SIZE.min(seq_len - offset);

                // Extract query chunk
                let q_chunk = q.narrow(2, offset, chunk_len)?;

                // Compute attention for this chunk
                let mut att = MatMul.matmul_affine_mul(
                    &q_chunk,
                    &k.t()?,
                    sdpa_params.softmax_scale.into(),
                )?;
                if let Some(softcap) = sdpa_params.softcap {
                    att = (att / softcap as f64)?;
                    att = att.tanh()?;
                    att = (att * softcap as f64)?;
                }

                // Apply mask if present
                if let Some(mask) = mask {
                    // Extract the corresponding mask chunk
                    let mask_chunk = mask.narrow(2, offset, chunk_len)?;
                    att = att.broadcast_add(&mask_chunk)?;
                }

                att = candle_nn::ops::softmax_last_dim(&att)?;
                let att_chunk = MatMul.matmul(&att, v)?;

                attn_chunks.push(att_chunk);
            }

            // Concatenate all chunks along the sequence dimension
            Tensor::cat(&attn_chunks, 2)
        } else {
            // Original implementation for shorter sequences
            if let Some(mask) = mask {
                let mut att =
                    MatMul.matmul_affine_mul(q, &k.t()?, sdpa_params.softmax_scale.into())?;
                if let Some(softcap) = sdpa_params.softcap {
                    att = (att / softcap as f64)?;
                    att = att.tanh()?;
                    att = (att * softcap as f64)?;
                }

                att = att.broadcast_add(mask)?;
                att = candle_nn::ops::softmax_last_dim(&att)?;

                MatMul.matmul(&att, v)
            } else {
                let mut att =
                    MatMul.matmul_affine_mul(q, &k.t()?, sdpa_params.softmax_scale.into())?;
                if let Some(softcap) = sdpa_params.softcap {
                    att = (att / softcap as f64)?;
                    att = att.tanh()?;
                    att = (att * softcap as f64)?;
                }

                att = candle_nn::ops::softmax_last_dim(&att)?;
                MatMul.matmul(&att, v)
            }
        }
    }
}
