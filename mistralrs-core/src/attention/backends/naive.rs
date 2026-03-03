#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::MemoryUsage;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::MatMul;

use crate::attention::{chunked_attention, SdpaParams};

/// Not *really* sure why this is necessary but it is.
pub(crate) fn maybe_synchronize(device: &Device) -> Result<()> {
    // If less that 4 GB available, synchronize
    #[cfg(target_pointer_width = "64")]
    const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
    #[cfg(not(target_pointer_width = "64"))]
    const FOUR_GIB: usize = usize::MAX;
    if MemoryUsage.get_memory_available(device)? < FOUR_GIB {
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
    maybe_synchronize(q.device())?;

    // Use chunked attention with a closure that captures the necessary parameters
    chunked_attention(q, k, v, mask, |q_chunk, k, v, mask_chunk| {
        let mut att =
            MatMul.matmul_affine_mul(q_chunk, &k.t()?, sdpa_params.softmax_scale.into())?;

        if let Some(softcap) = sdpa_params.softcap {
            att = (att / softcap as f64)?;
            att = att.tanh()?;
            att = (att * softcap as f64)?;
        }

        if let Some(mask) = mask_chunk {
            att = att.broadcast_add(mask)?;
        }

        att = candle_nn::ops::softmax_last_dim(&att)?;
        MatMul.matmul(&att, v)
    })
}
