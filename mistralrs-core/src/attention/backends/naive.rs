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

        if let Some(mask) = mask {
            let mut att = MatMul.matmul_affine_mul(q, &k.t()?, sdpa_params.softmax_scale.into())?;
            if let Some(softcap) = sdpa_params.softcap {
                att = (att / softcap as f64)?;
                att = att.tanh()?;
                att = (att * softcap as f64)?;
            }

            att = att.broadcast_add(mask)?;
            att = candle_nn::ops::softmax_last_dim(&att)?;

            MatMul.matmul(&att, v)
        } else {
            let mut att = MatMul.matmul_affine_mul(q, &k.t()?, sdpa_params.softmax_scale.into())?;
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
