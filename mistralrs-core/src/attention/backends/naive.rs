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

        // Upcast F16/BF16 to F32 before softcap and softmax to prevent:
        //   1. NaN overflow in tanh() for large attention values in F16 (±65504 range)
        //   2. Precision loss in exp() during softmax (BF16 has only 7 mantissa bits)
        // We upcast once here and stay in F32 through both softcap and softmax,
        // doing a single downcast at the end.
        let att_dtype = att.dtype();
        if att_dtype == candle_core::DType::BF16 || att_dtype == candle_core::DType::F16 {
            att = att.to_dtype(candle_core::DType::F32)?;
        }

        if let Some(softcap) = sdpa_params.softcap {
            att = (att / softcap as f64)?;
            att = att.tanh()?;
            att = (att * softcap as f64)?;
        }

        if let Some(mask) = mask_chunk {
            let mask = if mask.dtype() != att.dtype() {
                mask.to_dtype(att.dtype())?
            } else {
                mask.clone()
            };
            att = att.broadcast_add(&mask)?;
        }

        att = candle_nn::ops::softmax_last_dim(&att)?;
        if att.dtype() != att_dtype {
            att = att.to_dtype(att_dtype)?;
        }
        MatMul.matmul(&att, v)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::DType;

    #[test]
    fn softcap_attention_keeps_large_bf16_scores_finite() -> Result<()> {
        let device = Device::Cpu;
        let q = Tensor::from_vec(vec![1000f32, -1000., -1000., 1000.], (1, 1, 2, 2), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::from_vec(vec![1000f32, -1000., -1000., 1000.], (1, 1, 2, 2), &device)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 2, 2), &device)?
            .to_dtype(DType::BF16)?;
        let params = SdpaParams {
            n_kv_groups: 1,
            softcap: Some(50.0),
            softmax_scale: 1.0,
            sliding_window: None,
            sinks: None,
        };

        let got = naive_sdpa(&q, &k, &v, None, &params)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert!(got.iter().all(|x| x.is_finite()), "got {got:?}");
        Ok(())
    }

    #[test]
    fn softcap_attention_upcasts_bf16_mask_with_scores() -> Result<()> {
        let device = Device::Cpu;
        let q = Tensor::from_vec(vec![1000f32, -1000., -1000., 1000.], (1, 1, 2, 2), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::from_vec(vec![1000f32, -1000., -1000., 1000.], (1, 1, 2, 2), &device)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 2, 2), &device)?
            .to_dtype(DType::BF16)?;
        let mask = Tensor::from_vec(vec![0f32, 0., 0., 0.], (1, 1, 2, 2), &device)?
            .to_dtype(DType::BF16)?;
        let params = SdpaParams {
            n_kv_groups: 1,
            softcap: Some(50.0),
            softmax_scale: 1.0,
            sliding_window: None,
            sinks: None,
        };

        let got = naive_sdpa(&q, &k, &v, Some(&mask), &params)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert!(got.iter().all(|x| x.is_finite()), "got {got:?}");
        Ok(())
    }
}
