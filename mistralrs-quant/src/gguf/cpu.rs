//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes the weights and delegates to UnquantLinear's gather_forward.

use std::borrow::Cow;
use std::sync::Arc;

use candle_core::{
    quantized::{QMatMul, QStorage, QTensor},
    DType, Device, Result, Tensor,
};
use candle_nn::Linear;

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

/// Perform indexed MoE forward pass on a QTensor by dequantizing and using UnquantLinear.
///
/// # Arguments
/// * `qtensor` - The quantized weight tensor [num_experts, n, k]
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn qtensor_indexed_moe_forward(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    // Repacked per-expert gemv path; falls back to dequantize-and-gather only for
    // layouts the packed kernels cannot serve. Normalize the metal/cpu 4D/5D input
    // shapes to the (tokens, x_t, hidden) form the kernel expects.
    {
        let (x3, ids2, out_shape): (Tensor, Tensor, Option<Vec<usize>>) = match *x.dims() {
            [b, s, xt, h] => {
                let (ib, is, t) = ids.dims3()?;
                if ib == b && is == s {
                    (
                        x.reshape((b * s, xt, h))?,
                        ids.reshape((b * s, t))?,
                        Some(vec![b, s, t, 0]),
                    )
                } else {
                    (x.clone(), ids.clone(), None)
                }
            }
            [b, s, 1, 1, h] => {
                let (ib, is, t) = ids.dims3()?;
                if ib == b && is == s {
                    (
                        x.reshape((b * s, 1, h))?,
                        ids.reshape((b * s, t))?,
                        Some(vec![b, s, t, 0]),
                    )
                } else {
                    (x.clone(), ids.clone(), None)
                }
            }
            [_, _, _] => (x.clone(), ids.clone(), Some(vec![])),
            _ => (x.clone(), ids.clone(), None),
        };
        if let Some(shape) = out_shape.filter(|_| x3.rank() == 3 && ids2.rank() == 2) {
            if let Some(out) = qtensor.indexed_gemv(&x3, &ids2)? {
                return if shape.is_empty() {
                    Ok(out)
                } else {
                    let n_out = out.dim(2)?;
                    out.reshape((shape[0], shape[1], shape[2], n_out))
                };
            }
        }
    }

    let device = x.device();

    // Each expert is its own byte slab. Dequantize the ones this token named.
    if let Some((weights, ids)) = dequantize_selected(qtensor, ids, device)? {
        let unquant =
            UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;
        return unquant.gather_forward(x, &ids);
    }

    // Dequantize all weights to f32
    let weights = qtensor.dequantize(device)?;

    // Create an UnquantLinear and use its gather_forward
    let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;

    unquant.gather_forward(x, ids)
}

/// The experts named by `ids`, dequantized, with `ids` remapped onto that shorter stack.
fn dequantize_selected(
    qtensor: &QTensor,
    ids: &Tensor,
    device: &Device,
) -> Result<Option<(Tensor, Tensor)>> {
    let &[n_experts, out_features, in_features] = qtensor.shape().dims() else {
        return Ok(None);
    };
    if n_experts == 0 {
        return Ok(None);
    }
    let total = qtensor.storage_size_in_bytes();
    if !total.is_multiple_of(n_experts) {
        return Ok(None);
    }
    let stride = total / n_experts;

    let flat = ids.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
    if flat.is_empty() || flat.iter().any(|&id| id as usize >= n_experts) {
        return Ok(None);
    }
    let mut wanted: Vec<usize> = flat.iter().map(|&id| id as usize).collect();
    wanted.sort_unstable();
    wanted.dedup();
    // Naming every expert is the same work as one dequant of the stack.
    if wanted.len() == n_experts {
        return Ok(None);
    }

    let bytes = qtensor.data()?;
    let dtype = qtensor.dtype();
    let stored_on = qtensor.device();
    let mut slabs = Vec::with_capacity(wanted.len());
    for &expert in &wanted {
        let start = expert * stride;
        let Some(slab) = bytes.get(start..start + stride) else {
            return Ok(None);
        };
        let storage = QStorage::from_data(Cow::Borrowed(slab), &stored_on, dtype)?;
        let expert_q = QTensor::new(storage, (out_features, in_features))?;
        slabs.push(expert_q.dequantize(device)?);
    }
    let weights = Tensor::stack(&slabs, 0)?;

    let mut remap = vec![0u32; n_experts];
    for (row, &expert) in wanted.iter().enumerate() {
        remap[expert] = row as u32;
    }
    let remapped: Vec<u32> = flat.iter().map(|&id| remap[id as usize]).collect();
    let remapped = Tensor::from_vec(remapped, ids.dims(), ids.device())?;
    let remapped = if remapped.dtype() == ids.dtype() {
        remapped
    } else {
        remapped.to_dtype(ids.dtype())?
    };
    Ok(Some((weights, remapped)))
}

/// Perform indexed MoE forward pass on a QMatMul.
///
/// This is the main entry point for CPU/Metal GGUF quantized MoE forward.
///
/// # Arguments
/// * `qmatmul` - The quantized weight matrix
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn cpu_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => qtensor_indexed_moe_forward(qtensor, x, ids),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            // For non-quantized tensors, use UnquantLinear directly
            let unquant =
                UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(t.clone(), None)))?;
            unquant.gather_forward(x, ids)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::Linear;

    use crate::{QuantMethodConfig, UnquantLinear};

    fn whole_stack(q: &QTensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let weights = q.dequantize(x.device())?;
        let unquant =
            UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;
        unquant.gather_forward(x, ids)
    }

    fn bits(t: &Tensor) -> Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    fn assert_same(q: &Arc<QTensor>, x: &Tensor, ids: &Tensor) -> Result<()> {
        let got = qtensor_indexed_moe_forward(q, x, ids)?;
        let want = whole_stack(q, x, ids)?;
        assert_eq!(got.dims(), want.dims());
        assert_eq!(bits(&got)?, bits(&want)?);
        Ok(())
    }

    /// Q2_K and Q4_1 miss the packed gemv, and the ids leave an expert out.
    #[test]
    fn selected_experts_match_the_whole_stack() -> Result<()> {
        let dev = Device::Cpu;
        let cases = [
            (GgmlDType::Q4_1, 8usize, 32usize, 32usize),
            (GgmlDType::Q2K, 4usize, 256usize, 256usize),
        ];
        for (dtype, experts, out, k) in cases {
            let mut data = vec![0f32; experts * out * k];
            for expert in 0..experts {
                for i in 0..out * k {
                    data[expert * out * k + i] = (expert as f32) * 3.0 + (i as f32) * 0.015625;
                }
            }
            let src = Tensor::from_vec(data, (experts, out, k), &dev)?;
            let q = Arc::new(QTensor::quantize(&src, dtype)?);
            let (b, s, t) = (2usize, 2usize, 3usize);
            let ids: Vec<u32> = (0..b * s * t).map(|i| (i % (experts - 1)) as u32).collect();
            let ids3 = Tensor::from_vec(ids.clone(), (b, s, t), &dev)?;
            let ids2 = Tensor::from_vec(ids, (b * s, t), &dev)?;

            assert_same(
                &q,
                &Tensor::rand(-1f32, 1f32, (b, s, 1, 1, k), &dev)?,
                &ids3,
            )?;
            assert_same(&q, &Tensor::rand(-1f32, 1f32, (b, s, t, k), &dev)?, &ids3)?;
            assert_same(&q, &Tensor::rand(-1f32, 1f32, (b * s, 1, k), &dev)?, &ids2)?;
            assert_same(&q, &Tensor::rand(-1f32, 1f32, (b * s, t, k), &dev)?, &ids2)?;
        }
        Ok(())
    }

    /// A non-contiguous activation still matches the whole-stack matmul.
    #[test]
    fn selected_experts_match_when_the_activation_is_not_contiguous() -> Result<()> {
        let dev = Device::Cpu;
        let (experts, out, k) = (8usize, 32usize, 32usize);
        let src = Tensor::rand(-1f32, 1f32, (experts, out, k), &dev)?;
        let q = Arc::new(QTensor::quantize(&src, GgmlDType::Q4_1)?);
        let (b, s, t) = (1usize, 2usize, 4usize);
        let ids = Tensor::from_vec(vec![1u32, 1, 4, 6, 0, 0, 3, 3], (b, s, t), &dev)?;
        let wide = Tensor::rand(-1f32, 1f32, (b, s, 1, 1, k * 2), &dev)?;
        let x = wide.narrow(4, 1, k)?;
        assert!(!x.is_contiguous());
        assert_same(&q, &x, &ids)
    }
}
