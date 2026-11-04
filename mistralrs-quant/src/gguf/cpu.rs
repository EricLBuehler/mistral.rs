//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes the weights and delegates to UnquantLinear's gather_forward.

use candle_core::{
    quantized::{QMatMul, QStorage, QTensor},
    DType, Module, Result, Tensor,
};
use candle_nn::Linear;
use std::borrow::Cow;
use std::sync::Arc;

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

/// Per-expert quantized matmul: route each (token, slot) pair to its expert's slice of
/// the stacked [num_experts, n, k] QTensor without dequantizing the unused experts.
/// Returns None for layouts this path cannot serve.
fn sparse_indexed_moe(qtensor: &Arc<QTensor>, x: &Tensor, ids: &Tensor) -> Result<Option<Tensor>> {
    if !x.device().is_cpu() || !qtensor.device().is_cpu() {
        return Ok(None);
    }
    let Ok((n_experts, n, k)) = qtensor.shape().dims3() else {
        return Ok(None);
    };
    let Ok((batch, x_t, xk)) = x.dims3() else {
        return Ok(None);
    };
    let Ok((ids_b, topk)) = ids.dims2() else {
        return Ok(None);
    };
    if xk != k || ids_b != batch || (x_t != 1 && x_t != topk) {
        return Ok(None);
    }
    let in_dtype = x.dtype();
    let x = match in_dtype {
        DType::F32 => x.clone(),
        DType::F16 | DType::BF16 => x.to_dtype(DType::F32)?,
        _ => return Ok(None),
    };
    if !x.is_contiguous() {
        return Ok(None);
    }
    let ids_v: Vec<u32> = ids.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
    if ids_v.iter().any(|&e| e as usize >= n_experts) {
        candle_core::bail!("expert index out of range");
    }
    let n_pairs = ids_v.len();
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
    for (pair, &e) in ids_v.iter().enumerate() {
        buckets[e as usize].push(pair);
    }
    let n_rows = batch * x_t;
    let x_flat = x.reshape((n_rows, k))?;
    let bytes = qtensor.data()?;
    let total = qtensor.storage_size_in_bytes();
    if !total.is_multiple_of(n_experts) || bytes.len() < total {
        return Ok(None);
    }
    let per_expert = total / n_experts;
    let dtype = qtensor.dtype();
    let mut dst = vec![0f32; n_pairs * n];
    for (e, bucket) in buckets.iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let row_idx: Vec<u32> = bucket
            .iter()
            .map(|&pair| if n_rows == n_pairs { pair } else { pair / topk } as u32)
            .collect();
        let rows = x_flat.index_select(&Tensor::from_vec(row_idx, bucket.len(), x.device())?, 0)?;
        let storage = dtype.from_data(Cow::Borrowed(&bytes[e * per_expert..(e + 1) * per_expert]));
        let expert = QMatMul::from_qtensor(QTensor::new(QStorage::Cpu(storage), (n, k))?)?;
        let out = expert.forward(&rows)?.reshape((bucket.len() * n,))?;
        let out_v: Vec<f32> = out.to_vec1()?;
        for (i, &pair) in bucket.iter().enumerate() {
            dst[pair * n..(pair + 1) * n].copy_from_slice(&out_v[i * n..(i + 1) * n]);
        }
    }
    let out = Tensor::from_vec(dst, (batch, topk, n), x.device())?;
    let out = match in_dtype {
        DType::F32 => out,
        other => out.to_dtype(other)?,
    };
    Ok(Some(out))
}

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

    if let Some(out) = sparse_indexed_moe(qtensor, x, ids)? {
        return Ok(out);
    }

    let device = x.device();

    // Dequantize all weights to f32
    let weights = qtensor.dequantize(device)?;

    // Create an UnquantLinear and use its gather_forward
    let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;

    unquant.gather_forward(x, ids)
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
    use candle_core::quantized::GgmlDType;
    use candle_core::{Device, Tensor};

    #[test]
    fn sparse_indexed_moe_matches_dequantize_gather() -> Result<()> {
        let dev = Device::Cpu;
        let (e, n, k) = (8, 64, 128);
        let w = Tensor::randn(0f32, 1.0, (e, n, k), &dev)?;
        let qt = Arc::new(QTensor::quantize(&w, GgmlDType::F32)?);

        let (batch, topk) = (3, 2);
        let x = Tensor::randn(0f32, 1.0, (batch, 1, k), &dev)?;
        let ids = Tensor::from_vec(vec![0u32, 5, 7, 1, 3, 3], (batch, topk), &dev)?;

        let sparse = sparse_indexed_moe(&qt, &x, &ids)?.expect("sparse path should serve this");
        let weights = qt.dequantize(&dev)?;
        let unquant =
            UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;
        let reference = unquant.gather_forward(&x, &ids)?;

        let diff = (sparse - reference)?
            .abs()?
            .max(0)?
            .max(0)?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "max abs diff {diff}");
        Ok(())
    }

    #[test]
    fn sparse_indexed_moe_quantized_close_to_dequantize_gather() -> Result<()> {
        let dev = Device::Cpu;
        let (e, n, k) = (8, 64, 128);
        let w = Tensor::randn(0f32, 1.0, (e, n, k), &dev)?;
        let qt = Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0)?);

        let (batch, topk) = (3, 2);
        let x = Tensor::randn(0f32, 1.0, (batch, 1, k), &dev)?;
        let ids = Tensor::from_vec(vec![0u32, 5, 7, 1, 3, 3], (batch, topk), &dev)?;

        let sparse = sparse_indexed_moe(&qt, &x, &ids)?.expect("sparse path should serve this");
        let weights = qt.dequantize(&dev)?;
        let unquant =
            UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;
        let reference = unquant.gather_forward(&x, &ids)?;

        // quantized activation dot (ggml-style vec_dot) vs plain f32 matmul: allow
        // quantization noise, catch routing/indexing errors which are O(1) magnitude
        let diff = (sparse - &reference)?
            .abs()?
            .max(0)?
            .max(0)?
            .max(0)?
            .to_scalar::<f32>()?;
        let scale = reference
            .abs()?
            .max(0)?
            .max(0)?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < 0.05 * scale,
            "max abs diff {diff} vs output scale {scale}"
        );
        Ok(())
    }
}
