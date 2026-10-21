//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes the weights and delegates to UnquantLinear's gather_forward.

use candle_core::{
    quantized::{QMatMul, QTensor},
    Result, Tensor,
};
use candle_nn::Linear;
use std::sync::Arc;

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

    // Scoped path: dequantise only the experts the gather names. The stack is one quantised
    // block, so dequantising it costs the whole model per forward; see
    // `todo/mistralrs/02`.
    if let Some(out) = scoped_gather(qtensor, x, ids)? {
        return Ok(out);
    }

    // Dequantize all weights to f32
    let weights = qtensor.dequantize(device)?;

    // Create an UnquantLinear and use its gather_forward
    let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;

    unquant.gather_forward(x, ids)
}

/// The scoped gather: dequantise only the routed experts and run the per-expert gather.
///
/// Returns `None` when the shapes are not ones this handles, so the caller falls back to
/// dequantising the whole stack. The per-expert gather is bit-identical to the fused path by
/// construction, so this changes what is dequantised and not what is computed.
fn scoped_gather(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Option<Tensor>> {
    use crate::per_expert_quantized::PerExpertQuantizedLinear;

    // The shapes the MoE is called with, per the QuantMethod contract: x is
    // (b, s, 1, 1, h) or (tokens, 1, h), and ids is (b, s, t) or (tokens, t). Both carry one
    // activation row per token, which the token's t slots read in turn.
    // Only the 5D form is taken. The 3D form's whole-stack path is a matmul over the
    // selected experts rather than the gather kernel, and a matmul and the gather are not the
    // same arithmetic, so replacing it there would change the output rather than just what
    // is dequantised.
    let (n_tokens, h, out_shape) = match x.dims() {
        [b, s, 1, 1, h] => (b * s, *h, vec![*b, *s]),
        _ => return Ok(None),
    };
    let t = match ids.dims() {
        [ib, is, t] if ib * is == n_tokens => *t,
        _ => return Ok(None),
    };
    if t == 0 {
        return Ok(None);
    }

    let (n_expert, out_features, in_features) = match qtensor.shape().dims() {
        [n, out, k] => (*n, *out, *k),
        _ => return Ok(None),
    };
    if in_features != h {
        return Ok(None);
    }

    // The whole-stack path falls back to a matmul when the gather kernel would not serve the
    // shape, and a matmul and the gather are not the same arithmetic. So the scoped path takes
    // over only where the gather would have been used.
    let n_slots = ids.elem_count();
    if !crate::moe_gather_cpu::should_use_gather_gemv_shape(
        n_expert,
        out_features,
        in_features,
        out_features,
        n_slots,
    ) {
        return Ok(None);
    }

    let linear = PerExpertQuantizedLinear::new(
        qtensor.clone(),
        n_expert,
        out_features,
        in_features,
    )?;

    // One activation row per token; the token's t slots read it in turn.
    let rows = x.reshape((n_tokens, in_features))?;
    let a: Vec<f32> = rows
        .to_dtype(candle_core::DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let ids_flat: Vec<u32> = ids.flatten_all()?.to_vec1::<u32>()?.to_vec();
    let out = linear.gather_slots(&a, &ids_flat, t)?;
    let out = Tensor::from_vec(out, (n_tokens, t, out_features), x.device())?;
    let mut shape = out_shape;
    shape.push(t);
    shape.push(out_features);
    Ok(Some(out.reshape(shape)?))
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

    /// The scoped gather has to produce exactly what the whole-stack path produces. It changes
    /// what is dequantised, not what is computed, so any difference here is a bug in the
    /// slicing, the remap, or the row addressing.
    #[test]
    fn scoped_gather_matches_the_whole_stack_bit_exactly() -> Result<()> {
        use candle_core::DType;

        for (label, five_d) in [("5d", true)] {
            // The shapes have to be big enough for the gather kernel to serve them, because
            // that is when the whole-stack path uses it too.
            let (b, s, t, h, out, experts) = (2usize, 3usize, 3usize, 64usize, 64usize, 8usize);
            let dev = candle_core::Device::Cpu;

            // Distinctive values, so a wrong slab is obvious rather than merely different.
            let mut data = vec![0f32; experts * out * h];
            for e in 0..experts {
                for i in 0..out * h {
                    data[e * out * h + i] = (e as f32) * 7.0 + (i as f32) * 0.015625;
                }
            }
            let src = Tensor::from_vec(data, (experts, out, h), &dev)?;
            let q = std::sync::Arc::new(QTensor::quantize(&src, GgmlDType::Q4_0)?);

            let x = if five_d {
                Tensor::rand(-1f32, 1f32, (b, s, 1, 1, h), &dev)?
            } else {
                Tensor::rand(-1f32, 1f32, (b * s, 1, h), &dev)?
            };
            let ids = if five_d {
                Tensor::from_vec(
                    (0..b * s * t).map(|i| (i * 5 % experts) as u32).collect::<Vec<_>>(),
                    (b, s, t),
                    &dev,
                )?
            } else {
                Tensor::from_vec(
                    (0..b * s * t).map(|i| (i * 5 % experts) as u32).collect::<Vec<_>>(),
                    (b * s, t),
                    &dev,
                )?
            };

            let whole = {
                let weights = q.dequantize(&dev)?;
                let unquant =
                    UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;
                unquant.gather_forward(&x, &ids)?
            };
            let scoped = scoped_gather(&q, &x, &ids)?.expect("scoped path should handle these shapes");

            assert_eq!(whole.dims(), scoped.dims(), "{label}: shape");
            let a = whole.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let b = scoped.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(a, b, "{label}: values differ");
        }
        Ok(())
    }
}
