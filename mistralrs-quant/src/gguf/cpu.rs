//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes the weights and delegates to UnquantLinear's gather_forward.

use candle_core::{
    quantized::{ggml_file::qtensor_from_ggml, QMatMul, QTensor},
    DType, Module, Result, Tensor,
};
use candle_nn::Linear;
use std::sync::{Arc, OnceLock};

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

/// Per-expert quantized matmuls for one stacked expert tensor, built lazily on first use.
#[derive(Debug, Default)]
pub(crate) struct ExpertMatMuls {
    slots: OnceLock<Vec<OnceLock<QMatMul>>>,
}

impl ExpertMatMuls {
    fn get_or_build(
        &self,
        n_experts: usize,
        expert: usize,
        build: impl FnOnce() -> Result<QMatMul>,
    ) -> Result<QMatMul> {
        let slots = self
            .slots
            .get_or_init(|| (0..n_experts).map(|_| OnceLock::new()).collect());
        if let Some(matmul) = slots[expert].get() {
            return Ok(matmul.clone());
        }
        let matmul = build()?;
        Ok(slots[expert].get_or_init(|| matmul).clone())
    }
}

fn selected_experts_forward(
    qtensor: &QTensor,
    cache: Option<&ExpertMatMuls>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Option<Tensor>> {
    let Ok((n_experts, n_out, k)) = qtensor.shape().dims3() else {
        return Ok(None);
    };
    let Ok((tokens, x_t, x_k)) = x.dims3() else {
        return Ok(None);
    };
    let Ok((ids_tokens, topk)) = ids.dims2() else {
        return Ok(None);
    };
    if !x.device().is_cpu()
        || x.dtype() != DType::F32
        || x_k != k
        || ids_tokens != tokens
        || (x_t != 1 && x_t != topk)
    {
        return Ok(None);
    }
    let dtype = qtensor.dtype();
    let expert_elems = n_out * k;
    if expert_elems % dtype.block_size() != 0 {
        return Ok(None);
    }
    let expert_bytes = expert_elems / dtype.block_size() * dtype.type_size();
    let data = qtensor.data()?;
    if data.len() != n_experts * expert_bytes {
        return Ok(None);
    }

    let ids_flat = ids.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
    let mut by_expert: Vec<Vec<u32>> = vec![Vec::new(); n_experts];
    for (assignment, &id) in ids_flat.iter().enumerate() {
        let id = id as usize;
        if id >= n_experts {
            candle_core::bail!("expert index {id} out of range for {n_experts} experts");
        }
        by_expert[id].push(assignment as u32);
    }

    let device = x.device();
    let x_rows = x.reshape((tokens * x_t, k))?;
    let mut outputs = Vec::new();
    let mut order: Vec<u32> = Vec::with_capacity(ids_flat.len());
    for (expert, assignments) in by_expert.iter().enumerate() {
        if assignments.is_empty() {
            continue;
        }
        let build = || {
            let bytes = &data[expert * expert_bytes..(expert + 1) * expert_bytes];
            QMatMul::from_qtensor(qtensor_from_ggml(dtype, bytes, vec![n_out, k], device)?)
        };
        let matmul = match cache {
            Some(cache) => cache.get_or_build(n_experts, expert, build)?,
            None => build()?,
        };
        let rows: Vec<u32> = assignments
            .iter()
            .map(|&a| {
                let (token, slot) = (a as usize / topk, a as usize % topk);
                (token * x_t + if x_t == 1 { 0 } else { slot }) as u32
            })
            .collect();
        let rows = Tensor::from_vec(rows, assignments.len(), device)?;
        outputs.push(matmul.forward(&x_rows.index_select(&rows, 0)?)?);
        order.extend_from_slice(assignments);
    }

    let mut inverse = vec![0u32; order.len()];
    for (position, &assignment) in order.iter().enumerate() {
        inverse[assignment as usize] = position as u32;
    }
    let inverse = Tensor::from_vec(inverse, order.len(), device)?;
    Tensor::cat(&outputs, 0)?
        .index_select(&inverse, 0)?
        .reshape((tokens, topk, n_out))
        .map(Some)
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
pub(crate) fn qtensor_indexed_moe_forward(
    qtensor: &Arc<QTensor>,
    cache: Option<&ExpertMatMuls>,
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

    if let Some(out) = selected_experts_forward(qtensor, cache, x, ids)? {
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
    cpu_indexed_moe_forward_cached(qmatmul, None, x, ids)
}

pub(crate) fn cpu_indexed_moe_forward_cached(
    qmatmul: &QMatMul,
    cache: Option<&ExpertMatMuls>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => qtensor_indexed_moe_forward(qtensor, cache, x, ids),
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
    use candle_core::{quantized::GgmlDType, Device};

    use super::*;

    const EXPERTS: usize = 6;
    const N_OUT: usize = 8;
    const K: usize = 256;
    const MAX_REL_ERR: f32 = 0.05;

    fn reference(qtensor: &QTensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let weights = qtensor.dequantize(x.device())?;
        let unquant =
            UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;
        unquant.gather_forward(x, ids)
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        let max_abs = expected.abs()?.max_all()?.to_scalar::<f32>()?;
        let diff = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            diff <= MAX_REL_ERR * max_abs,
            "diff {diff} vs max {max_abs}"
        );
        Ok(())
    }

    fn run(x_t: usize) -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 1f32, (EXPERTS, N_OUT, K), &device)?;
        let qtensor = QTensor::quantize(&weight, GgmlDType::Q4K)?;
        let x = Tensor::randn(0f32, 1f32, (3, x_t, K), &device)?;
        let ids = Tensor::new(&[[0u32, 5], [5, 2], [2, 0]], &device)?;
        let expected = reference(&qtensor, &x, &ids)?;

        let cache = ExpertMatMuls::default();
        for _ in 0..2 {
            let actual = selected_experts_forward(&qtensor, Some(&cache), &x, &ids)?
                .expect("layout is supported");
            assert_close(&actual, &expected)?;
        }
        let uncached =
            selected_experts_forward(&qtensor, None, &x, &ids)?.expect("layout is supported");
        assert_close(&uncached, &expected)
    }

    #[test]
    fn selected_experts_match_full_dequantize_shared_input() -> Result<()> {
        run(1)
    }

    #[test]
    fn selected_experts_match_full_dequantize_per_slot_input() -> Result<()> {
        run(2)
    }

    #[test]
    fn selected_experts_reject_out_of_range_ids() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 1f32, (EXPERTS, N_OUT, K), &device)?;
        let qtensor = QTensor::quantize(&weight, GgmlDType::Q4K)?;
        let x = Tensor::randn(0f32, 1f32, (1, 1, K), &device)?;
        let ids = Tensor::new(&[[EXPERTS as u32, 0]], &device)?;
        assert!(selected_experts_forward(&qtensor, None, &x, &ids).is_err());
        Ok(())
    }
}
