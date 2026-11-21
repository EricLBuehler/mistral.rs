//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes the weights and delegates to UnquantLinear's gather_forward.

use candle_core::{
    quantized::{ggml_file::qtensor_from_ggml, QMatMul, QTensor},
    DType, Module, Result, Tensor,
};
use candle_nn::Linear;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

/// Default byte budget for the repacked per-expert matmul cache on one stacked expert tensor
/// (e.g. one MoE layer's gate_proj across all experts). Override with
/// `MISTRALRS_CPU_MOE_EXPERT_CACHE_BYTES`; `0` disables retention entirely (build every call).
const DEFAULT_EXPERT_CACHE_BYTES: usize = 64 * 1024 * 1024;
const EXPERT_CACHE_BYTES_ENV: &str = "MISTRALRS_CPU_MOE_EXPERT_CACHE_BYTES";

fn expert_cache_budget_bytes() -> usize {
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var(EXPERT_CACHE_BYTES_ENV)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_EXPERT_CACHE_BYTES)
    })
}

fn qmatmul_retained_bytes(m: &QMatMul) -> usize {
    match m {
        QMatMul::QTensor(t) => t.storage_size_in_bytes(),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.elem_count() * t.dtype().size_in_bytes(),
    }
}

#[derive(Debug, Default)]
enum ExpertSlot {
    #[default]
    Empty,
    Building,
    Ready(QMatMul, usize),
}

#[derive(Debug, Default)]
struct ExpertCacheMeta {
    slots: Vec<ExpertSlot>,
    lru: VecDeque<usize>,
    bytes: usize,
}

/// Per-expert quantized matmuls for one stacked expert tensor, built lazily and bounded to a
/// process-wide byte budget (see `DEFAULT_EXPERT_CACHE_BYTES`).
///
/// Construction is single-flight per expert: a thread that finds its expert `Building` waits on
/// `ready` instead of repacking the same bytes again. Different experts build fully concurrently,
/// since the mutex is only held for state transitions, never across `build()` or the matmul
/// itself. Eviction is plain LRU over the experts currently `Ready`; an expert whose repacked size
/// alone exceeds the budget still runs, it just isn't retained afterward. Accounting drops a slot
/// from the budget the instant it's evicted, even though a caller already holding a clone of that
/// `QMatMul` keeps its backing storage alive until it finishes using it, so live memory can
/// transiently exceed the budget by whatever a single in-flight expert needs.
#[derive(Debug, Default)]
pub(crate) struct ExpertMatMuls {
    meta: Mutex<Option<ExpertCacheMeta>>,
    ready: Condvar,
    #[cfg(test)]
    budget_override: OnceLock<usize>,
}

impl ExpertMatMuls {
    #[cfg(test)]
    fn set_budget_for_test(&self, bytes: usize) {
        self.budget_override
            .set(bytes)
            .expect("budget must be set before first use in tests");
    }

    fn budget_bytes(&self) -> usize {
        #[cfg(test)]
        if let Some(bytes) = self.budget_override.get() {
            return *bytes;
        }
        expert_cache_budget_bytes()
    }

    fn get_or_build(
        &self,
        n_experts: usize,
        expert: usize,
        build: impl FnOnce() -> Result<QMatMul>,
    ) -> Result<QMatMul> {
        let budget = self.budget_bytes();
        let mut guard = self.meta.lock().unwrap();
        if guard.is_none() {
            *guard = Some(ExpertCacheMeta {
                slots: (0..n_experts).map(|_| ExpertSlot::Empty).collect(),
                lru: VecDeque::new(),
                bytes: 0,
            });
        }
        loop {
            let meta = guard.as_mut().expect("initialized above");
            match &meta.slots[expert] {
                ExpertSlot::Ready(matmul, _) => return Ok(matmul.clone()),
                ExpertSlot::Building => {
                    guard = self.ready.wait(guard).unwrap();
                    continue;
                }
                ExpertSlot::Empty => {
                    meta.slots[expert] = ExpertSlot::Building;
                    break;
                }
            }
        }
        drop(guard);

        let outcome = build();

        let mut guard = self.meta.lock().unwrap();
        let meta = guard.as_mut().expect("initialized above, never cleared");
        let result = match outcome {
            Ok(matmul) => {
                let bytes = qmatmul_retained_bytes(&matmul);
                if bytes > budget {
                    meta.slots[expert] = ExpertSlot::Empty;
                } else {
                    while meta.bytes + bytes > budget {
                        let Some(victim) = meta.lru.pop_front() else {
                            break;
                        };
                        if let ExpertSlot::Ready(_, victim_bytes) =
                            std::mem::replace(&mut meta.slots[victim], ExpertSlot::Empty)
                        {
                            meta.bytes -= victim_bytes;
                        }
                    }
                    meta.slots[expert] = ExpertSlot::Ready(matmul.clone(), bytes);
                    meta.lru.push_back(expert);
                    meta.bytes += bytes;
                }
                Ok(matmul)
            }
            Err(e) => {
                meta.slots[expert] = ExpertSlot::Empty;
                Err(e)
            }
        };
        drop(guard);
        self.ready.notify_all();
        result
    }

    #[cfg(test)]
    fn cached_bytes(&self) -> usize {
        self.meta
            .lock()
            .unwrap()
            .as_ref()
            .map(|meta| meta.bytes)
            .unwrap_or(0)
    }

    #[cfg(test)]
    fn is_ready(&self, expert: usize) -> bool {
        self.meta
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|meta| matches!(meta.slots[expert], ExpertSlot::Ready(..)))
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::*;

    const EXPERTS: usize = 6;
    const N_OUT: usize = 8;
    const K: usize = 256;

    // `reference()` dequantizes to f32 and runs a plain f32 matmul. The optimized path instead
    // runs candle's quantized `k_quants::matmul`, which (for every k-quant type, not just this
    // PR's path) requantizes the f32 activations to the weight's `VecDotType` block format before
    // the dot product, same as llama.cpp's CPU kernels. That activation requantization, not
    // anything this PR changed, is what the two sides disagree on, and it's largest in relative
    // terms exactly where dot products cancel toward zero. LOOSE_* bounds that for our (EXPERTS,
    // N_OUT, K) test sizes with margin (see `measure_max_errors_per_dtype`, run with
    // `--ignored --nocapture`); TIGHT_* is for comparisons that stay on one side of that boundary
    // (cached vs uncached, pre- vs post-eviction), which should be reduction-order noise only.
    const LOOSE_ATOL: f32 = 1.0;
    const LOOSE_RTOL: f32 = 0.2;
    const TIGHT_ATOL: f32 = 1e-4;
    const TIGHT_RTOL: f32 = 1e-4;

    fn reference(qtensor: &QTensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let weights = qtensor.dequantize(x.device())?;
        let unquant =
            UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;
        unquant.gather_forward(x, ids)
    }

    /// Elementwise `abs(actual - expected) <= atol + rtol * abs(expected)`, reporting max
    /// abs/rel error on failure instead of just a single aggregate diff.
    fn assert_allclose(actual: &Tensor, expected: &Tensor, atol: f32, rtol: f32) -> Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let mut max_abs_err = 0f32;
        let mut max_rel_err = 0f32;
        let mut first_violation: Option<(usize, f32, f32)> = None;
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let abs_err = (a - e).abs();
            max_abs_err = max_abs_err.max(abs_err);
            max_rel_err = max_rel_err.max(abs_err / e.abs().max(1e-12));
            if abs_err > atol + rtol * e.abs() && first_violation.is_none() {
                first_violation = Some((i, a, e));
            }
        }
        if let Some((idx, a, e)) = first_violation {
            panic!(
                "allclose violated at index {idx}: actual={a} expected={e} \
                 (max_abs_err={max_abs_err}, max_rel_err={max_rel_err}, atol={atol}, rtol={rtol})"
            );
        }
        Ok(())
    }

    /// Deterministic, reproducible stand-in for `Tensor::randn`: candle's CPU RNG can't be
    /// seeded (`Device::Cpu::set_seed` errors), and these tests must not be flaky.
    fn deterministic_values(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed ^ 0x9E3779B97F4A7C15;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
            })
            .collect()
    }

    fn det_tensor(
        dims: impl Into<candle_core::Shape>,
        seed: u64,
        device: &Device,
    ) -> Result<Tensor> {
        let dims = dims.into();
        Tensor::from_vec(deterministic_values(dims.elem_count(), seed), dims, device)
    }

    fn random_qtensor(
        dtype: GgmlDType,
        n_experts: usize,
        n_out: usize,
        k: usize,
        device: &Device,
    ) -> Result<QTensor> {
        let seed = dtype.type_size() as u64 * 1_000_003 + (n_experts * n_out * k) as u64;
        let weight = det_tensor((n_experts, n_out, k), seed, device)?;
        QTensor::quantize(&weight, dtype)
    }

    fn run(x_t: usize, dtype: GgmlDType) -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(dtype, EXPERTS, N_OUT, K, &device)?;
        let x = det_tensor((3, x_t, K), 201, &device)?;
        let ids = Tensor::new(&[[0u32, 5], [5, 2], [2, 0]], &device)?;
        let expected = reference(&qtensor, &x, &ids)?;

        let cache = ExpertMatMuls::default();
        let mut cached_outputs = Vec::new();
        for _ in 0..2 {
            let actual = selected_experts_forward(&qtensor, Some(&cache), &x, &ids)?
                .expect("layout is supported");
            assert_allclose(&actual, &expected, LOOSE_ATOL, LOOSE_RTOL)?;
            cached_outputs.push(actual);
        }
        // Cold and warm cache must return identical results, not just both-close-to-reference.
        assert_allclose(
            &cached_outputs[0],
            &cached_outputs[1],
            TIGHT_ATOL,
            TIGHT_RTOL,
        )?;
        let uncached =
            selected_experts_forward(&qtensor, None, &x, &ids)?.expect("layout is supported");
        assert_allclose(&uncached, &expected, LOOSE_ATOL, LOOSE_RTOL)?;
        assert_allclose(&uncached, &cached_outputs[0], TIGHT_ATOL, TIGHT_RTOL)
    }

    #[test]
    fn selected_experts_match_full_dequantize_shared_input() -> Result<()> {
        run(1, GgmlDType::Q4K)
    }

    #[test]
    fn selected_experts_match_full_dequantize_per_slot_input() -> Result<()> {
        run(2, GgmlDType::Q4K)
    }

    #[test]
    fn selected_experts_match_full_dequantize_q6k() -> Result<()> {
        run(1, GgmlDType::Q6K)
    }

    #[test]
    fn selected_experts_match_full_dequantize_q8_0() -> Result<()> {
        run(1, GgmlDType::Q8_0)
    }

    #[test]
    fn selected_experts_reject_out_of_range_ids() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let x = det_tensor((1, 1, K), 209, &device)?;
        let ids = Tensor::new(&[[EXPERTS as u32, 0]], &device)?;
        assert!(selected_experts_forward(&qtensor, None, &x, &ids).is_err());
        Ok(())
    }

    /// Not a correctness assertion; run with `cargo test -p mistralrs-quant --lib \
    /// measure_max_errors_per_dtype -- --nocapture --ignored` to recheck the ATOL/RTOL margins
    /// above against fresh random weights.
    #[test]
    #[ignore]
    fn measure_max_errors_per_dtype() -> Result<()> {
        let device = Device::Cpu;
        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let mut max_abs = 0f32;
            let mut max_rel = 0f32;
            for seed in 0..20u64 {
                let weight = Tensor::randn(0f32, 1f32, (EXPERTS, N_OUT, K), &device)?
                    .affine((seed as f64 + 1.0) * 0.37, 0.0)?;
                let qtensor = QTensor::quantize(&weight, dtype)?;
                let x = Tensor::randn(0f32, 1f32, (3, 2, K), &device)?;
                let ids = Tensor::new(&[[0u32, 5], [5, 2], [2, 0]], &device)?;
                let expected = reference(&qtensor, &x, &ids)?;
                let actual = selected_experts_forward(&qtensor, None, &x, &ids)?.unwrap();
                let a = actual.flatten_all()?.to_vec1::<f32>()?;
                let e = expected.flatten_all()?.to_vec1::<f32>()?;
                for (a, e) in a.iter().zip(e.iter()) {
                    let abs_err = (a - e).abs();
                    let rel_err = abs_err / e.abs().max(1e-12);
                    max_abs = max_abs.max(abs_err);
                    max_rel = max_rel.max(rel_err);
                }
            }
            println!("{dtype:?}: max_abs_err={max_abs} max_rel_err={max_rel}");
        }
        Ok(())
    }

    fn allclose(actual: &Tensor, expected: &Tensor, atol: f32, rtol: f32) -> Result<bool> {
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        Ok(actual
            .iter()
            .zip(expected.iter())
            .all(|(&a, &e)| (a - e).abs() <= atol + rtol * e.abs()))
    }

    #[test]
    fn selected_experts_detect_wrong_accumulation() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let x = det_tensor((3, 1, K), 202, &device)?;
        let ids = Tensor::new(&[[0u32, 5], [5, 2], [2, 0]], &device)?;
        let expected = reference(&qtensor, &x, &ids)?;

        // Deliberately wrong routing: swap the two expert columns for every token. Compare the
        // optimized (cached) path against the correct reference using the same LOOSE bound real
        // callers get; it must still catch this, not just the within-quantized-domain TIGHT one.
        let wrong_ids = Tensor::new(&[[5u32, 0], [2, 5], [0, 2]], &device)?;
        let cache = ExpertMatMuls::default();
        let wrong = selected_experts_forward(&qtensor, Some(&cache), &x, &wrong_ids)?
            .expect("layout is supported");

        assert!(
            !allclose(&wrong, &expected, LOOSE_ATOL, LOOSE_RTOL)?,
            "LOOSE allclose failed to distinguish deliberately wrong expert routing"
        );
        Ok(())
    }

    #[test]
    fn repeated_and_reordered_expert_selection_match_reference() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let x = det_tensor((4, 2, K), 203, &device)?;
        // Repeated indices within a token, and different relative orderings across tokens.
        let ids = Tensor::new(&[[3u32, 3], [1, 4], [4, 1], [0, 0]], &device)?;
        let expected = reference(&qtensor, &x, &ids)?;
        let cache = ExpertMatMuls::default();
        let actual = selected_experts_forward(&qtensor, Some(&cache), &x, &ids)?
            .expect("layout is supported");
        assert_allclose(&actual, &expected, LOOSE_ATOL, LOOSE_RTOL)
    }

    #[test]
    fn near_zero_and_large_magnitude_outputs_match_reference() -> Result<()> {
        let device = Device::Cpu;
        // Scale weights up/down to exercise both large-magnitude and near-zero outputs.
        let weight = det_tensor((EXPERTS, N_OUT, K), 204, &device)?;
        let weight = Tensor::cat(
            &[
                (weight.narrow(0, 0, 3)? * 1000.0)?,
                (weight.narrow(0, 3, 3)? * 1e-4)?,
            ],
            0,
        )?;
        let qtensor = QTensor::quantize(&weight, GgmlDType::Q4K)?;
        let x = det_tensor((2, 2, K), 205, &device)?;
        let ids = Tensor::new(&[[0u32, 3], [4, 1]], &device)?;
        let expected = reference(&qtensor, &x, &ids)?;
        let cache = ExpertMatMuls::default();
        let actual = selected_experts_forward(&qtensor, Some(&cache), &x, &ids)?
            .expect("layout is supported");
        // The activation-requantization noise LOOSE_* bounds scales with weight magnitude; this
        // test deliberately uses a 1000x/1e-4x dynamic range, so it needs its own wider margin.
        assert_allclose(&actual, &expected, 10.0, 0.3)
    }

    // -----------------------------------------------------------------------
    // Cache memory budget
    // -----------------------------------------------------------------------

    fn build_counting(counter: Arc<AtomicUsize>, qtensor: &QTensor, expert: usize) -> QMatMul {
        counter.fetch_add(1, Ordering::SeqCst);
        let dtype = qtensor.dtype();
        let expert_bytes = N_OUT * K / dtype.block_size() * dtype.type_size();
        let data = qtensor.data().unwrap();
        let bytes = &data[expert * expert_bytes..(expert + 1) * expert_bytes];
        QMatMul::from_qtensor(
            qtensor_from_ggml(dtype, bytes, vec![N_OUT, K], &Device::Cpu).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn cache_stays_within_configured_budget() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let one_expert_bytes = {
            let m = build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 0);
            qmatmul_retained_bytes(&m)
        };
        let cache = ExpertMatMuls::default();
        cache.set_budget_for_test(one_expert_bytes * 2 + one_expert_bytes / 2);

        for expert in 0..EXPERTS {
            let counter = Arc::new(AtomicUsize::new(0));
            cache.get_or_build(EXPERTS, expert, || {
                Ok(build_counting(counter.clone(), &qtensor, expert))
            })?;
            assert!(
                cache.cached_bytes() <= one_expert_bytes * 2 + one_expert_bytes / 2,
                "cache exceeded its configured budget after expert {expert}"
            );
        }
        // Only the two most recently built experts should still be resident.
        assert!(!cache.is_ready(0));
        assert!(cache.is_ready(EXPERTS - 1));
        assert!(cache.is_ready(EXPERTS - 2));
        Ok(())
    }

    #[test]
    fn repeated_selection_hits_cache_without_rebuilding() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let cache = ExpertMatMuls::default();
        let counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..5 {
            cache.get_or_build(EXPERTS, 2, || {
                Ok(build_counting(counter.clone(), &qtensor, 2))
            })?;
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn new_expert_evicts_least_recently_used() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let one_expert_bytes =
            qmatmul_retained_bytes(&build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 0));
        let cache = ExpertMatMuls::default();
        cache.set_budget_for_test(one_expert_bytes + one_expert_bytes / 2);

        cache.get_or_build(EXPERTS, 0, || {
            Ok(build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 0))
        })?;
        assert!(cache.is_ready(0));
        cache.get_or_build(EXPERTS, 1, || {
            Ok(build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 1))
        })?;
        assert!(
            !cache.is_ready(0),
            "expert 0 should be evicted for expert 1"
        );
        assert!(cache.is_ready(1));
        Ok(())
    }

    #[test]
    fn evicted_expert_reconstructs_to_the_same_result() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let x = det_tensor((1, 1, K), 206, &device)?;
        let ids = Tensor::new(&[[0u32, 1]], &device)?;
        let expected = reference(&qtensor, &x, &ids)?;

        let one_expert_bytes =
            qmatmul_retained_bytes(&build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 0));
        let cache = ExpertMatMuls::default();
        cache.set_budget_for_test(one_expert_bytes + one_expert_bytes / 2);

        let first = selected_experts_forward(&qtensor, Some(&cache), &x, &ids)?.unwrap();
        assert_allclose(&first, &expected, LOOSE_ATOL, LOOSE_RTOL)?;
        // Force eviction of expert 0 by touching every other expert.
        for expert in 2..EXPERTS {
            cache.get_or_build(EXPERTS, expert, || {
                Ok(build_counting(
                    Arc::new(AtomicUsize::new(0)),
                    &qtensor,
                    expert,
                ))
            })?;
        }
        assert!(!cache.is_ready(0));
        let second = selected_experts_forward(&qtensor, Some(&cache), &x, &ids)?.unwrap();
        assert_allclose(&second, &first, TIGHT_ATOL, TIGHT_RTOL)?;
        assert_allclose(&second, &expected, LOOSE_ATOL, LOOSE_RTOL)
    }

    #[test]
    fn budget_smaller_than_one_expert_is_safe() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let x = det_tensor((1, 1, K), 207, &device)?;
        let ids = Tensor::new(&[[0u32, 1]], &device)?;
        let expected = reference(&qtensor, &x, &ids)?;

        let cache = ExpertMatMuls::default();
        cache.set_budget_for_test(1);
        let actual = selected_experts_forward(&qtensor, Some(&cache), &x, &ids)?.unwrap();
        assert_allclose(&actual, &expected, LOOSE_ATOL, LOOSE_RTOL)?;
        assert_eq!(cache.cached_bytes(), 0);
        assert!(!cache.is_ready(0) && !cache.is_ready(1));
        Ok(())
    }

    #[test]
    fn zero_budget_never_retains_anything() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let cache = ExpertMatMuls::default();
        cache.set_budget_for_test(0);
        for expert in 0..EXPERTS {
            cache.get_or_build(EXPERTS, expert, || {
                Ok(build_counting(
                    Arc::new(AtomicUsize::new(0)),
                    &qtensor,
                    expert,
                ))
            })?;
        }
        assert_eq!(cache.cached_bytes(), 0);
        for expert in 0..EXPERTS {
            assert!(!cache.is_ready(expert));
        }
        Ok(())
    }

    #[test]
    fn independent_model_instances_have_independent_caches() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let cache_a = ExpertMatMuls::default();
        let cache_b = ExpertMatMuls::default();
        cache_a.get_or_build(EXPERTS, 0, || {
            Ok(build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 0))
        })?;
        assert!(cache_a.is_ready(0));
        assert!(!cache_b.is_ready(0));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Concurrency
    // -----------------------------------------------------------------------

    #[test]
    fn concurrent_same_expert_builds_exactly_once() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = Arc::new(random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?);
        let cache = Arc::new(ExpertMatMuls::default());
        let counter = Arc::new(AtomicUsize::new(0));
        let n_threads = 8;
        let barrier = Arc::new(Barrier::new(n_threads));

        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let cache = cache.clone();
                let counter = counter.clone();
                let qtensor = qtensor.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    cache
                        .get_or_build(EXPERTS, 3, || {
                            Ok(build_counting(counter.clone(), &qtensor, 3))
                        })
                        .unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "expert should be built exactly once under concurrent access"
        );
        Ok(())
    }

    #[test]
    fn concurrent_different_experts_build_independently() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = Arc::new(random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?);
        let cache = Arc::new(ExpertMatMuls::default());
        let n_threads = 4;
        // Every build() blocks on this barrier until all n_threads have entered build(),
        // which can only happen if the cache lets distinct experts build concurrently
        // (a global lock held across build() would deadlock this).
        let entered = Arc::new(Barrier::new(n_threads));

        let handles: Vec<_> = (0..n_threads)
            .map(|expert| {
                let cache = cache.clone();
                let qtensor = qtensor.clone();
                let entered = entered.clone();
                thread::spawn(move || {
                    cache
                        .get_or_build(EXPERTS, expert, || {
                            entered.wait();
                            Ok(build_counting(
                                Arc::new(AtomicUsize::new(0)),
                                &qtensor,
                                expert,
                            ))
                        })
                        .unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        for expert in 0..n_threads {
            assert!(cache.is_ready(expert));
        }
        Ok(())
    }

    #[test]
    fn concurrent_reads_of_a_cached_expert_all_succeed() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = Arc::new(random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?);
        let cache = Arc::new(ExpertMatMuls::default());
        cache.get_or_build(EXPERTS, 0, || {
            Ok(build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 0))
        })?;

        let n_threads = 8;
        let barrier = Arc::new(Barrier::new(n_threads));
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let cache = cache.clone();
                let qtensor = qtensor.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    cache
                        .get_or_build(EXPERTS, 0, || {
                            panic!("already cached, should not rebuild");
                            #[allow(unreachable_code)]
                            Ok(build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 0))
                        })
                        .unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        Ok(())
    }

    #[test]
    fn eviction_does_not_disturb_an_in_flight_reference() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let x = det_tensor((1, 1, K), 208, &device)?;

        let one_expert_bytes =
            qmatmul_retained_bytes(&build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 0));
        let cache = ExpertMatMuls::default();
        cache.set_budget_for_test(one_expert_bytes + one_expert_bytes / 2);

        let held = cache.get_or_build(EXPERTS, 0, || {
            Ok(build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 0))
        })?;
        let x_rows = x.reshape((1, K))?;
        let expected_row = Module::forward(&held, &x_rows)?;

        // Evict expert 0 from the cache while `held` keeps it alive.
        cache.get_or_build(EXPERTS, 1, || {
            Ok(build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 1))
        })?;
        assert!(!cache.is_ready(0));

        // The already-obtained handle must still compute the same answer as before eviction.
        let out = Module::forward(&held, &x_rows)?;
        assert_allclose(&out, &expected_row, TIGHT_ATOL, TIGHT_RTOL)
    }

    #[test]
    fn failed_build_is_retryable() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?;
        let cache = ExpertMatMuls::default();

        let err = cache.get_or_build(EXPERTS, 0, || {
            candle_core::bail!("deliberate build failure")
        });
        assert!(err.is_err());
        assert!(!cache.is_ready(0));

        let ok = cache.get_or_build(EXPERTS, 0, || {
            Ok(build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 0))
        });
        assert!(ok.is_ok());
        assert!(cache.is_ready(0));
        Ok(())
    }

    #[test]
    fn concurrent_requests_with_a_small_budget_do_not_corrupt_state() -> Result<()> {
        let device = Device::Cpu;
        let qtensor = Arc::new(random_qtensor(GgmlDType::Q4K, EXPERTS, N_OUT, K, &device)?);
        let one_expert_bytes =
            qmatmul_retained_bytes(&build_counting(Arc::new(AtomicUsize::new(0)), &qtensor, 0));
        let cache = Arc::new(ExpertMatMuls::default());
        cache.set_budget_for_test(one_expert_bytes + one_expert_bytes / 2);

        let n_threads = 16;
        let barrier = Arc::new(Barrier::new(n_threads));
        let handles: Vec<_> = (0..n_threads)
            .map(|i| {
                let cache = cache.clone();
                let qtensor = qtensor.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let expert = i % EXPERTS;
                    barrier.wait();
                    cache
                        .get_or_build(EXPERTS, expert, || {
                            Ok(build_counting(
                                Arc::new(AtomicUsize::new(0)),
                                &qtensor,
                                expert,
                            ))
                        })
                        .unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(
            cache.cached_bytes() <= one_expert_bytes * 2 + one_expert_bytes,
            "cache exceeded its budget after concurrent load: {} bytes",
            cache.cached_bytes()
        );
        Ok(())
    }
}
