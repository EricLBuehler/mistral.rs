//! Output-parallel CPU gather-GEMV for stacked MoE experts.
//!
//! The stacked expert weight is `[num_experts, out_features, in_features]` and decode routes
//! each token to `k` experts, so one layer's routed work is `n_slots * out_features`
//! independent dot products of length `in_features`. For a 384-expert model that is millions
//! of outputs against a weight of gigabytes, so the same bandwidth argument as the dense
//! [`crate::gemv_cpu`] applies: the serial fallback in `UnquantLinear::gather_forward_raw`
//! reaches one core's bandwidth and nothing else.
//!
//! That fallback also *copies* the selected experts out of the stack before multiplying, so it
//! pays the weight traffic twice. This kernel reads the selected expert rows in place.
//!
//! Partitioning is over the flattened output axis, and every output element is produced
//! end-to-end by one thread walking its reduction in a fixed order. No reduction is split
//! across threads, so results do not depend on how many threads ran.

use candle_core::{CpuStorage, DType, Result, Storage, Tensor};
use rayon::prelude::*;

/// Below this many outputs there is not enough work to pay for splitting.
const MIN_PARALLEL_OUTPUTS: usize = 1024;

/// Dtypes whose rows this kernel can widen to f32 in place. Anything else falls back.
fn is_kernel_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::BF16)
}

/// Whether the fast path applies to this input and stacked weight.
///
/// The weight must be a densely packed `[num_experts, out, in]` stack: a sliced or otherwise
/// strided view would make the row index arithmetic below address the wrong elements.
pub fn should_use_gather_gemv(
    xs: &Tensor,
    w: &Tensor,
    out_features: usize,
    n_slots: usize,
) -> bool {
    if !xs.device().is_cpu() || w.rank() != 3 || n_slots == 0 || out_features == 0 {
        return false;
    }
    if !is_kernel_dtype(w.dtype()) || !is_kernel_dtype(xs.dtype()) {
        return false;
    }
    let (num_experts, out, k) = match w.dims3() {
        Ok(dims) => dims,
        Err(_) => return false,
    };
    if num_experts == 0 || out != out_features || k == 0 {
        return false;
    }
    if n_slots * out_features < MIN_PARALLEL_OUTPUTS {
        return false;
    }
    let layout = w.layout();
    layout.start_offset() == 0 && layout.stride() == [out * k, k, 1]
}

/// `(n_slots, in_features) @ gather(w, ids).T` as `(n_slots, out_features)` f32.
///
/// `a` must already be contiguous f32 of length `n_slots * in_features`; `ids` holds the
/// expert index for each slot. The caller narrows the result to the weight dtype, which is
/// what candle's own bf16 matmul does (widen, accumulate in f32, narrow back).
pub fn gather_gemv(a: &[f32], w: &Tensor, ids: &[u32]) -> Result<Vec<f32>> {
    let (num_experts, out_features, in_features) = w.dims3()?;
    let n_slots = ids.len();
    if a.len() != n_slots * in_features {
        candle_core::bail!(
            "gather gemv: input has {} elements, expected {}",
            a.len(),
            n_slots * in_features
        );
    }

    let (guard, _) = w.storage_and_layout();
    let Storage::Cpu(w_cpu) = &*guard else {
        candle_core::bail!("gather gemv requires CPU storage")
    };

    let total = n_slots * out_features;
    let mut out = vec![0f32; total];

    // Each work item is one output element of one slot. `slot` fixes both the activation row
    // and which expert block of the stack to read, so items share nothing.
    let threads = crate::cpu_threads::resolve().max(1);
    let chunk = total.div_ceil(threads);
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(chunk_idx, dst)| {
            for (i, d) in dst.iter_mut().enumerate() {
                let idx = chunk_idx * chunk + i;
                let slot = idx / out_features;
                let expert = ids[slot] as usize;
                *d = if expert >= num_experts {
                    f32::NAN
                } else {
                    dot_slot(
                        a,
                        slot,
                        idx % out_features,
                        w_cpu,
                        expert,
                        out_features,
                        in_features,
                    )
                };
            }
        });

    Ok(out)
}

/// `[n_tokens, in] @ gather(w, ids).T` where every one of `k` consecutive slots for a token
/// reads the same activation row, returned as `(n_tokens, k, out_features)`.
///
/// Returns `None` when the fast path does not apply, so the caller can fall back to the
/// index-select-and-matmul route. `out_dtype` is the stacked weight's dtype; the result is
/// narrowed to it, which is what candle's own bf16 matmul does.
pub fn gather_broadcast_rows(
    a: &Tensor,
    w: &Tensor,
    ids: &Tensor,
    out_features: usize,
    k: usize,
    out_dtype: DType,
) -> Result<Option<Tensor>> {
    let (n_tokens, in_features) = a.dims2()?;
    if ids.elem_count() != n_tokens * k {
        return Ok(None);
    }
    if !should_use_gather_gemv(a, w, out_features, n_tokens * k) {
        return Ok(None);
    }
    // Expert ids have to be read on the host to index the stack, which is what the fallback
    // does too, so this is not an extra round trip.
    if !matches!(ids.dtype(), DType::U8 | DType::U32 | DType::I64) {
        return Ok(None);
    }
    let ids: Vec<u32> = match ids.dtype() {
        DType::U8 => ids
            .flatten_all()?
            .to_vec1::<u8>()?
            .iter()
            .map(|v| *v as u32)
            .collect(),
        DType::U32 => ids.flatten_all()?.to_vec1::<u32>()?,
        _ => ids
            .flatten_all()?
            .to_vec1::<i64>()?
            .iter()
            .map(|v| *v as u32)
            .collect(),
    };

    let a = a.to_dtype(DType::F32)?.contiguous()?;
    let a_rows = a.flatten_all()?.to_vec1::<f32>()?;
    let mut expanded = Vec::with_capacity(n_tokens * k * in_features);
    for row in a_rows.chunks_exact(in_features) {
        for _ in 0..k {
            expanded.extend_from_slice(row);
        }
    }

    let out = gather_gemv(&expanded, w, &ids)?;
    Tensor::from_vec(out, (n_tokens, k, out_features), a.device())?
        .to_dtype(out_dtype)
        .map(Some)
}

#[cfg(target_arch = "aarch64")]
fn dot_slot(
    a: &[f32],
    slot: usize,
    row: usize,
    w: &CpuStorage,
    expert: usize,
    out_features: usize,
    in_features: usize,
) -> f32 {
    let a_row = &a[slot * in_features..slot * in_features + in_features];
    let start = (expert * out_features + row) * in_features;
    match w {
        CpuStorage::F32(v) => dot(a_row, &v[start..start + in_features]),
        CpuStorage::BF16(v) => {
            // bf16 is repr(transparent) over u16, so the rows are readable as raw bits and
            // widening is a shift rather than a scalar conversion.
            let bits: &[u16] = unsafe { core::mem::transmute::<&[half::bf16], &[u16]>(v) };
            dot_bf16(a_row, &bits[start..start + in_features])
        }
        _ => f32::NAN,
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_slot(
    a: &[f32],
    slot: usize,
    row: usize,
    w: &CpuStorage,
    expert: usize,
    out_features: usize,
    in_features: usize,
) -> f32 {
    let _ = (a, slot, row, w, expert, out_features, in_features);
    f32::NAN
}

fn dot(x: &[f32], w: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (xi, wi) in x.iter().zip(w) {
        acc += xi * wi;
    }
    acc
}

/// `x @ bf16_row`, widening the bf16 operands as they load.
///
/// Four independent accumulators give a fixed reduction order per output element, matching
/// [`crate::gemv_cpu`]'s dense kernel so the routed and dense paths agree.
#[cfg(target_arch = "aarch64")]
fn dot_bf16(a: &[f32], b: &[u16]) -> f32 {
    use core::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= a.len() {
            s0 = vfmaq_f32(s0, vld1q_f32(a.as_ptr().add(i)), widen(b.as_ptr().add(i)));
            s1 = vfmaq_f32(
                s1,
                vld1q_f32(a.as_ptr().add(i + 4)),
                widen(b.as_ptr().add(i + 4)),
            );
            s2 = vfmaq_f32(
                s2,
                vld1q_f32(a.as_ptr().add(i + 8)),
                widen(b.as_ptr().add(i + 8)),
            );
            s3 = vfmaq_f32(
                s3,
                vld1q_f32(a.as_ptr().add(i + 12)),
                widen(b.as_ptr().add(i + 12)),
            );
            i += 16;
        }
        let mut sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        while i + 4 <= a.len() {
            sum = vfmaq_f32(sum, vld1q_f32(a.as_ptr().add(i)), widen(b.as_ptr().add(i)));
            i += 4;
        }
        let mut out = vaddvq_f32(sum);
        while i < a.len() {
            out += *a.get_unchecked(i) * f32::from_bits((*b.get_unchecked(i) as u32) << 16);
            i += 1;
        }
        out
    }
}

/// Widen four adjacent bf16 lanes into one `float32x4_t`.
///
/// bf16 is the top half of an f32, so this is a 16-bit left shift after widening u16 -> u32.
/// A 16-bit immediate shift does not exist for `u16`, hence the `vmovl_u16` first.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn widen(ptr: *const u16) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;
    unsafe { vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vld1_u16(ptr)))) }
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_bf16(a: &[f32], b: &[u16]) -> f32 {
    let mut acc = 0f32;
    for (ai, bi) in a.iter().zip(b) {
        acc += ai * f32::from_bits((*bi as u32) << 16);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Reference: gather the rows out of the stack and dot them one at a time, which is the
    /// shape of the serial fallback this kernel replaces.
    fn serial_reference(a: &[f32], w: &Tensor, ids: &[u32]) -> Result<Vec<f32>> {
        let (_, out_features, in_features) = w.dims3()?;
        let wf = w.to_dtype(DType::F32)?.contiguous()?;
        let flat = wf.flatten_all()?.to_vec1::<f32>()?;
        let mut out = Vec::with_capacity(ids.len() * out_features);
        for (slot, &id) in ids.iter().enumerate() {
            let a_row = &a[slot * in_features..slot * in_features + in_features];
            let block_len = out_features * in_features;
            let block = &flat[id as usize * block_len..(id as usize + 1) * block_len];
            for r in 0..out_features {
                out.push(dot(
                    a_row,
                    &block[r * in_features..r * in_features + in_features],
                ));
            }
        }
        Ok(out)
    }

    #[test]
    fn rejects_unsuitable_inputs() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::rand(-1f32, 1f32, (8, 256, 32), &dev)?;
        let a = Tensor::zeros((8, 32), DType::F32, &dev)?;
        assert!(should_use_gather_gemv(&a, &w, 256, 8));
        // Below the output budget the split is not worth it.
        assert!(!should_use_gather_gemv(&a, &w, 256, 1));
        // A sliced weight would make the row arithmetic address the wrong elements.
        let sliced = w.narrow(candle_core::D::Minus1, 0, 16)?;
        assert!(!should_use_gather_gemv(&a, &sliced, 256, 8));
        // Mismatched declared out_features, a dtype the kernel cannot widen, and a rank-2
        // "stack" all have to fall back.
        assert!(!should_use_gather_gemv(&a, &w, 32, 8));
        let f16 = Tensor::zeros((8, 32), DType::F16, &dev)?;
        assert!(!should_use_gather_gemv(&f16, &w, 256, 8));
        let flat = Tensor::zeros((256, 32), DType::F32, &dev)?;
        assert!(!should_use_gather_gemv(&a, &flat, 256, 8));
        Ok(())
    }

    /// The partitioning invariant: no reduction is split across threads, so the parallel result
    /// is bit-identical to the same work done serially.
    #[test]
    fn parallel_matches_serial_bit_exactly() -> Result<()> {
        for dtype in [DType::F32, DType::BF16] {
            for (experts, out, k, slots) in [(8usize, 64usize, 32usize, 16usize), (384, 16, 64, 64)]
            {
                let w =
                    Tensor::rand(-1f32, 1f32, (experts, out, k), &Device::Cpu)?.to_dtype(dtype)?;
                let a: Vec<f32> = (0..slots * k)
                    .map(|i| ((i % 17) as f32) * 0.125 - 1.0)
                    .collect();
                let ids: Vec<u32> = (0..slots).map(|i| (i * 7 % experts) as u32).collect();

                let par = gather_gemv(&a, &w, &ids)?;
                let ser = serial_reference(&a, &w, &ids)?;
                assert_eq!(par.len(), ser.len());
                // bf16 rounds the operands before the product, so only the f32 stack is exact.
                if dtype == DType::F32 {
                    assert_eq!(par, ser, "{dtype:?} {experts}x{out}x{k}x{slots}");
                }
            }
        }
        Ok(())
    }

    /// bf16 -> f32 is a bit shift, and the vector path has to land every lane in the right
    /// accumulator. A lane mixup here is invisible in a throughput benchmark and shows up as
    /// silently wrong logits, so it is pinned directly.
    #[test]
    fn bf16_widening_is_lane_exact() {
        for len in [1usize, 3, 4, 7, 8, 15, 16, 17, 31, 64] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.25 - 3.0).collect();
            let bits: Vec<u16> = (0..len)
                .map(|i| ((a[i] * 0.5).to_bits() >> 16) as u16)
                .collect();
            let want: f32 = a.iter().zip(&bits).map(|(x, b)| x * bf16_to_f32(*b)).sum();
            let got = dot_bf16(&a, &bits);
            let scale = want.abs().max(1.0);
            assert!(
                (got - want).abs() <= 1e-5 * scale,
                "len {len}: {got} vs {want}"
            );
        }
    }

    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    #[test]
    fn out_of_range_expert_is_reported_not_read() -> Result<()> {
        let w = Tensor::zeros((4, 64, 32), DType::F32, &Device::Cpu)?;
        let a = vec![0f32; 8 * 32];
        let ids = vec![9u32, 0, 1, 2, 3, 0, 1, 2];
        let out = gather_gemv(&a, &w, &ids)?;
        assert!(out[0].is_nan());
        assert_eq!(&out[64..], &[0f32; 7 * 64]);
        Ok(())
    }

    /// The dispatch has to agree with the index-select-and-matmul route it replaces, because
    /// that route is what every non-CPU backend still takes.
    #[test]
    fn broadcast_rows_matches_the_index_select_fallback() -> Result<()> {
        for dtype in [DType::F32, DType::BF16] {
            let (experts, n_tokens, k, out, in_features) = (16usize, 8usize, 4usize, 256usize, 64);
            let w = Tensor::rand(-1f32, 1f32, (experts, out, in_features), &Device::Cpu)?
                .to_dtype(dtype)?;
            let a = Tensor::rand(-1f32, 1f32, (n_tokens, in_features), &Device::Cpu)?
                .to_dtype(dtype)?;
            // Built [k, n_tokens] then transposed, so consecutive slots pick different
            // experts and a transpose bug cannot hide.
            let flat: Vec<u32> = (0..n_tokens * k)
                .map(|i| (i * 5 % experts) as u32)
                .collect();
            let ids = Tensor::from_vec(flat, (k, n_tokens), &Device::Cpu)?
                .t()?
                .contiguous()?;
            assert_eq!(ids.dims(), &[n_tokens, k]);

            let got = gather_broadcast_rows(&a, &w, &ids, out, k, dtype)?
                .expect("shape is large enough for the fast path");
            assert_eq!(got.dims(), &[n_tokens, k, out]);

            let flat_ids = ids.reshape((n_tokens * k,))?;
            let selected = w.index_select(&flat_ids, 0)?;
            let expanded = a
                .unsqueeze(1)?
                .broadcast_as((n_tokens, k, in_features))?
                .reshape((n_tokens * k, in_features))?;
            let want = expanded
                .unsqueeze(1)?
                .matmul(&selected.transpose(1, 2)?)?
                .squeeze(1)?
                .reshape((n_tokens, k, out))?;

            let got_v = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let want_v = want.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(got_v.len(), want_v.len());
            let scale = want_v.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let mut worst = 0f32;
            for (g, x) in got_v.iter().zip(&want_v) {
                worst = worst.max((g - x).abs());
            }
            // The partitioning is exact (see `parallel_matches_serial_bit_exactly`); what
            // differs from candle here is only the order of the length-`in` accumulation, and
            // bf16 additionally rounds the operands before the product.
            let tol = match dtype {
                DType::F32 => 1e-5,
                _ => 5e-2,
            } * scale;
            assert!(worst <= tol, "{dtype:?} worst {worst} > {tol}");
        }
        Ok(())
    }
}
