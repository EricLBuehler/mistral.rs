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

use crate::dot_cpu::{dot_bf16, dot_f32 as dot};

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
    let (guard, _) = w.storage_and_layout();
    let Storage::Cpu(w_cpu) = &*guard else {
        candle_core::bail!("gather gemv requires CPU storage")
    };
    check_input(a.len(), ids.len(), in_features)?;
    par_fill(
        ids.len(),
        out_features,
        ids,
        num_experts,
        |slot, row, expert| dot_slot(a, slot, row, w_cpu, expert, out_features, in_features),
    )
}

/// The same gather against one view per expert instead of a fused stack, so a per-expert
/// checkpoint never has to be stacked to be used.
///
/// Each slab is a contiguous `[out_features, in_features]` tensor. For a memory-mapped
/// safetensors checkpoint these are zero-copy views of the mapping, so the whole model stays
/// in reclaimable page cache instead of being copied into anonymous memory.
pub fn gather_gemv_per_expert(
    a: &[f32],
    experts: &[Tensor],
    out_features: usize,
    in_features: usize,
    ids: &[u32],
) -> Result<Vec<f32>> {
    if experts.is_empty() {
        candle_core::bail!("gather gemv: no expert slabs")
    }
    check_input(a.len(), ids.len(), in_features)?;

    // dtype and residency are properties of how the checkpoint was loaded, not of the
    // routing, so they are checked once here rather than per work item.
    let dtype = experts[0].dtype();
    if experts.iter().any(|t| t.dtype() != dtype) {
        candle_core::bail!("gather gemv: expert slabs disagree on dtype")
    }
    if experts.iter().any(|t| !t.device().is_cpu()) {
        candle_core::bail!("gather gemv: expert slabs must be on CPU")
    }

    match dtype {
        DType::F32 => with_f32_rows(experts, |rows| {
            par_fill(
                ids.len(),
                out_features,
                ids,
                experts.len(),
                |slot, row, expert| {
                    dot(
                        a_row(a, slot, in_features),
                        &rows[expert][row * in_features..][..in_features],
                    )
                },
            )
        }),
        DType::BF16 => with_bf16_rows(experts, |rows| {
            par_fill(
                ids.len(),
                out_features,
                ids,
                experts.len(),
                |slot, row, expert| {
                    dot_bf16(
                        a_row(a, slot, in_features),
                        &rows[expert][row * in_features..][..in_features],
                    )
                },
            )
        }),
        other => candle_core::bail!("gather gemv: unsupported expert dtype {other:?}"),
    }
}

/// Hands each slab's rows to `f` as one contiguous slice.
///
/// The storage guards are taken here rather than in the caller because the row slices borrow
/// from the storage they lock; holding them in this frame is what keeps the slices alive for
/// the duration of the parallel loop without a self-referential struct.
fn with_f32_rows<T>(experts: &[Tensor], f: impl FnOnce(&[&[f32]]) -> Result<T>) -> Result<T> {
    let guards: Vec<_> = experts.iter().map(Tensor::storage_and_layout).collect();
    let rows = guards
        .iter()
        .zip(experts)
        .map(|((guard, layout), t)| {
            let Storage::Cpu(cpu) = &**guard else {
                candle_core::bail!("expert slab must be CPU storage")
            };
            let flat = cpu.as_slice::<f32>()?;
            let start = layout.start_offset();
            Ok(&flat[start..start + t.elem_count()])
        })
        .collect::<Result<Vec<_>>>()?;
    f(&rows)
}

fn with_bf16_rows<T>(experts: &[Tensor], f: impl FnOnce(&[&[u16]]) -> Result<T>) -> Result<T> {
    let guards: Vec<_> = experts.iter().map(Tensor::storage_and_layout).collect();
    let rows = guards
        .iter()
        .zip(experts)
        .map(|((guard, layout), t)| {
            let Storage::Cpu(cpu) = &**guard else {
                candle_core::bail!("expert slab must be CPU storage")
            };
            let flat = cpu.as_slice::<half::bf16>()?;
            // bf16 is repr(transparent) over u16, and the dot kernel reads raw bits.
            let flat: &[u16] = unsafe { core::mem::transmute::<&[half::bf16], &[u16]>(flat) };
            let start = layout.start_offset();
            Ok(&flat[start..start + t.elem_count()])
        })
        .collect::<Result<Vec<_>>>()?;
    f(&rows)
}

fn check_input(a_len: usize, n_slots: usize, in_features: usize) -> Result<()> {
    if a_len != n_slots * in_features {
        candle_core::bail!(
            "gather gemv: input has {} elements, expected {}",
            a_len,
            n_slots * in_features
        );
    }
    Ok(())
}

#[inline]
fn a_row(a: &[f32], slot: usize, in_features: usize) -> &[f32] {
    &a[slot * in_features..slot * in_features + in_features]
}

/// Partitions the `n_slots * out_features` outputs across the pool. `dot` receives the slot,
/// the output row, and an expert id already bounds-checked against `num_experts`.
fn par_fill(
    n_slots: usize,
    out_features: usize,
    ids: &[u32],
    num_experts: usize,
    dot: impl Fn(usize, usize, usize) -> f32 + Sync,
) -> Result<Vec<f32>> {
    let total = n_slots * out_features;
    let mut out = vec![0f32; total];
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
                    dot(slot, idx % out_features, expert)
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

    /// The per-expert form has to produce exactly what the fused stack produces, since it is
    /// a drop-in replacement for it rather than a second opinion on the math.
    #[test]
    fn per_expert_matches_the_fused_stack_bit_exactly() -> Result<()> {
        for dtype in [DType::F32, DType::BF16] {
            let (experts, out, k, slots) = (8usize, 64usize, 32usize, 16usize);
            let slabs: Vec<Tensor> = (0..experts)
                .map(|_| Tensor::rand(-1f32, 1f32, (out, k), &Device::Cpu)?.to_dtype(dtype))
                .collect::<Result<Vec<_>>>()?;
            // Same bytes either way, so only the layout the gather reads differs.
            let stack = Tensor::stack(&slabs, 0)?;
            let a: Vec<f32> = (0..slots * k)
                .map(|i| ((i % 17) as f32) * 0.125 - 1.0)
                .collect();
            let ids: Vec<u32> = (0..slots).map(|i| (i * 7 % experts) as u32).collect();

            let fused = gather_gemv(&a, &stack, &ids)?;
            let split = gather_gemv_per_expert(&a, &slabs, out, k, &ids)?;
            assert_eq!(fused.len(), split.len());
            assert_eq!(fused, split, "{dtype:?}");
        }
        Ok(())
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
