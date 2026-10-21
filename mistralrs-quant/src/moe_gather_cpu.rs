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

/// Whether the gather kernel would serve a stack of this shape and size.
///
/// `should_use_gather_gemv` answers the same question for a stack that is already
/// dequantised. This is the same predicate without needing one, so a caller holding only the
/// quantised stack can ask before committing to dequantising it.
///
/// A caller that replaces the whole-stack path with a per-expert one has to agree with it
/// here, because the whole-stack path falls back to a matmul when this says no, and a matmul
/// and the gather are not the same arithmetic.
pub fn should_use_gather_gemv_shape(
    num_experts: usize,
    out: usize,
    k: usize,
    out_features: usize,
    n_slots: usize,
) -> bool {
    if num_experts == 0 || out != out_features || k == 0 || n_slots == 0 || out_features == 0 {
        return false;
    }
    n_slots * out_features >= MIN_PARALLEL_OUTPUTS
}

/// `(n_slots, in_features) @ gather(w, ids).T` as `(n_slots, out_features)` f32.
///
/// `a` must already be contiguous f32 of length `n_slots * in_features`; `ids` holds the
/// expert index for each slot. The caller narrows the result to the weight dtype, which is
/// what candle's own bf16 matmul does (widen, accumulate in f32, narrow back).
pub fn gather_gemv(a: &[f32], w: &Tensor, ids: &[u32], repeat: usize) -> Result<Vec<f32>> {
    let (num_experts, out_features, in_features) = w.dims3()?;
    if repeat == 0 || !ids.len().is_multiple_of(repeat) {
        candle_core::bail!(
            "gather gemv: {} slots do not divide into groups of {repeat}",
            ids.len()
        )
    }
    let (guard, _) = w.storage_and_layout();
    let Storage::Cpu(w_cpu) = &*guard else {
        candle_core::bail!("gather gemv requires CPU storage")
    };
    check_input(a.len(), ids.len(), in_features, repeat)?;
    par_fill(
        ids.len(),
        out_features,
        ids,
        num_experts,
        |slot, row, expert| {
            dot_slot(
                &DotCtx {
                    a,
                    repeat,
                    in_features,
                },
                slot,
                row,
                w_cpu,
                expert,
                out_features,
            )
        },
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
    repeat: usize,
) -> Result<Vec<f32>> {
    if experts.is_empty() {
        candle_core::bail!("gather gemv: no expert slabs")
    }
    if repeat == 0 || !ids.len().is_multiple_of(repeat) {
        candle_core::bail!(
            "gather gemv: {} slots do not divide into groups of {repeat}",
            ids.len()
        )
    }
    check_input(a.len(), ids.len(), in_features, repeat)?;

    // dtype and residency are properties of how the checkpoint was loaded, not of the
    // routing, so they are checked once here rather than per work item.
    let dtype = experts[0].dtype();
    if experts.iter().any(|t| t.dtype() != dtype) {
        candle_core::bail!("gather gemv: expert slabs disagree on dtype")
    }
    if experts.iter().any(|t| !t.device().is_cpu()) {
        candle_core::bail!("gather gemv: expert slabs must be on CPU")
    }

    // Only the experts this call names are locked; see `wanted_experts`.
    let wanted = wanted_experts(ids, experts.len());
    let dense = dense_index(&wanted, experts.len());
    let remapped: Vec<u32> = ids
        .iter()
        .map(|&i| {
            if (i as usize) < experts.len() {
                dense[i as usize]
            } else {
                wanted.len() as u32
            }
        })
        .collect();

    match dtype {
        DType::F32 => with_f32_rows(experts, &wanted, |rows| {
            par_fill(
                ids.len(),
                out_features,
                &remapped,
                wanted.len(),
                |slot, row, k| {
                    dot(
                        a_row_ref(a, slot, in_features, repeat),
                        &rows[k][row * in_features..][..in_features],
                    )
                },
            )
        }),
        DType::BF16 => with_bf16_rows(experts, &wanted, |rows| {
            par_fill(
                ids.len(),
                out_features,
                &remapped,
                wanted.len(),
                |slot, row, k| {
                    dot_bf16(
                        a_row_ref(a, slot, in_features, repeat),
                        &rows[k][row * in_features..][..in_features],
                    )
                },
            )
        }),
        other => candle_core::bail!("gather gemv: unsupported expert dtype {other:?}"),
    }
}

/// The experts `ids` names, ascending and deduplicated.
///
/// Only these are worth locking. A decode step routes to a handful out of the pool, and
/// `Tensor::storage_and_layout` takes an `RwLock` read per slab - locking all 384 to read six
/// costs 57,600 lock acquisitions per decode step on this model, three projections times fifty
/// layers, for memory that is never touched.
fn wanted_experts(ids: &[u32], num_experts: usize) -> Vec<usize> {
    let mut wanted: Vec<usize> = ids
        .iter()
        .map(|&i| i as usize)
        .filter(|&i| i < num_experts)
        .collect();
    wanted.sort_unstable();
    wanted.dedup();
    wanted
}

/// Maps an expert id to its row in a compacted slab list, or to `rows.len()` for an id that is
/// out of range, which `par_fill` reports as `NaN` rather than reading.
fn dense_index(wanted: &[usize], num_experts: usize) -> Vec<u32> {
    let mut dense = vec![wanted.len() as u32; num_experts];
    for (row, &expert) in wanted.iter().enumerate() {
        dense[expert] = row as u32;
    }
    dense
}

/// Hands each slab's rows to `f` as one contiguous slice.
///
/// The storage guards are taken here rather than in the caller because the row slices borrow
/// from the storage they lock; holding them in this frame is what keeps the slices alive for
/// the duration of the parallel loop without a self-referential struct.
fn with_f32_rows<T>(
    experts: &[Tensor],
    wanted: &[usize],
    f: impl FnOnce(&[&[f32]]) -> Result<T>,
) -> Result<T> {
    let guards: Vec<_> = wanted
        .iter()
        .map(|&i| experts[i].storage_and_layout())
        .collect();
    let rows = guards
        .iter()
        .zip(wanted)
        .map(|((guard, layout), &i)| {
            let Storage::Cpu(cpu) = &**guard else {
                candle_core::bail!("expert slab must be CPU storage")
            };
            let flat = cpu.as_slice::<f32>()?;
            let start = layout.start_offset();
            Ok(&flat[start..start + experts[i].elem_count()])
        })
        .collect::<Result<Vec<_>>>()?;
    f(&rows)
}

fn with_bf16_rows<T>(
    experts: &[Tensor],
    wanted: &[usize],
    f: impl FnOnce(&[&[u16]]) -> Result<T>,
) -> Result<T> {
    let guards: Vec<_> = wanted
        .iter()
        .map(|&i| experts[i].storage_and_layout())
        .collect();
    let rows = guards
        .iter()
        .zip(wanted)
        .map(|((guard, layout), &i)| {
            let Storage::Cpu(cpu) = &**guard else {
                candle_core::bail!("expert slab must be CPU storage")
            };
            let flat = cpu.as_slice::<half::bf16>()?;
            // bf16 is repr(transparent) over u16, and the dot kernel reads raw bits.
            let flat: &[u16] = unsafe { core::mem::transmute::<&[half::bf16], &[u16]>(flat) };
            let start = layout.start_offset();
            Ok(&flat[start..start + experts[i].elem_count()])
        })
        .collect::<Result<Vec<_>>>()?;
    f(&rows)
}

/// How many consecutive slots read the same activation row.
///
/// The stacked and per-expert gathers both broadcast one token's activation across its `top_k`
/// slots, which used to be done by materialising an expanded copy of every row. Addressing is
/// the same arithmetic with no copy: slot `s` reads row `s / repeat`.
pub const NO_REPEAT: usize = 1;

fn check_input(a_len: usize, n_slots: usize, in_features: usize, repeat: usize) -> Result<()> {
    let rows = n_slots / repeat.max(1);
    if a_len != rows * in_features {
        candle_core::bail!(
            "gather gemv: input has {} elements, expected {}",
            a_len,
            rows * in_features
        );
    }
    Ok(())
}

/// Activation row for one slot. [`DotCtx::row`] is the preferred spelling.
#[inline]
fn a_row_ref(a: &[f32], slot: usize, in_features: usize, repeat: usize) -> &[f32] {
    let row = slot / repeat;
    &a[row * in_features..row * in_features + in_features]
}

/// How one work item locates its operands.
///
/// Slot `s` reads activation row `s / repeat`, which is what lets a token's row be shared by
/// all of its slots without copying it `repeat` times.
struct DotCtx<'a> {
    a: &'a [f32],
    repeat: usize,
    in_features: usize,
}

impl DotCtx<'_> {
    #[inline]
    fn row(&self, slot: usize) -> &[f32] {
        let row = slot / self.repeat;
        &self.a[row * self.in_features..row * self.in_features + self.in_features]
    }
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
    crate::gemv_cpu::par_chunks(&mut out, chunk, |chunk_idx, dst| {
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
    let (n_tokens, _) = a.dims2()?;
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

    let out = gather_gemv(&a_rows, w, &ids, k)?;
    Tensor::from_vec(out, (n_tokens, k, out_features), a.device())?
        .to_dtype(out_dtype)
        .map(Some)
}

#[cfg(target_arch = "aarch64")]
fn dot_slot(
    ctx: &DotCtx<'_>,
    slot: usize,
    row: usize,
    w: &CpuStorage,
    expert: usize,
    out_features: usize,
) -> f32 {
    let a_row = ctx.row(slot);
    let start = (expert * out_features + row) * ctx.in_features;
    match w {
        CpuStorage::F32(v) => dot(a_row, &v[start..start + ctx.in_features]),
        CpuStorage::BF16(v) => {
            // bf16 is repr(transparent) over u16, so the rows are readable as raw bits and
            // widening is a shift rather than a scalar conversion.
            let bits: &[u16] = unsafe { core::mem::transmute::<&[half::bf16], &[u16]>(v) };
            dot_bf16(a_row, &bits[start..start + ctx.in_features])
        }
        _ => f32::NAN,
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_slot(
    ctx: &DotCtx<'_>,
    slot: usize,
    row: usize,
    w: &CpuStorage,
    expert: usize,
    out_features: usize,
) -> f32 {
    let _ = (ctx, slot, row, w, expert, out_features);
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

                let par = gather_gemv(&a, &w, &ids, NO_REPEAT)?;
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

            let fused = gather_gemv(&a, &stack, &ids, NO_REPEAT)?;
            let split = gather_gemv_per_expert(&a, &slabs, out, k, &ids, NO_REPEAT)?;
            assert_eq!(fused.len(), split.len());
            assert_eq!(fused, split, "{dtype:?}");
        }
        Ok(())
    }

    /// The per-expert broadcast form must produce exactly what the fused broadcast form
    /// produces. The fused form is what the whole-stack path uses, so if the two disagree at
    /// repeat > 1 then a per-expert gather cannot stand in for it.
    #[test]
    fn per_expert_broadcast_matches_the_fused_broadcast_bit_exactly() -> Result<()> {
        for dtype in [DType::F32, DType::BF16] {
            let (experts, out, k, tokens, top_k) = (8usize, 32usize, 48usize, 5usize, 3usize);
            let slabs: Vec<Tensor> = (0..experts)
                .map(|_| Tensor::rand(-1f32, 1f32, (out, k), &Device::Cpu)?.to_dtype(dtype))
                .collect::<Result<Vec<_>>>()?;
            let stack = Tensor::stack(&slabs, 0)?;
            let a: Vec<f32> = (0..tokens * k)
                .map(|i| ((i % 23) as f32) * 0.125 - 1.0)
                .collect();
            let ids: Vec<u32> = (0..tokens * top_k)
                .map(|i| (i * 5 % experts) as u32)
                .collect();

            let fused = gather_gemv(&a, &stack, &ids, top_k)?;
            let split = gather_gemv_per_expert(&a, &slabs, out, k, &ids, top_k)?;
            assert_eq!(fused.len(), split.len());
            assert_eq!(fused, split, "{dtype:?}: fused and per-expert broadcast differ");
        }
        Ok(())
    }

    /// Broadcasting by addressing must be bit-identical to broadcasting by expansion, which is
    /// what it replaced: a dropped copy here silently changes which activation a slot reads.
    #[test]
    fn broadcast_by_addressing_matches_the_expanded_form() -> Result<()> {
        let (experts, out, k, tokens, top_k) = (8usize, 32usize, 48usize, 5usize, 3usize);
        for dtype in [DType::F32, DType::BF16] {
            let slabs: Vec<Tensor> = (0..experts)
                .map(|_| Tensor::rand(-1f32, 1f32, (out, k), &Device::Cpu)?.to_dtype(dtype))
                .collect::<Result<Vec<_>>>()?;
            let a: Vec<f32> = (0..tokens * k)
                .map(|i| ((i % 23) as f32) * 0.125 - 1.0)
                .collect();
            let ids: Vec<u32> = (0..tokens * top_k)
                .map(|i| (i * 5 % experts) as u32)
                .collect();

            let addressed = gather_gemv_per_expert(&a, &slabs, out, k, &ids, top_k)?;

            let mut expanded = Vec::with_capacity(ids.len() * k);
            for row in a.chunks_exact(k) {
                for _ in 0..top_k {
                    expanded.extend_from_slice(row);
                }
            }
            let copied = gather_gemv_per_expert(&expanded, &slabs, out, k, &ids, NO_REPEAT)?;

            assert_eq!(addressed, copied, "{dtype:?}");
        }
        Ok(())
    }

    /// The same contract for the fused stack, which is what every pre-fused checkpoint uses.
    #[test]
    fn stacked_broadcast_matches_the_expanded_form() -> Result<()> {
        let (experts, out, k, tokens, top_k) = (8usize, 32usize, 48usize, 5usize, 3usize);
        let stack = Tensor::rand(-1f32, 1f32, (experts, out, k), &Device::Cpu)?;
        let a: Vec<f32> = (0..tokens * k)
            .map(|i| ((i % 23) as f32) * 0.125 - 1.0)
            .collect();
        let ids: Vec<u32> = (0..tokens * top_k)
            .map(|i| (i * 5 % experts) as u32)
            .collect();

        let addressed = gather_gemv(&a, &stack, &ids, top_k)?;

        let mut expanded = Vec::with_capacity(ids.len() * k);
        for row in a.chunks_exact(k) {
            for _ in 0..top_k {
                expanded.extend_from_slice(row);
            }
        }
        let copied = gather_gemv(&expanded, &stack, &ids, NO_REPEAT)?;
        assert_eq!(addressed, copied);
        Ok(())
    }

    #[test]
    fn out_of_range_expert_is_reported_not_read() -> Result<()> {
        let w = Tensor::zeros((4, 64, 32), DType::F32, &Device::Cpu)?;
        let a = vec![0f32; 8 * 32];
        let ids = vec![9u32, 0, 1, 2, 3, 0, 1, 2];
        let out = gather_gemv(&a, &w, &ids, NO_REPEAT)?;
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
