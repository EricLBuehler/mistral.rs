//! Output-parallel CPU GEMV for batch-1 decode.
//!
//! A decode step is `[1, K] x [K, N]` and must stream the whole weight matrix once, so it is
//! purely bandwidth-bound: the only thing that matters is how many cores stream the weight at
//! once. Candle's CPU path runs this at roughly one core's bandwidth whatever the thread
//! count, so partitioning `N` across cores is what reaches aggregate bandwidth.
//!
//! Partitioning is over output rows only. Each output element is produced end-to-end by one
//! thread walking `k` in the same order a serial loop would, so results are bit-identical to
//! the serial path by construction and no cross-thread reduction is introduced.

use candle_core::{DType, Result, Storage, Tensor};
use rayon::prelude::*;

use crate::dot_cpu::{dot_bf16, dot_f32};

/// Below this many outputs there is not enough work to pay for splitting.
///
/// 512 rather than 1024: a GQA key/value projection is exactly 512 outputs at head_dim 128 with
/// 4 KV heads, and at 1024 it fell through to candle's generic GEMM. That path packs both
/// operands into scratch and builds a `BarrierPool` per call, and a decode-only profile put
/// k/v and up at 8.6% of runtime for 3.8% of the bytes - 5.8 GB/s where this kernel does the
/// same shape at 16.3.
const MIN_PARALLEL_OUTPUTS: usize = 512;

/// Overrides [`MIN_PARALLEL_OUTPUTS`] for measurement.
///
/// The thread sweep and the barrier-pool experiments both needed "run this shape serially"
/// without rebuilding, and the shape that matters is whatever a given model routes through a
/// small projection. Exposed as an override rather than a constant so the crossover can be
/// measured rather than argued.
fn min_parallel_outputs() -> usize {
    static OVERRIDE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("MISTRALRS_GEMV_MIN_OUTPUTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(MIN_PARALLEL_OUTPUTS)
    })
}

/// Weight dtypes this kernel reads directly. Anything else falls back to candle's matmul.
fn is_kernel_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::BF16)
}

/// Whether the decode-shaped fast path applies to this input and weight pair.
pub fn should_use_gemv(xs: &Tensor, w: &Tensor) -> bool {
    xs.device().is_cpu()
        && xs.rank() >= 2
        && xs.dim(xs.rank() - 2).is_ok_and(|rows| rows == 1)
        && w.dim(0).is_ok_and(|n| n >= min_parallel_outputs())
        && w.dim(1).is_ok_and(|k| k >= 1)
        && is_kernel_dtype(w.dtype())
}

/// `xs @ w_i^T` for several weights that share `K`, in one parallel region.
///
/// q/k/v are three GEMVs over the same activation. Done separately that is three regions joined
/// one at a time, and on a decode-only profile rayon synchronisation was 27% of runtime. Done
/// together they are one region, one pass over the activation, and the small one (a GQA k or v
/// at 512 outputs) rides along inside the large one's thread budget instead of paying a
/// barrier to parallelise 82 KB.
///
/// Returns `None` when the shapes or dtypes do not suit, so the caller keeps its own path.
pub fn try_gemv_fused(xs: &Tensor, ws: &[&Tensor]) -> Result<Option<Vec<Tensor>>> {
    if ws.len() < 2 || !xs.device().is_cpu() || xs.rank() < 2 {
        return Ok(None);
    }
    // Exactly one row of work: the leading dims all one. The single-weight path accepts a
    // wider shape and reads only the first row, which is not something to copy here.
    if !xs.dims()[..xs.rank() - 1].iter().all(|d| *d == 1) {
        return Ok(None);
    }
    let k = match ws[0].dim(1) {
        Ok(k) => k,
        Err(_) => return Ok(None),
    };
    if !ws
        .iter()
        .all(|w| w.dim(1).is_ok_and(|wk| wk == k) && is_kernel_dtype(w.dtype()))
    {
        return Ok(None);
    }
    if !matches!(xs.dtype(), DType::F32 | DType::BF16) {
        return Ok(None);
    }
    if !xs.dim(xs.rank() - 1).is_ok_and(|xk| xk == k) {
        return Ok(None);
    }
    // A strided weight would need a gather per row; leave it to the per-weight path.
    for w in ws {
        let Ok(n) = w.dim(0) else { return Ok(None) };
        let layout = w.layout();
        if layout.dims() != [n, k] || layout.stride()[0] != k {
            return Ok(None);
        }
    }

    let total: usize = ws.iter().filter_map(|w| w.dim(0).ok()).sum();
    if total == 0 {
        return Ok(None);
    }
    let outs = fused_into_vec(xs, ws, total, k)?;
    // Leading dims are carried through; the trailing one becomes this weight's output count.
    let leading: Vec<usize> = xs.dims()[..xs.rank() - 1].to_vec();
    let mut out = Vec::with_capacity(ws.len());
    let mut cursor = 0;
    for w in ws {
        let n = w.dim(0)?;
        let mut shape = leading.clone();
        shape.push(n);
        out.push(
            Tensor::from_vec(outs[cursor..cursor + n].to_vec(), shape, xs.device())?
                .to_dtype(w.dtype())?,
        );
        cursor += n;
    }
    Ok(Some(out))
}

fn fused_into_vec(xs: &Tensor, ws: &[&Tensor], total: usize, k: usize) -> Result<Vec<f32>> {
    let xs = xs.to_dtype(DType::F32)?.contiguous()?;
    let xs = &xs;
    let (x_guard, x_layout) = xs.storage_and_layout();
    let Storage::Cpu(x_cpu) = &*x_guard else {
        candle_core::bail!("CPU gemv requires CPU storage")
    };
    let x_start = x_layout.start_offset();
    let x = &x_cpu.as_slice::<f32>()?[x_start..x_start + k];

    // Guards held here rather than in the caller: the row slices borrow from the storage they
    // lock, so one frame has to own them for the whole parallel loop.
    let guards: Vec<_> = ws.iter().map(|w| w.storage_and_layout()).collect();
    let mut rows: Vec<Weight<'_>> = Vec::with_capacity(ws.len());
    for ((guard, _), w) in guards.iter().zip(ws) {
        let Storage::Cpu(cpu) = &**guard else {
            candle_core::bail!("CPU gemv requires CPU storage")
        };
        let n = w.dim(0)?;
        rows.push(match w.dtype() {
            DType::F32 => Weight::F32(&cpu.as_slice::<f32>()?[..n * k]),
            // bf16 is repr(transparent) over u16 and the dot kernel reads raw bits.
            DType::BF16 => Weight::Bf16(unsafe {
                core::mem::transmute::<&[half::bf16], &[u16]>(cpu.as_slice::<half::bf16>()?)
            }),
            other => candle_core::bail!("CPU gemv does not read {other:?} weights"),
        });
    }
    // Where each weight's outputs start in the flat result.
    let mut offsets = Vec::with_capacity(rows.len());
    let mut acc = 0;
    for r in &rows {
        offsets.push(acc);
        acc += r.len() / k;
    }

    let mut out = vec![0f32; total];
    let threads = crate::cpu_threads::resolve().max(1);
    let chunk = total.div_ceil(threads);
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(chunk_idx, dst)| {
            for (i, d) in dst.iter_mut().enumerate() {
                let idx = chunk_idx * chunk + i;
                // Three weights at most, so a linear scan of the offsets beats a binary search.
                let mut j = 0;
                while j + 1 < offsets.len() && idx >= offsets[j + 1] {
                    j += 1;
                }
                *d = rows[j].dot(x, idx - offsets[j], k);
            }
        });
    Ok(out)
}

/// `xs @ w^T` for a single leading row, with the `N` outputs split across the rayon pool.
pub fn gemv(xs: &Tensor, w: &Tensor) -> Result<Tensor> {
    let k = w.dim(1)?;
    let n = w.dim(0)?;
    if xs.dim(xs.rank() - 1)? != k {
        candle_core::bail!("CPU gemv shape mismatch: input K does not match weight K")
    }
    // A strided weight would need a gather per row; fall back rather than read wrong elements.
    let layout = w.layout();
    if layout.dims() != [n, k] || layout.stride()[0] != k {
        return xs.matmul(&w.t()?);
    }

    let xs = xs.to_dtype(DType::F32)?.contiguous()?;
    let out_dtype = w.dtype();
    let out = gemv_into_vec(&xs, w, n, k)?;
    // Preserve the input rank: the forward pipeline expects the trailing-dim layout back.
    let mut shape: Vec<usize> = xs.dims()[..xs.rank() - 1].to_vec();
    shape.push(n);
    Tensor::from_vec(out, shape, xs.device())?.to_dtype(out_dtype)
}

/// The weight, borrowed from a storage guard held for the call.
///
/// Borrowed rather than widened: at decode the weight is the largest thing that exists and
/// materialising it as f32 would double the streaming this kernel exists to amortise.
enum Weight<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
}

impl Weight<'_> {
    fn len(&self) -> usize {
        match self {
            Self::F32(w) => w.len(),
            Self::Bf16(w) => w.len(),
        }
    }

    fn dot(&self, x: &[f32], row: usize, k: usize) -> f32 {
        match self {
            Self::F32(w) => dot_f32(x, &w[row * k..row * k + k]),
            Self::Bf16(w) => dot_bf16(x, &w[row * k..row * k + k]),
        }
    }
}

pub fn gemv_into_vec(xs: &Tensor, w: &Tensor, n: usize, k: usize) -> Result<Vec<f32>> {
    let (x_guard, x_layout) = xs.storage_and_layout();
    let Storage::Cpu(x_cpu) = &*x_guard else {
        candle_core::bail!("CPU gemv requires CPU storage")
    };
    let x_start = x_layout.start_offset();
    let x = &x_cpu.as_slice::<f32>()?[x_start..x_start + k];

    let (w_guard, _) = w.storage_and_layout();
    let Storage::Cpu(w_cpu) = &*w_guard else {
        candle_core::bail!("CPU gemv requires CPU storage")
    };
    let weight = match w.dtype() {
        DType::F32 => Weight::F32(w_cpu.as_slice::<f32>()?),
        // bf16 is repr(transparent) over u16 and the dot kernel reads raw bits.
        DType::BF16 => Weight::Bf16(unsafe {
            core::mem::transmute::<&[half::bf16], &[u16]>(w_cpu.as_slice::<half::bf16>()?)
        }),
        other => candle_core::bail!("CPU gemv does not read {other:?} weights"),
    };

    let threads = crate::cpu_threads::resolve();
    if n < min_parallel_outputs() || threads <= 1 {
        return Ok((0..n).map(|row| weight.dot(x, row, k)).collect());
    }

    let chunk = n.div_ceil(threads);
    let mut out = vec![0f32; n];
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(chunk_idx, slot)| {
            slot.iter_mut()
                .enumerate()
                .for_each(|(i, dst)| *dst = weight.dot(x, chunk_idx * chunk + i, k));
        });

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn gemv_preserves_input_rank() -> Result<()> {
        let dev = Device::Cpu;
        let xs = Tensor::rand(0f32, 1f32, (1, 1, 256), &dev)?;
        let w = Tensor::rand(0f32, 1f32, (2048, 256), &dev)?;
        let got = gemv(&xs, &w)?;
        assert_eq!(got.dims(), &[1, 1, 2048]);
        let want = xs.broadcast_matmul(&w.t()?.unsqueeze(0)?)?;
        for (g, x) in got
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .zip(want.flatten_all()?.to_vec1::<f32>()?.iter())
        {
            assert!((g - x).abs() <= 1e-3 * x.abs().max(1e-3));
        }
        Ok(())
    }

    #[test]
    fn parallel_matches_serial_bit_exactly() -> Result<()> {
        let dev = Device::Cpu;
        for (k, n) in [(2048usize, 5632usize), (4096, 11008), (128, 2048)] {
            let xs = Tensor::rand(0f32, 1f32, (1, k), &dev)?;
            let w = Tensor::rand(0f32, 1f32, (n, k), &dev)?;
            let par = gemv_into_vec(&xs.to_dtype(DType::F32)?.contiguous()?, &w, n, k)?;
            let xs_row = xs.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?[0].clone();
            let w_flat = w
                .to_vec2::<f32>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            let ser = (0..n)
                .map(|row| dot_f32(&xs_row, &w_flat[row * k..row * k + k]))
                .collect::<Vec<_>>();
            // The partitioning invariant: splitting N across threads must not perturb any
            // row's accumulation, so this is exact equality, not a tolerance.
            assert_eq!(par, ser, "{k}x{n}");
        }
        Ok(())
    }

    /// The regression: `should_use_gemv` claimed every CPU weight, but the kernel only read
    /// f32, so a bf16 model failed inside the fast path instead of falling back. bf16 is the
    /// default checkpoint dtype, so this was not a corner.
    #[test]
    fn bf16_weights_take_the_fast_path_and_match_matmul() -> Result<()> {
        let dev = Device::Cpu;
        let (k, n) = (2048usize, 4096usize);
        let w = Tensor::rand(0f32, 1f32, (n, k), &dev)?.to_dtype(DType::BF16)?;
        for shape in [vec![1, 1, k], vec![1, k]] {
            let xs = Tensor::rand(0f32, 1f32, shape.as_slice(), &dev)?.to_dtype(DType::BF16)?;
            assert!(should_use_gemv(&xs, &w), "{shape:?} should be fast-pathed");
            let got = gemv(&xs, &w)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let want = xs
                .to_dtype(DType::F32)?
                .flatten_all()?
                .reshape((1, k))?
                .matmul(&w.to_dtype(DType::F32)?.t()?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(got.len(), want.len());
            // The result is narrowed back to bf16, so it cannot beat bf16 precision: 8 mantissa
            // bits put the relative resolution at 2^-8, about 0.4%.
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            for (g, wv) in got.iter().zip(&want) {
                assert!((g - wv).abs() <= 8e-3 * scale, "{g} vs {wv}");
            }
        }
        Ok(())
    }

    /// The fused form must produce exactly what three separate GEMVs produce, per weight: a
    /// mixed-up offset here would hand q the rows of k and be invisible in the shape.
    #[test]
    fn fused_matches_three_separate_gemvs() -> Result<()> {
        let dev = Device::Cpu;
        for dtype in [DType::F32, DType::BF16] {
            // Deliberately unequal, and with the small one last, which is the q/k/v shape.
            let ws: Vec<Tensor> = [1024usize, 512, 512]
                .iter()
                .map(|n| Tensor::rand(-1f32, 1f32, (*n, 768), &dev)?.to_dtype(dtype))
                .collect::<Result<Vec<_>>>()?;
            let xs = Tensor::rand(-1f32, 1f32, (1, 1, 768), &dev)?.to_dtype(dtype)?;
            let refs: Vec<&Tensor> = ws.iter().collect();

            let fused = try_gemv_fused(&xs, &refs)?.expect("should fuse");
            assert_eq!(fused.len(), 3);
            for (got, w) in fused.iter().zip(&ws) {
                let want = gemv(&xs, w)?;
                assert_eq!(got.dims(), want.dims(), "{dtype:?}");
                assert_eq!(got.dtype(), want.dtype(), "{dtype:?}");
                let g = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let x = want.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(g, x, "{dtype:?} n={}", w.dim(0)?);
            }
        }
        Ok(())
    }

    /// Anything the fused path does not recognise has to decline rather than guess.
    #[test]
    fn fused_declines_shapes_it_cannot_handle() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::rand(0f32, 1f32, (2048, 64), &dev)?;
        let b = Tensor::rand(0f32, 1f32, (512, 64), &dev)?;
        let xs = Tensor::rand(0f32, 1f32, (1, 1, 64), &dev)?;

        // Different K.
        let c = Tensor::rand(0f32, 1f32, (512, 32), &dev)?;
        assert!(try_gemv_fused(&xs, &[&a, &c])?.is_none());
        // More than one query row is not the decode shape this kernel handles.
        let xs_batch = Tensor::rand(0f32, 1f32, (2, 1, 64), &dev)?;
        assert!(try_gemv_fused(&xs_batch, &[&a, &b])?.is_none());
        // A dtype the kernel cannot read.
        let f16 = a.to_dtype(DType::F16)?;
        assert!(try_gemv_fused(&xs.to_dtype(DType::BF16)?, &[&f16, &b])?.is_none());
        Ok(())
    }

    /// A dtype the kernel does not read must fall back rather than be claimed and then fail.
    #[test]
    fn unsupported_weight_dtypes_are_not_claimed() -> Result<()> {
        let dev = Device::Cpu;
        let xs = Tensor::zeros((1, 1, 8), DType::F32, &dev)?;
        for dtype in [DType::F16, DType::U8, DType::BF16] {
            let w = Tensor::zeros((2048, 8), dtype, &dev)?;
            let claimed = should_use_gemv(&xs, &w);
            if claimed {
                // Claimed means the kernel must actually read it.
                gemv(&xs, &w)?;
            } else {
                assert!(matches!(dtype, DType::F16 | DType::U8), "{dtype:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn gemv_tracks_candle_matmul() -> Result<()> {
        let dev = Device::Cpu;
        let xs = Tensor::rand(0f32, 1f32, (1, 2048), &dev)?;
        let w = Tensor::rand(0f32, 1f32, (5632, 2048), &dev)?;
        let got = gemv(&xs, &w)?.to_vec2::<f32>()?[0].clone();
        let want = xs.matmul(&w.t()?)?.to_vec2::<f32>()?[0].clone();
        assert_eq!(got.len(), want.len());
        // Candle's gemm accumulates with FMA; this kernel accumulates serially, so the two
        // agree only to rounding. The gap is asserted so a real regression stands out.
        for (g, x) in got.iter().zip(&want) {
            assert!((g - x).abs() <= 1e-3 * x.abs().max(1e-3), "{g} vs {x}");
        }
        Ok(())
    }
}
