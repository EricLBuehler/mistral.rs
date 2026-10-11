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

/// Below this many outputs there is not enough work to pay for splitting.
const MIN_PARALLEL_OUTPUTS: usize = 1024;

/// Whether the decode-shaped fast path applies to this input and weight pair.
pub fn should_use_gemv(xs: &Tensor, w: &Tensor) -> bool {
    xs.device().is_cpu()
        && xs.rank() >= 2
        && xs.dim(xs.rank() - 2).is_ok_and(|rows| rows == 1)
        && w.dim(0).is_ok_and(|n| n >= MIN_PARALLEL_OUTPUTS)
        && w.dim(1).is_ok_and(|k| k >= 1)
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
    let out = gemv_into_vec(&xs, w, n, k)?;
    // Preserve the input rank: the forward pipeline expects the trailing-dim layout back.
    let mut shape: Vec<usize> = xs.dims()[..xs.rank() - 1].to_vec();
    shape.push(n);
    Tensor::from_vec(out, shape, xs.device())
}

fn gemv_into_vec(xs: &Tensor, w: &Tensor, n: usize, k: usize) -> Result<Vec<f32>> {
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
    let w = w_cpu.as_slice::<f32>()?;

    let threads = crate::cpu_threads::resolve();
    if n < MIN_PARALLEL_OUTPUTS || threads <= 1 {
        return Ok(serial(x, w, n, k));
    }

    let chunk = n.div_ceil(threads);
    let mut out = vec![0f32; n];
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(chunk_idx, slot)| {
            let rows = &w[chunk_idx * chunk * k..];
            slot.iter_mut()
                .enumerate()
                .for_each(|(i, dst)| *dst = dot(x, &rows[i * k..i * k + k]));
        });

    Ok(out)
}

fn dot(x: &[f32], w: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (xi, wi) in x.iter().zip(w) {
        acc += xi * wi;
    }
    acc
}

fn serial(x: &[f32], w: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n).map(|row| dot(x, &w[row * k..row * k + k])).collect()
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
            let ser = serial(
                &xs.to_dtype(DType::F32)?.contiguous()?.to_vec2::<f32>()?[0],
                &w.to_vec2::<f32>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
                n,
                k,
            );
            // The partitioning invariant: splitting N across threads must not perturb any
            // row's accumulation, so this is exact equality, not a tolerance.
            assert_eq!(par, ser, "{k}x{n}");
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
