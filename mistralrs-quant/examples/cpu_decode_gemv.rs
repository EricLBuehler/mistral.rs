//! Decode-shaped GEMV bandwidth probe for CPU quantized and dense linears.
//!
//! Isolates the per-token decode linear so CPU GEMV work can be iterated on without a
//! full model. Reports achieved weight-stream bandwidth against the node's ceiling.
//!
//! Run pinned to a NUMA node: `numactl --cpunodebind=3 --interleave=all cargo run --release
//! --example cpu_decode_gemv -p mistralrs-quant`.

use std::time::Instant;

use candle_core::{DType, Device, Tensor};

const HIDDEN: usize = 2048;
const FFN: usize = 5632;
const ITERS: usize = 32;
const WARMUP: usize = 4;

fn gemv_bench(name: &str, w: &Tensor, x: &Tensor, weight_bytes: usize) {
    for _ in 0..WARMUP {
        let _ = x.matmul(w).unwrap();
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = x.matmul(w).unwrap();
    }
    let dt = start.elapsed().as_secs_f64() / ITERS as f64;
    let bw = weight_bytes as f64 / dt / 1e9;
    println!("{:<28} {:8.3} ms/call  {:7.1} GB/s", name, dt * 1e3, bw);
}

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    println!("threads: {}", candle_core::utils::get_num_threads());

    // Decode shape: one token against a [K, N] weight, row-major so the read is contiguous.
    let x = Tensor::rand(0f32, 1f32, (1, HIDDEN), &dev)?;

    for (name, dtype, elem) in [
        ("f32 dense", DType::F32, 4usize),
        ("bf16 dense", DType::BF16, 2),
    ] {
        let w = Tensor::rand(0f32, 1f32, (FFN, HIDDEN), &dev)?.to_dtype(dtype)?;
        let xd = x.to_dtype(dtype)?;
        gemv_bench(name, &w.t().unwrap(), &xd, HIDDEN * FFN * elem);

        let wf = w.to_dtype(DType::F32)?.contiguous()?;
        let xf = xd.to_dtype(DType::F32)?.contiguous()?;
        for _ in 0..WARMUP {
            std::hint::black_box(mistralrs_quant::gemv_cpu::gemv(&xf, &wf).unwrap());
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(mistralrs_quant::gemv_cpu::gemv(&xf, &wf).unwrap());
        }
        let dt = start.elapsed().as_secs_f64() / ITERS as f64;
        let bytes = (HIDDEN * FFN * elem) as f64;
        println!(
            "{:<28} {:8.3} ms/call  {:7.1} GB/s",
            "gemv (output-parallel)",
            dt * 1e3,
            bytes / dt / 1e9
        );
    }

    // The decode bottleneck is streaming weights, so report the same GEMV as a pure
    // column-sum over the weight to expose the achievable read ceiling on this node.
    let w = Tensor::rand(0f32, 1f32, (HIDDEN, FFN), &dev)?;
    for _ in 0..WARMUP {
        let _ = w.sum(1).unwrap();
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = w.sum(1).unwrap();
    }
    let dt = start.elapsed().as_secs_f64() / ITERS as f64;
    let bytes = (HIDDEN * FFN * 4) as f64;
    println!(
        "{:<28} {:8.3} ms/call  {:7.1} GB/s",
        "weight read ceiling",
        dt * 1e3,
        bytes / dt / 1e9
    );

    Ok(())
}
