//! Isolated bandwidth probe for the CPU MoE gather-GEMV.
//!
//! Times the parallel kernel against the index-select-and-matmul route it replaces, on a
//! stacked-expert weight shaped like Kolibri-1's (384 experts, 2560 hidden, 512 inter).
//!
//!   cargo run --release --example moe_gather_cpu_bench -p mistralrs-quant

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use mistralrs_quant::moe_gather_cpu;

fn bench(name: &str, iters: usize, mut f: impl FnMut() -> Tensor) {
    // Warm the pages so the first iteration is not the only one that pays for the fault-in.
    let _ = f();
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(f());
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!("{name:<34} {:8.2} ms", elapsed * 1000.0 / iters as f64);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let experts = 384usize;
    let out_features = 512usize;
    let hidden = 2560usize;
    let n_tokens = 1usize;
    let k = 6usize;

    let dtype = match std::env::var("DTYPE").as_deref() {
        Ok("f32") => DType::F32,
        _ => DType::BF16,
    };
    let threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    println!(
        "stacked weight [{}, {}, {}] {:?}, {} tokens x {} experts, {} rayon threads",
        experts, out_features, hidden, dtype, n_tokens, k, threads
    );

    let w = Tensor::rand(-0.05f32, 0.05, (experts, out_features, hidden), &dev)?.to_dtype(dtype)?;
    let a = Tensor::rand(-1f32, 1f32, (n_tokens, hidden), &dev)?.to_dtype(dtype)?;
    // Built [k, n_tokens] then transposed so consecutive slots pick different experts.
    let ids = Tensor::from_vec(
        (0..n_tokens * k)
            .map(|i| (i * 61 % experts) as u32)
            .collect::<Vec<u32>>(),
        (k, n_tokens),
        &dev,
    )?
    .t()?
    .contiguous()?;

    // The routed weight traffic per call, which is the lower bound on the time.
    let bytes = (n_tokens * k * out_features * hidden * dtype.size_in_bytes()) as f64;
    println!("routed weight per call: {:.1} MB\n", bytes / 1e6);

    let flat_ids = ids.reshape((n_tokens * k,))?;
    bench("index_select + matmul (old)", 5, || {
        let selected = w.index_select(&flat_ids, 0).expect("index_select");
        let expanded = a
            .unsqueeze(1)
            .expect("unsqueeze")
            .broadcast_as((n_tokens, k, hidden))
            .expect("broadcast")
            .reshape((n_tokens * k, hidden))
            .expect("reshape");
        expanded
            .unsqueeze(1)
            .expect("unsqueeze")
            .matmul(&selected.transpose(1, 2).expect("t"))
            .expect("matmul")
            .squeeze(1)
            .expect("squeeze")
    });

    bench("gather gemv (new)", 5, || {
        moe_gather_cpu::gather_broadcast_rows(&a, &w, &ids, out_features, k, dtype)
            .expect("gather")
            .expect("fast path")
    });

    let n = 20;
    let start = Instant::now();
    for _ in 0..n {
        std::hint::black_box(moe_gather_cpu::gather_broadcast_rows(
            &a,
            &w,
            &ids,
            out_features,
            k,
            dtype,
        )?);
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "\nnew kernel effective bandwidth: {:.1} GB/s",
        bytes * n as f64 / elapsed / 1e9
    );
    Ok(())
}
