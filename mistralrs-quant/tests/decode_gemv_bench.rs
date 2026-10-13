//! Isolated decode-GEMV throughput on the shapes a 4B MoE actually uses.
//!
//! The model-level benchmark mixes this kernel with attention, norms and MoE, so a slow
//! decode step does not say which. This pins it: the same `gemv` call, the same weights and
//! dtypes as the real layers, timed with everything else removed.
//!
//! `KOLIBRI1_GEMV_BENCH=1` runs it; the shapes default to Kolibri-1's.

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use mistralrs_quant::{gemv_cpu, moe_gather_cpu};

fn timed_gemv(name: &str, n: usize, k: usize, dtype: DType, reps: usize) -> f64 {
    let dev = Device::Cpu;
    let w = Tensor::rand(-1f32, 1f32, (n, k), &dev)
        .expect("weight")
        .to_dtype(dtype)
        .expect("cast");
    let xs = Tensor::rand(-1f32, 1f32, (1, 1, k), &dev)
        .expect("input")
        .to_dtype(dtype)
        .expect("cast");

    // One untimed call so the first-touch cost of the weight is not in the measurement.
    let _ = gemv_cpu::gemv(&xs, &w).expect("warmup");

    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = gemv_cpu::gemv(&xs, &w).expect("gemv");
    }
    let secs = t0.elapsed().as_secs_f64() / reps as f64;
    let bytes = (n * k) as f64 * if dtype == DType::BF16 { 2.0 } else { 4.0 };
    let gbps = bytes / 1e9 / secs;
    println!(
        "  {name:<12} [{n:>6}, {k}] {dtype:?} {reps:>3} reps  {:>8.2} ms  {gbps:>6.1} GB/s",
        secs * 1e3
    );
    gbps
}

#[test]
fn decode_gemv_isolated_throughput() {
    if std::env::var("KOLIBRI1_GEMV_BENCH").is_err() {
        return;
    }
    let threads = mistralrs_quant::cpu_threads::resolve();
    println!("threads = {threads}");
    let reps = 20;
    // q_proj / k_proj / o_proj / lm_head / a shared expert, at Kolibri-1's shapes. `k_proj`
    // is 512 outputs, below `MIN_PARALLEL_OUTPUTS`, so the model does *not* reach this kernel
    // for it - the number here is what it would get if the floor allowed.
    let shapes: [(&str, usize, usize); 5] = [
        ("q_proj", 6144, 2560),
        ("k_proj", 512, 2560),
        ("o_proj", 2560, 6144),
        ("shared_exp", 2560, 512),
        ("lm_head", 128_000, 2560),
    ];
    for (name, n, k) in shapes {
        timed_gemv(name, n, k, DType::BF16, reps);
    }

    // The routed experts, which are half the MoE bytes and do not go through `gemv` at all.
    // `in_f32` is what a batch-1 token broadcasts across its six slots.
    timed_gather("expert_gate", 6, 512, 2560, reps);
    timed_gather("expert_down", 6, 2560, 512, reps);
}

/// Routed-expert gather at Kolibri-1's shape: one activation row, `top_k` slots, `top_k`
/// distinct `[out, in]` slabs.
fn timed_gather(name: &str, top_k: usize, out: usize, in_features: usize, reps: usize) -> f64 {
    let dev = Device::Cpu;
    let experts: Vec<Tensor> = (0..top_k)
        .map(|_| Tensor::rand(-1f32, 1f32, (out, in_features), &dev).expect("slab"))
        .collect();
    let a: Vec<f32> = (0..in_features).map(|i| (i % 7) as f32).collect();
    let ids: Vec<u32> = (0..top_k).map(|i| i as u32).collect();

    let _ = moe_gather_cpu::gather_gemv_per_expert(&a, &experts, out, in_features, &ids, top_k)
        .expect("warmup");

    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = moe_gather_cpu::gather_gemv_per_expert(&a, &experts, out, in_features, &ids, top_k)
            .expect("gather");
    }
    let secs = t0.elapsed().as_secs_f64() / reps as f64;
    let bytes = (top_k * out * in_features) as f64 * 2.0;
    let gbps = bytes / 1e9 / secs;
    println!(
        "  {name:<12} [{top_k} x ({out}, {in_features})] {reps:>3} reps  {:>8.2} ms  {gbps:>6.1} GB/s",
        secs * 1e3
    );
    gbps
}
