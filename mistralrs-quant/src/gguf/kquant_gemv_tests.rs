use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

struct GemvShape {
    n: usize,
    k: usize,
    dtype: GgmlDType,
    label: &'static str,
}

const SHAPES: [GemvShape; 4] = [
    GemvShape {
        n: 150_144,
        k: 2_048,
        dtype: GgmlDType::Q6K,
        label: "lm_head",
    },
    GemvShape {
        n: 4_096,
        k: 2_048,
        dtype: GgmlDType::Q4K,
        label: "attn_qo",
    },
    GemvShape {
        n: 512,
        k: 2_048,
        dtype: GgmlDType::Q4K,
        label: "attn_kv",
    },
    GemvShape {
        n: 1_024,
        k: 2_048,
        dtype: GgmlDType::Q4K,
        label: "shared",
    },
];

fn bytes_per_value(dtype: GgmlDType) -> f64 {
    match dtype {
        GgmlDType::Q4K => 4.5 / 8.0,
        GgmlDType::Q6K => 6.5625 / 8.0,
        other => panic!("unsupported dtype {other:?}"),
    }
}

fn pseudo_f32(len: usize, seed: usize) -> Vec<f32> {
    let mut state = seed as u64 | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 33) % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

fn bench_variant(dequantize_all: bool, iters: usize) {
    std::env::set_var(
        "CANDLE_DEQUANTIZE_ALL",
        if dequantize_all { "1" } else { "0" },
    );
    let device = Device::Cpu;
    for shape in SHAPES {
        let weight = pseudo_f32(shape.n * shape.k, shape.n);
        let src = Tensor::from_vec(weight, (shape.n, shape.k), &device).expect("src");
        let qtensor = QTensor::quantize(&src, shape.dtype).expect("quantize");
        let w = QMatMul::from_arc(Arc::new(qtensor)).expect("qmatmul");
        let x = Tensor::from_vec(pseudo_f32(shape.k, 7), (1, shape.k), &device).expect("x");

        let first = {
            let started = std::time::Instant::now();
            std::hint::black_box(w.forward(&x).expect("forward"));
            started.elapsed().as_secs_f64()
        };
        let steady = {
            let started = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(w.forward(&x).expect("forward"));
            }
            started.elapsed().as_secs_f64() / iters as f64
        };
        let gbytes = shape.n as f64 * shape.k as f64 * bytes_per_value(shape.dtype) / 1e9;
        println!(
            "{:<8} {:>7}x{:<5} {:?} first {:.2} ms, steady {:.3} ms, {:.1} GB/s",
            shape.label,
            shape.n,
            shape.k,
            shape.dtype,
            first * 1e3,
            steady * 1e3,
            gbytes / steady,
        );
    }
    std::env::remove_var("CANDLE_DEQUANTIZE_ALL");
}

/// GEMV rate of the packed K-quant path against the dequantized-f32 fallback, at the
/// Q4_K_M shapes. `--ignored`; CANDLE_DEQUANTIZE_ALL is toggled per thread.
#[test]
#[ignore = "measurement"]
fn kquant_gemv_rates() {
    let iters = 20;
    let packed = std::thread::spawn(move || bench_variant(false, iters));
    packed.join().expect("packed thread");
    println!("--- CANDLE_DEQUANTIZE_ALL=1 ---");
    let fallback = std::thread::spawn(move || bench_variant(true, iters));
    fallback.join().expect("fallback thread");
}

/// Streaming-read ceiling: how fast the pool can reduce a 252 MB buffer, the lm_head
/// footprint. Bounds what any matvec kernel can reach here. `--ignored`.
#[test]
#[ignore = "measurement"]
fn stream_reduce_rates() {
    let values = 63_000_000usize;
    let data = pseudo_f32(values, 3);
    let pool = candle_core::utils::barrier_pool();

    for _threads in [1usize, 16] {
        let started = std::time::Instant::now();
        let runs = 5;
        for _ in 0..runs {
            let acc_total = std::sync::atomic::AtomicU32::new(0);
            pool.execute_chunked(values / 4096, |range| {
                let start = range.start * 4096;
                let end = (range.end * 4096).min(values);
                let bits = data[start..end]
                    .iter()
                    .map(|v| v.to_bits())
                    .fold(0u64, |a, b| a.wrapping_add(b as u64));
                acc_total.fetch_add(bits as u32, std::sync::atomic::Ordering::Relaxed);
            });
            std::hint::black_box(acc_total.load(std::sync::atomic::Ordering::Relaxed));
        }
        let per = started.elapsed().as_secs_f64() / runs as f64;
        std::hint::black_box(&data);
        println!(
            "stream reduce {values} f32 ({} MiB): {:.2} ms, {:.1} GB/s (threads env)",
            values * 4 / 1_048_576,
            per * 1e3,
            values as f64 * 4.0 / per / 1e9,
        );
    }
}

fn q6k_tensor(n: usize, k: usize, seed: usize) -> (candle_core::quantized::QTensor, Vec<f32>) {
    let device = candle_core::Device::Cpu;
    let weight = pseudo_f32(n * k, seed);
    let src = candle_core::Tensor::from_vec(weight.clone(), (n, k), &device).expect("src");
    let qtensor =
        candle_core::quantized::QTensor::quantize(&src, candle_core::quantized::GgmlDType::Q6K)
            .expect("quantize");
    (qtensor, weight)
}

#[test]
fn q6k_gemv_matches_the_dequantized_reference() {
    use super::q6k_gemv;
    let (n, k) = (37usize, 512usize);
    let (qtensor, weight) = q6k_tensor(n, k, 11);
    let xs = pseudo_f32(k, 12);
    let bytes = qtensor.data().expect("bytes");

    let reference = qtensor
        .dequantize(&candle_core::Device::Cpu)
        .expect("dequantize")
        .matmul(
            &candle_core::Tensor::from_vec(xs.clone(), (k, 1), &candle_core::Device::Cpu)
                .expect("lhs"),
        )
        .expect("matmul")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec");

    let mut got = vec![0f32; n];
    assert!(q6k_gemv::gemv(n, k, &bytes, &xs, &mut got));
    // Candle's own packed path quantizes the lhs to q8 too; the dequantized f32 dot
    // carries the q8 rounding on top, so only the candle-vs-candle comparison is tight.
    let candle_q = candle_core::quantized::QMatMul::from_arc(std::sync::Arc::new(
        candle_core::quantized::QTensor::quantize(
            &candle_core::Tensor::from_vec(weight.clone(), (n, k), &candle_core::Device::Cpu)
                .expect("src"),
            candle_core::quantized::GgmlDType::Q6K,
        )
        .expect("quantize"),
    ))
    .expect("qmatmul")
    .forward(
        &candle_core::Tensor::from_vec(xs.clone(), (1, k), &candle_core::Device::Cpu).expect("x"),
    )
    .expect("candle forward")
    .flatten_all()
    .expect("flat")
    .to_vec1::<f32>()
    .expect("vec");
    for (r, (g, c)) in got.iter().zip(&candle_q).enumerate() {
        assert!(
            (g - c).abs() <= 1e-4 * (1.0 + c.abs()),
            "row {r}: gemv {g} against candle packed {c}"
        );
    }
    for (r, (g, e)) in got.iter().zip(&reference).enumerate() {
        assert!(
            (g - e).abs() <= 4e-2 * (1.0 + e.abs()),
            "row {r}: gemv {g} against dequantized {e}"
        );
    }

    let mut neon = vec![0f32; n];
    let k_blocks = k / 256;
    let q8 = q6k_gemv::test_quantize_row_q8k(&xs);
    for row in 0..n {
        let blocks = &bytes[row * k_blocks * q6k_gemv::Q6K_BLOCK_BYTES
            ..(row + 1) * k_blocks * q6k_gemv::Q6K_BLOCK_BYTES];
        let s = q6k_gemv::test_scalar_row_dot(blocks, k_blocks, &q8);
        #[cfg(target_arch = "aarch64")]
        let v = q6k_gemv::test_neon_row_dot(blocks, k_blocks, &q8);
        #[cfg(not(target_arch = "aarch64"))]
        let v = s;
        assert_eq!(v.to_bits(), s.to_bits(), "row {row}: neon against scalar");
        neon[row] = v;
    }
    for (r, (g, e)) in neon.iter().zip(&reference).enumerate() {
        assert!(
            (g - e).abs() <= 4e-2 * (1.0 + e.abs()),
            "row {r}: neon {g} against dequantized {e}"
        );
    }
}

/// Rate of the in-repo Q6K gemv against the packed candle path. `--ignored`.
#[test]
#[ignore = "measurement"]
fn q6k_ours_rate() {
    use super::q6k_gemv;
    let (n, k) = (150_144usize, 2_048usize);
    let (qtensor, _) = q6k_tensor(n, k, 5);
    let xs = pseudo_f32(k, 6);
    let bytes = qtensor.data().expect("bytes");
    let mut dst = vec![0f32; n];
    assert!(q6k_gemv::gemv(n, k, &bytes, &xs, &mut dst));

    let first = {
        let started = std::time::Instant::now();
        std::hint::black_box(q6k_gemv::gemv(n, k, &bytes, &xs, &mut dst));
        started.elapsed().as_secs_f64()
    };
    let iters = 20;
    let steady = {
        let started = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(q6k_gemv::gemv(n, k, &bytes, &xs, &mut dst));
        }
        started.elapsed().as_secs_f64() / iters as f64
    };
    let gbytes = n as f64 * k as f64 * 6.5625 / 8.0 / 1e9;
    println!(
        "ours     {n}x{k} Q6K first {:.2} ms, steady {:.3} ms, {:.1} GB/s",
        first * 1e3,
        steady * 1e3,
        gbytes / steady,
    );
}
