use candle_core::Device;

use super::*;

const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const LCG_MUL: u64 = 6364136223846793005;
const LCG_INC: u64 = 1442695040888963407;
const SCALE_VALUES: [f32; 6] = [0.001953125, 0.0546875, 0.3125, 1.5, 2.75, 13.0];
const GLOBAL_VALUES: [f32; 3] = [0.03125, 0.09375, 0.5];
const DENSE_SHAPES: [(usize, usize); 5] = [(5, 16), (77, 80), (200, 208), (130, 1024), (256, 1040)];
const GROUPED_SHAPES: [(usize, usize); 3] = [(77, 80), (200, 1040), (128, 256)];
const TOKENS: [usize; 4] = [1, 17, 100, 512];
const EXPERTS: usize = 5;
// The last expert never receives a route.
const USED_EXPERTS: usize = EXPERTS - 1;
const TOPKS: [usize; 2] = [1, 3];
const BF16_TOLERANCE: f64 = 1e-2;
const F16_TOLERANCE: f64 = 3e-3;
const FP4_CODE_COUNT: usize = 16;
const FINITE_POSITIVE_SCALE_COUNT: usize = 127;
const FULL_RANGE_GLOBAL: f32 = 1.0 / 1024.0;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
        self.0 >> 33
    }
}

struct Case {
    weight: Tensor,
    scales: Tensor,
    global: Tensor,
    // [E, N, K] f32 including the global multiplier
    dequantized: Vec<f32>,
}

fn case(experts: usize, n: usize, k: usize, varied_global: bool, device: &Device) -> Result<Case> {
    let mut rng = Rng(SEED ^ (experts * 131 + n * 7 + k) as u64);
    let codes = (0..experts * n * k / 2)
        .map(|_| rng.next() as u8)
        .collect::<Vec<_>>();
    let scales = (0..experts * n * k / NVFP4_BLOCK_SIZE)
        .map(|_| F8E4M3::from_f32(SCALE_VALUES[rng.next() as usize % SCALE_VALUES.len()]))
        .collect::<Vec<_>>();
    let global = (0..experts * n)
        .map(|i| {
            if varied_global {
                GLOBAL_VALUES[i % GLOBAL_VALUES.len()]
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();
    let mut dequantized = Vec::with_capacity(experts * n * k);
    for byte in &codes {
        for nibble in [byte & 15, byte >> 4] {
            let value = dequantized.len();
            let row = value / k;
            dequantized.push(
                super::super::unpack_fp4(nibble)
                    * scales[value / NVFP4_BLOCK_SIZE].to_f32()
                    * global[row],
            );
        }
    }
    let shape = |last: usize| -> Vec<usize> {
        if experts == 1 {
            vec![n, last]
        } else {
            vec![experts, n, last]
        }
    };
    let global_shape = if experts == 1 {
        vec![n]
    } else {
        vec![experts, n]
    };
    Ok(Case {
        weight: Tensor::from_vec(codes, shape(k / 2), device)?,
        scales: Tensor::from_vec(scales, shape(k / NVFP4_BLOCK_SIZE), device)?,
        global: Tensor::from_vec(global, global_shape, device)?,
        dequantized,
    })
}

fn input(rows: usize, k: usize, dtype: DType, device: &Device) -> Result<(Tensor, Vec<f32>)> {
    let x = Tensor::randn(0f32, 1f32, (rows, k), &Device::Cpu)?.to_dtype(dtype)?;
    let values = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok((x.to_device(device)?, values))
}

fn dot(x: &[f32], w: &[f32]) -> f32 {
    x.iter().zip(w).map(|(a, b)| a * b).sum()
}

fn rel_err(actual: &Tensor, expected: &[f32]) -> Result<f64> {
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    let (mut diff, mut norm) = (0f64, 0f64);
    for (a, e) in actual.iter().zip(expected) {
        assert!(a.is_finite(), "non-finite output {a}");
        diff += f64::from(a - e).powi(2);
        norm += f64::from(*e).powi(2);
    }
    Ok((diff / norm.max(f64::MIN_POSITIVE)).sqrt())
}

fn tolerance(dtype: DType) -> f64 {
    if dtype == DType::BF16 {
        BF16_TOLERANCE
    } else {
        F16_TOLERANCE
    }
}

fn cuda() -> Result<Option<Device>> {
    let device = Device::new_cuda(0)?;
    Ok(supported(device.as_cuda_device()?).then_some(device))
}

#[test]
fn dense_matches_dequantized_reference() -> Result<()> {
    let Some(device) = cuda()? else {
        return Ok(());
    };
    for (n, k) in DENSE_SHAPES {
        for varied_global in [false, true] {
            let case = case(1, n, k, varied_global, &device)?;
            for dtype in [DType::BF16, DType::F16] {
                for rows in TOKENS {
                    let (x, values) = input(rows, k, dtype, &device)?;
                    let y = matmul(
                        &x,
                        Weights {
                            weight: &case.weight,
                            scales: &case.scales,
                            global: &case.global,
                        },
                    )?;
                    assert_eq!(y.dims(), [rows, n]);
                    assert_eq!(y.dtype(), dtype);
                    let mut expected = Vec::with_capacity(rows * n);
                    for row in values.chunks(k) {
                        for weights in case.dequantized.chunks(k) {
                            expected.push(dot(row, weights));
                        }
                    }
                    let err = rel_err(&y, &expected)?;
                    assert!(
                        err <= tolerance(dtype),
                        "dense n={n} k={k} rows={rows} {dtype:?} global={varied_global}: rel err {err}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn grouped_matches_dequantized_reference() -> Result<()> {
    let Some(device) = cuda()? else {
        return Ok(());
    };
    for (n, k) in GROUPED_SHAPES {
        for varied_global in [false, true] {
            let case = case(EXPERTS, n, k, varied_global, &device)?;
            for dtype in [DType::BF16, DType::F16] {
                for tokens in TOKENS {
                    for topk in TOPKS {
                        for shared_input in [true, false] {
                            let routes = tokens * topk;
                            let input_rows = if shared_input { tokens } else { routes };
                            let (x, values) = input(input_rows, k, dtype, &device)?;
                            let x = x.reshape((tokens, input_rows / tokens, k))?;
                            let ids = (0..routes)
                                .map(|route| {
                                    let (token, slot) = (route / topk, route % topk);
                                    ((token * 3 + slot) % USED_EXPERTS) as u32
                                })
                                .collect::<Vec<_>>();
                            let indices = Tensor::from_vec(ids.clone(), (tokens, topk), &device)?;
                            let y = gather(
                                &x,
                                &indices,
                                Weights {
                                    weight: &case.weight,
                                    scales: &case.scales,
                                    global: &case.global,
                                },
                            )?;
                            assert_eq!(y.dims(), [tokens, topk, n]);
                            let mut expected = Vec::with_capacity(routes * n);
                            for (route, &expert) in ids.iter().enumerate() {
                                let row = if shared_input { route / topk } else { route };
                                let row = &values[row * k..(row + 1) * k];
                                let expert = expert as usize * n * k;
                                for weights in case.dequantized[expert..expert + n * k].chunks(k) {
                                    expected.push(dot(row, weights));
                                }
                            }
                            let err = rel_err(&y, &expected)?;
                            assert!(
                                err <= tolerance(dtype),
                                "grouped n={n} k={k} tokens={tokens} topk={topk} shared={shared_input} {dtype:?} global={varied_global}: rel err {err}"
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn finite_scale_range_matches_exact_dense_and_routed_reference() -> Result<()> {
    let Some(device) = cuda()? else {
        return Ok(());
    };
    let n = FP4_CODE_COUNT * FINITE_POSITIVE_SCALE_COUNT;
    let k = NVFP4_BLOCK_SIZE;
    let mut packed = Vec::with_capacity(n * k / 2);
    let mut scales = Vec::with_capacity(n);
    let mut expected = Vec::with_capacity(n);
    for bits in 0..FINITE_POSITIVE_SCALE_COUNT {
        let scale = F8E4M3::from_bits(bits as u8);
        for code in 0..FP4_CODE_COUNT {
            packed.extend(std::iter::repeat_n((code | code << 4) as u8, k / 2));
            scales.push(scale);
            expected.push(
                super::super::unpack_fp4(code as u8)
                    * scale.to_f32()
                    * k as f32
                    * FULL_RANGE_GLOBAL,
            );
        }
    }
    let weight = Tensor::from_vec(packed, (n, k / 2), &device)?;
    let scales = Tensor::from_vec(scales, (n, 1), &device)?;
    let global = Tensor::from_vec(vec![FULL_RANGE_GLOBAL; n], n, &device)?;
    let grouped_weight = Tensor::stack(&[&weight, &weight], 0)?;
    let grouped_scales = Tensor::stack(&[&scales, &scales], 0)?;
    let grouped_global = Tensor::stack(&[&global, &(&global * 2.0)?], 0)?;
    let indices = Tensor::from_vec(vec![1u32, 0], (1, 2), &device)?;
    let expected_grouped = expected
        .iter()
        .map(|value| value * 2.0)
        .chain(expected.iter().copied())
        .collect::<Vec<_>>();
    for dtype in [DType::BF16, DType::F16] {
        let x = Tensor::ones((1, k), dtype, &device)?;
        for (actual, expected) in [
            (
                matmul(
                    &x,
                    Weights {
                        weight: &weight,
                        scales: &scales,
                        global: &global,
                    },
                )?,
                &expected,
            ),
            (
                gather(
                    &x.reshape((1, 1, k))?,
                    &indices,
                    Weights {
                        weight: &grouped_weight,
                        scales: &grouped_scales,
                        global: &grouped_global,
                    },
                )?,
                &expected_grouped,
            ),
        ] {
            let expected = Tensor::from_vec(expected.clone(), expected.len(), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            assert_eq!(
                actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected,
                "{dtype:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn misaligned_views_are_realigned() -> Result<()> {
    let Some(device) = cuda()? else {
        return Ok(());
    };
    const ROWS: usize = 3;
    const K: usize = 48;
    let flat = Tensor::randn(0f32, 1f32, ROWS * K + 1, &device)?.to_dtype(DType::BF16)?;
    let view = flat.narrow(0, 1, ROWS * K)?.reshape((ROWS, K))?;
    let aligned = realigned(&view, ACTIVATION_ALIGNMENT)?;
    assert_eq!(aligned.layout().start_offset(), 0);
    assert_eq!(
        aligned.to_dtype(DType::F32)?.to_vec2::<f32>()?,
        view.to_dtype(DType::F32)?.to_vec2::<f32>()?
    );
    Ok(())
}

#[test]
fn split_k_partitions_cover_every_tile() {
    const TARGET: usize = 96;
    for blocks in [1, 2, 7, 95, 96, 400] {
        for k_tiles in [1, 3, 4, 16, 17, 64] {
            let (splits, per_split) = split_k(blocks, k_tiles, TARGET);
            assert!((1..=MAX_SPLITS).contains(&splits));
            assert!(splits * per_split >= k_tiles);
            assert!((splits - 1) * per_split < k_tiles);
        }
    }
}

const BENCH_DENSE: [(usize, usize, usize); 6] = [
    (1, 12288, 4096),
    (8, 12288, 4096),
    (1, 4096, 12288),
    (17, 12288, 4096),
    (512, 12288, 4096),
    (2048, 12288, 4096),
];
const BENCH_ITERATIONS: usize = 10;
const BENCH_SAMPLES: usize = 7;
const NVFP4_BYTES_PER_WEIGHT: f64 = 0.5625;

// Best of several samples: the GPU is shared, so the minimum is the least contended estimate.
fn time_ms(device: &Device, mut run: impl FnMut() -> Result<Tensor>) -> Result<f64> {
    run()?;
    device.synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..BENCH_SAMPLES {
        let start = std::time::Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            std::hint::black_box(run()?);
        }
        device.synchronize()?;
        best = best.min(start.elapsed().as_secs_f64() * 1e3 / BENCH_ITERATIONS as f64);
    }
    Ok(best)
}

#[test]
#[ignore = "benchmark"]
fn bench_dense_vs_bf16() -> Result<()> {
    let Some(device) = cuda()? else {
        return Ok(());
    };
    for (rows, n, k) in BENCH_DENSE {
        let case = case(1, n, k, true, &device)?;
        let (x, _) = input(rows, k, DType::BF16, &device)?;
        let weights = Weights {
            weight: &case.weight,
            scales: &case.scales,
            global: &case.global,
        };
        let ms = time_ms(&device, || matmul(&x, weights))?;
        let dense = Tensor::randn(0f32, 1f32, (n, k), &device)?.to_dtype(DType::BF16)?;
        let bf16 = time_ms(&device, || x.matmul(&dense.t()?))?;
        let gbs = (n * k) as f64 * NVFP4_BYTES_PER_WEIGHT / (ms * 1e-3) / 1e9;
        println!(
            "rows={rows} n={n} k={k}: w4a16 {ms:.3} ms ({gbs:.0} GB/s), bf16 {bf16:.3} ms, ratio {:.2}x",
            bf16 / ms
        );
    }
    Ok(())
}
