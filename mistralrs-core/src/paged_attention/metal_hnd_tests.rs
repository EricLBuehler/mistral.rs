use std::{collections::HashMap, time::Instant};

use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;
use mistralrs_paged_attn::{
    copy_blocks, gather_kv_cache, paged_attention_with_window, reshape_and_cache, swap_blocks,
};

const CACHE_VECTOR_BYTES: usize = 16;
const KEY_SCALE: f32 = 0.5;
const VALUE_SCALE: f32 = 0.25;
const BF16_TOLERANCE: f32 = 0.03;
const F16_TOLERANCE: f32 = 0.003;
const F32_TOLERANCE: f32 = 0.0002;
const EXTRA_SHORT_HEADS: [usize; 3] = [64, 96, 192];
const EXTRA_SHORT_LENGTHS: [&[usize]; 2] = [&[1], &[33]];
const EXCLUDED_VALUE_BIAS: f32 = 64.0;
const EXCLUDED_HEAD_BIAS: f32 = 4.0;
const WINDOW_VALUE_OSCILLATION_SCALE: f32 = 1.0 / 12.0;
const BENCH_DTYPE: DType = DType::BF16;
const BENCH_SHAPES: [(usize, usize, usize); 3] = [(16, 8, 128), (8, 2, 256), (8, 2, 512)];
const BENCH_BATCHES: [usize; 3] = [1, 4, 8];
const BENCH_CONTEXTS: [usize; 7] = [128, 256, 288, 320, 512, 1024, 2048];
const BENCH_BLOCK_SIZE: usize = 32;
const BENCH_ROUNDS: usize = 3;
const BENCH_WARMUP_ITERS: usize = 3;
const BENCH_TIMED_ITERS: usize = 10;

#[derive(Clone, Copy, Debug)]
struct Case<'a> {
    head_dim: usize,
    kv_heads: usize,
    query_heads: usize,
    block_size: usize,
    lengths: &'a [usize],
    window: Option<usize>,
    features: bool,
}

const CASES: &[Case<'static>] = &[
    Case {
        head_dim: 128,
        kv_heads: 8,
        query_heads: 16,
        block_size: 32,
        lengths: &[256],
        window: None,
        features: false,
    },
    Case {
        head_dim: 256,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[256],
        window: None,
        features: false,
    },
    Case {
        head_dim: 512,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[256],
        window: None,
        features: false,
    },
    Case {
        head_dim: 64,
        kv_heads: 2,
        query_heads: 32,
        block_size: 8,
        lengths: &[1, 9, 37],
        window: None,
        features: true,
    },
    Case {
        head_dim: 80,
        kv_heads: 2,
        query_heads: 6,
        block_size: 16,
        lengths: &[15, 17, 129],
        window: None,
        features: true,
    },
    Case {
        head_dim: 96,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[31, 513, 1025],
        window: None,
        features: false,
    },
    Case {
        head_dim: 112,
        kv_heads: 2,
        query_heads: 4,
        block_size: 8,
        lengths: &[7, 65, 513],
        window: None,
        features: true,
    },
    Case {
        head_dim: 192,
        kv_heads: 2,
        query_heads: 16,
        block_size: 32,
        lengths: &[1, 512, 1500],
        window: None,
        features: false,
    },
    Case {
        head_dim: 256,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[31, 513, 1500],
        window: Some(512),
        features: false,
    },
    Case {
        head_dim: 256,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[543],
        window: Some(512),
        features: false,
    },
    Case {
        head_dim: 512,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[37, 512, 1025],
        window: Some(65),
        features: true,
    },
    Case {
        head_dim: 256,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[1025],
        window: Some(31),
        features: false,
    },
    Case {
        head_dim: 256,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[31, 513, 1025],
        window: Some(31),
        features: false,
    },
    Case {
        head_dim: 256,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[513, 1025, 1537],
        window: Some(513),
        features: false,
    },
    Case {
        head_dim: 256,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[257],
        window: Some(1),
        features: false,
    },
    Case {
        head_dim: 512,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[513, 1025, 1537],
        window: Some(513),
        features: false,
    },
];

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let error = actual.iter().zip(expected).fold(0f32, |error, (a, b)| {
        let diff = (a - b).abs();
        if diff.is_finite() {
            error.max(diff)
        } else {
            f32::INFINITY
        }
    });
    assert!(
        error <= tolerance,
        "{label}: maximum absolute error {error}, tolerance {tolerance}"
    );
}

fn round_to(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => half::bf16::from_f32(value).to_f32(),
        DType::F16 => half::f16::from_f32(value).to_f32(),
        DType::F32 => value,
        _ => unreachable!(),
    }
}

struct Inputs {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    slots: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    cu_lengths: Tensor,
    q_cpu: Vec<f32>,
    k_cpu: Vec<f32>,
    v_cpu: Vec<f32>,
    blocks: usize,
    tokens: usize,
    scales: Option<(Tensor, Tensor)>,
}

impl Inputs {
    fn new(device: &Device, case: Case<'_>, dtype: DType, fp8: bool) -> Result<Self> {
        let batch = case.lengths.len();
        let tokens = case.lengths.iter().sum::<usize>();
        let max_blocks = case
            .lengths
            .iter()
            .copied()
            .max()
            .unwrap()
            .div_ceil(case.block_size);
        let blocks = batch * max_blocks;
        let mut q_data = vec![f32::NAN; batch * (case.query_heads + 1) * case.head_dim];
        let mut q_cpu = Vec::new();
        for sequence in 0..batch {
            for head in 0..case.query_heads {
                for dim in 0..case.head_dim {
                    let value =
                        round_to(((sequence * 13 + head * 7 + dim) as f32).sin() * 0.7, dtype);
                    q_data[(sequence * (case.query_heads + 1) + head + 1) * case.head_dim + dim] =
                        value;
                    q_cpu.push(value);
                }
            }
        }
        let query = Tensor::from_vec(
            q_data,
            (batch, case.query_heads + 1, case.head_dim),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?
        .to_device(device)?
        .narrow(1, 1, case.query_heads)?;
        let elements = tokens * case.kv_heads * case.head_dim;
        let source = |factor: usize| -> Vec<f32> {
            (0..elements)
                .map(|index| round_to(((index * factor) as f32).sin() * 0.6, dtype))
                .collect()
        };
        let k_input = source(3);
        let mut v_input = source(7);
        if let Some(window) = case.window {
            let mut token_base = 0;
            for &length in case.lengths {
                for token in 0..length {
                    let ramp = 2. * token as f32 / length.saturating_sub(1).max(1) as f32 - 1.;
                    for head in 0..case.kv_heads {
                        let start = ((token_base + token) * case.kv_heads + head) * case.head_dim;
                        let bias = if token < length.saturating_sub(window) {
                            EXCLUDED_VALUE_BIAS + head as f32 * EXCLUDED_HEAD_BIAS
                        } else {
                            0.
                        };
                        for value in &mut v_input[start..start + case.head_dim] {
                            *value = round_to(
                                ramp + *value * WINDOW_VALUE_OSCILLATION_SCALE + bias,
                                dtype,
                            );
                        }
                    }
                }
                token_base += length;
            }
        }
        let quantized = |input: &[f32], scale: f32| -> Vec<f32> {
            input
                .iter()
                .map(|&value| {
                    if fp8 {
                        round_to(
                            F8E4M3::from_f32(round_to(value / scale, dtype)).to_f32() * scale,
                            dtype,
                        )
                    } else {
                        value
                    }
                })
                .collect()
        };
        let k_cpu = quantized(&k_input, KEY_SCALE);
        let v_cpu = quantized(&v_input, VALUE_SCALE);
        let shape = (tokens, case.kv_heads, case.head_dim);
        let key = Tensor::from_vec(k_input, shape, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;
        let value = Tensor::from_vec(v_input, shape, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;
        let mut slot_data = vec![-1i64];
        let mut tables = vec![0u32; (batch + 1) * max_blocks];
        let mut cumulative = vec![0u32];
        for (sequence, &length) in case.lengths.iter().enumerate() {
            for page in 0..max_blocks {
                tables[(sequence + 1) * max_blocks + page] =
                    (blocks - 1 - (sequence * max_blocks + page)) as u32;
            }
            for token in 0..length {
                let physical =
                    tables[(sequence + 1) * max_blocks + token / case.block_size] as usize;
                slot_data.push((physical * case.block_size + token % case.block_size) as i64);
            }
            cumulative.push(cumulative.last().unwrap() + length as u32);
        }
        let slots = Tensor::from_vec(slot_data, tokens + 1, device)?.narrow(0, 1, tokens)?;
        let block_tables =
            Tensor::from_vec(tables, (batch + 1, max_blocks), device)?.narrow(0, 1, batch)?;
        let mut lengths = vec![0u32];
        lengths.extend(case.lengths.iter().map(|&length| length as u32));
        let context_lens = Tensor::new(lengths, device)?.narrow(0, 1, batch)?;
        let cu_lengths = Tensor::new(cumulative, device)?;
        let scales = if fp8 {
            Some((
                Tensor::new(&[KEY_SCALE], device)?,
                Tensor::new(&[VALUE_SCALE], device)?,
            ))
        } else {
            None
        };
        Ok(Self {
            query,
            key,
            value,
            slots,
            block_tables,
            context_lens,
            cu_lengths,
            q_cpu,
            k_cpu,
            v_cpu,
            blocks,
            tokens,
            scales,
        })
    }

    fn cache(
        &self,
        device: &Device,
        case: Case<'_>,
        dtype: DType,
        hnd: bool,
    ) -> Result<(Tensor, Tensor)> {
        let pack = CACHE_VECTOR_BYTES / dtype.size_in_bytes();
        let key_shape = if hnd {
            vec![
                self.blocks + 1,
                case.kv_heads,
                case.block_size,
                case.head_dim,
            ]
        } else {
            vec![
                self.blocks + 1,
                case.kv_heads,
                case.head_dim / pack,
                case.block_size,
                pack,
            ]
        };
        let value_shape = if hnd {
            key_shape.clone()
        } else {
            vec![
                self.blocks + 1,
                case.kv_heads,
                case.head_dim,
                case.block_size,
            ]
        };
        let cache = |shape: Vec<usize>| -> Result<Tensor> {
            Tensor::full(f32::NAN, shape, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(device)?
                .narrow(0, 1, self.blocks)
        };
        let key = cache(key_shape)?;
        let value = cache(value_shape)?;
        let (k_scale, v_scale) = self
            .scales
            .as_ref()
            .map_or((None, None), |(k, v)| (Some(k), Some(v)));
        reshape_and_cache(
            &self.key,
            &self.value,
            k_scale,
            v_scale,
            &key,
            &value,
            &self.slots,
        )?;
        let (gathered_k, gathered_v) = gather_kv_cache(
            &key,
            &value,
            k_scale,
            v_scale,
            &self.block_tables,
            &self.cu_lengths,
            self.tokens,
            self.query.dtype(),
        )?;
        let label = format!(
            "{case:?} input={:?} cache={dtype:?} hnd={hnd}",
            self.query.dtype()
        );
        assert_close(
            &values(&gathered_k)?,
            &self.k_cpu,
            0.,
            &format!("gathered K {label}"),
        );
        assert_close(
            &values(&gathered_v)?,
            &self.v_cpu,
            0.,
            &format!("gathered V {label}"),
        );
        Ok((key, value))
    }

    fn expected(&self, case: Case<'_>) -> Vec<f32> {
        let mut output = vec![0f32; self.q_cpu.len()];
        let scale = 1. / (case.head_dim as f64).sqrt();
        let mut token_base = 0;
        for (sequence, &length) in case.lengths.iter().enumerate() {
            let start = case
                .window
                .map_or(0, |window| length.saturating_sub(window));
            for head in 0..case.query_heads {
                let kv_head = head / (case.query_heads / case.kv_heads);
                let q_base = (sequence * case.query_heads + head) * case.head_dim;
                let mut logits = Vec::new();
                for token in start..length {
                    let kv_base = ((token_base + token) * case.kv_heads + kv_head) * case.head_dim;
                    let mut logit = (0..case.head_dim)
                        .map(|dim| {
                            self.q_cpu[q_base + dim] as f64 * self.k_cpu[kv_base + dim] as f64
                        })
                        .sum::<f64>()
                        * scale;
                    if case.features {
                        logit = (logit / 2.).tanh() * 2.;
                        logit +=
                            (0.01 * (head + 1) as f32) as f64 * (token as f64 - length as f64 + 1.);
                    }
                    logits.push(logit);
                }
                let sink = (0.1 + head as f32 * 0.05) as f64;
                let max = logits.iter().copied().fold(
                    if case.features {
                        sink
                    } else {
                        f64::NEG_INFINITY
                    },
                    f64::max,
                );
                let weights = logits
                    .iter()
                    .map(|logit| (logit - max).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f64>()
                    + if case.features {
                        (sink - max).exp()
                    } else {
                        0.
                    };
                for dim in 0..case.head_dim {
                    output[q_base + dim] = (weights
                        .iter()
                        .enumerate()
                        .map(|(offset, weight)| {
                            let kv_base = ((token_base + start + offset) * case.kv_heads + kv_head)
                                * case.head_dim;
                            weight * self.v_cpu[kv_base + dim] as f64
                        })
                        .sum::<f64>()
                        / denominator) as f32;
                }
            }
            token_base += length;
        }
        output
    }
}

fn run_case(device: &Device, case: Case<'_>, dtype: DType, fp8: bool) -> Result<()> {
    let inputs = Inputs::new(device, case, dtype, fp8)?;
    let expected = inputs.expected(case);
    let alibi = case
        .features
        .then(|| {
            Tensor::from_vec(
                (0..case.query_heads)
                    .map(|h| 0.01 * (h + 1) as f32)
                    .collect::<Vec<_>>(),
                case.query_heads,
                device,
            )
        })
        .transpose()?;
    let sinks = case
        .features
        .then(|| {
            Tensor::from_vec(
                (0..case.query_heads)
                    .map(|h| 0.1 + h as f32 * 0.05)
                    .collect::<Vec<_>>(),
                case.query_heads,
                device,
            )
        })
        .transpose()?;
    let (k_scale, v_scale) = inputs
        .scales
        .as_ref()
        .map_or((None, None), |(k, v)| (Some(k), Some(v)));
    let tolerance = match dtype {
        DType::BF16 => BF16_TOLERANCE,
        DType::F16 => F16_TOLERANCE,
        _ => F32_TOLERANCE,
    };
    for hnd in [true, false] {
        if !hnd && case.window.is_some() {
            continue;
        }
        let cache_dtype = if fp8 { DType::F8E4M3 } else { dtype };
        let (key, value) = inputs.cache(device, case, cache_dtype, hnd)?;
        let result = paged_attention_with_window(
            &inputs.query,
            k_scale,
            v_scale,
            &key,
            &value,
            &inputs.block_tables,
            &inputs.context_lens,
            alibi.as_ref(),
            *case.lengths.iter().max().unwrap(),
            1. / (case.head_dim as f32).sqrt(),
            if case.features { 2. } else { 1. },
            sinks.as_ref(),
            case.window,
        )?;
        assert_close(
            &values(&result)?,
            &expected,
            tolerance,
            &format!("{case:?} {dtype:?} fp8={fp8} hnd={hnd}"),
        );
    }
    Ok(())
}

#[test]
fn metal_hnd_decode_matches_cpu_with_ragged_pages_and_features() -> Result<()> {
    let device = Device::new_metal(0)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        for &case in CASES {
            run_case(&device, case, dtype, false)?;
        }
        for head_dim in EXTRA_SHORT_HEADS {
            for lengths in EXTRA_SHORT_LENGTHS {
                run_case(
                    &device,
                    Case {
                        head_dim,
                        lengths,
                        ..CASES[1]
                    },
                    dtype,
                    false,
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn metal_hnd_fp8_cache_matches_cpu() -> Result<()> {
    let device = Device::new_metal(0)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        for &case in &[
            CASES[0], CASES[2], CASES[4], CASES[8], CASES[10], CASES[12], CASES[13],
        ] {
            run_case(&device, case, dtype, true)?;
        }
    }
    Ok(())
}

#[test]
fn metal_fp8_cache_encoding_rounds_and_saturates_exactly() -> Result<()> {
    const HEAD_DIM: usize = 32;
    const BLOCK_SIZE: usize = 32;
    const FP8_MAX_CODE: u8 = 0x7e;
    const FP8_NAN_CODE: u8 = 0x7f;
    const FP8_MAX: f32 = 448.;

    let device = Device::new_metal(0)?;
    let scale = Tensor::new(&[1f32], &device)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let mut data = Vec::new();
        for code in 0..=FP8_MAX_CODE {
            let value = F8E4M3::from_bits(code).to_f32();
            data.extend([value, -value]);
            if code == FP8_MAX_CODE {
                continue;
            }
            let midpoint = (value + F8E4M3::from_bits(code + 1).to_f32()) / 2.;
            let neighbors = match dtype {
                DType::BF16 => {
                    let bits = half::bf16::from_f32(midpoint).to_bits();
                    [
                        half::bf16::from_bits(bits - 1).to_f32(),
                        midpoint,
                        half::bf16::from_bits(bits + 1).to_f32(),
                    ]
                }
                DType::F16 => {
                    let bits = half::f16::from_f32(midpoint).to_bits();
                    [
                        half::f16::from_bits(bits - 1).to_f32(),
                        midpoint,
                        half::f16::from_bits(bits + 1).to_f32(),
                    ]
                }
                _ => [
                    f32::from_bits(midpoint.to_bits() - 1),
                    midpoint,
                    f32::from_bits(midpoint.to_bits() + 1),
                ],
            };
            for value in neighbors {
                data.extend([value, -value]);
            }
        }
        data.extend([
            480.,
            -480.,
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
        ]);
        let blocks = data.len().div_ceil(BLOCK_SIZE * HEAD_DIM);
        data.resize(blocks * BLOCK_SIZE * HEAD_DIM, 0.);
        let tokens = blocks * BLOCK_SIZE;
        let expected = data
            .iter()
            .map(|&value| {
                let value = round_to(value, dtype);
                if value.is_nan() {
                    FP8_NAN_CODE
                } else {
                    F8E4M3::from_f32(value.clamp(-FP8_MAX, FP8_MAX)).to_bits()
                }
            })
            .collect::<Vec<_>>();
        let key = Tensor::from_vec(data.clone(), (tokens, 1, HEAD_DIM), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(&device)?;
        data.reverse();
        let value = Tensor::from_vec(data, (tokens, 1, HEAD_DIM), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(&device)?;
        let slots = Tensor::new((0..tokens as i64).collect::<Vec<_>>(), &device)?;
        let tables = Tensor::new((0..blocks as u32).collect::<Vec<_>>(), &device)?.unsqueeze(0)?;
        let cu_lengths = Tensor::new(&[0u32, tokens as u32], &device)?;
        for hnd in [false, true] {
            let key_shape = if hnd {
                vec![blocks, 1, BLOCK_SIZE, HEAD_DIM]
            } else {
                vec![
                    blocks,
                    1,
                    HEAD_DIM / CACHE_VECTOR_BYTES,
                    BLOCK_SIZE,
                    CACHE_VECTOR_BYTES,
                ]
            };
            let value_shape = if hnd {
                key_shape.clone()
            } else {
                vec![blocks, 1, HEAD_DIM, BLOCK_SIZE]
            };
            let key_cache =
                Tensor::zeros(key_shape, DType::F8E4M3, &Device::Cpu)?.to_device(&device)?;
            let value_cache =
                Tensor::zeros(value_shape, DType::F8E4M3, &Device::Cpu)?.to_device(&device)?;
            reshape_and_cache(
                &key,
                &value,
                Some(&scale),
                Some(&scale),
                &key_cache,
                &value_cache,
                &slots,
            )?;
            let key_cpu = key_cache.to_device(&Device::Cpu)?;
            let value_cpu = value_cache.to_device(&Device::Cpu)?;
            let caches = if hnd {
                [key_cpu, value_cpu]
            } else {
                [
                    key_cpu.permute((0, 1, 3, 2, 4))?.contiguous()?,
                    value_cpu.transpose(2, 3)?.contiguous()?,
                ]
            };
            let mut expected_value = expected.clone();
            expected_value.reverse();
            for (name, cache, expected) in [
                ("K", &caches[0], &expected),
                ("V", &caches[1], &expected_value),
            ] {
                let actual = cache.flatten_all()?.to_vec1::<F8E4M3>()?;
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        *expected,
                        "{dtype:?} hnd={hnd} {name}[{index}]: FP8 encoding"
                    );
                }
            }
            let (gathered_key, gathered_value) = gather_kv_cache(
                &key_cache,
                &value_cache,
                Some(&scale),
                Some(&scale),
                &tables,
                &cu_lengths,
                tokens,
                dtype,
            )?;
            for (name, gathered, expected) in [
                ("K", gathered_key, &expected),
                ("V", gathered_value, &expected_value),
            ] {
                for (index, (actual, expected)) in
                    values(&gathered)?.iter().zip(expected).enumerate()
                {
                    let expected = F8E4M3::from_bits(*expected).to_f32();
                    assert!(if expected.is_nan() { actual.is_nan() } else { *actual == expected }, "{dtype:?} hnd={hnd} {name}[{index}]: gathered {actual}, expected {expected}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn metal_hnd_block_copies_preserve_offsets_and_other_blocks() -> Result<()> {
    const BLOCKS: usize = 3;
    const ELEMENTS_PER_BLOCK: usize = 32;
    let device = Device::new_metal(0)?;
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F8E4M3] {
        let data = (0..(BLOCKS + 1) * ELEMENTS_PER_BLOCK)
            .map(|i| (i % 16) as f32 + (i / ELEMENTS_PER_BLOCK) as f32)
            .collect::<Vec<_>>();
        let source_storage = Tensor::from_vec(data, (BLOCKS + 1, 2, 4, 4), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(&device)?;
        // Metal Tensor::copy shares storage; cache writes need independent backing buffers.
        let key_storage = source_storage.force_contiguous()?;
        let value_storage = source_storage.force_contiguous()?;
        let source = source_storage.narrow(0, 1, BLOCKS)?;
        let mut key = key_storage.narrow(0, 1, BLOCKS)?;
        let mut value = value_storage.narrow(0, 1, BLOCKS)?;
        assert_eq!(key.layout().start_offset(), ELEMENTS_PER_BLOCK);
        assert_eq!(value.layout().start_offset(), ELEMENTS_PER_BLOCK);
        let full_before = values(&source_storage)?;
        let before = values(&key)?;
        copy_blocks(
            vec![&mut key],
            vec![&mut value],
            &HashMap::from([(0, vec![1, 2])]),
        )?;
        let mut expected = before.clone();
        for block in [1, 2] {
            expected[block * ELEMENTS_PER_BLOCK..(block + 1) * ELEMENTS_PER_BLOCK]
                .copy_from_slice(&before[..ELEMENTS_PER_BLOCK]);
        }
        assert_close(&values(&key)?, &expected, 0., "copied K blocks");
        assert_close(&values(&value)?, &expected, 0., "copied V blocks");
        let mut full_expected = full_before.clone();
        for block in [2, 3] {
            full_expected[block * ELEMENTS_PER_BLOCK..(block + 1) * ELEMENTS_PER_BLOCK]
                .copy_from_slice(&full_before[ELEMENTS_PER_BLOCK..2 * ELEMENTS_PER_BLOCK]);
        }
        assert_close(
            &values(&key_storage)?,
            &full_expected,
            0.,
            "copied K guard block",
        );
        assert_close(
            &values(&value_storage)?,
            &full_expected,
            0.,
            "copied V guard block",
        );
        for cpu_source in [false, true] {
            let swap_source = if cpu_source {
                source_storage
                    .to_device(&Device::Cpu)?
                    .narrow(0, 1, BLOCKS)?
            } else {
                source.clone()
            };
            assert_eq!(swap_source.layout().start_offset(), ELEMENTS_PER_BLOCK);
            let destination_storage = Tensor::zeros((BLOCKS + 1, 2, 4, 4), dtype, &device)?;
            let destination = destination_storage.narrow(0, 1, BLOCKS)?;
            unsafe {
                swap_blocks(swap_source, &destination, HashMap::from([(1, 2)]))?;
            }
            let mut expected_swap = vec![0.; BLOCKS * ELEMENTS_PER_BLOCK];
            expected_swap[2 * ELEMENTS_PER_BLOCK..]
                .copy_from_slice(&before[ELEMENTS_PER_BLOCK..2 * ELEMENTS_PER_BLOCK]);
            assert_close(&values(&destination)?, &expected_swap, 0., "swapped blocks");
            let mut full_expected_swap = vec![0.; (BLOCKS + 1) * ELEMENTS_PER_BLOCK];
            full_expected_swap[ELEMENTS_PER_BLOCK..].copy_from_slice(&expected_swap);
            assert_close(
                &values(&destination_storage)?,
                &full_expected_swap,
                0.,
                "swapped guard block",
            );
        }
        assert_close(
            &values(&source_storage)?,
            &full_before,
            0.,
            "swap source unchanged",
        );
    }
    Ok(())
}

#[test]
#[ignore = "benchmark; needs exclusive use of a Metal GPU"]
fn metal_hnd_decode_bench() -> Result<()> {
    let device = Device::new_metal(0)?;
    println!(
        "HND vs legacy cache layout: {BENCH_DTYPE:?}, host dispatch + allocation + synchronized GPU completion; excludes cache packing"
    );
    println!(
        "kernel override={:?}, warmup={BENCH_WARMUP_ITERS}, timed={BENCH_TIMED_ITERS}, rotating rounds={BENCH_ROUNDS}",
        std::env::var("MISTRALRS_METAL_PAGED_KERNEL").ok()
    );
    println!("q_heads,kv_heads,head_dim,batch,context,round,hnd_ms,legacy_ms,legacy_over_hnd");
    for (query_heads, kv_heads, head_dim) in BENCH_SHAPES {
        for batch in BENCH_BATCHES {
            for context in BENCH_CONTEXTS {
                let lengths = vec![context; batch];
                let case = Case {
                    head_dim,
                    kv_heads,
                    query_heads,
                    block_size: BENCH_BLOCK_SIZE,
                    lengths: &lengths,
                    window: None,
                    features: false,
                };
                let inputs = Inputs::new(&device, case, BENCH_DTYPE, false)?;
                let caches = [
                    inputs.cache(&device, case, BENCH_DTYPE, true)?,
                    inputs.cache(&device, case, BENCH_DTYPE, false)?,
                ];
                let decode = |layout: usize| {
                    let (key, value) = &caches[layout];
                    paged_attention_with_window(
                        &inputs.query,
                        None,
                        None,
                        key,
                        value,
                        &inputs.block_tables,
                        &inputs.context_lens,
                        None,
                        context,
                        1. / (head_dim as f32).sqrt(),
                        1.,
                        None,
                        None,
                    )
                };
                let expected = inputs.expected(case);
                for layout in 0..caches.len() {
                    assert_close(
                        &values(&decode(layout)?)?,
                        &expected,
                        BF16_TOLERANCE,
                        &format!("benchmark {case:?} hnd={}", layout == 0),
                    );
                    for _ in 0..BENCH_WARMUP_ITERS {
                        decode(layout)?;
                    }
                }
                device.synchronize()?;
                let mut timings = [[0f64; BENCH_ROUNDS]; 2];
                for round in 0..BENCH_ROUNDS {
                    for index in 0..caches.len() {
                        let layout = (index + round) % caches.len();
                        device.synchronize()?;
                        let start = Instant::now();
                        for _ in 0..BENCH_TIMED_ITERS {
                            decode(layout)?;
                        }
                        device.synchronize()?;
                        timings[layout][round] =
                            start.elapsed().as_secs_f64() * 1e3 / BENCH_TIMED_ITERS as f64;
                    }
                    let (hnd, legacy) = (timings[0][round], timings[1][round]);
                    println!(
                        "{query_heads},{kv_heads},{head_dim},{batch},{context},{round},{hnd:.6},{legacy:.6},{:.4}",
                        legacy / hnd,
                    );
                }
                for times in &mut timings {
                    times.sort_by(f64::total_cmp);
                }
                let (hnd, legacy) = (timings[0][BENCH_ROUNDS / 2], timings[1][BENCH_ROUNDS / 2]);
                println!(
                    "{query_heads},{kv_heads},{head_dim},{batch},{context},median,{hnd:.6},{legacy:.6},{:.4}",
                    legacy / hnd,
                );
            }
        }
    }
    Ok(())
}
