#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::{collections::HashMap, time::Instant};

use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;
use mistralrs_paged_attn::{
    copy_blocks, gather_kv_cache, metal_paged_prefill, paged_attention_with_window,
    reshape_and_cache, swap_blocks, MetalPagedPrefillRequest,
};

use crate::{
    attention::{AttentionMask, Sdpa, SdpaParams},
    pipeline::text_models_inputs_processor::PagedAttentionInputMetadata,
};

use super::{Fp8AttentionScales, PagedAttention};

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
const DONOR_QUERY_LENGTHS: [&[usize]; 3] =
    [&[224], &[543], &[31, 64, 224, 511, 512, 513, 529, 543]];
const DONOR_CROPPED_BLOCKS: usize = 7;
const PREFILL_BENCH_FUSED_HEAD_DIM: usize = 512;
const PREFILL_BENCH_HEAD_DIMS: [usize; 3] = [128, 256, PREFILL_BENCH_FUSED_HEAD_DIM];
const PREFILL_BENCH_QUERY_HEADS: usize = 8;
const PREFILL_BENCH_KV_HEADS: usize = 2;
const PREFILL_BENCH_QUERY_LEN: usize = 512;
const PREFILL_BENCH_CONTEXT_LEN: usize = 4096;
const PREFILL_BENCH_TOLERANCE: f32 = 0.006;
const PREFILL_BENCH_ITERS_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_ITERS";

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
        head_dim: 512,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[1025],
        window: None,
        features: false,
    },
    Case {
        head_dim: 512,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[2049],
        window: None,
        features: false,
    },
    Case {
        head_dim: 512,
        kv_heads: 2,
        query_heads: 8,
        block_size: 32,
        lengths: &[4096],
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
fn metal_hnd_reduced_donor_prefill_uses_pages_without_gathering() -> Result<()> {
    let device = Device::new_metal(0)?;
    let location = device.location();
    for head_dim in [256, 512] {
        for lengths in DONOR_QUERY_LENGTHS {
            for window in [None, Some(512)] {
                let case = Case {
                    head_dim,
                    kv_heads: 2,
                    query_heads: 8,
                    block_size: BENCH_BLOCK_SIZE,
                    lengths,
                    window,
                    features: false,
                };
                for (dtype, fp8) in [
                    (DType::BF16, false),
                    (DType::F16, false),
                    (DType::BF16, true),
                ] {
                    let inputs = Inputs::new(&device, case, dtype, fp8)?;
                    let cache_dtype = if fp8 { DType::F8E4M3 } else { dtype };
                    let (key, value) = inputs.cache(&device, case, cache_dtype, true)?;
                    let batch = lengths.len();
                    let mut metadata = PagedAttentionInputMetadata::dummy(&device)?;
                    metadata.block_size = Some(case.block_size);
                    metadata.block_tables =
                        Some(HashMap::from([(location, inputs.block_tables.clone())]));
                    metadata.paged_context_lens_cpu = Some(lengths.to_vec());
                    metadata.prefix_gather_workspace_limit = Some(0);
                    let cached = lengths
                        .iter()
                        .map(|&len| {
                            let cropped = if window.is_some_and(|window| len >= window) {
                                DONOR_CROPPED_BLOCKS * case.block_size
                            } else {
                                0
                            };
                            len + cropped - 1
                        })
                        .collect::<Vec<_>>();
                    let metadata = metadata
                        .for_reduced_prefill_queries(
                            std::slice::from_ref(&device),
                            &cached,
                            &vec![1; batch],
                        )
                        .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
                    let attention = PagedAttention::new_with_fp8_attention_scales(
                        head_dim,
                        &device,
                        None,
                        fp8.then_some(Fp8AttentionScales {
                            q: 1.,
                            k: KEY_SCALE,
                            v: VALUE_SCALE,
                        }),
                    )?;
                    let params = SdpaParams {
                        n_kv_groups: case.query_heads / case.kv_heads,
                        softmax_scale: 1. / (head_dim as f32).sqrt(),
                        softcap: None,
                        sliding_window: window,
                        sinks: None,
                    };
                    let output = attention.forward_donor_cache(
                        &inputs.query.unsqueeze(2)?,
                        &key,
                        &value,
                        &AttentionMask::None,
                        &metadata,
                        &params,
                        None,
                    )?;
                    assert_eq!(output.dims(), &[batch, 1, case.query_heads, head_dim]);
                    assert_close(
                        &values(&output)?,
                        &inputs.expected(case),
                        if dtype == DType::F16 {
                            F16_TOLERANCE
                        } else {
                            BF16_TOLERANCE
                        },
                        &format!("reduced donor {case:?} {dtype:?} fp8={fp8}"),
                    );
                    let mut noncausal = metadata.clone();
                    noncausal.has_noncausal_mm_context = true;
                    assert!(attention
                        .forward_donor_cache(
                            &inputs.query.unsqueeze(2)?,
                            &key,
                            &value,
                            &AttentionMask::None,
                            &noncausal,
                            &params,
                            None,
                        )
                        .is_err());
                }
            }
        }
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
            CASES[0], CASES[2], CASES[4], CASES[7], CASES[11], CASES[13], CASES[15], CASES[16],
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
                let [hnd_timings, legacy_timings] = &mut timings;
                for (round, (hnd, legacy)) in hnd_timings.iter_mut().zip(legacy_timings).enumerate()
                {
                    let round_timings = [hnd, legacy];
                    for index in 0..caches.len() {
                        let layout = (index + round) % caches.len();
                        device.synchronize()?;
                        let start = Instant::now();
                        for _ in 0..BENCH_TIMED_ITERS {
                            decode(layout)?;
                        }
                        device.synchronize()?;
                        *round_timings[layout] =
                            start.elapsed().as_secs_f64() * 1e3 / BENCH_TIMED_ITERS as f64;
                    }
                    let (hnd, legacy) = (*round_timings[0], *round_timings[1]);
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

#[test]
#[ignore = "benchmark; needs exclusive use of a Metal GPU"]
fn metal_native_prefill_vs_sdpa_bench() -> Result<()> {
    const QUERIES_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_QUERIES";
    const CONTEXTS_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_CONTEXTS";
    const DIMS_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_DIMS";
    const WINDOW_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_WINDOW";
    const BATCHES_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_BATCHES";
    const HEADS_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_HEADS";
    const KV_HEADS_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_KV_HEADS";
    const MASKLESS_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_MASKLESS";
    const MASKLESS_HEAD_DIMS: &[usize] = &[32, 64, 72, 80, 96, 128, 256];
    let sizes = |name: &str, defaults: &[usize]| {
        std::env::var(name).map_or_else(
            |_| defaults.to_vec(),
            |value| {
                value
                    .split(',')
                    .map(|part| {
                        let size = part.trim().parse::<usize>().expect(name);
                        assert!(size > 0, "{name} must contain positive integers");
                        size
                    })
                    .collect::<Vec<_>>()
            },
        )
    };
    let batches = sizes(BATCHES_ENV, &[1]);
    let heads = sizes(HEADS_ENV, &[PREFILL_BENCH_QUERY_HEADS]);
    let kv_heads = sizes(KV_HEADS_ENV, &[PREFILL_BENCH_KV_HEADS]);
    assert_eq!(heads.len(), 1, "{HEADS_ENV} must contain one integer");
    assert_eq!(kv_heads.len(), 1, "{KV_HEADS_ENV} must contain one integer");
    let (query_heads, kv_heads) = (heads[0], kv_heads[0]);
    assert_eq!(
        query_heads % kv_heads,
        0,
        "query heads must divide into KV groups"
    );
    let maskless = std::env::var(MASKLESS_ENV).is_ok_and(|value| match value.trim() {
        "1" | "true" => true,
        "0" | "false" => false,
        _ => panic!("{MASKLESS_ENV} must be 0, 1, false, or true"),
    });
    let dense_mode = if maskless {
        "maskless_causal"
    } else {
        "masked"
    };
    let query_lengths = sizes(QUERIES_ENV, &[PREFILL_BENCH_QUERY_LEN]);
    let context_lengths = sizes(CONTEXTS_ENV, &[PREFILL_BENCH_CONTEXT_LEN]);
    let head_dims = sizes(DIMS_ENV, &PREFILL_BENCH_HEAD_DIMS);
    let window = std::env::var(WINDOW_ENV)
        .ok()
        .map(|value| value.trim().parse::<usize>().expect(WINDOW_ENV))
        .filter(|&size| size > 0);
    let window_column = window.unwrap_or(0);
    let device = Device::new_metal(0)?;
    let iterations = std::env::var(PREFILL_BENCH_ITERS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(BENCH_TIMED_ITERS);
    assert!(iterations > 0);
    println!(
        "Native paged prefill vs engine SDPA: BF16, host dispatch + allocation + synchronized GPU completion; excludes mask construction and KV packing/gathering"
    );
    println!(
        "warmup={BENCH_WARMUP_ITERS}, timed={iterations}, rotating rounds={BENCH_ROUNDS}; D128/256 use Candle SDPA, D512 uses engine Metal FlashAttn"
    );
    println!("window={window:?}, dense={dense_mode}; query/context/dimension/batch environment values accept comma-separated lists; WINDOW=0 disables the window");
    println!("head_dim,query_heads,kv_heads,batch,query_len,context_len,window,dense_mode,round,native_ms,sdpa_ms,sdpa_over_native,max_abs_error");
    for head_dim in head_dims {
        for &query_len in &query_lengths {
            for &context_len in &context_lengths {
                assert!(
                    context_len >= query_len,
                    "context length must cover the query length"
                );
                if maskless {
                    assert!(
                        MASKLESS_HEAD_DIMS.contains(&head_dim)
                            && query_len > 1
                            && query_len == context_len
                            && window.is_none_or(|size| size >= context_len),
                        "maskless comparison requires supported D<=256, Q=K>1, and no excluded window keys"
                    );
                }
                for &batch in &batches {
                    let context_lens = vec![context_len; batch];
                    let query_lens = vec![query_len; batch];
                    let query_starts = vec![context_len - query_len; batch];
                    let key_starts = vec![0; batch];
                    let case = Case {
                        head_dim,
                        kv_heads,
                        query_heads,
                        block_size: BENCH_BLOCK_SIZE,
                        lengths: &context_lens,
                        window: None,
                        features: false,
                    };
                    let inputs = Inputs::new(&device, case, BENCH_DTYPE, false)?;
                    let (key_cache, value_cache) =
                        inputs.cache(&device, case, BENCH_DTYPE, true)?;
                    let key = inputs
                        .key
                        .reshape((batch, context_len, kv_heads, head_dim))?
                        .transpose(1, 2)?
                        .contiguous()?;
                    let value = inputs
                        .value
                        .reshape((batch, context_len, kv_heads, head_dim))?
                        .transpose(1, 2)?
                        .contiguous()?;
                    let query = Tensor::from_vec(
                        (0..batch * query_heads * query_len * head_dim)
                            .map(|index| ((index * 11 + 1) as f32).sin() * 0.7)
                            .collect::<Vec<_>>(),
                        (batch, query_heads, query_len, head_dim),
                        &Device::Cpu,
                    )?
                    .to_dtype(BENCH_DTYPE)?
                    .to_device(&device)?;
                    let prefix_len = context_len - query_len;
                    let mask = Tensor::from_vec(
                        (0..query_len)
                            .flat_map(|query| {
                                (0..context_len).map(move |key| {
                                    if key <= prefix_len + query
                                        && window.is_none_or(|size| prefix_len + query - key < size)
                                    {
                                        0f32
                                    } else {
                                        f32::NEG_INFINITY
                                    }
                                })
                            })
                            .collect::<Vec<_>>(),
                        (1, 1, query_len, context_len),
                        &Device::Cpu,
                    )?
                    .to_dtype(if head_dim == PREFILL_BENCH_FUSED_HEAD_DIM {
                        DType::F16
                    } else {
                        BENCH_DTYPE
                    })?
                    .to_device(&device)?;
                    let mask = AttentionMask::Custom(mask);
                    let params = SdpaParams {
                        n_kv_groups: query_heads / kv_heads,
                        softcap: None,
                        softmax_scale: 1. / (head_dim as f32).sqrt(),
                        sliding_window: window,
                        sinks: None,
                    };
                    let execute = |native: bool| {
                        if native {
                            metal_paged_prefill(MetalPagedPrefillRequest {
                                query: &query,
                                key_cache: &key_cache,
                                value_cache: &value_cache,
                                block_tables: &inputs.block_tables,
                                query_lens: &query_lens,
                                context_lens: &context_lens,
                                query_start_positions: &query_starts,
                                key_start_positions: &key_starts,
                                packed_queries: false,
                                causal: true,
                                sliding_window: window,
                                window_right: None,
                                attention_chunk_size: None,
                                multimodal_ranges: None,
                                additive_mask: None,
                                packed_mask: false,
                                mask_key_offsets: None,
                                alibi_slopes: None,
                                sinks: None,
                                k_scale: None,
                                v_scale: None,
                                scale: params.softmax_scale,
                                softcap: None,
                            })
                        } else if maskless {
                            Sdpa.run_attention_noflash(&query, &key, &value, None, &params, true)
                        } else {
                            Sdpa.run_attention(&query, &key, &value, &mask, None, &params)
                        }
                    };
                    let native = execute(true)?;
                    let dense = execute(false)?;
                    assert_eq!(native.shape(), dense.shape());
                    let native = values(&native)?;
                    let dense = values(&dense)?;
                    let label = format!(
                        "native prefill vs SDPA D{head_dim} H{query_heads}/KV{kv_heads} B{batch} Q{query_len} K{context_len} window={window:?} dense={dense_mode}"
                    );
                    assert_close(&native, &dense, PREFILL_BENCH_TOLERANCE, &label);
                    let max_error = native
                        .iter()
                        .zip(&dense)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    for native in [true, false] {
                        for _ in 0..BENCH_WARMUP_ITERS {
                            let output = execute(native)?;
                            device.synchronize()?;
                            std::hint::black_box(output);
                        }
                    }
                    let mut timings = [[0f64; BENCH_ROUNDS]; 2];
                    let [native_timings, dense_timings] = &mut timings;
                    for (round, (native, dense)) in
                        native_timings.iter_mut().zip(dense_timings).enumerate()
                    {
                        let round_timings = [native, dense];
                        for index in 0..round_timings.len() {
                            let backend = (index + round) % round_timings.len();
                            device.synchronize()?;
                            let start = Instant::now();
                            for _ in 0..iterations {
                                let output = execute(backend == 0)?;
                                device.synchronize()?;
                                std::hint::black_box(output);
                            }
                            *round_timings[backend] =
                                start.elapsed().as_secs_f64() * 1e3 / iterations as f64;
                        }
                        let (native, dense) = (*round_timings[0], *round_timings[1]);
                        println!(
                            "{head_dim},{query_heads},{kv_heads},{batch},{query_len},{context_len},{window_column},{dense_mode},{round},{native:.6},{dense:.6},{:.4},{max_error:.8}",
                            dense / native,
                        );
                    }
                    for samples in &mut timings {
                        samples.sort_by(f64::total_cmp);
                    }
                    let (native, dense) =
                        (timings[0][BENCH_ROUNDS / 2], timings[1][BENCH_ROUNDS / 2]);
                    println!(
                        "{head_dim},{query_heads},{kv_heads},{batch},{query_len},{context_len},{window_column},{dense_mode},median,{native:.6},{dense:.6},{:.4},{max_error:.8}",
                        dense / native,
                    );
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum PrefillMaskCase {
    Causal,
    Bidirectional,
    Chunked,
    Multimodal,
    Custom,
    CustomFullView,
    Speculative,
}

#[derive(Clone, Copy, Debug)]
struct PrefillWrapperCase {
    cold: bool,
    cropped: bool,
    donor: bool,
    mask: PrefillMaskCase,
}

fn prefill_wrapper_case(
    device: &Device,
    dtype: DType,
    hnd: bool,
    test: PrefillWrapperCase,
) -> Result<()> {
    const HEAD_DIM: usize = 64;
    const QUERY_HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const BLOCK_SIZE: usize = 8;
    const WINDOW: usize = 7;
    const CHUNK: usize = 8;
    let speculative = matches!(test.mask, PrefillMaskCase::Speculative);
    let query_lens = if speculative { [3, 3] } else { [3, 5] };
    let lengths = if test.cold { query_lens } else { [17, 23] };
    let starts = if test.cropped && !matches!(test.mask, PrefillMaskCase::CustomFullView) {
        [16, 24]
    } else {
        [0, 0]
    };
    let max_q = *query_lens.iter().max().unwrap();
    let max_k = *lengths.iter().max().unwrap();
    let case = Case {
        head_dim: HEAD_DIM,
        kv_heads: KV_HEADS,
        query_heads: QUERY_HEADS,
        block_size: BLOCK_SIZE,
        lengths: &lengths,
        window: None,
        features: false,
    };
    let inputs = Inputs::new(device, case, dtype, false)?;
    let (key_cache, value_cache) = inputs.cache(device, case, dtype, hnd)?;
    let mut query = vec![f32::NAN; 2 * QUERY_HEADS * max_q * HEAD_DIM];
    let mut key = vec![f32::NAN; 2 * KV_HEADS * max_q * HEAD_DIM];
    let mut value = key.clone();
    let mut slots = vec![-1i64; 2 * max_q];
    let original_slots = inputs.slots.to_device(&Device::Cpu)?.to_vec1::<i64>()?;
    let mut token_base = 0;
    for seq in 0..2 {
        for row in 0..query_lens[seq] {
            slots[seq * max_q + row] =
                original_slots[token_base + lengths[seq] - query_lens[seq] + row];
            for head in 0..QUERY_HEADS {
                for dim in 0..HEAD_DIM {
                    query[((seq * QUERY_HEADS + head) * max_q + row) * HEAD_DIM + dim] = round_to(
                        inputs.q_cpu[(seq * QUERY_HEADS + head) * HEAD_DIM + dim]
                            + row as f32 * 0.07,
                        dtype,
                    );
                }
            }
            for head in 0..KV_HEADS {
                let from = ((token_base + lengths[seq] - query_lens[seq] + row) * KV_HEADS + head)
                    * HEAD_DIM;
                let to = ((seq * KV_HEADS + head) * max_q + row) * HEAD_DIM;
                key[to..to + HEAD_DIM].copy_from_slice(&inputs.k_cpu[from..from + HEAD_DIM]);
                value[to..to + HEAD_DIM].copy_from_slice(&inputs.v_cpu[from..from + HEAD_DIM]);
            }
        }
        token_base += lengths[seq];
    }
    let q = Tensor::from_vec(
        query.clone(),
        (2, QUERY_HEADS, max_q, HEAD_DIM),
        &Device::Cpu,
    )?
    .to_dtype(dtype)?
    .to_device(device)?;
    let k = Tensor::from_vec(key, (2, KV_HEADS, max_q, HEAD_DIM), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)?;
    let v = Tensor::from_vec(value, (2, KV_HEADS, max_q, HEAD_DIM), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)?;
    let slot_tensor = Tensor::from_vec(slots, (2, max_q), device)?;
    if !test.donor {
        let zeros = Tensor::zeros((2 * max_q, KV_HEADS, HEAD_DIM), dtype, device)?;
        reshape_and_cache(
            &zeros,
            &zeros,
            None,
            None,
            &key_cache,
            &value_cache,
            &slot_tensor.flatten_all()?,
        )?;
    }
    let mut metadata = PagedAttentionInputMetadata::dummy(device)?;
    let location = device.location();
    metadata.block_size = Some(BLOCK_SIZE);
    metadata.slot_mappings = HashMap::from([(location, slot_tensor)]);
    metadata.block_tables = Some(HashMap::from([(location, inputs.block_tables.clone())]));
    metadata.paged_context_lens_cpu = Some(lengths.to_vec());
    metadata.full_paged_context_lens_cpu = Some(
        lengths
            .iter()
            .zip(starts)
            .map(|(len, start)| len + start)
            .collect(),
    );
    metadata.query_lens = Some(query_lens.to_vec());
    metadata.num_cached_tokens = (!test.cold).then(|| {
        lengths
            .iter()
            .zip(starts)
            .zip(query_lens)
            .map(|((len, start), q)| len + start - q)
            .collect()
    });
    metadata.is_first_prompt_chunk = test.cold;
    metadata.prefix_gather_workspace_limit = Some(0);
    let custom = matches!(
        test.mask,
        PrefillMaskCase::Custom | PrefillMaskCase::CustomFullView
    );
    if matches!(test.mask, PrefillMaskCase::CustomFullView) {
        metadata.full_block_tables = metadata.block_tables.clone();
        metadata.block_tables = Some(HashMap::from([(
            location,
            inputs
                .block_tables
                .narrow(1, 1, inputs.block_tables.dim(1)? - 1)?,
        )]));
        metadata.paged_context_lens_cpu =
            Some(lengths.iter().map(|len| len - BLOCK_SIZE).collect());
    }
    let mask_k = if test.cropped {
        lengths
            .iter()
            .zip(starts)
            .map(|(len, start)| len + start)
            .max()
            .unwrap()
    } else {
        max_k
    };
    let mut custom_values = vec![0f32; 2 * QUERY_HEADS * max_q * mask_k];
    for (index, value) in custom_values.iter_mut().enumerate() {
        *value = (index % 11) as f32 * 0.17 - 0.8;
    }
    let mask = match test.mask {
        PrefillMaskCase::Bidirectional => AttentionMask::Bidirectional,
        PrefillMaskCase::Chunked => AttentionMask::ChunkedCausal { chunk_size: CHUNK },
        PrefillMaskCase::Custom | PrefillMaskCase::CustomFullView => {
            AttentionMask::Custom(Tensor::from_vec(
                custom_values.clone(),
                (2, QUERY_HEADS, max_q, mask_k),
                &Device::Cpu,
            )?)
        }
        PrefillMaskCase::Speculative => AttentionMask::None,
        _ => AttentionMask::CausalFlash,
    };
    if matches!(test.mask, PrefillMaskCase::Multimodal) {
        let ranges = Tensor::from_vec(
            lengths
                .iter()
                .flat_map(|len| [(len - 4) as i32, *len as i32])
                .collect::<Vec<_>>(),
            (2, 1, 2),
            &Device::Cpu,
        )?;
        metadata.mm_prefix_ranges = Some(HashMap::from([
            (Device::Cpu.location(), ranges.clone()),
            (location, ranges.to_device(device)?),
        ]));
        metadata.has_noncausal_mm_context = true;
    }
    if speculative {
        let table = inputs
            .block_tables
            .to_device(&Device::Cpu)?
            .to_vec2::<u32>()?;
        let expanded = table
            .iter()
            .flat_map(|row| std::iter::repeat_n(row.clone(), max_q))
            .flatten()
            .collect::<Vec<_>>();
        metadata.block_tables = Some(HashMap::from([(
            location,
            Tensor::from_vec(expanded, (2 * max_q, inputs.block_tables.dim(1)?), device)?,
        )]));
        metadata.paged_context_lens_cpu = Some(
            lengths
                .iter()
                .flat_map(|len| (len - max_q + 1)..=*len)
                .collect(),
        );
        metadata.full_paged_context_lens_cpu = Some(
            lengths
                .iter()
                .zip(starts)
                .flat_map(|(len, start)| (len + start - max_q + 1)..=(len + start))
                .collect(),
        );
        metadata.query_lens = None;
    }
    let window = test.cropped.then_some(WINDOW);
    let params = SdpaParams {
        n_kv_groups: QUERY_HEADS / KV_HEADS,
        softmax_scale: 0.125,
        softcap: None,
        sliding_window: window,
        sinks: None,
    };
    let attention = PagedAttention::new(HEAD_DIM, device, None)?;
    let result = if test.donor {
        attention.forward_donor_cache(
            &q,
            &key_cache,
            &value_cache,
            &mask,
            &metadata,
            &params,
            None,
        )?
    } else {
        attention.forward(
            &q,
            &k,
            &v,
            &mask,
            Some(key_cache.clone()),
            Some(value_cache.clone()),
            &metadata,
            &params,
            None,
        )?
    };
    let result = if speculative {
        result.transpose(1, 2)?.contiguous()?
    } else {
        result
    };
    let actual = values(&result)?;
    let mut expected = Vec::new();
    let mut selected = Vec::new();
    token_base = 0;
    for seq in 0..2 {
        for head in 0..QUERY_HEADS {
            for row in 0..query_lens[seq] {
                let qpos = starts[seq] + lengths[seq] - query_lens[seq] + row;
                let qbase = ((seq * QUERY_HEADS + head) * max_q + row) * HEAD_DIM;
                let mut logits = Vec::new();
                for token in 0..lengths[seq] {
                    let kpos = starts[seq] + token;
                    let same_mm = matches!(test.mask, PrefillMaskCase::Multimodal)
                        && token >= lengths[seq] - 4
                        && qpos >= starts[seq] + lengths[seq] - 4;
                    let visible = custom
                        || same_mm
                        || match test.mask {
                            PrefillMaskCase::Bidirectional => {
                                window.is_none_or(|w| qpos.abs_diff(kpos) < w)
                            }
                            PrefillMaskCase::Chunked => {
                                kpos <= qpos
                                    && kpos / CHUNK == qpos / CHUNK
                                    && window.is_none_or(|w| qpos - kpos < w)
                            }
                            _ => kpos <= qpos && window.is_none_or(|w| qpos - kpos < w),
                        };
                    let base = ((token_base + token) * KV_HEADS + head / (QUERY_HEADS / KV_HEADS))
                        * HEAD_DIM;
                    let logit = if visible {
                        (0..HEAD_DIM)
                            .map(|dim| query[qbase + dim] as f64 * inputs.k_cpu[base + dim] as f64)
                            .sum::<f64>()
                            * 0.125
                            + if custom {
                                custom_values[((seq * QUERY_HEADS + head) * max_q + row) * mask_k
                                    + starts[seq]
                                    + token] as f64
                            } else {
                                0.
                            }
                    } else {
                        f64::NEG_INFINITY
                    };
                    logits.push(logit);
                }
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = logits
                    .iter()
                    .map(|logit| (logit - max).exp())
                    .collect::<Vec<_>>();
                let sum = weights.iter().sum::<f64>();
                for dim in 0..HEAD_DIM {
                    expected.push(
                        (weights
                            .iter()
                            .enumerate()
                            .map(|(token, weight)| {
                                weight
                                    * inputs.v_cpu[((token_base + token) * KV_HEADS
                                        + head / (QUERY_HEADS / KV_HEADS))
                                        * HEAD_DIM
                                        + dim] as f64
                            })
                            .sum::<f64>()
                            / sum) as f32,
                    );
                    selected.push(actual[qbase + dim]);
                }
            }
        }
        token_base += lengths[seq];
    }
    assert_close(
        &selected,
        &expected,
        match dtype {
            DType::BF16 => BF16_TOLERANCE,
            DType::F16 => F16_TOLERANCE,
            _ => F32_TOLERANCE,
        },
        &format!("prefill {test:?} {dtype:?} hnd={hnd}"),
    );
    let (gathered_k, gathered_v) = gather_kv_cache(
        &key_cache,
        &value_cache,
        None,
        None,
        &inputs.block_tables,
        &inputs.cu_lengths,
        inputs.tokens,
        dtype,
    )?;
    assert_close(&values(&gathered_k)?, &inputs.k_cpu, 0., "prefill K cache");
    assert_close(&values(&gathered_v)?, &inputs.v_cpu, 0., "prefill V cache");
    Ok(())
}

#[test]
fn metal_native_prefill_wrapper_matches_cpu_and_preserves_cache() -> Result<()> {
    let device = Device::new_metal(0)?;
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        for hnd in [false, true] {
            for donor in [false, true] {
                for (cold, cropped, mask) in [
                    (true, false, PrefillMaskCase::Causal),
                    (false, false, PrefillMaskCase::Causal),
                    (false, true, PrefillMaskCase::Causal),
                    (false, true, PrefillMaskCase::Bidirectional),
                    (false, true, PrefillMaskCase::Chunked),
                    (false, true, PrefillMaskCase::Multimodal),
                    (false, false, PrefillMaskCase::Custom),
                    (false, true, PrefillMaskCase::Custom),
                    (false, true, PrefillMaskCase::CustomFullView),
                    (false, true, PrefillMaskCase::Speculative),
                ] {
                    prefill_wrapper_case(
                        &device,
                        dtype,
                        hnd,
                        PrefillWrapperCase {
                            cold,
                            cropped,
                            donor,
                            mask,
                        },
                    )?;
                }
            }
        }
    }
    Ok(())
}
