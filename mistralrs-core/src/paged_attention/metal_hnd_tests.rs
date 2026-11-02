#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;
use mistralrs_paged_attn::{
    copy_blocks, gather_kv_cache, paged_attention_with_window, reshape_and_cache, swap_blocks,
};

use crate::{
    attention::{AttentionMask, SdpaParams},
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
const DONOR_QUERY_LENGTHS: [&[usize]; 3] =
    [&[224], &[543], &[31, 64, 224, 511, 512, 513, 529, 543]];
const DONOR_CROPPED_BLOCKS: usize = 7;
const DONOR_BLOCK_SIZE: usize = 32;

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
                    block_size: DONOR_BLOCK_SIZE,
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

mod decode {
    use candle_core::{DType, Device, Result, Tensor};

    const BLOCK_SIZE: usize = 32;
    const CACHE_VECTOR_BYTES: usize = 16;
    const SHORT_CONTEXT_LENS: [usize; 23] = [
        1, 31, 32, 33, 127, 128, 129, 231, 255, 256, 257, 287, 288, 289, 319, 320, 321, 511, 512,
        513, 1023, 1024, 1025,
    ];
    const SHORT_SHAPES: [(usize, usize, usize); 7] = [
        (4, 4, 64),
        (8, 4, 96),
        (12, 4, 128),
        (16, 4, 192),
        (32, 4, 256),
        (16, 8, 128),
        (16, 4, 256),
    ];
    const QUERY_PADDING_HEADS: usize = 2;
    const BF16_ABS_TOL: f32 = 3e-2;
    const F16_ABS_TOL: f32 = 3e-3;
    const GEMMA_DECODE_SHAPE: (usize, usize, usize) = (8, 2, 256);
    const GEMMA_GLOBAL_DECODE_SHAPE: (usize, usize, usize) = (8, 2, 512);
    const GEMMA_SLIDING_WINDOW: usize = 512;
    const MASKED_DECODE_CASES: [(usize, usize, usize); 2] = [(33, 0, 17), (543, 31, 543)];
    const MASK_CONTROL_CONTEXT: usize = 256;
    const MASK_QK_AMPLITUDE: f32 = 0.1;
    const MASK_EXCLUDED_VALUE_BIAS: f32 = 4.0;
    const CUSTOM_MASK_BATCH: usize = 2;
    const CUSTOM_MASK_CONTEXT: usize = 33;
    const CUSTOM_MASK_VISIBLE: usize = 9;
    const CUSTOM_MASK_BIAS: f32 = 0.25;
    const DECODE_CASES: [(usize, usize, usize, [usize; 4]); 7] = [
        (2, 2, 64, [1, 31, 32, 128]),
        (6, 2, 96, [1, 127, 128, 129]),
        (16, 2, 128, [33, 511, 512, 513]),
        // Eight KV heads and four sequences select 512-token partitions at this context length.
        (16, 8, 192, [1, 511, 512, 513]),
        (24, 4, 256, [1, 37, 700, 1500]),
        (4, 1, 256, [1, 512, 513, 2049]),
        (8, 2, 512, [1, 37, 90, 700]),
    ];

    fn max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        Ok(actual.iter().zip(&expected).fold(0.0f32, |max, (a, e)| {
            let diff = (a - e).abs();
            if diff.is_finite() {
                max.max(diff)
            } else {
                f32::INFINITY
            }
        }))
    }

    /// Packs dense `[kv_heads, len, D]` K/V into one sequence's paged blocks, padding past `len` with NaN.
    fn pack_seq(k: &Tensor, v: &Tensor, blocks: usize) -> Result<(Tensor, Tensor)> {
        let (kv_heads, len, d) = k.dims3()?;
        let x = CACHE_VECTOR_BYTES / k.dtype().size_in_bytes();
        let padded = blocks * BLOCK_SIZE;
        let pad = |t: &Tensor| -> Result<Tensor> {
            if padded == len {
                return Ok(t.clone());
            }
            let nan = Tensor::full(f32::NAN, (kv_heads, padded - len, d), t.device())?
                .to_dtype(t.dtype())?;
            Tensor::cat(&[t, &nan], 1)
        };
        let k = pad(k)?
            .reshape((kv_heads, blocks, BLOCK_SIZE, d / x, x))?
            .permute((1, 0, 3, 2, 4))?
            .contiguous()?;
        let v = pad(v)?
            .reshape((kv_heads, blocks, BLOCK_SIZE, d))?
            .permute((1, 0, 3, 2))?
            .contiguous()?;
        Ok((k, v))
    }

    fn cache_with_layout(k: &Tensor, v: &Tensor, hnd: bool) -> Result<(Tensor, Tensor)> {
        if !hnd {
            return Ok((k.clone(), v.clone()));
        }
        let (blocks, heads, head_chunks, block_size, x) = k.dims5()?;
        let k = k.permute((0, 1, 3, 2, 4))?.contiguous()?.reshape((
            blocks,
            heads,
            block_size,
            head_chunks * x,
        ))?;
        Ok((k, v.transpose(2, 3)?.contiguous()?))
    }

    fn remap_cache_blocks(
        key_cache: &Tensor,
        value_cache: &Tensor,
        batch: usize,
        blocks_per_seq: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let blocks = batch * blocks_per_seq;
        let device = key_cache.device();
        let order = (0..blocks)
            .rev()
            .flat_map(|block| [blocks as u32, block as u32])
            .chain(std::iter::once(blocks as u32))
            .collect::<Vec<_>>();
        let order = Tensor::new(order, device)?;
        let remap = |cache: &Tensor| -> Result<Tensor> {
            let poison = Tensor::full(f32::NAN, cache.narrow(0, 0, 1)?.shape(), device)?
                .to_dtype(cache.dtype())?;
            Tensor::cat(&[cache, &poison], 0)?.index_select(&order, 0)
        };
        let block_tables = (0..blocks)
            .map(|block| (2 * (blocks - block) - 1) as u32)
            .collect::<Vec<_>>();
        let block_tables = Tensor::new(block_tables, device)?.reshape((batch, blocks_per_seq))?;
        Ok((remap(key_cache)?, remap(value_cache)?, block_tables))
    }

    fn assert_decode_matches_dense(
        device: &Device,
        (q_heads, kv_heads, head_dim): (usize, usize, usize),
        lens: &[usize],
        dtype: DType,
        tolerance: f32,
    ) -> Result<()> {
        let batch = lens.len();
        let max_len = *lens.iter().max().unwrap();
        let blocks_per_seq = max_len.div_ceil(BLOCK_SIZE);
        let scale = 1. / (head_dim as f32).sqrt();
        let q = Tensor::randn(
            0f32,
            1.,
            (batch, q_heads + QUERY_PADDING_HEADS, head_dim),
            device,
        )?
        .to_dtype(dtype)?
        .narrow(1, QUERY_PADDING_HEADS / 2, q_heads)?;
        let (mut k_blocks, mut v_blocks, mut expected) = (vec![], vec![], vec![]);
        for (b, &len) in lens.iter().enumerate() {
            let k = Tensor::randn(0f32, 1., (kv_heads, len, head_dim), device)?.to_dtype(dtype)?;
            let v = Tensor::randn(0f32, 1., (kv_heads, len, head_dim), device)?.to_dtype(dtype)?;
            let (kb, vb) = pack_seq(&k, &v, blocks_per_seq)?;
            k_blocks.push(kb);
            v_blocks.push(vb);
            let gqa = q_heads / kv_heads;
            let qb = q
                .narrow(0, b, 1)?
                .to_dtype(DType::F32)?
                .reshape((kv_heads, gqa, head_dim))?;
            let kf = k.to_dtype(DType::F32)?;
            let att = (qb.matmul(&kf.t()?)? * scale as f64)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            let out = att.matmul(&v.to_dtype(DType::F32)?)?;
            expected.push(out.reshape((1, q_heads, head_dim))?);
        }
        let (key_cache, value_cache, block_tables) = remap_cache_blocks(
            &Tensor::cat(&k_blocks, 0)?,
            &Tensor::cat(&v_blocks, 0)?,
            batch,
            blocks_per_seq,
        )?;
        let context_lens = Tensor::new(lens.iter().map(|&l| l as u32).collect::<Vec<_>>(), device)?;
        let actual = mistralrs_paged_attn::paged_attention(
            &q,
            None,
            None,
            &key_cache,
            &value_cache,
            &block_tables,
            &context_lens,
            None,
            max_len,
            scale,
            1.,
            None,
        )?;
        let expected = Tensor::cat(&expected, 0)?;
        let diff = max_abs_diff(&actual, &expected)?;
        assert!(
            diff < tolerance,
            "{dtype:?} {q_heads}/{kv_heads}/{head_dim} {lens:?}: max abs diff {diff}"
        );
        Ok(())
    }

    #[test]
    fn paged_decode_matches_dense_sdpa() -> Result<()> {
        let device = Device::new_metal(0)?;
        for (dtype, tolerance) in [(DType::BF16, BF16_ABS_TOL), (DType::F16, F16_ABS_TOL)] {
            for (q_heads, kv_heads, head_dim, lens) in DECODE_CASES {
                assert_decode_matches_dense(
                    &device,
                    (q_heads, kv_heads, head_dim),
                    &lens,
                    dtype,
                    tolerance,
                )?;
            }
        }
        Ok(())
    }

    #[test]
    fn paged_decode_short_contexts_match_dense() -> Result<()> {
        let device = Device::new_metal(0)?;
        for (dtype, tolerance) in [(DType::BF16, BF16_ABS_TOL), (DType::F16, F16_ABS_TOL)] {
            for shape in SHORT_SHAPES {
                for len in SHORT_CONTEXT_LENS {
                    assert_decode_matches_dense(&device, shape, &[len], dtype, tolerance)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn sdpa_handles_token_major_kv_views() -> Result<()> {
        use crate::attention::{AttentionMask, Sdpa, SdpaParams};

        let device = Device::new_metal(0)?;
        let (q_heads, kv_heads, head_dim, kv_len) = (8, 2, 256, 90);
        let q =
            Tensor::randn(0f32, 1., (1, q_heads, 1, head_dim), &device)?.to_dtype(DType::BF16)?;
        let k_tokens = Tensor::randn(0f32, 1., (kv_len, kv_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let v_tokens = Tensor::randn(0f32, 1., (kv_len, kv_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = k_tokens.transpose(0, 1)?.unsqueeze(0)?;
        let v = v_tokens.transpose(0, 1)?.unsqueeze(0)?;
        let mask = Tensor::zeros((1, 1, 1, kv_len), DType::BF16, &device)?;
        let params = SdpaParams {
            n_kv_groups: q_heads / kv_heads,
            softcap: None,
            softmax_scale: 1. / (head_dim as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };
        for mask in [AttentionMask::None, AttentionMask::Custom(mask)] {
            let strided = Sdpa.run_attention(&q, &k, &v, &mask, None, &params)?;
            let packed =
                Sdpa.run_attention(&q, &k.contiguous()?, &v.contiguous()?, &mask, None, &params)?;
            let diff = (strided.to_dtype(DType::F32)? - packed.to_dtype(DType::F32)?)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-3, "max abs diff {diff}");
        }
        Ok(())
    }

    fn decode_mask_errors(
        device: &Device,
        dtype: DType,
        (q_heads, kv_heads, head_dim): (usize, usize, usize),
        (context, visible_start, visible_end): (usize, usize, usize),
    ) -> Result<(f32, f32, f32)> {
        use crate::attention::{AttentionMask, Sdpa, SdpaParams};

        let cpu = Device::Cpu;
        let q = Tensor::from_vec(
            (0..q_heads * head_dim)
                .map(|i| (i as f32).sin() * MASK_QK_AMPLITUDE)
                .collect::<Vec<_>>(),
            (1, q_heads, 1, head_dim),
            &cpu,
        )?
        .to_dtype(dtype)?;
        let k = Tensor::from_vec(
            (0..context * kv_heads * head_dim)
                .map(|i| (i as f32).cos() * MASK_QK_AMPLITUDE)
                .collect::<Vec<_>>(),
            (context, kv_heads, head_dim),
            &cpu,
        )?
        .to_dtype(dtype)?;
        let v = Tensor::from_vec(
            (0..context * kv_heads * head_dim)
                .map(|i| {
                    let token = i / (kv_heads * head_dim);
                    let bias = if (visible_start..visible_end).contains(&token) {
                        0.0
                    } else {
                        MASK_EXCLUDED_VALUE_BIAS
                    };
                    (i as f32).sin() + bias
                })
                .collect::<Vec<_>>(),
            (context, kv_heads, head_dim),
            &cpu,
        )?
        .to_dtype(dtype)?;
        let mask = Tensor::from_vec(
            (0..context)
                .map(|token| {
                    if (visible_start..visible_end).contains(&token) {
                        0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect::<Vec<_>>(),
            (1, 1, context),
            &cpu,
        )?;
        let q_cpu = q
            .to_dtype(DType::F32)?
            .reshape((kv_heads, q_heads / kv_heads, head_dim))?;
        let k_cpu = k.to_dtype(DType::F32)?.transpose(0, 1)?.contiguous()?;
        let v_cpu = v.to_dtype(DType::F32)?.transpose(0, 1)?.contiguous()?;
        let logits = q_cpu.matmul(&k_cpu.t()?)?.broadcast_add(&mask)?;
        let expected = candle_nn::ops::softmax_last_dim(&logits)?
            .matmul(&v_cpu)?
            .reshape((1, q_heads, 1, head_dim))?;

        let q = q.to_device(device)?;
        let k = k.to_device(device)?.transpose(0, 1)?.unsqueeze(0)?;
        let v = v.to_device(device)?.transpose(0, 1)?.unsqueeze(0)?;
        let mask = AttentionMask::Custom(mask.unsqueeze(0)?.to_dtype(dtype)?.to_device(device)?);
        let params = SdpaParams {
            n_kv_groups: q_heads / kv_heads,
            softcap: None,
            softmax_scale: 1.0,
            sliding_window: (head_dim == GEMMA_DECODE_SHAPE.2).then_some(GEMMA_SLIDING_WINDOW),
            sinks: None,
        };
        let masked = Sdpa.run_attention(&q, &k, &v, &mask, None, &params)?;
        let unmasked = Sdpa.run_attention(&q, &k, &v, &AttentionMask::None, None, &params)?;
        let trimmed = Sdpa.run_attention(
            &q,
            &k.narrow(2, visible_start, visible_end - visible_start)?,
            &v.narrow(2, visible_start, visible_end - visible_start)?,
            &AttentionMask::None,
            None,
            &params,
        )?;
        let masked_error = max_abs_diff(&masked, &expected)?;
        let masked_vs_unmasked = max_abs_diff(&masked, &unmasked)?;
        let trimmed_error = max_abs_diff(&trimmed, &expected)?;

        let full_context = visible_start == 0 && visible_end == context;
        let generated_window = params.sliding_window.is_some_and(|window| {
            visible_start == context.saturating_sub(window) && visible_end == context
        });
        if full_context || generated_window {
            use crate::{
                paged_attention::PagedAttention,
                pipeline::text_models_inputs_processor::PagedAttentionInputMetadata,
            };
            use std::collections::HashMap;

            let mut cache_key = k.squeeze(0)?;
            let mut cache_value = v.squeeze(0)?;
            if generated_window && visible_start > 0 {
                let poison = Tensor::full(f32::NAN, (kv_heads, visible_start, head_dim), device)?
                    .to_dtype(dtype)?;
                cache_key = Tensor::cat(
                    &[
                        &poison,
                        &cache_key.narrow(1, visible_start, context - visible_start)?,
                    ],
                    1,
                )?;
                cache_value = Tensor::cat(
                    &[
                        &poison,
                        &cache_value.narrow(1, visible_start, context - visible_start)?,
                    ],
                    1,
                )?;
            }
            let (key_cache, value_cache) =
                pack_seq(&cache_key, &cache_value, context.div_ceil(BLOCK_SIZE))?;
            for hnd in [false, true] {
                let (key_cache, value_cache) = cache_with_layout(&key_cache, &value_cache, hnd)?;
                let (key_cache, value_cache, block_tables) =
                    remap_cache_blocks(&key_cache, &value_cache, 1, context.div_ceil(BLOCK_SIZE))?;
                let context_lens = Tensor::new(&[context as u32], device)?;
                let tolerance = if dtype == DType::BF16 {
                    BF16_ABS_TOL
                } else {
                    F16_ABS_TOL
                };
                if full_context {
                    let paged = mistralrs_paged_attn::paged_attention(
                        &q.squeeze(2)?,
                        None,
                        None,
                        &key_cache,
                        &value_cache,
                        &block_tables,
                        &context_lens,
                        None,
                        context,
                        params.softmax_scale,
                        1.0,
                        None,
                    )?;
                    let paged_error = max_abs_diff(&paged, &expected)?;
                    assert!(
                        paged_error < tolerance,
                        "{dtype:?} hnd={hnd}: paged control error {paged_error}"
                    );
                }
                if generated_window {
                    let mut metadata = PagedAttentionInputMetadata::dummy(device)?;
                    metadata.is_first_prompt_chunk = false;
                    metadata.block_size = Some(BLOCK_SIZE);
                    metadata.block_tables =
                        Some(HashMap::from([(device.location(), block_tables)]));
                    metadata.context_lens =
                        Some(HashMap::from([(device.location(), context_lens)]));
                    metadata.paged_context_lens_cpu = Some(vec![context]);
                    metadata.max_context_len = Some(context);
                    let output = PagedAttention::new(head_dim, device, None)?.forward_donor_cache(
                        &q,
                        &key_cache,
                        &value_cache,
                        &AttentionMask::None,
                        &metadata,
                        &params,
                        None,
                    )?;
                    let error = max_abs_diff(&output, &expected)?;
                    assert!(
                        error < tolerance,
                        "{dtype:?} context={context} hnd={hnd}: paged window error {error}"
                    );
                }
            }
        }
        Ok((masked_error, masked_vs_unmasked, trimmed_error))
    }

    fn assert_decode_masks_match_cpu(
        dtype: DType,
        tolerance: f32,
        shape: (usize, usize, usize),
    ) -> Result<()> {
        let device = Device::new_metal(0)?;
        let mut failures = Vec::new();
        for case in MASKED_DECODE_CASES {
            let (masked, masked_vs_unmasked, trimmed) =
                decode_mask_errors(&device, dtype, shape, case)?;
            if masked >= tolerance || trimmed >= tolerance {
                failures.push(format!(
                    "{dtype:?} {shape:?} {case:?}: masked error {masked}, masked/unmasked diff {masked_vs_unmasked}, trimmed error {trimmed}"
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        Ok(())
    }

    #[test]
    fn sdpa_decode_masks_match_cpu_bf16() -> Result<()> {
        assert_decode_masks_match_cpu(DType::BF16, BF16_ABS_TOL, GEMMA_DECODE_SHAPE)
    }

    #[test]
    fn sdpa_decode_masks_match_cpu_f16() -> Result<()> {
        assert_decode_masks_match_cpu(DType::F16, F16_ABS_TOL, GEMMA_DECODE_SHAPE)
    }

    #[test]
    fn sdpa_decode_global_masks_match_cpu_bf16() -> Result<()> {
        assert_decode_masks_match_cpu(DType::BF16, BF16_ABS_TOL, GEMMA_GLOBAL_DECODE_SHAPE)
    }

    fn assert_decode_noop_mask_matches_cpu(
        dtype: DType,
        tolerance: f32,
        shape: (usize, usize, usize),
    ) -> Result<()> {
        let device = Device::new_metal(0)?;
        let (masked, masked_vs_unmasked, trimmed) = decode_mask_errors(
            &device,
            dtype,
            shape,
            (MASK_CONTROL_CONTEXT, 0, MASK_CONTROL_CONTEXT),
        )?;
        assert!(
            masked < tolerance && trimmed < tolerance && masked_vs_unmasked < tolerance,
            "{dtype:?} {shape:?}: no-op mask error {masked}, masked/unmasked diff {masked_vs_unmasked}, unmasked/trimmed error {trimmed}"
        );
        Ok(())
    }

    #[test]
    fn sdpa_decode_noop_sliding_mask_matches_cpu() -> Result<()> {
        for (dtype, tolerance) in [(DType::BF16, BF16_ABS_TOL), (DType::F16, F16_ABS_TOL)] {
            assert_decode_noop_mask_matches_cpu(dtype, tolerance, GEMMA_DECODE_SHAPE)?;
        }
        Ok(())
    }

    #[test]
    fn sdpa_decode_global_noop_mask_matches_cpu_bf16() -> Result<()> {
        assert_decode_noop_mask_matches_cpu(DType::BF16, BF16_ABS_TOL, GEMMA_GLOBAL_DECODE_SHAPE)
    }

    #[test]
    fn sdpa_decode_batch_head_masks_match_cpu() -> Result<()> {
        use crate::{
            attention::{AttentionMask, Sdpa, SdpaParams},
            paged_attention::PagedAttention,
            pipeline::text_models_inputs_processor::PagedAttentionInputMetadata,
        };
        use std::collections::HashMap;

        let device = Device::new_metal(0)?;
        let cpu = Device::Cpu;
        for ((q_heads, kv_heads, head_dim), dtype, tolerance) in [
            (GEMMA_DECODE_SHAPE, DType::BF16, BF16_ABS_TOL),
            (GEMMA_DECODE_SHAPE, DType::F16, F16_ABS_TOL),
            (GEMMA_GLOBAL_DECODE_SHAPE, DType::BF16, BF16_ABS_TOL),
        ] {
            let q = Tensor::from_vec(
                (0..CUSTOM_MASK_BATCH * q_heads * head_dim)
                    .map(|i| (i as f32).sin() * MASK_QK_AMPLITUDE)
                    .collect::<Vec<_>>(),
                (CUSTOM_MASK_BATCH, q_heads, 1, head_dim),
                &cpu,
            )?
            .to_dtype(dtype)?;
            let kv_shape = (CUSTOM_MASK_BATCH, kv_heads, CUSTOM_MASK_CONTEXT, head_dim);
            let kv_elements = CUSTOM_MASK_BATCH * kv_heads * CUSTOM_MASK_CONTEXT * head_dim;
            let k = Tensor::from_vec(
                (0..kv_elements)
                    .map(|i| (i as f32).cos() * MASK_QK_AMPLITUDE)
                    .collect::<Vec<_>>(),
                kv_shape,
                &cpu,
            )?
            .to_dtype(dtype)?;
            let v = Tensor::from_vec(
                (0..kv_elements)
                    .map(|i| {
                        (i as f32).sin()
                            + ((i / head_dim) % CUSTOM_MASK_CONTEXT) as f32
                                / CUSTOM_MASK_CONTEXT as f32
                    })
                    .collect::<Vec<_>>(),
                kv_shape,
                &cpu,
            )?
            .to_dtype(dtype)?;
            let mask = Tensor::from_vec(
                (0..CUSTOM_MASK_BATCH * q_heads * CUSTOM_MASK_CONTEXT)
                    .map(|i| {
                        let row = i / CUSTOM_MASK_CONTEXT;
                        let token = i % CUSTOM_MASK_CONTEXT;
                        let start = row % (CUSTOM_MASK_CONTEXT - CUSTOM_MASK_VISIBLE + 1);
                        if (start..start + CUSTOM_MASK_VISIBLE).contains(&token) {
                            (token % 2) as f32 * CUSTOM_MASK_BIAS
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect::<Vec<_>>(),
                (CUSTOM_MASK_BATCH, q_heads, 1, CUSTOM_MASK_CONTEXT),
                &cpu,
            )?
            .to_dtype(dtype)?;
            let mut expected = Vec::with_capacity(CUSTOM_MASK_BATCH);
            for batch in 0..CUSTOM_MASK_BATCH {
                let q = q.narrow(0, batch, 1)?.to_dtype(DType::F32)?.reshape((
                    kv_heads,
                    q_heads / kv_heads,
                    head_dim,
                ))?;
                let k = k.narrow(0, batch, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
                let v = v.narrow(0, batch, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
                let mask = mask.narrow(0, batch, 1)?.to_dtype(DType::F32)?.reshape((
                    kv_heads,
                    q_heads / kv_heads,
                    CUSTOM_MASK_CONTEXT,
                ))?;
                let logits = q.matmul(&k.t()?)?.broadcast_add(&mask)?;
                expected.push(
                    candle_nn::ops::softmax_last_dim(&logits)?
                        .matmul(&v)?
                        .reshape((1, q_heads, 1, head_dim))?,
                );
            }
            let params = SdpaParams {
                n_kv_groups: q_heads / kv_heads,
                softcap: None,
                softmax_scale: 1.0,
                sliding_window: None,
                sinks: None,
            };
            let q = q.to_device(&device)?;
            let k = k.to_device(&device)?;
            let v = v.to_device(&device)?;
            let mask = AttentionMask::Custom(mask.to_device(&device)?);
            let expected = Tensor::cat(&expected, 0)?;
            let actual = Sdpa.run_attention(&q, &k, &v, &mask, None, &params)?;
            let error = max_abs_diff(&actual, &expected)?;
            assert!(
                error < tolerance,
                "{dtype:?} head_dim={head_dim}: batch/head mask error {error}"
            );
            let blocks = CUSTOM_MASK_CONTEXT.div_ceil(BLOCK_SIZE);
            let mut key_blocks = Vec::with_capacity(CUSTOM_MASK_BATCH);
            let mut value_blocks = Vec::with_capacity(CUSTOM_MASK_BATCH);
            for batch in 0..CUSTOM_MASK_BATCH {
                let (key, value) = pack_seq(
                    &k.narrow(0, batch, 1)?.squeeze(0)?,
                    &v.narrow(0, batch, 1)?.squeeze(0)?,
                    blocks,
                )?;
                key_blocks.push(key);
                value_blocks.push(value);
            }
            let key_cache = Tensor::cat(&key_blocks, 0)?;
            let value_cache = Tensor::cat(&value_blocks, 0)?;
            for hnd in [false, true] {
                let (key_cache, value_cache) = cache_with_layout(&key_cache, &value_cache, hnd)?;
                let (key_cache, value_cache, block_tables) =
                    remap_cache_blocks(&key_cache, &value_cache, CUSTOM_MASK_BATCH, blocks)?;
                let context_lens =
                    Tensor::full(CUSTOM_MASK_CONTEXT as u32, CUSTOM_MASK_BATCH, &device)?;
                let mut metadata = PagedAttentionInputMetadata::dummy(&device)?;
                metadata.is_first_prompt_chunk = false;
                metadata.block_size = Some(BLOCK_SIZE);
                metadata.block_tables = Some(HashMap::from([(device.location(), block_tables)]));
                metadata.context_lens = Some(HashMap::from([(device.location(), context_lens)]));
                metadata.paged_context_lens_cpu =
                    Some(vec![CUSTOM_MASK_CONTEXT; CUSTOM_MASK_BATCH]);
                metadata.max_context_len = Some(CUSTOM_MASK_CONTEXT);
                metadata.query_lens = Some(vec![1; CUSTOM_MASK_BATCH]);
                let actual = PagedAttention::new(head_dim, &device, None)?.forward_donor_cache(
                    &q,
                    &key_cache,
                    &value_cache,
                    &mask,
                    &metadata,
                    &params,
                    None,
                )?;
                let error = max_abs_diff(&actual, &expected)?;
                assert!(
                    error < tolerance,
                    "{dtype:?} head_dim={head_dim} hnd={hnd}: paged batch/head mask error {error}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn sdpa_honors_offset_mask_views() -> Result<()> {
        use crate::attention::{AttentionMask, Sdpa, SdpaParams};

        let device = Device::new_metal(0)?;
        let (q_heads, kv_heads, head_dim, q_len, full_kv, kv_len) = (8, 2, 256, 79, 1103, 591);
        let q = Tensor::randn(0f32, 1., (1, q_heads, q_len, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1., (1, kv_heads, kv_len, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1., (1, kv_heads, kv_len, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let window = 512;
        let prefix = full_kv - q_len;
        let mut m = vec![0f32; q_len * full_kv];
        for i in 0..q_len {
            let qp = prefix + i;
            for j in 0..full_kv {
                if j > qp || j + window <= qp {
                    m[i * full_kv + j] = f32::NEG_INFINITY;
                }
            }
        }
        let full_mask = Tensor::from_vec(m, (q_len, full_kv), &device)?.to_dtype(DType::BF16)?;
        let view = full_mask.narrow(1, full_kv - kv_len, kv_len)?;
        let params = SdpaParams {
            n_kv_groups: q_heads / kv_heads,
            softcap: None,
            softmax_scale: 1. / (head_dim as f32).sqrt(),
            sliding_window: Some(window),
            sinks: None,
        };
        let from_view = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::Custom(view.clone()),
            None,
            &params,
        )?;
        let from_copy = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::Custom(view.force_contiguous()?),
            None,
            &params,
        )?;
        let diff = (from_view.to_dtype(DType::F32)? - from_copy.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-3, "max abs diff {diff}");
        Ok(())
    }
}
