#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;
use mistralrs_paged_attn::{metal_paged_prefill, MetalPagedPrefillRequest, MetalPagedPrefillShape};

const KEY_SCALE: f32 = 0.5;
const VALUE_SCALE: f32 = 0.25;
const BF16_TOLERANCE: f32 = 0.025;
const F16_TOLERANCE: f32 = 0.003;
const F32_TOLERANCE: f32 = 0.0003;
const LARGE_POSITION_ORIGIN: usize = 1 << 25;
const BF16_LARGE_SCALE: f32 = 1_048_576.0;
const BF16_LARGE_DOT_SCALE: f32 = 1024.0;
const LONG_PREFIX_SCORE_SCALE: f32 = 4.0;
const METADATA_BATCH_SIZES: [usize; 2] = [56, 57];
const BENCH_WARMUP_ITERATIONS: usize = 3;
const BENCH_DEFAULT_ITERATIONS: usize = 5;
const BENCH_ITERATIONS_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_ITERS";
const BENCH_DTYPES_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_DTYPES";
const BENCH_DIMS_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_DIMS";
const BENCH_QUERIES_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_QUERIES";
const BENCH_CONTEXTS_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_CONTEXTS";
const BENCH_BATCHES_ENV: &str = "MISTRALRS_METAL_PREFILL_BENCH_BATCHES";

#[derive(Clone, Copy, Debug, Default)]
enum MaskLayout {
    #[default]
    Broadcast,
    PerHead,
    PerBatchHead,
    Packed,
}

#[derive(Clone, Copy, Debug)]
struct Sequence {
    query: usize,
    context: usize,
    key_start: usize,
    query_start: usize,
}

const SEQUENCES: &[Sequence] = &[
    Sequence {
        query: 3,
        context: 17,
        key_start: 0,
        query_start: 14,
    },
    Sequence {
        query: 9,
        context: 65,
        key_start: 96,
        query_start: 149,
    },
    Sequence {
        query: 17,
        context: 97,
        key_start: 32,
        query_start: 112,
    },
];

const TILE_SEQUENCES: &[Sequence] = &[
    Sequence {
        query: 65,
        context: 129,
        key_start: 0,
        query_start: 64,
    },
    Sequence {
        query: 33,
        context: 97,
        key_start: 64,
        query_start: 128,
    },
];

#[derive(Clone, Copy, Debug)]
struct Case {
    key_dim: usize,
    value_dim: usize,
    heads: usize,
    kv_heads: usize,
    block: usize,
    dtype: DType,
    fp8: bool,
    hnd: bool,
    packed: bool,
    causal: bool,
    window: Option<usize>,
    window_right: Option<usize>,
    attention_chunk_size: Option<usize>,
    features: bool,
    custom_mask: bool,
    mask_layout: MaskLayout,
    masked_rows: bool,
    strided_caches: bool,
    multimodal: bool,
    historical_multimodal: bool,
    input_scales: [f32; 3],
    benchmark_iterations: usize,
}

impl Default for Case {
    fn default() -> Self {
        Self {
            key_dim: 64,
            value_dim: 64,
            heads: 4,
            kv_heads: 2,
            block: 32,
            dtype: DType::BF16,
            fp8: false,
            hnd: true,
            packed: false,
            causal: true,
            window: None,
            window_right: None,
            attention_chunk_size: None,
            features: false,
            custom_mask: false,
            mask_layout: MaskLayout::Broadcast,
            masked_rows: false,
            strided_caches: false,
            multimodal: false,
            historical_multimodal: false,
            input_scales: [1.0; 3],
            benchmark_iterations: 0,
        }
    }
}

fn rounded(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => half::bf16::from_f32(value).to_f32(),
        DType::F16 => half::f16::from_f32(value).to_f32(),
        DType::F32 => value,
        _ => unreachable!(),
    }
}

fn signal(index: usize, seed: usize, dtype: DType) -> f32 {
    rounded(((index * seed + seed) as f32).sin() * 0.6, dtype)
}

fn cache_value(value: f32, scale: f32, case: Case) -> f32 {
    if case.fp8 {
        F8E4M3::from_f32(value / scale).to_f32() * scale
    } else {
        value
    }
}

fn tensor(values: Vec<f32>, shape: Vec<usize>, dtype: DType, device: &Device) -> Result<Tensor> {
    Tensor::from_vec(values, shape, &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)
}

fn mask_value(case: Case, seq: usize, head: usize, query: usize, key: usize) -> f32 {
    if case.masked_rows && query == 0 {
        return f32::NEG_INFINITY;
    }
    let seq = if matches!(
        case.mask_layout,
        MaskLayout::PerBatchHead | MaskLayout::Packed
    ) {
        seq
    } else {
        0
    };
    let head = if matches!(case.mask_layout, MaskLayout::Broadcast) {
        0
    } else {
        head
    };
    if (query + 2 * key + 3 * seq + 5 * head).is_multiple_of(7) {
        -3.0 - head as f32 * 0.5 - seq as f32 * 0.25
    } else {
        0.0
    }
}

fn check_case(device: &Device, case: Case, sequences: &[Sequence]) -> Result<()> {
    let collect_reference = case.benchmark_iterations == 0;
    let batch = sequences.len();
    let max_q = sequences.iter().map(|s| s.query).max().unwrap();
    let max_k = sequences.iter().map(|s| s.context).max().unwrap();
    let total_q: usize = sequences.iter().map(|s| s.query).sum();
    let max_blocks = max_k.div_ceil(case.block);
    let blocks = batch * max_blocks;
    let cache_dtype = if case.fp8 { DType::F8E4M3 } else { case.dtype };
    let x = 16 / cache_dtype.size_in_bytes();
    let head_offset = usize::from(case.strided_caches);
    let cache_heads = case.kv_heads + head_offset;
    let key_shape = if case.hnd {
        vec![blocks + 1, cache_heads, case.block, case.key_dim]
    } else {
        vec![blocks + 1, cache_heads, case.key_dim / x, case.block, x]
    };
    let value_shape = if case.hnd {
        vec![blocks + 1, cache_heads, case.block, case.value_dim]
    } else {
        vec![blocks + 1, cache_heads, case.value_dim, case.block]
    };
    let mut keys = vec![f32::NAN; key_shape.iter().product()];
    let mut values = vec![f32::NAN; value_shape.iter().product()];
    let mut table = vec![0u32; (batch + 1) * max_blocks];
    let mut logical_keys = Vec::new();
    let mut logical_values = Vec::new();
    for (seq, sequence) in sequences.iter().enumerate() {
        for page in 0..max_blocks {
            table[(seq + 1) * max_blocks + page] = (blocks - 1 - seq * max_blocks - page) as u32;
        }
        let mut k_reference = Vec::new();
        let mut v_reference = Vec::new();
        for token in 0..sequence.context {
            let physical = table[(seq + 1) * max_blocks + token / case.block] as usize + 1;
            let slot = token % case.block;
            for head in 0..case.kv_heads {
                for dim in 0..case.key_dim {
                    let value = signal(
                        ((seq * max_k + token) * case.kv_heads + head) * case.key_dim + dim,
                        3,
                        case.dtype,
                    ) * case.input_scales[1];
                    if collect_reference {
                        k_reference.push(cache_value(value, KEY_SCALE, case));
                    }
                    let index = if case.hnd {
                        ((physical * cache_heads + head + head_offset) * case.block + slot)
                            * case.key_dim
                            + dim
                    } else {
                        (((physical * cache_heads + head + head_offset) * (case.key_dim / x)
                            + dim / x)
                            * case.block
                            + slot)
                            * x
                            + dim % x
                    };
                    keys[index] = if case.fp8 { value / KEY_SCALE } else { value };
                }
                for dim in 0..case.value_dim {
                    let value = signal(
                        ((seq * max_k + token) * case.kv_heads + head) * case.value_dim + dim,
                        7,
                        case.dtype,
                    ) * case.input_scales[2];
                    if collect_reference {
                        v_reference.push(cache_value(value, VALUE_SCALE, case));
                    }
                    let index = if case.hnd {
                        ((physical * cache_heads + head + head_offset) * case.block + slot)
                            * case.value_dim
                            + dim
                    } else {
                        ((physical * cache_heads + head + head_offset) * case.value_dim + dim)
                            * case.block
                            + slot
                    };
                    values[index] = if case.fp8 { value / VALUE_SCALE } else { value };
                }
            }
        }
        logical_keys.push(k_reference);
        logical_values.push(v_reference);
    }
    let key_cache = tensor(keys, key_shape, cache_dtype, device)?
        .narrow(0, 1, blocks)?
        .narrow(1, head_offset, case.kv_heads)?;
    let value_cache = tensor(values, value_shape, cache_dtype, device)?
        .narrow(0, 1, blocks)?
        .narrow(1, head_offset, case.kv_heads)?;
    let block_tables =
        Tensor::from_vec(table, (batch + 1, max_blocks), device)?.narrow(0, 1, batch)?;
    let (query_batch, query_rows) = if case.packed {
        (1, total_q)
    } else {
        (batch, max_q)
    };
    let mut queries = vec![f32::NAN; query_batch * case.heads * query_rows * (case.key_dim + 1)];
    let mut reference_queries = Vec::new();
    let mut q_offset = 0;
    for (seq, sequence) in sequences.iter().enumerate() {
        let mut q_reference = Vec::new();
        for q in 0..sequence.query {
            for head in 0..case.heads {
                for dim in 0..case.key_dim {
                    let value = signal(
                        ((seq * max_q + q) * case.heads + head) * case.key_dim + dim,
                        11,
                        case.dtype,
                    ) * case.input_scales[0];
                    if collect_reference {
                        q_reference.push(value);
                    }
                    let (b, row) = if case.packed {
                        (0, q_offset + q)
                    } else {
                        (seq, q)
                    };
                    queries[((b * case.heads + head) * query_rows + row) * (case.key_dim + 1)
                        + dim
                        + 1] = value;
                }
            }
        }
        q_offset += sequence.query;
        reference_queries.push(q_reference);
    }
    let query = tensor(
        queries,
        vec![query_batch, case.heads, query_rows, case.key_dim + 1],
        case.dtype,
        device,
    )?
    .narrow(3, 1, case.key_dim)?;
    let query_lens: Vec<_> = sequences.iter().map(|s| s.query).collect();
    let context_lens: Vec<_> = sequences.iter().map(|s| s.context).collect();
    let query_starts: Vec<_> = sequences.iter().map(|s| s.query_start).collect();
    let key_starts: Vec<_> = sequences.iter().map(|s| s.key_start).collect();
    let ranges: Vec<_> = sequences
        .iter()
        .map(|s| {
            let start = if case.historical_multimodal {
                s.key_start
            } else {
                s.query_start
            };
            vec![(start, start + s.query.min(5))]
        })
        .collect();
    let slopes: Vec<_> = (0..case.heads).map(|h| 0.02 * (h + 1) as f32).collect();
    let sink_values: Vec<_> = (0..case.heads).map(|h| 0.1 * h as f32 - 0.5).collect();
    let alibi = case
        .features
        .then(|| Tensor::new(slopes.clone(), device))
        .transpose()?;
    let sinks = case
        .features
        .then(|| Tensor::new(sink_values.clone(), device))
        .transpose()?;
    let k_scale = Tensor::new(&[KEY_SCALE], device)?;
    let v_scale = Tensor::new(&[VALUE_SCALE], device)?;
    let additive = if case.custom_mask {
        let packed_mask = matches!(case.mask_layout, MaskLayout::Packed);
        let mask_q = if packed_mask { total_q } else { max_q };
        let mask_k = if packed_mask {
            sequences.iter().map(|s| s.context).sum()
        } else {
            max_k
        };
        let mask_heads = if matches!(case.mask_layout, MaskLayout::Broadcast) {
            1
        } else {
            case.heads
        };
        let mask_batch = if matches!(case.mask_layout, MaskLayout::PerBatchHead) {
            batch
        } else {
            1
        };
        let mut shape = match case.mask_layout {
            MaskLayout::Broadcast => vec![mask_q + 1, mask_k + 1],
            MaskLayout::PerHead => vec![mask_heads, mask_q + 1, mask_k + 1],
            MaskLayout::PerBatchHead | MaskLayout::Packed => {
                vec![mask_batch, mask_heads, mask_q + 1, mask_k + 1]
            }
        };
        let mut data = vec![f32::NAN; shape.iter().product()];
        let mut q_start = 0;
        let mut k_start = 0;
        for (seq, sequence) in sequences.iter().enumerate() {
            if !packed_mask && mask_batch == 1 && seq > 0 {
                break;
            }
            let (query_count, key_count) = if packed_mask {
                (sequence.query, sequence.context)
            } else {
                (max_q, max_k)
            };
            for head in 0..mask_heads {
                for query in 0..query_count {
                    for key in 0..key_count {
                        let mask_seq = if mask_batch == 1 { 0 } else { seq };
                        let index =
                            ((mask_seq * mask_heads + head) * (mask_q + 1) + q_start + query + 1)
                                * (mask_k + 1)
                                + k_start
                                + key
                                + 1;
                        data[index] = mask_value(case, seq, head, query, key);
                    }
                }
            }
            if packed_mask {
                q_start += sequence.query;
                k_start += sequence.context;
            }
        }
        let rank = shape.len();
        Some(
            tensor(data, std::mem::take(&mut shape), case.dtype, device)?
                .narrow(rank - 2, 1, mask_q)?
                .narrow(rank - 1, 1, mask_k)?,
        )
    } else {
        None
    };
    let scale = 1.0 / (case.key_dim as f32).sqrt();
    let softcap = case.features.then_some(2.0);
    let execute = || {
        metal_paged_prefill(MetalPagedPrefillRequest {
            query: &query,
            key_cache: &key_cache,
            value_cache: &value_cache,
            block_tables: &block_tables,
            query_lens: &query_lens,
            context_lens: &context_lens,
            query_start_positions: &query_starts,
            key_start_positions: &key_starts,
            packed_queries: case.packed,
            causal: case.causal,
            sliding_window: case.window,
            window_right: case.window_right,
            attention_chunk_size: case.attention_chunk_size,
            multimodal_ranges: case.multimodal.then_some(ranges.as_slice()),
            additive_mask: additive.as_ref(),
            packed_mask: matches!(case.mask_layout, MaskLayout::Packed),
            mask_key_offsets: None,
            alibi_slopes: alibi.as_ref(),
            sinks: sinks.as_ref(),
            k_scale: case.fp8.then_some(&k_scale),
            v_scale: case.fp8.then_some(&v_scale),
            scale,
            softcap,
        })
    };
    if case.benchmark_iterations > 0 {
        for _ in 0..BENCH_WARMUP_ITERATIONS {
            let output = execute()?;
            device.synchronize()?;
            std::hint::black_box(output);
        }
        device.synchronize()?;
        let start = std::time::Instant::now();
        for _ in 0..case.benchmark_iterations {
            let output = execute()?;
            device.synchronize()?;
            std::hint::black_box(output);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "metal_prefill_bench,{:?},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6}",
            case.dtype,
            case.hnd,
            case.packed,
            batch,
            case.heads,
            case.kv_heads,
            case.key_dim,
            case.value_dim,
            max_q,
            max_k,
            total_q,
            sequences.iter().map(|s| s.context).sum::<usize>(),
            BENCH_WARMUP_ITERATIONS,
            case.benchmark_iterations,
            elapsed_ms / case.benchmark_iterations as f64,
        );
        return Ok(());
    }
    let output = execute()?;
    assert_eq!(
        output.dims(),
        &[query_batch, case.heads, query_rows, case.value_dim]
    );
    let output = output
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let tolerance = match case.dtype {
        DType::BF16 => BF16_TOLERANCE,
        DType::F16 => F16_TOLERANCE,
        _ => F32_TOLERANCE,
    };
    q_offset = 0;
    for (seq, sequence) in sequences.iter().enumerate() {
        for q in 0..sequence.query {
            let q_position = sequence.query_start + q;
            for head in 0..case.heads {
                let kv_head = head / (case.heads / case.kv_heads);
                let mut scores = Vec::new();
                for k in 0..sequence.context {
                    let k_position = sequence.key_start + k;
                    let bidirectional = case.multimodal
                        && ranges[seq].iter().any(|&(start, end)| {
                            (start..end).contains(&q_position) && (start..end).contains(&k_position)
                        });
                    let permitted = (!case.causal || k_position <= q_position)
                        && case
                            .window
                            .is_none_or(|w| k_position.saturating_add(w) > q_position)
                        && case
                            .window_right
                            .is_none_or(|w| k_position < q_position.saturating_add(w))
                        || bidirectional;
                    let permitted = permitted
                        && case
                            .attention_chunk_size
                            .is_none_or(|chunk| k_position / chunk == q_position / chunk);
                    if !permitted {
                        scores.push(f64::NEG_INFINITY);
                        continue;
                    }
                    let mut score: f64 = (0..case.key_dim)
                        .map(|dim| {
                            f64::from(
                                reference_queries[seq]
                                    [(q * case.heads + head) * case.key_dim + dim],
                            ) * f64::from(
                                logical_keys[seq]
                                    [(k * case.kv_heads + kv_head) * case.key_dim + dim],
                            )
                        })
                        .sum::<f64>()
                        * f64::from(scale);
                    if let Some(cap) = softcap {
                        score = (score / f64::from(cap)).tanh() * f64::from(cap);
                    }
                    if case.features {
                        score += f64::from(slopes[head]) * (k_position as f64 - q_position as f64);
                    }
                    if case.custom_mask {
                        score += f64::from(mask_value(case, seq, head, q, k));
                    }
                    scores.push(score);
                }
                let sink = if case.features {
                    f64::from(sink_values[head])
                } else {
                    f64::NEG_INFINITY
                };
                let max = scores.iter().copied().fold(sink, f64::max);
                let weight = |score: f64| {
                    if score == f64::NEG_INFINITY {
                        0.0
                    } else {
                        (score - max).exp()
                    }
                };
                let denominator = scores.iter().map(|&s| weight(s)).sum::<f64>() + weight(sink);
                let (b, row) = if case.packed {
                    (0, q_offset + q)
                } else {
                    (seq, q)
                };
                for dim in 0..case.value_dim {
                    let expected = if denominator == 0.0 {
                        0.0
                    } else {
                        scores
                            .iter()
                            .enumerate()
                            .map(|(k, score)| {
                                weight(*score)
                                    * f64::from(
                                        logical_values[seq]
                                            [(k * case.kv_heads + kv_head) * case.value_dim + dim],
                                    )
                            })
                            .sum::<f64>()
                            / denominator
                    };
                    let actual =
                        output[((b * case.heads + head) * query_rows + row) * case.value_dim + dim];
                    let normalized_error = (actual - expected as f32).abs() / case.input_scales[2];
                    assert!(actual.is_finite() && normalized_error < tolerance, "{case:?} seq={seq} q={q} head={head} dim={dim}: actual={actual} expected={expected}");
                }
            }
        }
        q_offset += sequence.query;
    }
    device.synchronize()?;
    Ok(())
}

#[test]
fn metal_prefill_matches_reference_across_geometry_and_layouts() -> Result<()> {
    let device = Device::new_metal(0)?;
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for (key_dim, value_dim) in [
            (64usize, 64usize),
            (80, 96),
            (128, 64),
            (256, 256),
            (384, 512),
            (512, 256),
            (511, 497),
            (512, 512),
        ] {
            for hnd in [true, false] {
                if !hnd && !key_dim.is_multiple_of(16 / dtype.size_in_bytes()) {
                    continue;
                }
                check_case(
                    &device,
                    Case {
                        key_dim,
                        value_dim,
                        dtype,
                        hnd,
                        ..Case::default()
                    },
                    SEQUENCES,
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn metal_prefill_preserves_metadata_across_inline_and_buffered_batches() -> Result<()> {
    let device = Device::new_metal(0)?;
    for batch in METADATA_BATCH_SIZES {
        check_case(&device, Case::default(), &vec![SEQUENCES[0]; batch])?;
    }
    Ok(())
}

#[test]
fn metal_prefill_preserves_windows_ranges_biases_and_quantized_kv() -> Result<()> {
    let device = Device::new_metal(0)?;
    for packed in [false, true] {
        for hnd in [true, false] {
            for dtype in [DType::F16, DType::BF16] {
                for window in [None, Some(1), Some(31), Some(65)] {
                    check_case(
                        &device,
                        Case {
                            dtype,
                            hnd,
                            packed,
                            window,
                            features: true,
                            fp8: true,
                            multimodal: true,
                            block: 16,
                            ..Case::default()
                        },
                        SEQUENCES,
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn metal_prefill_preserves_windows_with_historical_multimodal_ranges() -> Result<()> {
    let device = Device::new_metal(0)?;
    for dim in [64, 512] {
        for historical_multimodal in [false, true] {
            check_case(
                &device,
                Case {
                    key_dim: dim,
                    value_dim: dim,
                    window: Some(3),
                    multimodal: true,
                    historical_multimodal,
                    ..Case::default()
                },
                &[Sequence {
                    query: 65,
                    context: 513,
                    key_start: 256,
                    query_start: 704,
                }],
            )?;
        }
    }
    Ok(())
}

#[test]
fn metal_prefill_accepts_noncausal_and_custom_attention() -> Result<()> {
    let device = Device::new_metal(0)?;
    for causal in [false, true] {
        for custom_mask in [false, true] {
            check_case(
                &device,
                Case {
                    causal,
                    custom_mask,
                    features: true,
                    heads: 6,
                    kv_heads: 2,
                    block: 8,
                    ..Case::default()
                },
                SEQUENCES,
            )?;
        }
    }
    check_case(
        &device,
        Case {
            causal: false,
            window: Some(5),
            window_right: Some(5),
            ..Case::default()
        },
        SEQUENCES,
    )?;
    for chunk in [16, 32] {
        check_case(
            &device,
            Case {
                attention_chunk_size: Some(chunk),
                ..Case::default()
            },
            SEQUENCES,
        )?;
    }
    Ok(())
}

#[test]
fn metal_prefill_matches_reference_across_query_tiles() -> Result<()> {
    let device = Device::new_metal(0)?;
    for key_dim in [64, 128, 256, 512] {
        for dtype in [DType::F16, DType::BF16] {
            for packed in [false, true] {
                check_case(
                    &device,
                    Case {
                        key_dim,
                        value_dim: key_dim,
                        dtype,
                        packed,
                        heads: 2,
                        kv_heads: 1,
                        ..Case::default()
                    },
                    TILE_SEQUENCES,
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn metal_prefill_workspace_does_not_grow_with_cached_context() -> Result<()> {
    let short = MetalPagedPrefillShape {
        num_seqs: 4,
        num_heads: 8,
        num_kv_heads: 2,
        max_query_len: 4096,
        total_query_tokens: 4096,
        max_context_len: 4096,
        key_head_dim: 512,
        value_head_dim: 512,
        block_size: 32,
        num_multimodal_ranges: 4,
        query_dtype: DType::BF16,
        cache_dtype: DType::BF16,
        packed_queries: true,
        hnd: true,
    };
    let long = MetalPagedPrefillShape {
        max_context_len: 262144,
        ..short
    };
    assert!(short.supports() && long.supports());
    assert_eq!(short.workspace_bytes()?, long.workspace_bytes()?);
    let output_bytes = short.total_query_tokens
        * short.num_heads
        * short.value_head_dim
        * short.query_dtype.size_in_bytes();
    assert!(long.workspace_bytes()? < output_bytes * 2);
    Ok(())
}

#[test]
fn metal_prefill_vector_handles_single_queries_large_values_and_odd_dimensions() -> Result<()> {
    let device = Device::new_metal(0)?;
    let single_queries = [
        Sequence {
            query: 1,
            context: 33,
            key_start: 64,
            query_start: 96,
        },
        Sequence {
            query: 1,
            context: 17,
            key_start: 0,
            query_start: 16,
        },
    ];
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for hnd in [false, true] {
            check_case(
                &device,
                Case {
                    key_dim: 128,
                    value_dim: 768,
                    dtype,
                    hnd,
                    strided_caches: true,
                    features: true,
                    ..Case::default()
                },
                &single_queries,
            )?;
            check_case(
                &device,
                Case {
                    key_dim: 64,
                    value_dim: 768,
                    dtype,
                    hnd,
                    packed: true,
                    window: Some(31),
                    fp8: true,
                    strided_caches: true,
                    ..Case::default()
                },
                SEQUENCES,
            )?;
        }
        for sequences in [&single_queries[..], SEQUENCES] {
            check_case(
                &device,
                Case {
                    key_dim: 7,
                    value_dim: 13,
                    dtype,
                    heads: 6,
                    kv_heads: 2,
                    strided_caches: true,
                    ..Case::default()
                },
                sequences,
            )?;
        }
    }
    check_case(
        &device,
        Case {
            key_dim: 4096,
            value_dim: 4096,
            heads: 1,
            kv_heads: 1,
            dtype: DType::F32,
            ..Case::default()
        },
        &[Sequence {
            query: 1,
            context: 3,
            key_start: 0,
            query_start: 2,
        }],
    )?;
    Ok(())
}

#[test]
fn metal_prefill_preserves_bf16_dynamic_range_and_f32_accumulation() -> Result<()> {
    let device = Device::new_metal(0)?;
    for dim in [64, 128, 256, 512] {
        for hnd in [false, true] {
            for input_scales in [
                [BF16_LARGE_SCALE, BF16_LARGE_SCALE.recip(), BF16_LARGE_SCALE],
                [BF16_LARGE_SCALE.recip(), BF16_LARGE_SCALE, BF16_LARGE_SCALE],
            ] {
                check_case(
                    &device,
                    Case {
                        key_dim: dim,
                        value_dim: dim,
                        hnd,
                        input_scales,
                        strided_caches: true,
                        custom_mask: true,
                        mask_layout: MaskLayout::PerBatchHead,
                        features: true,
                        ..Case::default()
                    },
                    SEQUENCES,
                )?;
            }
        }
        check_case(
            &device,
            Case {
                key_dim: dim,
                value_dim: dim,
                packed: true,
                input_scales: [BF16_LARGE_DOT_SCALE, BF16_LARGE_DOT_SCALE, 1.0],
                ..Case::default()
            },
            TILE_SEQUENCES,
        )?;
    }
    Ok(())
}

#[test]
fn metal_prefill_matches_reference_across_long_cached_prefixes() -> Result<()> {
    let device = Device::new_metal(0)?;
    for (dim, dtype, fp8, hnd, query, context) in [
        (128, DType::F32, false, true, 2, 4096),
        (256, DType::F16, false, false, 9, 2048),
        (512, DType::BF16, false, true, 9, 4096),
        (128, DType::BF16, true, true, 9, 4096),
    ] {
        check_case(
            &device,
            Case {
                key_dim: dim,
                value_dim: dim,
                dtype,
                fp8,
                hnd,
                input_scales: [LONG_PREFIX_SCORE_SCALE, LONG_PREFIX_SCORE_SCALE, 1.0],
                ..Case::default()
            },
            &[Sequence {
                query,
                context,
                key_start: 0,
                query_start: context - query,
            }],
        )?;
    }
    Ok(())
}

#[test]
fn metal_prefill_preserves_alibi_distances_with_large_absolute_origins() -> Result<()> {
    let device = Device::new_metal(0)?;
    let sequences: Vec<_> = SEQUENCES
        .iter()
        .map(|s| Sequence {
            query_start: s.query_start + LARGE_POSITION_ORIGIN,
            key_start: s.key_start + LARGE_POSITION_ORIGIN,
            ..*s
        })
        .collect();
    for value_dim in [64, 768] {
        check_case(
            &device,
            Case {
                value_dim,
                dtype: DType::F32,
                features: true,
                causal: false,
                ..Case::default()
            },
            &sequences,
        )?;
    }
    Ok(())
}

#[test]
fn metal_prefill_reads_broadcast_packed_and_strided_masks_and_all_masked_rows() -> Result<()> {
    let device = Device::new_metal(0)?;
    for mask_layout in [
        MaskLayout::Broadcast,
        MaskLayout::PerHead,
        MaskLayout::PerBatchHead,
        MaskLayout::Packed,
    ] {
        for features in [false, true] {
            for value_dim in [64, 768] {
                check_case(
                    &device,
                    Case {
                        value_dim,
                        dtype: DType::F32,
                        heads: 6,
                        kv_heads: 2,
                        packed: matches!(mask_layout, MaskLayout::Packed),
                        custom_mask: true,
                        mask_layout,
                        masked_rows: true,
                        features,
                        strided_caches: true,
                        ..Case::default()
                    },
                    SEQUENCES,
                )?;
            }
        }
    }
    Ok(())
}

fn benchmark_sizes(name: &str, defaults: &[usize]) -> Result<Vec<usize>> {
    let Ok(value) = std::env::var(name) else {
        return Ok(defaults.to_vec());
    };
    value
        .split(',')
        .map(|value| {
            let parsed = value
                .trim()
                .parse::<usize>()
                .map_err(candle_core::Error::wrap)?;
            if parsed == 0 {
                candle_core::bail!("{name} entries must be nonzero");
            }
            Ok(parsed)
        })
        .collect()
}

#[test]
fn metal_prefill_handles_non_power_of_two_cache_blocks() -> Result<()> {
    let device = Device::new_metal(0)?;
    for key_dim in [64, 512] {
        for hnd in [true, false] {
            check_case(
                &device,
                Case {
                    key_dim,
                    value_dim: key_dim,
                    packed: true,
                    custom_mask: true,
                    block: 7,
                    hnd,
                    ..Case::default()
                },
                &[Sequence {
                    query: 9,
                    context: 45,
                    query_start: 36,
                    key_start: 0,
                }],
            )?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "exclusive Metal microbenchmark; includes host dispatch, allocation, and synchronization"]
fn metal_prefill_kernel_microbenchmark() -> Result<()> {
    let iterations = std::env::var(BENCH_ITERATIONS_ENV)
        .ok()
        .map(|value| value.parse::<usize>().map_err(candle_core::Error::wrap))
        .transpose()?
        .unwrap_or(BENCH_DEFAULT_ITERATIONS);
    if iterations == 0 {
        candle_core::bail!("{BENCH_ITERATIONS_ENV} must be nonzero");
    }
    let dtypes = std::env::var(BENCH_DTYPES_ENV)
        .unwrap_or_else(|_| "bf16".into())
        .split(',')
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "f16" => Ok(DType::F16),
            "bf16" => Ok(DType::BF16),
            "f32" => Ok(DType::F32),
            _ => candle_core::bail!("{BENCH_DTYPES_ENV} entries must be f16, bf16, or f32"),
        })
        .collect::<Result<Vec<_>>>()?;
    let dims = benchmark_sizes(BENCH_DIMS_ENV, &[64, 128, 256, 512])?;
    let queries = benchmark_sizes(BENCH_QUERIES_ENV, &[1, 32, 512])?;
    let contexts = benchmark_sizes(BENCH_CONTEXTS_ENV, &[512, 4096, 32768])?;
    let batches = benchmark_sizes(BENCH_BATCHES_ENV, &[1, 4])?;
    if batches.iter().any(|&batch| !matches!(batch, 1 | 4)) {
        candle_core::bail!("{BENCH_BATCHES_ENV} entries must be 1 or 4");
    }
    let device = Device::new_metal(0)?;
    println!("kind,dtype,hnd,packed,batch,heads,kv_heads,key_dim,value_dim,max_q,max_k,total_q,total_k,warmup,iterations,host_dispatch_alloc_sync_ms");
    for dtype in dtypes {
        for &dim in &dims {
            for &query in &queries {
                for &context in &contexts {
                    if query > context {
                        candle_core::bail!(
                            "benchmark query length {query} exceeds context {context}"
                        );
                    }
                    let case = Case {
                        key_dim: dim,
                        value_dim: dim,
                        heads: 8,
                        kv_heads: 2,
                        dtype,
                        benchmark_iterations: iterations,
                        ..Case::default()
                    };
                    if batches.contains(&1) {
                        check_case(
                            &device,
                            case,
                            &[Sequence {
                                query,
                                context,
                                query_start: context - query,
                                key_start: 0,
                            }],
                        )?;
                    }
                    if batches.contains(&4) && context <= 4096 && query > 1 {
                        let sequences: Vec<_> = (0..4)
                            .map(|row| {
                                let query = (query >> row).max(1);
                                let context = context.saturating_sub(row * 32).max(query);
                                Sequence {
                                    query,
                                    context,
                                    query_start: context - query,
                                    key_start: 0,
                                }
                            })
                            .collect();
                        check_case(&device, case, &sequences)?;
                    }
                }
            }
        }
    }
    Ok(())
}
