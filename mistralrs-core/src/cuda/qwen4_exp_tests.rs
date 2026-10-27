use super::*;
use candle_core::{cuda_backend::cudarc::driver::sys, Var};

const BLOCK_SIZE: usize = 4;
const RATIO: usize = 4;
const HALF_ROT: usize = 32;
const TOKENS: usize = 8;
const SLOTS: usize = 12;
const REPLACED_TOKENS: usize = 2;
const DENSE_TEST_BLOCK_SIZE: usize = 32;
const DENSE_TEST_Q_HEADS: usize = 24;
const DENSE_TEST_KV_HEADS: usize = 2;
const DENSE_TEST_CONTEXTS: &[usize] = &[
    0, 1, 3, 4, 7, 31, 32, 33, 511, 512, 513, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2079, 2080,
];
const DENSE_TEST_CASES: &[(usize, usize, usize)] = &[(2048, 1, 1), (2080, 1, 1), (2080, 2, 3)];

#[test]
#[ignore = "requires a CUDA device"]
fn qsa_dense_graph_matches_identity_selection() -> Result<()> {
    use crate::pipeline::cuda_graph::CudaGraphHandle;

    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    for dtype in [DType::BF16, DType::F16] {
        for &(capacity, batch, query_len) in DENSE_TEST_CASES {
            let blocks = capacity / DENSE_TEST_BLOCK_SIZE;
            let n_tokens = batch * query_len;
            let cache_shape = (
                batch * blocks,
                DENSE_TEST_KV_HEADS,
                DENSE_TEST_BLOCK_SIZE,
                ATTN_HEAD_DIM,
            );
            let cache_len =
                batch * blocks * DENSE_TEST_KV_HEADS * DENSE_TEST_BLOCK_SIZE * ATTN_HEAD_DIM;
            let key_cache = Tensor::from_vec(
                (0..cache_len)
                    .map(|i| ((i * 71 % 997) as f32 - 498.0) / 498.0)
                    .collect(),
                cache_shape,
                &device,
            )?
            .to_dtype(dtype)?;
            let value_cache = Tensor::from_vec(
                (0..cache_len)
                    .map(|i| ((i * 37 % 1009) as f32 - 504.0) / 504.0)
                    .collect(),
                cache_shape,
                &device,
            )?
            .to_dtype(dtype)?;
            let q = Tensor::from_vec(
                (0..n_tokens * DENSE_TEST_Q_HEADS * ATTN_HEAD_DIM)
                    .map(|i| ((i * 29 % 991) as f32 - 495.0) / 495.0)
                    .collect(),
                (n_tokens, DENSE_TEST_Q_HEADS, ATTN_HEAD_DIM),
                &device,
            )?
            .to_dtype(dtype)?;
            let tables = Tensor::from_vec(
                (0..batch)
                    .flat_map(|seq| {
                        (0..blocks).map(move |block| (seq * blocks + blocks - block - 1) as u32)
                    })
                    .collect(),
                (batch, blocks),
                &device,
            )?;
            let kv_lens = Var::from_vec(vec![capacity as u32; batch], batch, &device)?;
            let layout = TokenLayout::rectangular(batch, query_len);
            let paged = PagedView {
                block_tables: &tables,
                kv_lens: kv_lens.as_tensor(),
                block_size: DENSE_TEST_BLOCK_SIZE,
            };
            let shape = QsaShape {
                ratio: RATIO,
                topk: capacity / RATIO,
                half_rot: HALF_ROT,
                n_index_heads: 1,
            };
            let selected = Tensor::from_vec(
                (0..n_tokens)
                    .flat_map(|_| (0..shape.topk).map(|block| block as i32))
                    .collect(),
                (n_tokens, shape.topk),
                &device,
            )?;
            let n_selected = Var::from_vec(
                (0..n_tokens)
                    .map(|t| ((capacity - query_len + t % query_len + 1) / RATIO) as i32)
                    .collect(),
                n_tokens,
                &device,
            )?;
            let attend = |selection| {
                qsa_attention(QsaAttentionArgs {
                    q: &q,
                    key_cache: &key_cache,
                    value_cache: &value_cache,
                    layout: &layout,
                    paged: &paged,
                    selection,
                    shape: &shape,
                    n_kv_heads: DENSE_TEST_KV_HEADS,
                    scale: 1.0 / (ATTN_HEAD_DIM as f32).sqrt(),
                })
            };
            let identity = Some((&selected, n_selected.as_tensor()));
            let _ = attend(None)?;
            let _ = attend(identity)?;
            device.synchronize()?;
            let tracking = stream.context().is_event_tracking();
            if tracking {
                unsafe { stream.context().disable_event_tracking() };
            }
            if let Err(error) =
                stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            {
                if tracking {
                    unsafe { stream.context().enable_event_tracking() };
                }
                return Err(candle_core::Error::msg(error));
            }
            let captured: Result<(Tensor, Tensor)> = (|| Ok((attend(None)?, attend(identity)?)))();
            let graph = CudaGraphHandle::end_capture(&stream);
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            let (dense, sparse) = captured?;
            let graph =
                graph?.ok_or_else(|| candle_core::Error::msg("QSA capture returned no graph"))?;
            for &context in DENSE_TEST_CONTEXTS
                .iter()
                .filter(|&&context| context <= capacity)
            {
                let lengths = (0..batch)
                    .map(|seq| context.saturating_sub(seq) as u32)
                    .collect::<Vec<_>>();
                let counts = lengths
                    .iter()
                    .flat_map(|&len| {
                        (0..query_len).map(move |t| {
                            ((len as usize + t + 1).saturating_sub(query_len) / RATIO) as i32
                        })
                    })
                    .collect::<Vec<_>>();
                kv_lens.set(&Tensor::from_vec(lengths, batch, &device)?)?;
                n_selected.set(&Tensor::from_vec(counts, n_tokens, &device)?)?;
                graph.launch()?;
                device.synchronize()?;
                let dense = dense
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let sparse = sparse
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let tolerance = if dtype == DType::BF16 { 0.002 } else { 0.0002 };
                let mut max_error = 0.0f32;
                for (&a, &b) in dense.iter().zip(&sparse) {
                    assert!(a.is_finite() && b.is_finite());
                    max_error = max_error.max((a - b).abs());
                }
                assert!(max_error <= tolerance,
                        "{dtype:?} capacity={capacity} batch={batch} query_len={query_len} context={context} max_error={max_error}");
                if context == 0 {
                    assert!(dense.iter().all(|&value| value == 0.0));
                } else {
                    assert!(dense.iter().any(|&value| value.abs() > 0.001));
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn qsa_reopened_block_matches_fresh_prefill() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;
    let aux_dim = QSA_HEAD_DIM + 2 * HALF_ROT;
    let head_dim = u32::try_from(QSA_HEAD_DIM).unwrap();
    let half_rot = u32::try_from(HALF_ROT).unwrap();
    let raw = (Tensor::arange(0u32, u32::try_from(TOKENS).unwrap() * head_dim, &device)?
        .to_dtype(DType::F32)?
        .reshape((TOKENS, QSA_HEAD_DIM))?
        / f64::from(head_dim))?
    .sin()?
    .to_dtype(dtype)?;
    let angles = (Tensor::arange(0u32, u32::try_from(TOKENS).unwrap() * half_rot, &device)?
        .to_dtype(DType::F32)?
        .reshape((TOKENS, HALF_ROT))?
        / f64::from(half_rot))?;
    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;
    let slots = Tensor::new(&[8i64, 9, 10, 11, 0, 1, 2, 3], &device)?;
    let tables = Tensor::new(&[[2u32, 0]], &device)?;
    let kv_lens = Tensor::new(&[TOKENS as u32], &device)?;
    let paged = PagedView {
        block_tables: &tables,
        kv_lens: &kv_lens,
        block_size: BLOCK_SIZE,
    };
    let shape = QsaShape {
        ratio: RATIO,
        topk: 1,
        half_rot: HALF_ROT,
        n_index_heads: 1,
    };
    let weight = Tensor::ones(QSA_HEAD_DIM, DType::F32, &device)?;
    let full_layout = TokenLayout::from_host(&[(0, TOKENS)], &[TOKENS], TOKENS, &device)?;
    let suffix_layout =
        TokenLayout::from_host(&[(0, REPLACED_TOKENS)], &[TOKENS], REPLACED_TOKENS, &device)?;
    let aux = Tensor::zeros((SLOTS, aux_dim), dtype, &device)?;
    let keys = Tensor::zeros((SLOTS / RATIO, QSA_HEAD_DIM), dtype, &device)?;
    let finalize = |aux: &Tensor, keys: &Tensor, layout: &TokenLayout| {
        qsa_finalize(QsaFinalizeArgs {
            aux,
            block_keys: keys,
            layout,
            paged: &paged,
            norm_weight: &weight,
            shape: &shape,
            eps: 1e-6,
        })
    };
    qsa_aux_write(&raw, &cos, &sin, &slots, &aux)?;
    let initial_aux = aux.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    finalize(&aux, &keys, &full_layout)?;
    assert_eq!(aux.to_dtype(DType::F32)?.to_vec2::<f32>()?, initial_aux);

    let suffix_start = TOKENS - REPLACED_TOKENS;
    let replacement = raw.narrow(0, suffix_start, REPLACED_TOKENS)?.neg()?;
    qsa_aux_write(
        &replacement,
        &cos.narrow(0, suffix_start, REPLACED_TOKENS)?,
        &sin.narrow(0, suffix_start, REPLACED_TOKENS)?,
        &slots.narrow(0, suffix_start, REPLACED_TOKENS)?,
        &aux,
    )?;
    finalize(&aux, &keys, &suffix_layout)?;

    let fresh_aux = Tensor::zeros((SLOTS, aux_dim), dtype, &device)?;
    let fresh_keys = Tensor::zeros((SLOTS / RATIO, QSA_HEAD_DIM), dtype, &device)?;
    let accepted = Tensor::cat(&[raw.narrow(0, 0, suffix_start)?, replacement], 0)?;
    qsa_aux_write(&accepted, &cos, &sin, &slots, &fresh_aux)?;
    finalize(&fresh_aux, &fresh_keys, &full_layout)?;
    assert_eq!(
        keys.to_dtype(DType::F32)?.to_vec2::<f32>()?,
        fresh_keys.to_dtype(DType::F32)?.to_vec2::<f32>()?
    );
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn qsa_scored_topk_graph_ignores_poisoned_unused_pages_across_requests() -> Result<()> {
    use crate::pipeline::cuda_graph::CudaGraphHandle;

    const TOPK: usize = 512;
    const PHYSICAL_BLOCKS: usize = 83;
    const CONTEXTS: &[usize] = &[2048, 2049, 2050, 2051, 2052, 2053];
    const PREFIX: usize = CONTEXTS[0] - 1;
    const REQUESTS: usize = 3;
    const RNG_MULTIPLIER: u64 = 6364136223846793005;
    const RNG_INCREMENT: u64 = 1442695040888963407;
    const BF16_TOLERANCE: f32 = 0.004;
    const F16_TOLERANCE: f32 = 0.0005;

    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
    let table_blocks = CONTEXTS.last().unwrap().div_ceil(DENSE_TEST_BLOCK_SIZE);
    let cache_len = PHYSICAL_BLOCKS * DENSE_TEST_KV_HEADS * DENSE_TEST_BLOCK_SIZE * ATTN_HEAD_DIM;
    let aux_width = QSA_HEAD_DIM + 2 * HALF_ROT;
    let physical_tokens = PHYSICAL_BLOCKS * DENSE_TEST_BLOCK_SIZE;
    let layout = TokenLayout::rectangular(1, 1);
    let prefix_layout = TokenLayout::from_host(&[(0, PREFIX)], &[PREFIX], PREFIX, &device)?;
    let shape = QsaShape {
        ratio: RATIO,
        topk: TOPK,
        half_rot: HALF_ROT,
        n_index_heads: 1,
    };
    let norm_weight = Tensor::ones(QSA_HEAD_DIM, DType::F32, &device)?;
    for dtype in [DType::BF16, DType::F16] {
        let round = |value| match dtype {
            DType::BF16 => half::bf16::from_f32(value).to_f32(),
            DType::F16 => half::f16::from_f32(value).to_f32(),
            _ => unreachable!(),
        };
        let value = |token: usize, head: usize, dim: usize| {
            let token_value = if token < RATIO {
                16.0
            } else {
                (token % 7) as f32 / 8.0
            };
            round(token_value + head as f32 / 16.0 + (dim % 13) as f32 / 64.0)
        };
        for flashinfer_layout in [false, true] {
            let key_shape = if flashinfer_layout {
                vec![
                    PHYSICAL_BLOCKS,
                    DENSE_TEST_KV_HEADS,
                    DENSE_TEST_BLOCK_SIZE,
                    ATTN_HEAD_DIM,
                ]
            } else {
                vec![
                    PHYSICAL_BLOCKS,
                    DENSE_TEST_KV_HEADS,
                    ATTN_HEAD_DIM / 8,
                    DENSE_TEST_BLOCK_SIZE,
                    8,
                ]
            };
            let value_shape = if flashinfer_layout {
                key_shape.clone()
            } else {
                vec![
                    PHYSICAL_BLOCKS,
                    DENSE_TEST_KV_HEADS,
                    ATTN_HEAD_DIM,
                    DENSE_TEST_BLOCK_SIZE,
                ]
            };
            let key_cache = Var::zeros(key_shape.clone(), dtype, &device)?;
            let value_cache = Var::zeros(value_shape.clone(), dtype, &device)?;
            let aux = Var::zeros((physical_tokens, aux_width), dtype, &device)?;
            let block_keys = Var::zeros((physical_tokens / RATIO, QSA_HEAD_DIM), dtype, &device)?;
            let tables = Var::zeros((1, table_blocks), DType::U32, &device)?;
            let kv_lens = Var::zeros(1, DType::U32, &device)?;
            let slot_mapping = Var::zeros(1, DType::I64, &device)?;
            let raw_key = Var::zeros((1, QSA_HEAD_DIM), dtype, &device)?;
            let cos = Tensor::ones((1, HALF_ROT), dtype, &device)?;
            let sin = Tensor::zeros((1, HALF_ROT), dtype, &device)?;
            let q = Tensor::zeros((1, DENSE_TEST_Q_HEADS, ATTN_HEAD_DIM), dtype, &device)?;
            let mut score_query = vec![0.0f32; QSA_HEAD_DIM];
            score_query[0] = 1.0;
            let score_query =
                Tensor::from_vec(score_query, (1, 1, QSA_HEAD_DIM), &device)?.to_dtype(dtype)?;
            let paged = PagedView {
                block_tables: tables.as_tensor(),
                kv_lens: kv_lens.as_tensor(),
                block_size: DENSE_TEST_BLOCK_SIZE,
            };
            let physical_slot = |table: &[u32], token: usize| {
                table[token / DENSE_TEST_BLOCK_SIZE] as usize * DENSE_TEST_BLOCK_SIZE
                    + token % DENSE_TEST_BLOCK_SIZE
            };
            let write_cache =
                |keys: &mut [f32], values: &mut [f32], table: &[u32], token: usize| {
                    let slot = physical_slot(table, token);
                    let block = slot / DENSE_TEST_BLOCK_SIZE;
                    let offset = slot % DENSE_TEST_BLOCK_SIZE;
                    for head in 0..DENSE_TEST_KV_HEADS {
                        let head_base = block * DENSE_TEST_KV_HEADS + head;
                        for dim in 0..ATTN_HEAD_DIM {
                            let key_index = if flashinfer_layout {
                                (head_base * DENSE_TEST_BLOCK_SIZE + offset) * ATTN_HEAD_DIM + dim
                            } else {
                                ((head_base * (ATTN_HEAD_DIM / 8) + dim / 8)
                                    * DENSE_TEST_BLOCK_SIZE
                                    + offset)
                                    * 8
                                    + dim % 8
                            };
                            let value_index = if flashinfer_layout {
                                key_index
                            } else {
                                (head_base * ATTN_HEAD_DIM + dim) * DENSE_TEST_BLOCK_SIZE + offset
                            };
                            keys[key_index] = 0.0;
                            values[value_index] = value(token, head, dim);
                        }
                    }
                };
            let upload_cache = |keys: &[f32], values: &[f32]| -> Result<()> {
                key_cache.set(
                    &Tensor::from_vec(keys.to_vec(), key_shape.clone(), &device)?
                        .to_dtype(dtype)?,
                )?;
                value_cache.set(
                    &Tensor::from_vec(values.to_vec(), value_shape.clone(), &device)?
                        .to_dtype(dtype)?,
                )
            };
            let prepare_request = |request: usize| -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
                let mut permutation = (0..PHYSICAL_BLOCKS as u32).collect::<Vec<_>>();
                let mut random = request as u64 + 1;
                for index in (1..permutation.len()).rev() {
                    random = random
                        .wrapping_mul(RNG_MULTIPLIER)
                        .wrapping_add(RNG_INCREMENT);
                    permutation.swap(index, (random as usize) % (index + 1));
                }
                let table = permutation[..table_blocks].to_vec();
                tables.set(&Tensor::from_vec(
                    table.clone(),
                    (1, table_blocks),
                    &device,
                )?)?;
                let mut keys = vec![f32::NAN; cache_len];
                let mut values = vec![f32::NAN; cache_len];
                let mut raw = vec![f32::NAN; physical_tokens * aux_width];
                for token in 0..PREFIX {
                    write_cache(&mut keys, &mut values, &table, token);
                    let base = physical_slot(&table, token) * aux_width;
                    raw[base..base + QSA_HEAD_DIM].fill(1.0);
                    raw[base] = (token / RATIO + 1) as f32;
                    raw[base + QSA_HEAD_DIM..base + QSA_HEAD_DIM + HALF_ROT].fill(1.0);
                    raw[base + QSA_HEAD_DIM + HALF_ROT..base + aux_width].fill(0.0);
                }
                upload_cache(&keys, &values)?;
                aux.set(
                    &Tensor::from_vec(raw, (physical_tokens, aux_width), &device)?
                        .to_dtype(dtype)?,
                )?;
                block_keys.set(
                    &Tensor::full(f32::NAN, (physical_tokens / RATIO, QSA_HEAD_DIM), &device)?
                        .to_dtype(dtype)?,
                )?;
                kv_lens.set(&Tensor::new(&[PREFIX as u32], &device)?)?;
                qsa_finalize(QsaFinalizeArgs {
                    aux: aux.as_tensor(),
                    block_keys: block_keys.as_tensor(),
                    layout: &prefix_layout,
                    paged: &paged,
                    norm_weight: &norm_weight,
                    shape: &shape,
                    eps: 1e-6,
                })?;
                Ok((table, keys, values))
            };
            let prepare_token = |context: usize,
                                 table: &[u32],
                                 keys: &mut [f32],
                                 values: &mut [f32]|
             -> Result<()> {
                let token = context - 1;
                write_cache(keys, values, table, token);
                upload_cache(keys, values)?;
                kv_lens.set(&Tensor::new(&[context as u32], &device)?)?;
                slot_mapping.set(&Tensor::new(
                    &[physical_slot(table, token) as i64],
                    &device,
                )?)?;
                let mut raw = vec![1.0f32; QSA_HEAD_DIM];
                raw[0] = (token / RATIO + 1) as f32;
                raw_key.set(&Tensor::from_vec(raw, (1, QSA_HEAD_DIM), &device)?.to_dtype(dtype)?)
            };
            let forward = || -> Result<(Tensor, Tensor, Tensor)> {
                qsa_aux_write(
                    raw_key.as_tensor(),
                    &cos,
                    &sin,
                    slot_mapping.as_tensor(),
                    aux.as_tensor(),
                )?;
                qsa_finalize(QsaFinalizeArgs {
                    aux: aux.as_tensor(),
                    block_keys: block_keys.as_tensor(),
                    layout: &layout,
                    paged: &paged,
                    norm_weight: &norm_weight,
                    shape: &shape,
                    eps: 1e-6,
                })?;
                let (selected, counts) = qsa_select(
                    &score_query,
                    block_keys.as_tensor(),
                    &layout,
                    &paged,
                    &shape,
                    table_blocks * DENSE_TEST_BLOCK_SIZE / RATIO,
                )?;
                let output = qsa_attention(QsaAttentionArgs {
                    q: &q,
                    key_cache: key_cache.as_tensor(),
                    value_cache: value_cache.as_tensor(),
                    layout: &layout,
                    paged: &paged,
                    selection: Some((&selected, &counts)),
                    shape: &shape,
                    n_kv_heads: DENSE_TEST_KV_HEADS,
                    scale: 1.0 / (ATTN_HEAD_DIM as f32).sqrt(),
                })?;
                Ok((selected, counts, output))
            };
            let (table, mut keys, mut values) = prepare_request(0)?;
            prepare_token(CONTEXTS[0], &table, &mut keys, &mut values)?;
            let _ = forward()?;
            device.synchronize()?;
            let tracking = stream.context().is_event_tracking();
            if tracking {
                unsafe { stream.context().disable_event_tracking() };
            }
            if let Err(error) =
                stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            {
                if tracking {
                    unsafe { stream.context().enable_event_tracking() };
                }
                return Err(candle_core::Error::msg(error));
            }
            let captured = forward();
            let graph = CudaGraphHandle::end_capture(&stream);
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            let (selected, counts, output) = captured?;
            let graph = graph?.ok_or_else(|| {
                candle_core::Error::msg("QSA transition capture returned no graph")
            })?;
            for request in 0..REQUESTS {
                let (table, mut keys, mut values) = prepare_request(request)?;
                for &context in CONTEXTS {
                    prepare_token(context, &table, &mut keys, &mut values)?;
                    graph.launch()?;
                    let count = counts.to_vec1::<i32>()?[0] as usize;
                    let complete_blocks = context / RATIO;
                    assert_eq!(count, complete_blocks.min(TOPK));
                    let mut actual_blocks =
                        selected.flatten_all()?.to_vec1::<i32>()?[..count].to_vec();
                    actual_blocks.sort_unstable();
                    let first_block = complete_blocks.saturating_sub(TOPK);
                    assert_eq!(actual_blocks, (first_block..complete_blocks).map(|block| block as i32).collect::<Vec<_>>(),
                            "{dtype:?} flashinfer={flashinfer_layout} request={request} context={context}");
                    let first_token = first_block * RATIO;
                    let actual = output
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let tolerance = if dtype == DType::BF16 {
                        BF16_TOLERANCE
                    } else {
                        F16_TOLERANCE
                    };
                    for (index, actual) in actual.into_iter().enumerate() {
                        let query_head = index / ATTN_HEAD_DIM;
                        let kv_head = query_head / (DENSE_TEST_Q_HEADS / DENSE_TEST_KV_HEADS);
                        let dim = index % ATTN_HEAD_DIM;
                        let expected = round(
                            (first_token..context)
                                .map(|token| value(token, kv_head, dim) as f64)
                                .sum::<f64>() as f32
                                / (context - first_token) as f32,
                        );
                        assert!(actual.is_finite() && (actual - expected).abs() <= tolerance,
                                "{dtype:?} flashinfer={flashinfer_layout} request={request} context={context} element={index}: {actual} != {expected}");
                    }
                }
            }
        }
    }
    Ok(())
}
