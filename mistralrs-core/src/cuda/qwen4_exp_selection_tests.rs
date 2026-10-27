use super::*;
use crate::pipeline::cuda_graph::CudaGraphHandle;
use candle_core::{cuda_backend::cudarc::driver::sys, Var};

const SELECTION_GROUPS: &[usize] = &[0, 511, 512, 513, 527, 544, 2048, 4097];
const MAX_GROUPS: usize = 4097;
const PAGE_SIZE: usize = 32;
const TOPK: usize = 512;
const RNG_MULTIPLIER: u64 = 6364136223846793005;
const RNG_INCREMENT: u64 = 1442695040888963407;

fn capture<T>(
    device: &Device,
    forward: impl FnOnce() -> Result<T>,
) -> Result<(CudaGraphHandle, T)> {
    device.synchronize()?;
    let stream = device.as_cuda_device()?.cuda_stream();
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
    let output = forward();
    let graph = CudaGraphHandle::end_capture(&stream);
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let output = output?;
    let graph = graph?.ok_or_else(|| candle_core::Error::msg("QSA selection capture was empty"))?;
    Ok((graph, output))
}

fn ordered(value: f32) -> u32 {
    if value.is_nan() {
        0
    } else if value.is_sign_negative() {
        !value.to_bits()
    } else {
        value.to_bits() ^ (1 << 31)
    }
}

fn assert_selection(scores: &[f32], selected: &[i32], count: i32, topk: usize) {
    assert_eq!(count as usize, scores.len().min(topk));
    if scores.len() <= topk {
        assert_eq!(
            &selected[..scores.len()],
            &(0..scores.len() as i32).collect::<Vec<_>>()
        );
        return;
    }
    let keys = scores.iter().copied().map(ordered).collect::<Vec<_>>();
    let mut ranked = keys.clone();
    ranked.sort_unstable_by(|a, b| b.cmp(a));
    let threshold = ranked[topk - 1];
    let above = keys.iter().filter(|&&key| key > threshold).count();
    let mut seen = vec![false; scores.len()];
    for (rank, &index) in selected[..topk].iter().enumerate() {
        let index = index as usize;
        assert!(index < scores.len() && !seen[index]);
        seen[index] = true;
        if rank < above {
            assert!(keys[index] > threshold);
        } else {
            assert_eq!(keys[index], threshold);
        }
    }
    for (index, &key) in keys.iter().enumerate() {
        assert!(key <= threshold || seen[index]);
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn qsa_topk_preserves_float_order_and_boundary_ties_on_graph_replay() -> Result<()> {
    const ROWS: usize = 2;
    const RATIO: usize = 4;
    let device = Device::new_cuda_with_stream(0)?;
    let _htod_cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
    let scores = Var::zeros((ROWS, MAX_GROUPS), DType::F32, &device)?;
    let lengths = Var::zeros(ROWS, DType::U32, &device)?;
    let tables = Tensor::zeros((ROWS, 1), DType::U32, &device)?;
    let layout = TokenLayout::rectangular(ROWS, 1);
    let paged = PagedView {
        block_tables: &tables,
        kv_lens: lengths.as_tensor(),
        block_size: PAGE_SIZE,
    };
    for topk in [1, 31, TOPK] {
        let selected = Tensor::zeros((ROWS, topk), DType::I32, &device)?;
        let counts = Tensor::zeros(ROWS, DType::I32, &device)?;
        let forward = || -> Result<()> {
            unsafe {
                ffi::qwen4_qsa_topk(
                    dev_ptr(scores.as_tensor())? as *const f32,
                    MAX_GROUPS as i32,
                    layout.ffi()?,
                    paged.ffi()?,
                    RATIO as i32,
                    topk as i32,
                    dev_ptr(&selected)? as *mut i32,
                    dev_ptr(&counts)? as *mut i32,
                    stream(&device)?,
                );
            }
            Ok(())
        };
        forward()?;
        let (graph, ()) = capture(&device, forward)?;
        for &groups in SELECTION_GROUPS {
            for pattern in 0..5 {
                let live = [groups, MAX_GROUPS - groups];
                let mut host = vec![f32::NAN; ROWS * MAX_GROUPS];
                let mut random = groups as u64 + pattern;
                for (row, &count) in live.iter().enumerate() {
                    for index in 0..count {
                        random = random
                            .wrapping_mul(RNG_MULTIPLIER)
                            .wrapping_add(RNG_INCREMENT);
                        host[row * MAX_GROUPS + index] = match pattern {
                            0 => index as f32 / 8.0 - 300.0,
                            1 => (random % 17) as f32 - 8.0,
                            2 => 0.0,
                            3 => match index % 6 {
                                0 => f32::NAN,
                                1 => f32::INFINITY,
                                2 => f32::NEG_INFINITY,
                                3 => -0.0,
                                _ => (random % 100) as f32 / 64.0,
                            },
                            _ => f32::from_bits((random >> 32) as u32),
                        };
                    }
                }
                scores.set(&Tensor::from_vec(
                    host.clone(),
                    (ROWS, MAX_GROUPS),
                    &device,
                )?)?;
                lengths.set(&Tensor::new(
                    live.map(|count| (count * RATIO) as u32).as_slice(),
                    &device,
                )?)?;
                graph.launch()?;
                let actual = selected.to_vec2::<i32>()?;
                let counts = counts.to_vec1::<i32>()?;
                for row in 0..ROWS {
                    assert_selection(
                        &host[row * MAX_GROUPS..row * MAX_GROUPS + live[row]],
                        &actual[row],
                        counts[row],
                        topk,
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn qsa_scoring_preserves_selection_with_changed_heads_pages_and_dtype() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let _htod_cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
    for dtype in [DType::BF16, DType::F16] {
        for (heads, ratio, key_stride) in [
            (1, 2, 128),
            (3, 4, 136),
            (4, 4, 128),
            (4, 4, 129),
            (8, 8, 256),
        ] {
            let pages = (MAX_GROUPS * ratio).div_ceil(PAGE_SIZE) + 1;
            let physical_groups = pages * PAGE_SIZE / ratio;
            let keys = Var::zeros((physical_groups, key_stride), dtype, &device)?;
            let tables = Var::zeros((1, pages), DType::U32, &device)?;
            let lengths = Var::zeros(1, DType::U32, &device)?;
            let query = Var::zeros((1, heads, QSA_HEAD_DIM), dtype, &device)?;
            let layout = TokenLayout::rectangular(1, 1);
            let paged = PagedView {
                block_tables: tables.as_tensor(),
                kv_lens: lengths.as_tensor(),
                block_size: PAGE_SIZE,
            };
            let shape = QsaShape {
                ratio,
                topk: TOPK,
                half_rot: 0,
                n_index_heads: heads,
            };
            let forward = || {
                qsa_select(
                    query.as_tensor(),
                    keys.as_tensor(),
                    &layout,
                    &paged,
                    &shape,
                    physical_groups,
                )
            };
            let _ = forward()?;
            let (graph, (selected, counts)) = capture(&device, forward)?;
            for request in 0..2 {
                let mut table = (0..pages as u32).collect::<Vec<_>>();
                let mut random = request as u64 + 1;
                for index in (1..pages).rev() {
                    random = random
                        .wrapping_mul(RNG_MULTIPLIER)
                        .wrapping_add(RNG_INCREMENT);
                    table.swap(index, random as usize % (index + 1));
                }
                tables.set(&Tensor::from_vec(table.clone(), (1, pages), &device)?)?;
                let direction = if request == 0 { 1.0 } else { -1.0 };
                let mut q = vec![0.0f32; heads * QSA_HEAD_DIM];
                for head in 0..heads {
                    q[head * QSA_HEAD_DIM] = direction;
                }
                query.set(
                    &Tensor::from_vec(q, (1, heads, QSA_HEAD_DIM), &device)?.to_dtype(dtype)?,
                )?;
                for &groups in SELECTION_GROUPS {
                    let mut host = vec![f32::NAN; physical_groups * key_stride];
                    let mut expected = Vec::with_capacity(groups);
                    for index in 0..groups {
                        let token = index * ratio;
                        let physical = (table[token / PAGE_SIZE] as usize * PAGE_SIZE
                            + token % PAGE_SIZE)
                            / ratio;
                        let row =
                            &mut host[physical * key_stride..physical * key_stride + QSA_HEAD_DIM];
                        row.fill(0.0);
                        row[0] = (index % 129) as f32 - 64.0;
                        expected.push((direction * row[0]).max(0.0));
                    }
                    keys.set(
                        &Tensor::from_vec(host, (physical_groups, key_stride), &device)?
                            .to_dtype(dtype)?,
                    )?;
                    lengths.set(&Tensor::new(&[(groups * ratio) as u32], &device)?)?;
                    graph.launch()?;
                    assert_selection(
                        &expected,
                        &selected.to_vec2::<i32>()?[0],
                        counts.to_vec1::<i32>()?[0],
                        TOPK,
                    );
                }
            }
        }
    }
    Ok(())
}
