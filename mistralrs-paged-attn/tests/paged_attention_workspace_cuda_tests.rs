#![cfg(all(feature = "cuda", target_family = "unix"))]

use candle_core::cuda::cudarc::driver::sys;
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_paged_attn::paged_attention_with_window;

const QUERY_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const BLOCK_SIZE: usize = 32;
const CACHE_VECTOR: usize = 4;
const PARTITION_SIZE: usize = 512;
const SHORT_CONTEXT: usize = 513;
const WINDOW_CONTEXT: usize = 4097;
const LARGE_CONTEXT: usize = 8193;
const PRESSURE_BUFFERS: usize = 64;
const PRESSURE_SENTINEL: f32 = 1234.0;
const TOLERANCE: f32 = 0.00002;

fn value(position: usize, head: usize, dim: usize) -> f32 {
    (position % 17) as f32 * 0.125 + head as f32 * 0.5 + dim as f32 * 0.0078125
}

#[test]
fn captured_v2_workspace_survives_growth_and_allocator_reuse() -> Result<()> {
    for window in [None, Some(SHORT_CONTEXT)] {
        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let blocks = LARGE_CONTEXT.div_ceil(BLOCK_SIZE);
        let query = Tensor::zeros((1, QUERY_HEADS, HEAD_DIM), DType::F32, &device)?;
        let keys = Tensor::zeros(
            (
                blocks,
                KV_HEADS,
                HEAD_DIM / CACHE_VECTOR,
                BLOCK_SIZE,
                CACHE_VECTOR,
            ),
            DType::F32,
            &device,
        )?;
        let values = (0..blocks * KV_HEADS * HEAD_DIM * BLOCK_SIZE)
            .map(|index| {
                let token = index % BLOCK_SIZE;
                let dim = index / BLOCK_SIZE % HEAD_DIM;
                let head = index / (BLOCK_SIZE * HEAD_DIM) % KV_HEADS;
                let block = index / (BLOCK_SIZE * HEAD_DIM * KV_HEADS);
                value(block * BLOCK_SIZE + token, head, dim)
            })
            .collect::<Vec<_>>();
        let values = Tensor::from_vec(values, (blocks, KV_HEADS, HEAD_DIM, BLOCK_SIZE), &device)?;
        let tables =
            Tensor::from_vec((0..blocks as u32).collect::<Vec<_>>(), (1, blocks), &device)?;
        let context = if window.is_some() {
            WINDOW_CONTEXT
        } else {
            SHORT_CONTEXT
        };
        let contexts = Tensor::new(&[context as u32], &device)?;
        let large_contexts = Tensor::new(&[LARGE_CONTEXT as u32], &device)?;
        let sinks = Tensor::full(std::f32::consts::LN_2, QUERY_HEADS, &device)?;
        let decode = |contexts: &Tensor, max_context, window| {
            paged_attention_with_window(
                &query,
                None,
                None,
                &keys,
                &values,
                &tables,
                contexts,
                None,
                max_context,
                1.0,
                1.0,
                Some(&sinks),
                window,
            )
        };
        decode(&contexts, context, window)?;
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
            return Err(candle_core::Error::msg(error.to_string()));
        }
        let captured = decode(&contexts, context, window);
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let output = captured?;
        let graph = graph
            .map_err(|error| candle_core::Error::msg(error.to_string()))?
            .ok_or_else(|| candle_core::Error::msg("PagedAttention capture produced no graph"))?;

        decode(&large_contexts, LARGE_CONTEXT, None)?;
        device.synchronize()?;

        let short_workspace_floats =
            QUERY_HEADS * SHORT_CONTEXT.div_ceil(PARTITION_SIZE) * (HEAD_DIM + 2);
        let pressure = (0..PRESSURE_BUFFERS)
            .map(|index| {
                let elements = if index % 2 == 0 {
                    short_workspace_floats
                } else {
                    short_workspace_floats.next_power_of_two()
                };
                Tensor::full(PRESSURE_SENTINEL, elements, &device)
            })
            .collect::<Result<Vec<_>>>()?;
        graph
            .launch()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        device.synchronize()?;

        let actual = output.flatten_all()?.to_vec1::<f32>()?;
        let start = window.map_or(0, |window| context.saturating_sub(window));
        for head in 0..QUERY_HEADS {
            for dim in 0..HEAD_DIM {
                let numerator = (start..context)
                    .map(|position| {
                        f64::from(value(position, head / (QUERY_HEADS / KV_HEADS), dim))
                    })
                    .sum::<f64>();
                let expected = (numerator / (context - start + 2) as f64) as f32;
                let actual = actual[head * HEAD_DIM + dim];
                assert!(
                    (actual - expected).abs() < TOLERANCE,
                    "window {window:?}, head {head}, dim {dim}: {actual} vs {expected}"
                );
            }
        }
        for buffer in pressure {
            assert!(
                buffer
                    .to_vec1::<f32>()?
                    .into_iter()
                    .all(|value| value == PRESSURE_SENTINEL),
                "window {window:?}: captured workspace overwrote a live allocation"
            );
        }
    }
    Ok(())
}
