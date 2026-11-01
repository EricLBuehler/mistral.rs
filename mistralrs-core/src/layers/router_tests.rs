use candle_core::{cuda::cudarc::driver::sys, DType, Device, Var};

use super::*;
use crate::ops::{
    moe_router_topk, MoeRouterScoreFunction, MoeRouterSelectedWeight, MoeRouterTopKConfig,
};

const TOP_K: usize = 10;
const REPLAY_VALUES: [f32; 4] = [0.0, 1.0, -0.5, 0.25];

fn topk(logits: &Tensor) -> Result<Vec<u32>> {
    moe_router_topk(
        logits,
        MoeRouterTopKConfig {
            top_k: TOP_K,
            score_function: MoeRouterScoreFunction::Softmax,
            selected_weight: MoeRouterSelectedWeight::Score,
            renormalize: true,
            norm_min: 0.0,
            output_scale: 1.0,
            logit_clip: None,
        },
        None,
        None,
    )?
    .indices
    .flatten_all()?
    .to_vec1()
}

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
}

fn fixture(rows: usize, columns: usize, dtype: DType, device: &Device) -> Result<Linear> {
    let mut weights = vec![0f32; rows * columns];
    for row in 0..rows {
        weights[row * columns] = 0.125 + (row / 3) as f32 / 256.0;
        weights[row * columns + columns - 1] = -0.0625;
    }
    Ok(Linear::new(
        Tensor::from_vec(weights, (rows, columns), device)?.to_dtype(dtype)?,
        None,
    ))
}

#[test]
#[ignore = "requires CUDA"]
fn router_linear_matches_independent_logits_and_topk() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        for (rows, columns) in [(32, 512), (128, 2048), (512, 2560), (576, 4096)] {
            let linear = fixture(rows, columns, dtype, &device)?;
            let weights = values(linear.weight())?;
            for batch in [1, 2, 17] {
                let input = Tensor::ones((batch, columns), dtype, &device)?;
                let expected = Tensor::from_vec(
                    (0..batch * rows)
                        .map(|i| {
                            weights[(i % rows) * columns] + weights[(i % rows + 1) * columns - 1]
                        })
                        .collect::<Vec<_>>(),
                    (batch, rows),
                    &Device::Cpu,
                )?
                .to_dtype(dtype)?
                .to_device(&device)?;
                let actual = router_linear(&linear, &input)?;
                assert_eq!(values(&actual)?, values(&expected)?);
                assert_eq!(topk(&actual)?, topk(&expected)?);
                if batch != 1 || dtype == DType::F32 {
                    assert_eq!(values(&actual)?, values(&linear.forward(&input)?)?);
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn router_linear_preserves_bias_rounding_and_view_fallbacks() -> Result<()> {
    const ROWS: usize = 64;
    const COLUMNS: usize = 64;
    let device = Device::new_cuda_with_stream(0)?;
    for dtype in [DType::BF16, DType::F16] {
        let delta = if dtype == DType::BF16 {
            1.0 / 512.0
        } else {
            1.0 / 4096.0
        };
        let mut weights = vec![0f32; ROWS * COLUMNS];
        for row in 0..ROWS {
            weights[row * COLUMNS] = 1.0;
            weights[row * COLUMNS + 1] = delta;
        }
        let weight = Tensor::from_vec(weights, (ROWS, COLUMNS), &device)?.to_dtype(dtype)?;
        let bias = Tensor::full(-1f32, ROWS, &device)?.to_dtype(dtype)?;
        let input = Tensor::ones((1, COLUMNS), dtype, &device)?;
        let linear = Linear::new(weight.clone(), Some(bias));
        assert_eq!(values(&router_linear(&linear, &input)?)?, vec![0f32; ROWS]);
        let padding = Tensor::zeros(1, dtype, &device)?;
        let offset_weight = Tensor::cat(&[&padding, &weight.flatten_all()?], 0)?
            .narrow(0, 1, ROWS * COLUMNS)?
            .reshape((ROWS, COLUMNS))?;
        let offset_input = Tensor::cat(&[&padding, &input.flatten_all()?], 0)?
            .narrow(0, 1, COLUMNS)?
            .reshape((1, COLUMNS))?;
        for (weight, input) in [
            (offset_weight, input.clone()),
            (weight.t()?, input.clone()),
            (weight, offset_input),
        ] {
            let linear = Linear::new(weight, None);
            assert_eq!(
                values(&router_linear(&linear, &input)?)?,
                values(&linear.forward(&input)?)?
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn router_gemv_orders_cross_stream_weights_and_graph_replay() -> Result<()> {
    const ROWS: usize = 128;
    const COLUMNS: usize = 2048;
    let device = Device::new_cuda_with_stream(0)?;
    let producer = Device::new_cuda_with_stream(0)?;
    for dtype in [DType::BF16, DType::F16] {
        let original = fixture(ROWS, COLUMNS, dtype, &producer)?;
        let weight = Var::from_tensor(original.weight())?;
        let linear = Linear::new(weight.as_tensor().clone(), None);
        let input = Tensor::ones((1, COLUMNS), dtype, &device)?;
        for value in REPLAY_VALUES {
            let updated = (original.weight() * value as f64)?;
            weight.set(&updated)?;
            // Separate Candle device IDs bypass the router capability guard.
            let actual = mistralrs_quant::gemv::gemv(&input, linear.weight(), None)?;
            let expected = Linear::new(updated.to_device(&device)?, None).forward(&input)?;
            assert_eq!(values(&actual)?, values(&expected)?);
        }
        let linear = fixture(ROWS, COLUMNS, dtype, &device)?;
        let input = Var::zeros((1, COLUMNS), dtype, &device)?;
        let _ = router_linear(&linear, &input)?;
        device.synchronize()?;
        let cuda = device.as_cuda_device()?;
        let _htod = cuda.enable_cuda_graph_htod_cache();
        let stream = cuda.cuda_stream();
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
            return Err(candle_core::Error::wrap(error));
        }
        let actual = router_linear(&linear, &input);
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let actual = actual?;
        let graph = graph.map_err(candle_core::Error::wrap)?.unwrap();
        for value in REPLAY_VALUES {
            input.set(&Tensor::full(value, (1, COLUMNS), &device)?.to_dtype(dtype)?)?;
            graph.launch().map_err(candle_core::Error::wrap)?;
            let expected = linear.forward(&input)?;
            assert_eq!(values(&actual)?, values(&expected)?);
            assert_eq!(topk(&actual)?, topk(&expected)?);
        }
    }
    Ok(())
}
