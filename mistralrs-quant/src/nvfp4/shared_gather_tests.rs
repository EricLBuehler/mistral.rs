use std::sync::Mutex;

use candle_core::{cuda::cudarc::driver::sys, DType, Device, Result, Tensor};
use float8::F8E4M3;

use super::{Nvfp4Layer, Nvfp4LayerParts};
use crate::{
    try_gather_forward_with_shared_quantized_activation, Nvfp4ActivationMode, QuantMethod,
    NVFP4_BLOCK_SIZE,
};

const EXPERTS: usize = 9;
const TOPK: usize = 4;
const COLUMNS: usize = 512;
const REDUCTION: usize = 640;
const SHORT_ROWS: usize = 513;
const SOURCE_QUANTIZED_ROWS: usize = 4097;
const VALUES_PER_BYTE: usize = 2;
const CALIBRATION: f32 = 1.25;
const OTHER_CALIBRATION: f32 = 2.5;
const WEIGHT_CODES: [u8; 8] = [0x22, 0x32, 0x42, 0x23, 0x34, 0x54, 0x12, 0x35];
const BLOCK_SCALES: [f32; 3] = [0.25, 0.5, 1.0];
const GLOBAL_PERIOD: usize = 7;
const GLOBAL_DIVISOR: f32 = 512.0;
const INPUT_PERIOD: usize = 23;
const INPUT_DIVISOR: f32 = 8.0;
const INPUT_ROW_MULTIPLIER: usize = 11;
const INPUT_COLUMN_MULTIPLIER: usize = 3;
const ROUTING_TOKEN_MULTIPLIER: usize = 3;
const ROUTING_SLOT_MULTIPLIER: usize = 2;
const PROJECTION_SEEDS: [usize; 2] = [0, 5];
const REPLAY_PHASES: [usize; 3] = [0, 1, 7];

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn projection(
    device: &Device,
    dtype: DType,
    seed: usize,
    calibration: &[f32],
    immutable: bool,
) -> Result<Nvfp4Layer> {
    Nvfp4Layer::from_normalized_parts(
        Nvfp4LayerParts {
            weight: Tensor::from_vec(
                (0..EXPERTS * COLUMNS * REDUCTION / VALUES_PER_BYTE)
                    .map(|index| WEIGHT_CODES[(index + seed) % WEIGHT_CODES.len()])
                    .collect::<Vec<_>>(),
                (EXPERTS, COLUMNS, REDUCTION / VALUES_PER_BYTE),
                device,
            )?,
            scales: Tensor::from_vec(
                (0..EXPERTS * COLUMNS * REDUCTION / NVFP4_BLOCK_SIZE)
                    .map(|index| {
                        F8E4M3::from_f32(BLOCK_SCALES[(index + seed) % BLOCK_SCALES.len()])
                    })
                    .collect::<Vec<_>>(),
                (EXPERTS, COLUMNS, REDUCTION / NVFP4_BLOCK_SIZE),
                device,
            )?,
            global_scales: Tensor::from_vec(
                (0..EXPERTS * COLUMNS)
                    .map(|index| ((index + seed) % GLOBAL_PERIOD + 1) as f32 / GLOBAL_DIVISOR)
                    .collect::<Vec<_>>(),
                (EXPERTS, COLUMNS),
                device,
            )?,
            input_scale: Some(Tensor::from_slice(calibration, EXPERTS, device)?),
            activation: Nvfp4ActivationMode::DynamicBlock,
            bias: None,
            dtype,
        },
        immutable,
    )
}

fn projections(device: &Device, dtype: DType) -> Result<[Nvfp4Layer; 2]> {
    Ok([
        projection(
            device,
            dtype,
            PROJECTION_SEEDS[0],
            &[CALIBRATION; EXPERTS],
            true,
        )?,
        projection(
            device,
            dtype,
            PROJECTION_SEEDS[1],
            &[CALIBRATION; EXPERTS],
            true,
        )?,
    ])
}

fn input(
    rows: usize,
    routes: usize,
    phase: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let values = (0..rows * routes * REDUCTION)
        .map(|index| {
            let row = index / REDUCTION;
            let column = index % REDUCTION;
            let value = ((row * INPUT_ROW_MULTIPLIER + column * INPUT_COLUMN_MULTIPLIER + phase)
                % INPUT_PERIOD
                + 1) as f32
                / INPUT_DIVISOR;
            if phase.is_multiple_of(2) {
                value
            } else {
                -value
            }
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (rows, routes, REDUCTION), device)?.to_dtype(dtype)
}

fn indices(rows: usize, phase: usize, device: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        (0..rows * TOPK)
            .map(|route| {
                ((route / TOPK * ROUTING_TOKEN_MULTIPLIER
                    + route % TOPK * ROUTING_SLOT_MULTIPLIER
                    + phase)
                    % EXPERTS) as u32
            })
            .collect::<Vec<_>>(),
        (rows, TOPK),
        device,
    )
}

fn difference(left: &Tensor, right: &Tensor) -> Result<f32> {
    (left.to_dtype(DType::F32)? - right.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()
}

fn compare(actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(difference(actual, expected)?, 0.0);
    Ok(())
}

fn shared(input: &Tensor, indices: &Tensor, layers: &[Nvfp4Layer; 2]) -> Result<Vec<Tensor>> {
    try_gather_forward_with_shared_quantized_activation(input, indices, &[&layers[0], &layers[1]])?
        .ok_or_else(|| candle_core::Error::msg("compatible expert projections declined sharing"))
}

#[test]
fn shared_expert_gather_matches_distinct_projections_and_routed_sources() -> Result<()> {
    let _lock = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    for dtype in [DType::BF16, DType::F16] {
        let layers = projections(&device, dtype)?;
        for (rows, routes) in [
            (SHORT_ROWS, 1),
            (SHORT_ROWS, TOPK),
            (SOURCE_QUANTIZED_ROWS, 1),
        ] {
            let input = input(rows, routes, 0, dtype, &device)?;
            let input = if rows == SHORT_ROWS && routes == 1 {
                input.squeeze(1)?
            } else {
                input
            };
            let indices = indices(rows, 0, &device)?;
            let outputs = shared(&input, &indices, &layers)?;
            assert_eq!(outputs.len(), layers.len());
            for (output, layer) in outputs.iter().zip(&layers) {
                compare(output, &layer.gather_forward(&input, &indices)?)?;
            }
            assert!(difference(&outputs[0], &outputs[1])? > 0.0);
        }
    }
    Ok(())
}

#[test]
fn shared_expert_gather_rejects_unproven_or_mismatched_calibration() -> Result<()> {
    let _lock = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let dtype = DType::BF16;
    let first = projection(
        &device,
        dtype,
        PROJECTION_SEEDS[0],
        &[CALIBRATION; EXPERTS],
        true,
    )?;
    let input = input(SHORT_ROWS, 1, 0, dtype, &device)?;
    let indices = indices(SHORT_ROWS, 0, &device)?;
    let mut nonuniform = [CALIBRATION; EXPERTS];
    nonuniform[EXPERTS - 1] = OTHER_CALIBRATION;
    for (calibration, immutable) in [
        ([OTHER_CALIBRATION; EXPERTS], true),
        ([CALIBRATION; EXPERTS], false),
        (nonuniform, true),
    ] {
        let other = projection(&device, dtype, PROJECTION_SEEDS[1], &calibration, immutable)?;
        for methods in [
            [&first as &dyn QuantMethod, &other as &dyn QuantMethod],
            [&other as &dyn QuantMethod, &first as &dyn QuantMethod],
        ] {
            assert!(try_gather_forward_with_shared_quantized_activation(
                &input, &indices, &methods,
            )?
            .is_none());
        }
        assert_eq!(
            other.gather_forward(&input, &indices)?.dims(),
            &[SHORT_ROWS, TOPK, COLUMNS]
        );
    }
    Ok(())
}

#[test]
fn shared_expert_gather_graph_replays_current_inputs_and_routes() -> Result<()> {
    let _lock = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda_with_stream(0)?;
    let dev = device.as_cuda_device()?;
    if !crate::cutile::nvfp4_supported(dev) {
        return Ok(());
    }
    let stream = dev.cuda_stream();
    let _cache = dev.enable_cuda_graph_htod_cache();
    for dtype in [DType::BF16, DType::F16] {
        let layers = projections(&device, dtype)?;
        let source = input(SOURCE_QUANTIZED_ROWS, 1, REPLAY_PHASES[0], dtype, &device)?;
        let routing = indices(SOURCE_QUANTIZED_ROWS, REPLAY_PHASES[0], &device)?;
        let warm = shared(&source, &routing, &layers)?;
        for (output, layer) in warm.iter().zip(&layers) {
            compare(output, &layer.gather_forward(&source, &routing)?)?;
        }
        drop(warm);
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
            return Err(candle_core::Error::wrap(error));
        }
        let outputs = shared(&source, &routing, &layers);
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let outputs = outputs?;
        let graph = graph
            .map_err(candle_core::Error::wrap)?
            .ok_or_else(|| candle_core::Error::msg("shared expert capture produced no graph"))?;
        let mut previous = None;
        for phase in REPLAY_PHASES {
            source.slice_set(
                &input(SOURCE_QUANTIZED_ROWS, 1, phase, dtype, &device)?,
                0,
                0,
            )?;
            routing.slice_set(&indices(SOURCE_QUANTIZED_ROWS, phase, &device)?, 0, 0)?;
            graph.launch().map_err(candle_core::Error::wrap)?;
            device.synchronize()?;
            let reference_source = source.copy()?;
            let reference_routing = routing.copy()?;
            for (output, layer) in outputs.iter().zip(&layers) {
                compare(
                    output,
                    &layer.gather_forward(&reference_source, &reference_routing)?,
                )?;
            }
            if let Some(previous) = &previous {
                assert!(difference(&outputs[0], previous)? > 0.0);
            }
            previous = Some(outputs[0].copy()?);
        }
        drop(graph);
    }
    Ok(())
}

#[test]
fn shared_expert_decode_warmup_and_graph_replay() -> Result<()> {
    const DECODE_EXPERTS: usize = 128;
    const DECODE_TOPK: usize = 4;
    const DECODE_N: usize = 256;
    const DECODE_K: usize = 1024;

    let _lock = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda_with_stream(0)?;
    let dev = device.as_cuda_device()?;
    if !crate::cutile::nvfp4_supported(dev) {
        return Ok(());
    }
    let first_rows = crate::cutile::device_multiprocessor_count(dev)
        .div_ceil(DECODE_TOPK)
        .max(2);
    let dense_rows = crate::cutile::nvfp4_grouped_min_tokens(DECODE_EXPERTS, DECODE_TOPK);
    if first_rows >= dense_rows {
        return Ok(());
    }
    crate::cutile::register_nvfp4_routing(&device, DECODE_EXPERTS, DECODE_TOPK)?;
    let stream = dev.cuda_stream();
    let _cache = dev.enable_cuda_graph_htod_cache();
    for dtype in [DType::BF16, DType::F16] {
        let layers = PROJECTION_SEEDS
            .map(|seed| {
                Nvfp4Layer::from_normalized_parts(
                    Nvfp4LayerParts {
                        weight: Tensor::full(
                            0x22u8,
                            (DECODE_EXPERTS, DECODE_N, DECODE_K / 2),
                            &device,
                        )?,
                        scales: Tensor::full(
                            F8E4M3::from_f32(0.5),
                            (DECODE_EXPERTS, DECODE_N, DECODE_K / NVFP4_BLOCK_SIZE),
                            &device,
                        )?,
                        global_scales: Tensor::from_vec(
                            (0..DECODE_EXPERTS * DECODE_N)
                                .map(|index| (index / DECODE_N + seed + 1) as f32 / GLOBAL_DIVISOR)
                                .collect::<Vec<_>>(),
                            (DECODE_EXPERTS, DECODE_N),
                            &device,
                        )?,
                        input_scale: Some(Tensor::ones(DECODE_EXPERTS, DType::F32, &device)?),
                        activation: Nvfp4ActivationMode::DynamicBlock,
                        bias: None,
                        dtype,
                    },
                    true,
                )
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let layers: [Nvfp4Layer; 2] = layers.try_into().ok().unwrap();
        crate::cutile::warmup_moe_kernels(&device, true)?;
        let compiled = cutile::tile_kernel::jit_compile_count();
        for rows in [first_rows, first_rows + 1, 16, 32]
            .into_iter()
            .filter(|&rows| rows >= first_rows && rows < dense_rows)
        {
            for routes in [1, DECODE_TOPK] {
                let make_input = |phase: usize| {
                    Tensor::from_vec(
                        (0..rows * routes * DECODE_K)
                            .map(|index| ((index / DECODE_K + phase) % 3 + 1) as f32)
                            .collect::<Vec<_>>(),
                        (rows, routes, DECODE_K),
                        &device,
                    )?
                    .to_dtype(dtype)
                };
                let make_indices = |phase: usize| {
                    Tensor::from_vec(
                        (0..rows * DECODE_TOPK)
                            .map(|index| {
                                ((index / DECODE_TOPK * 3 + index % DECODE_TOPK + phase)
                                    % DECODE_EXPERTS) as u32
                            })
                            .collect::<Vec<_>>(),
                        (rows, DECODE_TOPK),
                        &device,
                    )
                };
                let source = make_input(0)?;
                let routing = make_indices(0)?;
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
                    return Err(candle_core::Error::wrap(error));
                }
                let outputs = shared(&source, &routing, &layers);
                let graph = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                if tracking {
                    unsafe { stream.context().enable_event_tracking() };
                }
                let outputs = outputs?;
                let graph = graph.map_err(candle_core::Error::wrap)?.unwrap();
                assert_eq!(
                    cutile::tile_kernel::jit_compile_count(),
                    compiled,
                    "decode warmup missed rows={rows} routes={routes}"
                );
                for phase in REPLAY_PHASES {
                    source.slice_set(&make_input(phase)?, 0, 0)?;
                    routing.slice_set(&make_indices(phase)?, 0, 0)?;
                    graph.launch().map_err(candle_core::Error::wrap)?;
                    device.synchronize()?;
                    for (output, layer) in outputs.iter().zip(&layers) {
                        compare(output, &layer.gather_forward(&source, &routing)?)?;
                    }
                }
                assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
                drop(graph);
            }
        }
    }
    Ok(())
}
