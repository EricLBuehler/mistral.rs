use std::{hint::black_box, time::Instant};

use candle_core::{cuda::cudarc::driver::sys, Device, IndexOp};

use super::*;
use crate::{nvfp4::Nvfp4LayerParts, Nvfp4ActivationMode, Nvfp4Layer, QuantMethod};

const EXPERTS: usize = 3;
const ROUTES: usize = 3;
const SHAPES: [(usize, usize); 7] = [
    (17, 16),
    (83, 80),
    (259, 144),
    (1027, 640),
    (2560, 640),
    (79, 144),
    (640, 2560),
];
const BATCHES: [usize; 6] = [1, 2, 4, 8, 16, 32];
const INPUT_GLOBAL: [f32; EXPERTS] = [0.015625, 0.09375, 0.0078125];
const WEIGHT_GLOBAL: [f32; EXPERTS] = [0.03125, 0.09375, 0.15625];
const IDS: [u32; ROUTES] = [2, 0, 2];
const NEXT_IDS: [u32; ROUTES] = [1, 2, 0];
const BF16_TOLERANCE: f32 = 0.012;
const F16_TOLERANCE: f32 = 0.002;
const BENCH_SHAPES: [(usize, usize); 5] = [
    (2560, 640),
    (640, 2560),
    (1024, 1024),
    (2048, 512),
    (1280, 2560),
];
const BENCH_ROUTING: [(usize, usize); 4] = [(512, 10), (8, 2), (64, 4), (512, 1)];
const BENCH_ROTATIONS: usize = 64;
const BENCH_WARMUP: usize = 4;
const BENCH_ITERATIONS: usize = 4;
const BENCH_SAMPLES: usize = 5;
const BENCH_ROUTE_STEP: usize = 13;
const BENCH_ROTATION_STEP: usize = 17;
const BENCH_INPUT: f32 = 6.0;
const BENCH_SCALE_VARIANTS: usize = 4;

fn view(tensor: Tensor) -> Result<Tensor> {
    let count = tensor.elem_count();
    let shape = tensor.shape().clone();
    let padding = tensor.flatten_all()?.narrow(0, 0, 1)?;
    let tensor = Tensor::cat(&[&padding, &tensor.flatten_all()?], 0)?;
    tensor.narrow(0, 1, count)?.reshape(shape)
}

fn layer(n: usize, k: usize, routed: bool, dtype: DType, device: &Device) -> Result<Nvfp4Layer> {
    let experts = if routed { EXPERTS } else { 1 };
    let mut shape = vec![experts, n, k / 2];
    if !routed {
        shape.remove(0);
    }
    let weight = (0..experts * n * k / 2)
        .map(|i| ((i * 37 + i / (k / 2) * 13) % 256) as u8)
        .collect::<Vec<_>>();
    let weight = view(Tensor::from_vec(weight, shape.clone(), device)?)?;
    *shape.last_mut().unwrap() = k / NVFP4_BLOCK_SIZE;
    let scales = (0..experts * n * k / NVFP4_BLOCK_SIZE)
        .map(|i| F8E4M3::from_f32([0.3125, 0.75, 1.5, 2.75, 5.5][i % 5]))
        .collect::<Vec<_>>();
    let scales = view(Tensor::from_vec(scales, shape.clone(), device)?)?;
    shape.pop();
    let global = (0..experts * n)
        .map(|i| WEIGHT_GLOBAL[i % WEIGHT_GLOBAL.len()])
        .collect::<Vec<_>>();
    let global_scales = view(Tensor::from_vec(global, shape.clone(), device)?)?;
    let bias = (0..experts * n)
        .map(|i| (i % 11) as f32 / 64.0)
        .collect::<Vec<_>>();
    let bias = view(Tensor::from_vec(bias, shape, device)?.to_dtype(dtype)?)?;
    let input_scale = if routed {
        Tensor::from_slice(&INPUT_GLOBAL, EXPERTS, device)?
    } else {
        Tensor::new(INPUT_GLOBAL[0], device)?
    };
    Nvfp4Layer::from_parts(Nvfp4LayerParts {
        weight,
        scales,
        global_scales,
        input_scale: Some(view(input_scale)?),
        activation: Nvfp4ActivationMode::DynamicBlock,
        bias: Some(bias),
        dtype,
    })
}

fn input(k: usize, shape: (usize, usize), dtype: DType, device: &Device) -> Result<Tensor> {
    let (tokens, rows) = shape;
    let data = (0..tokens * rows * k)
        .map(|i| {
            if i % k / NVFP4_BLOCK_SIZE == 1 {
                0.0
            } else {
                ((i * 17 + i / k * 7) % 113) as f32 / 29.0 - 1.75
            }
        })
        .collect::<Vec<_>>();
    view(Tensor::from_vec(data, (tokens, rows, k), device)?.to_dtype(dtype)?)
}

fn indices(tokens: usize, ids: &[u32], device: &Device) -> Result<Tensor> {
    let data = (0..tokens)
        .flat_map(|token| (0..ids.len()).map(move |route| ids[(token + route) % ids.len()]))
        .collect::<Vec<_>>();
    view(Tensor::from_vec(data, (tokens, ids.len()), device)?)
}

fn compare(actual: &Tensor, expected: &Tensor) -> Result<()> {
    compare_context(actual, expected, "")
}

fn compare_context(actual: &Tensor, expected: &Tensor, context: &str) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{context}");
    let tolerance = match actual.dtype() {
        DType::BF16 => BF16_TOLERANCE,
        DType::F16 => F16_TOLERANCE,
        _ => unreachable!(),
    };
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let maximum = expected.iter().map(|x| x.abs()).fold(1.0f32, f32::max);
    for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * maximum,
            "{context}: element {i}: {actual} vs {expected}"
        );
    }
    Ok(())
}

fn native_with_bias(layer: &Nvfp4Layer, x: &Tensor, indices: Option<&Tensor>) -> Result<Tensor> {
    let output = forward(x, indices, layer.gemm_args())?;
    let bias = layer.parts.bias.as_ref().unwrap();
    match indices {
        Some(ids) => output.broadcast_add(
            &bias
                .index_select(&ids.flatten_all()?, 0)?
                .reshape(output.shape())?,
        ),
        None => output.broadcast_add(bias),
    }
}

#[test]
fn nvfp4_decode_dense_and_routed_match_cpu_with_views() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    for dtype in [DType::BF16, DType::F16] {
        for (n, k) in SHAPES {
            for (routed, per_route) in [(false, false), (true, false), (true, true)] {
                let gpu = layer(n, k, routed, dtype, &device)?;
                let cpu = layer(n, k, routed, dtype, &Device::Cpu)?;
                let x = input(k, (1, if per_route { ROUTES } else { 1 }), dtype, &device)?;
                let ids = view(Tensor::new(&[IDS], &device)?)?;
                let (actual, expected) = if routed {
                    (
                        native_with_bias(&gpu, &x, Some(&ids))?,
                        cpu.gather_forward_raw(
                            &x.to_device(&Device::Cpu)?,
                            &ids.to_device(&Device::Cpu)?,
                        )?,
                    )
                } else {
                    let x = x.i(0)?;
                    (
                        native_with_bias(&gpu, &x, None)?,
                        cpu.forward_raw(&x.to_device(&Device::Cpu)?)?,
                    )
                };
                compare(&actual, &expected)?;
            }
        }
    }
    Ok(())
}

#[test]
fn nvfp4_decode_batched_routes_match_cpu() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let (n, k) = SHAPES[2];
    for dtype in [DType::BF16, DType::F16] {
        let gpu = layer(n, k, true, dtype, &device)?;
        let cpu = layer(n, k, true, dtype, &Device::Cpu)?;
        for tokens in BATCHES {
            for topk in [1, ROUTES] {
                let ids = indices(tokens, &IDS[..topk], &device)?;
                for per_route in [false, true] {
                    let context = format!(
                        "n={n} k={k} dtype={dtype:?} tokens={tokens} topk={topk} per_route={per_route}"
                    );
                    let check = |actual: &Tensor, expected: &Tensor, path: &str| {
                        compare_context(actual, expected, &format!("{context} path={path}"))
                    };
                    let x = input(
                        k,
                        (tokens, if per_route { topk } else { 1 }),
                        dtype,
                        &device,
                    )?;
                    let expected = cpu.gather_forward_raw(
                        &x.to_device(&Device::Cpu)?,
                        &ids.to_device(&Device::Cpu)?,
                    )?;
                    check(
                        &native_with_bias(&gpu, &x, Some(&ids))?,
                        &expected,
                        "native",
                    )?;
                    check(&gpu.gather_forward_raw(&x, &ids)?, &expected, "public-3d")?;
                    if !per_route {
                        check(
                            &gpu.gather_forward_raw(&x.squeeze(1)?, &ids)?,
                            &expected,
                            "public-2d",
                        )?;
                    }
                    let batched_ids = ids.unsqueeze(0)?;
                    let batched_expected = expected.reshape((1, tokens, topk, 1, n))?;
                    check(
                        &gpu.gather_forward_raw(&x.unsqueeze(0)?.unsqueeze(3)?, &batched_ids)?,
                        &batched_expected,
                        "public-5d",
                    )?;
                    if per_route {
                        check(
                            &gpu.gather_forward_raw(&x.unsqueeze(0)?, &batched_ids)?,
                            &batched_expected,
                            "public-4d",
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn nvfp4_decode_graph_replay_tracks_inputs_and_routes() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let _htod_cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
    let (n, k) = SHAPES[2];
    let gpu = layer(n, k, true, DType::BF16, &device)?;
    let cpu = layer(n, k, true, DType::BF16, &Device::Cpu)?;
    let tokens = BATCHES[3];
    let x = input(k, (tokens, ROUTES), DType::BF16, &device)?;
    let ids = indices(tokens, &IDS, &device)?;
    let expected =
        cpu.gather_forward_raw(&x.to_device(&Device::Cpu)?, &ids.to_device(&Device::Cpu)?)?;
    compare(&native_with_bias(&gpu, &x, Some(&ids))?, &expected)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    device.synchronize()?;
    let tracking = stream.context().is_event_tracking();
    if tracking {
        unsafe { stream.context().disable_event_tracking() };
    }
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(candle_core::Error::wrap)?;
    let output = native_with_bias(&gpu, &x, Some(&ids))?;
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(candle_core::Error::wrap)?
        .unwrap();
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    graph.launch().map_err(candle_core::Error::wrap)?;
    compare(&output, &expected)?;
    ids.slice_set(&indices(tokens, &NEXT_IDS, &device)?, 0, 0)?;
    x.slice_set(&(x.affine(-0.75, 0.125)?), 0, 0)?;
    let expected =
        cpu.gather_forward_raw(&x.to_device(&Device::Cpu)?, &ids.to_device(&Device::Cpu)?)?;
    graph.launch().map_err(candle_core::Error::wrap)?;
    compare(&output, &expected)?;
    Ok(())
}

#[test]
fn nvfp4_decode_dispatch_preserves_prefill_and_k_heavy_paths() -> Result<()> {
    let dense = layer(83, 80, false, DType::F16, &Device::Cpu)?;
    assert!(supports_on_device(1, 1, DType::F16, dense.gemm_args(), 48));
    assert!(!supports_on_device(2, 1, DType::F16, dense.gemm_args(), 48));
    assert!(!supports_on_device(1, 1, DType::F32, dense.gemm_args(), 48));
    let routed = layer(83, 80, true, DType::F16, &Device::Cpu)?;
    for (experts, topk, boundary) in [(8, 2, 8), (64, 4, 32), (512, 10, MAX_TOKENS + 1)] {
        let weights = Tensor::zeros((experts, 1024, 40), DType::U8, &Device::Cpu)?;
        let args = Nvfp4GemmArgs {
            weights: &weights,
            ..routed.gemm_args()
        };
        assert!(supports_on_device(1, topk, DType::F16, args, 48));
        assert!(supports_on_device(boundary - 1, topk, DType::F16, args, 48));
        assert!(!supports_on_device(boundary, topk, DType::F16, args, 48));
    }
    assert!(!supports_on_device(
        0,
        ROUTES,
        DType::F16,
        routed.gemm_args(),
        48
    ));
    assert!(!supports_on_device(
        MAX_TOKENS + 1,
        ROUTES,
        DType::F16,
        routed.gemm_args(),
        48,
    ));
    assert!(!supports_on_device(
        1,
        0,
        DType::F16,
        routed.gemm_args(),
        48
    ));
    assert!(!supports_on_device(
        2,
        MAX_ROUTES,
        DType::F16,
        routed.gemm_args(),
        48
    ));
    assert!(!supports_on_device(
        2,
        usize::MAX,
        DType::F16,
        routed.gemm_args(),
        48
    ));
    let k_heavy = layer(17, 80, false, DType::F16, &Device::Cpu)?;
    assert!(!supports_on_device(
        1,
        1,
        DType::F16,
        k_heavy.gemm_args(),
        48
    ));
    let mut args = dense.gemm_args();
    args.activation_global_scale = None;
    assert!(!supports_on_device(1, 1, DType::F16, args, 48));
    Ok(())
}

#[test]
fn nvfp4_decode_dispatch_leaves_one_wave_to_gemv() -> Result<()> {
    let routed = layer(256, 256, true, DType::BF16, &Device::Cpu)?;
    let weights = Tensor::zeros((128, 256, 128), DType::U8, &Device::Cpu)?;
    let args = Nvfp4GemmArgs {
        weights: &weights,
        ..routed.gemm_args()
    };
    for (tokens, topk, multiprocessors, expected) in [
        (1, 2, 48, true),
        (2, 2, 48, false),
        (3, 2, 48, false),
        (4, 2, 48, true),
        (2, 4, 48, true),
        (4, 2, 80, false),
        (8, 2, 80, true),
        (16, 2, 48, true),
        (32, 2, 48, true),
    ] {
        assert_eq!(
            supports_on_device(tokens, topk, DType::BF16, args, multiprocessors),
            expected,
            "tokens={tokens} topk={topk} multiprocessors={multiprocessors}"
        );
    }
    Ok(())
}

fn benchmark_graph(
    device: &Device,
    ids: &[Tensor],
    mut launch: impl FnMut(&Tensor) -> Result<Tensor>,
) -> Result<(f64, f64)> {
    let cuda = device.as_cuda_device()?;
    let _htod_cache = cuda.enable_cuda_graph_htod_cache();
    for _ in 0..BENCH_WARMUP {
        black_box(launch(&ids[0])?);
    }
    device.synchronize()?;
    let stream = cuda.cuda_stream();
    let tracking = stream.context().is_event_tracking();
    if tracking {
        unsafe { stream.context().disable_event_tracking() };
    }
    let begin = stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED);
    if let Err(error) = begin {
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        return Err(candle_core::Error::wrap(error));
    }
    let outputs = ids.iter().map(&mut launch).collect::<Result<Vec<_>>>();
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let outputs = outputs?;
    let graph = graph.map_err(candle_core::Error::wrap)?.unwrap();
    graph.launch().map_err(candle_core::Error::wrap)?;
    device.synchronize()?;
    let mut gpu = Vec::with_capacity(BENCH_SAMPLES);
    let mut host = Vec::with_capacity(BENCH_SAMPLES);
    let calls = (BENCH_ITERATIONS * ids.len()) as f64;
    for _ in 0..BENCH_SAMPLES {
        let begin = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::wrap)?;
        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            graph.launch().map_err(candle_core::Error::wrap)?;
        }
        let end = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::wrap)?;
        end.synchronize().map_err(candle_core::Error::wrap)?;
        gpu.push(begin.elapsed_ms(&end).map_err(candle_core::Error::wrap)? as f64 * 1000.0 / calls);
        host.push(start.elapsed().as_secs_f64() * 1e6 / calls);
    }
    gpu.sort_by(f64::total_cmp);
    host.sort_by(f64::total_cmp);
    drop(outputs);
    device.synchronize()?;
    drop(graph);
    Ok((gpu[BENCH_SAMPLES / 2], host[BENCH_SAMPLES / 2]))
}

#[test]
#[ignore = "GPU crossover benchmark; run with --ignored --nocapture --test-threads=1"]
fn nvfp4_decode_crossover() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    for (n, k) in BENCH_SHAPES {
        for (experts, topk) in BENCH_ROUTING {
            let weights = Tensor::full(0x22u8, (experts, n, k / 2), &device)?;
            let scales = Tensor::full(
                F8E4M3::from_f32(1.0),
                (experts, n, k / NVFP4_BLOCK_SIZE),
                &device,
            )?;
            let globals = Tensor::from_vec(
                (0..experts)
                    .flat_map(|expert| {
                        std::iter::repeat_n(
                            (expert % BENCH_SCALE_VARIANTS + 1) as f32 / k as f32,
                            n,
                        )
                    })
                    .collect::<Vec<_>>(),
                (experts, n),
                &device,
            )?;
            let input_global = Tensor::ones(experts, DType::F32, &device)?;
            let args = Nvfp4GemmArgs {
                weights: &weights,
                weight_scales: &scales,
                weight_global_scale: &globals,
                activation_global_scale: Some(&input_global),
            };
            for tokens in BATCHES {
                for per_route in [false, true] {
                    let x = Tensor::full(
                        BENCH_INPUT,
                        (tokens, if per_route { topk } else { 1 }, k),
                        &device,
                    )?
                    .to_dtype(DType::BF16)?;
                    for rotating in [false, true] {
                        let ids = (0..BENCH_ROTATIONS)
                            .map(|rotation| {
                                let offset = if rotating {
                                    rotation * BENCH_ROTATION_STEP
                                } else {
                                    0
                                };
                                let ids = (0..tokens * topk)
                                    .map(|route| {
                                        ((offset + route * BENCH_ROUTE_STEP) % experts) as u32
                                    })
                                    .collect::<Vec<_>>();
                                Tensor::from_vec(ids, (tokens, topk), &device)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let native = forward(&x, Some(&ids[0]), args)?;
                        let cutile =
                            crate::cutile::cutile_nvfp4_gather_calibrated(&x, &ids[0], args, true)?;
                        compare(&native, &cutile)?;
                        let (native_gpu_us, native_host_us) =
                            benchmark_graph(&device, &ids, |ids| forward(&x, Some(ids), args))?;
                        let (cutile_gpu_us, cutile_host_us) =
                            benchmark_graph(&device, &ids, |ids| {
                                crate::cutile::cutile_nvfp4_gather_calibrated(&x, ids, args, true)
                            })?;
                        println!(
                            "{}",
                            serde_json::json!({
                                "tokens": tokens, "n": n, "k": k,
                                "experts": experts, "topk": topk, "per_route": per_route,
                                "dtype": "BF16", "graph": true,
                                "weight_cache": if rotating { "rotating" } else { "warm" },
                                "graph_calls": BENCH_ROTATIONS,
                                "iterations": BENCH_ITERATIONS, "samples": BENCH_SAMPLES,
                                "native_gpu_us": native_gpu_us, "native_host_us": native_host_us,
                                "cutile_gpu_us": cutile_gpu_us, "cutile_host_us": cutile_host_us,
                                "speedup": cutile_gpu_us / native_gpu_us,
                            })
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
