use super::*;
use candle_core::cuda::cudarc::driver::sys;
use float8::F8E4M3;

const PROBE_EXPERTS: usize = 512;
const PROBE_TOPK: usize = 10;
const PROBE_BATCHES: [usize; 4] = [1, 2, 4, 8];
const PROBE_SHAPES: [(usize, usize); 2] = [(2560, 640), (640, 2560)];
const PROBE_SHARED_ROUTES: [usize; 4] = [0, 5, 8, 10];
const PROBE_ROTATIONS: usize = 64;
const PROBE_WARMUP: usize = 4;
const PROBE_ITERATIONS: usize = 4;
const PROBE_SAMPLES: usize = 5;
const PROBE_CALIBRATION: [f32; 3] = [0.015625, 0.09375, 0.0078125];
const PROBE_BLOCK_SCALES: [f32; 4] = [0.3125, 0.75, 1.5, 2.75];
const PROBE_TOLERANCE: f32 = 0.012;
const PROBE_SEED: u64 = 0x243f6a8885a308d3;
const PROBE_RANDOM_MULTIPLIER: u64 = 6364136223846793005;
const PROBE_UNIFORM_CALIBRATION: f32 = 0.00390625;
const PROBE_PROJECTION_SEEDS: [usize; 2] = [0, 5];
const PROBE_SHARED_SHAPE: (usize, usize) = (640, 2560);
const BOUNDARY_SHAPES: [(usize, usize); 6] = [
    (256, 256),
    (256, 1024),
    (1024, 256),
    (1024, 1024),
    (1024, 4096),
    (4096, 1024),
];
const BOUNDARY_ROUTING: [(usize, usize); 6] =
    [(8, 2), (128, 2), (8, 4), (256, 4), (16, 10), (512, 10)];
const BOUNDARY_BATCHES: [usize; 3] = [2, 4, 8];
const BOUNDARY_CACHE_MULTIPLIER: usize = 2;
const BOUNDARY_ALLOCATION_MULTIPLIER: usize = 2;
const BOUNDARY_BANK_SEED_STRIDE: usize = 7;

fn probe_permute(values: &mut [u32], state: &mut u64) {
    for index in (1..values.len()).rev() {
        *state = state.wrapping_mul(PROBE_RANDOM_MULTIPLIER).wrapping_add(1);
        values.swap(index, (*state >> 32) as usize % (index + 1));
    }
}

fn probe_routing(tokens: usize, shared: usize) -> Vec<Vec<u32>> {
    let mut state = PROBE_SEED;
    (0..PROBE_ROTATIONS)
        .map(|_| {
            let mut experts = (0..PROBE_EXPERTS as u32).collect::<Vec<_>>();
            probe_permute(&mut experts, &mut state);
            let mut ids = Vec::with_capacity(tokens * PROBE_TOPK);
            for _ in 0..tokens {
                ids.extend_from_slice(&experts[..shared]);
                probe_permute(&mut experts[shared..], &mut state);
                ids.extend_from_slice(&experts[shared..PROBE_TOPK]);
            }
            ids
        })
        .collect()
}

fn probe_compare(actual: &Tensor, expected: &Tensor, context: &str) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{context}");
    let width = actual.dim(2)?;
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for (row, (actual, expected)) in actual.chunks(width).zip(expected.chunks(width)).enumerate() {
        let maximum = expected
            .iter()
            .map(|value| value.abs())
            .fold(1.0f32, f32::max);
        for (column, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= PROBE_TOLERANCE * maximum,
                "{context}: row={row} column={column}: {actual} vs {expected}"
            );
        }
    }
    Ok(())
}

fn probe_graph<C, T>(
    device: &Device,
    ids: &[C],
    mut launch: impl FnMut(&C) -> Result<T>,
) -> Result<(f64, f64)> {
    let dev = device.as_cuda_device()?;
    let _cache = dev.enable_cuda_graph_htod_cache();
    for _ in 0..PROBE_WARMUP {
        std::hint::black_box(launch(&ids[0])?);
    }
    device.synchronize()?;
    let stream = dev.cuda_stream();
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
    let mut gpu = Vec::with_capacity(PROBE_SAMPLES);
    let mut host = Vec::with_capacity(PROBE_SAMPLES);
    let calls = (PROBE_ITERATIONS * ids.len()) as f64;
    for _ in 0..PROBE_SAMPLES {
        let begin = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::wrap)?;
        let start = std::time::Instant::now();
        for _ in 0..PROBE_ITERATIONS {
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
    Ok((gpu[PROBE_SAMPLES / 2], host[PROBE_SAMPLES / 2]))
}

fn probe_weights(
    n: usize,
    k: usize,
    seed: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    probe_weights_for_experts(PROBE_EXPERTS, n, k, seed, device)
}

fn probe_weights_for_experts(
    experts: usize,
    n: usize,
    k: usize,
    seed: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let weight = Tensor::from_vec(
        (0..experts * n * k / 2)
            .map(|index| ((index * 37 + index / (k / 2) * 13 + seed * 19) % 256) as u8)
            .collect::<Vec<_>>(),
        (experts, n, k / 2),
        device,
    )?;
    let scales = Tensor::from_vec(
        (0..experts * n * k / BLOCK_SIZE)
            .map(|index| {
                F8E4M3::from_f32(PROBE_BLOCK_SCALES[(index + seed) % PROBE_BLOCK_SCALES.len()])
            })
            .collect::<Vec<_>>(),
        (experts, n, k / BLOCK_SIZE),
        device,
    )?;
    let globals = Tensor::from_vec(
        (0..experts * n)
            .map(|index| ((index + seed) % PROBE_CALIBRATION.len() + 1) as f32 / k as f32)
            .collect::<Vec<_>>(),
        (experts, n),
        device,
    )?;
    Ok((weight, scales, globals))
}

#[test]
#[ignore = "GPU routed GEMM crossover probe; run with --ignored --nocapture --test-threads=1"]
fn nvfp4_grouped_decode_crossover() -> Result<()> {
    use crate::{Nvfp4ActivationMode, Nvfp4Layer, Nvfp4LayerParts, QuantMethod};
    use std::collections::HashSet;

    let device = Device::new_cuda_with_stream(0)?;
    if !nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    for (n, k) in PROBE_SHAPES {
        let (weight, scales, globals) = probe_weights(n, k, PROBE_PROJECTION_SEEDS[0], &device)?;
        let calibration = Tensor::from_vec(
            (0..PROBE_EXPERTS)
                .map(|expert| PROBE_CALIBRATION[expert % PROBE_CALIBRATION.len()])
                .collect::<Vec<_>>(),
            PROBE_EXPERTS,
            &device,
        )?;
        let layer = Nvfp4Layer::from_parts(Nvfp4LayerParts {
            weight: weight.clone(),
            scales: scales.clone(),
            global_scales: globals.clone(),
            input_scale: Some(calibration.clone()),
            activation: Nvfp4ActivationMode::DynamicBlock,
            bias: None,
            dtype: DType::BF16,
        })?;
        let args = Nvfp4GemmArgs {
            weights: &weight,
            weight_scales: &scales,
            weight_global_scale: &globals,
            activation_global_scale: Some(&calibration),
        };
        for tokens in PROBE_BATCHES {
            for per_route in [false, true] {
                let input_routes = if per_route { PROBE_TOPK } else { 1 };
                let input = Tensor::from_vec(
                    (0..tokens * input_routes * k)
                        .map(|index| ((index * 17 + index / k * 7) % 113) as f32 / 29.0 - 1.75)
                        .collect::<Vec<_>>(),
                    (tokens, input_routes, k),
                    &device,
                )?
                .to_dtype(DType::BF16)?;
                for shared in PROBE_SHARED_ROUTES {
                    let host_ids = probe_routing(tokens, shared);
                    let unique = host_ids
                        .iter()
                        .map(|ids| ids.iter().copied().collect::<HashSet<_>>().len())
                        .collect::<Vec<_>>();
                    let working_experts = host_ids
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<HashSet<_>>()
                        .len();
                    let ids = host_ids
                        .into_iter()
                        .map(|ids| Tensor::from_vec(ids, (tokens, PROBE_TOPK), &device))
                        .collect::<Result<Vec<_>>>()?;
                    let current = |ids: &Tensor| layer.gather_forward_raw(&input, ids);
                    let gemv = |ids: &Tensor| {
                        super::super::nvfp4_gemv::launch(&input, Some(ids), args, false)
                    };
                    let grouped = |ids: &Tensor| {
                        super::super::nvfp4_matmul::launch_gather(&input, ids, args, false, false)
                    };
                    for (rotation, ids) in ids.iter().enumerate() {
                        let expected = current(ids)?;
                        let context = format!("n={n} k={k} tokens={tokens} per_route={per_route} shared={shared} rotation={rotation}");
                        probe_compare(&gemv(ids)?, &expected, &format!("{context} gemv"))?;
                        probe_compare(&grouped(ids)?, &expected, &format!("{context} grouped"))?;
                    }
                    let (current_gpu_us, current_host_us) = probe_graph(&device, &ids, current)?;
                    let (gemv_gpu_us, gemv_host_us) = probe_graph(&device, &ids, gemv)?;
                    let (grouped_gpu_us, grouped_host_us) = probe_graph(&device, &ids, grouped)?;
                    let expert_bytes =
                        n * k / 2 + n * k / BLOCK_SIZE + n * std::mem::size_of::<f32>();
                    println!(
                        "{}",
                        serde_json::json!({
                            "tokens": tokens, "n": n, "k": k, "experts": PROBE_EXPERTS,
                            "topk": PROBE_TOPK, "per_route": per_route, "dtype": "BF16", "graph": true,
                            "routing": if shared == 0 { "independent" } else { "correlated" },
                            "shared_routes_per_token": shared,
                            "unique_experts_min": unique.iter().min().unwrap(),
                            "unique_experts_max": unique.iter().max().unwrap(),
                            "unique_experts_mean": unique.iter().sum::<usize>() as f64 / unique.len() as f64,
                            "working_experts": working_experts,
                            "working_weight_bytes": working_experts * expert_bytes,
                            "weight_cache": "rotating", "graph_calls": PROBE_ROTATIONS,
                            "iterations": PROBE_ITERATIONS, "samples": PROBE_SAMPLES,
                            "current_path": if n >= k { "native" } else { "gemv" },
                            "current_gpu_us": current_gpu_us, "current_host_us": current_host_us,
                            "gemv_gpu_us": gemv_gpu_us, "gemv_host_us": gemv_host_us,
                            "grouped_gpu_us": grouped_gpu_us, "grouped_host_us": grouped_host_us,
                            "grouped_speedup": current_gpu_us / grouped_gpu_us,
                        })
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU shared routed GEMM crossover probe; run with --ignored --nocapture --test-threads=1"]
fn nvfp4_shared_grouped_decode_crossover() -> Result<()> {
    use super::super::nvfp4_matmul::{launch_gather, launch_gather_prepared, prepare_gather};
    use std::collections::HashSet;

    let device = Device::new_cuda_with_stream(0)?;
    if !nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let (n, k) = PROBE_SHARED_SHAPE;
    let weights = PROBE_PROJECTION_SEEDS
        .map(|seed| probe_weights(n, k, seed, &device))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let calibration = PROBE_PROJECTION_SEEDS
        .map(|_| Tensor::full(PROBE_UNIFORM_CALIBRATION, PROBE_EXPERTS, &device))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let args = weights
        .iter()
        .zip(&calibration)
        .map(|((weight, scales, globals), calibration)| Nvfp4GemmArgs {
            weights: weight,
            weight_scales: scales,
            weight_global_scale: globals,
            activation_global_scale: Some(calibration),
        })
        .collect::<Vec<_>>();
    for tokens in PROBE_BATCHES {
        for per_route in [false, true] {
            let input_routes = if per_route { PROBE_TOPK } else { 1 };
            let input = Tensor::from_vec(
                (0..tokens * input_routes * k)
                    .map(|index| ((index * 17 + index / k * 7) % 113) as f32 / 29.0 - 1.75)
                    .collect::<Vec<_>>(),
                (tokens, input_routes, k),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            for shared in PROBE_SHARED_ROUTES {
                let host_ids = probe_routing(tokens, shared);
                let unique = host_ids
                    .iter()
                    .map(|ids| ids.iter().copied().collect::<HashSet<_>>().len())
                    .collect::<Vec<_>>();
                let working_experts = host_ids
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<HashSet<_>>()
                    .len();
                let ids = host_ids
                    .into_iter()
                    .map(|ids| Tensor::from_vec(ids, (tokens, PROBE_TOPK), &device))
                    .collect::<Result<Vec<_>>>()?;
                let gemv = |ids: &Tensor| -> Result<[Tensor; 2]> {
                    Ok([
                        super::super::nvfp4_gemv::launch(&input, Some(ids), args[0], false)?,
                        super::super::nvfp4_gemv::launch(&input, Some(ids), args[1], false)?,
                    ])
                };
                let independent_grouped = |ids: &Tensor| -> Result<[Tensor; 2]> {
                    Ok([
                        launch_gather(&input, ids, args[0], true, false)?,
                        launch_gather(&input, ids, args[1], true, false)?,
                    ])
                };
                let shared_grouped = |ids: &Tensor| -> Result<[Tensor; 2]> {
                    let prepared = prepare_gather(&input, ids, args[0], true, false)?;
                    Ok([
                        launch_gather_prepared(&prepared, args[0], false)?,
                        launch_gather_prepared(&prepared, args[1], false)?,
                    ])
                };
                for (rotation, ids) in ids.iter().enumerate() {
                    let expected = gemv(ids)?;
                    let independent = independent_grouped(ids)?;
                    let shared_outputs = shared_grouped(ids)?;
                    for projection in 0..args.len() {
                        let context = format!("tokens={tokens} per_route={per_route} shared={shared} rotation={rotation} projection={projection}");
                        probe_compare(
                            &independent[projection],
                            &expected[projection],
                            &format!("{context} independent"),
                        )?;
                        probe_compare(
                            &shared_outputs[projection],
                            &expected[projection],
                            &format!("{context} shared"),
                        )?;
                    }
                }
                let (gemv_gpu_us, gemv_host_us) = probe_graph(&device, &ids, gemv)?;
                let (independent_gpu_us, independent_host_us) =
                    probe_graph(&device, &ids, independent_grouped)?;
                let (shared_gpu_us, shared_host_us) = probe_graph(&device, &ids, shared_grouped)?;
                let expert_bytes = n * k / 2 + n * k / BLOCK_SIZE + n * std::mem::size_of::<f32>();
                println!(
                    "{}",
                    serde_json::json!({
                        "tokens": tokens, "n": n, "k": k, "experts": PROBE_EXPERTS,
                        "topk": PROBE_TOPK, "per_route": per_route, "dtype": "BF16", "graph": true,
                        "projections": args.len(), "input_calibration": PROBE_UNIFORM_CALIBRATION,
                        "routing": if shared == 0 { "independent" } else { "correlated" },
                        "shared_routes_per_token": shared,
                        "unique_experts_min": unique.iter().min().unwrap(),
                        "unique_experts_max": unique.iter().max().unwrap(),
                        "unique_experts_mean": unique.iter().sum::<usize>() as f64 / unique.len() as f64,
                        "working_experts": working_experts,
                        "working_weight_bytes": args.len() * working_experts * expert_bytes,
                        "weight_cache": "rotating", "graph_calls": PROBE_ROTATIONS,
                        "iterations": PROBE_ITERATIONS, "samples": PROBE_SAMPLES,
                        "gemv_gpu_us": gemv_gpu_us, "gemv_host_us": gemv_host_us,
                        "independent_grouped_gpu_us": independent_gpu_us,
                        "independent_grouped_host_us": independent_host_us,
                        "shared_grouped_gpu_us": shared_gpu_us, "shared_grouped_host_us": shared_host_us,
                        "shared_speedup": gemv_gpu_us / shared_gpu_us,
                        "preparation_sharing_speedup": independent_gpu_us / shared_gpu_us,
                    })
                );
            }
        }
    }
    Ok(())
}

struct BoundaryRouting {
    experts: usize,
    topk: usize,
    banks: usize,
    pair_expert_bytes: usize,
    working_bytes: usize,
}

struct BoundaryCase {
    ids: Tensor,
    bank: usize,
}

impl BoundaryRouting {
    fn cases(
        &self,
        tokens: usize,
        shared: usize,
        device: &Device,
    ) -> Result<(Vec<BoundaryCase>, usize, f64)> {
        use std::collections::HashSet;

        let mut state = PROBE_SEED;
        let mut cases = Vec::new();
        let mut working = HashSet::new();
        let mut unique_total = 0;
        while cases.len() < PROBE_ROTATIONS
            || working.len() * self.pair_expert_bytes < self.working_bytes
        {
            let bank = cases.len() % self.banks;
            let mut experts = (0..self.experts as u32).collect::<Vec<_>>();
            probe_permute(&mut experts, &mut state);
            let mut ids = Vec::with_capacity(tokens * self.topk);
            for _ in 0..tokens {
                ids.extend_from_slice(&experts[..shared]);
                probe_permute(&mut experts[shared..], &mut state);
                ids.extend_from_slice(&experts[shared..self.topk]);
            }
            unique_total += ids.iter().copied().collect::<HashSet<_>>().len();
            working.extend(ids.iter().map(|&expert| (bank, expert)));
            cases.push(BoundaryCase {
                ids: Tensor::from_vec(ids, (tokens, self.topk), device)?,
                bank,
            });
        }
        let unique_mean = unique_total as f64 / cases.len() as f64;
        Ok((cases, working.len() * self.pair_expert_bytes, unique_mean))
    }
}

#[test]
#[ignore = "GPU shared routed GEMM boundary probe; run with --ignored --nocapture --test-threads=1"]
fn nvfp4_shared_grouped_decode_boundaries() -> Result<()> {
    shared_decode_boundaries(
        &BOUNDARY_SHAPES,
        &BOUNDARY_ROUTING,
        &BOUNDARY_BATCHES,
        false,
    )
}

#[test]
#[ignore = "GPU newly admitted shared GEMM probe; run with --ignored --nocapture --test-threads=1"]
fn nvfp4_shared_grouped_decode_larger_batches() -> Result<()> {
    shared_decode_boundaries(
        &[(256, 1024), (640, 2560), (1024, 4096)],
        &[(128, 2), (256, 4), (512, 10)],
        &[16, 32],
        true,
    )
}

fn shared_decode_boundaries(
    shapes: &[(usize, usize)],
    routing_shapes: &[(usize, usize)],
    batches: &[usize],
    newly_admitted_only: bool,
) -> Result<()> {
    use super::super::nvfp4_matmul::{
        launch_gather, launch_gather_prepared, prepare_gather, routed_geometry,
    };
    use crate::{Nvfp4ActivationMode, Nvfp4Layer, Nvfp4LayerParts, QuantMethod};
    use candle_core::cuda::cudarc::driver::result;

    let device = Device::new_cuda_with_stream(0)?;
    let dev = device.as_cuda_device()?;
    if !nvfp4_supported(dev) {
        return Ok(());
    }
    let ordinal = dev.cuda_stream().context().ordinal();
    let handle = result::device::get(ordinal as i32).map_err(candle_core::Error::wrap)?;
    let l2_bytes = unsafe {
        result::device::get_attribute(
            handle,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        )
    }
    .map_err(candle_core::Error::wrap)? as usize;
    assert!(l2_bytes > 0);
    let working_bytes = l2_bytes * BOUNDARY_CACHE_MULTIPLIER;
    for &(n, k) in shapes {
        let expert_bytes = n * k / 2 + n * k / BLOCK_SIZE + n * std::mem::size_of::<f32>();
        let pair_expert_bytes = PROBE_PROJECTION_SEEDS.len() * expert_bytes;
        for &(experts, topk) in routing_shapes {
            let banks = (working_bytes * BOUNDARY_ALLOCATION_MULTIPLIER)
                .div_ceil(experts * pair_expert_bytes)
                .max(1);
            let weights = (0..banks)
                .map(|bank| {
                    PROBE_PROJECTION_SEEDS
                        .map(|seed| {
                            probe_weights_for_experts(
                                experts,
                                n,
                                k,
                                seed + bank * BOUNDARY_BANK_SEED_STRIDE,
                                &device,
                            )
                        })
                        .into_iter()
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            let calibration = PROBE_PROJECTION_SEEDS
                .map(|_| Tensor::full(PROBE_UNIFORM_CALIBRATION, experts, &device))
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let projections = weights
                .iter()
                .map(|bank| {
                    bank.iter()
                        .zip(&calibration)
                        .map(|((weight, scales, globals), calibration)| Nvfp4GemmArgs {
                            weights: weight,
                            weight_scales: scales,
                            weight_global_scale: globals,
                            activation_global_scale: Some(calibration),
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let layers = weights
                .iter()
                .map(|bank| {
                    bank.iter()
                        .zip(&calibration)
                        .map(|((weight, scales, globals), calibration)| {
                            Nvfp4Layer::from_parts(Nvfp4LayerParts {
                                weight: weight.clone(),
                                scales: scales.clone(),
                                global_scales: globals.clone(),
                                input_scale: Some(calibration.clone()),
                                activation: Nvfp4ActivationMode::DynamicBlock,
                                bias: None,
                                dtype: DType::BF16,
                            })
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            let routing = BoundaryRouting {
                experts,
                topk,
                banks,
                pair_expert_bytes,
                working_bytes,
            };
            let mut test_batches = batches.to_vec();
            if newly_admitted_only {
                test_batches.push(grouped_min_tokens(experts, topk) - 1);
                test_batches.sort_unstable();
                test_batches.dedup();
            }
            for tokens in test_batches {
                let newly_selected = tokens < grouped_min_tokens(experts, topk);
                let standalone_path = if !newly_selected {
                    "grouped"
                } else if n >= k
                    && tokens <= 32
                    && tokens * topk * n.div_ceil(super::super::nvfp4_gemv::geometry(n, k).0)
                        > super::super::device_multiprocessor_count(dev)
                {
                    "native"
                } else {
                    "gemv"
                };
                let current_path = if newly_selected {
                    standalone_path
                } else {
                    "shared_grouped"
                };
                let geometry = routed_geometry(tokens, n, k, true);
                let shared_min = shared_grouped_min_tokens(
                    n,
                    k,
                    topk,
                    super::super::device_multiprocessor_count(dev),
                    geometry,
                );
                if newly_admitted_only && (!newly_selected || tokens < shared_min) {
                    continue;
                }
                let input = Tensor::from_vec(
                    (0..tokens * k)
                        .map(|index| ((index * 17 + index / k * 7) % 113) as f32 / 29.0 - 1.75)
                        .collect::<Vec<_>>(),
                    (tokens, 1, k),
                    &device,
                )?
                .to_dtype(DType::BF16)?;
                for shared in [0, topk] {
                    let (cases, active_weight_bytes, unique_mean) =
                        routing.cases(tokens, shared, &device)?;
                    let standalone = |case: &BoundaryCase| -> Result<[Tensor; 2]> {
                        let bank = &layers[case.bank];
                        Ok([
                            bank[0].gather_forward_raw(&input, &case.ids)?,
                            bank[1].gather_forward_raw(&input, &case.ids)?,
                        ])
                    };
                    let current = |case: &BoundaryCase| -> Result<[Tensor; 2]> {
                        if !newly_selected {
                            let outputs = try_cutile_nvfp4_gather_shared(
                                &input,
                                &case.ids,
                                &projections[case.bank],
                            )?
                            .unwrap();
                            Ok(outputs.try_into().unwrap())
                        } else {
                            standalone(case)
                        }
                    };
                    let gemv = |case: &BoundaryCase| -> Result<[Tensor; 2]> {
                        let args = &projections[case.bank];
                        Ok([
                            super::super::nvfp4_gemv::launch(
                                &input,
                                Some(&case.ids),
                                args[0],
                                false,
                            )?,
                            super::super::nvfp4_gemv::launch(
                                &input,
                                Some(&case.ids),
                                args[1],
                                false,
                            )?,
                        ])
                    };
                    let independent_grouped = |case: &BoundaryCase| -> Result<[Tensor; 2]> {
                        let args = &projections[case.bank];
                        Ok([
                            launch_gather(&input, &case.ids, args[0], true, false)?,
                            launch_gather(&input, &case.ids, args[1], true, false)?,
                        ])
                    };
                    let shared_grouped = |case: &BoundaryCase| -> Result<[Tensor; 2]> {
                        let args = &projections[case.bank];
                        let prepared = prepare_gather(&input, &case.ids, args[0], true, false)?;
                        Ok([
                            launch_gather_prepared(&prepared, args[0], false)?,
                            launch_gather_prepared(&prepared, args[1], false)?,
                        ])
                    };
                    for (rotation, case) in cases.iter().enumerate() {
                        let expected = current(case)?;
                        let vectors = gemv(case)?;
                        let standalone_outputs = standalone(case)?;
                        let independent = independent_grouped(case)?;
                        let shared_outputs = shared_grouped(case)?;
                        for projection in 0..PROBE_PROJECTION_SEEDS.len() {
                            let context = format!("n={n} k={k} experts={experts} topk={topk} tokens={tokens} shared={shared} rotation={rotation} projection={projection}");
                            probe_compare(
                                &vectors[projection],
                                &expected[projection],
                                &format!("{context} gemv"),
                            )?;
                            probe_compare(
                                &standalone_outputs[projection],
                                &expected[projection],
                                &format!("{context} standalone"),
                            )?;
                            probe_compare(
                                &independent[projection],
                                &expected[projection],
                                &format!("{context} independent"),
                            )?;
                            probe_compare(
                                &shared_outputs[projection],
                                &expected[projection],
                                &format!("{context} shared"),
                            )?;
                        }
                    }
                    let (current_gpu_us, current_host_us) = probe_graph(&device, &cases, current)?;
                    let (standalone_gpu_us, standalone_host_us) =
                        probe_graph(&device, &cases, standalone)?;
                    let (gemv_gpu_us, gemv_host_us) = probe_graph(&device, &cases, gemv)?;
                    let (independent_gpu_us, independent_host_us) =
                        probe_graph(&device, &cases, independent_grouped)?;
                    let (shared_gpu_us, shared_host_us) =
                        probe_graph(&device, &cases, shared_grouped)?;
                    let serde_json::Value::Object(mut result) = serde_json::json!({
                        "tokens": tokens, "n": n, "k": k, "experts": experts, "topk": topk,
                        "per_route": false, "dtype": "BF16", "graph": true,
                        "projections": PROBE_PROJECTION_SEEDS.len(), "input_calibration": PROBE_UNIFORM_CALIBRATION,
                        "routing": if shared == 0 { "independent" } else { "correlated" },
                        "shared_routes_per_token": shared, "unique_experts_mean": unique_mean,
                        "active_grouped_blocks_mean": unique_mean * n.div_ceil(geometry.columns) as f64,
                        "grouped_tile_rows": geometry.rows, "grouped_tile_columns": geometry.columns,
                        "grouped_tile_k": geometry.k, "l2_bytes": l2_bytes, "weight_banks": banks,
                        "allocated_weight_bytes": banks * experts * pair_expert_bytes,
                        "working_weight_bytes": active_weight_bytes, "weight_cache": "rotating",
                        "graph_calls": cases.len(), "iterations": PROBE_ITERATIONS, "samples": PROBE_SAMPLES,
                        "newly_selected": newly_selected, "current_path": current_path,
                        "standalone_path": standalone_path,
                    }) else {
                        unreachable!()
                    };
                    let serde_json::Value::Object(timings) = serde_json::json!({
                        "current_gpu_us": current_gpu_us, "current_host_us": current_host_us,
                        "standalone_gpu_us": standalone_gpu_us, "standalone_host_us": standalone_host_us,
                        "gemv_gpu_us": gemv_gpu_us, "gemv_host_us": gemv_host_us,
                        "independent_grouped_gpu_us": independent_gpu_us,
                        "independent_grouped_host_us": independent_host_us,
                        "shared_grouped_gpu_us": shared_gpu_us, "shared_grouped_host_us": shared_host_us,
                        "shared_speedup": current_gpu_us / shared_gpu_us,
                        "shared_speedup_vs_gemv": gemv_gpu_us / shared_gpu_us,
                        "preparation_sharing_speedup": independent_gpu_us / shared_gpu_us,
                    }) else {
                        unreachable!()
                    };
                    result.extend(timings);
                    println!("{}", serde_json::Value::Object(result));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn routed_prefill_geometry_preserves_small_and_a16_paths() {
    use super::super::nvfp4_matmul::routed_geometry;
    let short = routed_geometry(512, 2560, 640, true);
    assert_eq!(
        (short.rows, short.columns, short.k, short.persistent),
        (64, 128, 128, true)
    );
    let long = routed_geometry(512, 640, 2560, true);
    assert_eq!(
        (long.rows, long.columns, long.k, long.persistent),
        (16, 128, 256, true)
    );
    let boundary = routed_geometry(4095, 640, 2560, true);
    assert_eq!(boundary, long);
    for tokens in [4096, 4097, 8192] {
        let large = routed_geometry(tokens, 640, 2560, true);
        assert_eq!(
            (large.rows, large.columns, large.k, large.persistent),
            (32, 128, 256, true)
        );
        assert_eq!(routed_geometry(tokens, 2560, 640, true), short);
    }
    for cfg in [
        routed_geometry(511, 2560, 640, true),
        routed_geometry(512, 2560, 640, false),
        routed_geometry(512, 73, 144, true),
    ] {
        assert_eq!(
            (cfg.rows, cfg.columns, cfg.k, cfg.persistent),
            (16, 64, 256, false)
        );
    }
}

#[test]
fn routed_prefill_warmup_covers_alignment_tails_and_graph_replay() -> Result<()> {
    const EXPERTS: usize = 9;
    const TOPK: usize = 2;
    let device = Device::new_cuda_with_stream(0)?;
    let dev = device.as_cuda_device()?;
    if !nvfp4_supported(dev) {
        return Ok(());
    }
    let stream = dev.cuda_stream();
    let _cache = dev.enable_cuda_graph_htod_cache();
    for dtype in [DType::BF16, DType::F16] {
        for (n, k, test_rows) in [
            (513, 640, &[512, 513, 4096, 4097][..]),
            (641, 512, &[512, 513][..]),
        ] {
            let weights = Tensor::from_vec(
                vec![0x22u8; EXPERTS * n * k / 2],
                (EXPERTS, n, k / 2),
                &device,
            )?;
            let scales = Tensor::from_vec(
                vec![F8E4M3::from_f32(0.5); EXPERTS * n * k / 16],
                (EXPERTS, n, k / 16),
                &device,
            )?;
            let globals = Tensor::from_vec(
                (0..EXPERTS * n)
                    .map(|index| (index / n + 1) as f32 / 4.0)
                    .collect::<Vec<_>>(),
                (EXPERTS, n),
                &device,
            )?;
            let activation = Tensor::ones(EXPERTS, DType::F32, &device)?;
            let args = Nvfp4GemmArgs {
                weights: &weights,
                weight_scales: &scales,
                weight_global_scale: &globals,
                activation_global_scale: Some(&activation),
            };
            let geometry = |tokens| super::super::nvfp4_matmul::routed_geometry(tokens, n, k, true);
            for rows in grouped_warmup_rows(EXPERTS, TOPK, geometry)
                .into_iter()
                .filter(|&rows| rows >= super::super::nvfp4_matmul::ROUTED_PREFILL_MIN_TOKENS)
            {
                let indices = Tensor::zeros((rows, TOPK), DType::U32, &device)?;
                let input = Tensor::zeros((rows, k), dtype, &device)?;
                launch(&input, Some(&indices), args, true)?;
            }
            let compiled = cutile::tile_kernel::jit_compile_count();
            for &rows in test_rows {
                let ids = (0..rows * TOPK)
                    .map(|route| ((route / TOPK * 3 + route % TOPK * 7) % EXPERTS) as u32)
                    .collect::<Vec<_>>();
                let indices = Tensor::from_vec(ids.clone(), (rows, TOPK), &device)?;
                let input = Tensor::ones((rows, k), dtype, &device)?;
                let base = ids
                    .iter()
                    .flat_map(|&expert| {
                        std::iter::repeat_n(k as f32 * 33.0 * (expert + 1) as f32 / 256.0, n)
                    })
                    .collect::<Vec<_>>();
                let compare = |output: &Tensor, multiplier: f32| -> Result<()> {
                    let expected = Tensor::from_vec(
                        base.iter().map(|v| v * multiplier).collect::<Vec<_>>(),
                        (rows, TOPK, n),
                        &Device::Cpu,
                    )?
                    .to_dtype(dtype)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                    assert_eq!(
                        output
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        expected,
                        "rows={rows} n={n} k={k} dtype={dtype:?}"
                    );
                    Ok(())
                };
                compare(&launch(&input, Some(&indices), args, false)?, 1.0)?;
                assert_eq!(
                    cutile::tile_kernel::jit_compile_count(),
                    compiled,
                    "warmup missed rows={rows} n={n} k={k}"
                );
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
                let output = launch(&input, Some(&indices), args, false);
                let graph = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                if tracking {
                    unsafe { stream.context().enable_event_tracking() };
                }
                let output = output?;
                let graph = graph
                    .map_err(candle_core::Error::wrap)?
                    .ok_or_else(|| candle_core::Error::msg("NVFP4 capture produced no graph"))?;
                for multiplier in [1f32, 2.0] {
                    input.slice_set(
                        &Tensor::full(multiplier, (rows, k), &device)?.to_dtype(dtype)?,
                        0,
                        0,
                    )?;
                    graph.launch().map_err(candle_core::Error::wrap)?;
                    device.synchronize()?;
                    compare(&output, multiplier)?;
                }
                assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
            }
        }
    }
    Ok(())
}
