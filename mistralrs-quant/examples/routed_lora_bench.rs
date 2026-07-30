#[cfg(feature = "cuda")]
mod cuda_bench {
    use std::{env, sync::Arc, time::Instant};

    use candle_core::{
        cuda::cudarc::driver::{sys, DevicePtr, DevicePtrMut},
        DType, Device, Result, Storage, Tensor,
    };
    use half::bf16;
    #[cfg(feature = "cutile")]
    use mistralrs_quant::cutile::{
        try_cutile_routed_lora, try_cutile_routed_lora_no_sort, CutileRoutedLoraLaunch,
        CutileRoutedLoraStatus,
    };
    use mistralrs_quant::{
        add_expert_delta_reference, launch_routed_lora_direct, launch_routed_lora_grouped,
        with_lora_execution, LoraExecution, LoraExecutionArena, LoraExpertDelta,
        LoraExpertExecution, LoraExpertInputMode, LoraExpertProjection, LoraExpertProjectionNames,
        LoraExpertProjectionWeights, LoraExpertSiteSpec, LoraExpertWeights, LoraLayerRegistry,
        LoraSiteKey, RoutedLoraAdapterWeight, RoutedLoraCudaMetadata, RoutedLoraCudaWeightTable,
        RoutedLoraDirectLaunch, RoutedLoraGroupedLaunch, RoutedLoraInputMode,
        RoutedLoraMetadataLayout, RoutedLoraProjectionLayout, Shard,
    };

    const NUM_EXPERTS: usize = 128;
    const TOP_K: usize = 8;
    const HIDDEN_SIZE: usize = 4096;
    const MOE_INTERMEDIATE_SIZE: usize = 1536;
    const BENCH_RANKS: [usize; 3] = [8, 64, 128];
    const WARMUP: usize = 10;
    const REFERENCE_WARMUP: usize = 1;
    const DEFAULT_ITERS: usize = 50;
    const DEFAULT_REFERENCE_ITERS: usize = 1;
    const DEFAULT_SAMPLES: usize = 10;
    const GENERIC_FALLBACK_CASE: &str = "generic-rank512";

    #[derive(Clone, Copy, Debug)]
    enum RoutePattern {
        Hotset,
        Spread,
        Skewed,
    }

    impl RoutePattern {
        fn name(self) -> &'static str {
            match self {
                Self::Hotset => "hotset",
                Self::Spread => "spread",
                Self::Skewed => "skewed",
            }
        }
    }

    #[derive(Clone, Copy)]
    struct BenchCase {
        name: &'static str,
        num_tokens: usize,
        input_features: usize,
        output_features: usize,
        num_slices: usize,
        input_mode: RoutedLoraInputMode,
        routed_weights: bool,
        num_adapters: usize,
        route_pattern: RoutePattern,
    }

    #[derive(Clone, Copy)]
    struct BenchWork {
        tokens: usize,
        routes: usize,
    }

    fn iterations() -> usize {
        env::var("MISTRALRS_LORA_BENCH_ITERS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|iterations| *iterations != 0)
            .unwrap_or(DEFAULT_ITERS)
    }

    fn reference_iterations() -> usize {
        env::var("MISTRALRS_LORA_BENCH_REFERENCE_ITERS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_REFERENCE_ITERS)
    }

    fn samples(iterations: usize) -> usize {
        env::var("MISTRALRS_LORA_BENCH_SAMPLES")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|samples| *samples != 0)
            .unwrap_or(DEFAULT_SAMPLES)
            .min(iterations)
    }

    fn token_slots(case: BenchCase, alternate: bool) -> Vec<u32> {
        (0..case.num_tokens)
            .map(|token| {
                let slot = if case.num_adapters == 1 {
                    0
                } else {
                    match case.route_pattern {
                        RoutePattern::Hotset => 0,
                        RoutePattern::Spread => token % case.num_adapters,
                        RoutePattern::Skewed if token % 4 != 3 => 0,
                        RoutePattern::Skewed => 1 + (token / 4) % (case.num_adapters - 1),
                    }
                };
                if alternate {
                    ((slot + 1) % case.num_adapters) as u32
                } else {
                    slot as u32
                }
            })
            .collect()
    }

    fn expert_routes(case: BenchCase) -> Vec<u32> {
        (0..case.num_tokens)
            .flat_map(|token| {
                (0..TOP_K).map(move |lane| match case.route_pattern {
                    RoutePattern::Hotset => lane as u32,
                    RoutePattern::Spread => ((token * TOP_K + lane) % NUM_EXPERTS) as u32,
                    RoutePattern::Skewed if lane < TOP_K / 2 => 0,
                    RoutePattern::Skewed => ((token * 3 + lane) % NUM_EXPERTS) as u32,
                })
            })
            .collect()
    }

    fn percentile(sorted: &[f64], percentile: f64) -> f64 {
        let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
        sorted[index]
    }

    fn measure(
        device: &Device,
        name: &str,
        work: BenchWork,
        iterations: usize,
        launch: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        measure_with_warmup(device, name, work, WARMUP, iterations, launch)
    }

    fn measure_with_warmup(
        device: &Device,
        name: &str,
        work: BenchWork,
        warmup: usize,
        iterations: usize,
        mut launch: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        for _ in 0..warmup {
            launch()?;
        }
        device.synchronize()?;

        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        let num_samples = samples(iterations);
        let launches_per_sample = iterations / num_samples;
        let extra_launches = iterations % num_samples;
        let mut gpu_samples = Vec::with_capacity(num_samples);
        let mut wall_samples = Vec::with_capacity(num_samples);
        let mut total_gpu_micros = 0.0;
        let mut total_wall_micros = 0.0;
        for sample in 0..num_samples {
            let launches = launches_per_sample + usize::from(sample < extra_launches);
            let start_event = stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| {
                    candle_core::Error::Msg(format!("CUDA start event failed: {error}"))
                })?;
            let wall_start = Instant::now();
            for _ in 0..launches {
                launch()?;
            }
            let end_event = stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| {
                    candle_core::Error::Msg(format!("CUDA end event failed: {error}"))
                })?;
            end_event.synchronize().map_err(|error| {
                candle_core::Error::Msg(format!("CUDA end event synchronization failed: {error}"))
            })?;
            let wall_micros = wall_start.elapsed().as_secs_f64() * 1e6;
            let gpu_micros = start_event.elapsed_ms(&end_event).map_err(|error| {
                candle_core::Error::Msg(format!("CUDA event timing failed: {error}"))
            })? as f64
                * 1e3;
            gpu_samples.push(gpu_micros / launches as f64);
            wall_samples.push(wall_micros / launches as f64);
            total_gpu_micros += gpu_micros;
            total_wall_micros += wall_micros;
        }
        gpu_samples.sort_by(f64::total_cmp);
        wall_samples.sort_by(f64::total_cmp);
        let gpu_mean = total_gpu_micros / iterations as f64;
        let wall_mean = total_wall_micros / iterations as f64;
        let tokens_per_second = work.tokens as f64 * 1e6 / gpu_mean;
        let routes_per_second = work.routes as f64 * 1e6 / gpu_mean;
        println!(
            "{name}: gpu mean={gpu_mean:.3} us sample-p50={:.3} us sample-p95={:.3} us; wall mean={wall_mean:.3} us sample-p95={:.3} us; {tokens_per_second:.0} tok/s {routes_per_second:.0} routes/s",
            percentile(&gpu_samples, 0.50),
            percentile(&gpu_samples, 0.95),
            percentile(&wall_samples, 0.95),
        );
        Ok(())
    }

    fn run_case(device: &Device, case: BenchCase, rank: usize) -> Result<()> {
        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        let routes = case.num_tokens * TOP_K;
        let work = BenchWork {
            tokens: case.num_tokens,
            routes,
        };
        let layout =
            RoutedLoraMetadataLayout::new(case.num_tokens, TOP_K, NUM_EXPERTS, case.num_adapters)?;
        let projection = RoutedLoraProjectionLayout::new(
            case.input_features,
            case.output_features,
            case.output_features * case.num_slices,
            case.output_features,
            case.num_slices,
            rank,
            case.input_mode,
        )?;

        let input_rows = match case.input_mode {
            RoutedLoraInputMode::TokenRows => case.num_tokens,
            RoutedLoraInputMode::RoutedRows => routes,
        };
        let input = Tensor::ones((input_rows, case.input_features), DType::BF16, device)?;
        let mut output =
            cuda.alloc_zeros::<bf16>(routes * case.output_features * case.num_slices)?;
        let rank_stride = rank;
        let a_adapter_elements = NUM_EXPERTS * rank_stride * case.input_features;
        let b_adapter_elements = NUM_EXPERTS * case.output_features * rank_stride;
        let a_slice_elements = case.num_adapters * a_adapter_elements;
        let b_slice_elements = case.num_adapters * b_adapter_elements;
        let a = Tensor::ones((case.num_slices * a_slice_elements,), DType::BF16, device)?;
        let b = Tensor::ones((case.num_slices * b_slice_elements,), DType::BF16, device)?;
        let mut scales = unsafe { cuda.alloc::<f32>(case.num_adapters * NUM_EXPERTS)? };
        cuda.memcpy_htod(
            &(0..case.num_adapters * NUM_EXPERTS)
                .map(|index| 0.5 + (index % 17) as f32 / 32.0)
                .collect::<Vec<_>>(),
            &mut scales,
        )?;
        let mut route_scales = unsafe { cuda.alloc::<f32>(routes)? };
        cuda.memcpy_htod(
            &(0..routes)
                .map(|route| 0.5 + (route % 7) as f32 / 8.0)
                .collect::<Vec<_>>(),
            &mut route_scales,
        )?;
        let mut device_token_slots = unsafe { cuda.alloc::<u32>(case.num_tokens)? };
        cuda.memcpy_htod(&token_slots(case, false), &mut device_token_slots)?;
        let mut topk_ids = unsafe { cuda.alloc::<u32>(routes)? };
        cuda.memcpy_htod(&expert_routes(case), &mut topk_ids)?;

        let (input_storage, _) = input.storage_and_layout();
        let Storage::Cuda(input_storage) = &*input_storage else {
            unreachable!()
        };
        let (input_ptr, _input_guard) = input_storage.as_cuda_slice::<bf16>()?.device_ptr(&stream);
        let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);
        let (a_storage, _) = a.storage_and_layout();
        let Storage::Cuda(a_storage) = &*a_storage else {
            unreachable!()
        };
        let (a_ptr, _a_guard) = a_storage.as_cuda_slice::<bf16>()?.device_ptr(&stream);
        let (b_storage, _) = b.storage_and_layout();
        let Storage::Cuda(b_storage) = &*b_storage else {
            unreachable!()
        };
        let (b_ptr, _b_guard) = b_storage.as_cuda_slice::<bf16>()?.device_ptr(&stream);
        let (scales_ptr, _scales_guard) = scales.device_ptr_mut(&stream);
        let (route_scales_ptr, _route_scales_guard) = route_scales.device_ptr_mut(&stream);
        let (token_slots_ptr, _token_slots_guard) = device_token_slots.device_ptr(&stream);
        let (topk_ptr, _topk_guard) = topk_ids.device_ptr(&stream);
        let element_bytes = std::mem::size_of::<bf16>() as u64;
        let descriptors = (0..case.num_slices)
            .flat_map(|slice| {
                (0..case.num_adapters).map(move |adapter| RoutedLoraAdapterWeight {
                    a: a_ptr
                        + (slice * a_slice_elements + adapter * a_adapter_elements) as u64
                            * element_bytes,
                    b: b_ptr
                        + (slice * b_slice_elements + adapter * b_adapter_elements) as u64
                            * element_bytes,
                    scales: scales_ptr
                        + (adapter * NUM_EXPERTS * std::mem::size_of::<f32>()) as u64,
                    rank: rank as u32,
                    rank_stride: rank_stride as u32,
                    scale: 1.0 / (adapter + 1) as f32,
                    flags: 0,
                })
            })
            .collect::<Vec<_>>();
        let table =
            RoutedLoraCudaWeightTable::new(cuda, &descriptors, case.num_slices, case.num_adapters)?;
        let iterations = iterations();
        let prefix = format!(
            "{} routes={} top_k={} K={} N={} rank={} adapters={} pattern={}",
            case.name,
            routes,
            TOP_K,
            case.input_features,
            case.output_features,
            rank,
            case.num_adapters,
            case.route_pattern.name(),
        );

        let route_scales_ptr = if case.routed_weights {
            route_scales_ptr
        } else {
            0
        };
        let small_direct = rank <= 64 && routes * rank <= 1024;
        let sparse_no_sort = rank <= 128 && routes * 8 <= NUM_EXPERTS * case.num_adapters;
        if small_direct {
            for output_splits in [Some(1), None] {
                let label = match output_splits {
                    Some(1) => format!("{prefix} kernel-steady-state native-direct split=1"),
                    _ => format!("{prefix} kernel-steady-state native-direct split=auto"),
                };
                measure(device, &label, work, iterations, || unsafe {
                    launch_routed_lora_direct(
                        &table,
                        RoutedLoraDirectLaunch {
                            input: input_ptr,
                            output: output_ptr,
                            token_adapter_slots: token_slots_ptr,
                            topk_expert_ids: topk_ptr,
                            route_input_rows: 0,
                            route_output_rows: 0,
                            route_output_scales: route_scales_ptr,
                            dtype: DType::BF16,
                            metadata: layout,
                            projection,
                            weight_slice_offset: 0,
                            output_splits,
                        },
                    )
                })?;
                device.synchronize()?;
            }
        } else if sparse_no_sort {
            for output_splits in [Some(1), None] {
                let suffix = if output_splits.is_some() {
                    "split=1"
                } else {
                    "split=auto"
                };
                measure(
                    device,
                    &format!("{prefix} kernel-steady-state native-direct-generic {suffix}"),
                    work,
                    iterations,
                    || unsafe {
                        launch_routed_lora_direct(
                            &table,
                            RoutedLoraDirectLaunch {
                                input: input_ptr,
                                output: output_ptr,
                                token_adapter_slots: token_slots_ptr,
                                topk_expert_ids: topk_ptr,
                                route_input_rows: 0,
                                route_output_rows: 0,
                                route_output_scales: route_scales_ptr,
                                dtype: DType::BF16,
                                metadata: layout,
                                projection,
                                weight_slice_offset: 0,
                                output_splits,
                            },
                        )
                    },
                )?;
            }
            #[cfg(feature = "cutile")]
            {
                let launch = CutileRoutedLoraLaunch {
                    input: input_ptr,
                    output: output_ptr,
                    route_input_rows: 0,
                    route_output_rows: 0,
                    route_output_scales: route_scales_ptr,
                    dtype: DType::BF16,
                    projection,
                    weight_slice_offset: 0,
                };
                match unsafe {
                    try_cutile_routed_lora_no_sort(
                        cuda,
                        layout,
                        Some(&device_token_slots),
                        &topk_ids,
                        0,
                        &table,
                        launch,
                    )?
                } {
                    CutileRoutedLoraStatus::Launched => measure(
                        device,
                        &format!("{prefix} kernel-steady-state cutile-no-sort"),
                        work,
                        iterations,
                        || match unsafe {
                            try_cutile_routed_lora_no_sort(
                                cuda,
                                layout,
                                Some(&device_token_slots),
                                &topk_ids,
                                0,
                                &table,
                                launch,
                            )?
                        } {
                            CutileRoutedLoraStatus::Launched => Ok(()),
                            CutileRoutedLoraStatus::Unsupported(reason) => {
                                candle_core::bail!(
                                    "cuTile no-sort routed LoRA became unsupported: {reason:?}"
                                )
                            }
                        },
                    )?,
                    CutileRoutedLoraStatus::Unsupported(reason) => {
                        println!("{prefix} cutile-no-sort: skipped ({reason:?})");
                    }
                }
            }
        } else {
            for output_splits in [Some(1), None] {
                let suffix = if output_splits.is_some() {
                    "split=1"
                } else {
                    "split=auto"
                };
                measure(
                    device,
                    &format!("{prefix} kernel-steady-state native-direct-generic {suffix}"),
                    work,
                    iterations,
                    || unsafe {
                        launch_routed_lora_direct(
                            &table,
                            RoutedLoraDirectLaunch {
                                input: input_ptr,
                                output: output_ptr,
                                token_adapter_slots: token_slots_ptr,
                                topk_expert_ids: topk_ptr,
                                route_input_rows: 0,
                                route_output_rows: 0,
                                route_output_scales: route_scales_ptr,
                                dtype: DType::BF16,
                                metadata: layout,
                                projection,
                                weight_slice_offset: 0,
                                output_splits,
                            },
                        )
                    },
                )?;
            }
            let mut metadata = RoutedLoraCudaMetadata::new(cuda, layout)?;
            unsafe { metadata.build(token_slots_ptr, topk_ptr)? };
            measure(
                device,
                &format!("{prefix} routing-metadata-rebuild-only"),
                work,
                iterations,
                || unsafe { metadata.build(token_slots_ptr, topk_ptr) },
            )?;
            let mut scratch =
                cuda.alloc_zeros::<f32>(layout.hidden_elements(case.num_slices, rank)?)?;
            let (scratch_ptr, _scratch_guard) = scratch.device_ptr_mut(&stream);
            measure(
                device,
                &format!("{prefix} kernel-steady-state native-grouped"),
                work,
                iterations,
                || unsafe {
                    launch_routed_lora_grouped(
                        &metadata,
                        &table,
                        RoutedLoraGroupedLaunch {
                            input: input_ptr,
                            hidden: scratch_ptr,
                            output: output_ptr,
                            route_input_rows: 0,
                            route_output_rows: 0,
                            route_output_scales: route_scales_ptr,
                            dtype: DType::BF16,
                            projection,
                            weight_slice_offset: 0,
                        },
                    )
                },
            )?;
            #[cfg(feature = "cutile")]
            {
                let launch = CutileRoutedLoraLaunch {
                    input: input_ptr,
                    output: output_ptr,
                    route_input_rows: 0,
                    route_output_rows: 0,
                    route_output_scales: route_scales_ptr,
                    dtype: DType::BF16,
                    projection,
                    weight_slice_offset: 0,
                };
                match unsafe { try_cutile_routed_lora(cuda, &metadata, &table, launch)? } {
                    CutileRoutedLoraStatus::Launched => measure(
                        device,
                        &format!("{prefix} kernel-steady-state cutile-grouped"),
                        work,
                        iterations,
                        || match unsafe { try_cutile_routed_lora(cuda, &metadata, &table, launch)? }
                        {
                            CutileRoutedLoraStatus::Launched => Ok(()),
                            CutileRoutedLoraStatus::Unsupported(reason) => {
                                candle_core::bail!(
                                    "cuTile routed LoRA became unsupported: {reason:?}"
                                )
                            }
                        },
                    )?,
                    CutileRoutedLoraStatus::Unsupported(reason) => {
                        println!("{prefix} cutile-grouped: skipped ({reason:?})");
                    }
                }
            }
        }
        Ok(())
    }

    fn projection_weights(
        device: &Device,
        input_features: usize,
        output_features: usize,
        rank: usize,
    ) -> Result<LoraExpertProjectionWeights> {
        LoraExpertProjectionWeights::new(
            Tensor::ones((NUM_EXPERTS, rank, input_features), DType::BF16, device)?,
            Tensor::ones((NUM_EXPERTS, output_features, rank), DType::BF16, device)?,
            Tensor::ones(NUM_EXPERTS, DType::F32, device)?,
        )
    }

    fn run_dispatch_case(device: &Device, case: BenchCase, rank: usize) -> Result<()> {
        let registry = LoraLayerRegistry::new();
        let site = registry.register_expert(
            LoraSiteKey::new(format!("bench.{}", case.name)),
            LoraExpertSiteSpec::new(
                NUM_EXPERTS,
                HIDDEN_SIZE,
                MOE_INTERMEDIATE_SIZE,
                LoraExpertProjectionNames::new("gate_proj", "up_proj", "down_proj"),
                Shard::default(),
                Shard::default(),
            )?,
            DType::BF16,
            device.clone(),
        )?;
        registry.finalize()?;
        let weights = (0..case.num_adapters)
            .map(|_| {
                let weights = if case.num_slices == 2 {
                    LoraExpertWeights::new(
                        &site,
                        Some(projection_weights(
                            device,
                            HIDDEN_SIZE,
                            MOE_INTERMEDIATE_SIZE,
                            rank,
                        )?),
                        Some(projection_weights(
                            device,
                            HIDDEN_SIZE,
                            MOE_INTERMEDIATE_SIZE,
                            rank,
                        )?),
                        None,
                    )?
                } else {
                    LoraExpertWeights::new(
                        &site,
                        None,
                        None,
                        Some(projection_weights(
                            device,
                            MOE_INTERMEDIATE_SIZE,
                            HIDDEN_SIZE,
                            rank,
                        )?),
                    )?
                };
                Ok(Arc::new(weights))
            })
            .collect::<Result<Vec<_>>>()?;
        let arena = Arc::new(LoraExecutionArena::new());
        let make_execution = |alternate: bool| -> Result<Arc<LoraExecution>> {
            let row_slots = token_slots(case, alternate)
                .into_iter()
                .map(|slot| Some(slot + 1))
                .collect();
            let mut execution =
                LoraExecution::new_with_arena(registry.runtime_id(), row_slots, arena.clone());
            for (adapter, weights) in weights.iter().enumerate() {
                execution.insert_expert_shared(&site, adapter as u32 + 1, weights.clone())?;
            }
            Ok(Arc::new(execution))
        };
        let execution = make_execution(false)?;
        let alternate_execution = (case.num_adapters > 1)
            .then(|| make_execution(true))
            .transpose()?;
        let routes = case.num_tokens * TOP_K;
        let work = BenchWork {
            tokens: case.num_tokens,
            routes,
        };
        let topk_ids = Tensor::from_vec(expert_routes(case), (case.num_tokens, TOP_K), device)?;
        let input_shape = match case.input_mode {
            RoutedLoraInputMode::TokenRows => (case.num_tokens, case.input_features),
            RoutedLoraInputMode::RoutedRows => (routes, case.input_features),
        };
        let input = Tensor::ones(input_shape, DType::BF16, device)?;
        let base = Tensor::zeros(
            (routes, case.output_features * case.num_slices),
            DType::BF16,
            device,
        )?;
        let routed_weights = if case.routed_weights {
            Some(Tensor::from_vec(
                (0..routes)
                    .map(|route| 0.5 + (route % 7) as f32 / 8.0)
                    .collect::<Vec<_>>(),
                (case.num_tokens, TOP_K),
                device,
            )?)
        } else {
            None
        };
        let prefix = format!(
            "{} routes={} top_k={} K={} N={} rank={} adapters={} pattern={}",
            case.name,
            routes,
            TOP_K,
            case.input_features,
            case.output_features,
            rank,
            case.num_adapters,
            case.route_pattern.name(),
        );

        if case.num_slices == 2 {
            measure(
                device,
                &format!("{prefix} dispatch-steady-state"),
                work,
                iterations(),
                || {
                    with_lora_execution(Some(execution.clone()), || -> Result<()> {
                        let expert = LoraExpertExecution::current(&site)?
                            .expect("benchmark expert LoRA site is active");
                        let _ = expert.add_gate_up_delta_combined_owned(
                            &input,
                            base.clone(),
                            &topk_ids,
                        )?;
                        Ok(())
                    })
                },
            )?;
            if let Some(alternate_execution) = &alternate_execution {
                let before = arena.cuda_stats();
                let mut next_execution = 0;
                measure(
                    device,
                    &format!("{prefix} dispatch-reconfigure+kernel"),
                    work,
                    iterations(),
                    || {
                        let active = if next_execution % 2 == 0 {
                            execution.clone()
                        } else {
                            alternate_execution.clone()
                        };
                        next_execution += 1;
                        with_lora_execution(Some(active), || -> Result<()> {
                            let expert = LoraExpertExecution::current(&site)?
                                .expect("benchmark expert LoRA site is active");
                            let _ = expert.add_gate_up_delta_combined_owned(
                                &input,
                                base.clone(),
                                &topk_ids,
                            )?;
                            Ok(())
                        })
                    },
                )?;
                let after = arena.cuda_stats();
                println!(
                    "{prefix} reconfigure counters including warmup: weight_tables={} token_slots={} metadata_builds={}",
                    after.weight_table_uploads - before.weight_table_uploads,
                    after.token_slot_uploads - before.token_slot_uploads,
                    after.metadata_builds - before.metadata_builds,
                );
            }
            let shaped = base.reshape((case.num_tokens, TOP_K, MOE_INTERMEDIATE_SIZE * 2))?;
            let gate_base = shaped.narrow(candle_core::D::Minus1, 0, MOE_INTERMEDIATE_SIZE)?;
            let up_base = shaped.narrow(
                candle_core::D::Minus1,
                MOE_INTERMEDIATE_SIZE,
                MOE_INTERMEDIATE_SIZE,
            )?;
            let reference_iterations = reference_iterations();
            if reference_iterations != 0 {
                measure_with_warmup(
                    device,
                    &format!("{prefix} candle-reference"),
                    work,
                    REFERENCE_WARMUP,
                    reference_iterations,
                    || {
                        let _ = add_expert_delta_reference(
                            &execution,
                            &site,
                            LoraExpertDelta::new(
                                LoraExpertProjection::Gate,
                                &input,
                                gate_base.clone(),
                                &topk_ids,
                                LoraExpertInputMode::TokenRows,
                            ),
                        )?;
                        let _ = add_expert_delta_reference(
                            &execution,
                            &site,
                            LoraExpertDelta::new(
                                LoraExpertProjection::Up,
                                &input,
                                up_base.clone(),
                                &topk_ids,
                                LoraExpertInputMode::TokenRows,
                            ),
                        )?;
                        Ok(())
                    },
                )?;
            }
        } else {
            let input = input.reshape((case.num_tokens, TOP_K, case.input_features))?;
            let base = base.reshape((case.num_tokens, TOP_K, case.output_features))?;
            measure(
                device,
                &format!("{prefix} dispatch-steady-state"),
                work,
                iterations(),
                || {
                    with_lora_execution(Some(execution.clone()), || -> Result<()> {
                        let expert = LoraExpertExecution::current(&site)?
                            .expect("benchmark expert LoRA site is active");
                        let _ = expert.add_delta_owned(
                            LoraExpertProjection::Down,
                            &input,
                            base.clone(),
                            &topk_ids,
                            routed_weights.as_ref(),
                            LoraExpertInputMode::RoutedRows,
                        )?;
                        Ok(())
                    })
                },
            )?;
            if let Some(alternate_execution) = &alternate_execution {
                let before = arena.cuda_stats();
                let mut next_execution = 0;
                measure(
                    device,
                    &format!("{prefix} dispatch-reconfigure+kernel"),
                    work,
                    iterations(),
                    || {
                        let active = if next_execution % 2 == 0 {
                            execution.clone()
                        } else {
                            alternate_execution.clone()
                        };
                        next_execution += 1;
                        with_lora_execution(Some(active), || -> Result<()> {
                            let expert = LoraExpertExecution::current(&site)?
                                .expect("benchmark expert LoRA site is active");
                            let _ = expert.add_delta_owned(
                                LoraExpertProjection::Down,
                                &input,
                                base.clone(),
                                &topk_ids,
                                routed_weights.as_ref(),
                                LoraExpertInputMode::RoutedRows,
                            )?;
                            Ok(())
                        })
                    },
                )?;
                let after = arena.cuda_stats();
                println!(
                    "{prefix} reconfigure counters including warmup: weight_tables={} token_slots={} metadata_builds={}",
                    after.weight_table_uploads - before.weight_table_uploads,
                    after.token_slot_uploads - before.token_slot_uploads,
                    after.metadata_builds - before.metadata_builds,
                );
            }
            let reference_iterations = reference_iterations();
            if reference_iterations != 0 {
                measure_with_warmup(
                    device,
                    &format!("{prefix} candle-reference"),
                    work,
                    REFERENCE_WARMUP,
                    reference_iterations,
                    || {
                        let mut delta = LoraExpertDelta::new(
                            LoraExpertProjection::Down,
                            &input,
                            base.clone(),
                            &topk_ids,
                            LoraExpertInputMode::RoutedRows,
                        );
                        if let Some(routed_weights) = &routed_weights {
                            delta = delta.with_routed_weights(routed_weights);
                        }
                        let _ = add_expert_delta_reference(&execution, &site, delta)?;
                        Ok(())
                    },
                )?;
            }
        }
        println!("{prefix} full-dispatch cache: {:?}", arena.cuda_stats());
        Ok(())
    }

    fn benchmark_ranks(profile: &str) -> Result<Vec<usize>> {
        if let Ok(ranks) = env::var("MISTRALRS_LORA_BENCH_RANKS") {
            let ranks = ranks
                .split(',')
                .map(|rank| {
                    rank.trim().parse::<usize>().map_err(|error| {
                        candle_core::Error::Msg(format!("invalid benchmark rank {rank:?}: {error}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if ranks.is_empty() || ranks.contains(&0) {
                candle_core::bail!("benchmark ranks must be nonzero");
            }
            return Ok(ranks);
        }
        Ok(match profile {
            "quick" => vec![8],
            "full" => vec![8, 64, 96, 128],
            _ => BENCH_RANKS.to_vec(),
        })
    }

    fn case_is_selected(name: &str) -> bool {
        env::var("MISTRALRS_LORA_BENCH_CASES")
            .ok()
            .is_none_or(|selected| {
                selected
                    .split(',')
                    .map(str::trim)
                    .any(|selected| selected == name)
            })
    }

    fn benchmark_cases(profile: &str) -> Result<Vec<BenchCase>> {
        let mut cases = vec![
            BenchCase {
                name: "decode-gate-t1",
                num_tokens: 1,
                input_features: HIDDEN_SIZE,
                output_features: MOE_INTERMEDIATE_SIZE,
                num_slices: 2,
                input_mode: RoutedLoraInputMode::TokenRows,
                routed_weights: false,
                num_adapters: 1,
                route_pattern: RoutePattern::Hotset,
            },
            BenchCase {
                name: "decode-down-t1",
                num_tokens: 1,
                input_features: MOE_INTERMEDIATE_SIZE,
                output_features: HIDDEN_SIZE,
                num_slices: 1,
                input_mode: RoutedLoraInputMode::RoutedRows,
                routed_weights: true,
                num_adapters: 1,
                route_pattern: RoutePattern::Hotset,
            },
            BenchCase {
                name: "batch-gate-t16",
                num_tokens: 16,
                input_features: HIDDEN_SIZE,
                output_features: MOE_INTERMEDIATE_SIZE,
                num_slices: 2,
                input_mode: RoutedLoraInputMode::TokenRows,
                routed_weights: false,
                num_adapters: 4,
                route_pattern: RoutePattern::Spread,
            },
        ];
        if profile != "quick" {
            cases.extend([
                BenchCase {
                    name: "batch-down-t64",
                    num_tokens: 64,
                    input_features: MOE_INTERMEDIATE_SIZE,
                    output_features: HIDDEN_SIZE,
                    num_slices: 1,
                    input_mode: RoutedLoraInputMode::RoutedRows,
                    routed_weights: true,
                    num_adapters: 4,
                    route_pattern: RoutePattern::Skewed,
                },
                BenchCase {
                    name: "prefill-gate-t256",
                    num_tokens: 256,
                    input_features: HIDDEN_SIZE,
                    output_features: MOE_INTERMEDIATE_SIZE,
                    num_slices: 2,
                    input_mode: RoutedLoraInputMode::TokenRows,
                    routed_weights: false,
                    num_adapters: 4,
                    route_pattern: RoutePattern::Spread,
                },
            ]);
        }
        if profile == "full" {
            cases.extend([
                BenchCase {
                    name: "batch-down-t16",
                    num_tokens: 16,
                    input_features: MOE_INTERMEDIATE_SIZE,
                    output_features: HIDDEN_SIZE,
                    num_slices: 1,
                    input_mode: RoutedLoraInputMode::RoutedRows,
                    routed_weights: true,
                    num_adapters: 4,
                    route_pattern: RoutePattern::Spread,
                },
                BenchCase {
                    name: "batch-gate-t64",
                    num_tokens: 64,
                    input_features: HIDDEN_SIZE,
                    output_features: MOE_INTERMEDIATE_SIZE,
                    num_slices: 2,
                    input_mode: RoutedLoraInputMode::TokenRows,
                    routed_weights: false,
                    num_adapters: 4,
                    route_pattern: RoutePattern::Skewed,
                },
                BenchCase {
                    name: "prefill-down-t256",
                    num_tokens: 256,
                    input_features: MOE_INTERMEDIATE_SIZE,
                    output_features: HIDDEN_SIZE,
                    num_slices: 1,
                    input_mode: RoutedLoraInputMode::RoutedRows,
                    routed_weights: true,
                    num_adapters: 4,
                    route_pattern: RoutePattern::Spread,
                },
            ]);
        }
        if let Ok(selected) = env::var("MISTRALRS_LORA_BENCH_CASES") {
            cases.retain(|case| {
                selected
                    .split(',')
                    .map(str::trim)
                    .any(|selected| selected == case.name)
            });
            if cases.is_empty() && !(profile == "full" && case_is_selected(GENERIC_FALLBACK_CASE)) {
                candle_core::bail!("MISTRALRS_LORA_BENCH_CASES selected no benchmark cases");
            }
        }
        Ok(cases)
    }

    pub fn main() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let profile =
            env::var("MISTRALRS_LORA_BENCH_PROFILE").unwrap_or_else(|_| "standard".to_string());
        if !matches!(profile.as_str(), "quick" | "standard" | "full") {
            candle_core::bail!("benchmark profile must be quick, standard, or full");
        }
        let ranks = benchmark_ranks(&profile)?;
        let cases = benchmark_cases(&profile)?;
        println!(
            "routed LoRA profile={profile} ranks={ranks:?} cases={} iterations={} samples={} reference_iterations={}",
            cases.len(),
            iterations(),
            samples(iterations()),
            reference_iterations(),
        );
        println!("gpu timings use CUDA events; wall timings include launch and dispatch overhead");
        for rank in ranks {
            for &case in &cases {
                run_case(&device, case, rank)?;
                run_dispatch_case(&device, case, rank)?;
            }
        }
        if profile == "full" && case_is_selected(GENERIC_FALLBACK_CASE) {
            run_case(
                &device,
                BenchCase {
                    name: GENERIC_FALLBACK_CASE,
                    num_tokens: 4,
                    input_features: 257,
                    output_features: 193,
                    num_slices: 1,
                    input_mode: RoutedLoraInputMode::RoutedRows,
                    routed_weights: true,
                    num_adapters: 4,
                    route_pattern: RoutePattern::Skewed,
                },
                512,
            )?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn main() -> candle_core::Result<()> {
    cuda_bench::main()
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("routed_lora_bench requires the cuda feature");
}
