use super::*;
use crate::{mxfp8::Mxfp8Linear, QuantMethod};
use candle_core::Device;
use cutile::cutile_compiler::specialization::DivHint;

const SMALL_BATCH_TOLERANCE: f32 = 0.02;
const BENCH_WARMUP: usize = 5;
const BENCH_TARGET_CALLS: usize = 100;
const BENCH_MIN_GRAPH_REPLAYS: usize = 2;
const BENCH_CACHE_MULTIPLIER: usize = 2;

fn small_batch_operands(
    rows: usize,
    n: usize,
    k: usize,
    dtype: DType,
    device: &Device,
) -> Result<(QuantizedActivation, Tensor, Tensor)> {
    let weight = Tensor::from_vec(
        (0..n * k)
            .map(|i| F8E4M3::from_f32(((i * 13 % 47) as f32 - 23.0) * 0.125))
            .collect::<Vec<_>>(),
        (n, k),
        device,
    )?;
    let scales = Tensor::from_vec(
        (0..n * k / MXFP8_BLOCK_SIZE)
            .map(|i| 123 + (i % 7) as u8)
            .collect::<Vec<_>>(),
        (n, k / MXFP8_BLOCK_SIZE),
        device,
    )?;
    let input = Tensor::from_vec(
        (0..rows * k)
            .map(|i| ((i * 19 % 97) as f32 - 48.0) * 0.03125)
            .collect::<Vec<_>>(),
        (rows, k),
        device,
    )?
    .to_dtype(dtype)?;
    let layer = Mxfp8Linear::from_parts(weight.clone(), scales.clone(), None, dtype)?;
    Ok((layer.quantize_activation(&input)?, weight, scales))
}

#[test]
fn split_k_matches_unsplit_for_partial_tiles_and_dtypes() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !super::super::block_scaled_mma_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let multiprocessors = device_multiprocessor_count(device.as_cuda_device()?);
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        for (rows, n, k) in [(2, 96, 640), (3, 129, 1056), (8, 257, 2560), (17, 65, 160)] {
            let (input, weight, scales) = small_batch_operands(rows, n, k, dtype, &device)?;
            let expected = forward(&input, &weight, &scales)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (bn, splits) in [(64, 2), (128, 4)] {
                let cfg = Config {
                    bn,
                    splits,
                    ..config(rows, n, k, multiprocessors)
                };
                let actual = forward_config(&input, &weight, &scales, Some(cfg))?;
                assert_eq!(actual.dtype(), dtype);
                assert_eq!(actual.dims(), [rows, n]);
                let actual = actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (actual - expected).abs() <= SMALL_BATCH_TOLERANCE * expected.abs().max(1.0),
                        "{dtype:?} rows={rows} n={n} k={k} bn={bn} splits={splits} index={index}: {actual} != {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn split_k_graph_replay_reads_new_inputs_with_temporary_workspace() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;
    let device = Device::new_cuda_with_stream(0)?;
    if !super::super::block_scaled_mma_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let multiprocessors = device_multiprocessor_count(device.as_cuda_device()?);
    let (rows, n, k) = (17, 129, 1056);
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let weight = Tensor::from_vec(vec![F8E4M3::from_f32(1.0); n * k], (n, k), &device)?;
        let scales = Tensor::from_vec(
            vec![127u8; n * k / MXFP8_BLOCK_SIZE],
            (n, k / MXFP8_BLOCK_SIZE),
            &device,
        )?;
        let layer = Mxfp8Linear::from_parts(weight.clone(), scales.clone(), None, dtype)?;
        let input = Tensor::ones((rows, k), dtype, &device)?;
        let cfg = Config {
            bm: MIN_TILE_M,
            bn: 64,
            splits: 4,
            ..config(rows, n, k, multiprocessors)
        };
        let launch = || {
            forward_config(
                &layer.quantize_activation(&input)?,
                &weight,
                &scales,
                Some(cfg),
            )
        };
        let _ = launch()?;
        device.synchronize()?;
        let compiled = cutile::tile_kernel::jit_compile_count();
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
            return Err(candle_core::Error::wrap(error));
        }
        let output = launch();
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let output = output?;
        let graph = graph.map_err(candle_core::Error::wrap)?.unwrap();
        assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
        for multiplier in [1.0f32, -0.5, 2.0, 0.25] {
            let values = (0..rows * k)
                .map(|i| [1.0, -2.0, 0.5][(i / k) % 3] * multiplier)
                .collect::<Vec<_>>();
            input.slice_set(
                &Tensor::from_vec(values, (rows, k), &device)?.to_dtype(dtype)?,
                0,
                0,
            )?;
            let scratch = Tensor::full(
                -1.0f32,
                (rows.div_ceil(cfg.bm) * cfg.bm * cfg.splits, n),
                &device,
            )?;
            graph.launch().map_err(candle_core::Error::wrap)?;
            let actual = output.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            for (row, values) in actual.iter().enumerate() {
                let expected = [1.0, -2.0, 0.5][row % 3] * multiplier * k as f32;
                assert!(
                    values.iter().all(|&value| value == expected),
                    "{dtype:?} row={row}"
                );
            }
            std::hint::black_box(&scratch);
        }
    }
    Ok(())
}

fn graph_time<T>(device: &Device, calls: usize, launch: impl Fn() -> Result<T>) -> Result<f64> {
    use candle_core::cuda::cudarc::driver::sys;
    let _ = launch()?;
    device.synchronize()?;
    let stream = device.as_cuda_device()?.cuda_stream();
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
    let output = launch();
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let output = output?;
    let graph = graph.map_err(candle_core::Error::wrap)?.unwrap();
    for _ in 0..BENCH_WARMUP.div_ceil(calls) {
        graph.launch().map_err(candle_core::Error::wrap)?;
    }
    device.synchronize()?;
    let start = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(candle_core::Error::wrap)?;
    let iterations = BENCH_TARGET_CALLS
        .div_ceil(calls)
        .max(BENCH_MIN_GRAPH_REPLAYS);
    for _ in 0..iterations {
        graph.launch().map_err(candle_core::Error::wrap)?;
    }
    let end = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(candle_core::Error::wrap)?;
    end.synchronize().map_err(candle_core::Error::wrap)?;
    std::hint::black_box((&output, &graph));
    Ok(
        f64::from(start.elapsed_ms(&end).map_err(candle_core::Error::wrap)?) * 1_000.0
            / (iterations * calls) as f64,
    )
}

#[test]
#[ignore]
fn benchmark_small_batch_mxfp8_geometry() -> Result<()> {
    benchmark_shapes(
        &[
            (96, 2560),
            (640, 2560),
            (1280, 2560),
            (2560, 640),
            (2560, 6144),
            (13312, 2560),
            (16384, 2560),
        ],
        &[2, 4, 8],
        &[
            (64, 1),
            (64, 2),
            (64, 4),
            (64, 8),
            (128, 1),
            (128, 2),
            (128, 4),
            (128, 8),
        ],
    )
}

#[test]
#[ignore]
fn benchmark_underfilled_mxfp8_boundaries() -> Result<()> {
    benchmark_shapes(
        &[
            (32, 2560),
            (64, 2560),
            (128, 2560),
            (192, 2560),
            (256, 2560),
            (512, 2560),
            (1024, 2560),
            (1536, 2560),
            (2048, 2560),
            (96, 512),
            (96, 1024),
            (96, 2048),
            (96, 4096),
            (96, 6144),
            (640, 1024),
            (640, 4096),
            (1280, 1024),
            (1280, 4096),
            (1536, 640),
            (1536, 6144),
        ],
        &[8],
        &[(128, 1), (64, 1), (64, 4)],
    )
}

fn benchmark_shapes(
    shapes: &[(usize, usize)],
    row_counts: &[usize],
    configurations: &[(usize, usize)],
) -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;
    let device = Device::new_cuda_with_stream(0)?;
    let multiprocessors = device_multiprocessor_count(device.as_cuda_device()?);
    let cache_bytes = device
        .as_cuda_device()?
        .cuda_stream()
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .map_err(candle_core::Error::wrap)? as usize;
    println!("cache,rows,n,k,bn,splits,weight_sets,working_set_bytes,time_us");
    for &(n, k) in shapes {
        for &rows in row_counts {
            let (input, weight, scales) = small_batch_operands(rows, n, k, DType::BF16, &device)?;
            let weight_bytes = n * k + n * k / MXFP8_BLOCK_SIZE;
            let count = (cache_bytes * BENCH_CACHE_MULTIPLIER).div_ceil(weight_bytes);
            let weights = (0..count)
                .map(|_| Ok((weight.copy()?, scales.copy()?)))
                .collect::<Result<Vec<_>>>()?;
            for &(bn, splits) in configurations {
                let cfg = Config {
                    bn,
                    splits,
                    ..config(rows, n, k, multiprocessors)
                };
                let us = graph_time(&device, 1, || {
                    forward_config(&input, &weight, &scales, Some(cfg))
                })?;
                println!("warm,{rows},{n},{k},{bn},{splits},1,{weight_bytes},{us:.3}");
                let us = graph_time(&device, count, || {
                    weights
                        .iter()
                        .map(|(weight, scales)| forward_config(&input, weight, scales, Some(cfg)))
                        .collect::<Result<Vec<_>>>()
                })?;
                let working_set = count * weight_bytes;
                println!("rotating,{rows},{n},{k},{bn},{splits},{count},{working_set},{us:.3}");
            }
        }
    }
    Ok(())
}

#[test]
fn warmup_covers_row_specializations_and_dispatch_boundaries() {
    for multiprocessors in [24, 48, 96, 132] {
        let key = |rows, n, k| {
            let cfg = config(rows, n, k, multiprocessors);
            (
                cfg.bm,
                cfg.bn,
                cfg.bk,
                cfg.splits,
                cfg.static_k,
                cfg.group_m,
                cfg.blocks_per_sm,
                DivHint::from_value(rows as i32).divisor,
            )
        };
        for n in [
            1, 128, 129, 256, 257, 768, 769, 1536, 1537, 3968, 4096, 8064, 8192, 16384,
        ] {
            for k in [160, 1024, 2016, 2048, 2560, 6144] {
                let warmed = WARM_ROWS
                    .into_iter()
                    .map(|rows| key(rows, n, k))
                    .collect::<HashSet<_>>();
                for rows in 2..=8192 {
                    assert!(
                        warmed.contains(&key(rows, n, k)),
                        "missing rows={rows} n={n} k={k}"
                    );
                }
            }
        }
    }
}

#[test]
fn small_batches_only_split_underfilled_grids_with_sufficient_reduction_work() {
    let split_min_k = TILE_K * SMALL_BATCH_SPLITS * SMALL_BATCH_MIN_K_TILES_PER_SPLIT;
    for multiprocessors in [24, 48, 96, 132] {
        let narrow_max_n = multiprocessors / SMALL_BATCH_NARROW_SM_DIVISOR * TILE_N;
        let split_max_n = multiprocessors / SMALL_BATCH_SPLIT_SM_DIVISOR / SMALL_BATCH_SPLITS
            * SMALL_BATCH_TILE_N;
        for rows in [2, SMALL_BATCH_MAX_ROWS] {
            assert_eq!(
                config(rows, narrow_max_n, split_min_k, multiprocessors).bn,
                SMALL_BATCH_TILE_N
            );
            assert_eq!(
                config(rows, narrow_max_n + 1, split_min_k, multiprocessors).bn,
                TILE_N
            );
            assert_eq!(
                config(rows, split_max_n, split_min_k, multiprocessors).splits,
                SMALL_BATCH_SPLITS
            );
            assert_eq!(
                config(rows, split_max_n + 1, split_min_k, multiprocessors).splits,
                1
            );
            assert_eq!(
                config(
                    rows,
                    split_max_n,
                    split_min_k - MXFP8_BLOCK_SIZE,
                    multiprocessors
                )
                .splits,
                1
            );
        }
        for rows in [1, SMALL_BATCH_MAX_ROWS + 1, 64, 2048] {
            let cfg = config(rows, SMALL_BATCH_TILE_N, split_min_k, multiprocessors);
            assert_eq!((cfg.bn, cfg.splits), (TILE_N, 1));
        }
    }
}

#[test]
fn static_k_is_limited_to_large_wide_prefill() {
    for (rows, n, k) in [(2047, 8192, 2048), (4096, 8064, 2048), (4096, 8192, 2016)] {
        let cfg = config(rows, n, k, 48);
        assert!(!cfg.static_k);
        assert_eq!(cfg.bk, TILE_K);
    }
    for rows in [2048, 2049, 4095, 4096, 4097, 8192] {
        let cfg = config(rows, 8192, 2048, 48);
        assert!(cfg.static_k);
        assert_eq!(
            (cfg.bm, cfg.bk, cfg.group_m, cfg.blocks_per_sm),
            (128, 64, 16, 1)
        );
    }
}

#[test]
fn warmed_offset_scale_views_capture_without_jit() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;
    let device = Device::new_cuda_with_stream(0)?;
    let dev = device.as_cuda_device()?;
    if !super::super::block_scaled_mma_supported(dev) {
        return Ok(());
    }
    let cases = [
        (
            129,
            2080,
            DType::BF16,
            vec![
                (2, DType::BF16),
                (3, DType::BF16),
                (4, DType::BF16),
                (8, DType::BF16),
            ],
        ),
        (
            129,
            160,
            DType::BF16,
            vec![
                (63, DType::BF16),
                (66, DType::BF16),
                (68, DType::BF16),
                (72, DType::BF16),
                (80, DType::BF16),
                (63, DType::F16),
                (63, DType::F32),
            ],
        ),
        (
            8193,
            2080,
            DType::BF16,
            vec![
                (1985, DType::BF16),
                (2048, DType::BF16),
                (2049, DType::BF16),
            ],
        ),
        (
            8193,
            2080,
            DType::F16,
            vec![(1985, DType::F16), (2048, DType::F16), (2049, DType::F16)],
        ),
        (
            8193,
            2080,
            DType::F32,
            vec![(1985, DType::F32), (2048, DType::F32), (2049, DType::F32)],
        ),
    ];
    for (n, k, dtype, row_cases) in cases {
        let layer = Mxfp8Linear::from_parts(
            Tensor::from_vec(
                vec![F8E4M3::from_f32(1.0); (n + 1) * k],
                (n + 1, k),
                &device,
            )?
            .narrow(0, 1, n)?,
            Tensor::from_vec(
                vec![127u8; (n + 1) * k / MXFP8_BLOCK_SIZE],
                (n + 1, k / MXFP8_BLOCK_SIZE),
                &device,
            )?
            .narrow(0, 1, n)?,
            None,
            dtype,
        )?;
        let shape = WarmShape {
            ordinal: dev.cuda_stream().context().ordinal(),
            n,
            k,
            dtype,
            scale_offset: (k / MXFP8_BLOCK_SIZE) % MAX_POINTER_ALIGNMENT,
        };
        let rows = warm_rows(shape.key(), device_multiprocessor_count(dev));
        warm_shape(
            shape,
            &rows,
            dev,
            &WarmupProgress::new("MXFP8 test", rows.len(), true),
        )?;
        let compiled = cutile::tile_kernel::jit_compile_count();
        let stream = dev.cuda_stream();
        for (rows, input_dtype) in row_cases {
            let input = Tensor::ones((rows, k), input_dtype, &device)?;
            if input_dtype == dtype {
                let _ = layer.quantize_activation(&input)?;
            } else {
                assert!(layer.activation_quantization_scheme_for(&input).is_none());
                assert!(layer.quantize_activation(&input).is_err());
                let _ = layer.forward(&input)?;
            }
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
            let output = layer.forward(&input);
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            let output = output?;
            assert_eq!(output.dtype(), input_dtype);
            let graph = graph
                .map_err(candle_core::Error::wrap)?
                .ok_or_else(|| candle_core::Error::msg("MXFP8 capture produced no graph"))?;
            assert_eq!(
                cutile::tile_kernel::jit_compile_count(),
                compiled,
                "warmup missed rows={rows}"
            );
            for value in [1f32, 2.0] {
                input.slice_set(
                    &Tensor::full(value, (rows, k), &device)?.to_dtype(input_dtype)?,
                    0,
                    0,
                )?;
                graph.launch().map_err(candle_core::Error::wrap)?;
                device.synchronize()?;
                assert!(output
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|&output| output == k as f32 * value));
            }
        }
    }
    Ok(())
}

#[test]
fn tune_buckets_never_mix_policies_and_include_decode_rows() {
    for multiprocessors in [24, 48, 132] {
        for (n, k) in [(128, 2048), (4096, 4096), (12288, 4096), (8192, 2048)] {
            let key = TuneKey {
                ordinal: 0,
                n,
                k,
                dtype: DType::BF16,
            };
            let buckets = tune_buckets(key, multiprocessors);
            assert!(buckets[0].probe <= 16);
            let mut lower = TUNE_MIN_ROWS + 1;
            for bucket in &buckets {
                assert!(bucket.probe >= lower && bucket.probe <= bucket.upper);
                let policy = config(bucket.probe, n, k, multiprocessors);
                for rows in lower..=bucket.upper.min(TUNE_SCAN_ROWS * 2) {
                    assert_eq!(config(rows, n, k, multiprocessors), policy, "rows={rows}");
                }
                lower = bucket.upper.saturating_add(1);
            }
            let policy = config(buckets[0].probe, n, k, multiprocessors);
            let configs = tune_space(buckets[0], key, multiprocessors, SMALL_SMEM).configs();
            assert_eq!(Config::from_config(&configs[0]), Some(policy));
            assert!(configs
                .iter()
                .filter_map(Config::from_config)
                .all(|cfg| cfg == policy || fits_smem(cfg, SMALL_SMEM)));
        }
    }
}

const SMALL_SMEM: usize = 99 << 10;
const OVERSIZED_WEIGHT_DIM: usize = 1 << 20;

#[test]
fn cached_and_disabled_tuning_do_not_materialize_weights() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dev = device.as_cuda_device()?;
    if !super::super::block_scaled_mma_supported(dev) {
        return Ok(());
    }
    let key = TuneKey {
        ordinal: dev.cuda_stream().context().ordinal(),
        n: OVERSIZED_WEIGHT_DIM,
        k: OVERSIZED_WEIGHT_DIM,
        dtype: DType::BF16,
    };
    for mode in [TuneMode::Off, TuneMode::Cached] {
        let configs = tune_shape(key, dev, mode)?;
        assert!(!configs.is_empty());
    }
    Ok(())
}

#[test]
fn launch_config_consults_the_table_for_decode_and_prefill_rows() {
    let key = TuneKey {
        ordinal: 0,
        n: 4224,
        k: 4096,
        dtype: DType::F16,
    };
    let forced = Config {
        bm: 128,
        group_m: 4,
        ..config(512, key.n, key.k, 48)
    };
    TUNED.set(
        key,
        &[Tuned::measured(
            Bucket {
                upper: usize::MAX,
                probe: 512,
            },
            forced.to_config(),
        )],
        Config::from_config,
    );
    assert_eq!(launch_config(key, 512, 48), forced);
    assert_eq!(
        launch_config(key, TUNE_MIN_ROWS, 48),
        config(TUNE_MIN_ROWS, key.n, key.k, 48)
    );
    assert_eq!(launch_config(key, 8, 48), forced);
    let rows = warm_rows(key, 48);
    assert!(rows.contains(&(TUNE_MIN_ROWS + 1)) && rows.contains(&2049));
}

#[test]
#[ignore = "GPU MXFP8 tuning sweep; run with --ignored --nocapture --test-threads=1"]
fn mxfp8_tuned_vs_policy_sweep() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dev = device.as_cuda_device()?;
    if !super::super::block_scaled_mma_supported(dev) {
        return Ok(());
    }
    for dtype in [DType::BF16, DType::F16] {
        for (n, k) in [(12288, 4096), (4096, 12288)] {
            let key = TuneKey {
                ordinal: dev.cuda_stream().context().ordinal(),
                n,
                k,
                dtype,
            };
            super::super::tune::report(
                &format!("mxfp8 {dtype:?} n={n} k={k}"),
                &tune_shape(key, dev, TuneMode::Force)?,
            );
        }
    }
    Ok(())
}

fn dequantized(activation: &QuantizedActivation, dtype: DType) -> Result<Tensor> {
    Mxfp8Linear::from_parts(
        activation.quantized().clone(),
        activation.scales().clone(),
        None,
        dtype,
    )?
    .dequantize_w()?
    .to_dtype(DType::F32)
}

#[test]
#[ignore = "requires a CUDA device with cuTile support"]
fn tunable_configs_match_the_reference_at_odd_rows() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dev = device.as_cuda_device()?;
    if !super::super::block_scaled_mma_supported(dev) {
        return Ok(());
    }
    let (n, k) = (320, 1056);
    for dtype in [DType::BF16, DType::F16] {
        let weight = random_quantized(n, k, dtype, dev)?;
        let reference_weight = dequantized(&weight, dtype)?;
        for rows in [65, 129, 333] {
            let activation = random_quantized(rows, k, dtype, dev)?;
            let reference = dequantized(&activation, dtype)?.matmul(&reference_weight.t()?)?;
            let policy = config(rows, n, k, device_multiprocessor_count(dev));
            for cfg in [
                policy,
                Config {
                    bm: 128,
                    bn: 64,
                    bk: 64,
                    static_k: true,
                    group_m: 4,
                    latency: 2,
                    ..policy
                },
                Config {
                    bm: 32,
                    bk: 256,
                    blocks_per_sm: 3,
                    latency: 4,
                    ..policy
                },
            ] {
                let actual =
                    forward_config(&activation, weight.quantized(), weight.scales(), Some(cfg))?
                        .to_dtype(DType::F32)?;
                let error = (actual - &reference)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                let scale = reference.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(
                    error <= 1e-2 * scale,
                    "{dtype:?} rows={rows} {cfg:?}: {error} vs {scale}"
                );
            }
        }
    }
    Ok(())
}
