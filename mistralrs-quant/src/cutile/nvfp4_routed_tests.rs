use super::*;
use candle_core::cuda::cudarc::driver::sys;
use float8::F8E4M3;

#[test]
fn routed_prefill_geometry_preserves_small_and_a16_paths() {
    use super::super::nvfp4_matmul::routed_geometry;
    let short = routed_geometry(512, 2560, 640, true, 12);
    assert_eq!(
        (short.rows, short.columns, short.k, short.persistent),
        (64, 128, 128, true)
    );
    let long = routed_geometry(512, 640, 2560, true, 12);
    assert_eq!(
        (long.rows, long.columns, long.k, long.persistent),
        (16, 128, 256, true)
    );
    let boundary = routed_geometry(4095, 640, 2560, true, 12);
    assert_eq!(boundary, long);
    for tokens in [4096, 4097, 8192] {
        let large = routed_geometry(tokens, 640, 2560, true, 12);
        assert_eq!(
            (large.rows, large.columns, large.k, large.persistent),
            (32, 128, 256, true)
        );
        assert_eq!(routed_geometry(tokens, 2560, 640, true, 12), short);
    }
    for cfg in [
        routed_geometry(511, 2560, 640, true, 12),
        routed_geometry(512, 2560, 640, false, 12),
        routed_geometry(512, 2560, 640, true, 10),
        routed_geometry(512, 73, 144, true, 12),
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
    let major = super::super::device_compute_major(dev);
    if !nvfp4_supported(dev) || major != 12 {
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
            let geometry =
                |tokens| super::super::nvfp4_matmul::routed_geometry(tokens, n, k, true, major);
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
