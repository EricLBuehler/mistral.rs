use super::*;
use crate::{mxfp8::Mxfp8Linear, QuantMethod};
use candle_core::Device;
use cutile::cutile_compiler::specialization::DivHint;

#[test]
fn warmup_covers_row_specializations_and_dispatch_boundaries() {
    let key = |rows, n, k| {
        let cfg = config(rows, n, k);
        (
            cfg.bm,
            cfg.bk,
            cfg.static_k,
            cfg.group_m,
            cfg.blocks_per_sm,
            DivHint::from_value(rows as i32).divisor,
        )
    };
    for n in [1, 128, 129, 3968, 4096, 8064, 8192, 16384] {
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

#[test]
fn static_k_is_limited_to_large_wide_prefill() {
    for (rows, n, k) in [(2047, 8192, 2048), (4096, 8064, 2048), (4096, 8192, 2016)] {
        let cfg = config(rows, n, k);
        assert!(!cfg.static_k);
        assert_eq!(cfg.bk, TILE_K);
    }
    for rows in [2048, 2049, 4095, 4096, 4097, 8192] {
        let cfg = config(rows, 8192, 2048);
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
        warm_shape(
            WarmShape {
                ordinal: dev.cuda_stream().context().ordinal(),
                n,
                k,
                dtype,
                scale_offset: (k / MXFP8_BLOCK_SIZE) % MAX_POINTER_ALIGNMENT,
            },
            dev,
            &WarmupProgress::new("MXFP8 test", WARM_ROWS.len(), true),
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
