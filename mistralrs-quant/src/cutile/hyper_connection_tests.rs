use super::*;
use candle_core::cuda::cudarc::driver::sys;
use cutile::cutile_compiler::specialization::{compute_spec, DivHint};
use std::collections::HashSet;

const MAX_TEST_ROWS: usize = 16384;
const TEST_ROW_OFFSETS: [usize; 5] = [0, 1, 2, 4, 8];
const TEST_ROW_ALIGNMENT: usize = 16;
const TEST_UNSEEN_ROW_OFFSET: usize = 32;
const TEST_HIDDEN: usize = 2048;
const TEST_RANK: usize = 256;
const TEST_STREAMS: usize = 4;

#[test]
fn hc_prefill_warmup_covers_all_dispatch_row_keys() {
    let warmed = WARMUP_ROWS
        .into_iter()
        .map(|rows| DivHint::from_value(rows as i32))
        .collect::<HashSet<_>>();
    for rows in MIN_ROWS..=MAX_TEST_ROWS {
        assert!(
            warmed.contains(&DivHint::from_value(rows as i32)),
            "missing rows={rows}"
        );
    }
    for hidden in [512, 1024, 2048, 4096, 8192] {
        for rank in [64, 128, 320, 512, 1024] {
            for streams in [2, 3, 4, 8] {
                let spec = |rows: usize| {
                    [
                        compute_spec(16, &[rows as i32, rank], &[rank, 1], 2),
                        compute_spec(16, &[streams, hidden, rank], &[hidden * rank, rank, 1], 2),
                        compute_spec(
                            16,
                            &[rows as i32, streams, hidden],
                            &[streams * hidden, hidden, 1],
                            2,
                        ),
                        compute_spec(16, &[rows as i32, hidden], &[hidden, 1], 2),
                    ]
                };
                let warmed = WARMUP_ROWS.into_iter().map(spec).collect::<HashSet<_>>();
                for rows in MIN_ROWS..=MAX_TEST_ROWS {
                    assert!(
                        warmed.contains(&spec(rows)),
                        "missing rows={rows} hidden={hidden} rank={rank} streams={streams}"
                    );
                }
            }
        }
    }
}

#[test]
fn hc_prefill_metadata_warmup_avoids_capture_jit() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let dev = device.as_cuda_device()?;
    if !hc_prefill_supported(dev) {
        return Ok(());
    }
    let width = TEST_STREAMS * TEST_HIDDEN;
    let hardware = DeviceGeometry::read(dev).expect("supported CUDA device has cache metadata");
    let minimum = MIN_ROWS
        .max(
            hardware
                .cache_budget
                .div_ceil(width * DType::BF16.size_in_bytes()),
        )
        .max(
            (hardware.multiprocessors * MIN_BLOCKS_PER_SM).div_ceil(TEST_HIDDEN / TILE_COLUMNS)
                * TILE_ROWS,
        );
    let base = (minimum + TEST_UNSEEN_ROW_OFFSET).next_multiple_of(TEST_ROW_ALIGNMENT);
    for dtype in [DType::BF16, DType::F16] {
        let weight = Tensor::ones((width, TEST_RANK), dtype, &device)?;
        let lo = Tensor::ones((MIN_ROWS, TEST_RANK), dtype, &device)?;
        let xn = Tensor::ones((MIN_ROWS, width), dtype, &device)?;
        assert!(try_hc_prefill_up_mix(&xn, &lo, &weight, TEST_STREAMS)?.is_none());
        register_hc_up_mix(&weight, TEST_STREAMS);
        assert!(try_hc_prefill_up_mix(&xn, &lo, &weight, TEST_STREAMS)?.is_none());
        HYPER_CONNECTION.warm(dev, true)?;
        let compiled = cutile::tile_kernel::jit_compile_count();
        for rows in TEST_ROW_OFFSETS.map(|offset| base + offset) {
            let lo = Tensor::ones((1, rows, TEST_RANK), dtype, &device)?;
            let xn = Tensor::ones((1, rows, width), dtype, &device)?;
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
                return Err(candle_core::Error::msg(error));
            }
            let output = try_hc_prefill_up_mix(&xn, &lo, &weight, TEST_STREAMS);
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            let output = output?.expect("warmed geometry should select fusion");
            let graph = graph
                .map_err(candle_core::Error::msg)?
                .expect("captured graph");
            assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
            graph.launch().map_err(candle_core::Error::msg)?;
            assert_eq!(output.dims(), [1, rows, TEST_HIDDEN]);
            let error = (&output.to_dtype(DType::F32)? - 1.0)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert_eq!(error, 0.0);
        }
    }
    Ok(())
}

#[test]
fn hc_prefill_policy_tracks_cache_and_occupancy() {
    const MIB: usize = 1024 * 1024;
    let gb10 = DeviceGeometry {
        multiprocessors: 48,
        cache_budget: 8 * MIB,
    };
    let larger = DeviceGeometry {
        multiprocessors: 170,
        cache_budget: 32 * MIB,
    };
    for (rows, hidden, rank, streams, gb10_expected, larger_expected) in [
        (512usize, 4096usize, 128usize, 2usize, true, false),
        (2048, 4096, 128, 2, true, true),
        (512, 1024, 128, 8, false, false),
        (4096, 1024, 128, 8, true, true),
        (512, 2048, 1024, 2, false, false),
        (4096, 2048, 1024, 2, true, true),
        (4096, 8192, 512, 2, false, true),
        (8192, 2560, 320, 4, true, true),
    ] {
        let width = hidden * streams;
        let weight_bytes = width * rank * DType::BF16.size_in_bytes();
        let input_bytes = rows * width * DType::BF16.size_in_bytes();
        let blocks = rows.div_ceil(TILE_ROWS) * hidden.div_ceil(TILE_COLUMNS);
        assert_eq!(
            gb10.permits_workload(rows, weight_bytes, input_bytes, blocks),
            gb10_expected
        );
        assert_eq!(
            larger.permits_workload(rows, weight_bytes, input_bytes, blocks),
            larger_expected
        );
    }
}
