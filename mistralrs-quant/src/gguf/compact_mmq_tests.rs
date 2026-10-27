use super::*;
use candle_core::cuda::cudarc::driver::sys;
use candle_core::quantized::QStorage;
use std::borrow::Cow;

const ASSIGNMENTS: usize = COMPACT_TOKENS * COMPACT_TOPK;
const ROUTING_STRIDE: usize = 37;
const BALANCED: usize = 0;
const SINGLE_HOT: usize = 1;
const CONCENTRATED: usize = 2;
const MAXIMUM_TILES: usize = 3;

fn weights(device: &Device, dtype: GgmlDType, rows: usize, cols: usize) -> Result<QTensor> {
    let blocks_per_expert = rows * cols / dtype.block_size();
    let mut bytes = vec![0x5a; COMPACT_EXPERTS * blocks_per_expert * dtype.type_size()];
    let scale = half::f16::from_f32(0.002).to_bits().to_le_bytes();
    let minimum = half::f16::from_f32(0.001).to_bits().to_le_bytes();
    for (index, block) in bytes.chunks_exact_mut(dtype.type_size()).enumerate() {
        block[..2].copy_from_slice(&scale);
        block[2..4].copy_from_slice(&minimum);
        block[4] = (index / blocks_per_expert + index) as u8;
    }
    QTensor::new(
        QStorage::from_data(Cow::Owned(bytes), device, dtype)?,
        (COMPACT_EXPERTS, rows, cols),
    )
}

fn routing(mode: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut experts = vec![Vec::new(); COMPACT_EXPERTS];
    if mode == MAXIMUM_TILES {
        let larger_experts = (ASSIGNMENTS - COMPACT_EXPERTS) / COMPACT_COLUMNS;
        let mut assignment = 0;
        for (expert, routes) in experts.iter_mut().enumerate() {
            let count = 1 + usize::from(expert < larger_experts) * COMPACT_COLUMNS;
            for _ in 0..count {
                let token = assignment % COMPACT_TOKENS;
                let slot = assignment / COMPACT_TOKENS;
                routes.push((token as u32, (token * COMPACT_TOPK + slot) as u32));
                assignment += 1;
            }
        }
    } else {
        for token in 0..COMPACT_TOKENS {
            for slot in 0..COMPACT_TOPK {
                let expert = match mode {
                    BALANCED => (token + slot * ROUTING_STRIDE) % COMPACT_EXPERTS,
                    SINGLE_HOT if slot == 0 => 0,
                    SINGLE_HOT => 1 + (token + slot * ROUTING_STRIDE) % (COMPACT_EXPERTS - 1),
                    _ => slot,
                };
                experts[expert].push((token as u32, (token * COMPACT_TOPK + slot) as u32));
            }
        }
    }
    let tiles: usize = experts
        .iter()
        .map(|expert| expert.len().div_ceil(COMPACT_COLUMNS))
        .sum();
    assert!(tiles <= ASSIGNMENTS / COMPACT_COLUMNS + COMPACT_EXPERTS);
    if mode == MAXIMUM_TILES {
        assert_eq!(
            tiles,
            (ASSIGNMENTS - COMPACT_EXPERTS) / COMPACT_COLUMNS + COMPACT_EXPERTS
        );
    }
    let mut source = Vec::with_capacity(ASSIGNMENTS);
    let mut destination = Vec::with_capacity(ASSIGNMENTS);
    let mut bounds = Vec::with_capacity(COMPACT_EXPERTS + 1);
    for expert in experts {
        bounds.push(source.len() as u32);
        for (src, dst) in expert {
            source.push(src);
            destination.push(dst);
        }
    }
    bounds.push(ASSIGNMENTS as u32);
    (source, destination, bounds)
}

#[test]
#[ignore = "requires an SM121 CUDA device and 2 GiB of memory"]
fn compact_mmq_graph_replay_rebuilds_routing_and_reuses_workspace() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    if get_device_info(cuda).cc != COMPACT_CAPABILITY {
        return Ok(());
    }
    let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
    for dtype in [
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q4_1,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
    ] {
        let paired = matches!(dtype, GgmlDType::Q4K | GgmlDType::Q5K);
        let (rows, cols) = if paired {
            (COMPACT_INTERMEDIATE, COMPACT_HIDDEN)
        } else {
            (COMPACT_HIDDEN, COMPACT_INTERMEDIATE)
        };
        let weight = weights(&device, dtype, rows, cols)?;
        let xs = if paired {
            tests::patterned((COMPACT_TOKENS, cols), 3, 0.2)?.to_dtype(DType::BF16)?
        } else {
            tests::patterned((ASSIGNMENTS, 2 * cols), 3, 0.2)?
        }
        .to_device(&device)?;
        let mut source = unsafe { cuda.alloc::<u32>(ASSIGNMENTS)? };
        let mut destination = unsafe { cuda.alloc::<u32>(ASSIGNMENTS)? };
        let mut bounds = unsafe { cuda.alloc::<u32>(COMPACT_EXPERTS + 1)? };
        let initial = routing(BALANCED);
        cuda.memcpy_htod(&initial.0, &mut source)?;
        cuda.memcpy_htod(&initial.1, &mut destination)?;
        cuda.memcpy_htod(&initial.2, &mut bounds)?;
        let project = |source, destination, bounds| {
            if paired {
                grouped_pair_packed(
                    &weight,
                    &weight,
                    &xs,
                    source,
                    destination,
                    bounds,
                    ASSIGNMENTS,
                    COMPACT_TOPK,
                    COMPACT_EXPERTS,
                    cuda,
                )
            } else {
                grouped_from_glu_packed(
                    &weight,
                    &xs,
                    destination,
                    destination,
                    bounds,
                    ASSIGNMENTS,
                    COMPACT_TOKENS,
                    COMPACT_EXPERTS,
                    GluActivationType::Silu as i32,
                    cuda,
                )
            }
        };
        let _ = project(&source, &destination, &bounds)?;
        device.synchronize()?;
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
            return Err(candle_core::Error::msg(error));
        }
        let output = project(&source, &destination, &bounds);
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let output = output?;
        let graph = graph
            .map_err(candle_core::Error::msg)?
            .ok_or_else(|| candle_core::Error::msg("grouped MMQ capture returned no graph"))?;
        for mode in [BALANCED, SINGLE_HOT, CONCENTRATED, MAXIMUM_TILES, BALANCED] {
            let routes = routing(mode);
            cuda.memcpy_htod(&routes.0, &mut source)?;
            cuda.memcpy_htod(&routes.1, &mut destination)?;
            cuda.memcpy_htod(&routes.2, &mut bounds)?;
            graph.launch().map_err(candle_core::Error::msg)?;
            device.synchronize()?;
            // A larger safe column bound selects the original kernel without changing the routes.
            let reference = if paired {
                grouped(
                    &weight,
                    &xs,
                    &source,
                    &destination,
                    &bounds,
                    ASSIGNMENTS,
                    COMPACT_TOKENS + 1,
                    COMPACT_EXPERTS,
                    cuda,
                )?
            } else {
                grouped_from_glu_packed(
                    &weight,
                    &xs,
                    &destination,
                    &destination,
                    &bounds,
                    ASSIGNMENTS,
                    COMPACT_TOKENS + 1,
                    COMPACT_EXPERTS,
                    GluActivationType::Silu as i32,
                    cuda,
                )?
            };
            if paired {
                tests::assert_close(&output.narrow(1, 0, rows)?, &reference)?;
                tests::assert_close(&output.narrow(1, rows, rows)?, &reference)?;
            } else {
                tests::assert_close(&output, &reference)?;
            }
        }
    }
    Ok(())
}
