use super::*;
use candle_core::{cuda::cudarc::driver::sys, quantized::QStorage, Var};
use std::borrow::Cow;

const ATOMIC_TOLERANCE: f32 = 1e-5;

fn weight(
    device: &Device,
    quant: GgmlDType,
    rows: usize,
    cols: usize,
    salt: usize,
) -> Result<QTensor> {
    let blocks_per_expert = rows * cols / quant.block_size();
    let mut bytes = vec![0u8; DECODE_SM121_EXPERTS * blocks_per_expert * quant.type_size()];
    for (index, block) in bytes.chunks_exact_mut(quant.type_size()).enumerate() {
        for (offset, value) in block.iter_mut().enumerate() {
            *value = (index * 17 + offset * 53 + index / blocks_per_expert + salt) as u8;
        }
        let (scale, minimum) = if matches!(quant, GgmlDType::Q4K | GgmlDType::Q5K) {
            (
                0.0001 * (1 + index % 5) as f32,
                0.00006 * (1 + index % 3) as f32,
            )
        } else {
            (
                0.002 * (1 + index % 5) as f32,
                -0.031 * (1 + index % 5) as f32,
            )
        };
        block[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        if quant != GgmlDType::Q8_0 {
            block[2..4].copy_from_slice(&f16::from_f32(minimum).to_bits().to_le_bytes());
        }
    }
    QTensor::new(
        QStorage::from_data(Cow::Owned(bytes), device, quant)?,
        (DECODE_SM121_EXPERTS, rows, cols),
    )
}

#[test]
#[ignore = "requires an SM121 CUDA device and 3 GiB of memory"]
fn sm121_moe_graph_replay_matches_legacy_with_changing_routing() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    if !super::super::fast_mmvq::is_sm121(cuda)? {
        return Ok(());
    }
    let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
    for (gate_quant, down_quant) in [
        (GgmlDType::Q4K, GgmlDType::Q4_1),
        (GgmlDType::Q4K, GgmlDType::Q5_1),
        (GgmlDType::Q4K, GgmlDType::Q8_0),
        (GgmlDType::Q5K, GgmlDType::Q8_0),
    ] {
        let gate = weight(
            &device,
            gate_quant,
            DECODE_SM121_INTERMEDIATE,
            DECODE_SM121_HIDDEN,
            17,
        )?;
        let up = weight(
            &device,
            gate_quant,
            DECODE_SM121_INTERMEDIATE,
            DECODE_SM121_HIDDEN,
            29,
        )?;
        let down = weight(
            &device,
            down_quant,
            DECODE_SM121_HIDDEN,
            DECODE_SM121_INTERMEDIATE,
            23,
        )?;
        let mut ids = cuda.alloc_zeros::<u32>(DECODE_SM121_TOPK)?;
        let mut legacy_ids = cuda.alloc_zeros::<u32>(2 * DECODE_SM121_TOPK)?;
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let input = Var::zeros((1, DECODE_SM121_HIDDEN), dtype, &device)?;
            let routing = Var::zeros((1, DECODE_SM121_TOPK), DType::F32, &device)?;
            let project = |xs: &Tensor, ids: &CudaSlice<u32>, weights: &Tensor, batch| {
                let (storage, _) = weights.storage_and_layout();
                let Storage::Cuda(storage) = &*storage else {
                    unreachable!()
                };
                let slice = storage.as_cuda_slice::<f32>()?;
                let stream = cuda.cuda_stream();
                let (pointer, _guard) = slice.device_ptr(&stream);
                unsafe {
                    indexed_moe_fused_decode(
                        &gate,
                        &up,
                        &down,
                        xs,
                        ids,
                        pointer as *const f32,
                        batch,
                        DECODE_SM121_TOPK,
                        ACT_SILU,
                        cuda,
                    )
                }
            };
            let _ = project(&input, &ids, &routing, 1)?;
            let _ = project(
                &Tensor::cat(&[input.as_tensor(), input.as_tensor()], 0)?,
                &legacy_ids,
                &Tensor::cat(&[routing.as_tensor(), routing.as_tensor()], 0)?,
                2,
            )?;
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
            let output = project(&input, &ids, &routing, 1);
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            let output = output?;
            let graph = graph
                .map_err(candle_core::Error::msg)?
                .ok_or_else(|| candle_core::Error::msg("MoE capture returned no graph"))?;
            for (case, scale) in [0.0f32, 0.05, -0.11, 0.25, 0.05].into_iter().enumerate() {
                let values = (0..DECODE_SM121_HIDDEN)
                    .map(|i| (i as f32 * 0.03 + case as f32).sin() * scale)
                    .collect();
                input.set(
                    &Tensor::from_vec(values, (1, DECODE_SM121_HIDDEN), &device)?
                        .to_dtype(dtype)?,
                )?;
                let experts = (0..DECODE_SM121_TOPK)
                    .map(|slot| ((case * 127 + slot * 37) % DECODE_SM121_EXPERTS) as u32)
                    .collect::<Vec<_>>();
                cuda.memcpy_htod(&experts, &mut ids)?;
                cuda.memcpy_htod(&experts.repeat(2), &mut legacy_ids)?;
                let weights = (0..DECODE_SM121_TOPK)
                    .map(|slot| (slot + case + 1) as f32 * scale)
                    .collect();
                routing.set(&Tensor::from_vec(weights, (1, DECODE_SM121_TOPK), &device)?)?;
                graph.launch().map_err(candle_core::Error::msg)?;
                device.synchronize()?;
                // Batch two selects the legacy kernels and overwrites the same decode workspaces.
                let reference = project(
                    &Tensor::cat(&[input.as_tensor(), input.as_tensor()], 0)?,
                    &legacy_ids,
                    &Tensor::cat(&[routing.as_tensor(), routing.as_tensor()], 0)?,
                    2,
                )?
                .narrow(0, 0, 1)?;
                let actual = output.flatten_all()?.to_vec1::<f32>()?;
                let expected = reference.flatten_all()?.to_vec1::<f32>()?;
                for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
                    assert!(
                    (actual - expected).abs()
                        <= ATOMIC_TOLERANCE * (1.0 + actual.abs().max(expected.abs())),
                    "{gate_quant:?}/{down_quant:?} {dtype:?} case={case} index={index}: {actual} != {expected}"
                );
                }
            }
        }
    }
    Ok(())
}
