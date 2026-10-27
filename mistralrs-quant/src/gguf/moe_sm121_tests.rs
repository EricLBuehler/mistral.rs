use super::*;
use candle_core::{cuda::cudarc::driver::sys, Var};

const DECODE_SM121_HIDDEN: usize = 768;
const DECODE_SM121_INTERMEDIATE: usize = 512;
const DECODE_SM121_EXPERTS: usize = 17;
const DECODE_SM121_TOPK: usize = 3;

const ATOMIC_TOLERANCE: f32 = 1e-5;

#[test]
fn staged_moe_preserves_cache_fitting_and_batched_dispatch() {
    const CACHE_BYTES: usize = 4096;
    assert!(moe_staging_bank_eligible(
        true,
        1,
        CACHE_BYTES + 1,
        CACHE_BYTES
    ));
    assert!(!moe_staging_bank_eligible(
        true,
        1,
        CACHE_BYTES,
        CACHE_BYTES
    ));
    assert!(!moe_staging_bank_eligible(
        true,
        2,
        CACHE_BYTES * 2,
        CACHE_BYTES
    ));
    assert!(!moe_staging_bank_eligible(
        false,
        1,
        CACHE_BYTES * 2,
        CACHE_BYTES
    ));
}

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn staged_moe_launches_all_formats_and_rejects_unsupported_geometry() -> Result<()> {
    const BUFFER_BYTES: usize = 64 * 1024;
    const ROWS: usize = 35;
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    if !super::super::fast_mmvq::is_sm121(cuda)? {
        return Ok(());
    }
    let stream = cuda.cuda_stream();
    let weights = cuda.alloc_zeros::<u8>(BUFFER_BYTES)?;
    let input = cuda.alloc_zeros::<u8>(BUFFER_BYTES)?;
    let indices = cuda.alloc_zeros::<u32>(1)?;
    let routing = cuda.alloc_zeros::<f32>(1)?;
    let mut output = cuda.alloc_zeros::<f32>(ROWS)?;
    let (weight_ptr, _wg) = weights.device_ptr(&stream);
    let (input_ptr, _ig) = input.device_ptr(&stream);
    let (index_ptr, _idg) = indices.device_ptr(&stream);
    let (routing_ptr, _rg) = routing.device_ptr(&stream);
    for format in [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14] {
        cuda.memcpy_htod(&vec![f32::NAN; ROWS], &mut output)?;
        {
            let (output_ptr, _og) = slice_ptr(&output, 0);
            let args = ffi::MoeDecodeArgs {
                weights: weight_ptr as *const _,
                up: weight_ptr as *const _,
                input: input_ptr as *const _,
                indices: index_ptr as *const _,
                routing: routing_ptr as *const _,
                output: output_ptr as *mut _,
                rows: ROWS as i32,
                columns: 512,
                topk: 1,
                input_stride: 512,
                activation: ACT_SILU,
            };
            assert!(unsafe {
                ffi::launch_gguf_moe_staged(&args, format, true, stream.cu_stream() as *mut _)
            });
        }
        assert!(cuda.clone_dtoh(&output)?.iter().all(|value| *value == 0.0));
    }
    let (output_ptr, _og) = slice_ptr(&output, 0);
    for (format, columns, offset) in [(11, 256, 0), (8, 512, 2), (8, 32768, 0), (14, 2560, 0)] {
        let args = ffi::MoeDecodeArgs {
            weights: (weight_ptr + offset) as *const _,
            up: (weight_ptr + offset) as *const _,
            input: input_ptr as *const _,
            indices: index_ptr as *const _,
            routing: routing_ptr as *const _,
            output: output_ptr as *mut _,
            rows: 1,
            columns,
            topk: 1,
            input_stride: columns,
            activation: ACT_SILU,
        };
        assert!(!unsafe {
            ffi::launch_gguf_moe_staged(&args, format, false, stream.cu_stream() as *mut _)
        });
    }
    Ok(())
}

fn weight(
    device: &Device,
    quant: GgmlDType,
    (rows, cols): (usize, usize),
    salt: usize,
    copies: usize,
) -> Result<QTensor> {
    let source = QTensor::quantize(
        &super::super::fast_mmq::tests::patterned((DECODE_SM121_EXPERTS, rows, cols), salt, 0.03)?,
        quant,
    )?;
    let bytes = source.data()?.repeat(copies);
    super::super::archive::qtensor_from_gguf_data(
        quant,
        &bytes,
        vec![DECODE_SM121_EXPERTS * copies, rows, cols],
        device,
    )
}

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn sm121_moe_graph_replay_matches_legacy_with_changing_routing() -> Result<()> {
    replay_moe(false)
}

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn sm121_staged_moe_large_banks_replay_matches_legacy() -> Result<()> {
    replay_moe(true)
}

fn replay_moe(large_bank: bool) -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    if !super::super::fast_mmvq::is_sm121(cuda)? {
        return Ok(());
    }
    const STAGED_HIDDEN: usize = 1024;
    let hidden = if large_bank {
        STAGED_HIDDEN
    } else {
        DECODE_SM121_HIDDEN
    };
    let l2 = moe_l2_bytes(cuda)?;
    let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
    for gate_quant in [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        let down_quant = gate_quant;
        let base_bytes = DECODE_SM121_EXPERTS * DECODE_SM121_INTERMEDIATE * hidden
            / gate_quant.block_size()
            * gate_quant.type_size();
        let copies = if large_bank { l2 / base_bytes + 1 } else { 1 };
        let experts = DECODE_SM121_EXPERTS * copies;
        let gate = weight(
            &device,
            gate_quant,
            (DECODE_SM121_INTERMEDIATE, hidden),
            17,
            copies,
        )?;
        let up = weight(
            &device,
            gate_quant,
            (DECODE_SM121_INTERMEDIATE, hidden),
            29,
            copies,
        )?;
        let down = weight(
            &device,
            down_quant,
            (hidden, DECODE_SM121_INTERMEDIATE),
            23,
            copies,
        )?;
        assert_eq!(
            moe_staging_bank_eligible(true, 1, quantized_weight_bytes(&down), l2),
            large_bank,
        );
        for tokens in [1, 2] {
            let mut ids = cuda.alloc_zeros::<u32>(tokens * DECODE_SM121_TOPK)?;
            let mut legacy_ids = cuda.alloc_zeros::<u32>(3 * tokens * DECODE_SM121_TOPK)?;
            for dtype in [DType::BF16, DType::F16, DType::F32] {
                let input = Var::zeros((tokens, hidden), dtype, &device)?;
                let routing = Var::zeros((tokens, DECODE_SM121_TOPK), DType::F32, &device)?;
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
                let _ = project(&input, &ids, &routing, tokens)?;
                let _ = project(
                    &Tensor::cat(
                        &[input.as_tensor(), input.as_tensor(), input.as_tensor()],
                        0,
                    )?,
                    &legacy_ids,
                    &Tensor::cat(
                        &[
                            routing.as_tensor(),
                            routing.as_tensor(),
                            routing.as_tensor(),
                        ],
                        0,
                    )?,
                    3 * tokens,
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
                let output = project(&input, &ids, &routing, tokens);
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
                    let values = (0..tokens * hidden)
                        .map(|i| (i as f32 * 0.03 + case as f32).sin() * scale)
                        .collect();
                    input.set(
                        &Tensor::from_vec(values, (tokens, hidden), &device)?.to_dtype(dtype)?,
                    )?;
                    let experts = (0..tokens * DECODE_SM121_TOPK)
                        .map(|slot| ((case * 127 + slot * 37) % experts) as u32)
                        .collect::<Vec<_>>();
                    cuda.memcpy_htod(&experts, &mut ids)?;
                    cuda.memcpy_htod(&experts.repeat(3), &mut legacy_ids)?;
                    let weights = (0..tokens * DECODE_SM121_TOPK)
                        .map(|slot| (slot + case + 1) as f32 * scale)
                        .collect();
                    routing.set(&Tensor::from_vec(
                        weights,
                        (tokens, DECODE_SM121_TOPK),
                        &device,
                    )?)?;
                    graph.launch().map_err(candle_core::Error::msg)?;
                    device.synchronize()?;
                    // A larger batch selects the legacy kernels and overwrites the same decode workspaces.
                    let reference = project(
                        &Tensor::cat(
                            &[input.as_tensor(), input.as_tensor(), input.as_tensor()],
                            0,
                        )?,
                        &legacy_ids,
                        &Tensor::cat(
                            &[
                                routing.as_tensor(),
                                routing.as_tensor(),
                                routing.as_tensor(),
                            ],
                            0,
                        )?,
                        3 * tokens,
                    )?
                    .narrow(0, 0, tokens)?;
                    let actual = output.flatten_all()?.to_vec1::<f32>()?;
                    let expected = reference.flatten_all()?.to_vec1::<f32>()?;
                    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate()
                    {
                        assert!(
                    (actual - expected).abs()
                        <= ATOMIC_TOLERANCE * (1.0 + actual.abs().max(expected.abs())),
                    "{gate_quant:?}/{down_quant:?} {dtype:?} tokens={tokens} case={case} index={index}: {actual} != {expected}"
                );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn q8_1_experts_treat_block_sums_as_metadata() -> Result<()> {
    const EXPERTS: usize = 3;
    const WIDTH: usize = 256;
    const TOPK: usize = 2;
    const INPUT: f32 = 0.125;
    const GATE: f32 = 0.03125;
    const UP: f32 = 0.015625;
    const DOWN: f32 = 0.0078125;
    const TOLERANCE: f32 = 2e-3;
    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?;
    let weights = [GATE, UP, DOWN]
        .into_iter()
        .map(|value| {
            QTensor::quantize_onto(
                &Tensor::full(value, (EXPERTS, WIDTH, WIDTH), &Device::Cpu)?,
                GgmlDType::Q8_1,
                &device,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let ids = cuda.clone_htod(&[0u32, 2])?;
    let routing = cuda.clone_htod(&[0.25f32, 0.75])?;
    let input = Tensor::full(INPUT, (1, WIDTH), &device)?;
    let stream = cuda.cuda_stream();
    let (routing_ptr, _routing_guard) = routing.device_ptr(&stream);
    let output = unsafe {
        indexed_moe_fused_decode(
            &weights[0],
            &weights[1],
            &weights[2],
            &input,
            &ids,
            routing_ptr as *const f32,
            1,
            TOPK,
            ACT_SILU,
            cuda,
        )?
    };
    let gate = WIDTH as f32 * INPUT * GATE;
    let up = WIDTH as f32 * INPUT * UP;
    let expected = WIDTH as f32 * DOWN * up * gate / (1.0 + (-gate).exp());
    for value in output.flatten_all()?.to_vec1::<f32>()? {
        assert!(
            (value - expected).abs() < TOLERANCE,
            "{value} != {expected}"
        );
    }
    Ok(())
}
