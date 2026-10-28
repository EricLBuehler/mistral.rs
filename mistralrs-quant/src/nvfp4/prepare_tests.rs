use candle_core::{cuda::CudaStorageSlice, Device, Storage, Tensor};
use float8::F8E4M3;

use super::*;
use crate::utils::slice_ptr_on_stream;

const TOKENS: usize = 3;
const TOPK: usize = 3;
const EXPERTS: usize = 4;
const CALIBRATION: [f32; EXPERTS] = [0.015625, 0.09375, 0.0078125, 0.125];
const WIDTHS: [usize; 6] = [16, 80, 128, 272, 640, 2560];
const ALIGNMENTS: [usize; 4] = [8, 16, 32, 64];
const UNTOUCHED_BYTE: u8 = 0xab;
const COPY_GUARD_BYTES: usize = 16;
const COPY_OFFSETS: [(usize, usize, usize, usize); 8] = [
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (8, 16, 8, 16),
    (16, 8, 16, 8),
    (4, 2, 4, 2),
];

struct GuardedBytes {
    storage: Tensor,
    view: Tensor,
    prefix: usize,
}

impl GuardedBytes {
    fn new(elements: usize, offset: usize, device: &Device) -> Result<Self> {
        let prefix = COPY_GUARD_BYTES + offset;
        let storage = Tensor::full(UNTOUCHED_BYTE, prefix + elements + COPY_GUARD_BYTES, device)?;
        let view = storage.narrow(0, prefix, elements)?;
        Ok(Self {
            storage,
            view,
            prefix,
        })
    }

    fn check_guards(&self) -> Result<()> {
        let values = self.storage.to_vec1::<u8>()?;
        assert!(values[..self.prefix]
            .iter()
            .all(|&value| value == UNTOUCHED_BYTE));
        assert!(values[self.prefix + self.view.elem_count()..]
            .iter()
            .all(|&value| value == UNTOUCHED_BYTE));
        Ok(())
    }
}

fn ptr(tensor: &Tensor) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        unreachable!()
    };
    let stream = storage.device.cuda_stream();
    macro_rules! pointer {
        ($slice:expr) => {
            slice_ptr_on_stream($slice, layout.start_offset(), &stream).0
        };
    }
    Ok(match &storage.slice {
        CudaStorageSlice::U8(x) => pointer!(x),
        CudaStorageSlice::U32(x) => pointer!(x),
        CudaStorageSlice::F32(x) => pointer!(x),
        CudaStorageSlice::F16(x) => pointer!(x),
        CudaStorageSlice::BF16(x) => pointer!(x),
        _ => unreachable!(),
    })
}

fn offset(tensor: Tensor) -> Result<Tensor> {
    let count = tensor.elem_count();
    let flat = tensor.flatten_all()?;
    Tensor::cat(&[&flat.narrow(0, 0, 1)?, &flat], 0)?.narrow(0, 1, count)
}

fn fp4_code(value: f32) -> u8 {
    let rounded = crate::nvfp4::quantize_fp4(value);
    let magnitude = crate::nvfp4::FP4_VALUES
        .iter()
        .position(|&candidate| candidate == rounded.abs())
        .unwrap() as u8;
    magnitude | (u8::from(rounded.is_sign_negative()) << 3)
}

#[test]
fn nvfp4_routed_prepare_matches_reference_and_preserves_unused_rows() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let stream = device.as_cuda_device()?.cuda_stream();
    let routes = TOKENS * TOPK;
    for alignment in ALIGNMENTS {
        let mut sorted = Vec::new();
        let mut experts = Vec::new();
        for expert in 0..EXPERTS {
            let ids = (0..routes).filter(|route| route % EXPERTS == expert);
            sorted.extend(ids.map(|id| id as u32));
            sorted.resize(sorted.len().div_ceil(alignment) * alignment, routes as u32);
            experts.push(expert as u32);
        }
        let count = sorted.len();
        let capacity = count + alignment - 1;
        sorted.resize(capacity, u32::MAX);
        experts.push(u32::MAX);
        let ids = offset(Tensor::from_slice(&sorted, capacity, &device)?)?;
        let eids = offset(Tensor::from_slice(&experts, experts.len(), &device)?)?;
        let num_padded = offset(Tensor::new(&[count as u32], &device)?)?;
        let global = offset(Tensor::new(&CALIBRATION, &device)?)?;
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            for k in WIDTHS {
                for shared in [false, true] {
                    let input_rows = if shared { TOKENS } else { routes };
                    let input_stride = if shared { TOPK } else { 1 };
                    let values = (0..input_rows * k)
                        .map(|i| {
                            if i % k / crate::NVFP4_BLOCK_SIZE == 1 {
                                0.0
                            } else if i % 127 == 0 {
                                32768.0
                            } else if i % 127 == 1 {
                                -1.0e-7
                            } else {
                                ((i * 17 + i / k * 7) % 113) as f32 / 29.0 - 1.75
                            }
                        })
                        .collect::<Vec<_>>();
                    let input = offset(
                        Tensor::from_vec(values, (input_rows, k), &device)?.to_dtype(dtype)?,
                    )?;
                    let source = input.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    let packed = offset(Tensor::full(UNTOUCHED_BYTE, capacity * k / 2, &device)?)?;
                    let scales = offset(Tensor::full(
                        UNTOUCHED_BYTE,
                        capacity * k / crate::NVFP4_BLOCK_SIZE,
                        &device,
                    )?)?;
                    let aligned =
                        offset(Tensor::full(1f32, capacity * k, &device)?.to_dtype(dtype)?)?;
                    let mut args = RoutedActivation {
                        input: ptr(&input)? as *const _,
                        global: ptr(&global)? as *const _,
                        sorted_ids: ptr(&ids)? as *const _,
                        expert_ids: ptr(&eids)? as *const _,
                        num_padded: ptr(&num_padded)? as *const _,
                        packed: ptr(&packed)? as *mut _,
                        scales: ptr(&scales)? as *mut _,
                        aligned: ptr(&aligned)? as *mut _,
                        rows: capacity as i32,
                        routes: routes as i32,
                        k: k as i32,
                        input_stride: input_stride as i32,
                        alignment: alignment as i32,
                        a4: 1,
                    };
                    unsafe { launch(&args, dtype, stream.cu_stream())? };
                    let actual = packed.to_vec1::<u8>()?;
                    let actual_scales = scales.to_vec1::<u8>()?;
                    let mut expected = vec![UNTOUCHED_BYTE; capacity * k / 2];
                    let mut expected_scales =
                        vec![UNTOUCHED_BYTE; capacity * k / crate::NVFP4_BLOCK_SIZE];
                    let mut expected_aligned = vec![1f32; capacity * k];
                    for row in 0..count {
                        let route = sorted[row] as usize;
                        let global = CALIBRATION[experts[row / alignment] as usize];
                        for group in 0..k / crate::NVFP4_BLOCK_SIZE {
                            let mut block = [0f32; crate::NVFP4_BLOCK_SIZE];
                            for (column, value) in block.iter_mut().enumerate() {
                                let column = group * crate::NVFP4_BLOCK_SIZE + column;
                                let x = if route < routes {
                                    source[(route / input_stride) * k + column]
                                } else {
                                    0.0
                                };
                                expected_aligned[row * k + column] = x;
                                *value = x / global;
                            }
                            let maximum = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                            let scale = F8E4M3::from_f32(
                                (maximum / crate::nvfp4::FP4_MAX).min(crate::nvfp4::FP8_MAX),
                            );
                            expected_scales[row * k / crate::NVFP4_BLOCK_SIZE + group] =
                                scale.to_bits();
                            let denominator = if scale.to_f32() == 0.0 {
                                1.0
                            } else {
                                scale.to_f32()
                            };
                            for (pair, values) in block.as_chunks::<2>().0.iter().enumerate() {
                                expected
                                    [row * k / 2 + group * crate::NVFP4_BLOCK_SIZE / 2 + pair] =
                                    fp4_code(values[0] / denominator)
                                        | (fp4_code(values[1] / denominator) << 4);
                            }
                        }
                    }
                    assert_eq!(
                        actual, expected,
                        "packed {dtype:?} k={k} shared={shared} alignment={alignment}"
                    );
                    assert_eq!(
                        actual_scales, expected_scales,
                        "scales {dtype:?} k={k} shared={shared} alignment={alignment}"
                    );
                    args.a4 = 0;
                    unsafe { launch(&args, dtype, stream.cu_stream())? };
                    assert_eq!(
                        aligned.to_dtype(DType::F32)?.to_vec1::<f32>()?,
                        expected_aligned
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn nvfp4_uniform_source_policy_requires_reused_large_inputs() {
    assert!(supports_source_quantization(512, 2560, 10));
    assert!(supports_source_quantization(4096, 2560, 4));
    assert!(!supports_source_quantization(512, 2560, 1));
    assert!(!supports_source_quantization(512, 2560, 2));
    assert!(!supports_source_quantization(512, 80, 10));
}

#[test]
fn nvfp4_uniform_source_prepare_matches_routed_with_views_and_padding() -> Result<()> {
    const GLOBAL: f32 = 0.0035807292;
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let stream = device.as_cuda_device()?.cuda_stream();
    let routes = TOKENS * TOPK;
    for alignment in ALIGNMENTS {
        let mut sorted = Vec::new();
        let mut experts = Vec::new();
        for expert in 0..EXPERTS {
            sorted.extend(
                (0..routes)
                    .filter(|route| route % EXPERTS == expert)
                    .map(|route| route as u32),
            );
            sorted.resize(sorted.len().div_ceil(alignment) * alignment, routes as u32);
            experts.push(expert as u32);
        }
        let count = sorted.len();
        let capacity = count + alignment - 1;
        sorted.resize(capacity, u32::MAX);
        experts.push(u32::MAX);
        let ids = offset(Tensor::from_slice(&sorted, capacity, &device)?)?;
        let eids = offset(Tensor::from_slice(&experts, experts.len(), &device)?)?;
        let num_padded = offset(Tensor::new(&[count as u32], &device)?)?;
        let global = offset(Tensor::full(GLOBAL, EXPERTS, &device)?)?;
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            for k in WIDTHS {
                let values = (0..TOKENS * k)
                    .map(|i| {
                        if i % k / crate::NVFP4_BLOCK_SIZE == 1 {
                            0.0
                        } else if i % 127 == 0 {
                            32768.0
                        } else if i % 127 == 1 {
                            -1.0e-7
                        } else {
                            ((i * 17 + i / k * 7) % 113) as f32 / 29.0 - 1.75
                        }
                    })
                    .collect::<Vec<_>>();
                let input =
                    offset(Tensor::from_vec(values, (TOKENS, k), &device)?.to_dtype(dtype)?)?;
                for (packed_offset, scale_offset, source_packed_offset, source_scale_offset) in
                    COPY_OFFSETS
                {
                    let packed_buffer =
                        GuardedBytes::new(capacity * k / 2, packed_offset, &device)?;
                    let scale_buffer = GuardedBytes::new(
                        capacity * k / crate::NVFP4_BLOCK_SIZE,
                        scale_offset,
                        &device,
                    )?;
                    let source_packed_buffer =
                        GuardedBytes::new(TOKENS * k / 2, source_packed_offset, &device)?;
                    let source_scale_buffer = GuardedBytes::new(
                        TOKENS * k / crate::NVFP4_BLOCK_SIZE,
                        source_scale_offset,
                        &device,
                    )?;
                    let packed = &packed_buffer.view;
                    let scales = &scale_buffer.view;
                    let source_packed = &source_packed_buffer.view;
                    let source_scales = &source_scale_buffer.view;
                    let args = RoutedActivation {
                        input: ptr(&input)? as *const _,
                        global: ptr(&global)? as *const _,
                        sorted_ids: ptr(&ids)? as *const _,
                        expert_ids: ptr(&eids)? as *const _,
                        num_padded: ptr(&num_padded)? as *const _,
                        packed: ptr(packed)? as *mut _,
                        scales: ptr(scales)? as *mut _,
                        aligned: std::ptr::null_mut(),
                        rows: capacity as i32,
                        routes: routes as i32,
                        k: k as i32,
                        input_stride: TOPK as i32,
                        alignment: alignment as i32,
                        a4: 1,
                    };
                    unsafe { launch(&args, dtype, stream.cu_stream())? };
                    let expected = packed.to_vec1::<u8>()?;
                    let expected_scales = scales.to_vec1::<u8>()?;
                    packed.slice_set(
                        &Tensor::full(UNTOUCHED_BYTE, capacity * k / 2, &device)?,
                        0,
                        0,
                    )?;
                    scales.slice_set(
                        &Tensor::full(
                            UNTOUCHED_BYTE,
                            capacity * k / crate::NVFP4_BLOCK_SIZE,
                            &device,
                        )?,
                        0,
                        0,
                    )?;
                    unsafe {
                        launch_source(
                            &args,
                            &SourceActivation {
                                packed: ptr(source_packed)? as *mut _,
                                scales: ptr(source_scales)? as *mut _,
                            },
                            dtype,
                            stream.cu_stream(),
                        )?
                    };
                    assert_eq!(
                        packed.to_vec1::<u8>()?,
                        expected,
                        "{dtype:?} k={k} alignment={alignment} offsets={packed_offset}/{scale_offset}/{source_packed_offset}/{source_scale_offset}"
                    );
                    assert_eq!(scales.to_vec1::<u8>()?, expected_scales);
                    for buffer in [
                        &packed_buffer,
                        &scale_buffer,
                        &source_packed_buffer,
                        &source_scale_buffer,
                    ] {
                        buffer.check_guards()?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn nvfp4_uniform_source_gather_matches_fallback_and_replays_graph() -> Result<()> {
    use crate::{Nvfp4ActivationMode, QuantMethod};
    use candle_core::cuda::cudarc::driver::sys;
    const ROWS: usize = 512;
    const K: usize = 2048;
    const N: usize = 17;
    const EXPERT_COUNT: usize = 5;
    const ROUTES_PER_TOKEN: usize = 4;
    const GLOBAL: f32 = 0.0035807292;
    let device = Device::new_cuda_with_stream(0)?;
    let dev = device.as_cuda_device()?;
    if !crate::cutile::nvfp4_supported(dev) {
        return Ok(());
    }
    let stream = dev.cuda_stream();
    let _cache = dev.enable_cuda_graph_htod_cache();
    for dtype in [DType::BF16, DType::F16] {
        let layer = super::super::Nvfp4Layer::from_normalized_parts(
            super::super::Nvfp4LayerParts {
                weight: Tensor::full(0x22u8, (EXPERT_COUNT, N, K / 2), &device)?,
                scales: Tensor::from_vec(
                    vec![F8E4M3::from_f32(0.5); EXPERT_COUNT * N * K / crate::NVFP4_BLOCK_SIZE],
                    (EXPERT_COUNT, N, K / crate::NVFP4_BLOCK_SIZE),
                    &device,
                )?,
                global_scales: Tensor::ones((EXPERT_COUNT, N), DType::F32, &device)?,
                input_scale: Some(Tensor::full(GLOBAL, EXPERT_COUNT, &device)?),
                activation: Nvfp4ActivationMode::DynamicBlock,
                bias: None,
                dtype,
            },
            true,
        )?;
        assert_eq!(layer.input_global_scale, Some(GLOBAL));
        let ids = Tensor::from_vec(
            (0..ROWS * ROUTES_PER_TOKEN)
                .map(|i| ((i * 3 + i / ROUTES_PER_TOKEN) % EXPERT_COUNT) as u32)
                .collect::<Vec<_>>(),
            (ROWS, ROUTES_PER_TOKEN),
            &device,
        )?;
        let input = Tensor::ones((ROWS, 1, K), dtype, &device)?;
        let reference = crate::cutile::cutile_nvfp4_gather(&input, &ids, layer.gemm_args())?;
        let actual = layer.gather_forward(&input, &ids)?;
        assert_eq!(
            actual
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            reference
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        device.synchronize()?;
        let compiled = cutile::tile_kernel::jit_compile_count();
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
        let output = layer.gather_forward(&input, &ids);
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let output = output?;
        let graph = graph
            .map_err(candle_core::Error::wrap)?
            .ok_or_else(|| candle_core::Error::msg("NVFP4 source capture produced no graph"))?;
        assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
        for value in [0.5f32, 2.0] {
            input.slice_set(
                &Tensor::full(value, (ROWS, 1, K), &device)?.to_dtype(dtype)?,
                0,
                0,
            )?;
            let reference = crate::cutile::cutile_nvfp4_gather(&input, &ids, layer.gemm_args())?;
            graph.launch().map_err(candle_core::Error::wrap)?;
            assert_eq!(
                output
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                reference
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            );
        }
    }
    Ok(())
}
