use candle_core::{cuda::cudarc::driver::sys, Device, IndexOp};

use super::*;
use crate::{nvfp4::Nvfp4LayerParts, Nvfp4ActivationMode, Nvfp4Layer, QuantMethod};

const EXPERTS: usize = 3;
const ROUTES: usize = 3;
const SHAPES: [(usize, usize); 5] = [(17, 16), (83, 80), (259, 144), (1027, 640), (2560, 640)];
const INPUT_GLOBAL: [f32; EXPERTS] = [0.015625, 0.09375, 0.0078125];
const WEIGHT_GLOBAL: [f32; EXPERTS] = [0.03125, 0.09375, 0.15625];
const IDS: [u32; ROUTES] = [2, 0, 2];
const NEXT_IDS: [u32; ROUTES] = [1, 2, 0];
const BF16_TOLERANCE: f32 = 0.012;
const F16_TOLERANCE: f32 = 0.002;

fn view(tensor: Tensor) -> Result<Tensor> {
    let count = tensor.elem_count();
    let shape = tensor.shape().clone();
    let padding = tensor.flatten_all()?.narrow(0, 0, 1)?;
    let tensor = Tensor::cat(&[&padding, &tensor.flatten_all()?], 0)?;
    tensor.narrow(0, 1, count)?.reshape(shape)
}

fn layer(n: usize, k: usize, routed: bool, dtype: DType, device: &Device) -> Result<Nvfp4Layer> {
    let experts = if routed { EXPERTS } else { 1 };
    let mut shape = vec![experts, n, k / 2];
    if !routed {
        shape.remove(0);
    }
    let weight = (0..experts * n * k / 2)
        .map(|i| ((i * 37 + i / (k / 2) * 13) % 256) as u8)
        .collect::<Vec<_>>();
    let weight = view(Tensor::from_vec(weight, shape.clone(), device)?)?;
    *shape.last_mut().unwrap() = k / NVFP4_BLOCK_SIZE;
    let scales = (0..experts * n * k / NVFP4_BLOCK_SIZE)
        .map(|i| F8E4M3::from_f32([0.3125, 0.75, 1.5, 2.75, 5.5][i % 5]))
        .collect::<Vec<_>>();
    let scales = view(Tensor::from_vec(scales, shape.clone(), device)?)?;
    shape.pop();
    let global = (0..experts * n)
        .map(|i| WEIGHT_GLOBAL[i % WEIGHT_GLOBAL.len()])
        .collect::<Vec<_>>();
    let global_scales = view(Tensor::from_vec(global, shape.clone(), device)?)?;
    let bias = (0..experts * n)
        .map(|i| (i % 11) as f32 / 64.0)
        .collect::<Vec<_>>();
    let bias = view(Tensor::from_vec(bias, shape, device)?.to_dtype(dtype)?)?;
    let input_scale = if routed {
        Tensor::from_slice(&INPUT_GLOBAL, EXPERTS, device)?
    } else {
        Tensor::new(INPUT_GLOBAL[0], device)?
    };
    Nvfp4Layer::from_parts(Nvfp4LayerParts {
        weight,
        scales,
        global_scales,
        input_scale: Some(view(input_scale)?),
        activation: Nvfp4ActivationMode::DynamicBlock,
        bias: Some(bias),
        dtype,
    })
}

fn input(k: usize, rows: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let data = (0..rows * k)
        .map(|i| {
            if i % k / NVFP4_BLOCK_SIZE == 1 {
                0.0
            } else {
                ((i * 17 + i / k * 7) % 113) as f32 / 29.0 - 1.75
            }
        })
        .collect::<Vec<_>>();
    view(Tensor::from_vec(data, (1, rows, k), device)?.to_dtype(dtype)?)
}

fn compare(actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let tolerance = match actual.dtype() {
        DType::BF16 => BF16_TOLERANCE,
        DType::F16 => F16_TOLERANCE,
        _ => unreachable!(),
    };
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let maximum = expected.iter().map(|x| x.abs()).fold(1.0f32, f32::max);
    for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * maximum,
            "element {i}: {actual} vs {expected}"
        );
    }
    Ok(())
}

#[test]
fn nvfp4_decode_dense_and_routed_match_cpu_with_views() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    for dtype in [DType::BF16, DType::F16] {
        for (n, k) in SHAPES {
            for (routed, per_route) in [(false, false), (true, false), (true, true)] {
                let gpu = layer(n, k, routed, dtype, &device)?;
                let cpu = layer(n, k, routed, dtype, &Device::Cpu)?;
                let x = input(k, if per_route { ROUTES } else { 1 }, dtype, &device)?;
                let ids = view(Tensor::new(&[IDS], &device)?)?;
                assert!(supports(
                    1,
                    if routed { ROUTES } else { 1 },
                    dtype,
                    gpu.gemm_args()
                ));
                let (actual, expected) = if routed {
                    (
                        gpu.gather_forward_raw(&x, &ids)?,
                        cpu.gather_forward_raw(
                            &x.to_device(&Device::Cpu)?,
                            &ids.to_device(&Device::Cpu)?,
                        )?,
                    )
                } else {
                    let x = x.i(0)?;
                    (
                        gpu.forward_raw(&x)?,
                        cpu.forward_raw(&x.to_device(&Device::Cpu)?)?,
                    )
                };
                compare(&actual, &expected)?;
            }
        }
    }
    Ok(())
}

#[test]
fn nvfp4_decode_graph_replay_tracks_inputs_and_routes() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let _htod_cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
    let (n, k) = SHAPES[2];
    let gpu = layer(n, k, true, DType::BF16, &device)?;
    let cpu = layer(n, k, true, DType::BF16, &Device::Cpu)?;
    let x = input(k, ROUTES, DType::BF16, &device)?;
    let ids = view(Tensor::new(&[IDS], &device)?)?;
    let expected =
        cpu.gather_forward_raw(&x.to_device(&Device::Cpu)?, &ids.to_device(&Device::Cpu)?)?;
    compare(&gpu.gather_forward_raw(&x, &ids)?, &expected)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    device.synchronize()?;
    let tracking = stream.context().is_event_tracking();
    if tracking {
        unsafe { stream.context().disable_event_tracking() };
    }
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(candle_core::Error::wrap)?;
    let output = gpu.gather_forward_raw(&x, &ids)?;
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(candle_core::Error::wrap)?
        .unwrap();
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    graph.launch().map_err(candle_core::Error::wrap)?;
    compare(&output, &expected)?;
    ids.slice_set(&Tensor::new(&[NEXT_IDS], &device)?, 0, 0)?;
    x.slice_set(&(x.affine(-0.75, 0.125)?), 0, 0)?;
    let expected =
        cpu.gather_forward_raw(&x.to_device(&Device::Cpu)?, &ids.to_device(&Device::Cpu)?)?;
    graph.launch().map_err(candle_core::Error::wrap)?;
    compare(&output, &expected)?;
    Ok(())
}

#[test]
fn nvfp4_decode_dispatch_preserves_prefill_and_k_heavy_paths() -> Result<()> {
    let dense = layer(83, 80, false, DType::F16, &Device::Cpu)?;
    assert!(supports(1, 1, DType::F16, dense.gemm_args()));
    assert!(!supports(2, 1, DType::F16, dense.gemm_args()));
    assert!(!supports(1, 1, DType::F32, dense.gemm_args()));
    let k_heavy = layer(17, 80, false, DType::F16, &Device::Cpu)?;
    assert!(!supports(1, 1, DType::F16, k_heavy.gemm_args()));
    let mut args = dense.gemm_args();
    args.activation_global_scale = None;
    assert!(!supports(1, 1, DType::F16, args));
    Ok(())
}
