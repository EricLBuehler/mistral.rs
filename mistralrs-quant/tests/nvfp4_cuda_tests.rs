#![cfg(all(feature = "cuda", feature = "cutile"))]

use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;
use mistralrs_quant::cutile::{cutile_nvfp4, cutile_nvfp4_gather, Nvfp4GemmArgs};

const BLOCK_SIZE: usize = 16;
const FP4_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const FP8_MAX: f32 = 448.0;
const ACTIVATION_GLOBAL: f32 = 0.015625;
const BLOCK_SCALES: [f32; 5] = [0.3125, 0.75, 1.5, 2.75, 5.5];
const GLOBAL_SCALES: [f32; 3] = [0.03125, 0.09375, 0.15625];

const DENSE_SHAPES: [(usize, usize, usize); 8] = [
    (1, 19, 80),
    (4, 37, 528),
    (5, 73, 144),
    (71, 73, 144),
    (1, 19, 8208),
    (7, 129, 1040),
    (1, 4096, 80),
    (1, 4097, 8208),
];
const BF16_TOLERANCE: f32 = 0.012;
const F16_TOLERANCE: f32 = 0.002;
const GATHER_TOKENS: usize = 3;
const GROUPED_TOKENS: usize = 37;
const GATHER_TOPK: usize = 2;
const GATHER_EXPERTS: usize = 4;
const GATHER_N: usize = 73;
const GATHER_K: usize = 80;
const GATHER_ROUTES: [u32; GATHER_TOKENS * GATHER_TOPK] = [2, 0, 2, 1, 2, 2];
const GATHER_ACTIVATION_GLOBALS: [f32; GATHER_EXPERTS] = [0.015625, 0.03125, 0.0078125, 0.0625];
const GRAPH_ROWS: usize = 37;
const GRAPH_N: usize = 73;
const GRAPH_K: usize = 80;
const GRAPH_EXPERTS: usize = 2;
const GRAPH_TOPK: usize = 2;
const EXTREME_SHAPE: (usize, usize, usize) = (3, 13, 32);
const SATURATING_ACTIVATION: f32 = 2048.0;
const UNDERFLOW_ACTIVATION: f32 = 1.0e-6;
const ROUNDING_GLOBAL: f32 = f32::from_bits(0x3b08_8889);
const ROUNDING_SMALL_INPUT: f32 = 1.0 / 512.0;
const ROUNDING_LARGE_INPUT: f32 = 1.0 / 64.0;
const ROUNDING_WEIGHT_GLOBAL: f32 = 1024.0;

// Capture requires exclusive use of the shared primary CUDA context.
static CUDA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fp4(bits: u8) -> f32 {
    FP4_VALUES[usize::from(bits & 7)] * if bits & 8 == 0 { 1.0 } else { -1.0 }
}

fn round_fp4(value: f32) -> f32 {
    let mut closest = 0;
    for (index, &candidate) in FP4_VALUES.iter().enumerate().skip(1) {
        let distance = (value.abs() - candidate).abs();
        let previous = (value.abs() - FP4_VALUES[closest]).abs();
        if distance < previous || (distance == previous && index % 2 == 0) {
            closest = index;
        }
    }
    FP4_VALUES[closest].copysign(value)
}

fn round_activation(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => half::bf16::from_f32(value).to_f32(),
        DType::F16 => half::f16::from_f32(value).to_f32(),
        _ => unreachable!(),
    }
}

struct Fixture {
    rows: usize,
    n: usize,
    k: usize,
    dtype: DType,
    x: Vec<f32>,
    packed: Vec<u8>,
    scales: Vec<F8E4M3>,
    global: Vec<f32>,
}

impl Fixture {
    fn new(rows: usize, n: usize, k: usize, dtype: DType) -> Self {
        let x = (0..rows * k)
            .map(|i| {
                let value = if i / k % 13 == 12 || i % k / BLOCK_SIZE == 1 {
                    0.0
                } else {
                    ((i * 17 + i / k * 7) % 113) as f32 / 29.0 - 1.75
                };
                round_activation(value, dtype)
            })
            .collect();
        let packed = (0..n * k / 2)
            .map(|i| {
                let lower = ((i * 3 + i / (k / 2)) % 16) as u8;
                let upper = ((i * 7 + i / (k / 2) * 5 + 3) % 16) as u8;
                lower | (upper << 4)
            })
            .collect();
        let scales = (0..n * k / BLOCK_SIZE)
            .map(|i| {
                F8E4M3::from_f32(BLOCK_SCALES[(i + i / (k / BLOCK_SIZE)) % BLOCK_SCALES.len()])
            })
            .collect();
        let global = (0..n)
            .map(|i| GLOBAL_SCALES[i % GLOBAL_SCALES.len()])
            .collect();
        Self {
            rows,
            n,
            k,
            dtype,
            x,
            packed,
            scales,
            global,
        }
    }

    fn reference(&self, a4: bool, activation_global: f32) -> Vec<f32> {
        let mut x = self.x.clone();
        if a4 {
            for block in x.as_chunks_mut::<BLOCK_SIZE>().0 {
                let normalized: Vec<_> = block.iter().map(|x| x / activation_global).collect();
                let maximum = normalized.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = F8E4M3::from_f32((maximum / FP4_VALUES[7]).min(FP8_MAX)).to_f32();
                let denominator = if scale == 0.0 { 1.0 } else { scale };
                for (value, normalized) in block.iter_mut().zip(normalized) {
                    *value = round_fp4(normalized / denominator) * scale;
                }
            }
        }
        let mut result = vec![0.0; self.rows * self.n];
        for row in 0..self.rows {
            for out in 0..self.n {
                let mut acc = 0.0;
                for col in 0..self.k {
                    let byte = self.packed[out * self.k / 2 + col / 2];
                    let nibble = if col % 2 == 0 { byte & 15 } else { byte >> 4 };
                    let scale = self.scales[out * self.k / BLOCK_SIZE + col / BLOCK_SIZE].to_f32();
                    let weight = fp4(nibble) * scale;
                    let weight = if a4 {
                        weight
                    } else {
                        round_activation(weight, self.dtype)
                    };
                    acc += x[row * self.k + col] * weight;
                }
                let global = if a4 { activation_global } else { 1.0 };
                result[row * self.n + out] =
                    round_activation(acc * self.global[out] * global, self.dtype);
            }
        }
        result
    }

    fn check(&self, a4: bool, device: &Device) -> Result<()> {
        self.check_with_global(a4, ACTIVATION_GLOBAL, device)
    }

    fn check_with_global(&self, a4: bool, activation_scale: f32, device: &Device) -> Result<()> {
        let x =
            Tensor::from_vec(self.x.clone(), (self.rows, self.k), device)?.to_dtype(self.dtype)?;
        let weights = Tensor::from_vec(self.packed.clone(), (self.n, self.k / 2), device)?;
        let scales = Tensor::from_vec(self.scales.clone(), (self.n, self.k / BLOCK_SIZE), device)?;
        let global = Tensor::from_vec(self.global.clone(), self.n, device)?;
        let activation_global = Tensor::new(activation_scale, device)?;
        let actual = cutile_nvfp4(
            &x,
            Nvfp4GemmArgs {
                weights: &weights,
                weight_scales: &scales,
                weight_global_scale: &global,
                activation_global_scale: a4.then_some(&activation_global),
            },
        )?;
        assert_eq!(actual.dims(), [self.rows, self.n]);
        assert_eq!(actual.dtype(), self.dtype);
        let actual = actual
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let tolerance = if self.dtype == DType::BF16 {
            BF16_TOLERANCE
        } else {
            F16_TOLERANCE
        };
        for (index, (actual, expected)) in actual
            .iter()
            .zip(self.reference(a4, activation_scale))
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                "NVFP4 {:?} a4={a4} M={} N={} K={} output[{index}]={actual}, reference={expected}",
                self.dtype,
                self.rows,
                self.n,
                self.k
            );
        }
        Ok(())
    }
}

#[test]
fn nvfp4_dense_matches_quantized_reference() -> Result<()> {
    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    for dtype in [DType::BF16, DType::F16] {
        for a4 in [false, true] {
            for (m, n, k) in DENSE_SHAPES {
                Fixture::new(m, n, k, dtype).check(a4, &device)?;
            }
        }
    }
    Ok(())
}

#[test]
fn nvfp4_activation_scale_extremes_match_reference() -> Result<()> {
    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    let (rows, n, k) = EXTREME_SHAPE;
    for dtype in [DType::BF16, DType::F16] {
        let mut fixture = Fixture::new(rows, n, k, dtype);
        for (index, value) in fixture.x.iter_mut().enumerate() {
            let magnitude = match index / k {
                0 => 0.0,
                1 => SATURATING_ACTIVATION,
                _ => UNDERFLOW_ACTIVATION,
            };
            *value = round_activation(
                if index % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                },
                dtype,
            );
        }
        fixture.check(true, &device)?;
    }
    Ok(())
}

#[test]
fn nvfp4_activation_rounding_boundary_matches_reference() -> Result<()> {
    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    for dtype in [DType::BF16, DType::F16] {
        for rows in [1, 5] {
            let mut fixture = Fixture::new(rows, 1, BLOCK_SIZE, dtype);
            fixture.x.fill(0.0);
            for block in fixture.x.as_chunks_mut::<BLOCK_SIZE>().0 {
                block[0] = ROUNDING_SMALL_INPUT;
                block[1] = ROUNDING_LARGE_INPUT;
            }
            fixture.packed.fill(0);
            fixture.packed[0] = 2;
            fixture.scales.fill(F8E4M3::from_f32(1.0));
            fixture.global.fill(ROUNDING_WEIGHT_GLOBAL);
            fixture.check_with_global(true, ROUNDING_GLOBAL, &device)?;
        }
    }
    Ok(())
}

#[test]
fn nvfp4_gather_matches_selected_experts() -> Result<()> {
    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    for tokens in [GATHER_TOKENS, GROUPED_TOKENS] {
        let routes: Vec<_> = GATHER_ROUTES
            .into_iter()
            .cycle()
            .take(tokens * GATHER_TOPK)
            .collect();
        let indices = Tensor::from_vec(routes.clone(), (tokens, GATHER_TOPK), &device)?;
        let activation_globals =
            Tensor::from_vec(GATHER_ACTIVATION_GLOBALS.to_vec(), GATHER_EXPERTS, &device)?;
        for dtype in [DType::BF16, DType::F16] {
            for input_layout in [0, 1, 2] {
                let per_route_input = input_layout == 2;
                let input_rows = if per_route_input {
                    tokens * GATHER_TOPK
                } else {
                    tokens
                };
                let fixtures: Vec<_> = (0..GATHER_EXPERTS)
                    .map(|expert| {
                        let mut fixture = Fixture::new(input_rows, GATHER_N, GATHER_K, dtype);
                        fixture.packed.rotate_left(expert * 5);
                        fixture.scales.rotate_left(expert * 7);
                        fixture
                            .global
                            .iter_mut()
                            .for_each(|scale| *scale *= (expert + 1) as f32);
                        fixture
                    })
                    .collect();
                let x = Tensor::from_vec(fixtures[0].x.clone(), (input_rows, GATHER_K), &device)?
                    .to_dtype(dtype)?;
                let x = match input_layout {
                    1 => x.reshape((tokens, 1, GATHER_K))?,
                    2 => x.reshape((tokens, GATHER_TOPK, GATHER_K))?,
                    _ => x,
                };
                let weights = Tensor::from_vec(
                    fixtures
                        .iter()
                        .flat_map(|fixture| fixture.packed.clone())
                        .collect(),
                    (GATHER_EXPERTS, GATHER_N, GATHER_K / 2),
                    &device,
                )?;
                let scales = Tensor::from_vec(
                    fixtures
                        .iter()
                        .flat_map(|fixture| fixture.scales.clone())
                        .collect(),
                    (GATHER_EXPERTS, GATHER_N, GATHER_K / BLOCK_SIZE),
                    &device,
                )?;
                let globals = Tensor::from_vec(
                    fixtures
                        .iter()
                        .flat_map(|fixture| fixture.global.clone())
                        .collect(),
                    (GATHER_EXPERTS, GATHER_N),
                    &device,
                )?;
                for a4 in [false, true] {
                    let actual = cutile_nvfp4_gather(
                        &x,
                        &indices,
                        Nvfp4GemmArgs {
                            weights: &weights,
                            weight_scales: &scales,
                            weight_global_scale: &globals,
                            activation_global_scale: a4.then_some(&activation_globals),
                        },
                    )?;
                    assert_eq!(actual.dims(), [tokens, GATHER_TOPK, GATHER_N]);
                    let actual = actual
                        .to_dtype(DType::F32)?
                        .to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let references: Vec<_> = fixtures
                        .iter()
                        .zip(GATHER_ACTIVATION_GLOBALS)
                        .map(|(fixture, global)| fixture.reference(a4, global))
                        .collect();
                    let tolerance = if dtype == DType::BF16 {
                        BF16_TOLERANCE
                    } else {
                        F16_TOLERANCE
                    };
                    for (route, &expert) in routes.iter().enumerate() {
                        let input_row = if per_route_input {
                            route
                        } else {
                            route / GATHER_TOPK
                        };
                        for col in 0..GATHER_N {
                            let actual = actual[route * GATHER_N + col];
                            let expected = references[expert as usize][input_row * GATHER_N + col];
                            assert!((actual - expected).abs() <= tolerance * expected.abs().max(1.0), "NVFP4 gather dtype={dtype:?} a4={a4} per_route={per_route_input} route={route} col={col}: {actual} != {expected}");
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn nvfp4_warmup_supports_cuda_graph_replay() -> Result<()> {
    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    use candle_core::cuda::cudarc::driver::sys;
    use mistralrs_quant::cutile::{
        register_nvfp4_routing, register_nvfp4_shape, warmup_moe_kernels,
    };

    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    register_nvfp4_routing(&device, GRAPH_EXPERTS, GRAPH_TOPK)?;
    let fixture = Fixture::new(GRAPH_ROWS, GRAPH_N, GRAPH_K, DType::BF16);
    let source = Tensor::from_vec(fixture.x.clone(), (GRAPH_ROWS, GRAPH_K), &device)?
        .to_dtype(DType::BF16)?;
    let alternate = source.neg()?;
    let x = Tensor::zeros((GRAPH_ROWS, GRAPH_K), DType::BF16, &device)?;
    x.slice_set(&source, 0, 0)?;
    let routes: Vec<_> = (0..GRAPH_ROWS * GRAPH_TOPK)
        .map(|route| (route % GRAPH_EXPERTS) as u32)
        .collect();
    let indices = Tensor::from_vec(routes.clone(), (GRAPH_ROWS, GRAPH_TOPK), &device)?;
    let alternate_indices = Tensor::from_vec(
        routes.iter().map(|route| 1 - route).collect::<Vec<_>>(),
        (GRAPH_ROWS, GRAPH_TOPK),
        &device,
    )?;
    let original_indices = Tensor::from_vec(routes.clone(), (GRAPH_ROWS, GRAPH_TOPK), &device)?;
    let weights = Tensor::from_vec(fixture.packed.clone(), (GRAPH_N, GRAPH_K / 2), &device)?;
    let scales = Tensor::from_vec(
        fixture.scales.clone(),
        (GRAPH_N, GRAPH_K / BLOCK_SIZE),
        &device,
    )?;
    let global = Tensor::from_vec(fixture.global.clone(), GRAPH_N, &device)?;
    let activation_global = Tensor::new(ACTIVATION_GLOBAL, &device)?;
    let routed_weights = Tensor::stack(&[&weights, &weights], 0)?;
    let routed_scales = Tensor::stack(&[&scales, &scales], 0)?;
    let routed_globals = Tensor::stack(&[&global, &global.affine(2.0, 0.0)?], 0)?;
    let routed_activation_globals = Tensor::from_vec(
        vec![ACTIVATION_GLOBAL; GRAPH_EXPERTS],
        GRAPH_EXPERTS,
        &device,
    )?;
    let decode_x = x.narrow(0, 0, 1)?;
    let routed_x = x.unsqueeze(1)?;
    for a4 in [false, true] {
        let dense = Nvfp4GemmArgs {
            weights: &weights,
            weight_scales: &scales,
            weight_global_scale: &global,
            activation_global_scale: a4.then_some(&activation_global),
        };
        let routed = Nvfp4GemmArgs {
            weights: &routed_weights,
            weight_scales: &routed_scales,
            weight_global_scale: &routed_globals,
            activation_global_scale: a4.then_some(&routed_activation_globals),
        };
        register_nvfp4_shape(dense, DType::BF16);
        register_nvfp4_shape(routed, DType::BF16);
        warmup_moe_kernels(&device)?;
        let _htod_cache_guard = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
        for rows in [1, GRAPH_ROWS, GRAPH_ROWS * GRAPH_TOPK] {
            Tensor::zeros((rows, GRAPH_N.next_power_of_two()), DType::BF16, &device)?
                .narrow(1, 0, GRAPH_N)?
                .contiguous()?;
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
            return Err(candle_core::Error::msg(error.to_string()));
        }
        let captured = (|| -> Result<_> {
            Ok((
                cutile_nvfp4(&decode_x, dense)?,
                cutile_nvfp4(&x, dense)?,
                cutile_nvfp4_gather(&routed_x, &indices, routed)?,
            ))
        })();
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let (decode, prefill, gathered) = captured?;
        let graph = graph
            .map_err(|error| candle_core::Error::msg(error.to_string()))?
            .ok_or_else(|| candle_core::Error::msg("NVFP4 capture produced no graph"))?;
        let reference = fixture.reference(a4, ACTIVATION_GLOBAL);
        for alternate_input in [false, true, false] {
            x.slice_set(if alternate_input { &alternate } else { &source }, 0, 0)?;
            indices.slice_set(
                if alternate_input {
                    &alternate_indices
                } else {
                    &original_indices
                },
                0,
                0,
            )?;
            graph
                .launch()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            device.synchronize()?;
            let sign = if alternate_input { -1.0 } else { 1.0 };
            for (output, rows) in [(&decode, 1), (&prefill, GRAPH_ROWS)] {
                let actual = output
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(actual.len(), rows * GRAPH_N);
                for (index, &actual) in actual.iter().enumerate() {
                    let expected = reference[index] * sign;
                    assert!((actual - expected).abs() <= BF16_TOLERANCE * expected.abs().max(1.0), "NVFP4 graph dense a4={a4} alternate={alternate_input} index={index}: {actual} != {expected}");
                }
            }
            let actual = gathered
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (route, &id) in routes.iter().enumerate() {
                let expert = if alternate_input { 1 - id } else { id };
                for col in 0..GRAPH_N {
                    let expected =
                        reference[route / GRAPH_TOPK * GRAPH_N + col] * sign * (expert + 1) as f32;
                    let actual = actual[route * GRAPH_N + col];
                    assert!((actual - expected).abs() <= BF16_TOLERANCE * expected.abs().max(1.0), "NVFP4 graph gather a4={a4} alternate={alternate_input} route={route} col={col}: {actual} != {expected}");
                }
            }
        }
        drop((decode, prefill, gathered));
        device.synchronize()?;
        drop(graph);
        x.slice_set(&source, 0, 0).map_err(|error| {
            error.context("NVFP4 activations must remain writable after graph destruction")
        })?;
        let after_destroy = cutile_nvfp4(&x, dense)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (actual, expected) in after_destroy.into_iter().zip(&reference) {
            assert!(
                (actual - expected).abs() <= BF16_TOLERANCE * expected.abs().max(1.0),
                "NVFP4 graph destruction changed subsequent inference: {actual} != {expected}"
            );
        }
        device.synchronize()?;
    }
    Ok(())
}
