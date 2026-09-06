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
const PREFILL_ROWS: [usize; 2] = [129, 1153];
const PREFILL_N: usize = 11009;
const PREFILL_K: usize = 4112;
const PREFILL_COLUMNS: [usize; 10] = [
    0,
    1,
    63,
    64,
    127,
    128,
    255,
    256,
    PREFILL_N / 2,
    PREFILL_N - 1,
];
const PREFILL_A16_SHAPE: (usize, usize, usize) = (129, 4097, 8208);
const PREFILL_A16_COLUMNS: [usize; 5] = [0, 63, 64, 128, 4096];
const GROUPED_GRAPH_ROWS: usize = 129;

const MERGED_FIRST_N: usize = 3;
const MERGED_SECOND_N: usize = 5;
const MERGED_K: usize = 80;
const MERGED_ROWS: usize = 5;
const MEDIUM_GRAPH_ROWS: [usize; 13] = [16, 17, 18, 20, 23, 24, 31, 32, 33, 34, 36, 40, 48];
const MEDIUM_GRAPH_MAX_ROWS: usize = 48;
const MEDIUM_GRAPH_N: usize = 1025;
const MEDIUM_GRAPH_K: usize = 4112;
const MEDIUM_GRAPH_WEIGHT_OFFSET: usize = 1;
const MEDIUM_GRAPH_COLUMNS: [usize; 10] = [0, 1, 63, 64, 127, 128, 255, 256, 512, 1024];

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
        self.reference_columns(a4, activation_global, &(0..self.n).collect::<Vec<_>>())
    }

    fn reference_columns(&self, a4: bool, activation_global: f32, columns: &[usize]) -> Vec<f32> {
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
        let mut result = vec![0.0; self.rows * columns.len()];
        for row in 0..self.rows {
            for (out_index, &out) in columns.iter().enumerate() {
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
                result[row * columns.len() + out_index] =
                    round_activation(acc * self.global[out] * global, self.dtype);
            }
        }
        result
    }

    fn check(&self, a4: bool, device: &Device) -> Result<()> {
        self.check_with_global(a4, ACTIVATION_GLOBAL, device)
    }

    fn check_with_global(&self, a4: bool, activation_scale: f32, device: &Device) -> Result<()> {
        self.check_columns(
            a4,
            activation_scale,
            device,
            &(0..self.n).collect::<Vec<_>>(),
        )
    }

    fn check_columns(
        &self,
        a4: bool,
        activation_scale: f32,
        device: &Device,
        columns: &[usize],
    ) -> Result<()> {
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
        for (sample, expected) in self
            .reference_columns(a4, activation_scale, columns)
            .into_iter()
            .enumerate()
        {
            let index = sample / columns.len() * self.n + columns[sample % columns.len()];
            let actual = actual[index];
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
fn nvfp4_prefill_preserves_partial_tiles_and_groups() -> Result<()> {
    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    for dtype in [DType::BF16, DType::F16] {
        for rows in PREFILL_ROWS {
            Fixture::new(rows, PREFILL_N, PREFILL_K, dtype).check_columns(
                true,
                ACTIVATION_GLOBAL,
                &device,
                &PREFILL_COLUMNS,
            )?;
        }
        let (rows, n, k) = PREFILL_A16_SHAPE;
        Fixture::new(rows, n, k, dtype).check_columns(
            false,
            ACTIVATION_GLOBAL,
            &device,
            &PREFILL_A16_COLUMNS,
        )?;
    }
    Ok(())
}

#[test]
fn nvfp4_medium_decode_warmup_supports_offset_views_and_graph_replay() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;
    use mistralrs_quant::cutile::{
        cutile_nvfp4_prequantized, cutile_nvfp4_quantize, register_nvfp4_shape, warmup_moe_kernels,
    };

    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let stored_n = MEDIUM_GRAPH_N + MEDIUM_GRAPH_WEIGHT_OFFSET;
    let columns = MEDIUM_GRAPH_COLUMNS.map(|column| column + MEDIUM_GRAPH_WEIGHT_OFFSET);
    for dtype in [DType::BF16, DType::F16] {
        let fixture = Fixture::new(MEDIUM_GRAPH_MAX_ROWS, stored_n, MEDIUM_GRAPH_K, dtype);
        let source = Tensor::from_vec(
            fixture.x.clone(),
            (MEDIUM_GRAPH_MAX_ROWS, MEDIUM_GRAPH_K),
            &device,
        )?
        .to_dtype(dtype)?;
        let alternate = source.neg()?;
        let x = Tensor::zeros((MEDIUM_GRAPH_MAX_ROWS, MEDIUM_GRAPH_K), dtype, &device)?;
        let weights = Tensor::from_vec(
            fixture.packed.clone(),
            (stored_n, MEDIUM_GRAPH_K / 2),
            &device,
        )?
        .narrow(0, MEDIUM_GRAPH_WEIGHT_OFFSET, MEDIUM_GRAPH_N)?;
        let scales = Tensor::from_vec(
            fixture.scales.clone(),
            (stored_n, MEDIUM_GRAPH_K / BLOCK_SIZE),
            &device,
        )?
        .narrow(0, MEDIUM_GRAPH_WEIGHT_OFFSET, MEDIUM_GRAPH_N)?;
        let global = Tensor::from_vec(fixture.global.clone(), stored_n, &device)?.narrow(
            0,
            MEDIUM_GRAPH_WEIGHT_OFFSET,
            MEDIUM_GRAPH_N,
        )?;
        for tensor in [&weights, &scales, &global] {
            assert!(tensor.is_contiguous());
            assert!(tensor.storage_and_layout().1.start_offset() > 0);
        }
        let activation_global = Tensor::new(ACTIVATION_GLOBAL, &device)?;
        let args = Nvfp4GemmArgs {
            weights: &weights,
            weight_scales: &scales,
            weight_global_scale: &global,
            activation_global_scale: Some(&activation_global),
        };
        register_nvfp4_shape(args, dtype)?;
        warmup_moe_kernels(&device)?;
        let warmed_count = cutile::tile_kernel::jit_compile_count();
        let expected = fixture.reference_columns(true, ACTIVATION_GLOBAL, &columns);
        let _htod_cache_guard = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
        x.slice_set(&source, 0, 0)?;
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
            MEDIUM_GRAPH_ROWS
                .into_iter()
                .map(|rows| {
                    let input = x.narrow(0, 0, rows)?;
                    let ordinary = cutile_nvfp4(&input, args)?;
                    let (packed, scales) = cutile_nvfp4_quantize(&input, &activation_global)?;
                    let shared = cutile_nvfp4_prequantized(&packed, &scales, dtype, args)?;
                    Ok((rows, ordinary, shared))
                })
                .collect::<Result<Vec<_>>>()
        })();
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let outputs = captured?;
        let graph = graph
            .map_err(|error| candle_core::Error::msg(error.to_string()))?
            .ok_or_else(|| candle_core::Error::msg("NVFP4 medium capture produced no graph"))?;
        assert_eq!(cutile::tile_kernel::jit_compile_count(), warmed_count);
        let tolerance = if dtype == DType::BF16 {
            BF16_TOLERANCE
        } else {
            F16_TOLERANCE
        };
        for alternate_input in [false, true, false] {
            x.slice_set(if alternate_input { &alternate } else { &source }, 0, 0)?;
            graph
                .launch()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            device.synchronize()?;
            let sign = if alternate_input { -1.0 } else { 1.0 };
            for (rows, ordinary, shared) in &outputs {
                for (path, output) in [("ordinary", ordinary), ("shared", shared)] {
                    assert_eq!(output.dims(), [*rows, MEDIUM_GRAPH_N]);
                    let actual = output
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    for row in 0..*rows {
                        for (sample, &column) in MEDIUM_GRAPH_COLUMNS.iter().enumerate() {
                            let expected =
                                expected[row * MEDIUM_GRAPH_COLUMNS.len() + sample] * sign;
                            let value = actual[row * MEDIUM_GRAPH_N + column];
                            assert!(
                                (value - expected).abs() <= tolerance * expected.abs().max(1.0),
                                "NVFP4 medium graph {dtype:?} {path} M={rows} alternate={alternate_input} row={row} col={column}: {value} != {expected}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(cutile::tile_kernel::jit_compile_count(), warmed_count);
        drop(outputs);
        device.synchronize()?;
        drop(graph);
        x.slice_set(&source, 0, 0)?;
        device.synchronize()?;
    }
    Ok(())
}

#[test]
fn nvfp4_grouped_prefill_warmup_supports_graph_replay() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;
    use mistralrs_quant::cutile::{register_nvfp4_shape, warmup_moe_kernels};

    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let fixture = Fixture::new(GROUPED_GRAPH_ROWS, PREFILL_N, PREFILL_K, DType::BF16);
    let source = Tensor::from_vec(fixture.x.clone(), (GROUPED_GRAPH_ROWS, PREFILL_K), &device)?
        .to_dtype(DType::BF16)?;
    let alternate = source.neg()?;
    let x = Tensor::zeros((GROUPED_GRAPH_ROWS, PREFILL_K), DType::BF16, &device)?;
    let weights = Tensor::from_vec(fixture.packed.clone(), (PREFILL_N, PREFILL_K / 2), &device)?;
    let scales = Tensor::from_vec(
        fixture.scales.clone(),
        (PREFILL_N, PREFILL_K / BLOCK_SIZE),
        &device,
    )?;
    let global = Tensor::from_vec(fixture.global.clone(), PREFILL_N, &device)?;
    let activation_global = Tensor::new(ACTIVATION_GLOBAL, &device)?;
    let args = Nvfp4GemmArgs {
        weights: &weights,
        weight_scales: &scales,
        weight_global_scale: &global,
        activation_global_scale: Some(&activation_global),
    };
    register_nvfp4_shape(args, DType::BF16)?;
    warmup_moe_kernels(&device)?;
    let _htod_cache_guard = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
    x.slice_set(&source, 0, 0)?;
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
    let captured = cutile_nvfp4(&x, args);
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let output = captured?;
    let graph = graph
        .map_err(|error| candle_core::Error::msg(error.to_string()))?
        .ok_or_else(|| candle_core::Error::msg("NVFP4 grouped capture produced no graph"))?;
    let expected = fixture.reference_columns(true, ACTIVATION_GLOBAL, &PREFILL_COLUMNS);
    for alternate_input in [false, true, false] {
        x.slice_set(if alternate_input { &alternate } else { &source }, 0, 0)?;
        graph
            .launch()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        device.synchronize()?;
        let actual = output
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let sign = if alternate_input { -1.0 } else { 1.0 };
        for (sample, &expected) in expected.iter().enumerate() {
            let index = sample / PREFILL_COLUMNS.len() * PREFILL_N
                + PREFILL_COLUMNS[sample % PREFILL_COLUMNS.len()];
            let expected = expected * sign;
            assert!(
                (actual[index] - expected).abs() <= BF16_TOLERANCE * expected.abs().max(1.0),
                "NVFP4 grouped graph alternate={alternate_input} index={index}: {} != {expected}",
                actual[index]
            );
        }
    }
    drop(output);
    device.synchronize()?;
    drop(graph);
    x.slice_set(&source, 0, 0)?;
    device.synchronize()?;
    Ok(())
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
        register_nvfp4_shape(dense, DType::BF16)?;
        register_nvfp4_shape(routed, DType::BF16)?;
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

#[test]
fn nvfp4_merged_views_and_shared_activation_support_graph_replay() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;
    use mistralrs_quant::cutile::warmup_moe_kernels;
    use mistralrs_quant::{ColumnParallelLayer, Comm, Id, QuantizedConfig, ShardedSafeTensors};
    use std::{collections::HashMap, sync::Arc};

    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 0, 1)?);
    for dtype in [DType::BF16, DType::F16] {
        for a4 in [false, true] {
            let n = MERGED_FIRST_N + MERGED_SECOND_N;
            let fixture = Fixture::new(MERGED_ROWS, n, MERGED_K, dtype);
            let source = Tensor::from_vec(fixture.x.clone(), (MERGED_ROWS, MERGED_K), &device)?
                .to_dtype(dtype)?;
            let alternate = source.neg()?;
            let x = Tensor::zeros((MERGED_ROWS, MERGED_K), dtype, &device)?;
            let mut tensors = HashMap::new();
            let mut row_offset = 0;
            for (name, rows) in [("first", MERGED_FIRST_N), ("second", MERGED_SECOND_N)] {
                tensors.insert(
                    format!("{name}.weight"),
                    Tensor::from_vec(
                        fixture.packed
                            [row_offset * MERGED_K / 2..(row_offset + rows) * MERGED_K / 2]
                            .to_vec(),
                        (rows, MERGED_K / 2),
                        &device,
                    )?,
                );
                tensors.insert(
                    format!("{name}.weight_scale"),
                    Tensor::from_vec(
                        fixture.scales[row_offset * MERGED_K / BLOCK_SIZE
                            ..(row_offset + rows) * MERGED_K / BLOCK_SIZE]
                            .to_vec(),
                        (rows, MERGED_K / BLOCK_SIZE),
                        &device,
                    )?,
                );
                tensors.insert(
                    format!("{name}.weight_scale_2"),
                    Tensor::from_vec(
                        fixture.global[row_offset..row_offset + rows].to_vec(),
                        rows,
                        &device,
                    )?,
                );
                if a4 {
                    tensors.insert(
                        format!("{name}.input_scale"),
                        Tensor::new(ACTIVATION_GLOBAL, &device)?,
                    );
                }
                row_offset += rows;
            }
            let config = Some(serde_json::from_value::<QuantizedConfig>(serde_json::json!({
                "quant_method": "modelopt", "quant_algo": if a4 { "NVFP4" } else { "W4A16_NVFP4" }, "group_size": BLOCK_SIZE,
            })).map_err(candle_core::Error::msg)?);
            let group = ColumnParallelLayer::new_packed(
                MERGED_K,
                &[MERGED_FIRST_N, MERGED_SECOND_N],
                &["first", "second"],
                &config,
                false,
                &comm,
                None,
                ShardedSafeTensors::wrap(tensors, dtype, device.clone()),
            )?
            .expect("compatible projections should merge");
            warmup_moe_kernels(&device)?;
            let _htod_cache_guard = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
            x.slice_set(&source, 0, 0)?;
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
                let packed = group.packed.forward(&x)?;
                let decode = group.constituents[1].forward(&x.narrow(0, 0, 1)?)?;
                let separate = if a4 {
                    let (first, second, _) = mistralrs_quant::try_fused_quantized_qkv(
                        &x,
                        &*group.constituents[0],
                        &*group.constituents[1],
                        &*group.constituents[0],
                    )?
                    .expect("compatible W4A4 projections should share quantization");
                    vec![first, second]
                } else {
                    group
                        .constituents
                        .iter()
                        .map(|layer| layer.forward(&x))
                        .collect::<Result<Vec<_>>>()?
                };
                Ok((packed, decode, separate))
            })();
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            let (packed, decode, separate) = captured?;
            let graph = graph
                .map_err(|error| candle_core::Error::msg(error.to_string()))?
                .ok_or_else(|| {
                    candle_core::Error::msg("NVFP4 merged graph capture returned no graph")
                })?;
            let reference = fixture.reference(a4, ACTIVATION_GLOBAL);
            let tolerance = if dtype == DType::BF16 {
                BF16_TOLERANCE
            } else {
                F16_TOLERANCE
            };
            for alternate_input in [false, true, false] {
                x.slice_set(if alternate_input { &alternate } else { &source }, 0, 0)?;
                graph
                    .launch()
                    .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                device.synchronize()?;
                let sign = if alternate_input { -1.0 } else { 1.0 };
                for (output, start, columns, rows) in [
                    (&packed, 0, n, MERGED_ROWS),
                    (&decode, MERGED_FIRST_N, MERGED_SECOND_N, 1),
                    (&separate[0], 0, MERGED_FIRST_N, MERGED_ROWS),
                    (&separate[1], MERGED_FIRST_N, MERGED_SECOND_N, MERGED_ROWS),
                ] {
                    let actual = output
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    for row in 0..rows {
                        for column in 0..columns {
                            let expected = reference[row * n + start + column] * sign;
                            let value = actual[row * columns + column];
                            assert!((value - expected).abs() <= tolerance * expected.abs().max(1.0),
                                "merged/shared NVFP4 dtype={dtype:?} a4={a4} row={row} col={column}: {value} != {expected}");
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn nvfp4_prequantized_forward_materializes_strided_values_and_scales() -> Result<()> {
    use mistralrs_quant::{
        Nvfp4ActivationMode, Nvfp4Layer, Nvfp4LayerParts, QuantMethod, QuantizedActivation,
    };
    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    for dtype in [DType::BF16, DType::F16] {
        let n = MERGED_FIRST_N + MERGED_SECOND_N;
        let fixture = Fixture::new(MERGED_ROWS, n, MERGED_K, dtype);
        let source = Tensor::from_vec(fixture.x.clone(), (MERGED_ROWS, MERGED_K), &device)?
            .to_dtype(dtype)?;
        let layer = Nvfp4Layer::from_parts(Nvfp4LayerParts {
            weight: Tensor::from_vec(fixture.packed.clone(), (n, MERGED_K / 2), &device)?,
            scales: Tensor::from_vec(fixture.scales.clone(), (n, MERGED_K / BLOCK_SIZE), &device)?,
            global_scales: Tensor::from_vec(fixture.global.clone(), n, &device)?,
            input_scale: Some(Tensor::new(ACTIVATION_GLOBAL, &device)?),
            activation: Nvfp4ActivationMode::DynamicBlock,
            bias: None,
            dtype,
        })?;
        let quantized = layer.quantize_activation(&source)?;
        let packed = Tensor::cat(&[quantized.quantized(), quantized.quantized()], 1)?.narrow(
            1,
            MERGED_K / 2,
            MERGED_K / 2,
        )?;
        let scales = Tensor::cat(&[quantized.scales(), quantized.scales()], 1)?.narrow(
            1,
            MERGED_K / BLOCK_SIZE,
            MERGED_K / BLOCK_SIZE,
        )?;
        assert!(!packed.is_contiguous());
        assert!(!scales.is_contiguous());
        let strided = QuantizedActivation::new_nvfp4(packed, scales, &source, ACTIVATION_GLOBAL)?;
        let output = layer
            .forward_quantized(&strided)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let reference = fixture.reference(true, ACTIVATION_GLOBAL);
        let tolerance = if dtype == DType::BF16 {
            BF16_TOLERANCE
        } else {
            F16_TOLERANCE
        };
        for (actual, expected) in output.into_iter().zip(reference) {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                "strided packed NVFP4 {dtype:?}: {actual} != {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn nvfp4_shared_activation_rejects_diverging_live_calibration() -> Result<()> {
    use mistralrs_quant::{
        try_forward_with_shared_quantized_activation, Nvfp4ActivationMode, Nvfp4Layer,
        Nvfp4LayerParts, QuantMethod,
    };
    let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    for dtype in [DType::BF16, DType::F16] {
        let n = MERGED_FIRST_N + MERGED_SECOND_N;
        let fixture = Fixture::new(MERGED_ROWS, n, MERGED_K, dtype);
        let input = Tensor::from_vec(fixture.x.clone(), (MERGED_ROWS, MERGED_K), &device)?
            .to_dtype(dtype)?;
        let make_layer = |scale: Tensor| {
            Nvfp4Layer::from_parts(Nvfp4LayerParts {
                weight: Tensor::from_vec(fixture.packed.clone(), (n, MERGED_K / 2), &device)?,
                scales: Tensor::from_vec(
                    fixture.scales.clone(),
                    (n, MERGED_K / BLOCK_SIZE),
                    &device,
                )?,
                global_scales: Tensor::from_vec(fixture.global.clone(), n, &device)?,
                input_scale: Some(scale),
                activation: Nvfp4ActivationMode::DynamicBlock,
                bias: None,
                dtype,
            })
        };
        let shared_scale = Tensor::new(ACTIVATION_GLOBAL, &device)?;
        let independent_scale = Tensor::new(ACTIVATION_GLOBAL, &device)?;
        let first = make_layer(shared_scale.clone())?;
        let same = make_layer(shared_scale.clone())?;
        let independent = make_layer(independent_scale.clone())?;
        assert!(
            try_forward_with_shared_quantized_activation(&input, &[&first, &independent])?
                .is_none()
        );
        independent_scale.reshape(1)?.slice_set(
            &Tensor::new(&[ACTIVATION_GLOBAL * 2.0], &device)?,
            0,
            0,
        )?;
        assert_eq!(
            first.activation_quantization_global_scale(),
            independent.activation_quantization_global_scale()
        );
        assert!(
            try_forward_with_shared_quantized_activation(&input, &[&first, &independent])?
                .is_none()
        );
        for factor in [1.0f32, 2.0, 0.5, 1.0] {
            shared_scale.reshape(1)?.slice_set(
                &Tensor::new(&[ACTIVATION_GLOBAL * factor], &device)?,
                0,
                0,
            )?;
            let outputs = try_forward_with_shared_quantized_activation(&input, &[&first, &same])?
                .expect("one live calibration buffer must remain eligible");
            for (output, method) in outputs.iter().zip([&first, &same]) {
                let expected = method
                    .forward(&input)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let actual = output
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(actual.iter().all(|value| value.is_finite()));
                assert_eq!(actual, expected);
            }
        }
        let separate = independent.forward(&input)?;
        assert!(separate
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite()));
    }
    Ok(())
}
