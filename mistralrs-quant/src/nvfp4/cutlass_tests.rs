use std::sync::Mutex;

use candle_core::{cuda::cudarc::driver::sys, DType, Device, Result, Tensor};
use float8::F8E4M3;
use half::{bf16, f16};

use super::{Nvfp4Layer, Nvfp4LayerParts};
use crate::{Nvfp4ActivationMode, QuantMethod, QuantizedActivation};

const ROWS: usize = 1024;
const COLUMNS: usize = 4096;
const REDUCTION: usize = 4096;
const FALLBACK_COLUMNS: usize = 1024;
const CONSTITUENT_COLUMNS: usize = COLUMNS / 2;
const BLOCK: usize = 16;
const PACK: usize = 2;
const ODD_OFFSET: usize = 1;
const ACTIVATION_GLOBAL: f32 = 1.5;
const MISMATCHED_GLOBAL: f32 = 3.0;
const GLOBAL_DENOMINATOR: f32 = 1024.0;
const HASH_ROW: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_COLUMN: u64 = 0xbf58_476d_1ce4_e5b9;
const HASH_FINAL: u64 = 0x94d0_49bb_1331_11eb;
const FP4: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
const SCALAR_INTERIOR_COLUMN: usize = 7;
const EXACT_FULL_WEIGHT_MAX_K: usize = 14336;
const EXACT_BOUNDED_WEIGHT_MAX_K: usize = 28672;
const BOUNDED_WEIGHT_CODE: usize = 5;
const REPLAY_SHAPES: [(usize, usize, usize); 16] = [
    (ROWS, COLUMNS, REDUCTION),
    (2, 11008, 4096),
    (3, 16416, 6208),
    (4, 11008, 4096),
    (5, 16416, 6208),
    (8, 11008, 4096),
    (9, 16416, 6208),
    (15, 16416, 6208),
    (16, 32768, 6144),
    (17, 16384, 6144),
    (31, 16384, 6144),
    (32, 16384, 6144),
    (33, 8192, 12288),
    (33, 16416, 6208),
    (63, 8192, 12288),
    (64, 6144, 24576),
];
const REPLAY_FACTORS: [f32; 4] = [1.0, -1.0, 2.0, 1.0];

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn coordinate(row: usize, column: usize) -> usize {
    let mut value = (row as u64).wrapping_mul(HASH_ROW) ^ (column as u64).wrapping_mul(HASH_COLUMN);
    value ^= value >> 30;
    value = value.wrapping_mul(HASH_COLUMN);
    value ^= value >> 27;
    value = value.wrapping_mul(HASH_FINAL);
    (value ^ (value >> 31)) as usize
}

fn rounded(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => bf16::from_f32(value).to_f32(),
        DType::F16 => f16::from_f32(value).to_f32(),
        _ => unreachable!(),
    }
}

fn activation_code(row: usize, column: usize) -> usize {
    if column % BLOCK == 0 {
        FP4.len() / 2 - 1
    } else {
        coordinate(row, column) % FP4.len()
    }
}

fn activation_scale(row: usize, group: usize) -> f32 {
    if coordinate(row, group + 17) % 2 == 0 {
        0.5
    } else {
        1.0
    }
}

fn reduction_scale(group: usize) -> f32 {
    if coordinate(13, group) % 2 == 0 {
        0.5
    } else {
        1.0
    }
}

fn reduction_sign(group: usize) -> f32 {
    if coordinate(19, group) % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

struct WeightRow {
    codes: [usize; PACK],
    scale: f32,
    global: f32,
    bias: f32,
}

struct Fixture {
    dtype: DType,
    rows: usize,
    columns: usize,
    reduction: usize,
    input: Vec<f32>,
    sums: Vec<[f64; PACK]>,
    weight_rows: Vec<WeightRow>,
}

impl Fixture {
    fn new(dtype: DType) -> Self {
        Self::with_shape(dtype, ROWS, COLUMNS, REDUCTION)
    }

    fn with_shape(dtype: DType, rows: usize, columns: usize, reduction: usize) -> Self {
        assert!(reduction <= EXACT_BOUNDED_WEIGHT_MAX_K);
        let weight_code = |column, seed| {
            let code = coordinate(column, seed) % FP4.len();
            if reduction <= EXACT_FULL_WEIGHT_MAX_K {
                code
            } else {
                (code & (FP4.len() / 2)) | (code % (FP4.len() / 2)).min(BOUNDED_WEIGHT_CODE)
            }
        };
        let mut input = Vec::with_capacity(rows * reduction);
        let mut sums = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut sum = [0.0; PACK];
            for column in 0..reduction {
                let group = column / BLOCK;
                let value = FP4[activation_code(row, column)] * activation_scale(row, group);
                let source = value * ACTIVATION_GLOBAL;
                assert_eq!(rounded(source, dtype), source);
                input.push(source);
                sum[column % PACK] +=
                    value as f64 * reduction_scale(group) as f64 * reduction_sign(group) as f64;
            }
            sums.push(sum);
        }
        let global_base = columns.next_power_of_two() as f32;
        let weight_rows = (0..columns)
            .map(|column| WeightRow {
                codes: [weight_code(column, 3), weight_code(column, 7)],
                scale: if coordinate(column, 5) % 2 == 0 {
                    0.5
                } else {
                    1.0
                },
                global: (global_base + column as f32) / (global_base * GLOBAL_DENOMINATOR),
                bias: 0.0,
            })
            .collect();
        let mut fixture = Self {
            dtype,
            rows,
            columns,
            reduction,
            input,
            sums,
            weight_rows,
        };
        for column in 0..columns {
            fixture.weight_rows[column].bias = -rounded(fixture.unrounded(0, column, 1.0), dtype);
        }
        assert!(fixture
            .weight_rows
            .windows(2)
            .all(|rows| rows[0].global != rows[1].global));
        assert!(
            (0..columns).any(|column| {
                let row = &fixture.weight_rows[column];
                rounded(fixture.unrounded(0, column, 1.0) + row.bias, dtype) != 0.0
            }),
            "fixture must distinguish fused bias from a separate output-dtype add"
        );
        fixture.check_scalar_spots();
        fixture
    }

    fn accumulator(&self, row: usize, column: usize) -> f32 {
        let weight = &self.weight_rows[column];
        let exact = (self.sums[row][0] * FP4[weight.codes[0]] as f64
            + self.sums[row][1] * FP4[weight.codes[1]] as f64)
            * weight.scale as f64;
        let accumulator = exact as f32;
        assert_eq!(accumulator as f64, exact);
        accumulator
    }

    fn unrounded(&self, row: usize, column: usize, factor: f32) -> f32 {
        (self.accumulator(row, column) * factor * self.weight_rows[column].global)
            * ACTIVATION_GLOBAL
    }

    fn expected(&self, row: usize, column: usize, factor: f32, biased: bool) -> f32 {
        let output = rounded(self.unrounded(row, column, factor), self.dtype);
        if biased {
            rounded(output + self.weight_rows[column].bias, self.dtype)
        } else {
            output
        }
    }

    fn check_scalar_spots(&self) {
        for row in [0, 1, self.rows / 2, self.rows - 1] {
            for column in [
                0,
                1,
                SCALAR_INTERIOR_COLUMN,
                self.columns / 2,
                self.columns - 1,
            ] {
                let weight = &self.weight_rows[column];
                let mut direct = 0.0f64;
                for reduction in 0..self.reduction {
                    let group = reduction / BLOCK;
                    let x = FP4[activation_code(row, reduction)] as f64
                        * activation_scale(row, group) as f64;
                    let w = FP4[weight.codes[reduction % PACK]] as f64
                        * reduction_sign(group) as f64
                        * (weight.scale * reduction_scale(group)) as f64;
                    direct += x * w;
                }
                assert_eq!(direct, self.accumulator(row, column) as f64);
            }
        }
    }

    fn source(&self, device: &Device) -> Result<Tensor> {
        Tensor::from_vec(self.input.clone(), (self.rows, self.reduction), device)?
            .to_dtype(self.dtype)
    }

    fn layer(
        &self,
        device: &Device,
        start: usize,
        columns: usize,
        biased: bool,
        odd_weight: bool,
        calibration: f32,
    ) -> Result<Nvfp4Layer> {
        let mut packed =
            Vec::with_capacity(columns * self.reduction / PACK + usize::from(odd_weight));
        if odd_weight {
            packed.push(0u8);
        }
        for column in start..start + columns {
            let codes = self.weight_rows[column].codes;
            for pair in 0..self.reduction / PACK {
                let sign = if reduction_sign(pair * PACK / BLOCK) < 0.0 {
                    0x88
                } else {
                    0
                };
                packed.push((codes[0] | (codes[1] << 4)) as u8 ^ sign);
            }
        }
        let packed_len = packed.len();
        let flat = Tensor::from_vec(packed, packed_len, device)?;
        let weight = flat
            .narrow(0, usize::from(odd_weight), columns * self.reduction / PACK)?
            .reshape((columns, self.reduction / PACK))?;
        let scales = (start..start + columns)
            .flat_map(|column| {
                (0..self.reduction / BLOCK).map(move |group| {
                    F8E4M3::from_f32(self.weight_rows[column].scale * reduction_scale(group))
                })
            })
            .collect::<Vec<_>>();
        let global = self.weight_rows[start..start + columns]
            .iter()
            .map(|row| row.global)
            .collect::<Vec<_>>();
        let bias = if biased {
            Some(
                Tensor::from_vec(
                    self.weight_rows[start..start + columns]
                        .iter()
                        .map(|row| row.bias)
                        .collect::<Vec<_>>(),
                    columns,
                    device,
                )?
                .to_dtype(self.dtype)?,
            )
        } else {
            None
        };
        Nvfp4Layer::from_parts(Nvfp4LayerParts {
            weight,
            scales: Tensor::from_vec(scales, (columns, self.reduction / BLOCK), device)?,
            global_scales: Tensor::from_vec(global, columns, device)?,
            input_scale: Some(Tensor::new(calibration, device)?),
            activation: Nvfp4ActivationMode::DynamicBlock,
            bias,
            dtype: self.dtype,
        })
    }

    fn check(
        &self,
        output: &Tensor,
        start: usize,
        columns: usize,
        factor: f32,
        biased: bool,
    ) -> Result<()> {
        assert_eq!(output.dtype(), self.dtype);
        assert_eq!(output.elem_count(), self.rows * columns);
        let actual = output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (index, actual) in actual.into_iter().enumerate() {
            let row = index / columns;
            let column = start + index % columns;
            let expected = self.expected(row, column, factor, biased);
            assert!(actual.is_finite() && expected.is_finite());
            assert_eq!(
                actual, expected,
                "dtype={:?} row={row} column={column} factor={factor}",
                self.dtype
            );
        }
        Ok(())
    }
}

fn sm121_device() -> Result<Option<Device>> {
    let device = Device::new_cuda(0)?;
    let supported = crate::cutile::device_compute_capability(device.as_cuda_device()?) == (12, 1);
    Ok(supported.then_some(device))
}

fn odd_packed_view(activation: &QuantizedActivation) -> Result<Tensor> {
    let packed = activation.quantized();
    let prefix = Tensor::zeros(ODD_OFFSET, DType::U8, packed.device())?;
    Tensor::cat(&[&prefix, &packed.flatten_all()?], 0)?
        .narrow(0, ODD_OFFSET, packed.elem_count())?
        .reshape(packed.shape())
}

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn native_loading_mutable_shared_scales_and_post_cast_bias_match_oracle() -> Result<()> {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    let Some(device) = sm121_device()? else {
        return Ok(());
    };
    for dtype in [DType::BF16, DType::F16] {
        let fixture = Fixture::new(dtype);
        let source = fixture.source(&device)?;
        let strided = source.t()?.contiguous()?.t()?.unsqueeze(0)?;
        assert!(!strided.is_contiguous());
        let layer = fixture.layer(&device, 0, COLUMNS, true, true, ACTIVATION_GLOBAL)?;
        assert_eq!(
            layer.parts.weight.storage_and_layout().1.start_offset(),
            ODD_OFFSET
        );
        assert!(layer.native.as_ref().unwrap().supports(ROWS, dtype));
        let direct = layer.forward(&strided)?;
        assert_eq!(direct.dims(), &[1, ROWS, COLUMNS]);
        fixture.check(&direct, 0, COLUMNS, 1.0, true)?;

        let shared = layer.quantize_activation(&strided)?;
        let cloned = shared.clone();
        let fallback =
            fixture.layer(&device, 0, FALLBACK_COLUMNS, true, false, ACTIVATION_GLOBAL)?;
        assert!(fallback.native.is_none());
        fixture.check(
            &fallback.forward_quantized(&shared)?,
            0,
            FALLBACK_COLUMNS,
            1.0,
            true,
        )?;
        fixture.check(&layer.forward_quantized(&shared)?, 0, COLUMNS, 1.0, true)?;
        fixture.check(&layer.forward_quantized(&cloned)?, 0, COLUMNS, 1.0, true)?;

        let original_scales = shared.scales().copy()?;
        cloned.scales().zero_set()?;
        fixture.check(&layer.forward_quantized(&shared)?, 0, COLUMNS, 0.0, true)?;
        fixture.check(
            &fallback.forward_quantized(&cloned)?,
            0,
            FALLBACK_COLUMNS,
            0.0,
            true,
        )?;
        shared.scales().slice_set(&original_scales, 0, 0)?;
        fixture.check(&layer.forward_quantized(&cloned)?, 0, COLUMNS, 1.0, true)?;

        let odd = odd_packed_view(&shared)?;
        assert_eq!(odd.storage_and_layout().1.start_offset(), ODD_OFFSET);
        let odd = QuantizedActivation::new_nvfp4(
            odd,
            shared.scales().clone(),
            &strided,
            ACTIVATION_GLOBAL,
        )?;
        fixture.check(&layer.forward_quantized(&odd)?, 0, COLUMNS, 1.0, true)?;
        device.synchronize()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn native_shared_activation_created_inside_graph_replays_fresh_scales() -> Result<()> {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    let Some(device) = sm121_device()? else {
        return Ok(());
    };
    for (rows, columns, reduction) in REPLAY_SHAPES {
        for dtype in [DType::BF16, DType::F16] {
            let fixture = Fixture::with_shape(dtype, rows, columns, reduction);
            let source = fixture.source(&device)?;
            let negative = source.neg()?;
            let doubled = (&source * 2.0)?;
            let input = source.zeros_like()?;
            let layer = fixture.layer(&device, 0, columns, true, true, ACTIVATION_GLOBAL)?;
            assert!(layer.native.as_ref().unwrap().supports(rows, dtype));
            let cuda = device.as_cuda_device()?;
            let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
            input.slice_set(&source, 0, 0)?;
            let warmed = layer.quantize_activation(&input)?;
            let _ = layer.forward_quantized(&warmed)?;
            let _ = layer.forward_quantized(&warmed.clone())?;
            let _ = layer.forward(&input)?;
            device.synchronize()?;
            let compiled = cutile::tile_kernel::jit_compile_count();
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
            let captured = (|| -> Result<_> {
                let activation = layer.quantize_activation(&input)?;
                let cloned = activation.clone();
                let direct = layer.forward(&input)?;
                let shared = layer.forward_quantized(&activation)?;
                let shared_clone = layer.forward_quantized(&cloned)?;
                Ok((direct, shared, shared_clone, activation, cloned))
            })();
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            let (direct, shared, shared_clone, activation, cloned) = captured?;
            let graph = graph
                .map_err(candle_core::Error::msg)?
                .ok_or_else(|| candle_core::Error::msg("native NVFP4 capture produced no graph"))?;
            assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
            for factor in REPLAY_FACTORS {
                let current = if factor < 0.0 {
                    &negative
                } else if factor == 2.0 {
                    &doubled
                } else {
                    &source
                };
                input.slice_set(current, 0, 0)?;
                graph.launch().map_err(candle_core::Error::msg)?;
                device.synchronize()?;
                for output in [&direct, &shared, &shared_clone] {
                    fixture.check(output, 0, columns, factor, true)?;
                }
                assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
            }
            drop(graph);
            drop((direct, shared, shared_clone, activation, cloned));
            input.slice_set(&source, 0, 0)?;
            fixture.check(&layer.forward(&input)?, 0, columns, 1.0, true)?;
            device.synchronize()?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn merged_rows_select_native_and_preserve_constituent_calibration() -> Result<()> {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    let Some(device) = sm121_device()? else {
        return Ok(());
    };
    for dtype in [DType::BF16, DType::F16] {
        let fixture = Fixture::new(dtype);
        let source = fixture.source(&device)?;
        let first = fixture.layer(
            &device,
            0,
            CONSTITUENT_COLUMNS,
            false,
            false,
            ACTIVATION_GLOBAL,
        )?;
        let second = fixture.layer(
            &device,
            CONSTITUENT_COLUMNS,
            CONSTITUENT_COLUMNS,
            false,
            false,
            ACTIVATION_GLOBAL,
        )?;
        let second = Nvfp4Layer::from_parts(Nvfp4LayerParts {
            input_scale: first.parts.input_scale.clone(),
            ..second.parts
        })?;
        assert!(first.native.is_none() && second.native.is_none());
        let activation = first.quantize_activation(&source)?;
        assert_eq!(activation.quantized().dims2()?, (ROWS, REDUCTION / PACK));
        let packed =
            Nvfp4Layer::merge(vec![first, second])?.expect("compatible bias-free rows must merge");
        assert_eq!(packed.rows_per_rank, vec![CONSTITUENT_COLUMNS; PACK]);
        let output = packed.packed.forward_quantized(&activation)?;
        assert_eq!(output.dims(), &[ROWS, COLUMNS]);
        fixture.check(&output, 0, COLUMNS, 1.0, false)?;
        fixture.check(&packed.packed.forward(&source)?, 0, COLUMNS, 1.0, false)?;
        for (index, constituent) in packed.constituents.iter().enumerate() {
            assert_eq!(
                constituent.activation_quantization_global_scale(),
                Some(ACTIVATION_GLOBAL)
            );
            let separate = constituent.forward_quantized(&activation)?;
            fixture.check(
                &separate,
                index * CONSTITUENT_COLUMNS,
                CONSTITUENT_COLUMNS,
                1.0,
                false,
            )?;
            let merged_slice =
                output.narrow(1, index * CONSTITUENT_COLUMNS, CONSTITUENT_COLUMNS)?;
            assert_eq!(
                separate
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                merged_slice
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            );
        }
        let first = fixture.layer(
            &device,
            0,
            CONSTITUENT_COLUMNS,
            false,
            false,
            ACTIVATION_GLOBAL,
        )?;
        let mismatch = fixture.layer(
            &device,
            CONSTITUENT_COLUMNS,
            CONSTITUENT_COLUMNS,
            false,
            false,
            MISMATCHED_GLOBAL,
        )?;
        assert!(Nvfp4Layer::merge(vec![first, mismatch])?.is_none());
        let biased = fixture.layer(
            &device,
            0,
            CONSTITUENT_COLUMNS,
            true,
            false,
            ACTIVATION_GLOBAL,
        )?;
        let second = fixture.layer(
            &device,
            CONSTITUENT_COLUMNS,
            CONSTITUENT_COLUMNS,
            false,
            false,
            ACTIVATION_GLOBAL,
        )?;
        assert!(Nvfp4Layer::merge(vec![biased, second])?.is_none());
        device.synchronize()?;
    }
    Ok(())
}
