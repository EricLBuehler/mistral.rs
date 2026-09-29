use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;

use super::{Nvfp4Layer, Nvfp4LayerParts};
use crate::{Nvfp4ActivationMode, QuantMethod};

const ROW_COUNTS: [usize; 2] = [33, 3];
const COLUMNS: usize = 16384;
const REDUCTION: usize = 6144;
const BLOCK: usize = 16;
const PACK: usize = 2;
const SCALE_COLUMNS: usize = REDUCTION / BLOCK;
const ACTIVATION_GLOBAL: f32 = 1.3125;
const SCALE_FIRST_BITS: u8 = 0x18;
const SCALE_VALUES: u8 = 40;
const SOURCE_MODULUS: u64 = 1_000_003;
const SOURCE_DENOMINATOR: f32 = 285_715.0;
const SOURCE_SHIFT: f32 = 1.75;
const WEIGHT_GLOBAL_BASE: f32 = 0.5;
const INPUT_SEED: u64 = 0x243f_6a88_85a3_08d3;
const WEIGHT_SEED: u64 = 0x1319_8a2e_0370_7344;
const SCALE_SEED: u64 = 0xa409_3822_299f_31d0;
const HASH_FIRST: u64 = 0xbf58_476d_1ce4_e5b9;
const HASH_SECOND: u64 = 0x94d0_49bb_1331_11eb;
const FP32_UNIT: f64 = 1.0 / 16_777_216.0;
const FP64_UNIT: f64 = 1.0 / 9_007_199_254_740_992.0;
const BF16_UNIT: f64 = 1.0 / 256.0;
const F16_UNIT: f64 = 1.0 / 2048.0;
const BF16_HALF_SUBNORMAL: f64 = 4.591_774_807_899_561e-41;
const F16_HALF_SUBNORMAL: f64 = 2.980_232_238_769_531_3e-8;
const EPILOGUE_MULTIPLIES: usize = 2;
const HOST_BOUND_OPERATIONS: usize = REDUCTION + 8;
const SAMPLE_COLUMNS: [usize; 8] = [0, 1, 31, 127, 128, 4093, COLUMNS / 2 + 1, COLUMNS - 1];
const FP4: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
const FP4_MAX: f64 = 6.0;

fn hash(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(HASH_FIRST);
    value ^= value >> 27;
    value = value.wrapping_mul(HASH_SECOND);
    value ^ (value >> 31)
}

fn gamma(operations: usize, unit: f64) -> f64 {
    let product = operations as f64 * unit;
    assert!(product < 1.0);
    product / (1.0 - product)
}

fn upward(value: f64) -> f64 {
    assert!(value.is_finite() && value >= 0.0);
    value.next_up()
}

fn host_upper_bound(value: f64) -> f64 {
    upward(value / (1.0 - gamma(HOST_BOUND_OPERATIONS, FP64_UNIT)))
}

fn fp4_value(packed: &[u8], index: usize) -> f64 {
    let byte = packed[index / PACK];
    let code = if index.is_multiple_of(PACK) {
        byte & 0xf
    } else {
        byte >> 4
    };
    FP4[code as usize]
}

struct HostOperands {
    dtype: DType,
    activations: Vec<u8>,
    activation_scales: Vec<F8E4M3>,
    weight_scales: Vec<F8E4M3>,
    weight_globals: Vec<f32>,
    activation_global: f32,
}

impl HostOperands {
    fn activation(&self, row: usize, column: usize) -> f64 {
        fp4_value(&self.activations, row * REDUCTION + column)
            * self.activation_scales[row * SCALE_COLUMNS + column / BLOCK].to_f32() as f64
    }

    fn scaled_absolute_sum(&self, sum: f64, column: usize) -> f64 {
        host_upper_bound((sum * self.weight_globals[column] as f64) * self.activation_global as f64)
    }

    fn accumulation_radius(&self, scaled_absolute_sum: f64) -> f64 {
        upward(gamma(REDUCTION + EPILOGUE_MULTIPLIES, FP32_UNIT) * scaled_absolute_sum)
    }

    fn rounding_radius(&self, output: f32) -> f64 {
        assert!(output.is_finite());
        let (unit, half_subnormal) = match self.dtype {
            DType::BF16 => (BF16_UNIT, BF16_HALF_SUBNORMAL),
            DType::F16 => (F16_UNIT, F16_HALF_SUBNORMAL),
            _ => unreachable!(),
        };
        let numerator = upward(upward(unit * (output as f64).abs()) + half_subnormal);
        upward(numerator / (1.0 - unit))
    }

    fn reference(&self, packed_weights: &[u8], row: usize, column: usize) -> (f64, f64) {
        let mut dot = 0.0;
        let mut sum_absolute = 0.0;
        for reduction in 0..REDUCTION {
            let weight = fp4_value(packed_weights, reduction)
                * self.weight_scales[column * SCALE_COLUMNS + reduction / BLOCK].to_f32() as f64;
            let product = self.activation(row, reduction) * weight;
            dot += product;
            sum_absolute += product.abs();
        }
        (
            (dot * self.weight_globals[column] as f64) * self.activation_global as f64,
            self.scaled_absolute_sum(sum_absolute, column),
        )
    }
}

fn random_layer(device: &Device, dtype: DType) -> Result<Nvfp4Layer> {
    let mut codes_seen = 0u16;
    let packed: Vec<u8> = (0..COLUMNS * REDUCTION / PACK)
        .map(|index| {
            let byte = hash(index as u64 ^ WEIGHT_SEED) as u8;
            codes_seen |= 1 << (byte & 0xf);
            codes_seen |= 1 << (byte >> 4);
            byte
        })
        .collect();
    assert_eq!(codes_seen, u16::MAX);
    let scales: Vec<F8E4M3> = (0..COLUMNS * SCALE_COLUMNS)
        .map(|index| {
            F8E4M3::from_bits(
                SCALE_FIRST_BITS + (hash(index as u64 ^ SCALE_SEED) % SCALE_VALUES as u64) as u8,
            )
        })
        .collect();
    let globals: Vec<f32> = (0..COLUMNS)
        .map(|column| WEIGHT_GLOBAL_BASE + (column + 1) as f32 / (COLUMNS + 1) as f32)
        .collect();
    assert!(globals.windows(2).all(|pair| pair[0] != pair[1]));
    Nvfp4Layer::from_parts(Nvfp4LayerParts {
        weight: Tensor::from_vec(packed, (COLUMNS, REDUCTION / PACK), device)?,
        scales: Tensor::from_vec(scales, (COLUMNS, SCALE_COLUMNS), device)?,
        global_scales: Tensor::from_vec(globals, COLUMNS, device)?,
        input_scale: Some(Tensor::new(ACTIVATION_GLOBAL, device)?),
        activation: Nvfp4ActivationMode::DynamicBlock,
        bias: None,
        dtype,
    })
}

#[test]
#[ignore = "requires an SM121 CUDA device; run CUDA tests with one test thread"]
fn native_decode_arbitrary_data_matches_cutile_and_sampled_fp64_bounds() -> Result<()> {
    let device = Device::new_cuda(0)?;
    if crate::cutile::device_compute_capability(device.as_cuda_device()?) != (12, 1) {
        return Ok(());
    }
    for dtype in [DType::BF16, DType::F16] {
        let layer = random_layer(&device, dtype)?;
        for rows in ROW_COUNTS {
            let mut sample_rows = vec![0, 1, rows / 2, rows - 1];
            sample_rows.dedup();
            let input: Vec<f32> = (0..rows * REDUCTION)
                .map(|index| {
                    (hash(index as u64 ^ INPUT_SEED) % SOURCE_MODULUS) as f32 / SOURCE_DENOMINATOR
                        - SOURCE_SHIFT
                })
                .collect();
            let input = Tensor::from_vec(input, (rows, REDUCTION), &device)?.to_dtype(dtype)?;
            assert!(layer.native.as_ref().unwrap().supports(rows, dtype));
            let activation = layer.quantize_activation(&input)?;
            let native = layer.forward_quantized(&activation)?;
            let control = crate::cutile::cutile_nvfp4_prequantized(
                activation.quantized(),
                activation.scales(),
                dtype,
                layer.gemm_args(),
            )?;
            assert_eq!(native.dtype(), dtype);
            assert_eq!(control.dtype(), dtype);
            assert_eq!(native.dims(), &[rows, COLUMNS]);
            assert_eq!(control.dims(), &[rows, COLUMNS]);
            let native = native
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let control = control
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let host = HostOperands {
                dtype,
                activations: activation.quantized().flatten_all()?.to_vec1::<u8>()?,
                activation_scales: activation.scales().flatten_all()?.to_vec1::<F8E4M3>()?,
                weight_scales: layer.parts.scales.flatten_all()?.to_vec1::<F8E4M3>()?,
                weight_globals: layer.parts.global_scales.to_vec1::<f32>()?,
                activation_global: layer
                    .parts
                    .input_scale
                    .as_ref()
                    .unwrap()
                    .to_scalar::<f32>()?,
            };
            assert!(host.activation_global.is_finite() && host.activation_global > 0.0);
            assert!(host
                .weight_globals
                .iter()
                .all(|value| value.is_finite() && *value > 0.0));
            assert!(host.activation_scales.iter().all(|value| {
                let value = value.to_f32();
                value.is_finite() && value >= 0.0
            }));
            let mut scale_values_seen = 0u64;
            for scale in &host.weight_scales {
                let bits = scale.to_bits();
                assert!((SCALE_FIRST_BITS..SCALE_FIRST_BITS + SCALE_VALUES).contains(&bits));
                scale_values_seen |= 1u64 << (bits - SCALE_FIRST_BITS);
            }
            assert_eq!(scale_values_seen, (1u64 << SCALE_VALUES) - 1);
            let activation_l1: Vec<f64> = (0..rows)
                .map(|row| {
                    (0..REDUCTION)
                        .map(|column| host.activation(row, column).abs())
                        .sum()
                })
                .collect();
            assert!(activation_l1
                .iter()
                .all(|value| value.is_finite() && *value > 0.0));
            let weight_max: Vec<f64> = host
                .weight_scales
                .as_chunks::<SCALE_COLUMNS>()
                .0
                .iter()
                .map(|scales| {
                    scales
                        .iter()
                        .map(|scale| FP4_MAX * scale.to_f32() as f64)
                        .fold(0.0, f64::max)
                })
                .collect();
            let mut maximum_gap = 0.0f64;
            let mut maximum_bound_fraction = 0.0f64;
            for (index, (&actual, &accepted)) in native.iter().zip(&control).enumerate() {
                assert!(
                    actual.is_finite() && accepted.is_finite(),
                    "dtype={dtype:?} index={index}"
                );
                let row = index / COLUMNS;
                let column = index % COLUMNS;
                let sum_bound =
                    host.scaled_absolute_sum(activation_l1[row] * weight_max[column], column);
                let bound = upward(
                    upward(
                        2.0 * host.accumulation_radius(sum_bound) + host.rounding_radius(actual),
                    ) + host.rounding_radius(accepted),
                );
                let gap = (actual as f64 - accepted as f64).abs();
                assert!(gap <= bound,
                "dtype={dtype:?} row={row} column={column} native={actual} cutile={accepted} gap={gap} bound={bound}");
                maximum_gap = maximum_gap.max(gap);
                maximum_bound_fraction = maximum_bound_fraction.max(gap / bound);
            }
            let mut maximum_reference_gap = 0.0f64;
            for column in SAMPLE_COLUMNS {
                // Compact the row before readback; a view readback copies its entire backing storage.
                let weights = layer
                    .parts
                    .weight
                    .narrow(0, column, 1)?
                    .force_contiguous()?
                    .flatten_all()?
                    .to_vec1::<u8>()?;
                for &row in &sample_rows {
                    let (reference, sum_absolute) = host.reference(&weights, row, column);
                    assert!(reference.is_finite());
                    let fp64_radius =
                        upward(gamma(HOST_BOUND_OPERATIONS, FP64_UNIT) * sum_absolute);
                    for (name, output) in [("native", &native), ("cutile", &control)] {
                        let actual = output[row * COLUMNS + column];
                        let bound = upward(
                            upward(
                                host.accumulation_radius(sum_absolute)
                                    + host.rounding_radius(actual),
                            ) + fp64_radius,
                        );
                        let gap = (actual as f64 - reference).abs();
                        assert!(gap <= bound,
                        "{name} dtype={dtype:?} row={row} column={column} actual={actual} fp64={reference} gap={gap} bound={bound}");
                        maximum_reference_gap = maximum_reference_gap.max(gap);
                    }
                }
            }
            eprintln!(
            "NVFP4 arbitrary {dtype:?} rows={rows}: outputs={} fp64_samples={} max_native_cutile_gap={maximum_gap} max_pair_bound_fraction={maximum_bound_fraction} max_fp64_gap={maximum_reference_gap}",
            rows * COLUMNS, sample_rows.len() * SAMPLE_COLUMNS.len()
        );
            device.synchronize()?;
        }
    }
    Ok(())
}
