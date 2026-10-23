#![cfg(feature = "cuda")]

use candle_core::{
    quantized::{GgmlDType, QStorage, QTensor},
    DType, Device, Result, Tensor,
};
use mistralrs_quant::{grouped_moe_mmq, grouped_moe_mmq_from_glu_pair};

const QUANT_TYPES: [GgmlDType; 5] = [
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_1,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
];
const INPUT_TYPES: [DType; 3] = [DType::BF16, DType::F16, DType::F32];
const MAGNITUDES: [f32; 9] = [
    1.0, 2047.0, 2048.0, 8192.0, -1.0, -2047.0, -2048.0, -8192.0, 60000.0,
];
const ROWS: usize = 17;
const HIDDEN: usize = 512;
const OUTPUT_ROWS: usize = 65;
const WEIGHT_PERIOD: usize = 37;
const WEIGHT_MINIMUM: f32 = -0.5;
const WEIGHT_SPAN: f32 = 0.25;
const GLU_GATE: f32 = 128.0;
const SILU: i32 = 0;
const TOLERANCE: f32 = 5e-3;

fn check_output(actual: Tensor, expected: &[f32], quant: GgmlDType, path: &str) -> Result<()> {
    let actual = actual
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "{quant:?}/{path}: nonfinite output {index}"
        );
        let error = (actual - expected).abs() / (1.0 + expected.abs());
        assert!(
            error <= TOLERANCE,
            "{quant:?}/{path}, index {index}: {actual} vs {expected}, error {error}"
        );
    }
    Ok(())
}

fn reference(weights: &[f32], inputs: &Tensor) -> Result<Vec<f32>> {
    let inputs = inputs
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut expected = vec![0.0; ROWS * OUTPUT_ROWS];
    for row in 0..ROWS {
        for output in 0..OUTPUT_ROWS {
            expected[row * OUTPUT_ROWS + output] = (0..HIDDEN)
                .map(|column| {
                    f64::from(inputs[row * HIDDEN + column])
                        * f64::from(weights[output * HIDDEN + column])
                })
                .sum::<f64>() as f32;
        }
    }
    Ok(expected)
}

#[test]
fn grouped_mmq_large_partial_sums_match_cpu_reference() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dev = device.as_cuda_device()?;
    let ids = dev.clone_htod(&(0..ROWS as u32).collect::<Vec<_>>())?;
    let bounds = dev.clone_htod(&[0u32, ROWS as u32])?;
    // The 2047/2048 rows straddle the largest finite half sum of a 32-value block.
    let input = Tensor::from_vec(
        (0..ROWS)
            .flat_map(|row| std::iter::repeat_n(MAGNITUDES[row % MAGNITUDES.len()], HIDDEN))
            .collect(),
        (ROWS, HIDDEN),
        &Device::Cpu,
    )?;
    let weights = Tensor::from_vec(
        (0..OUTPUT_ROWS * HIDDEN)
            .map(|index| {
                WEIGHT_MINIMUM
                    + WEIGHT_SPAN * (index % WEIGHT_PERIOD) as f32 / (WEIGHT_PERIOD - 1) as f32
            })
            .collect(),
        (1, OUTPUT_ROWS, HIDDEN),
        &Device::Cpu,
    )?;
    let gate = Tensor::full(GLU_GATE, (ROWS, HIDDEN), &device)?;
    let up = input
        .affine(f64::from(GLU_GATE).recip(), 0.0)?
        .to_device(&device)?;

    for quant in QUANT_TYPES {
        let cpu_weight = QTensor::quantize(&weights, quant)?;
        let dequantized = cpu_weight
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let weight = QTensor::new(
            QStorage::from_data(cpu_weight.data()?, &device, quant)?,
            (1, OUTPUT_ROWS, HIDDEN),
        )?;
        for dtype in INPUT_TYPES {
            let typed_input = input.to_dtype(dtype)?;
            let expected = reference(&dequantized, &typed_input)?;
            let output = grouped_moe_mmq(
                &weight,
                &typed_input.to_device(&device)?,
                &ids,
                &ids,
                &bounds,
                ROWS,
                ROWS,
                1,
                dev,
            )?;
            check_output(output, &expected, quant, &format!("{dtype:?}"))?;
        }
        let output = grouped_moe_mmq_from_glu_pair(
            &weight, &gate, &up, &ids, &ids, &bounds, ROWS, ROWS, 1, SILU, dev,
        )?;
        check_output(
            output,
            &reference(&dequantized, &input)?,
            quant,
            "fused GLU",
        )?;
    }
    Ok(())
}
