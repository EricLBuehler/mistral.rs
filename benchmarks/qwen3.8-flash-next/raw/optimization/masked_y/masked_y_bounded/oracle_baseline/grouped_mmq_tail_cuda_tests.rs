#![cfg(feature = "cuda")]

use candle_core::{
    cuda::cudarc::driver::sys::CUdevice_attribute,
    quantized::{GgmlDType, QStorage, QTensor},
    DType, Device, Result, Tensor,
};
use mistralrs_quant::grouped_moe_mmq;

const QUANT_TYPES: [GgmlDType; 4] = [
    GgmlDType::Q4K,
    GgmlDType::Q4_1,
    GgmlDType::Q6K,
    GgmlDType::Q2K,
];
const INPUT_TYPES: [DType; 3] = [DType::BF16, DType::F16, DType::F32];
const ACTIVATION_BLOCK: usize = 128;
const ACTIVATION_MAGNITUDE: f32 = 127.0;
const POSITION_ROW_STRIDE: usize = 37;
const POSITION_BLOCK_STRIDE: usize = 19;
const WEIGHT_PERIOD: usize = 211;
const WEIGHT_STRIDE: usize = 37;
const WEIGHT_MIDPOINT: f32 = 105.0;
const WEIGHT_SCALE: f32 = 0.03;
// Matches gguf::fast_mmq's existing numerical bound; the activation quantization below is exact.
const TOLERANCE: f32 = 5e-3;
const MIN_REFERENCE_SIGNAL: f32 = 0.1;
const STREAM_K_MIN_MAJOR: i32 = 7;
const STREAM_K_MIN_SMS: i32 = 2;
const STREAM_K_OUTPUT_ROWS: usize = 127;
const STREAM_K_HIDDEN: usize = 4096;

static CUDA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Case {
    counts: &'static [usize],
    hidden: usize,
    output_rows: usize,
}

const TAIL_CASES: [Case; 3] = [
    Case {
        counts: &[0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 0],
        hidden: 512,
        output_rows: 129,
    },
    Case {
        counts: &[0, 1, 31, 32, 33, 63, 64, 0],
        hidden: 512,
        output_rows: 129,
    },
    Case {
        counts: &[0, 1, 127, 128, 129, 256, 0],
        hidden: 512,
        output_rows: 129,
    },
];

fn activation_entry(row: usize, block: usize) -> (usize, f32) {
    let offset = (row * POSITION_ROW_STRIDE + block * POSITION_BLOCK_STRIDE) % ACTIVATION_BLOCK;
    let value = if (row + block) % 2 == 0 {
        ACTIVATION_MAGNITUDE
    } else {
        -ACTIVATION_MAGNITUDE
    };
    (block * ACTIVATION_BLOCK + offset, value)
}

fn run_case(case: &Case, device: &Device) -> Result<()> {
    let dev = device.as_cuda_device()?;
    let assignments = case.counts.iter().sum::<usize>();
    let mut bounds = vec![0u32];
    let mut experts = Vec::with_capacity(assignments);
    for (expert, &count) in case.counts.iter().enumerate() {
        experts.extend(std::iter::repeat_n(expert, count));
        bounds.push(u32::try_from(experts.len()).unwrap());
    }
    let destinations = (0..assignments)
        .rev()
        .map(|index| u32::try_from(index).unwrap())
        .collect::<Vec<_>>();
    let sources = destinations
        .iter()
        .map(|&destination| (destination + 1) % u32::try_from(assignments).unwrap())
        .collect::<Vec<_>>();
    let device_bounds = dev.clone_htod(&bounds)?;
    let device_sources = dev.clone_htod(&sources)?;
    let device_destinations = dev.clone_htod(&destinations)?;

    // A single +/-127 per block gives integer Q8 values, unit scales, and exact partial sums in every layout.
    let mut values = vec![0.0f32; assignments * case.hidden];
    for row in 0..assignments {
        for block in 0..case.hidden / ACTIVATION_BLOCK {
            let (column, value) = activation_entry(row, block);
            values[row * case.hidden + column] = value;
        }
    }
    let xs = Tensor::from_vec(values, (assignments, case.hidden), &Device::Cpu)?;
    let shape = (case.counts.len(), case.output_rows, case.hidden);
    let values = (0..shape.0 * shape.1 * shape.2)
        .map(|index| {
            let value = u16::try_from(index * WEIGHT_STRIDE % WEIGHT_PERIOD).unwrap();
            (f32::from(value) / WEIGHT_MIDPOINT - 1.0) * WEIGHT_SCALE
        })
        .collect::<Vec<_>>();
    let weights = Tensor::from_vec(values, shape, &Device::Cpu)?;

    for quant in QUANT_TYPES {
        let cpu_weight = QTensor::quantize(&weights, quant)?;
        let reference_weights = cpu_weight
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let weight = QTensor::new(
            QStorage::from_data(cpu_weight.data()?, device, quant)?,
            shape,
        )?;
        let mut expected = vec![0.0f32; assignments * case.output_rows];
        for (sorted, &expert) in experts.iter().enumerate() {
            let source = sources[sorted] as usize;
            let destination = destinations[sorted] as usize;
            for output in 0..case.output_rows {
                let weight_row = (expert * case.output_rows + output) * case.hidden;
                let value = (0..case.hidden / ACTIVATION_BLOCK)
                    .map(|block| {
                        let (column, activation) = activation_entry(source, block);
                        f64::from(reference_weights[weight_row + column]) * f64::from(activation)
                    })
                    .sum::<f64>();
                expected[destination * case.output_rows + output] = value as f32;
            }
        }
        for row in expected.chunks_exact(case.output_rows) {
            assert!(row.iter().any(|value| value.abs() > MIN_REFERENCE_SIGNAL));
        }

        for input_type in INPUT_TYPES {
            let input = xs.to_dtype(input_type)?.to_device(device)?;
            let actual = grouped_moe_mmq(
                &weight,
                &input,
                &device_sources,
                &device_destinations,
                &device_bounds,
                assignments,
                *case.counts.iter().max().unwrap(),
                case.counts.len(),
                dev,
            )?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            assert_eq!(actual.len(), expected.len());
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    actual.is_finite(),
                    "{quant:?}/{input_type:?}: index {index}"
                );
                let error = (actual - expected).abs() / (1.0 + expected.abs());
                assert!(
                    error <= TOLERANCE,
                    "{quant:?}/{input_type:?}, counts {:?}, index {index}: {actual} vs {expected}, error {error}",
                    case.counts,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn grouped_mmq_sparse_expert_tails_match_cpu_reference() -> Result<()> {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    for case in &TAIL_CASES {
        run_case(case, &device)?;
    }
    Ok(())
}

#[test]
fn grouped_mmq_partial_columns_survive_stream_k_fixup() -> Result<()> {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    let device = Device::new_cuda(0)?;
    let context = device.as_cuda_device()?.cuda_stream().context().clone();
    let major = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(candle_core::Error::wrap)?;
    let multiprocessors = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(candle_core::Error::wrap)?;
    if major < STREAM_K_MIN_MAJOR || multiprocessors < STREAM_K_MIN_SMS {
        eprintln!("stream-K fixup requires Volta or newer with multiple SMs");
        return Ok(());
    }
    // One output tile forces the stream-K launcher to split its K iterations across SMs and run fixup.
    run_case(
        &Case {
            counts: &[3],
            hidden: STREAM_K_HIDDEN,
            output_rows: STREAM_K_OUTPUT_ROWS,
        },
        &device,
    )
}
