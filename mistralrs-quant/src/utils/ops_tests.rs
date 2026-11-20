fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tolerance,
            "mismatch at {i}: actual={a}, expected={e}, diff={diff}"
        );
    }
}

#[test]
fn test_fused_split_glu_cpu_matches_pair() {
    use super::{fused_glu, fused_split_glu, GluActivationType};
    use candle_core::{Device, Tensor};

    const ROWS: usize = 3;
    const SPLIT: usize = 7;

    let device = Device::Cpu;
    let gate_data = (0..ROWS * SPLIT)
        .map(|index| (index as f32 - 9.0) / 7.0)
        .collect::<Vec<_>>();
    let up_data = (0..ROWS * SPLIT)
        .map(|index| (13.0 - index as f32) / 11.0)
        .collect::<Vec<_>>();
    let gate = Tensor::from_vec(gate_data, (ROWS, SPLIT), &device).unwrap();
    let up = Tensor::from_vec(up_data, (ROWS, SPLIT), &device).unwrap();
    let packed = Tensor::cat(&[&gate, &up], 1).unwrap();

    for activation in [
        GluActivationType::Silu,
        GluActivationType::Gelu,
        GluActivationType::Relu,
        GluActivationType::GeluErf,
        GluActivationType::Sigmoid,
    ] {
        let expected = fused_glu(&gate, &up, activation)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = fused_split_glu(&packed, SPLIT, activation)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close(&actual, &expected, 1e-6);
    }
}

#[test]
fn test_fused_sigmoid_glu_matches_candle_extremes() {
    use super::{fused_glu, GluActivationType};
    use candle_core::{Device, Tensor};

    let gate = Tensor::new(
        &[
            [f32::NEG_INFINITY, -100.0, -20.0],
            [-2.0, 0.0, 2.0],
            [20.0, 100.0, f32::INFINITY],
        ],
        &Device::Cpu,
    )
    .unwrap();
    let up = Tensor::new(
        &[[1.5f32, -2.0, 0.25], [3.0, -4.0, 5.0], [-0.5, 2.5, -3.0]],
        &Device::Cpu,
    )
    .unwrap();
    let expected = candle_nn::ops::sigmoid(&gate)
        .unwrap()
        .broadcast_mul(&up)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let actual = fused_glu(&gate, &up, GluActivationType::Sigmoid)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_close(&actual, &expected, 1e-6);
    assert!(actual.iter().all(|value| value.is_finite()));
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_split_glu_cuda_scalar_and_vector_paths() {
    use super::{fused_glu, fused_split_glu, GluActivationType};
    use candle_core::{DType, Device, Tensor};

    const ROWS: usize = 3;

    let cpu = Device::Cpu;
    let cuda = Device::new_cuda(0).unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for split in [7, 8] {
            let gate_data = (0..ROWS * split)
                .map(|index| (index as f32 - 10.0) / 8.0)
                .collect::<Vec<_>>();
            let up_data = (0..ROWS * split)
                .map(|index| (15.0 - index as f32) / 12.0)
                .collect::<Vec<_>>();
            let gate = Tensor::from_vec(gate_data, (ROWS, split), &cuda)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let up = Tensor::from_vec(up_data, (ROWS, split), &cuda)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let packed = Tensor::cat(&[&gate, &up], 1).unwrap();
            let tolerance = if dtype == DType::F32 { 1e-6 } else { 2e-2 };

            for activation in [
                GluActivationType::Silu,
                GluActivationType::Gelu,
                GluActivationType::Relu,
                GluActivationType::GeluErf,
                GluActivationType::Sigmoid,
            ] {
                let expected = fused_glu(&gate, &up, activation)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_device(&cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let actual = fused_split_glu(&packed, split, activation)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_device(&cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert_close(&actual, &expected, tolerance);
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_split_glu_fp8_quantization_cuda() {
    use super::{fused_split_glu, fused_split_glu_quantized_bf16, GluActivationType};
    use candle_core::{DType, Device, Tensor};
    use float8::F8E4M3;

    const ROWS: usize = 5;
    const SPLIT: usize = 256;
    const GROUP_SIZE: usize = 128;
    const SCALE_STRIDE: usize = 8;
    const PACKED: usize = SPLIT * 2;
    const SCALE_TOLERANCE: f32 = 1.0e-6;
    const QUANTIZATION_TOLERANCE: f32 = 0.08;

    let cuda = Device::new_cuda(0).unwrap();
    let values = (0..ROWS * PACKED + 1)
        .map(|index| {
            let value = ((index * 17 + 11) % 97) as f32 - 48.0;
            value / 13.0
        })
        .collect::<Vec<_>>();
    let storage = Tensor::from_vec(values, ROWS * PACKED + 1, &cuda)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let input = storage
        .narrow(0, 1, ROWS * PACKED)
        .unwrap()
        .reshape((ROWS, PACKED))
        .unwrap();
    assert!(input.is_contiguous());
    assert_eq!(input.layout().start_offset(), 1);

    for activation in [
        GluActivationType::Silu,
        GluActivationType::Gelu,
        GluActivationType::Relu,
        GluActivationType::GeluErf,
        GluActivationType::Sigmoid,
    ] {
        let expected = fused_split_glu(&input, SPLIT, activation)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (quantized, scales) =
            fused_split_glu_quantized_bf16(&input, SPLIT, GROUP_SIZE, SCALE_STRIDE, activation)
                .unwrap();
        let quantized = quantized
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<F8E4M3>()
            .unwrap();
        let scales = scales
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        for row in 0..ROWS {
            for group in 0..SPLIT / GROUP_SIZE {
                let start = row * SPLIT + group * GROUP_SIZE;
                let end = start + GROUP_SIZE;
                let maximum = expected[start..end]
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .fold(0.0f32, f32::max);
                let expected_scale = if maximum == 0.0 { 1.0 } else { maximum / 448.0 };
                assert!(
                    (scales[group][row] - expected_scale).abs() <= SCALE_TOLERANCE,
                    "scale mismatch at row {row}, group {group}"
                );
                for column in group * GROUP_SIZE..(group + 1) * GROUP_SIZE {
                    let index = row * SPLIT + column;
                    let actual = quantized[index].to_f32() * scales[group][row];
                    let tolerance =
                        QUANTIZATION_TOLERANCE * expected[index].abs().max(maximum / 64.0);
                    assert!(
                            (actual - expected[index]).abs() <= tolerance,
                            "value mismatch at row {row}, column {column}: actual={actual}, expected={}",
                            expected[index]
                        );
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_glu_fp8_quantization_preserves_bf16_rounding_and_row_strides() {
    use super::{fused_glu, fused_glu_quantized_bf16, GluActivationType};
    use candle_core::{DType, Device, Tensor};
    use float8::F8E4M3;

    const ROWS: usize = 5;
    const COLUMNS: usize = 256;
    const GROUP_SIZE: usize = 128;
    const SCALE_STRIDE: usize = 8;
    const GATE_WIDTH: usize = 280;
    const VALUE_WIDTH: usize = 296;
    const GATE_START: usize = 8;
    const VALUE_START: usize = 16;
    const QUANTIZATION_TOLERANCE: f32 = 0.08;

    let cuda = Device::new_cuda(0).unwrap();
    let gate_values = (0..ROWS * GATE_WIDTH)
        .map(|index| (((index * 17 + 5) % 101) as f32 - 50.0) / 19.0)
        .collect::<Vec<_>>();
    let value_values = (0..ROWS * VALUE_WIDTH)
        .map(|index| (((index * 13 + 7) % 89) as f32 - 44.0) / 17.0)
        .collect::<Vec<_>>();
    let gate = Tensor::from_vec(gate_values, (ROWS, 1, GATE_WIDTH), &cuda)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .narrow(2, GATE_START, COLUMNS)
        .unwrap();
    let value = Tensor::from_vec(value_values, (ROWS, 1, VALUE_WIDTH), &cuda)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .narrow(2, VALUE_START, COLUMNS)
        .unwrap();
    assert!(!gate.is_contiguous());
    assert!(!value.is_contiguous());
    assert_eq!(gate.layout().stride()[1], GATE_WIDTH);
    assert_eq!(value.layout().stride()[1], VALUE_WIDTH);

    let expected = fused_glu(&gate, &value, GluActivationType::Sigmoid)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let (quantized, scales) = fused_glu_quantized_bf16(
        &gate,
        &value,
        GROUP_SIZE,
        SCALE_STRIDE,
        GluActivationType::Sigmoid,
    )
    .unwrap()
    .unwrap();
    assert_eq!(quantized.dims(), &[ROWS, COLUMNS]);
    assert_eq!(scales.dims(), &[COLUMNS / GROUP_SIZE, SCALE_STRIDE]);
    let quantized = quantized
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<F8E4M3>()
        .unwrap();
    let scales = scales
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    for row in 0..ROWS {
        for group in 0..COLUMNS / GROUP_SIZE {
            let start = row * COLUMNS + group * GROUP_SIZE;
            let end = start + GROUP_SIZE;
            let maximum = expected[start..end]
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0f32, f32::max);
            let expected_scale = if maximum == 0.0 { 1.0 } else { maximum / 448.0 };
            assert!((scales[group][row] - expected_scale).abs() <= 1.0e-6);
            for column in group * GROUP_SIZE..(group + 1) * GROUP_SIZE {
                let index = row * COLUMNS + column;
                let actual = quantized[index].to_f32() * scales[group][row];
                let tolerance = QUANTIZATION_TOLERANCE * expected[index].abs().max(maximum / 64.0);
                assert!(
                    (actual - expected[index]).abs() <= tolerance,
                    "value mismatch at row {row}, column {column}: actual={actual}, expected={}",
                    expected[index]
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_sigmoid_glu_cuda_row_strides_and_offsets() {
    use super::{fused_glu, GluActivationType};
    use candle_core::{DType, Device, Tensor};

    const ROWS: usize = 3;
    const COLS: usize = 8;

    let cpu = Device::Cpu;
    let cuda = Device::new_cuda(0).unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for (a_width, a_start, b_width, b_start) in [(13, 1, 15, 2), (16, 4, 16, 4)] {
            let a_data = (0..ROWS * a_width)
                .map(|index| (index as f32 - 17.0) / 9.0)
                .collect::<Vec<_>>();
            let b_data = (0..ROWS * b_width)
                .map(|index| (23.0 - index as f32) / 13.0)
                .collect::<Vec<_>>();
            let a = Tensor::from_vec(a_data, (ROWS, 1, a_width), &cuda)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .narrow(2, a_start, COLS)
                .unwrap();
            let b = Tensor::from_vec(b_data, (ROWS, 1, b_width), &cuda)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .narrow(2, b_start, COLS)
                .unwrap();
            assert!(!a.is_contiguous());
            assert!(!b.is_contiguous());
            assert_ne!(a.layout().start_offset(), 0);
            assert_ne!(b.layout().start_offset(), 0);

            let expected = candle_nn::ops::sigmoid(&a)
                .unwrap()
                .broadcast_mul(&b)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let actual = fused_glu(&a, &b, GluActivationType::Sigmoid)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let tolerance = if dtype == DType::F32 { 1e-6 } else { 2e-2 };
            assert_close(&actual, &expected, tolerance);
        }
    }
}

#[test]
fn test_softcap_cpu_f32() {
    use super::softcap;
    use candle_core::Tensor;

    let device = candle_core::Device::Cpu;
    let cap = 30.0;
    let data: Vec<f32> = (-64..64).map(|i| i as f32 * 0.75).collect();
    let expected: Vec<f32> = data.iter().map(|x| (x / cap).tanh() * cap).collect();
    let input = Tensor::from_vec(data, &[8, 16], &device).unwrap();
    let output = softcap(&input, cap)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_close(&output, &expected, 1e-6);
}

#[cfg(feature = "cuda")]
#[test]
fn test_softcap_cuda_f32() {
    use super::softcap;
    use candle_core::Tensor;

    let cpu = candle_core::Device::Cpu;
    let cuda = candle_core::Device::new_cuda(0).unwrap();
    let cap = 30.0;
    let data: Vec<f32> = (-128..128).map(|i| i as f32 * 0.5).collect();
    let input = Tensor::from_vec(data, &[4, 64], &cuda).unwrap();
    let expected = ((&input / cap as f64).unwrap().tanh().unwrap() * cap as f64)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let output = softcap(&input, cap)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_close(&output, &expected, 5e-4);
}

#[cfg(feature = "metal")]
#[test]
fn test_softcap_metal_f32() {
    use super::softcap;
    use candle_core::Tensor;

    let cpu = candle_core::Device::Cpu;
    let metal = candle_core::Device::new_metal(0).unwrap();
    let cap = 30.0;
    let data: Vec<f32> = (-128..128).map(|i| i as f32 * 0.5).collect();
    let input = Tensor::from_vec(data, &[4, 64], &metal).unwrap();
    let expected = ((&input / cap as f64).unwrap().tanh().unwrap() * cap as f64)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let output = softcap(&input, cap)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_close(&output, &expected, 1e-4);
}

#[test]
fn test_cumsum_exclusive_forward_cpu() {
    use crate::utils::ops::CumSumOp;
    use candle_core::Tensor;
    let device = candle_core::Device::Cpu;
    let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
    let b = a.fast_cumsum(0).unwrap().to_vec1::<i64>().unwrap();
    assert_eq!(b, [0, 1, 3, 6]);
}

#[test]
fn test_cumsum_inclusive_forward_cpu() {
    use crate::utils::ops::CumSumOp;
    use candle_core::Tensor;
    let device = candle_core::Device::Cpu;
    let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
    let b = a
        .fast_cumsum_config(0, true, false)
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    assert_eq!(b, [1, 3, 6, 10]);
}

#[test]
fn test_cumsum_exclusive_reverse_cpu() {
    use crate::utils::ops::CumSumOp;
    use candle_core::Tensor;
    let device = candle_core::Device::Cpu;
    let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
    let b = a
        .fast_cumsum_config(0, false, true)
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    assert_eq!(b, [9, 7, 4, 0]);
}

#[test]
fn test_cumsum_inclusive_reverse_cpu() {
    use crate::utils::ops::CumSumOp;
    use candle_core::Tensor;
    let device = candle_core::Device::Cpu;
    let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
    let b = a
        .fast_cumsum_config(0, true, true)
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    assert_eq!(b, [10, 9, 7, 4]);
}

#[cfg(feature = "metal")]
#[test]
fn test_cumsum_exclusive_forward_metal() {
    use crate::utils::ops::CumSumOp;
    use candle_core::Tensor;
    let device = candle_core::Device::new_metal(0).unwrap();
    let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
    let b = a.fast_cumsum(0).unwrap().to_vec1::<i64>().unwrap();
    assert_eq!(b, [0, 1, 3, 6]);
}

#[cfg(feature = "metal")]
#[test]
fn test_cumsum_inclusive_forward_metal() {
    use crate::utils::ops::CumSumOp;
    use candle_core::Tensor;
    let device = candle_core::Device::new_metal(0).unwrap();
    let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
    let b = a
        .fast_cumsum_config(0, true, false)
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    assert_eq!(b, [1, 3, 6, 10]);
}

#[cfg(feature = "metal")]
#[test]
fn test_cumsum_exclusive_reverse_metal() {
    use crate::utils::ops::CumSumOp;
    use candle_core::Tensor;
    let device = candle_core::Device::new_metal(0).unwrap();
    let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
    let b = a
        .fast_cumsum_config(0, false, true)
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    assert_eq!(b, [9, 7, 4, 0]);
}

#[cfg(feature = "metal")]
#[test]
fn test_cumsum_inclusive_reverse_metal() {
    use crate::utils::ops::CumSumOp;
    use candle_core::Tensor;
    let device = candle_core::Device::new_metal(0).unwrap();
    let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
    let b = a
        .fast_cumsum_config(0, true, true)
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    assert_eq!(b, [10, 9, 7, 4]);
}

#[test]
fn test_nonzero_cpu() {
    use crate::utils::ops::NonZeroOp;
    use candle_core::Tensor;
    let device = candle_core::Device::Cpu;
    let a = Tensor::from_vec(
        vec![1f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
        &[2, 4],
        &device,
    )
    .unwrap();
    let b = a.nonzero().unwrap().to_vec2::<u32>().unwrap();
    assert_eq!(b, [[0, 0], [0, 2], [1, 0], [1, 2]]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_nonzero_cuda() {
    use crate::utils::ops::NonZeroOp;
    use candle_core::Tensor;
    let device = candle_core::Device::new_cuda(0).unwrap();
    let a = Tensor::from_vec(
        vec![1f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
        &[2, 4],
        &device,
    )
    .unwrap();
    let b = a.nonzero().unwrap().to_vec2::<u32>().unwrap();
    assert_eq!(b, [[0, 0], [0, 2], [1, 0], [1, 2]]);
}

#[test]
fn test_bitwise_and_cpu() {
    use crate::utils::ops::BitWiseOp;
    use candle_core::Tensor;
    let device = candle_core::Device::Cpu;
    let a = Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
    let b = Tensor::from_vec(vec![-1i64, 2, 3, -1, 1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
    let c = a.bitwise_and(&b).unwrap().to_vec2::<i64>().unwrap();
    assert_eq!(c, [[1, 2], [3, -1], [1, -1], [-1, 4], [5, 7]]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_bitwise_and_cuda() {
    use crate::utils::ops::BitWiseOp;
    use candle_core::Tensor;
    let device = candle_core::Device::new_cuda(0).unwrap();
    let a = Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
    let b = Tensor::from_vec(vec![-1i64, 2, 3, -1, 1, -1, -1, 4, 0, 7], (5, 2), &device).unwrap();
    let c = a.bitwise_and(&b).unwrap().to_vec2::<i64>().unwrap();
    assert_eq!(c, [[1, 2], [3, -1], [1, -1], [-1, 4], [0, 7]]);
}

#[test]
fn test_bitwise_or_cpu() {
    use crate::utils::ops::BitWiseOp;
    use candle_core::Tensor;
    let device = candle_core::Device::Cpu;
    let a = Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
    let b = Tensor::from_vec(vec![-1i64, 0, 0, 0, 0, 0, 0, 0, 0, 8], (5, 2), &device).unwrap();
    let c = a.bitwise_or(&b).unwrap().to_vec2::<i64>().unwrap();
    assert_eq!(c, [[-1, 2], [3, -1], [-1, -1], [-1, 4], [5, 15]]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_bitwise_or_cuda() {
    use crate::utils::ops::BitWiseOp;
    use candle_core::Tensor;
    let device = candle_core::Device::new_cuda(0).unwrap();
    let a = Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
    let b = Tensor::from_vec(vec![-1i64, 0, 0, 0, 0, 0, 0, 0, 0, 8], (5, 2), &device).unwrap();
    let c = a.bitwise_or(&b).unwrap().to_vec2::<i64>().unwrap();
    assert_eq!(c, [[-1, 2], [3, -1], [-1, -1], [-1, 4], [5, 15]]);
}

#[test]
fn test_bitwise_xor_cpu() {
    use crate::utils::ops::BitWiseOp;
    use candle_core::Tensor;
    let device = candle_core::Device::Cpu;
    let a = Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
    let b = Tensor::from_vec(vec![-1i64, 0, 0, 0, 0, 0, 0, 0, 0, 8], (5, 2), &device).unwrap();
    let c = a.bitwise_xor(&b).unwrap().to_vec2::<i64>().unwrap();
    assert_eq!(c, [[-2, 2], [3, -1], [-1, -1], [-1, 4], [5, 15]]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_bitwise_xor_cuda() {
    use crate::utils::ops::BitWiseOp;
    use candle_core::Tensor;
    let device = candle_core::Device::new_cuda(0).unwrap();
    let a = Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
    let b = Tensor::from_vec(vec![-1i64, 0, 0, 0, 0, 0, 0, 0, 0, 8], (5, 2), &device).unwrap();
    let c = a.bitwise_xor(&b).unwrap().to_vec2::<i64>().unwrap();
    assert_eq!(c, [[-2, 2], [3, -1], [-1, -1], [-1, 4], [5, 15]]);
}

#[cfg(feature = "metal")]
#[test]
fn test_bitwise_metal_unaligned_length() -> candle_core::Result<()> {
    use super::{BitWiseOp, LeftshiftOp};
    use candle_core::{Device, Tensor};

    const TEST_LENGTH: usize = 2049;
    const TEST_SHIFT: usize = 3;

    let device = Device::new_metal(0)?;
    let lhs = (0..TEST_LENGTH)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect::<Vec<_>>();
    let rhs = (0..TEST_LENGTH)
        .map(|index| (index.wrapping_mul(19).wrapping_add(7) & 0xff) as u8)
        .collect::<Vec<_>>();
    let lhs_tensor = Tensor::from_vec(lhs.clone(), TEST_LENGTH, &device)?;
    let rhs_tensor = Tensor::from_vec(rhs.clone(), TEST_LENGTH, &device)?;

    let and = lhs_tensor
        .bitwise_and(&rhs_tensor)?
        .to_device(&Device::Cpu)?
        .to_vec1::<u8>()?;
    let or = lhs_tensor
        .bitwise_or(&rhs_tensor)?
        .to_device(&Device::Cpu)?
        .to_vec1::<u8>()?;
    let xor = lhs_tensor
        .bitwise_xor(&rhs_tensor)?
        .to_device(&Device::Cpu)?
        .to_vec1::<u8>()?;
    let not = lhs_tensor
        .bitwise_not()?
        .to_device(&Device::Cpu)?
        .to_vec1::<u8>()?;
    let shifted = lhs_tensor
        .leftshift(TEST_SHIFT)?
        .to_device(&Device::Cpu)?
        .to_vec1::<u8>()?;

    assert_eq!(
        and,
        lhs.iter()
            .zip(&rhs)
            .map(|(lhs, rhs)| lhs & rhs)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        or,
        lhs.iter()
            .zip(&rhs)
            .map(|(lhs, rhs)| lhs | rhs)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        xor,
        lhs.iter()
            .zip(&rhs)
            .map(|(lhs, rhs)| lhs ^ rhs)
            .collect::<Vec<_>>()
    );
    assert_eq!(not, lhs.iter().map(|value| !value).collect::<Vec<_>>());
    assert_eq!(
        shifted,
        lhs.iter()
            .map(|value| value.wrapping_shl(TEST_SHIFT as u32))
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn test_nonzero_and() {
    use crate::utils::ops::{BitWiseOp, NonZeroOp};
    use candle_core::{Device, Tensor};

    let input1 = Tensor::from_vec(
        vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7],
        (10,),
        &Device::Cpu,
    )
    .unwrap();
    let input2 = Tensor::from_vec(
        vec![-1i64, 2, 3, -1, 1, -1, -1, 4, 5, 7],
        (10,),
        &Device::Cpu,
    )
    .unwrap();
    let input = Tensor::stack(&[input1, input2], 0).unwrap();

    let lt = input.lt(0.0).unwrap();
    let gt = input.gt(-10.0).unwrap();
    let res = lt
        .bitwise_and(&gt)
        .unwrap()
        .nonzero()
        .unwrap()
        .to_vec2::<u32>()
        .unwrap();

    assert_eq!(
        res,
        [
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [1, 0],
            [1, 3],
            [1, 5],
            [1, 6]
        ]
    );
}

#[cfg(feature = "cuda")]
#[test]
fn nonzero_and_cuda() {
    use crate::utils::ops::{BitWiseOp, NonZeroOp};
    use candle_core::{Device, Tensor};

    let device = Device::new_cuda(0).unwrap();
    let input1 =
        Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (10,), &device).unwrap();
    let input2 =
        Tensor::from_vec(vec![-1i64, 2, 3, -1, 1, -1, -1, 4, 5, 7], (10,), &device).unwrap();
    let input = Tensor::stack(&[input1, input2], 0).unwrap();

    let lt = input.lt(0.0).unwrap();
    let gt = input.gt(-10.0).unwrap();
    let res = lt
        .bitwise_and(&gt)
        .unwrap()
        .nonzero()
        .unwrap()
        .to_vec2::<u32>()
        .unwrap();

    assert_eq!(
        res,
        [
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [1, 0],
            [1, 3],
            [1, 5],
            [1, 6]
        ]
    );
}

#[test]
fn test_bitpack_8bit_cpu() {
    use crate::HqqBits;
    use candle_core::{Device, Tensor};
    let bits = HqqBits::Eight;
    let device = Device::Cpu;
    let wq = Tensor::from_vec(vec![257_i32, 258, 259, 260, 511, 512], (3, 2), &device).unwrap();
    let c = bits.bitpack_type()(wq.clone())
        .unwrap()
        .to_vec2::<u8>()
        .unwrap();
    assert_eq!(c, [[1, 2], [3, 4], [255, 0]]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_bitpack_8bit_cuda() {
    use crate::HqqBits;
    use candle_core::{Device, Tensor};
    let bits = HqqBits::Eight;
    let device = Device::new_cuda(0).unwrap();
    // Use U8 tensor directly to avoid candle's to_dtype which may not have
    // PTX compiled for newer GPU architectures (e.g., SM 120)
    let wq = Tensor::from_vec(vec![1_u8, 2, 3, 4, 255, 0], (3, 2), &device).unwrap();
    let c = bits.bitpack_type()(wq.clone())
        .unwrap()
        .to_vec2::<u8>()
        .unwrap();
    assert_eq!(c, [[1, 2], [3, 4], [255, 0]]);
}

#[cfg(feature = "metal")]
#[test]
fn test_bitpack_8bit_metal() {
    use crate::HqqBits;
    use candle_core::{Device, Tensor};
    let bits = HqqBits::Eight;
    let device = Device::new_metal(0).unwrap();
    let wq = Tensor::from_vec(vec![257_i32, 258, 259, 260, 511, 512], (3, 2), &device).unwrap();
    let c = bits.bitpack_type()(wq.clone())
        .unwrap()
        .to_vec2::<u8>()
        .unwrap();
    assert_eq!(c, [[1, 2], [3, 4], [255, 0]]);
}

#[test]
fn test_bitpack_4bit() {
    use crate::HqqBits;
    use candle_core::{Device, Tensor};
    let bits = HqqBits::Four;
    let device = Device::Cpu;
    let wq = Tensor::from_vec(vec![1_u8, 2, 3, 4, 5, 6], (3, 2), &device).unwrap();
    let c = bits.bitpack_type()(wq.clone())
        .unwrap()
        .to_vec2::<u8>()
        .unwrap();
    assert_eq!(c, [[19, 36]]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_bitpack_4bit_cuda() {
    use crate::HqqBits;
    use candle_core::{Device, Tensor};
    let bits = HqqBits::Four;
    let device = Device::new_cuda(0).unwrap();
    let wq = Tensor::from_vec(vec![1_u8, 2, 3, 4, 5, 6], (3, 2), &device).unwrap();
    let c = bits.bitpack_type()(wq.clone())
        .unwrap()
        .to_vec2::<u8>()
        .unwrap();
    assert_eq!(c, [[19, 36]]);
}

#[cfg(feature = "metal")]
#[test]
fn test_bitpack_4bit_metal() {
    use crate::HqqBits;
    use candle_core::{Device, Tensor};
    let bits = HqqBits::Four;
    let device = Device::new_metal(0).unwrap();
    let wq = Tensor::from_vec(vec![1_u8, 2, 3, 4, 5, 6], (3, 2), &device).unwrap();
    let c = bits.bitpack_type()(wq.clone())
        .unwrap()
        .to_vec2::<u8>()
        .unwrap();
    assert_eq!(c, [[19, 36]]);
}
// ─────────────────────────────── Sort / ArgSort ────────────────────────────────
#[cfg(feature = "metal")]
#[test]
fn test_sort_and_argsort_vector_metal() {
    use crate::utils::ops::SortOp;
    use candle_core::Tensor;

    let device = candle_core::Device::new_metal(0).unwrap();
    let a = Tensor::from_vec(vec![3i32, 1, 4, 2], &[4], &device).unwrap();

    // sort (ascending)
    let sorted = a.fast_sort_asc(0).unwrap().to_vec1::<i32>().unwrap();
    assert_eq!(sorted, [1, 2, 3, 4]);

    // argsort (ascending indices)
    let idx = a.fast_argsort_asc(0).unwrap().to_vec1::<u32>().unwrap();
    assert_eq!(idx, [1, 3, 0, 2]);
}

#[cfg(feature = "metal")]
#[test]
fn test_sort_and_argsort_matrix_axis1_metal() {
    use crate::utils::ops::SortOp;
    use candle_core::Tensor;

    let device = candle_core::Device::new_metal(0).unwrap();
    // 2 × 3 matrix:
    // [[3, 1, 2],
    //  [0, 4, 5]]
    let a = Tensor::from_vec(vec![3i32, 1, 2, 0, 4, 5], &[2, 3], &device).unwrap();

    // Sort along axis=1 (second dimension)
    let sorted = a.fast_sort_asc(1).unwrap().to_vec2::<i32>().unwrap();
    assert_eq!(sorted, [[1, 2, 3], [0, 4, 5]]);

    // ArgSort indices along axis=1
    let idx = a.fast_argsort_asc(1).unwrap().to_vec2::<u32>().unwrap();
    assert_eq!(idx, [[1, 2, 0], [0, 1, 2]]);
}

// ─────────────────────────────── 2 048-element vector ────────────────────────────────
#[cfg(feature = "metal")]
#[test]
fn test_sort_and_argsort_vector_2048_metal() {
    use crate::utils::ops::SortOp;
    use candle_core::Tensor;

    const N: usize = 4096;

    let device = candle_core::Device::new_metal(0).expect("Metal device");

    // Create a descending vector [4095, 4094, …, 0]
    let vals: Vec<i32> = (0..N as i32).rev().collect();
    let a = Tensor::from_vec(vals.clone(), &[N], &device).unwrap();

    // ---- sort (ascending) ---------------------------------------------------------
    let sorted = a.fast_sort_asc(0).unwrap().to_vec1::<i32>().unwrap();
    let expected: Vec<i32> = (0..N as i32).collect();
    assert_eq!(sorted, expected);

    // ---- argsort (indices that would sort) ---------------------------------------
    let idx = a.fast_argsort_asc(0).unwrap().to_vec1::<u32>().unwrap();
    // Because the input is reversed, the correct indices are likewise reversed
    for (i, &v) in idx.iter().enumerate() {
        assert_eq!(v as usize, N - 1 - i);
    }
}

#[cfg(feature = "metal")]
#[test]
fn test_fused_glu_metal_silu_f32() {
    use super::{fused_glu, GluActivationType};
    use candle_core::Tensor;

    let cpu = candle_core::Device::Cpu;
    let metal = candle_core::Device::new_metal(0).unwrap();

    let a_data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
    let b_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.7 - 90.0) / 50.0).collect();

    let a_cpu = Tensor::from_vec(a_data.clone(), &[4, 64], &cpu).unwrap();
    let b_cpu = Tensor::from_vec(b_data.clone(), &[4, 64], &cpu).unwrap();
    let a_metal = Tensor::from_vec(a_data, &[4, 64], &metal).unwrap();
    let b_metal = Tensor::from_vec(b_data, &[4, 64], &metal).unwrap();

    let cpu_result = fused_glu(&a_cpu, &b_cpu, GluActivationType::Silu)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let metal_result = fused_glu(&a_metal, &b_metal, GluActivationType::Silu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    for (row_cpu, row_metal) in cpu_result.iter().zip(metal_result.iter()) {
        for (c, m) in row_cpu.iter().zip(row_metal.iter()) {
            let diff = (c - m).abs();
            assert!(
                diff < 1e-4,
                "SiLU F32 mismatch: cpu={c}, metal={m}, diff={diff}"
            );
        }
    }
}

#[cfg(feature = "metal")]
#[test]
fn test_fused_glu_metal_silu_f16() {
    use super::{fused_glu, GluActivationType};
    use candle_core::{DType, Tensor};

    let cpu = candle_core::Device::Cpu;
    let metal = candle_core::Device::new_metal(0).unwrap();

    let a_data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
    let b_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.7 - 90.0) / 50.0).collect();

    let a_cpu = Tensor::from_vec(a_data.clone(), &[256], &cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let b_cpu = Tensor::from_vec(b_data.clone(), &[256], &cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let a_metal = Tensor::from_vec(a_data, &[256], &metal)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let b_metal = Tensor::from_vec(b_data, &[256], &metal)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();

    let cpu_result = fused_glu(&a_cpu, &b_cpu, GluActivationType::Silu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let metal_result = fused_glu(&a_metal, &b_metal, GluActivationType::Silu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    for (i, (c, m)) in cpu_result.iter().zip(metal_result.iter()).enumerate() {
        let diff = (c - m).abs();
        assert!(
            diff < 1e-2,
            "SiLU F16 mismatch at {i}: cpu={c}, metal={m}, diff={diff}"
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn test_fused_glu_metal_all_activations() {
    use super::{fused_glu, GluActivationType};
    use candle_core::Tensor;

    let cpu = candle_core::Device::Cpu;
    let metal = candle_core::Device::new_metal(0).unwrap();

    let a_data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 32.0).collect();
    let b_data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.5 - 32.0) / 20.0).collect();

    for act in [
        GluActivationType::Silu,
        GluActivationType::Gelu,
        GluActivationType::Relu,
        GluActivationType::GeluErf,
        GluActivationType::Sigmoid,
    ] {
        let a_cpu = Tensor::from_vec(a_data.clone(), &[128], &cpu).unwrap();
        let b_cpu = Tensor::from_vec(b_data.clone(), &[128], &cpu).unwrap();
        let a_metal = Tensor::from_vec(a_data.clone(), &[128], &metal).unwrap();
        let b_metal = Tensor::from_vec(b_data.clone(), &[128], &metal).unwrap();

        let cpu_result = fused_glu(&a_cpu, &b_cpu, act)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let metal_result = fused_glu(&a_metal, &b_metal, act)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (i, (c, m)) in cpu_result.iter().zip(metal_result.iter()).enumerate() {
            let diff = (c - m).abs();
            assert!(
                diff < 1e-4,
                "{act:?} F32 mismatch at {i}: cpu={c}, metal={m}, diff={diff}"
            );
        }
    }
}

/// Test that fused_glu matches candle's fallback path (a.gelu() * b) for BF16.
/// This was the exact scenario that caused model failure (Gemma 3 4B, BF16, GeluPytorchTanh).
#[cfg(feature = "metal")]
#[test]
fn test_fused_glu_matches_candle_fallback_bf16() {
    use super::{fused_glu, GluActivationType};
    use candle_core::{DType, Tensor};

    let metal = candle_core::Device::new_metal(0).unwrap();

    // Use realistic-sized data matching model dimensions
    let n = 10240;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32 - 5120.0) / 2560.0).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 - 1500.0) / 1000.0).collect();

    let a_metal = Tensor::from_vec(a_data.clone(), &[1, 2, n / 2], &metal)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let b_metal = Tensor::from_vec(b_data.clone(), &[1, 2, n / 2], &metal)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // Fused path
    let fused = fused_glu(&a_metal, &b_metal, GluActivationType::Gelu).unwrap();

    // Candle's fallback: a.gelu() * b (the tanh-approx GELU)
    let fallback = (a_metal.gelu().unwrap() * &b_metal).unwrap();

    let fused_f32 = fused
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let fallback_f32 = fallback
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut max_diff: f32 = 0.0;
    let mut num_mismatches = 0;
    for (f, fb) in fused_f32.iter().zip(fallback_f32.iter()) {
        let diff = (f - fb).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.0 {
            num_mismatches += 1;
        }
    }
    eprintln!(
        "BF16 Gelu fused vs fallback: max_diff={max_diff}, mismatches={num_mismatches}/{}",
        fused_f32.len()
    );
    // Allow up to 1 BF16 ULP difference (0.015625 at values around 1-2)
    // This is acceptable since Metal compiler may keep intermediate precision
    assert!(
        max_diff <= 0.015625,
        "BF16 Gelu fused vs candle fallback max_diff {max_diff} exceeds 1 BF16 ULP"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_glu_cuda_silu_f32() {
    use super::{fused_glu, GluActivationType};
    use candle_core::Tensor;

    let cpu = candle_core::Device::Cpu;
    let cuda = candle_core::Device::new_cuda(0).unwrap();

    let a_data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
    let b_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.7 - 90.0) / 50.0).collect();

    let a_cpu = Tensor::from_vec(a_data.clone(), &[4, 64], &cpu).unwrap();
    let b_cpu = Tensor::from_vec(b_data.clone(), &[4, 64], &cpu).unwrap();
    let a_cuda = Tensor::from_vec(a_data, &[4, 64], &cuda).unwrap();
    let b_cuda = Tensor::from_vec(b_data, &[4, 64], &cuda).unwrap();

    let cpu_result = fused_glu(&a_cpu, &b_cpu, GluActivationType::Silu)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let cuda_result = fused_glu(&a_cuda, &b_cuda, GluActivationType::Silu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    for (row_cpu, row_cuda) in cpu_result.iter().zip(cuda_result.iter()) {
        for (c, g) in row_cpu.iter().zip(row_cuda.iter()) {
            let diff = (c - g).abs();
            assert!(
                diff < 1e-4,
                "SiLU F32 mismatch: cpu={c}, cuda={g}, diff={diff}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_glu_cuda_silu_f16() {
    use super::{fused_glu, GluActivationType};
    use candle_core::{DType, Tensor};

    let cpu = candle_core::Device::Cpu;
    let cuda = candle_core::Device::new_cuda(0).unwrap();

    let a_data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
    let b_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.7 - 90.0) / 50.0).collect();

    let a_cpu = Tensor::from_vec(a_data.clone(), &[256], &cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let b_cpu = Tensor::from_vec(b_data.clone(), &[256], &cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let a_cuda = Tensor::from_vec(a_data, &[256], &cuda)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let b_cuda = Tensor::from_vec(b_data, &[256], &cuda)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();

    let cpu_result = fused_glu(&a_cpu, &b_cpu, GluActivationType::Silu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let cuda_result = fused_glu(&a_cuda, &b_cuda, GluActivationType::Silu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    for (i, (c, g)) in cpu_result.iter().zip(cuda_result.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff < 1e-2,
            "SiLU F16 mismatch at {i}: cpu={c}, cuda={g}, diff={diff}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_glu_cuda_all_activations() {
    use super::{fused_glu, GluActivationType};
    use candle_core::Tensor;

    let cpu = candle_core::Device::Cpu;
    let cuda = candle_core::Device::new_cuda(0).unwrap();

    let a_data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 32.0).collect();
    let b_data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.5 - 32.0) / 20.0).collect();

    for act in [
        GluActivationType::Silu,
        GluActivationType::Gelu,
        GluActivationType::Relu,
        GluActivationType::GeluErf,
        GluActivationType::Sigmoid,
    ] {
        let a_cpu = Tensor::from_vec(a_data.clone(), &[128], &cpu).unwrap();
        let b_cpu = Tensor::from_vec(b_data.clone(), &[128], &cpu).unwrap();
        let a_cuda = Tensor::from_vec(a_data.clone(), &[128], &cuda).unwrap();
        let b_cuda = Tensor::from_vec(b_data.clone(), &[128], &cuda).unwrap();

        let cpu_result = fused_glu(&a_cpu, &b_cpu, act)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cuda_result = fused_glu(&a_cuda, &b_cuda, act)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (i, (c, g)) in cpu_result.iter().zip(cuda_result.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(
                diff < 1e-4,
                "{act:?} F32 mismatch at {i}: cpu={c}, cuda={g}, diff={diff}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_fused_glu_matches_candle_fallback_bf16_cuda() {
    use super::{fused_glu, GluActivationType};
    use candle_core::{DType, Tensor};

    let cuda = candle_core::Device::new_cuda(0).unwrap();

    let n = 10240;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32 - 5120.0) / 2560.0).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 - 1500.0) / 1000.0).collect();

    let a_cuda = Tensor::from_vec(a_data.clone(), &[1, 2, n / 2], &cuda)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let b_cuda = Tensor::from_vec(b_data.clone(), &[1, 2, n / 2], &cuda)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // Fused path
    let fused = fused_glu(&a_cuda, &b_cuda, GluActivationType::Gelu).unwrap();

    // Candle's fallback: a.gelu() * b (the tanh-approx GELU)
    let fallback = (a_cuda.gelu().unwrap() * &b_cuda).unwrap();

    let fused_f32 = fused
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let fallback_f32 = fallback
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut max_diff: f32 = 0.0;
    let mut num_mismatches = 0;
    for (f, fb) in fused_f32.iter().zip(fallback_f32.iter()) {
        let diff = (f - fb).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.0 {
            num_mismatches += 1;
        }
    }
    eprintln!(
        "CUDA BF16 Gelu fused vs fallback: max_diff={max_diff}, mismatches={num_mismatches}/{}",
        fused_f32.len()
    );
    // Allow up to 1 BF16 ULP difference (0.015625 at values around 1-2)
    assert!(
        max_diff <= 0.015625,
        "CUDA BF16 Gelu fused vs candle fallback max_diff {max_diff} exceeds 1 BF16 ULP"
    );
}
