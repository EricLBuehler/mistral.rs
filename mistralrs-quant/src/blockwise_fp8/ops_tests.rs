use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};
use float8::F8E4M3;
use half::bf16;

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
use crate::blockwise_fp8::deepgemm;
use crate::blockwise_fp8::ops;

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
#[test]
fn deepgemm_ffi_plan_layout() {
    use crate::blockwise_fp8::ffi::{DeepGemmPlan, DeepGemmPrepared};

    assert_eq!(std::mem::offset_of!(DeepGemmPlan, workspace_bytes), 56);
    assert_eq!(std::mem::offset_of!(DeepGemmPlan, cache_key), 64);
    assert_eq!(std::mem::size_of::<DeepGemmPlan>(), 72);
    assert_eq!(std::mem::offset_of!(DeepGemmPrepared, function), 72);
    assert_eq!(std::mem::size_of::<DeepGemmPrepared>(), 80);
}

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
#[test]
fn fp8_workspace_keys_per_thread_default_streams() {
    let current = ops::fp8_workspace_thread(ops::CUDA_STREAM_PER_THREAD_HANDLE).unwrap();
    let other = std::thread::spawn(|| {
        ops::fp8_workspace_thread(ops::CUDA_STREAM_PER_THREAD_HANDLE).unwrap()
    })
    .join()
    .unwrap();
    assert_ne!(current, other);
    assert_eq!(ops::fp8_workspace_thread(0x1000), None);
}

#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
#[test]
fn fused_add_rms_norm_quantized_matches_bf16_reference() -> Result<()> {
    use crate::{ActivationQuantizationScheme, ActivationScaleLayout};

    const COLUMNS: usize = 256;
    const GROUP_SIZE: usize = 128;
    const EPSILON: f32 = 1.0e-6;
    const FP8_MAX: f32 = 448.0;

    let device = Device::new_cuda(0)?;
    let mut weight_values = Vec::with_capacity(COLUMNS + 1);
    weight_values.push(bf16::from_f32(7.0));
    weight_values.extend((0..COLUMNS).map(|column| {
        if column < GROUP_SIZE {
            bf16::ZERO
        } else {
            bf16::from_f32(0.75 + (column % 19) as f32 / 32.0)
        }
    }));
    let weight =
        Tensor::from_vec(weight_values.clone(), COLUMNS + 1, &device)?.narrow(0, 1, COLUMNS)?;
    assert_eq!(weight.layout().start_offset(), 1);

    let scheme = ActivationQuantizationScheme {
        dtype: DType::F8E4M3,
        scale_encoding: crate::ActivationScaleEncoding::F32,
        block_shape: [1, GROUP_SIZE],
    };
    let scale_layout = ActivationScaleLayout::GroupMajor {
        row_alignment: std::num::NonZeroUsize::new(4).unwrap(),
    };

    for rows in [1usize, 5] {
        let elements = rows * COLUMNS;
        let mut input_values = Vec::with_capacity(elements + 1);
        let mut residual_values = Vec::with_capacity(elements + 1);
        input_values.push(bf16::from_f32(11.0));
        residual_values.push(bf16::from_f32(-13.0));
        input_values.extend((0..elements).map(|index| {
            bf16::from_f32(((index * 17 + index / COLUMNS * 11) % 89) as f32 / 19.0 - 2.0)
        }));
        residual_values.extend((0..elements).map(|index| {
            bf16::from_f32(((index * 7 + index / COLUMNS * 5) % 43) as f32 / 31.0 - 0.6)
        }));
        let input = Tensor::from_vec(input_values.clone(), elements + 1, &device)?
            .narrow(0, 1, elements)?
            .reshape((rows, COLUMNS))?;
        let residual = Tensor::from_vec(residual_values.clone(), elements + 1, &device)?
            .narrow(0, 1, elements)?
            .reshape((rows, COLUMNS))?;
        assert_eq!(input.layout().start_offset(), 1);
        assert_eq!(residual.layout().start_offset(), 1);

        let (fused, normalized) = ops::fused_add_rms_norm_quantized_with_normalized(
            &input,
            &residual,
            &weight,
            EPSILON,
            scheme,
            scale_layout,
        )?;
        assert_eq!(fused.residual().dtype(), DType::BF16);
        assert_eq!(fused.residual().dims(), &[rows, COLUMNS]);
        assert_eq!(fused.activation().source_shape(), &[rows, COLUMNS]);
        assert_eq!(fused.activation().scale_layout(), scale_layout);
        assert_eq!(
            fused.activation().scales().dims(),
            &[COLUMNS / GROUP_SIZE, rows.div_ceil(4) * 4]
        );

        let actual_residual = fused
            .residual()
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<bf16>()?;
        let actual_quantized = fused
            .activation()
            .quantized()
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<F8E4M3>()?;
        let actual_normalized = normalized
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<bf16>()?;
        let actual_scales = fused
            .activation()
            .scales()
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;

        let input_values = &input_values[1..];
        let residual_values = &residual_values[1..];
        let weight_values = &weight_values[1..];
        let mut expected_residual = Vec::with_capacity(elements);
        let mut expected_normed = Vec::with_capacity(elements);
        for row in 0..rows {
            let row_start = row * COLUMNS;
            let mut square_sum = 0.0f32;
            for column in 0..COLUMNS {
                let index = row_start + column;
                let sum = input_values[index].to_f32() + residual_values[index].to_f32();
                let rounded = bf16::from_f32(sum);
                square_sum += rounded.to_f32() * rounded.to_f32();
                expected_residual.push(rounded);
            }
            let inverse_rms = (square_sum / COLUMNS as f32 + EPSILON).sqrt().recip();
            for column in 0..COLUMNS {
                let index = row_start + column;
                expected_normed.push(
                    bf16::from_f32(
                        expected_residual[index].to_f32()
                            * inverse_rms
                            * weight_values[column].to_f32(),
                    )
                    .to_f32(),
                );
            }
        }
        assert_eq!(actual_residual, expected_residual);
        assert_eq!(
            actual_normalized,
            expected_normed
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>()
        );

        let scale_stride = rows.div_ceil(4) * 4;
        for group in 0..COLUMNS / GROUP_SIZE {
            for row in 0..rows {
                let start = row * COLUMNS + group * GROUP_SIZE;
                let maximum = expected_normed[start..start + GROUP_SIZE]
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .fold(0.0f32, f32::max);
                let expected_scale = if maximum == 0.0 {
                    1.0
                } else {
                    maximum / FP8_MAX
                };
                let actual_scale = actual_scales[group][row];
                let scale_tolerance = 1.0e-7 + expected_scale.abs() * 2.0e-3;
                assert!(
                        (actual_scale - expected_scale).abs() <= scale_tolerance,
                        "rows={rows}, row={row}, group={group}: expected scale {expected_scale}, got {actual_scale}"
                    );
                for column in group * GROUP_SIZE..(group + 1) * GROUP_SIZE {
                    let index = row * COLUMNS + column;
                    let actual = actual_quantized[index].to_f32() * actual_scale;
                    let expected = expected_normed[index];
                    let tolerance = 1.0e-4 + 0.09 * expected.abs().max(maximum / 64.0);
                    assert!(
                            (actual - expected).abs() <= tolerance,
                            "rows={rows}, row={row}, column={column}: expected {expected}, got {actual}"
                        );
                }
            }
            assert!(actual_scales[group][rows..scale_stride]
                .iter()
                .all(|scale| *scale == 0.0));
        }
    }
    Ok(())
}

#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
#[test]
fn fused_add_rms_norm_quantized_aligned_store_normalized_matches_reference() -> Result<()> {
    use crate::{ActivationQuantizationScheme, ActivationScaleLayout};

    const COLUMNS: usize = 5120;
    const GROUP_SIZE: usize = 128;
    const ROW_ALIGNMENT: usize = 4;
    const EPSILON: f32 = 1.0e-6;
    const FP8_MAX: f32 = 448.0;

    let device = Device::new_cuda(0)?;
    let weight_values = (0..COLUMNS)
        .map(|column| {
            if column < GROUP_SIZE {
                bf16::ZERO
            } else {
                bf16::from_f32(0.5 + (column % 16) as f32 / 16.0)
            }
        })
        .collect::<Vec<_>>();
    let weight = Tensor::from_vec(weight_values.clone(), COLUMNS, &device)?;
    assert_eq!(weight.layout().start_offset(), 0);

    let scheme = ActivationQuantizationScheme {
        dtype: DType::F8E4M3,
        scale_encoding: crate::ActivationScaleEncoding::F32,
        block_shape: [1, GROUP_SIZE],
    };
    let scale_layout = ActivationScaleLayout::GroupMajor {
        row_alignment: std::num::NonZeroUsize::new(ROW_ALIGNMENT).unwrap(),
    };

    for rows in [1usize, 5, 128] {
        let elements = rows * COLUMNS;
        let values = (0..elements)
            .map(|index| {
                let row = index / COLUMNS;
                let magnitude = [0.5f32, 1.0, 2.0, 4.0][(index + row * 3) % 4];
                if (index + row).is_multiple_of(2) {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect::<Vec<_>>();
        let input_values = values
            .iter()
            .copied()
            .map(|value| bf16::from_f32(value * 0.75))
            .collect::<Vec<_>>();
        let residual_values = values
            .iter()
            .copied()
            .map(|value| bf16::from_f32(value * 0.25))
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input_values.clone(), (rows, COLUMNS), &device)?;
        let residual = Tensor::from_vec(residual_values.clone(), (rows, COLUMNS), &device)?;
        assert_eq!(input.layout().start_offset(), 0);
        assert_eq!(residual.layout().start_offset(), 0);

        let (fused, normalized) = ops::fused_add_rms_norm_quantized_with_normalized(
            &input,
            &residual,
            &weight,
            EPSILON,
            scheme,
            scale_layout,
        )?;
        assert_eq!(fused.residual().dims(), &[rows, COLUMNS]);
        assert_eq!(normalized.dims(), &[rows, COLUMNS]);
        assert_eq!(fused.activation().source_shape(), &[rows, COLUMNS]);
        assert_eq!(fused.activation().scale_layout(), scale_layout);

        let actual_residual = fused
            .residual()
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<bf16>()?;
        let actual_normalized = normalized
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<bf16>()?;
        let actual_quantized = fused
            .activation()
            .quantized()
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<F8E4M3>()?;
        let actual_scales = fused
            .activation()
            .scales()
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;

        let mut expected_residual = Vec::with_capacity(elements);
        let mut expected_normalized = Vec::with_capacity(elements);
        for row in 0..rows {
            let row_start = row * COLUMNS;
            let mut square_sum = 0.0f32;
            for column in 0..COLUMNS {
                let index = row_start + column;
                let sum = input_values[index].to_f32() + residual_values[index].to_f32();
                let rounded = bf16::from_f32(sum);
                square_sum += rounded.to_f32() * rounded.to_f32();
                expected_residual.push(rounded);
            }
            let inverse_rms = (square_sum / COLUMNS as f32 + EPSILON).sqrt().recip();
            for column in 0..COLUMNS {
                let index = row_start + column;
                expected_normalized.push(bf16::from_f32(
                    expected_residual[index].to_f32()
                        * inverse_rms
                        * weight_values[column].to_f32(),
                ));
            }
        }
        assert_eq!(actual_residual, expected_residual);
        for (index, (actual, expected)) in actual_normalized
            .iter()
            .zip(&expected_normalized)
            .enumerate()
        {
            let actual = actual.to_f32();
            let expected = expected.to_f32();
            let tolerance = 1.0e-4 + expected.abs() * 1.0e-2;
            assert!(
                (actual - expected).abs() <= tolerance,
                "rows={rows}, index={index}: expected normalized {expected}, got {actual}"
            );
        }

        let scale_stride = rows.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
        assert_eq!(actual_scales.len(), COLUMNS / GROUP_SIZE);
        assert!(actual_scales
            .iter()
            .all(|scales| scales.len() == scale_stride));
        for group in 0..COLUMNS / GROUP_SIZE {
            for row in 0..rows {
                let start = row * COLUMNS + group * GROUP_SIZE;
                let maximum = actual_normalized[start..start + GROUP_SIZE]
                    .iter()
                    .map(|value| value.to_f32().abs())
                    .fold(0.0f32, f32::max);
                let expected_scale = if maximum == 0.0 {
                    1.0
                } else {
                    maximum / FP8_MAX
                };
                let actual_scale = actual_scales[group][row];
                let scale_tolerance = 1.0e-7 + expected_scale.abs() * 2.0e-3;
                assert!(
                        (actual_scale - expected_scale).abs() <= scale_tolerance,
                        "rows={rows}, row={row}, group={group}: expected scale {expected_scale}, got {actual_scale}"
                    );
                for column in group * GROUP_SIZE..(group + 1) * GROUP_SIZE {
                    let index = row * COLUMNS + column;
                    let actual = actual_quantized[index].to_f32() * actual_scale;
                    let expected = actual_normalized[index].to_f32();
                    let tolerance = 1.0e-4 + 0.09 * expected.abs().max(maximum / 64.0);
                    assert!(
                            (actual - expected).abs() <= tolerance,
                            "rows={rows}, row={row}, column={column}: expected {expected}, got {actual}"
                        );
                }
            }
            assert!(actual_scales[group][rows..scale_stride]
                .iter()
                .all(|scale| *scale == 0.0));
        }
    }
    Ok(())
}

#[test]
fn test_fp8_blockwise_dequant() -> Result<()> {
    let dev = &Device::Cpu;
    let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
    let weight_block_size = vec![2, 2];
    let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

    let dequant =
        ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::F32)?;

    let res = dequant.to_vec2::<f32>()?;
    assert_eq!(
        res,
        vec![
            vec![0., 0., 1., 1., 2.],
            vec![0., 0., 1., 1., 2.],
            vec![3., 3., 4., 4., 5.],
            vec![3., 3., 4., 4., 5.],
            vec![6., 6., 7., 7., 8.],
        ]
    );

    Ok(())
}

#[test]
fn test_fp8_blockwise_dequant_rectangular_blocks() -> Result<()> {
    let dev = &Device::Cpu;
    let weight = Tensor::ones((5, 7), DType::F8E4M3, dev)?;
    let inv_scales = Tensor::arange(0f32, 9f32, dev)?.reshape((3, 3))?;
    let dequant = ops::fp8_blockwise_dequantize(&weight, &inv_scales, vec![2, 3], DType::F32)?;

    assert_eq!(
        dequant.to_vec2::<f32>()?,
        vec![
            vec![0., 0., 0., 1., 1., 1., 2.],
            vec![0., 0., 0., 1., 1., 1., 2.],
            vec![3., 3., 3., 4., 4., 4., 5.],
            vec![3., 3., 3., 4., 4., 4., 5.],
            vec![6., 6., 6., 7., 7., 7., 8.],
        ]
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_fp8_blockwise_dequant_cuda() -> Result<()> {
    let truth = {
        let dev = &Device::Cpu;
        let weight = Tensor::ones((5, 7), DType::F8E4M3, dev)?;
        let weight_block_size = vec![2, 3];
        let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

        let dequant =
            ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::F32)?;

        dequant.to_vec2::<f32>()?
    };
    let test = {
        let dev = &Device::new_cuda(0)?;
        // Create FP8 weight by first creating on CPU then moving to CUDA
        let weight_cpu = Tensor::ones((5, 7), DType::F8E4M3, &Device::Cpu)?;
        let weight = weight_cpu.to_device(dev)?;
        let weight_block_size = vec![2, 3];
        let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

        let dequant =
            ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::F32)?;

        dequant.to_vec2::<f32>()?
    };

    assert_eq!(test, truth);
    assert_eq!(
        test,
        vec![
            vec![0., 0., 0., 1., 1., 1., 2.],
            vec![0., 0., 0., 1., 1., 1., 2.],
            vec![3., 3., 3., 4., 4., 4., 5.],
            vec![3., 3., 3., 4., 4., 4., 5.],
            vec![6., 6., 6., 7., 7., 7., 8.],
        ]
    );

    Ok(())
}

#[test]
fn test_fp8_blockwise_dequant_bf16() -> Result<()> {
    let dev = &Device::Cpu;
    let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
    let weight_block_size = vec![2, 2];
    let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

    let dequant =
        ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::BF16)?;

    let res = dequant.to_vec2::<bf16>()?;
    assert_eq!(
        res,
        vec![
            vec![
                bf16::from_f32(0.),
                bf16::from_f32(0.),
                bf16::from_f32(1.),
                bf16::from_f32(1.),
                bf16::from_f32(2.)
            ],
            vec![
                bf16::from_f32(0.),
                bf16::from_f32(0.),
                bf16::from_f32(1.),
                bf16::from_f32(1.),
                bf16::from_f32(2.)
            ],
            vec![
                bf16::from_f32(3.),
                bf16::from_f32(3.),
                bf16::from_f32(4.),
                bf16::from_f32(4.),
                bf16::from_f32(5.)
            ],
            vec![
                bf16::from_f32(3.),
                bf16::from_f32(3.),
                bf16::from_f32(4.),
                bf16::from_f32(4.),
                bf16::from_f32(5.)
            ],
            vec![
                bf16::from_f32(6.),
                bf16::from_f32(6.),
                bf16::from_f32(7.),
                bf16::from_f32(7.),
                bf16::from_f32(8.)
            ],
        ]
    );

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_fp8_blockwise_dequant_cuda_bf16() -> Result<()> {
    let truth = {
        let dev = &Device::Cpu;
        let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
        let weight_block_size = vec![2, 2];
        let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

        let dequant =
            ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::BF16)?;

        dequant.to_vec2::<bf16>()?
    };
    let test = {
        let dev = &Device::new_cuda(0)?;
        // Create FP8 weight by first creating on CPU then moving to CUDA
        let weight_cpu = Tensor::ones((5, 5), DType::F8E4M3, &Device::Cpu)?;
        let weight = weight_cpu.to_device(dev)?;
        let weight_block_size = vec![2, 2];
        let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

        let dequant =
            ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::BF16)?;

        dequant.to_vec2::<bf16>()?
    };

    assert_eq!(test, truth);
    assert_eq!(
        test,
        vec![
            vec![
                bf16::from_f32(0.),
                bf16::from_f32(0.),
                bf16::from_f32(1.),
                bf16::from_f32(1.),
                bf16::from_f32(2.)
            ],
            vec![
                bf16::from_f32(0.),
                bf16::from_f32(0.),
                bf16::from_f32(1.),
                bf16::from_f32(1.),
                bf16::from_f32(2.)
            ],
            vec![
                bf16::from_f32(3.),
                bf16::from_f32(3.),
                bf16::from_f32(4.),
                bf16::from_f32(4.),
                bf16::from_f32(5.)
            ],
            vec![
                bf16::from_f32(3.),
                bf16::from_f32(3.),
                bf16::from_f32(4.),
                bf16::from_f32(4.),
                bf16::from_f32(5.)
            ],
            vec![
                bf16::from_f32(6.),
                bf16::from_f32(6.),
                bf16::from_f32(7.),
                bf16::from_f32(7.),
                bf16::from_f32(8.)
            ],
        ]
    );

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_fp8_blockwise_quant_dequant_roundtrip() -> Result<()> {
    let dev = &Device::new_cuda(0)?;

    // Create test input
    let input = Tensor::randn(0f32, 2f32, (8, 8), dev)?;
    let weight_block_size = vec![4, 4];

    // Quantize
    let (quantized, scales) = ops::fp8_blockwise_quantize(&input, weight_block_size.clone())?;

    // Verify shapes
    assert_eq!(quantized.shape(), input.shape());
    assert_eq!(scales.dims2()?, (2, 2)); // 8/4 = 2 blocks in each dimension

    // Dequantize
    let dequantized =
        ops::fp8_blockwise_dequantize(&quantized, &scales, weight_block_size, input.dtype())?;

    // Check that shapes match
    assert_eq!(dequantized.shape(), input.shape());

    // The values won't be exactly the same due to quantization loss,
    // but they should be reasonably close
    let input_vec = input.to_vec2::<f32>()?;
    let dequant_vec = dequantized.to_vec2::<f32>()?;

    let mut max_error = 0f32;
    for (row_in, row_out) in input_vec.iter().zip(dequant_vec.iter()) {
        for (val_in, val_out) in row_in.iter().zip(row_out.iter()) {
            let error = (val_in - val_out).abs();
            max_error = max_error.max(error);
        }
    }

    // FP8 E4M3 has limited precision, so we expect some error
    // but it should be reasonable
    assert!(max_error < 0.16, "Max error {} is too large", max_error);

    Ok(())
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
#[test]
fn test_cutlass_blockwise_fp8_gemm() -> Result<()> {
    const BLOCK_SIZE: usize = 128;
    const FP8_E4M3_MAX: f32 = 448.0;
    const SCALE_ABS_TOLERANCE: f32 = 1.0e-8;
    const SCALE_REL_TOLERANCE: f32 = 1.0e-5;
    const K: usize = 256;
    const N: usize = 256;

    let dev = Device::new_cuda(0)?;
    let weight_values = (0..N * K)
        .map(|index| ((index * 7 + index / K * 13) % 23) as f32 * 0.03 - 0.33)
        .collect::<Vec<_>>();
    let weight = Tensor::from_vec(weight_values, (N, K), &dev)?.to_dtype(DType::BF16)?;
    let (weight_q, weight_scales) = ops::fp8_blockwise_quantize(&weight, vec![128, 128])?;
    assert!(ops::cutlass_fp8_blockwise_supported(
        &weight_q,
        &weight_scales,
        &[128, 128]
    ));
    let reference_weight = weight.to_dtype(DType::F32)?.t()?;

    for rows in [1usize, 8] {
        for dtype in [DType::F16, DType::BF16] {
            let k_blocks = K / BLOCK_SIZE;
            let input_values = (0..rows * K)
                .map(|index| {
                    let row = index / K;
                    let k_block = index % K / BLOCK_SIZE;
                    (row * k_blocks + k_block + 1) as f32 / 16.0
                })
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(input_values, (rows, K), &dev)?.to_dtype(dtype)?;
            let reference = input.to_dtype(DType::F32)?.matmul(&reference_weight)?;
            let (input_q, input_scales) = ops::fp8_quantize_activation_cutlass(&input)?;
            let input_scale_values = input_scales.flatten_all()?.to_vec1::<f32>()?;
            let expected_scales = (0..k_blocks)
                .flat_map(|k_block| {
                    (0..rows)
                        .map(move |row| (row * k_blocks + k_block + 1) as f32 / 16.0 / FP8_E4M3_MAX)
                })
                .collect::<Vec<_>>();
            for (index, (actual, expected)) in input_scale_values
                .iter()
                .zip(expected_scales.iter())
                .enumerate()
            {
                let tolerance = SCALE_ABS_TOLERANCE + SCALE_REL_TOLERANCE * expected.abs();
                assert!(
                        (actual - expected).abs() <= tolerance,
                        "rows={rows}, dtype={dtype:?}, scale index {index}: expected {expected}, got {actual}"
                    );
            }
            let output = ops::fp8_blockwise_matmul_cutlass(
                &input_q,
                &input_scales,
                &weight_q,
                &weight_scales,
                dtype,
            )?
            .to_dtype(DType::F32)?;

            let reference = reference.flatten_all()?.to_vec1::<f32>()?;
            let output = output.flatten_all()?.to_vec1::<f32>()?;
            let max_reference = reference.iter().copied().map(f32::abs).fold(0f32, f32::max);
            let max_error = reference
                .iter()
                .zip(output.iter())
                .map(|(reference, output)| (reference - output).abs())
                .fold(0f32, f32::max);
            assert!(
                    max_error <= 0.12 + 0.08 * max_reference,
                    "rows={rows}, dtype={dtype:?}, max error {max_error}, max reference {max_reference}"
                );
        }
    }
    Ok(())
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
#[test]
#[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
fn test_deepgemm_blockwise_fp8_and_cuda_graph() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    const BLOCK_SIZE: usize = 128;
    const K: usize = 256;
    const N: usize = 256;

    fn assert_close(rows: usize, reference: &Tensor, output: &Tensor) -> Result<()> {
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;
        let max_reference = reference.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let max_error = reference
            .iter()
            .zip(&output)
            .map(|(reference, output)| (reference - output).abs())
            .fold(0.0, f32::max);
        assert!(
            max_error <= 0.02 + 0.02 * max_reference,
            "rows={rows}: max error {max_error}, max reference {max_reference}"
        );
        Ok(())
    }

    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cuda_dev) = &dev else {
        unreachable!()
    };
    let stream = cuda_dev.cuda_stream();
    unsafe { stream.context().disable_event_tracking() };
    let weight_values = (0..N * K)
        .map(|index| {
            let row = index / K;
            let column = index % K;
            let block = row / BLOCK_SIZE * 2 + column / BLOCK_SIZE;
            let amplitude = [0.04, 0.18, 0.75, 1.6][block];
            let value = ((row * 17 + column * 29) % 31) as f32 - 15.0;
            value * amplitude
        })
        .collect::<Vec<_>>();
    let weight = Tensor::from_vec(weight_values, (N, K), &dev)?.to_dtype(DType::BF16)?;
    let (weight, weight_scales) =
        ops::fp8_blockwise_quantize(&weight, vec![BLOCK_SIZE, BLOCK_SIZE])?;
    let weight_scale_values = weight_scales.flatten_all()?.to_vec1::<f32>()?;
    let min_scale = weight_scale_values
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_scale = weight_scale_values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(max_scale > min_scale * 8.0);
    assert!(deepgemm::supported(
        &weight,
        &weight_scales,
        &[BLOCK_SIZE, BLOCK_SIZE]
    ));
    let live_event = stream
        .record_event(None)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    stream
        .wait(&live_event)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    dev.synchronize()?;
    let prepared = deepgemm::prepare(&weight, &weight_scales, &[BLOCK_SIZE, BLOCK_SIZE])?;
    stream
        .wait(&live_event)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    dev.synchronize()?;
    for rows in [
        1usize, 8, 16, 21, 24, 28, 35, 42, 49, 129, 192, 255, 257, 511, 513, 1023, 1920, 2304, 4095,
    ] {
        let input_values = (0..rows * K)
            .map(|index| {
                let row = index / K;
                let column = index % K;
                let block = column / BLOCK_SIZE;
                let amplitude = (row + 1) as f32 * (block + 1) as f32 * 0.025;
                let value = ((row * 11 + column * 7) % 29) as f32 - 14.0;
                value * amplitude
            })
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input_values, (rows, K), &dev)?.to_dtype(DType::BF16)?;
        let (activation, activation_scales) = ops::fp8_quantize_activation_cutlass(&input)?;
        dev.synchronize()?;
        let reference = ops::fp8_blockwise_matmul_cutlass(
            &activation,
            &activation_scales,
            &weight,
            &weight_scales,
            DType::BF16,
        )?;
        dev.synchronize()?;
        let reference = reference.to_dtype(DType::F32)?;
        dev.synchronize()?;
        let output = deepgemm::matmul(&prepared, &input, &weight, &weight_scales)?;
        dev.synchronize()?;
        let output = output.to_dtype(DType::F32)?;
        dev.synchronize()?;
        assert_close(rows, &reference, &output)?;
        dev.synchronize()?;

        if rows != 21 {
            continue;
        }

        let graph_output = Tensor::zeros((rows, N), DType::BF16, &dev)?;
        let restore_event_tracking = stream.context().is_event_tracking();
        if restore_event_tracking {
            unsafe { stream.context().disable_event_tracking() };
        }
        if let Err(error) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            if restore_event_tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            return Err(candle_core::Error::msg(error.to_string()));
        }
        let captured =
            deepgemm::matmul(&prepared, &input, &weight, &weight_scales).and_then(|captured| {
                use crate::utils::slice_ptr_on_stream;
                use candle_core::Storage;

                let status = {
                    let (src_storage, src_layout) = captured.storage_and_layout();
                    let Storage::Cuda(src_storage) = &*src_storage else {
                        unreachable!()
                    };
                    let (dst_storage, dst_layout) = graph_output.storage_and_layout();
                    let Storage::Cuda(dst_storage) = &*dst_storage else {
                        unreachable!()
                    };
                    let (src_ptr, src_guard) = slice_ptr_on_stream(
                        src_storage.as_cuda_slice::<bf16>()?,
                        src_layout.start_offset(),
                        &stream,
                    );
                    let (dst_ptr, dst_guard) = slice_ptr_on_stream(
                        dst_storage.as_cuda_slice::<bf16>()?,
                        dst_layout.start_offset(),
                        &stream,
                    );
                    let status = unsafe {
                        sys::cuMemcpyDtoDAsync_v2(
                            dst_ptr,
                            src_ptr,
                            rows * N * std::mem::size_of::<bf16>(),
                            stream.cu_stream(),
                        )
                    };
                    drop((src_guard, dst_guard));
                    status
                };
                drop(captured);
                if status != sys::cudaError_enum::CUDA_SUCCESS {
                    candle_core::bail!("CUDA graph output copy failed: {status:?}")
                }
                Ok(())
            });
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if restore_event_tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        captured?;
        let graph = graph
            .map_err(|error| candle_core::Error::msg(error.to_string()))?
            .ok_or_else(|| candle_core::Error::msg("CUDA graph capture returned no graph"))?;
        graph
            .launch()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        graph
            .launch()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        stream
            .synchronize()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        assert_close(rows, &reference, &graph_output.to_dtype(DType::F32)?)?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
#[test]
#[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
fn test_deepgemm_shared_prepared_state_across_ptds_threads() -> Result<()> {
    const BLOCK_SIZE: usize = 128;
    const ITERATIONS: usize = 32;
    const K: usize = 256;
    const N: usize = 256;
    const ROWS: usize = 8;

    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cuda_dev) = &dev else {
        unreachable!()
    };
    unsafe { cuda_dev.cuda_stream().context().disable_event_tracking() };
    let weight = Tensor::ones((N, K), DType::F8E4M3, &Device::Cpu)?.to_device(&dev)?;
    let weight_scales = Tensor::ones((N / BLOCK_SIZE, K / BLOCK_SIZE), DType::F32, &dev)?
        .affine(1.0 / K as f64, 0.0)?;
    let prepared = deepgemm::prepare(&weight, &weight_scales, &[BLOCK_SIZE, BLOCK_SIZE])?;
    dev.synchronize()?;

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    std::thread::scope(|scope| -> Result<()> {
        let spawn = |value: f32| {
            let dev = dev.clone();
            let weight = weight.clone();
            let weight_scales = weight_scales.clone();
            let prepared = prepared.clone();
            let barrier = barrier.clone();
            scope.spawn(move || -> Result<()> {
                let values = vec![bf16::from_f32(value); ROWS * K];
                let input = Tensor::from_vec(values, (ROWS, K), &dev)?;
                barrier.wait();
                let mut output = None;
                for _ in 0..ITERATIONS {
                    output = Some(deepgemm::matmul(
                        &prepared,
                        &input,
                        &weight,
                        &weight_scales,
                    )?);
                    std::thread::yield_now();
                }
                dev.synchronize()?;
                let output = output.unwrap().to_dtype(DType::F32)?;
                let max_error = output
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(|output| (output - value).abs())
                    .fold(0.0, f32::max);
                assert!(max_error <= 0.02, "value={value}: max error {max_error}");
                Ok(())
            })
        };
        let first = spawn(0.5);
        let second = spawn(2.0);
        first.join().expect("first PTDS worker panicked")?;
        second.join().expect("second PTDS worker panicked")?;
        Ok(())
    })
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
#[derive(Clone, Copy)]
struct BlockwiseFp8BenchShape {
    name: &'static str,
    n: usize,
    k: usize,
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
fn blockwise_fp8_bench_iterations(variable: &str, default: usize) -> Result<usize> {
    match std::env::var(variable) {
        Ok(value) => value
            .parse()
            .ok()
            .filter(|value| *value != 0)
            .ok_or_else(|| {
                candle_core::Error::msg(format!("{variable} must be a positive integer"))
            }),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(candle_core::Error::msg(format!(
            "failed to read {variable}: {error}"
        ))),
    }
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
fn measure_blockwise_fp8_cuda_us<T>(
    dev: &Device,
    warmup: usize,
    iterations: usize,
    mut launch: impl FnMut() -> Result<T>,
) -> Result<f64> {
    use candle_core::cuda::cudarc::driver::sys;

    for _ in 0..warmup {
        drop(launch()?);
    }
    dev.synchronize()?;

    let stream = dev.as_cuda_device()?.cuda_stream();
    let start = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| candle_core::Error::msg(format!("CUDA start event failed: {error}")))?;
    for _ in 0..iterations {
        drop(launch()?);
    }
    let end = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| candle_core::Error::msg(format!("CUDA end event failed: {error}")))?;
    end.synchronize().map_err(|error| {
        candle_core::Error::msg(format!("CUDA event synchronization failed: {error}"))
    })?;
    let elapsed_ms = start
        .elapsed_ms(&end)
        .map_err(|error| candle_core::Error::msg(format!("CUDA timing failed: {error}")))?;
    Ok(f64::from(elapsed_ms) * 1_000.0 / iterations as f64)
}

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
fn measure_blockwise_fp8_cuda_graph_us<T>(
    dev: &Device,
    warmup: usize,
    iterations: usize,
    samples: usize,
    capture: impl FnOnce() -> Result<T>,
) -> Result<Vec<f64>> {
    use candle_core::cuda::cudarc::driver::sys;

    dev.synchronize()?;
    let Device::Cuda(cuda_dev) = dev else {
        candle_core::bail!("CUDA graph timing requires a CUDA device")
    };
    let stream = cuda_dev.cuda_stream();
    let restore_event_tracking = stream.context().is_event_tracking();
    if restore_event_tracking {
        unsafe { stream.context().disable_event_tracking() };
    }
    if let Err(error) =
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
    {
        if restore_event_tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        return Err(candle_core::Error::msg(error.to_string()));
    }
    let captured = capture();
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if restore_event_tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let captured = captured?;
    let graph = graph
        .map_err(|error| candle_core::Error::msg(error.to_string()))?
        .ok_or_else(|| candle_core::Error::msg("CUDA graph capture returned no graph"))?;
    for _ in 0..warmup {
        graph
            .launch()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    }
    stream
        .synchronize()
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;

    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| {
                candle_core::Error::msg(format!("CUDA start event failed: {error}"))
            })?;
        for _ in 0..iterations {
            graph
                .launch()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        }
        let end = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| candle_core::Error::msg(format!("CUDA end event failed: {error}")))?;
        end.synchronize().map_err(|error| {
            candle_core::Error::msg(format!("CUDA event synchronization failed: {error}"))
        })?;
        let elapsed_ms = start
            .elapsed_ms(&end)
            .map_err(|error| candle_core::Error::msg(format!("CUDA timing failed: {error}")))?;
        timings.push(f64::from(elapsed_ms) * 1_000.0 / iterations as f64);
    }
    drop((captured, graph));
    Ok(timings)
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
fn validate_blockwise_fp8_bench_case(
    activation_scales: &Tensor,
    output: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<f32> {
    const FP8_E4M3_MAX: f32 = 448.0;
    const SCALE_TOLERANCE: f32 = 1.0e-7;
    const OUTPUT_TOLERANCE: f32 = 0.02;

    let expected_scale = 1.0 / FP8_E4M3_MAX;
    let scale_error = activation_scales
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|scale| (scale - expected_scale).abs())
        .fold(0.0, f32::max);
    assert!(
        scale_error <= SCALE_TOLERANCE,
        "M={m}, N={n}, K={k}: activation scale error {scale_error}"
    );

    assert_eq!(output.dims2()?, (m, n));
    let output_error = output
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|value| (value - 1.0).abs())
        .fold(0.0, f32::max);
    assert!(
        output_error <= OUTPUT_TOLERANCE,
        "M={m}, N={n}, K={k}: output error {output_error}"
    );
    Ok(output_error)
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
#[test]
#[ignore = "requires an SM90 GPU and reports latency to stdout"]
fn bench_cutlass_blockwise_fp8_production_shapes_sm90() -> Result<()> {
    const BLOCK_SIZE: usize = 128;
    const DEFAULT_WARMUP: usize = 10;
    const DEFAULT_ITERATIONS: usize = 100;
    const M_VALUES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64];
    const SHAPES: [BlockwiseFp8BenchShape; 5] = [
        BlockwiseFp8BenchShape {
            name: "gdn_qkvz",
            n: 16_384,
            k: 5_120,
        },
        BlockwiseFp8BenchShape {
            name: "attention_qkv",
            n: 14_336,
            k: 5_120,
        },
        BlockwiseFp8BenchShape {
            name: "output_projection",
            n: 5_120,
            k: 6_144,
        },
        BlockwiseFp8BenchShape {
            name: "mlp_gate_up",
            n: 34_816,
            k: 5_120,
        },
        BlockwiseFp8BenchShape {
            name: "mlp_down",
            n: 5_120,
            k: 17_408,
        },
    ];

    let warmup =
        blockwise_fp8_bench_iterations("MISTRALRS_BLOCKWISE_FP8_BENCH_WARMUP", DEFAULT_WARMUP)?;
    let iterations = blockwise_fp8_bench_iterations(
        "MISTRALRS_BLOCKWISE_FP8_BENCH_ITERATIONS",
        DEFAULT_ITERATIONS,
    )?;
    let dev = Device::new_cuda(0)?;
    println!("shape,m,n,k,quantize_gpu_us,gemm_gpu_us,max_abs_error,warmup,iterations");

    for shape in SHAPES {
        let weight =
            Tensor::ones((shape.n, shape.k), DType::F8E4M3, &Device::Cpu)?.to_device(&dev)?;
        let weight_scales = Tensor::ones(
            (shape.n / BLOCK_SIZE, shape.k / BLOCK_SIZE),
            DType::F32,
            &dev,
        )?
        .affine(1.0 / shape.k as f64, 0.0)?;
        assert!(ops::cutlass_fp8_blockwise_supported(
            &weight,
            &weight_scales,
            &[BLOCK_SIZE, BLOCK_SIZE]
        ));

        for m in M_VALUES {
            let input = Tensor::ones((m, shape.k), DType::BF16, &dev)?;
            let quantize_us = measure_blockwise_fp8_cuda_us(&dev, warmup, iterations, || {
                ops::fp8_quantize_activation_cutlass(&input)
            })?;
            let (activation, activation_scales) = ops::fp8_quantize_activation_cutlass(&input)?;
            let gemm_us = measure_blockwise_fp8_cuda_us(&dev, warmup, iterations, || {
                ops::fp8_blockwise_matmul_cutlass(
                    &activation,
                    &activation_scales,
                    &weight,
                    &weight_scales,
                    DType::BF16,
                )
            })?;
            let output = ops::fp8_blockwise_matmul_cutlass(
                &activation,
                &activation_scales,
                &weight,
                &weight_scales,
                DType::BF16,
            )?;
            let max_error = validate_blockwise_fp8_bench_case(
                &activation_scales,
                &output,
                m,
                shape.n,
                shape.k,
            )?;
            println!(
                "{},{},{},{},{quantize_us:.3},{gemm_us:.3},{max_error:.6},{warmup},{iterations}",
                shape.name, m, shape.n, shape.k
            );
        }
    }
    Ok(())
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
#[test]
#[ignore = "requires an SM90 GPU and reports latency to stdout"]
fn bench_deepgemm_blockwise_fp8_production_shapes_sm90() -> Result<()> {
    const BLOCK_SIZE: usize = 128;
    const DEFAULT_WARMUP: usize = 10;
    const DEFAULT_ITERATIONS: usize = 100;
    const OUTPUT_TOLERANCE: f32 = 0.02;
    const M_VALUES: [usize; 14] = [
        1, 8, 16, 21, 24, 32, 64, 128, 192, 256, 512, 2304, 3840, 4095,
    ];
    const SHAPES: [BlockwiseFp8BenchShape; 5] = [
        BlockwiseFp8BenchShape {
            name: "gdn_qkvz",
            n: 16_384,
            k: 5_120,
        },
        BlockwiseFp8BenchShape {
            name: "attention_qkv",
            n: 14_336,
            k: 5_120,
        },
        BlockwiseFp8BenchShape {
            name: "output_projection",
            n: 5_120,
            k: 6_144,
        },
        BlockwiseFp8BenchShape {
            name: "mlp_gate_up",
            n: 34_816,
            k: 5_120,
        },
        BlockwiseFp8BenchShape {
            name: "mlp_down",
            n: 5_120,
            k: 17_408,
        },
    ];

    let warmup =
        blockwise_fp8_bench_iterations("MISTRALRS_BLOCKWISE_FP8_BENCH_WARMUP", DEFAULT_WARMUP)?;
    let iterations = blockwise_fp8_bench_iterations(
        "MISTRALRS_BLOCKWISE_FP8_BENCH_ITERATIONS",
        DEFAULT_ITERATIONS,
    )?;
    let dev = Device::new_cuda(0)?;
    println!("shape,m,n,k,fused_gpu_us,max_abs_error,warmup,iterations");

    for shape in SHAPES {
        let weight =
            Tensor::ones((shape.n, shape.k), DType::F8E4M3, &Device::Cpu)?.to_device(&dev)?;
        let weight_scales = Tensor::ones(
            (shape.n / BLOCK_SIZE, shape.k / BLOCK_SIZE),
            DType::F32,
            &dev,
        )?
        .affine(1.0 / shape.k as f64, 0.0)?;
        let prepared = deepgemm::prepare(&weight, &weight_scales, &[BLOCK_SIZE, BLOCK_SIZE])?;

        for m in M_VALUES {
            let input = Tensor::ones((m, shape.k), DType::BF16, &dev)?;
            let fused_us = measure_blockwise_fp8_cuda_us(&dev, warmup, iterations, || {
                deepgemm::matmul(&prepared, &input, &weight, &weight_scales)
            })?;
            let output = deepgemm::matmul(&prepared, &input, &weight, &weight_scales)?;
            let max_error = output
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(|value| (value - 1.0).abs())
                .fold(0.0, f32::max);
            assert!(
                max_error <= OUTPUT_TOLERANCE,
                "M={m}, N={}, K={}: output error {max_error}",
                shape.n,
                shape.k
            );
            println!(
                "{},{},{},{},{fused_us:.3},{max_error:.6},{warmup},{iterations}",
                shape.name, m, shape.n, shape.k
            );
        }
    }
    Ok(())
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
fn median_cuda_us(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
#[test]
#[ignore = "requires an SM90 GPU and reports latency to stdout"]
fn bench_cutlass_vs_deepgemm_fused_production_shapes_sm90() -> Result<()> {
    const BLOCK_SIZE: usize = 128;
    const DEFAULT_WARMUP: usize = 10;
    const DEFAULT_ITERATIONS: usize = 100;
    const SAMPLES: usize = 7;
    const M_VALUES: [usize; 14] = [
        1, 8, 16, 21, 24, 32, 64, 128, 192, 256, 512, 2304, 3840, 4095,
    ];
    const SHAPES: [BlockwiseFp8BenchShape; 5] = [
        BlockwiseFp8BenchShape {
            name: "gdn_qkvz",
            n: 16_384,
            k: 5_120,
        },
        BlockwiseFp8BenchShape {
            name: "attention_qkv",
            n: 14_336,
            k: 5_120,
        },
        BlockwiseFp8BenchShape {
            name: "output_projection",
            n: 5_120,
            k: 6_144,
        },
        BlockwiseFp8BenchShape {
            name: "mlp_gate_up",
            n: 34_816,
            k: 5_120,
        },
        BlockwiseFp8BenchShape {
            name: "mlp_down",
            n: 5_120,
            k: 17_408,
        },
    ];

    let warmup =
        blockwise_fp8_bench_iterations("MISTRALRS_BLOCKWISE_FP8_BENCH_WARMUP", DEFAULT_WARMUP)?;
    let iterations = blockwise_fp8_bench_iterations(
        "MISTRALRS_BLOCKWISE_FP8_BENCH_ITERATIONS",
        DEFAULT_ITERATIONS,
    )?;
    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cuda_dev) = &dev else {
        unreachable!()
    };
    unsafe { cuda_dev.cuda_stream().context().disable_event_tracking() };
    println!(
            "shape,m,n,k,cutlass_quant_median_us,cutlass_gemm_median_us,cutlass_fused_median_us,deepgemm_fused_median_us,deepgemm_speedup,cutlass_graph_median_us,deepgemm_graph_median_us,deepgemm_graph_speedup,samples,warmup,iterations"
        );

    for shape in SHAPES {
        let weight =
            Tensor::ones((shape.n, shape.k), DType::F8E4M3, &Device::Cpu)?.to_device(&dev)?;
        let weight_scales = Tensor::ones(
            (shape.n / BLOCK_SIZE, shape.k / BLOCK_SIZE),
            DType::F32,
            &dev,
        )?
        .affine(1.0 / shape.k as f64, 0.0)?;
        let prepared = deepgemm::prepare(&weight, &weight_scales, &[BLOCK_SIZE, BLOCK_SIZE])?;

        for m in M_VALUES {
            let input = Tensor::ones((m, shape.k), DType::BF16, &dev)?;
            let (activation, activation_scales) = ops::fp8_quantize_activation_cutlass(&input)?;
            let mut cutlass_quant = Vec::with_capacity(SAMPLES);
            let mut cutlass_gemm = Vec::with_capacity(SAMPLES);
            let mut cutlass_fused = Vec::with_capacity(SAMPLES);
            let mut deepgemm_fused = Vec::with_capacity(SAMPLES);
            for sample in 0..SAMPLES {
                if sample % 2 == 0 {
                    cutlass_fused.push(measure_blockwise_fp8_cuda_us(
                        &dev,
                        warmup,
                        iterations,
                        || {
                            let (activation, scales) =
                                ops::fp8_quantize_activation_cutlass(&input)?;
                            ops::fp8_blockwise_matmul_cutlass(
                                &activation,
                                &scales,
                                &weight,
                                &weight_scales,
                                DType::BF16,
                            )
                        },
                    )?);
                    deepgemm_fused.push(measure_blockwise_fp8_cuda_us(
                        &dev,
                        warmup,
                        iterations,
                        || deepgemm::matmul(&prepared, &input, &weight, &weight_scales),
                    )?);
                } else {
                    deepgemm_fused.push(measure_blockwise_fp8_cuda_us(
                        &dev,
                        warmup,
                        iterations,
                        || deepgemm::matmul(&prepared, &input, &weight, &weight_scales),
                    )?);
                    cutlass_fused.push(measure_blockwise_fp8_cuda_us(
                        &dev,
                        warmup,
                        iterations,
                        || {
                            let (activation, scales) =
                                ops::fp8_quantize_activation_cutlass(&input)?;
                            ops::fp8_blockwise_matmul_cutlass(
                                &activation,
                                &scales,
                                &weight,
                                &weight_scales,
                                DType::BF16,
                            )
                        },
                    )?);
                }
                cutlass_quant.push(measure_blockwise_fp8_cuda_us(
                    &dev,
                    warmup,
                    iterations,
                    || ops::fp8_quantize_activation_cutlass(&input),
                )?);
                cutlass_gemm.push(measure_blockwise_fp8_cuda_us(
                    &dev,
                    warmup,
                    iterations,
                    || {
                        ops::fp8_blockwise_matmul_cutlass(
                            &activation,
                            &activation_scales,
                            &weight,
                            &weight_scales,
                            DType::BF16,
                        )
                    },
                )?);
            }
            let cutlass_quant = median_cuda_us(cutlass_quant);
            let cutlass_gemm = median_cuda_us(cutlass_gemm);
            let cutlass_fused = median_cuda_us(cutlass_fused);
            let deepgemm_fused = median_cuda_us(deepgemm_fused);
            let cutlass_graph = median_cuda_us(measure_blockwise_fp8_cuda_graph_us(
                &dev,
                warmup,
                iterations,
                SAMPLES,
                || {
                    let (activation, scales) = ops::fp8_quantize_activation_cutlass(&input)?;
                    let output = ops::fp8_blockwise_matmul_cutlass(
                        &activation,
                        &scales,
                        &weight,
                        &weight_scales,
                        DType::BF16,
                    )?;
                    Ok((activation, scales, output))
                },
            )?);
            let deepgemm_graph = median_cuda_us(measure_blockwise_fp8_cuda_graph_us(
                &dev,
                warmup,
                iterations,
                SAMPLES,
                || deepgemm::matmul(&prepared, &input, &weight, &weight_scales),
            )?);
            println!(
                    "{},{},{},{},{cutlass_quant:.3},{cutlass_gemm:.3},{cutlass_fused:.3},{deepgemm_fused:.3},{:.3},{cutlass_graph:.3},{deepgemm_graph:.3},{:.3},{SAMPLES},{warmup},{iterations}",
                    shape.name,
                    m,
                    shape.n,
                    shape.k,
                    cutlass_fused / deepgemm_fused,
                    cutlass_graph / deepgemm_graph
                );
        }
    }
    Ok(())
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
#[test]
fn test_cutlass_blockwise_fp8_cuda_graph() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    const K: usize = 128;
    const N: usize = 128;
    const ROWS: usize = 8;

    let dev = Device::new_cuda(0)?;
    let weight = Tensor::randn(0f32, 0.25, (N, K), &dev)?.to_dtype(DType::BF16)?;
    let input = Tensor::randn(0f32, 0.25, (ROWS, K), &dev)?.to_dtype(DType::BF16)?;
    let (weight_q, weight_scales) = ops::fp8_blockwise_quantize(&weight, vec![128, 128])?;

    let (warmup_q, warmup_scales) = ops::fp8_quantize_activation_cutlass(&input)?;
    let warmup_output = ops::fp8_blockwise_matmul_cutlass(
        &warmup_q,
        &warmup_scales,
        &weight_q,
        &weight_scales,
        DType::BF16,
    )?;
    drop((warmup_q, warmup_scales, warmup_output));
    dev.synchronize()?;

    let Device::Cuda(cuda_dev) = &dev else {
        unreachable!()
    };
    let stream = cuda_dev.cuda_stream();
    let restore_event_tracking = stream.context().is_event_tracking();
    if restore_event_tracking {
        unsafe { stream.context().disable_event_tracking() };
    }
    if let Err(error) =
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
    {
        if restore_event_tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        return Err(candle_core::Error::msg(error.to_string()));
    }

    let captured =
        ops::fp8_quantize_activation_cutlass(&input).and_then(|(activation, activation_scales)| {
            let output = ops::fp8_blockwise_matmul_cutlass(
                &activation,
                &activation_scales,
                &weight_q,
                &weight_scales,
                DType::BF16,
            )?;
            drop((activation, activation_scales, output));
            Ok(())
        });
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if restore_event_tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    captured?;
    let graph = graph
        .map_err(|error| candle_core::Error::msg(error.to_string()))?
        .ok_or_else(|| candle_core::Error::msg("CUDA graph capture returned no graph"))?;
    graph
        .launch()
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    graph
        .launch()
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    stream
        .synchronize()
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    Ok(())
}
