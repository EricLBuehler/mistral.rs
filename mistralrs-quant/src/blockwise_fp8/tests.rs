use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

use super::*;
use crate::{QuantizedConfig, ShardedSafeTensors};

fn routed_biases(device: &Device, dtype: DType) -> Result<()> {
    const FEATURES: usize = 128;
    const TOPK: usize = 2;
    for experts in [2, 4] {
        for per_expert in [false, true] {
            let bias = if per_expert {
                Tensor::from_vec(
                    (1..=experts)
                        .flat_map(|expert| vec![expert as f32; FEATURES])
                        .collect::<Vec<_>>(),
                    (experts, FEATURES),
                    device,
                )?
            } else {
                Tensor::full(5f32, FEATURES, device)?
            };
            let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
                weight: Tensor::zeros((experts, FEATURES, FEATURES), DType::F8E4M3, device)?,
                weight_scale_inv: Tensor::ones((experts, 1, 1), DType::F32, device)?,
                bias: Some(bias.to_dtype(dtype)?),
                dequant_dtype: dtype,
                weight_block_size: vec![FEATURES, FEATURES],
                activation_scheme: None,
            })?;
            for tokens in [1, 32] {
                let ids = Tensor::from_vec(
                    (0..tokens)
                        .flat_map(|_| [(experts - 1) as u32, 0])
                        .collect::<Vec<_>>(),
                    (tokens, TOPK),
                    device,
                )?;
                for input_routes in [1, TOPK] {
                    let input = Tensor::zeros((tokens, input_routes, FEATURES), dtype, device)?;
                    let result = layer.gather_forward(&input, &ids)?.to_dtype(DType::F32)?;
                    for token in result.to_vec3::<f32>()? {
                        assert_eq!(
                            token[0],
                            vec![if per_expert { experts as f32 } else { 5.0 }; FEATURES]
                        );
                        assert_eq!(token[1], vec![if per_expert { 1.0 } else { 5.0 }; FEATURES]);
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn cpu_routed_biases_follow_expert_ids() -> Result<()> {
    routed_biases(&Device::Cpu, DType::F32)
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_routed_biases_survive_gemv_and_grouped_dispatch() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for dtype in [DType::F16, DType::BF16] {
        routed_biases(&device, dtype)?;
    }
    Ok(())
}

fn fp8_config(exclusions: &[&str]) -> QuantizedConfig {
    QuantizedConfig::Fp8 {
        weight_block_size: Some(vec![128, 128]),
        activation_scheme: Some(Fp8ActivationScheme::Dynamic),
        fmt: Some("e4m3".to_string()),
        modules_to_not_convert: exclusions
            .iter()
            .map(|module| (*module).to_string())
            .collect(),
    }
}

#[cfg(all(
    feature = "cuda",
    has_cutlass_fp8_sm90_kernels,
    has_deepgemm_fp8_sm90_provider
))]
fn deepgemm_fused_glu_test_layer(
    dev: &Device,
    output_features: usize,
    input_features: usize,
) -> Result<BlockwiseFP8Linear> {
    const BLOCK_SIZE: usize = 128;

    let weight_values = (0..output_features * input_features)
        .map(|index| {
            let row = index / input_features;
            let column = index % input_features;
            let block = row / BLOCK_SIZE * (input_features / BLOCK_SIZE) + column / BLOCK_SIZE;
            let amplitude = [0.04, 0.18, 0.75, 1.6][block % 4];
            let value = ((row * 17 + column * 29) % 31) as f32 - 15.0;
            value * amplitude
        })
        .collect::<Vec<_>>();
    let weight = Tensor::from_vec(weight_values, (output_features, input_features), dev)?
        .to_dtype(DType::BF16)?;
    let (weight, weight_scale_inv) =
        ops::fp8_blockwise_quantize(&weight, vec![BLOCK_SIZE, BLOCK_SIZE])?;
    let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
        weight,
        weight_scale_inv,
        bias: None,
        dequant_dtype: DType::BF16,
        weight_block_size: vec![BLOCK_SIZE, BLOCK_SIZE],
        activation_scheme: Some(Fp8ActivationScheme::Dynamic),
    })?;
    assert!(matches!(
        &layer.provider,
        BlockwiseFp8Provider::DeepGemmSm90(_)
    ));
    Ok(layer)
}

#[cfg(all(
    feature = "cuda",
    has_cutlass_fp8_sm90_kernels,
    has_deepgemm_fp8_sm90_provider
))]
fn deepgemm_fused_glu_test_input(
    rows: usize,
    split_size: usize,
    phase: usize,
    dev: &Device,
) -> Result<Tensor> {
    const GROUP_SIZE: usize = 128;

    let packed_features = split_size * 2;
    let values = (0..rows * packed_features)
        .map(|index| {
            let row = index / packed_features;
            let packed_column = index % packed_features;
            let column = packed_column % split_size;
            let half = packed_column / split_size;
            let amplitude =
                [0.2, 0.75][column / GROUP_SIZE % 2] * if half == 0 { 1.0 } else { 0.6 };
            let value = ((row * 37 + column * 19 + half * 11 + phase * 23) % 101) as f32 - 50.0;
            value * amplitude / 17.0
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (rows, packed_features), dev)?.to_dtype(DType::BF16)
}

#[cfg(all(
    feature = "cuda",
    has_cutlass_fp8_sm90_kernels,
    has_deepgemm_fp8_sm90_provider
))]
fn assert_deepgemm_fused_glu_close(label: &str, reference: &Tensor, output: &Tensor) -> Result<()> {
    let reference = reference
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let output = output
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let max_reference = reference.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let max_error = reference
        .iter()
        .zip(&output)
        .map(|(reference, output)| (reference - output).abs())
        .fold(0.0, f32::max);
    assert!(
        max_error <= 0.02 + 0.02 * max_reference,
        "{label}: max error {max_error}, max reference {max_reference}"
    );
    Ok(())
}

#[cfg(all(
    feature = "cuda",
    has_cutlass_fp8_sm90_kernels,
    has_deepgemm_fp8_sm90_provider
))]
fn copy_cuda_bf16(source: &Tensor, destination: &Tensor) -> Result<()> {
    use candle_core::{cuda::cudarc::driver::sys, Storage};
    use half::bf16;

    if source.dims() != destination.dims()
        || source.dtype() != DType::BF16
        || destination.dtype() != DType::BF16
        || !source.device().same_device(destination.device())
    {
        candle_core::bail!("CUDA BF16 test copy requires matching tensors")
    }
    let Device::Cuda(dev) = source.device() else {
        candle_core::bail!("CUDA BF16 test copy requires CUDA tensors")
    };
    let stream = dev.cuda_stream();
    let (source_storage, source_layout) = source.storage_and_layout();
    let Storage::Cuda(source_storage) = &*source_storage else {
        unreachable!()
    };
    let (destination_storage, destination_layout) = destination.storage_and_layout();
    let Storage::Cuda(destination_storage) = &*destination_storage else {
        unreachable!()
    };
    let (source_ptr, source_guard) = crate::utils::slice_ptr_on_stream(
        source_storage.as_cuda_slice::<bf16>()?,
        source_layout.start_offset(),
        &stream,
    );
    let (destination_ptr, destination_guard) = crate::utils::slice_ptr_on_stream(
        destination_storage.as_cuda_slice::<bf16>()?,
        destination_layout.start_offset(),
        &stream,
    );
    let status = unsafe {
        sys::cuMemcpyDtoDAsync_v2(
            destination_ptr,
            source_ptr,
            source.elem_count() * std::mem::size_of::<bf16>(),
            stream.cu_stream(),
        )
    };
    drop((source_guard, destination_guard));
    if status != sys::cudaError_enum::CUDA_SUCCESS {
        candle_core::bail!("CUDA BF16 test copy failed: {status:?}")
    }
    Ok(())
}

#[cfg(all(
    feature = "cuda",
    has_cutlass_fp8_sm90_kernels,
    has_deepgemm_fp8_sm90_provider
))]
#[test]
#[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
fn deepgemm_fused_glu_matches_unfused_sm90() -> Result<()> {
    const SPLIT_SIZE: usize = 256;
    const OUTPUT_FEATURES: usize = 256;
    const ROW_COUNTS: [usize; 8] = [1, 3, 4, 5, 16, 33, 129, 257];

    let dev = Device::new_cuda(0)?;
    let layer = deepgemm_fused_glu_test_layer(&dev, OUTPUT_FEATURES, SPLIT_SIZE)?;
    for rows in ROW_COUNTS {
        let input = deepgemm_fused_glu_test_input(rows, SPLIT_SIZE, rows, &dev)?;
        let intermediate =
            crate::utils::fused_split_glu(&input, SPLIT_SIZE, GluActivationType::Silu)?;
        let reference = layer.forward(&intermediate)?;
        let output = layer
            .forward_glu(
                crate::GluInput::Packed {
                    input: &input,
                    split_size: SPLIT_SIZE,
                },
                GluActivationType::Silu,
            )?
            .ok_or_else(|| candle_core::Error::msg("DeepGEMM fused GLU was not selected"))?;
        dev.synchronize()?;
        assert_eq!(output.dims(), [rows, OUTPUT_FEATURES]);
        assert_deepgemm_fused_glu_close(&format!("rows={rows}"), &reference, &output)?;
    }

    let input = deepgemm_fused_glu_test_input(6, SPLIT_SIZE, 91, &dev)?.reshape((2, 3, 512))?;
    let intermediate = crate::utils::fused_split_glu(&input, SPLIT_SIZE, GluActivationType::Silu)?;
    let reference = layer.forward(&intermediate)?;
    let output = layer
        .forward_glu(
            crate::GluInput::Packed {
                input: &input,
                split_size: SPLIT_SIZE,
            },
            GluActivationType::Silu,
        )?
        .ok_or_else(|| candle_core::Error::msg("DeepGEMM fused GLU was not selected"))?;
    dev.synchronize()?;
    assert_eq!(output.dims(), [2, 3, OUTPUT_FEATURES]);
    assert_deepgemm_fused_glu_close("rank=3", &reference, &output)
}

#[cfg(all(
    feature = "cuda",
    has_cutlass_fp8_sm90_kernels,
    has_deepgemm_fp8_sm90_provider
))]
#[test]
#[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
fn deepgemm_consumes_fused_rms_norm_group_major_activation() -> Result<()> {
    use float8::F8E4M3;

    const ROWS: usize = 5;
    const FEATURES: usize = 256;
    const OUTPUT_FEATURES: usize = 256;
    const EPSILON: f32 = 1.0e-6;

    let device = Device::new_cuda(0)?;
    let layer = deepgemm_fused_glu_test_layer(&device, OUTPUT_FEATURES, FEATURES)?;
    let input_values = (0..ROWS * FEATURES)
        .map(|index| ((index * 17 + index / FEATURES * 13) % 79) as f32 / 23.0 - 1.5)
        .collect::<Vec<_>>();
    let residual_values = (0..ROWS * FEATURES)
        .map(|index| ((index * 11 + index / FEATURES * 7) % 47) as f32 / 31.0 - 0.7)
        .collect::<Vec<_>>();
    let norm_weight_values = (0..FEATURES)
        .map(|column| 0.8 + (column % 29) as f32 / 64.0)
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(input_values, (ROWS, FEATURES), &device)?.to_dtype(DType::BF16)?;
    let residual =
        Tensor::from_vec(residual_values, (ROWS, FEATURES), &device)?.to_dtype(DType::BF16)?;
    let norm_weight =
        Tensor::from_vec(norm_weight_values, FEATURES, &device)?.to_dtype(DType::BF16)?;
    let scheme = layer
        .activation_quantization_scheme_for(&input)
        .ok_or_else(|| candle_core::Error::msg("DeepGEMM activation scheme is unavailable"))?;
    let scale_layout = layer
        .preferred_activation_scale_layout_for(&input)
        .ok_or_else(|| candle_core::Error::msg("DeepGEMM scale layout is unavailable"))?;
    assert_eq!(
        scale_layout,
        ActivationScaleLayout::GroupMajor {
            row_alignment: std::num::NonZeroUsize::new(4).unwrap(),
        }
    );
    let fused = fused_add_rms_norm_quantized(
        &input,
        &residual,
        &norm_weight,
        EPSILON,
        scheme,
        scale_layout,
    )?;
    assert_eq!(fused.activation().scales().dims(), &[2, 8]);
    let output = layer.forward_quantized(fused.activation())?;
    device.synchronize()?;

    let activation = fused
        .activation()
        .quantized()
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<F8E4M3>()?;
    let activation_scales = fused
        .activation()
        .scales()
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;
    let weight = ops::fp8_blockwise_dequantize(
        &layer.weight,
        &layer.weight_scale_inv,
        vec![128, 128],
        DType::F32,
    )?
    .to_device(&Device::Cpu)?
    .to_vec2::<f32>()?;
    let output = output
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;
    let mut max_reference = 0.0f32;
    let mut max_error = 0.0f32;
    for row in 0..ROWS {
        for output_column in 0..OUTPUT_FEATURES {
            let mut reference = 0.0f32;
            for column in 0..FEATURES {
                let activation_value = activation[row * FEATURES + column].to_f32()
                    * activation_scales[column / 128][row];
                reference += activation_value * weight[output_column][column];
            }
            max_reference = max_reference.max(reference.abs());
            max_error = max_error.max((output[row][output_column] - reference).abs());
        }
    }
    assert!(
        max_error <= 0.02 + 0.02 * max_reference,
        "DeepGEMM group-major prequantized output error {max_error}, max reference {max_reference}"
    );
    Ok(())
}

#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
fn cutile_test_layer(
    dev: &Device,
    output_features: usize,
    input_features: usize,
) -> Result<BlockwiseFP8Linear> {
    const BLOCK_SIZE: usize = 128;

    let weight_values = (0..output_features * input_features)
        .map(|index| {
            let row = index / input_features;
            let column = index % input_features;
            let block = row / BLOCK_SIZE * (input_features / BLOCK_SIZE) + column / BLOCK_SIZE;
            let amplitude = [0.04, 0.18, 0.75, 1.6][block % 4];
            let value = ((row * 17 + column * 29) % 31) as f32 - 15.0;
            value * amplitude
        })
        .collect::<Vec<_>>();
    let weight = Tensor::from_vec(weight_values, (output_features, input_features), dev)?
        .to_dtype(DType::BF16)?;
    let (weight, weight_scale_inv) =
        ops::fp8_blockwise_quantize(&weight, vec![BLOCK_SIZE, BLOCK_SIZE])?;
    let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
        weight,
        weight_scale_inv,
        bias: None,
        dequant_dtype: DType::BF16,
        weight_block_size: vec![BLOCK_SIZE, BLOCK_SIZE],
        activation_scheme: Some(Fp8ActivationScheme::Dynamic),
    })?;
    assert!(matches!(
        &layer.provider,
        BlockwiseFp8Provider::TensorCoreGemv
    ));
    Ok(layer)
}

#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
#[test]
fn cutile_w8a16_preserves_empty_leading_dimensions() -> Result<()> {
    const FEATURES: usize = 256;

    let device = Device::new_cuda(0)?;
    let weight = Tensor::zeros((FEATURES, FEATURES), DType::BF16, &device)?;
    let (weight, weight_scale_inv) = ops::fp8_blockwise_quantize(&weight, vec![128, 128])?;
    let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
        weight,
        weight_scale_inv,
        bias: None,
        dequant_dtype: DType::BF16,
        weight_block_size: vec![128, 128],
        activation_scheme: None,
    })?;
    assert!(matches!(layer.provider, BlockwiseFp8Provider::CutileW8A16));
    let output = layer.forward(&Tensor::zeros((2, 0, FEATURES), DType::BF16, &device)?)?;
    assert_eq!(output.dims(), [2, 0, FEATURES]);
    Ok(())
}

#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
fn assert_close(label: &str, reference: &Tensor, output: &Tensor) -> Result<()> {
    let reference = reference.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let output = output.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let max_error = output
        .sub(&reference)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let max_reference = reference.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        max_error <= 0.02 + 0.02 * max_reference,
        "{label}: error {max_error}, max reference {max_reference}"
    );
    Ok(())
}

#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
#[test]
fn cutile_fused_glu_matches_unfused() -> Result<()> {
    const OUTPUT_FEATURES: usize = 256;
    const SPLIT_SIZE: usize = 512;

    let dev = Device::new_cuda(0)?;
    let layer = cutile_test_layer(&dev, OUTPUT_FEATURES, SPLIT_SIZE)?;
    let input_of = |shape: &[usize], phase: usize| -> Result<Tensor> {
        let count = shape.iter().product::<usize>();
        let values = (0..count)
            .map(|index| ((index * 13 + phase * 7) % 89) as f32 / 29.0 - 1.4)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &dev)?.to_dtype(DType::BF16)
    };
    for shape in [
        &[64usize, 2 * SPLIT_SIZE][..],
        &[300, 2 * SPLIT_SIZE],
        &[2, 40, 2 * SPLIT_SIZE],
    ] {
        let input = input_of(shape, shape.len())?;
        let reference = layer.forward(&crate::utils::fused_split_glu(
            &input,
            SPLIT_SIZE,
            GluActivationType::Silu,
        )?)?;
        let output = layer
            .forward_glu(
                crate::GluInput::Packed {
                    input: &input,
                    split_size: SPLIT_SIZE,
                },
                GluActivationType::Silu,
            )?
            .ok_or_else(|| candle_core::Error::msg("cuTile fused GLU was not selected"))?;
        assert_eq!(output.dims(), reference.dims());
        assert_close(&format!("shape {shape:?}"), &reference, &output)?;
    }
    let small = input_of(&[8, 2 * SPLIT_SIZE], 5)?;
    assert!(layer
        .forward_glu(
            crate::GluInput::Packed {
                input: &small,
                split_size: SPLIT_SIZE
            },
            GluActivationType::Silu
        )?
        .is_none());
    Ok(())
}

#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
#[test]
fn cutile_consumes_fused_rms_norm_group_major_activation() -> Result<()> {
    const ROWS: usize = 300;
    const FEATURES: usize = 256;
    const OUTPUT_FEATURES: usize = 256;
    const EPSILON: f32 = 1.0e-6;

    let device = Device::new_cuda(0)?;
    let layer = cutile_test_layer(&device, OUTPUT_FEATURES, FEATURES)?;
    let input_values = (0..ROWS * FEATURES)
        .map(|index| ((index * 17 + index / FEATURES * 13) % 79) as f32 / 23.0 - 1.5)
        .collect::<Vec<_>>();
    let residual_values = (0..ROWS * FEATURES)
        .map(|index| ((index * 11 + index / FEATURES * 7) % 47) as f32 / 31.0 - 0.7)
        .collect::<Vec<_>>();
    let norm_weight_values = (0..FEATURES)
        .map(|column| 0.8 + (column % 29) as f32 / 64.0)
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(input_values, (ROWS, FEATURES), &device)?.to_dtype(DType::BF16)?;
    let residual =
        Tensor::from_vec(residual_values, (ROWS, FEATURES), &device)?.to_dtype(DType::BF16)?;
    let norm_weight =
        Tensor::from_vec(norm_weight_values, FEATURES, &device)?.to_dtype(DType::BF16)?;
    assert_eq!(
        layer.preferred_activation_scale_layout_for(&input.narrow(0, 0, 8)?),
        Some(ActivationScaleLayout::RowMajor)
    );
    let scheme = layer
        .activation_quantization_scheme_for(&input)
        .ok_or_else(|| candle_core::Error::msg("cuTile activation scheme is unavailable"))?;
    let scale_layout = layer
        .preferred_activation_scale_layout_for(&input)
        .ok_or_else(|| candle_core::Error::msg("cuTile scale layout is unavailable"))?;
    assert_eq!(
        scale_layout,
        ActivationScaleLayout::GroupMajor {
            row_alignment: std::num::NonZeroUsize::new(crate::cutile::FP8_GEMM_BLOCK_ROWS).unwrap(),
        }
    );
    let fused = fused_add_rms_norm_quantized(
        &input,
        &residual,
        &norm_weight,
        EPSILON,
        scheme,
        scale_layout,
    )?;
    assert_eq!(fused.activation().scales().dims(), &[2, 384]);
    let output = layer.forward_quantized(fused.activation())?;
    let residual_ref =
        (input.to_dtype(DType::F32)? + residual.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)?;
    assert_close("residual", &residual_ref, fused.residual())?;
    // the GEMM must read the producer's padded storage right: compare against the dequantized
    // activation it was handed rather than re-quantizing a bf16 normalization
    let cpu = Device::Cpu;
    let values = fused
        .activation()
        .quantized()
        .to_device(&cpu)?
        .to_dtype(DType::F32)?;
    let row_scales = fused
        .activation()
        .scales()
        .to_device(&cpu)?
        .t()?
        .narrow(0, 0, ROWS)?
        .contiguous()?;
    let dequantized = values
        .reshape((ROWS, FEATURES / 128, 128))?
        .broadcast_mul(&row_scales.reshape((ROWS, FEATURES / 128, 1))?)?
        .reshape((ROWS, FEATURES))?;
    let weight = ops::fp8_blockwise_dequantize(
        &layer.weight,
        &layer.weight_scale_inv,
        vec![128, 128],
        DType::F32,
    )?
    .to_device(&cpu)?;
    assert_close("output", &dequantized.matmul(&weight.t()?)?, &output)?;
    Ok(())
}

#[cfg(all(
    feature = "cuda",
    has_cutlass_fp8_sm90_kernels,
    has_deepgemm_fp8_sm90_provider
))]
#[test]
#[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
fn deepgemm_fused_rms_norm_cuda_graph_replays_sm90() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;
    use half::bf16;

    const ROWS: usize = 5;
    const FEATURES: usize = 256;
    const OUTPUT_FEATURES: usize = 256;
    const EPSILON: f32 = 1.0e-6;

    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda_device) = &device else {
        unreachable!()
    };
    let stream = cuda_device.cuda_stream();
    let layer = deepgemm_fused_glu_test_layer(&device, OUTPUT_FEATURES, FEATURES)?;
    let norm_weight_values = (0..FEATURES)
        .map(|column| 0.8 + (column % 29) as f32 / 64.0)
        .collect::<Vec<_>>();
    let norm_weight =
        Tensor::from_vec(norm_weight_values, FEATURES, &device)?.to_dtype(DType::BF16)?;
    let make_inputs = |phase: usize| -> Result<(Tensor, Tensor)> {
        let input_values = (0..ROWS * FEATURES)
            .map(|index| {
                ((index * 17 + index / FEATURES * 13 + phase * 19) % 79) as f32 / 23.0 - 1.5
            })
            .collect::<Vec<_>>();
        let residual_values = (0..ROWS * FEATURES)
            .map(|index| {
                ((index * 11 + index / FEATURES * 7 + phase * 23) % 47) as f32 / 31.0 - 0.7
            })
            .collect::<Vec<_>>();
        Ok((
            Tensor::from_vec(input_values, (ROWS, FEATURES), &device)?.to_dtype(DType::BF16)?,
            Tensor::from_vec(residual_values, (ROWS, FEATURES), &device)?.to_dtype(DType::BF16)?,
        ))
    };
    let (input_a, residual_a) = make_inputs(7)?;
    let (input_b, residual_b) = make_inputs(53)?;
    let make_reference = |input: &Tensor, residual: &Tensor| -> Result<(Tensor, Tensor)> {
        let residual_output = (input + residual)?;
        let normalized =
            candle_nn::ops::rms_norm(&residual_output.contiguous()?, &norm_weight, EPSILON)?;
        let output = layer.forward(&normalized)?;
        Ok((residual_output, output))
    };
    let (reference_residual_a, reference_output_a) = make_reference(&input_a, &residual_a)?;
    let (reference_residual_b, reference_output_b) = make_reference(&input_b, &residual_b)?;

    let graph_input = Tensor::zeros((ROWS, FEATURES), DType::BF16, &device)?;
    let graph_residual = Tensor::zeros((ROWS, FEATURES), DType::BF16, &device)?;
    let graph_residual_output = Tensor::zeros((ROWS, FEATURES), DType::BF16, &device)?;
    let graph_output = Tensor::zeros((ROWS, OUTPUT_FEATURES), DType::BF16, &device)?;
    copy_cuda_bf16(&input_a, &graph_input)?;
    copy_cuda_bf16(&residual_a, &graph_residual)?;
    let scheme = layer
        .activation_quantization_scheme_for(&graph_input)
        .ok_or_else(|| candle_core::Error::msg("DeepGEMM activation scheme is unavailable"))?;
    let scale_layout = layer
        .preferred_activation_scale_layout_for(&graph_input)
        .ok_or_else(|| candle_core::Error::msg("DeepGEMM scale layout is unavailable"))?;
    assert_eq!(
        scale_layout,
        ActivationScaleLayout::GroupMajor {
            row_alignment: std::num::NonZeroUsize::new(4).unwrap(),
        }
    );
    let warmup = fused_add_rms_norm_quantized(
        &graph_input,
        &graph_residual,
        &norm_weight,
        EPSILON,
        scheme,
        scale_layout,
    )?;
    assert_eq!(warmup.activation().scales().dims(), &[2, 8]);
    let warmup_output = layer.forward_quantized(warmup.activation())?;
    drop((warmup_output, warmup));
    device.synchronize()?;

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
    let captured = (|| -> Result<()> {
        let fused = fused_add_rms_norm_quantized(
            &graph_input,
            &graph_residual,
            &norm_weight,
            EPSILON,
            scheme,
            scale_layout,
        )?;
        let output = layer.forward_quantized(fused.activation())?;
        copy_cuda_bf16(fused.residual(), &graph_residual_output)?;
        copy_cuda_bf16(&output, &graph_output)?;
        drop((output, fused));
        Ok(())
    })();
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

    let expected_residual_a = reference_residual_a
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<bf16>()?;
    let expected_residual_b = reference_residual_b
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<bf16>()?;
    let replays = [
        (
            &input_a,
            &residual_a,
            &reference_output_a,
            &expected_residual_a,
        ),
        (
            &input_a,
            &residual_a,
            &reference_output_a,
            &expected_residual_a,
        ),
        (
            &input_b,
            &residual_b,
            &reference_output_b,
            &expected_residual_b,
        ),
        (
            &input_a,
            &residual_a,
            &reference_output_a,
            &expected_residual_a,
        ),
    ];
    let mut first_output = None;
    for (replay, (input, residual, reference_output, expected_residual)) in
        replays.into_iter().enumerate()
    {
        copy_cuda_bf16(input, &graph_input)?;
        copy_cuda_bf16(residual, &graph_residual)?;
        graph
            .launch()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        stream
            .synchronize()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;

        let actual_residual = graph_residual_output
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<bf16>()?;
        assert_eq!(actual_residual.as_slice(), expected_residual.as_slice());
        assert_deepgemm_fused_glu_close(
            &format!("fused RMSNorm graph replay {replay}"),
            reference_output,
            &graph_output,
        )?;
        let actual_output = graph_output
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<bf16>()?;
        match replay {
            0 => first_output = Some(actual_output),
            1 | 3 => assert_eq!(
                actual_output.as_slice(),
                first_output.as_ref().unwrap().as_slice()
            ),
            2 => assert_ne!(
                actual_output.as_slice(),
                first_output.as_ref().unwrap().as_slice()
            ),
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "cuda",
    has_cutlass_fp8_sm90_kernels,
    has_deepgemm_fp8_sm90_provider
))]
#[test]
#[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
fn deepgemm_fused_glu_cuda_graph_replays_sm90() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    const ROWS: usize = 16;
    const SPLIT_SIZE: usize = 256;
    const OUTPUT_FEATURES: usize = 256;
    const REPLAY_COUNT: usize = 3;

    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cuda_dev) = &dev else {
        unreachable!()
    };
    let stream = cuda_dev.cuda_stream();
    let layer = deepgemm_fused_glu_test_layer(&dev, OUTPUT_FEATURES, SPLIT_SIZE)?;
    let input_a = deepgemm_fused_glu_test_input(ROWS, SPLIT_SIZE, 7, &dev)?;
    let input_b = deepgemm_fused_glu_test_input(ROWS, SPLIT_SIZE, 53, &dev)?;
    let graph_input = input_a.copy()?;
    let graph_output = Tensor::zeros((ROWS, OUTPUT_FEATURES), DType::BF16, &dev)?;
    let reference_a = layer.forward(&crate::utils::fused_split_glu(
        &input_a,
        SPLIT_SIZE,
        GluActivationType::Silu,
    )?)?;
    let reference_b = layer.forward(&crate::utils::fused_split_glu(
        &input_b,
        SPLIT_SIZE,
        GluActivationType::Silu,
    )?)?;
    let warmup = layer
        .forward_glu(
            crate::GluInput::Packed {
                input: &graph_input,
                split_size: SPLIT_SIZE,
            },
            GluActivationType::Silu,
        )?
        .ok_or_else(|| candle_core::Error::msg("DeepGEMM fused GLU was not selected"))?;
    drop(warmup);
    dev.synchronize()?;

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
    let captured = layer
        .forward_glu(
            crate::GluInput::Packed {
                input: &graph_input,
                split_size: SPLIT_SIZE,
            },
            GluActivationType::Silu,
        )
        .and_then(|output| {
            let output = output
                .ok_or_else(|| candle_core::Error::msg("DeepGEMM fused GLU was not selected"))?;
            copy_cuda_bf16(&output, &graph_output)?;
            drop(output);
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

    for replay in 0..REPLAY_COUNT {
        let (source, reference) = if replay == 1 {
            (&input_b, &reference_b)
        } else {
            (&input_a, &reference_a)
        };
        let churn = (0..32)
            .map(|_| Tensor::zeros((ROWS, SPLIT_SIZE), DType::BF16, &dev))
            .collect::<Result<Vec<_>>>()?;
        drop(churn);
        copy_cuda_bf16(source, &graph_input)?;
        graph
            .launch()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        stream
            .synchronize()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        assert_deepgemm_fused_glu_close(
            &format!("graph replay {replay}"),
            reference,
            &graph_output,
        )?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
#[test]
#[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
fn deepgemm_production_shapes_use_linear_dispatch_sm90() -> Result<()> {
    const BLOCK_SIZE: usize = 128;
    const M: usize = 512;
    const OUTPUT_TOLERANCE: f32 = 0.02;
    const SHAPES: [(usize, usize); 5] = [
        (16_384, 5_120),
        (14_336, 5_120),
        (5_120, 6_144),
        (34_816, 5_120),
        (5_120, 17_408),
    ];

    let dev = Device::new_cuda(0)?;
    for (n, k) in SHAPES {
        let weight = Tensor::ones((n, k), DType::F8E4M3, &Device::Cpu)?.to_device(&dev)?;
        let weight_scales = Tensor::ones((n / BLOCK_SIZE, k / BLOCK_SIZE), DType::F32, &dev)?
            .affine(1.0 / k as f64, 0.0)?;
        let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
            weight,
            weight_scale_inv: weight_scales,
            bias: None,
            dequant_dtype: DType::BF16,
            weight_block_size: vec![BLOCK_SIZE, BLOCK_SIZE],
            activation_scheme: Some(Fp8ActivationScheme::Dynamic),
        })?;
        assert!(matches!(
            &layer.provider,
            BlockwiseFp8Provider::DeepGemmSm90(_)
        ));

        let input = Tensor::ones((M, k), DType::BF16, &dev)?;
        let output = layer.forward_raw(&input)?;
        dev.synchronize()?;
        let max_error = output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|value| (value - 1.0).abs())
            .fold(0.0, f32::max);
        assert!(
            max_error <= OUTPUT_TOLERANCE,
            "M={M}, N={n}, K={k}: output error {max_error}"
        );
    }
    Ok(())
}

#[test]
fn scale_shard_follows_simple_weight_partition() -> Result<()> {
    let shard = scale_shard_from_weight_shard(
        [12_288, 5_120],
        [128, 128],
        Shard::Simple {
            dim: 0,
            rank: 3,
            world_size: 8,
        },
    )?;
    assert_eq!(
        shard,
        Shard::Offset {
            dim: 0,
            offset: 36,
            len: 12,
        }
    );

    let shard = scale_shard_from_weight_shard(
        [5_120, 6_144],
        [128, 128],
        Shard::Simple {
            dim: 1,
            rank: 7,
            world_size: 8,
        },
    )?;
    assert_eq!(
        shard,
        Shard::Offset {
            dim: 1,
            offset: 42,
            len: 6,
        }
    );
    Ok(())
}

#[test]
fn scale_shard_maps_replicated_kv_offset_to_scale_rows() -> Result<()> {
    let shard = scale_shard_from_weight_shard(
        [1_024, 5_120],
        [128, 128],
        Shard::Offset {
            dim: 0,
            offset: 512,
            len: 256,
        },
    )?;
    assert_eq!(
        shard,
        Shard::Offset {
            dim: 0,
            offset: 4,
            len: 2,
        }
    );
    Ok(())
}

#[test]
fn derived_scale_offset_selects_matching_scale_blocks() -> Result<()> {
    let scales = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6., 7., 8.], (4, 2), &Device::Cpu)?;
    let scale_shard = scale_shard_from_weight_shard(
        [8, 4],
        [2, 2],
        Shard::Offset {
            dim: 0,
            offset: 4,
            len: 2,
        },
    )?;
    assert_eq!(
        scale_shard.apply_to(&scales)?.to_vec2::<f32>()?,
        [vec![5., 6.]]
    );
    Ok(())
}

#[test]
fn scale_shard_rejects_unaligned_weight_partition() {
    let err = scale_shard_from_weight_shard(
        [1_024, 5_120],
        [128, 128],
        Shard::Offset {
            dim: 0,
            offset: 64,
            len: 256,
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("not aligned"));
}

#[test]
fn excluded_fp8_linear_preserves_weight_shard() -> Result<()> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "foo.weight".to_string(),
        Tensor::zeros((8, 4), DType::F32, &Device::Cpu)?,
    );
    let vb = ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu).pp("foo");
    let layer = crate::fp8_config::checkpoint_linear_b(
        4,
        8,
        &fp8_config(&["foo"]),
        false,
        Shard::Simple {
            dim: 0,
            rank: 1,
            world_size: 2,
        },
        vb,
    )?;
    assert_eq!(layer.dequantize_w()?.dims(), &[4, 4]);
    Ok(())
}

#[test]
fn official_language_model_alias_is_excluded() -> Result<()> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "model.language_model.embed_tokens.weight".to_string(),
        Tensor::zeros((8, 4), DType::F32, &Device::Cpu)?,
    );
    let vb = ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu)
        .pp("model")
        .pp("language_model")
        .pp("embed_tokens");
    let layer = crate::fp8_config::checkpoint_linear_b(
        4,
        8,
        &fp8_config(&["model.embed_tokens"]),
        false,
        Shard::default(),
        vb,
    )?;
    assert_eq!(layer.dequantize_w()?.dims(), &[8, 4]);
    Ok(())
}

#[test]
fn declared_exclusions_make_unlisted_missing_scale_an_error() -> Result<()> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "foo.weight".to_string(),
        Tensor::zeros((8, 4), DType::F32, &Device::Cpu)?,
    );
    let vb = ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu).pp("foo");
    let err = crate::fp8_config::checkpoint_linear_b(
        4,
        8,
        &fp8_config(&["bar"]),
        false,
        Shard::default(),
        vb,
    )
    .unwrap_err();
    assert!(err.to_string().contains("missing FP8 weight scale"));
    Ok(())
}
