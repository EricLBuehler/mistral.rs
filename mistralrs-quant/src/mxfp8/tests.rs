use super::*;

#[test]
fn e8m0_preserves_exponents_and_nan() {
    assert_eq!(e8m0(0).to_bits(), 1 << 22);
    assert_eq!(e8m0(126), 0.5);
    assert_eq!(e8m0(127), 1.0);
    assert_eq!(e8m0(128), 2.0);
    assert_eq!(e8m0(254), 2f32.powi(127));
    assert!(e8m0(255).is_nan());
}

#[test]
fn activation_uses_power_of_two_scales_per_group() -> Result<()> {
    let mut input = vec![0f32; MXFP8_BLOCK_SIZE * 3];
    input[0] = 1.07;
    input[1] = 32.0;
    input[MXFP8_BLOCK_SIZE] = -1.07;
    input[MXFP8_BLOCK_SIZE + 1] = 224.0;
    let output = Tensor::from_vec(input, (1, MXFP8_BLOCK_SIZE * 3), &Device::Cpu)?
        .apply_op1_no_bwd(&Quantize)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(output[0], 1.125);
    assert_eq!(output[1], 32.0);
    assert_eq!(output[MXFP8_BLOCK_SIZE], -1.125);
    assert_eq!(output[MXFP8_BLOCK_SIZE + 1], 224.0);
    assert!(output[MXFP8_BLOCK_SIZE * 2..]
        .iter()
        .all(|value| *value == 0.0));
    Ok(())
}

#[test]
fn exact_scale_boundaries_preserve_small_representable_values() -> Result<()> {
    fn check_device(device: Device) -> Result<()> {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let mut values = vec![0f32; MXFP8_BLOCK_SIZE];
            values[0] = 3.5;
            values[1] = 2f32.powi(-16);
            let input = Tensor::from_vec(values.clone(), (1, MXFP8_BLOCK_SIZE), &device)?
                .to_dtype(dtype)?;
            let output = input.apply_op1_no_bwd(&Quantize)?.to_dtype(DType::F32)?;
            assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, values);
        }
        Ok(())
    }
    check_device(Device::Cpu)?;
    #[cfg(feature = "cuda")]
    check_device(Device::new_cuda_with_stream(0)?)?;
    Ok(())
}

#[test]
fn compact_layer_scales_each_group_and_rounds_activation() -> Result<()> {
    let device = Device::Cpu;
    let mut input = vec![0f32; MXFP8_BLOCK_SIZE * 2];
    input[0] = 1.07;
    input[1] = 32.0;
    input[MXFP8_BLOCK_SIZE] = -1.07;
    input[MXFP8_BLOCK_SIZE + 1] = 224.0;
    let layer = Mxfp8Linear::from_parts(
        Tensor::from_vec(
            vec![F8E4M3::from_f32(1.0); MXFP8_BLOCK_SIZE * 2],
            (1, MXFP8_BLOCK_SIZE * 2),
            &device,
        )?,
        Tensor::from_vec(vec![126u8, 128], (1, 2), &device)?,
        Some(Tensor::new(&[3f32], &device)?),
        DType::F32,
    )?;
    let output = layer.forward(&Tensor::from_vec(
        input,
        (1, 1, MXFP8_BLOCK_SIZE * 2),
        &device,
    )?)?;
    assert_eq!(output.dims(), [1, 1, 1]);
    assert_eq!(
        output.flatten_all()?.to_vec1::<f32>()?,
        vec![(1.125 + 32.0) * 0.5 + (-1.125 + 224.0) * 2.0 + 3.0]
    );
    assert_eq!(layer.weight.dtype(), DType::F8E4M3);
    assert_eq!(layer.scales.dtype(), DType::U8);
    let dequant = layer.dequantize_w()?.to_vec2::<f32>()?;
    assert!(dequant[0][..MXFP8_BLOCK_SIZE]
        .iter()
        .all(|value| *value == 0.5));
    assert!(dequant[0][MXFP8_BLOCK_SIZE..]
        .iter()
        .all(|value| *value == 2.0));
    Ok(())
}

#[test]
fn merged_projections_preserve_constituents_and_activation_rounding() -> Result<()> {
    let device = Device::Cpu;
    let k = MXFP8_BLOCK_SIZE * 2;
    let layers = [3usize, 5]
        .into_iter()
        .map(|n| {
            Mxfp8Linear::from_parts(
                Tensor::from_vec(
                    (0..n * k)
                        .map(|i| F8E4M3::from_f32((i % 13) as f32 * 0.25 - 1.0))
                        .collect(),
                    (n, k),
                    &device,
                )?,
                Tensor::from_vec(
                    (0..n * 2).map(|i| 125 + (i % 5) as u8).collect(),
                    (n, 2),
                    &device,
                )?,
                None,
                DType::F32,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let input = Tensor::from_vec(
        (0..2 * k).map(|i| (i % 19) as f32 * 0.13 - 0.71).collect(),
        (2, k),
        &device,
    )?;
    let expected = Tensor::cat(
        &layers
            .iter()
            .map(|layer| layer.forward(&input))
            .collect::<Result<Vec<_>>>()?,
        1,
    )?;
    let merged = Mxfp8Linear::merge(layers.clone())?.unwrap();
    assert_eq!(merged.rows_per_rank, [3, 5]);
    assert_eq!(merged.packed.name(), "MXFP8");
    assert_eq!(
        merged.packed.forward(&input)?.to_vec2::<f32>()?,
        expected.to_vec2::<f32>()?
    );
    for (original, constituent) in layers.iter().zip(merged.constituents) {
        assert_eq!(constituent.name(), "MXFP8");
        assert_eq!(
            constituent.dequantize_w()?.to_vec2::<f32>()?,
            original.dequantize_w()?.to_vec2::<f32>()?
        );
    }
    let mut biased = layers.clone();
    biased[0].bias = Some(Tensor::zeros(3, DType::F32, &device)?);
    assert!(Mxfp8Linear::merge(biased)?.is_none());
    Ok(())
}

#[cfg(feature = "cuda")]
fn fixture(
    rows: usize,
    n: usize,
    k: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Mxfp8Linear, Tensor)> {
    let weight = (0..n * k)
        .map(|i| F8E4M3::from_f32(((i * 13 % 47) as f32 - 23.0) * 0.125))
        .collect::<Vec<_>>();
    let scales = (0..n * k / MXFP8_BLOCK_SIZE)
        .map(|i| 123 + (i % 5) as u8)
        .collect::<Vec<_>>();
    let layer = Mxfp8Linear::from_parts(
        Tensor::from_vec(weight, (n, k), device)?,
        Tensor::from_vec(scales, (n, k / MXFP8_BLOCK_SIZE), device)?,
        Some(
            Tensor::from_vec(
                (0..n).map(|i| (i % 7) as f32 * 0.125).collect::<Vec<_>>(),
                n,
                device,
            )?
            .to_dtype(dtype)?,
        ),
        dtype,
    )?;
    let input = Tensor::from_vec(
        (0..(rows + 1) * k)
            .map(|i| ((i * 19 % 97) as f32 - 48.0) * 0.03125)
            .collect::<Vec<_>>(),
        (rows + 1, k),
        device,
    )?
    .to_dtype(dtype)?
    .narrow(0, 1, rows)?;
    Ok((layer, input))
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mxfp8_matches_cpu_prefill_decode_and_views() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for (rows, n, k) in [(1, 13, 64), (2, 65, 160), (17, 64, 64)] {
            let (cpu, x) = fixture(rows, n, k, dtype, &Device::Cpu)?;
            let (cuda, cuda_x) = fixture(rows, n, k, dtype, &device)?;
            let reference = cpu
                .forward(&x)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let output = cuda
                .forward(&cuda_x)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let tolerance = match dtype {
                DType::F32 => 1e-4,
                DType::F16 => 0.002,
                _ => 0.015,
            };
            for (i, (actual, expected)) in output.iter().zip(&reference).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                    "{dtype:?} {rows}x{n}x{k} element {i}: {actual} vs {expected}"
                );
            }
            assert_eq!(
                cuda.dequantize_w()?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                cpu.dequantize_w()?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            );
        }
    }
    Ok(())
}

fn routed(device: &Device) -> Result<Vec<f32>> {
    let k = MXFP8_BLOCK_SIZE;
    let layers = (0..3)
        .map(|expert| {
            Mxfp8Linear::from_parts(
                Tensor::from_vec(vec![F8E4M3::from_f32(1.0); 2 * k], (2, k), device)?,
                Tensor::from_vec(vec![127 + expert as u8; 2], (2, 1), device)?,
                None,
                DType::F32,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let layer = Mxfp8Linear::stack(layers)?;
    let indices = Tensor::new(&[[2u32, 0], [1, 2]], device)?;
    let input = Tensor::ones((2, 1, k), DType::F32, device)?;
    let output = layer.gather_forward(&input, &indices)?;
    assert_eq!(output.dims(), [2, 2, 2]);
    output.flatten_all()?.to_vec1::<f32>()
}

#[test]
fn routed_mxfp8_preserves_expert_selection() -> Result<()> {
    assert_eq!(
        routed(&Device::Cpu)?,
        vec![128.0, 128.0, 32.0, 32.0, 64.0, 64.0, 128.0, 128.0]
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_routed_mxfp8_matches_cpu() -> Result<()> {
    assert_eq!(routed(&Device::new_cuda(0)?)?, routed(&Device::Cpu)?);
    Ok(())
}

#[test]
fn packed_activation_matches_reference_bytes_and_scale_boundaries() -> Result<()> {
    fn check(device: &Device) -> Result<()> {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let k = MXFP8_BLOCK_SIZE * 4;
            let mut values = vec![0f32; k * 3];
            for row in 1..3 {
                values[row * k] = 3.5;
                values[row * k + 1] = 2f32.powi(-16);
                values[row * k + MXFP8_BLOCK_SIZE] = 448.0;
                values[row * k + MXFP8_BLOCK_SIZE + 1] = -0.001953125;
                values[row * k + MXFP8_BLOCK_SIZE * 2] = 449.0;
            }
            let input = Tensor::from_vec(values, (3, k), device)?
                .to_dtype(dtype)?
                .narrow(0, 1, 2)?;
            let packed = quantize_packed(&input)?;
            let reference = quantize_packed(&input.to_device(&Device::Cpu)?)?;
            assert_eq!(packed.scheme(), MXFP8_ACTIVATION_SCHEME);
            assert_eq!(packed.source_dtype(), dtype);
            assert_eq!(packed.source_shape(), [2, k]);
            let q = packed.quantized().flatten_all()?.to_vec1::<F8E4M3>()?;
            let reference_q = reference.quantized().flatten_all()?.to_vec1::<F8E4M3>()?;
            assert_eq!(
                q.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                reference_q.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            let scales = packed.scales().flatten_all()?.to_vec1::<u8>()?;
            assert_eq!(scales, reference.scales().flatten_all()?.to_vec1::<u8>()?);
            assert_eq!(scales[0], 120);
            assert_eq!(scales[1], 127);
            assert_eq!(scales[2], if dtype == DType::BF16 { 127 } else { 128 });
            assert_eq!(scales[3], 0);
            assert_eq!(q[0].to_f32(), 448.0);
            assert_eq!(q[1].to_f32(), 0.001953125);
            assert!(q[MXFP8_BLOCK_SIZE * 3..k].iter().all(|x| x.to_f32() == 0.0));
        }
        Ok(())
    }
    check(&Device::Cpu)?;
    #[cfg(feature = "cuda")]
    check(&Device::new_cuda_with_stream(0)?)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_c1_keeps_unrounded_activation_for_vector_gemv() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let (layer, input) = fixture(1, 17, 640, dtype, &device)?;
        let reference = input
            .apply_op3_no_bwd(&layer.weight, &layer.scales, &Matmul { indices: None })?
            .broadcast_add(layer.bias.as_ref().unwrap())?;
        assert!(layer.activation_quantization_scheme_for(&input).is_none());
        let actual = layer.forward(&input)?;
        assert_eq!(
            actual.to_dtype(DType::F32)?.to_vec2::<f32>()?,
            reference.to_dtype(DType::F32)?.to_vec2::<f32>()?
        );
    }
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cutile"))]
#[test]
fn native_prefill_rejects_unregistered_activation_dtypes() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::block_scaled_mma_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let (mut layer, input) = fixture(2, 13, 160, DType::BF16, &device)?;
    layer.bias = None;
    assert_eq!(
        layer.activation_quantization_scheme_for(&input),
        Some(MXFP8_ACTIVATION_SCHEME)
    );
    for dtype in [DType::F32, DType::F16] {
        let input = input.to_dtype(dtype)?;
        assert!(layer.activation_quantization_scheme_for(&input).is_none());
        assert!(layer.quantize_activation(&input).is_err());
        let activation = quantize_packed(&input)?;
        assert!(layer.forward_quantized(&activation).is_err());
    }
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cutile"))]
#[test]
fn native_prefill_preserves_shared_activation_and_packed_views() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !crate::cutile::block_scaled_mma_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for rows in [17, 65] {
            let (mut first, input) = fixture(rows, 65, 160, dtype, &device)?;
            let (mut second, _) = fixture(rows, 13, 160, dtype, &device)?;
            first.bias = None;
            second.bias = None;
            let packed = Mxfp8Linear::merge(vec![first.clone(), second.clone()])?.unwrap();
            let quantized = first.quantize_activation(&input)?;
            assert_eq!(
                first.activation_quantization_scheme_for(&input),
                Some(MXFP8_ACTIVATION_SCHEME)
            );
            let combined = packed.packed.forward_quantized(&quantized)?;
            let shared = if dtype == DType::F32 {
                packed
                    .constituents
                    .iter()
                    .map(|method| method.forward_quantized(&quantized))
                    .collect::<Result<Vec<_>>>()?
            } else {
                let methods = packed
                    .constituents
                    .iter()
                    .map(AsRef::as_ref)
                    .collect::<Vec<_>>();
                crate::forward_shared_activation(&input, &methods)?.unwrap()
            };
            let separate = Tensor::cat(&shared, 1)?;
            assert_eq!(
                combined.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                separate.to_dtype(DType::F32)?.to_vec2::<f32>()?
            );
            let x = input.to_device(&Device::Cpu)?.apply_op1_no_bwd(&Quantize)?;
            let weight = Tensor::cat(
                &[
                    first.weight.to_device(&Device::Cpu)?,
                    second.weight.to_device(&Device::Cpu)?,
                ],
                0,
            )?;
            let scales = Tensor::cat(
                &[
                    first.scales.to_device(&Device::Cpu)?,
                    second.scales.to_device(&Device::Cpu)?,
                ],
                0,
            )?;
            let reference = x
                .apply_op3_no_bwd(&weight, &scales, &Matmul { indices: None })?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = combined
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (actual, expected) in actual.into_iter().zip(reference) {
                let tolerance = if dtype == DType::F32 {
                    0.0001
                } else if dtype == DType::F16 {
                    0.002
                } else {
                    0.02
                };
                assert!(
                    (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                    "{dtype:?}: {actual} != {expected}"
                );
            }
        }
    }
    Ok(())
}
