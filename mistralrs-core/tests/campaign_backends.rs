use candle_core::{Device, Result, Tensor};

fn check_backend(device: &Device) -> Result<()> {
    let input = Tensor::new(&[[1f32, 2.], [3., 4.]], device)?;
    let result = input.matmul(&input)?;
    device.synchronize()?;
    assert_eq!(
        result.to_device(&Device::Cpu)?.to_vec2::<f32>()?,
        vec![vec![7., 10.], vec![15., 22.]]
    );
    Ok(())
}

#[test]
fn campaign_cpu_backend() -> Result<()> {
    check_backend(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn campaign_cuda_backend() -> Result<()> {
    check_backend(&Device::new_cuda(0)?)
}

const NVFP4_K: usize = 32;

fn nvfp4_fixture_parts(
    device: &Device,
    dtype: candle_core::DType,
) -> Result<mistralrs_quant::nvfp4::Nvfp4LayerParts> {
    use candle_core::DType;
    use mistralrs_quant::nvfp4::Nvfp4LayerParts;
    use mistralrs_quant::Nvfp4ActivationMode;
    Ok(Nvfp4LayerParts {
        weight: Tensor::from_vec(vec![0x22u8; NVFP4_K], (2, NVFP4_K / 2), device)?,
        scales: Tensor::ones((2, NVFP4_K / 16), DType::F32, &Device::Cpu)?
            .to_dtype(DType::F8E4M3)?
            .to_device(device)?,
        global_scales: Tensor::new(&[0.5f32, 2.0], device)?,
        input_scale: None,
        activation: Nvfp4ActivationMode::None,
        bias: None,
        dtype,
    })
}

#[test]
fn campaign_cpu_nvfp4_w4a16() -> Result<()> {
    use candle_core::quantized::nvfp4_e7c_safetensors::ModelOptNvfp4Shards;
    use candle_core::DType;
    use mistralrs_quant::nvfp4::Nvfp4Layer;
    use mistralrs_quant::QuantMethod;
    let parts = nvfp4_fixture_parts(&Device::Cpu, DType::F32)?;
    let path = std::env::temp_dir().join(format!(
        "haricot-nvfp4-fixture-{}.safetensors",
        std::process::id()
    ));
    candle_core::safetensors::save(
        &std::collections::HashMap::from([
            ("fixture.weight", parts.weight.clone()),
            ("fixture.weight_scale", parts.scales.clone()),
            ("fixture.weight_scale_2", Tensor::new(1f32, &Device::Cpu)?),
        ]),
        &path,
    )?;
    let payload = ModelOptNvfp4Shards::single_file(&path).load(2, NVFP4_K, "fixture");
    std::fs::remove_file(path)?;
    payload?;
    let layer = Nvfp4Layer::from_parts(parts)?;
    let input = Tensor::new(&[[1f32; NVFP4_K], [2f32; NVFP4_K]], &Device::Cpu)?;
    assert_eq!(
        layer.forward(&input)?.to_vec2::<f32>()?,
        vec![vec![16., 64.], vec![32., 128.]]
    );
    Ok(())
}

fn mxfp4_embedding(device: &Device, dtype: candle_core::DType) -> Result<()> {
    use candle_core::{
        quantized::{GgmlDType, QTensor},
        DType,
    };
    use mistralrs_quant::EmbeddingTable;
    let dense = Tensor::from_vec(
        (0..64)
            .map(|i| (i as f32 - 31.5) * 0.125)
            .collect::<Vec<_>>(),
        (2, 32),
        &Device::Cpu,
    )?;
    let weights = QTensor::quantize(&dense, GgmlDType::Mxfp4)?;
    let ids = Tensor::new(&[1u32, 0, 1], &Device::Cpu)?;
    let reference = weights
        .dequantize(&Device::Cpu)?
        .index_select(&ids, 0)?
        .to_dtype(dtype)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    #[cfg(feature = "cuda")]
    let weights = if device.is_cuda() {
        use candle_core::quantized::QStorage;
        QTensor::new(
            QStorage::from_data(weights.data()?, device, weights.dtype())?,
            weights.shape().clone(),
        )?
    } else {
        weights
    };
    let actual =
        EmbeddingTable::quantized(&weights)?.gather(&ids.to_device(device)?, dtype.into())?;
    device.synchronize()?;
    assert_eq!(
        actual
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?,
        reference
    );
    Ok(())
}

#[test]
fn campaign_cpu_mxfp4_embedding() -> Result<()> {
    mxfp4_embedding(&Device::Cpu, candle_core::DType::F32)
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp4"))]
#[test]
fn campaign_cuda_nvfp4_sm61_w4a16() -> Result<()> {
    use candle_core::DType;
    use mistralrs_quant::nvfp4::Nvfp4Layer;
    use mistralrs_quant::QuantMethod;
    let device = Device::new_cuda(0)?;
    let input = Tensor::new(&[[1f32; NVFP4_K], [2f32; NVFP4_K]], &device)?;
    for dtype in [DType::F16, DType::BF16] {
        let layer = Nvfp4Layer::from_parts(nvfp4_fixture_parts(&device, dtype)?)?;
        let actual = layer.forward(&input.to_dtype(dtype)?)?;
        device.synchronize()?;
        assert_eq!(actual.dtype(), dtype);
        assert_eq!(
            actual
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?,
            vec![vec![16., 64.], vec![32., 128.]]
        );
    }
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        mxfp4_embedding(&device, dtype)?;
    }
    Ok(())
}
