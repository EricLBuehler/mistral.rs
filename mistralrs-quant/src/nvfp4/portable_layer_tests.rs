use super::*;

const EXPERTS: usize = 3;
const TOPK: usize = 2;
const PRECISION_COLUMNS: usize = NVFP4_BLOCK_SIZE;
const PRECISION_OUTPUTS: usize = 2 * PRECISION_COLUMNS;
const PRECISION_BATCHES: [usize; 3] = [1, 2, 17];

fn portable_layer(parts: Nvfp4LayerParts) -> Result<Nvfp4Layer> {
    force_portable_kernels(true);
    let layer = Nvfp4Layer::from_parts(parts);
    force_portable_kernels(false);
    let layer = layer?;
    assert!(layer.portable);
    Ok(layer)
}

fn cuda() -> Result<Option<Device>> {
    let device = Device::new_cuda(0)?;
    Ok(portable::supported(device.as_cuda_device()?).then_some(device))
}

#[test]
fn portable_activation_precision_is_independent_of_batch_size() -> Result<()> {
    let Some(device) = cuda()? else {
        return Ok(());
    };
    for dtype in [DType::BF16, DType::F16] {
        for experts in [None, Some(EXPERTS)] {
            let count = experts.unwrap_or(1);
            let mut shape = experts.into_iter().collect::<Vec<_>>();
            shape.push(PRECISION_OUTPUTS);
            let with_columns = |columns| {
                let mut dims = shape.clone();
                dims.push(columns);
                dims
            };
            let layer = portable_layer(Nvfp4LayerParts {
                weight: Tensor::from_vec(
                    vec![0x22u8; count * PRECISION_OUTPUTS * PRECISION_COLUMNS / 2],
                    with_columns(PRECISION_COLUMNS / 2),
                    &device,
                )?,
                scales: Tensor::from_vec(
                    vec![F8E4M3::from_f32(1.0); count * PRECISION_OUTPUTS],
                    with_columns(1),
                    &device,
                )?,
                global_scales: Tensor::ones(shape, DType::F32, &device)?,
                input_scale: Some(match experts {
                    Some(count) => Tensor::ones(count, DType::F32, &device)?,
                    None => Tensor::new(1f32, &device)?,
                }),
                activation: Nvfp4ActivationMode::DynamicBlock,
                bias: None,
                dtype,
            })?;
            let mut row = [0.2f32; PRECISION_COLUMNS];
            row[0] = 1.0;
            for batch in PRECISION_BATCHES {
                let x = Tensor::from_vec(row.repeat(batch), (batch, PRECISION_COLUMNS), &device)?
                    .to_dtype(dtype)?;
                let expected = x.to_dtype(DType::F32)?.sum_keepdim(1)?.to_dtype(dtype)?;
                let y = match experts {
                    Some(_) => {
                        let ids = Tensor::zeros((batch, TOPK), DType::U32, &device)?;
                        layer.gather_forward(&x.unsqueeze(1)?, &ids)?
                    }
                    None => layer.forward(&x)?,
                };
                let expected = match experts {
                    Some(_) => expected.unsqueeze(1)?,
                    None => expected,
                };
                let error = y
                    .broadcast_sub(&expected)?
                    .abs()?
                    .max_all()?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?;
                assert_eq!(error, 0.0, "{dtype:?} experts={experts:?} batch={batch}");
            }
        }
    }
    Ok(())
}
