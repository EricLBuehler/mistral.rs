use super::*;

const N: usize = 77;
const K: usize = 208;
const EXPERTS: usize = 3;
const TOPK: usize = 2;
const TOKENS: [usize; 3] = [1, 17, 512];
const PREFILL_TOKENS: [usize; 2] = [17, 512];
const INPUT_GLOBAL: f32 = 0.0625;
const WEIGHT_GLOBAL: [f32; 3] = [0.03125, 0.5, 0.09375];
const SCALES: [f32; 4] = [0.3125, 0.75, 1.5, 2.75];
const TOLERANCE: f64 = 1e-2;

fn parts(experts: Option<usize>, w4a4: bool, device: &Device) -> Result<Nvfp4LayerParts> {
    let count = experts.unwrap_or(1);
    let mut shape = experts.map_or(vec![], |e| vec![e]);
    shape.push(N);
    let rows = shape.clone();
    let weight = (0..count * N * K / 2)
        .map(|i| ((i * 37 + i / (K / 2) * 13) % 256) as u8)
        .collect::<Vec<_>>();
    let scales = (0..count * N * K / NVFP4_BLOCK_SIZE)
        .map(|i| F8E4M3::from_f32(SCALES[i % SCALES.len()]))
        .collect::<Vec<_>>();
    let global = (0..count * N)
        .map(|i| WEIGHT_GLOBAL[i % WEIGHT_GLOBAL.len()])
        .collect::<Vec<_>>();
    let bias = (0..count * N)
        .map(|i| (i % 11) as f32 / 64.0)
        .collect::<Vec<_>>();
    let with_last = |last: usize| {
        let mut dims = rows.clone();
        dims.push(last);
        dims
    };
    Ok(Nvfp4LayerParts {
        weight: Tensor::from_vec(weight, with_last(K / 2), device)?,
        scales: Tensor::from_vec(scales, with_last(K / NVFP4_BLOCK_SIZE), device)?,
        global_scales: Tensor::from_vec(global, rows.clone(), device)?,
        input_scale: w4a4
            .then(|| match experts {
                Some(e) => Tensor::full(INPUT_GLOBAL, e, device),
                None => Tensor::new(INPUT_GLOBAL, device),
            })
            .transpose()?,
        activation: if w4a4 {
            Nvfp4ActivationMode::DynamicBlock
        } else {
            Nvfp4ActivationMode::None
        },
        bias: Some(Tensor::from_vec(bias, rows, device)?.to_dtype(DType::BF16)?),
        dtype: DType::BF16,
    })
}

fn rel_err(actual: &Tensor, expected: &Tensor) -> Result<f64> {
    let actual = actual.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let expected = expected.to_dtype(DType::F32)?;
    let diff = (&actual - &expected)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    let norm = expected.sqr()?.sum_all()?.to_scalar::<f32>()?;
    Ok(f64::from(diff / norm).sqrt())
}

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

// W4A16 reference: dequantized weights against unquantized activations, which is what W4A4 checkpoints get here too.
fn dense_reference(cpu: &Nvfp4Layer, x: &Tensor) -> Result<Tensor> {
    let weight = cpu.dequantize_w()?.to_dtype(DType::F32)?;
    x.to_dtype(DType::F32)?
        .matmul(&weight.t()?)?
        .broadcast_add(&cpu.parts.bias.as_ref().unwrap().to_dtype(DType::F32)?)
}

#[test]
fn forced_dense_layer_matches_w4a16_reference() -> Result<()> {
    let Some(device) = cuda()? else {
        return Ok(());
    };
    for w4a4 in [false, true] {
        let cpu = Nvfp4Layer::from_parts(parts(None, w4a4, &Device::Cpu)?)?;
        let gpu = portable_layer(parts(None, w4a4, &device)?)?;
        let tokens: &[usize] = if w4a4 { &PREFILL_TOKENS } else { &TOKENS };
        for &rows in tokens {
            let x = Tensor::randn(0f32, 1f32, (rows, K), &Device::Cpu)?.to_dtype(DType::BF16)?;
            let gpu_x = x.to_device(&device)?;
            assert!(gpu.activation_quantization_scheme_for(&gpu_x).is_none());
            let y = gpu.forward(&gpu_x)?;
            let err = rel_err(&y, &dense_reference(&cpu, &x)?)?;
            assert!(err <= TOLERANCE, "w4a4={w4a4} rows={rows}: rel err {err}");
        }
    }
    Ok(())
}

#[test]
fn forced_expert_gather_matches_w4a16_reference() -> Result<()> {
    let Some(device) = cuda()? else {
        return Ok(());
    };
    for w4a4 in [false, true] {
        let cpu = Nvfp4Layer::from_parts(parts(Some(EXPERTS), false, &Device::Cpu)?)?;
        let gpu = portable_layer(parts(Some(EXPERTS), w4a4, &device)?)?;
        let tokens: &[usize] = if w4a4 { &PREFILL_TOKENS } else { &TOKENS };
        for &rows in tokens {
            let x = Tensor::randn(0f32, 1f32, (rows, 1, K), &Device::Cpu)?.to_dtype(DType::BF16)?;
            let ids = (0..rows * TOPK)
                .map(|i| ((i / TOPK + i % TOPK) % EXPERTS) as u32)
                .collect::<Vec<_>>();
            let cpu_ids = Tensor::from_vec(ids, (rows, TOPK), &Device::Cpu)?;
            let gpu_ids = cpu_ids.to_device(&device)?;
            let gpu_x = x.to_device(&device)?;
            assert!(gpu
                .group_member(&gpu_x, crate::GroupKind::Gather { indices: &gpu_ids })?
                .is_none());
            let y = gpu.gather_forward(&gpu_x, &gpu_ids)?;
            let expected = cpu.gather_forward(&x.to_dtype(DType::F32)?, &cpu_ids)?;
            assert_eq!(y.dims(), expected.dims());
            let err = rel_err(&y, &expected)?;
            assert!(err <= TOLERANCE, "w4a4={w4a4} rows={rows}: rel err {err}");
        }
    }
    Ok(())
}
