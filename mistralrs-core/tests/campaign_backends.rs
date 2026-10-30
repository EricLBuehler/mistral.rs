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
