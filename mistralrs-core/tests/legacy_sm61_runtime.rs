#![cfg(feature = "cuda")]
use candle_core::{DType, Device, Result, Tensor};

#[test]
fn legacy_sm61_f32_bf16_fp8_roundtrip() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let input = Tensor::new(&[1f32, 2., 3., 4.], &device)?;
    let doubled = (&input + &input)?;
    assert_eq!(
        doubled.to_device(&Device::Cpu)?.to_vec1::<f32>()?,
        vec![2., 4., 6., 8.]
    );
    for dtype in [DType::BF16, DType::F8E4M3] {
        let tensor = input.to_dtype(dtype)?;
        let restored = tensor.to_dtype(DType::F32)?;
        device.synchronize()?;
        let values = restored.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        for (value, expected) in values.iter().zip([1., 2., 3., 4.]) {
            assert!(
                (value - expected).abs() < 0.1,
                "{dtype:?}: {value} != {expected}"
            );
        }
    }
    Ok(())
}
