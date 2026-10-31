use super::*;
use candle_core::Device;
use float8::F8E4M3;

const EXPERTS: usize = 3;
const TOPK: usize = 2;
const ROWS: usize = 513;
const OUTPUTS: usize = 513;
const INPUTS: usize = 640;
const WEIGHT_SCALE: f32 = 0.5;
const GLOBAL_SCALE_STEP: f32 = 0.25;
// E4M3 rounds the activation scale 1/6 to 11/64 before multiplying by FP4's maximum of 6.
const ACTIVATION_ROUNDTRIP: f32 = 33.0 / 32.0;

#[test]
fn routed_a4_prefill_matches_expert_scaled_reference() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    if !nvfp4_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let weights = Tensor::from_vec(
        vec![0x22u8; EXPERTS * OUTPUTS * INPUTS / 2],
        (EXPERTS, OUTPUTS, INPUTS / 2),
        &device,
    )?;
    let scales = Tensor::from_vec(
        vec![F8E4M3::from_f32(WEIGHT_SCALE); EXPERTS * OUTPUTS * INPUTS / BLOCK_SIZE],
        (EXPERTS, OUTPUTS, INPUTS / BLOCK_SIZE),
        &device,
    )?;
    let globals = Tensor::from_vec(
        (0..EXPERTS * OUTPUTS)
            .map(|i| (i / OUTPUTS + 1) as f32 * GLOBAL_SCALE_STEP)
            .collect::<Vec<_>>(),
        (EXPERTS, OUTPUTS),
        &device,
    )?;
    let activation = Tensor::ones(EXPERTS, DType::F32, &device)?;
    let ids = (0..ROWS * TOPK)
        .map(|i| (i % EXPERTS) as u32)
        .collect::<Vec<_>>();
    let indices = Tensor::from_vec(ids.clone(), (ROWS, TOPK), &device)?;
    let input = Tensor::ones((ROWS, INPUTS), DType::BF16, &device)?;
    let output = launch(
        &input,
        Some(&indices),
        Nvfp4GemmArgs {
            weights: &weights,
            weight_scales: &scales,
            weight_global_scale: &globals,
            activation_global_scale: Some(&activation),
        },
        false,
    )?;
    let expected = ids
        .into_iter()
        .flat_map(|expert| {
            std::iter::repeat_n(
                INPUTS as f32
                    * ACTIVATION_ROUNDTRIP
                    * WEIGHT_SCALE
                    * GLOBAL_SCALE_STEP
                    * (expert + 1) as f32,
                OUTPUTS,
            )
        })
        .collect::<Vec<_>>();
    let expected =
        Tensor::from_vec(expected, (ROWS, TOPK, OUTPUTS), &Device::Cpu)?.to_dtype(DType::BF16)?;
    assert_eq!(
        output.flatten_all()?.to_vec1::<half::bf16>()?,
        expected.flatten_all()?.to_vec1::<half::bf16>()?
    );
    Ok(())
}
