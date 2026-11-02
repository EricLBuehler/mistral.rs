use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{QuantMethod, QuantMethodConfig, UnquantLinear};

use super::grouped_rms_norm;

const UP_MIX_ROWS_PER_MULTIPROCESSOR: usize = 64;
const UP_MIX_ROUNDING_RANK: usize = 64;
const UP_MIX_ROUNDING_CASES: [(DType, f32, f32, f32, f32); 2] = [
    (
        DType::BF16,
        1.0 / 256.0,
        37.0 / 128.0,
        27.0 / 128.0,
        217.0 / 1024.0,
    ),
    (
        DType::F16,
        1.0 / 2048.0,
        9.0 / 32.0,
        1684.0 / 8192.0,
        1685.0 / 8192.0,
    ),
];

fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()
}

fn up_mix_test_hidden(device: &Device) -> Result<usize> {
    use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;
    let multiprocessors = device
        .as_cuda_device()?
        .cuda_stream()
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(candle_core::Error::wrap)?;
    Ok(multiprocessors as usize * UP_MIX_ROWS_PER_MULTIPROCESSOR + 3)
}

#[test]
fn vector_hyper_connection_upmix_rounds_gates_before_sigmoid() -> Result<()> {
    use crate::cuda::hyper_connection::try_hc_up_mix;

    let device = Device::new_cuda(0)?;
    let hidden = up_mix_test_hidden(&device)?;
    for (dtype, delta, residual, rounded, unrounded) in UP_MIX_ROUNDING_CASES {
        assert_ne!(rounded, unrounded);
        let mut row = vec![0f32; UP_MIX_ROUNDING_RANK];
        row[0] = 1.0;
        row[1] = delta;
        let weight = Tensor::from_vec(row.repeat(hidden), (hidden, UP_MIX_ROUNDING_RANK), &device)?
            .to_dtype(dtype)?;
        row[1] = 1.0;
        let lo = Tensor::from_vec(row, (1, UP_MIX_ROUNDING_RANK), &device)?.to_dtype(dtype)?;
        let input = Tensor::full(residual, (1, hidden), &device)?.to_dtype(dtype)?;
        let expected = Tensor::full(rounded, (1, hidden), &device)?.to_dtype(dtype)?;
        let projection = UnquantLinear::new(QuantMethodConfig::Unquantized(
            candle_nn::Linear::new(weight, None),
        ))?;
        let actual = try_hc_up_mix(&input, &lo, &projection, 1)?.unwrap();
        assert_eq!(max_diff(&actual, &expected)?, 0.0, "{dtype:?}");
    }
    Ok(())
}

#[test]
fn fused_hyper_connection_kernels_match_reference() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for (tokens, hc, hidden) in [(5, 4, 96), (1, 2, 193), (17, 3, 511)] {
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let res = Tensor::randn(0f32, 1.0, (tokens, hc * hidden), &device)?.to_dtype(dtype)?;
            let weight = Tensor::randn(1f32, 0.1, hc * hidden, &device)?;
            let eps = 1e-6;

            let fused = crate::cuda::hyper_connection::hc_norm(&res, &weight, hc, eps)?;
            let reference = grouped_rms_norm(&res, &weight, hc, eps)?;
            assert!(max_diff(&fused, &reference)? < 0.05);

            let gate = Tensor::randn(0f32, 1.0, (tokens, hc * hidden), &device)?.to_dtype(dtype)?;
            let mixed = crate::cuda::hyper_connection::hc_mix(&fused, &gate, hc)?;
            let reference = (fused.to_dtype(DType::F32)?
                * candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?)?
            .reshape((tokens, hc, hidden))?
            .mean(1)?;
            assert!(max_diff(&mixed, &reference)? < 0.02);

            let out = Tensor::randn(0f32, 1.0, (tokens, hidden), &device)?.to_dtype(dtype)?;
            let inject = Tensor::randn(0f32, 1.0, (tokens, hc), &device)?.to_dtype(dtype)?;
            let (combined, next) = crate::cuda::hyper_connection::hc_combine(
                &res,
                &out,
                &inject,
                Some(&weight),
                hc,
                eps,
            )?;
            let w = (candle_nn::ops::sigmoid(&(inject.to_dtype(DType::F32)? / hc as f64)?)? * 2.0)?;
            let reference = (res.to_dtype(DType::F32)?.reshape((tokens, hc, hidden))?
                + out
                    .to_dtype(DType::F32)?
                    .unsqueeze(1)?
                    .broadcast_mul(&w.unsqueeze(2)?)?)?
            .reshape((tokens, hc * hidden))?;
            assert!(max_diff(&combined, &reference)? < 0.05);
            let next_reference = grouped_rms_norm(&combined, &weight, hc, eps)?;
            assert!(max_diff(&next.unwrap(), &next_reference)? < 0.05);
        }
    }
    Ok(())
}

#[test]
fn hyper_connection_orders_cross_stream_inputs_and_outputs() -> Result<()> {
    use crate::cuda::hyper_connection::{hc_combine, hc_mix, hc_norm, hc_silu};

    const TOKENS: usize = 129;
    const STREAMS: usize = 3;
    const HIDDEN: usize = 768;
    const WIDTH: usize = STREAMS * HIDDEN;
    const UPDATES: usize = 32;
    const EPS: f64 = 1e-6;
    const TOLERANCE: f32 = 1e-5;
    let first = Device::new_cuda_with_stream(0)?;
    let second = Device::new_cuda_with_stream(0)?;
    let third = Device::new_cuda_with_stream(0)?;
    let matrix = |rows, cols, salt: f32, device: &Device| {
        Tensor::from_vec(
            (0..rows * cols)
                .map(|i| (i as f32 * 0.17 + salt).sin())
                .collect::<Vec<_>>(),
            (rows, cols),
            device,
        )
    };
    let residual = matrix(TOKENS + 1, WIDTH, 0.2, &first)?.narrow(0, 1, TOKENS)?;
    let mut gate = matrix(TOKENS + 1, WIDTH, 0.8, &second)?;
    let mut weight = matrix(1, WIDTH, 1.3, &third)?.flatten_all()?;
    let injection = matrix(TOKENS, STREAMS, 1.7, &third)?;
    for _ in 0..UPDATES {
        gate = gate.affine(0.99, 0.01)?;
        weight = weight.affine(0.99, 0.01)?;
    }
    let gate = gate.narrow(0, 1, TOKENS)?;
    let normalized = hc_norm(&residual, &weight, STREAMS, EPS)?;
    let activated = hc_silu(&gate, STREAMS)?;
    let mixed = hc_mix(&activated, &normalized, STREAMS)?;
    let (combined, next) = hc_combine(&residual, &mixed, &injection, Some(&weight), STREAMS, EPS)?;
    let consumed = hc_mix(&gate, &next.unwrap(), STREAMS)?.to_device(&Device::Cpu)?;

    let residual = residual.to_device(&Device::Cpu)?;
    let gate = gate.to_device(&Device::Cpu)?;
    let weight = weight.to_device(&Device::Cpu)?;
    let injection = injection.to_device(&Device::Cpu)?;
    let normalized = grouped_rms_norm(&residual, &weight, STREAMS, EPS)?;
    let activated = candle_nn::ops::silu(&(&gate / STREAMS as f64)?)?;
    let mixed = (activated * candle_nn::ops::sigmoid(&normalized)?)?
        .reshape((TOKENS, STREAMS, HIDDEN))?
        .mean(1)?;
    let injection = (candle_nn::ops::sigmoid(&(injection / STREAMS as f64)?)? * 2.0)?;
    let expected = (residual.reshape((TOKENS, STREAMS, HIDDEN))?
        + mixed
            .unsqueeze(1)?
            .broadcast_mul(&injection.unsqueeze(2)?)?)?
    .reshape((TOKENS, WIDTH))?;
    assert!(max_diff(&combined.to_device(&Device::Cpu)?, &expected)? < TOLERANCE);
    let expected = (gate
        * candle_nn::ops::sigmoid(&grouped_rms_norm(&expected, &weight, STREAMS, EPS)?)?)?
    .reshape((TOKENS, STREAMS, HIDDEN))?
    .mean(1)?;
    assert!(max_diff(&consumed, &expected)? < TOLERANCE);
    Ok(())
}

#[test]
fn hyper_connection_consumes_offset_and_strided_projection_views() -> Result<()> {
    use crate::cuda::hyper_connection::{hc_combine, hc_silu};

    const TOKENS: usize = 17;
    const STREAMS: usize = 3;
    const HIDDEN: usize = 193;
    const RANK: usize = 32;
    const PAD: usize = 7;
    const EPS: f64 = 1e-6;
    let device = Device::new_cuda(0)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        for batch in [1, 2] {
            let combined = Tensor::randn(
                0f32,
                1.0,
                (batch, TOKENS + 1, RANK + STREAMS + PAD),
                &device,
            )?
            .to_dtype(dtype)?
            .narrow(1, 1, TOKENS)?;
            let down = combined.narrow(2, 1, RANK)?;
            let inject = combined.narrow(2, RANK + 1, STREAMS)?;
            assert!(!down.is_contiguous());
            assert!(down.layout().start_offset() > 0);
            assert_eq!(
                max_diff(
                    &hc_silu(&down, STREAMS)?,
                    &hc_silu(&down.contiguous()?, STREAMS)?
                )?,
                0.0,
            );
            let residual = Tensor::randn(0f32, 1.0, (batch, TOKENS, STREAMS * HIDDEN), &device)?
                .to_dtype(dtype)?;
            let output =
                Tensor::randn(0f32, 1.0, (batch, TOKENS, HIDDEN), &device)?.to_dtype(dtype)?;
            let norm = Tensor::ones(STREAMS * HIDDEN, DType::F32, &device)?;
            let actual = hc_combine(&residual, &output, &inject, Some(&norm), STREAMS, EPS)?;
            let expected = hc_combine(
                &residual,
                &output,
                &inject.contiguous()?,
                Some(&norm),
                STREAMS,
                EPS,
            )?;
            assert_eq!(max_diff(&actual.0, &expected.0)?, 0.0);
            assert_eq!(max_diff(&actual.1.unwrap(), &expected.1.unwrap())?, 0.0);
        }
    }
    Ok(())
}
