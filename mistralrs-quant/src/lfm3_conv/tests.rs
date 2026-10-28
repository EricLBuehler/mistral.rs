use candle_core::Device;

use super::*;

const BATCH: usize = 2;
const HIDDEN: usize = 5;
const SEQUENCE: usize = 13;
const CHUNKS: [usize; 4] = [2, 1, 6, 4];

fn values(length: usize, period: usize) -> Vec<f32> {
    (0..length)
        .map(|index| ((index % period) as f32 - (period / 2) as f32) / period as f32)
        .collect()
}

fn round(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => half::f16::from_f32(value).to_f32(),
        DType::BF16 => half::bf16::from_f32(value).to_f32(),
        _ => value,
    }
}

fn reference(
    projected: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    state: &Tensor,
    use_existing: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let dtype = projected.dtype();
    let (batch, seq, projection) = projected.dims3()?;
    let hidden = projection / PROJECTION_PARTS;
    let width = weight.dim(2)?;
    let x = projected
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let weight = weight
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let bias = bias
        .map(|bias| {
            bias.to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()
        })
        .transpose()?;
    let mut history = state
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    if !use_existing {
        for row in &mut history {
            for channel in row {
                channel.fill(0.0);
            }
        }
    }
    let mut output = Vec::with_capacity(batch * seq * hidden);
    for batch in 0..batch {
        for token in &x[batch] {
            for channel in 0..hidden {
                let gated = round(token[channel] * token[2 * hidden + channel], dtype);
                history[batch][channel].rotate_left(1);
                history[batch][channel][width - 1] = gated;
                let mut sum = 0.0;
                for tap in 0..width {
                    sum += history[batch][channel][tap] * weight[channel][0][tap];
                }
                if let Some(bias) = &bias {
                    sum += bias[channel];
                }
                output.push(round(round(sum, dtype) * token[hidden + channel], dtype));
            }
        }
    }
    Ok((output, history.into_iter().flatten().flatten().collect()))
}

fn assert_close(actual: &Tensor, expected: &[f32], dtype: DType) -> Result<()> {
    let actual = actual
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let tolerance = match dtype {
        DType::F16 => 1e-4,
        DType::BF16 => 1e-3,
        _ => 1e-6,
    };
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: {actual} != {expected}"
        );
    }
    Ok(())
}

fn check_backend(device: &Device) -> Result<()> {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for width in [3, OPTIMIZED_KERNEL_WIDTH] {
            let projected_shape = (BATCH + 2, PROJECTION_PARTS * HIDDEN, SEQUENCE + 2);
            let projected = Tensor::from_vec(
                values(
                    projected_shape.0 * projected_shape.1 * projected_shape.2,
                    29,
                ),
                projected_shape,
                device,
            )?
            .to_dtype(dtype)?
            .narrow(0, 1, BATCH)?
            .narrow(2, 1, SEQUENCE)?
            .transpose(1, 2)?;
            let weight = Tensor::from_vec(
                values(width * (HIDDEN + 2), 17),
                (width, HIDDEN + 2),
                device,
            )?
            .to_dtype(dtype)?
            .narrow(1, 1, HIDDEN)?
            .t()?
            .unsqueeze(1)?;
            let state = Tensor::from_vec(
                values((BATCH + 2) * width * (HIDDEN + 2), 19),
                (BATCH + 2, width, HIDDEN + 2),
                device,
            )?
            .to_dtype(dtype)?
            .narrow(0, 1, BATCH)?
            .narrow(2, 1, HIDDEN)?
            .transpose(1, 2)?;
            let bias = Tensor::from_vec(values(HIDDEN * 2, 11), (HIDDEN, 2), device)?
                .to_dtype(dtype)?
                .narrow(1, 1, 1)?
                .squeeze(1)?;
            for bias in [None, Some(&bias)] {
                for use_existing in [false, true] {
                    let (expected, expected_state) =
                        reference(&projected, &weight, bias, &state, use_existing)?;
                    let (output, next_state) =
                        lfm3_causal_conv1d(&projected, &weight, bias, &state, use_existing)?;
                    assert_close(&output, &expected, dtype)?;
                    assert_close(&next_state, &expected_state, dtype)?;

                    let mut offset = 0;
                    let mut cached = state.clone();
                    let mut chunks = Vec::new();
                    for length in CHUNKS {
                        let (output, next) = lfm3_causal_conv1d(
                            &projected.narrow(1, offset, length)?,
                            &weight,
                            bias,
                            &cached,
                            offset != 0 || use_existing,
                        )?;
                        chunks.push(output);
                        cached = next;
                        offset += length;
                    }
                    assert_close(&Tensor::cat(&chunks, 1)?, &expected, dtype)?;
                    assert_close(&cached, &expected_state, dtype)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn cpu_prefill_decode_and_cached_chunks_match_reference() -> Result<()> {
    check_backend(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn metal_prefill_decode_and_cached_chunks_match_reference() -> Result<()> {
    check_backend(&Device::new_metal(0)?)
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_prefill_decode_and_cached_chunks_match_reference() -> Result<()> {
    check_backend(&Device::new_cuda(0)?)
}
