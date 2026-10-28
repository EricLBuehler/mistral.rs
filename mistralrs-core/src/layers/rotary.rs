use candle_core::{DType, Result, Tensor, D};

// Unlike the fused attention norm, this preserves the dtype rounding before rotation.
pub(crate) fn rms_norm_partial_rotary_neox(
    input: &Tensor,
    weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(output) = crate::cuda::rounded_rope::try_rms_norm_partial_rotary_neox(
        input, weight, cos, sin, epsilon,
    )? {
        return Ok(output);
    }
    let input = candle_nn::ops::rms_norm(&input.contiguous()?, weight, epsilon)?;
    partial_rotary_neox(&input, cos, sin)
}

fn partial_rotary_neox(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = cos.dim(D::Minus1)?;
    let columns = input.dim(D::Minus1)?;
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(D::Minus2)?;
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(D::Minus2)?;
    let values = input.to_dtype(DType::F32)?;
    let x = values.narrow(D::Minus1, 0, half)?;
    let y = values.narrow(D::Minus1, half, half)?;
    let first = (x.broadcast_mul(&cos)? - y.broadcast_mul(&sin)?)?;
    let second = (y.broadcast_mul(&cos)? + x.broadcast_mul(&sin)?)?;
    let mut parts = vec![first, second];
    if columns > 2 * half {
        parts.push(values.narrow(D::Minus1, 2 * half, columns - 2 * half)?);
    }
    Tensor::cat(&parts, D::Minus1)?.to_dtype(input.dtype())
}

#[cfg(test)]
#[path = "rotary_tests.rs"]
mod tests;
