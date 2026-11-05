use candle_core::{Result, Tensor};

pub(crate) fn sigmoid_gated_add(
    residual: &Tensor,
    value: &Tensor,
    gate: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(output) = crate::cuda::gated_add::try_sigmoid_gated_add(residual, value, gate)? {
        return Ok(output);
    }
    residual + value.broadcast_mul(&candle_nn::ops::sigmoid(gate)?)?
}

#[cfg(all(test, feature = "cuda"))]
#[path = "gated_add_tests.rs"]
mod tests;
