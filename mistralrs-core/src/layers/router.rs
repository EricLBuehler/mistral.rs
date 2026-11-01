use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

#[cfg(feature = "cuda")]
const ROUTER_VECTOR_ALIGNMENT: usize = 2;

pub(crate) fn router_linear(linear: &Linear, input: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let weight = linear.weight();
        if matches!(
            input.dtype(),
            candle_core::DType::BF16 | candle_core::DType::F16
        ) && input.elem_count() == input.dim(candle_core::D::Minus1)?
            && weight.is_contiguous()
            && input.device().same_device(weight.device())
            && input
                .layout()
                .start_offset()
                .is_multiple_of(ROUTER_VECTOR_ALIGNMENT)
            && weight
                .layout()
                .start_offset()
                .is_multiple_of(ROUTER_VECTOR_ALIGNMENT)
            && mistralrs_quant::gemv::should_use_gemv(input, weight)
        {
            let output = mistralrs_quant::gemv::gemv(input, weight, None)?;
            return match linear.bias() {
                Some(bias) => output.broadcast_add(bias),
                None => Ok(output),
            };
        }
    }
    linear.forward(input)
}

#[cfg(all(test, feature = "cuda"))]
#[path = "router_tests.rs"]
mod tests;
