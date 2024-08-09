use std::sync::Arc;

use candle_core::{quantized::QMatMul, DType, Result, Tensor};
use candle_nn::{Linear, Module};

use crate::{QuantMethod, QuantMethodConfig};

pub struct UnquantLinear(Linear);

impl QuantMethod for UnquantLinear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { q_weight: _ }
            | QuantMethodConfig::Gptq {
                bits: _,
                use_exllama: _,
                q_weight: _,
                gptq_qzeros: _,
                gptq_scales: _,
                g_idx: _,
                bias: _,
            } => unreachable!(),
            QuantMethodConfig::Unquantized(l) => Ok(Self(l)),
        }
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        self.0.forward(a)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self(Linear::new(
            (self.0.weight() + delta)?,
            self.0.bias().cloned(),
        ))))
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (self.0.weight().dtype(), self.0.weight().device().clone())
    }

    fn get_qmatmul(&mut self) -> Option<&mut QMatMul> {
        None
    }
}
