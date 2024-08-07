use std::sync::Arc;

use candle_core::{quantized::QMatMul, DType, Result, Tensor};
use candle_nn::{Linear, Module};

use crate::{GgufMatMul, QuantMethod, QuantMethodConfig};

pub struct UnquantLinear(Linear);

impl QuantMethod for UnquantLinear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { q_weight: _, b: _ }
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

    fn get_bias_mut(&mut self) -> Option<&mut Tensor> {
        None
    }

    fn convert_to_isq(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>> {
        let w = self.0.weight().clone();
        let b = self.0.bias().cloned();

        Ok(Arc::new(GgufMatMul {
            w: QMatMul::Tensor(w),
            b,
        }))
    }
}
