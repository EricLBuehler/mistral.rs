use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{quantized::GgmlDType, DType, Result, Tensor};
use candle_nn::{Linear, Module};

use crate::{generate_isq, GgufMatMul, QuantMethod, QuantMethodConfig};

#[derive(Debug)]
pub struct UnquantLinear(Linear);

impl QuantMethod for UnquantLinear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Gptq { .. }
            | QuantMethodConfig::Hqq { .. } => unreachable!(),
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

    fn get_bias_mut(&mut self) -> Option<&mut Tensor> {
        None
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: GgmlDType,
        n_quantized: &AtomicUsize,
    ) -> Result<Arc<dyn QuantMethod>> {
        let res = generate_isq!(
            self.0.weight(),
            self.0.weight().device(),
            dtype,
            n_quantized
        );
        Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: res,
            b: self
                .0
                .bias()
                .cloned()
                .map(|b| b.to_dtype(DType::F32).unwrap()),
        })?))
    }
}
