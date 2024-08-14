use std::{
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{quantized::GgmlDType, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

use crate::{generate_isq, GgufMatMul, IsqType, QuantMethod, QuantMethodConfig};

#[derive(Debug)]
pub struct UnquantLinear(Linear);

impl QuantMethod for UnquantLinear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. } | QuantMethodConfig::Gptq { .. } => unreachable!(),
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
        dtype: IsqType,
        device: Device,
        n_quantized: &AtomicUsize,
    ) -> Result<Arc<dyn QuantMethod>> {
        match dtype {
            IsqType::Q2K
            | IsqType::Q3K
            | IsqType::Q4K
            | IsqType::Q4_0
            | IsqType::Q4_1
            | IsqType::Q5K
            | IsqType::Q5_0
            | IsqType::Q5_1
            | IsqType::Q6K
            | IsqType::Q8K
            | IsqType::Q8_0
            | IsqType::Q8_1 => {
                let dtype = dtype.try_into()?;
                let res = generate_isq!(self.0.weight(), device, dtype, n_quantized);
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
    }

    fn get_max_isq_cpu_threads(&self, dtype: IsqType) -> Option<NonZeroUsize> {
        match dtype {
            IsqType::Q2K
            | IsqType::Q3K
            | IsqType::Q4K
            | IsqType::Q4_0
            | IsqType::Q4_1
            | IsqType::Q5K
            | IsqType::Q5_0
            | IsqType::Q5_1
            | IsqType::Q6K
            | IsqType::Q8K
            | IsqType::Q8_0
            | IsqType::Q8_1 => None,
        }
    }
}
