use candle_core::{Result, Tensor};

use crate::{QuantMethod, QuantMethodConfig};

pub struct GptQMatMul;

impl QuantMethod for GptQMatMul {
    fn new(method: QuantMethodConfig) -> Self
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::GptQ {
                bits: _,
                use_exllama: _,
                q_weight: _,
                gptq_qzeros: _,
                gptq_scales: _,
                g_idx: _,
            } => todo!(),
        }
    }

    fn matmul(&mut self, _a: &Tensor) -> Result<Tensor> {
        todo!()
    }
}
