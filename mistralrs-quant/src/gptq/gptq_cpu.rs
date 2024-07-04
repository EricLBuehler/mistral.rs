use candle_core::{Result, Tensor};

use crate::{QuantMethod, QuantMethodConfig};

pub struct GptqMatMul;

impl QuantMethod for GptqMatMul {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gptq {
                bits: _,
                use_exllama: _,
                q_weight: _,
                gptq_qzeros: _,
                gptq_scales: _,
                g_idx: _,
                bias: _,
            } => todo!(),
            QuantMethodConfig::Gguf { q_weight: _ } => unreachable!(),
        }
    }

    fn matmul(&self, _a: &Tensor) -> Result<Tensor> {
        todo!()
    }
}
