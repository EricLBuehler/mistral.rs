use candle_core::{quantized::QMatMul, Result, Tensor};
use candle_nn::Module;

use crate::{QuantMethod, QuantMethodConfig};

pub struct GgufMatMul(QMatMul);

impl QuantMethod for GgufMatMul {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { q_weight } => Ok(Self(QMatMul::from_arc(q_weight)?)),
            QuantMethodConfig::Gptq {
                bits: _,
                use_exllama: _,
                q_weight: _,
                gptq_qzeros: _,
                gptq_scales: _,
                g_idx: _,
                bias: _,
            } => unreachable!(),
        }
    }

    fn matmul(&self, a: &Tensor) -> Result<Tensor> {
        self.0.forward(a)
    }

    fn matmul_via_half(&self, a: &Tensor) -> Result<Tensor> {
        self.0.forward_via_f16(a)
    }
}
