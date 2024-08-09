use candle_core::{quantized::QMatMul, DType, Result, Tensor};
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
            }
            | QuantMethodConfig::Unquantized(_) => unreachable!(),
        }
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        self.0.forward(a)
    }

    fn forward_via_half(&self, a: &Tensor) -> Result<Tensor> {
        self.0.forward_via_f16(a)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        Some(DType::F32)
    }
}
