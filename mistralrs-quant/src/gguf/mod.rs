use std::sync::Arc;

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

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        match self {
            Self(QMatMul::Tensor(w)) => Ok(Arc::new(Self(QMatMul::Tensor((w + delta)?)))),
            Self(QMatMul::TensorF16(w)) => Ok(Arc::new(Self(QMatMul::TensorF16((w + delta)?)))),
            Self(QMatMul::QTensor(w)) => {
                let (w, dtype) = (w.dequantize(&w.device())?, w.dtype());
                Ok(Arc::new(Self(QMatMul::QTensor(std::sync::Arc::new(
                    candle_core::quantized::QTensor::quantize(&(w + delta)?, dtype)?,
                )))))
            }
        }
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        match &self.0 {
            QMatMul::QTensor(q) => (DType::F32, q.device()),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => (t.dtype(), t.device().clone()),
        }
    }
}
