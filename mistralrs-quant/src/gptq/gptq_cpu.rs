use crate::{QuantMethod, QuantMethodConfig};
use candle_core::{quantized::QMatMul, DType, Result, Tensor};
use std::sync::Arc;

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
            } => candle_core::bail!("GPTQ is only supported on CUDA."),
            QuantMethodConfig::Gguf { q_weight: _, b: _ } | QuantMethodConfig::Unquantized(_) => {
                unreachable!()
            }
        }
    }

    fn forward(&self, _a: &Tensor) -> Result<Tensor> {
        todo!()
    }

    fn quantized_act_type(&self) -> Option<DType> {
        todo!()
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        todo!()
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        todo!()
    }

    fn get_qmatmul(&mut self) -> Option<&mut QMatMul> {
        todo!()
    }

    fn get_bias_mut(&mut self) -> Option<&mut Tensor> {
        todo!()
    }

    fn convert_to_isq(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>> {
        todo!()
    }
}
