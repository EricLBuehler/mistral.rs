use std::sync::Arc;

use candle_core::{quantized::QTensor, Result, Tensor};

mod gguf;
mod gptq;

pub use gguf::GgufMatMul;
pub use gptq::GptQMatMul;

#[derive(Debug, Clone)]
pub enum QuantMethodConfig {
    GptQ {
        bits: i32,
        use_exllama: bool,
        q_weight: Tensor,
        gptq_qzeros: Tensor,
        gptq_scales: Tensor,
        g_idx: Tensor,
    },
    Gguf {
        q_weight: Arc<QTensor>,
    },
}

pub trait QuantMethod {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized;

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    fn matmul(&mut self, a: &Tensor) -> Result<Tensor>;

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    /// This may go via half precision if it is supported.
    fn matmul_via_half(&mut self, a: &Tensor) -> Result<Tensor> {
        self.matmul(a)
    }
}
