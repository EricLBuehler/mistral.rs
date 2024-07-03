use candle_core::{Result, Tensor};

pub mod gptq;

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
}

pub trait QuantMethod {
    fn new(method: QuantMethodConfig) -> Self
    where
        Self: Sized;

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    fn matmul(&mut self, a: &Tensor) -> Result<Tensor>;
}
