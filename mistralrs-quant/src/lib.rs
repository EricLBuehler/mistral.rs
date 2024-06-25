use serde::Deserialize;

pub mod gptq;

#[derive(Debug, Clone, Deserialize)]
pub enum QuantMethod {
    GptQ,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizedConfig {
    bits: i32,
    group_size: i32,
    quant_method: QuantMethod,
}
