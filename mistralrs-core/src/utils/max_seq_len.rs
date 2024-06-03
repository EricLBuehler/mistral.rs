use candle_core::{
    quantized::gguf_file::{Value, ValueType},
    Result,
};
use tracing::warn;

/// Extract a u32 or u8 max seq len. Warns if error and then uses a default
pub(crate) fn get_gguf_max_seq_len(max_seq_len: Result<&Value>, default: u64) -> u64 {
    match max_seq_len {
        Ok(m) => match m.value_type() {
            ValueType::U32 => m.to_u32().unwrap() as u64,
            ValueType::U64 => m.to_u64().unwrap(),
            _ => default,
        },
        Err(_) => {
            warn!("GGUF file does not specify a context window, using {default}.");
            default
        }
    }
}
