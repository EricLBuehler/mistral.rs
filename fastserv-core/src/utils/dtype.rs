use anyhow::Result;
use candle_core::DType;
use thiserror::Error;

#[derive(Error, Debug)]
enum DTypeConversionError {
    #[error("`{0}` is not a supported Candle data type.")]
    Unsupported(String),
}

pub(crate) fn get_dtype_from_torch_dtype(torch_dtype: String) -> Result<DType> {
    Ok(match torch_dtype.as_str() {
        "float32" | "float" => DType::F32,
        "float64" | "double" => DType::F64,
        "float16" | "half" => DType::F16,
        "bfloat16" => DType::BF16,
        "uint8" => DType::U8,
        "int64" | "long" => DType::I64,
        other => Result::Err(DTypeConversionError::Unsupported(other.to_string()))?,
    })
}
