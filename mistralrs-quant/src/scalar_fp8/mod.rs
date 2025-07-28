#[cfg(feature = "cuda")]
mod ffi;

mod ops;
pub use ops::{dtype_to_fp8, fp8_to_dtype};