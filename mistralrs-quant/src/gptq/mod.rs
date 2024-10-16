#[cfg(feature = "cuda")]
mod ffi;
#[cfg(not(feature = "cuda"))]
mod gptq_cpu;
#[cfg(feature = "cuda")]
mod gptq_cuda;
#[cfg(feature = "cuda")]
mod marlin_backend;

#[cfg(not(feature = "cuda"))]
pub use gptq_cpu::{gptq_linear, GptqLayer};
#[cfg(feature = "cuda")]
pub use gptq_cuda::{gptq_linear, GptqLayer};
