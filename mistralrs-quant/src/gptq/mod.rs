#[cfg(feature = "cuda")]
mod ffi;
#[cfg(not(feature = "cuda"))]
mod gptq_cpu;
#[cfg(feature = "cuda")]
mod gptq_cuda;

#[cfg(not(feature = "cuda"))]
pub use gptq_cpu::GptqLayer;
#[cfg(feature = "cuda")]
pub use gptq_cuda::GptqLayer;
