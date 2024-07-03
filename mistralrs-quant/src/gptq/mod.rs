mod ffi;
#[cfg(feature = "cuda")]
mod gptq_cuda;
#[cfg(not(feature = "cuda"))]
mod gptq_cpu;

#[cfg(feature = "cuda")]
pub use gptq_cuda::GptQMatMul;
#[cfg(not(feature = "cuda"))]
pub use gptq_cpu::GptQMatMul;
