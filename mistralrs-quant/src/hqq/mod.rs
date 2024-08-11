#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
mod hqq_cuda;

mod hqq_cpu;
