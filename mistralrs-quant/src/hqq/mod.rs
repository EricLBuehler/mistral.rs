#[cfg(feature = "cuda")]
mod ffi;

#[cfg(not(feature = "cuda"))]
mod hqq_cpu;

#[cfg(feature = "cuda")]
mod hqq;
