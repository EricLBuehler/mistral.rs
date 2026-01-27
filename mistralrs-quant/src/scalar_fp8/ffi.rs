#![allow(dead_code)]

use float8::F8E4M3;
use half::{bf16, f16};

pub(crate) const HAVE_SCALAR_FP8_KERNELS: bool = cfg!(has_scalar_fp8_kernels);

extern "C" {
    pub(crate) fn launch_fp8_to_f32_kernel(
        d_input: *const F8E4M3,
        d_output: *mut f32,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_fp8_to_f16_kernel(
        d_input: *const F8E4M3,
        d_output: *mut f16,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_fp8_to_bf16_kernel(
        d_input: *const F8E4M3,
        d_output: *mut bf16,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_f32_to_fp8_kernel(
        d_input: *const f32,
        d_output: *mut F8E4M3,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_f16_to_fp8_kernel(
        d_input: *const f16,
        d_output: *mut F8E4M3,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_bf16_to_fp8_kernel(
        d_input: *const bf16,
        d_output: *mut F8E4M3,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
