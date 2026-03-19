use float8::F8E4M3;
use half::{bf16, f16};

pub(crate) const HAVE_VECTOR_DEQUANT_KERNELS: bool = cfg!(has_vector_fp8_kernels);
pub(crate) const HAVE_VECTOR_QUANT_KERNELS: bool = cfg!(has_vector_fp8_kernels);

extern "C" {
    pub(crate) fn launch_dequant_fp8_vector_kernel_f32(
        d_weight: *const F8E4M3,
        d_scale: *const f32,
        d_output: *mut f32,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_dequant_fp8_vector_kernel_f16(
        d_weight: *const F8E4M3,
        d_scale: *const f32,
        d_output: *mut f16,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_dequant_fp8_vector_kernel_bf16(
        d_weight: *const F8E4M3,
        d_scale: *const f32,
        d_output: *mut bf16,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_quant_fp8_vector_kernel_f32(
        d_input: *const f32,
        d_weight: *mut F8E4M3,
        d_scale: *mut f32,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_quant_fp8_vector_kernel_f16(
        d_input: *const f16,
        d_weight: *mut F8E4M3,
        d_scale: *mut f32,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_quant_fp8_vector_kernel_bf16(
        d_input: *const bf16,
        d_weight: *mut F8E4M3,
        d_scale: *mut f32,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
