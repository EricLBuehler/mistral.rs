use float8::F8E4M3;
use half::{bf16, f16};

pub(crate) const HAVE_FP8_QUANT_KERNELS: bool = true;

extern "C" {
    pub fn quantize_scalar_fp8_f32(
        d_in: *const f32,
        d_out: *mut F8E4M3,
        s_out: *mut f32,
        elem_count: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub fn quantize_scalar_fp8_bf16(
        d_in: *const bf16,
        d_out: *mut F8E4M3,
        s_out: *mut f32,
        elem_count: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub fn quantize_scalar_fp8_f16(
        d_in: *const f16,
        d_out: *mut F8E4M3,
        s_out: *mut f32,
        elem_count: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
