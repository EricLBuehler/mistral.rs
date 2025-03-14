use float8::F8E4M3;
use half::{bf16, f16};

pub(crate) const HAVE_BLOCKWISE_DEQUANT_KERNELS: bool = true;

extern "C" {
    pub(crate) fn launch_dequant_fp8_blockwise_kernel_f32(
        d_weight: *const F8E4M3,
        d_scale: *const f32,
        d_output: *mut f32,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_dequant_fp8_blockwise_kernel_f16(
        d_weight: *const F8E4M3,
        d_scale: *const f32,
        d_output: *mut f16,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_dequant_fp8_blockwise_kernel_bf16(
        d_weight: *const F8E4M3,
        d_scale: *const f32,
        d_output: *mut bf16,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
