use float8::F8E4M3;
use half::{bf16, f16};

pub(crate) const HAVE_BLOCKWISE_DEQUANT_KERNELS: bool = cfg!(has_blockwise_fp8_kernels);
pub(crate) const HAVE_BLOCKWISE_QUANT_KERNELS: bool = cfg!(has_blockwise_fp8_kernels);
pub(crate) const HAVE_BLOCKWISE_GEMM_KERNELS: bool = cfg!(has_blockwise_fp8_kernels);
pub(crate) const HAVE_CUTLASS_FP8_SM90_KERNELS: bool = cfg!(has_cutlass_fp8_sm90_kernels);

#[cfg(has_cutlass_fp8_sm90_kernels)]
extern "C" {
    pub(crate) fn mistralrs_cutlass_fp8_error_string(status: i32) -> *const core::ffi::c_char;

    pub(crate) fn mistralrs_cutlass_fp8_blockwise_prepare() -> i32;

    pub(crate) fn mistralrs_cutlass_fp8_blockwise_workspace_size(
        m: i32,
        n: i32,
        k: i32,
        output_dtype: i32,
        sm_count: i32,
        bytes: *mut usize,
    ) -> i32;

    pub(crate) fn mistralrs_cutlass_fp8_blockwise_gemm(
        activation: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        activation_scales: *const f32,
        weight_scales: *const f32,
        output: *mut core::ffi::c_void,
        m: i32,
        n: i32,
        k: i32,
        output_dtype: i32,
        workspace: *mut core::ffi::c_void,
        workspace_bytes: usize,
        sm_count: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn mistralrs_fp8_quantize_activation_f16(
        input: *const f16,
        output: *mut F8E4M3,
        scales: *mut f32,
        rows: i32,
        cols: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn mistralrs_fp8_quantize_activation_bf16(
        input: *const bf16,
        output: *mut F8E4M3,
        scales: *mut f32,
        rows: i32,
        cols: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

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

    pub(crate) fn launch_quant_fp8_blockwise_kernel_f32(
        d_input: *const f32,
        d_weight: *mut F8E4M3,
        d_scale: *mut f32,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_quant_fp8_blockwise_kernel_f16(
        d_input: *const f16,
        d_weight: *mut F8E4M3,
        d_scale: *mut f32,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_quant_fp8_blockwise_kernel_bf16(
        d_input: *const bf16,
        d_weight: *mut F8E4M3,
        d_scale: *mut f32,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // FP8 Matmul kernels (for forward method)
    pub(crate) fn launch_fp8_matmul_f16(
        input: *const f16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut f16,
        m: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_fp8_matmul_bf16(
        input: *const bf16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut bf16,
        m: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // FP8 Indexed MoE GEMM kernels (for gather_forward method)
    pub(crate) fn launch_fp8_indexed_moe_gemm_f16(
        input: *const f16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        indices: *const u32,
        output: *mut f16,
        num_tokens: i32,
        topk: i32,
        num_experts: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        input_has_topk_dim: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_fp8_indexed_moe_gemm_bf16(
        input: *const bf16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        indices: *const u32,
        output: *mut bf16,
        num_tokens: i32,
        topk: i32,
        num_experts: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        input_has_topk_dim: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
