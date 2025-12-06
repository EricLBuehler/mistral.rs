use float8::F8E4M3;
use half::{bf16, f16};

pub(crate) const HAVE_BLOCKWISE_DEQUANT_KERNELS: bool = true;
pub(crate) const HAVE_BLOCKWISE_QUANT_KERNELS: bool = true;

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

    // GEMM kernels for blockwise FP8
    pub(crate) fn launch_blockwise_fp8_gemm_f16(
        input: *const f16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut f16,
        m: i32,
        n: i32,
        k: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        scale_stride: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_blockwise_fp8_gemm_bf16(
        input: *const bf16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut bf16,
        m: i32,
        n: i32,
        k: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        scale_stride: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_blockwise_fp8_gemm_f32(
        input: *const f32,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        scale_stride: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // MoE GEMM kernels (reserved for future use)
    #[allow(dead_code)]
    pub(crate) fn launch_blockwise_fp8_moe_gemm_f16(
        input: *const f16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut f16,
        num_experts: i32,
        topk: i32,
        m: i32,
        n: i32,
        k: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        scale_n_blocks: i32,
        scale_k_blocks: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    #[allow(dead_code)]
    pub(crate) fn launch_blockwise_fp8_moe_gemm_bf16(
        input: *const bf16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut bf16,
        num_experts: i32,
        topk: i32,
        m: i32,
        n: i32,
        k: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        scale_n_blocks: i32,
        scale_k_blocks: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    #[allow(dead_code)]
    pub(crate) fn launch_blockwise_fp8_moe_gemm_transposed_f16(
        input: *const f16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut f16,
        num_experts: i32,
        topk: i32,
        m: i32,
        n: i32,
        k: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        scale_k_blocks: i32,
        scale_n_blocks: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    #[allow(dead_code)]
    pub(crate) fn launch_blockwise_fp8_moe_gemm_transposed_bf16(
        input: *const bf16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut bf16,
        num_experts: i32,
        topk: i32,
        m: i32,
        n: i32,
        k: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        scale_k_blocks: i32,
        scale_n_blocks: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // WMMA MoE GEMM kernels (reserved for future use)
    #[allow(dead_code)]
    pub(crate) fn launch_blockwise_fp8_moe_gemm_wmma_f16(
        input: *const f16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut f16,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        scale_n_blocks: i32,
        scale_k_blocks: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    #[allow(dead_code)]
    pub(crate) fn launch_blockwise_fp8_moe_gemm_wmma_bf16(
        input: *const bf16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut bf16,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        scale_n_blocks: i32,
        scale_k_blocks: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    #[allow(dead_code)]
    pub(crate) fn launch_blockwise_fp8_moe_gemm_wmma_transposed_f16(
        input: *const f16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut f16,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        scale_k_blocks: i32,
        scale_n_blocks: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    #[allow(dead_code)]
    pub(crate) fn launch_blockwise_fp8_moe_gemm_wmma_transposed_bf16(
        input: *const bf16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut bf16,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        scale_k_blocks: i32,
        scale_n_blocks: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
