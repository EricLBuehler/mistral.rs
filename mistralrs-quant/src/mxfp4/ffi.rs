use half::{bf16, f16};

pub(crate) const HAVE_MXFP4_GEMM_KERNELS: bool = cfg!(has_mxfp4_kernels);
pub(crate) const HAVE_MXFP4_WMMA_KERNELS: bool = cfg!(has_mxfp4_wmma_kernels);

extern "C" {
    pub(crate) fn launch_mxfp4_matmul_f16(
        input: *const f16,
        weight: *const u8,
        weight_scale: *const u8,
        bias: *const f16,
        output: *mut f16,
        m: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_mxfp4_matmul_bf16(
        input: *const bf16,
        weight: *const u8,
        weight_scale: *const u8,
        bias: *const bf16,
        output: *mut bf16,
        m: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_mxfp4_matmul_wmma_f16(
        input: *const f16,
        weight: *const u8,
        weight_scale: *const u8,
        bias: *const f16,
        output: *mut f16,
        m: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_mxfp4_matmul_wmma_bf16(
        input: *const bf16,
        weight: *const u8,
        weight_scale: *const u8,
        bias: *const bf16,
        output: *mut bf16,
        m: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_mxfp4_indexed_moe_gemm_f16(
        input: *const f16,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const f16,
        indices: *const u32,
        output: *mut f16,
        num_tokens: i32,
        topk: i32,
        num_experts: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_mxfp4_indexed_moe_gemm_bf16(
        input: *const bf16,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const bf16,
        indices: *const u32,
        output: *mut bf16,
        num_tokens: i32,
        topk: i32,
        num_experts: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn mxfp4_get_max_smem_optin() -> i32;

    pub(crate) fn launch_mxfp4_moe_grouped_gemm_f16(
        input: *const f16,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const f16,
        indices: *const u32,
        output: *mut f16,
        num_tokens: i32,
        topk: i32,
        num_experts: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_mxfp4_moe_grouped_gemm_bf16(
        input: *const bf16,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const bf16,
        indices: *const u32,
        output: *mut bf16,
        num_tokens: i32,
        topk: i32,
        num_experts: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_mxfp4_moe_grouped_gemm_wmma_f16(
        input: *const f16,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const f16,
        indices: *const u32,
        output: *mut f16,
        num_tokens: i32,
        topk: i32,
        num_experts: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_mxfp4_moe_grouped_gemm_wmma_bf16(
        input: *const bf16,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const bf16,
        indices: *const u32,
        output: *mut bf16,
        num_tokens: i32,
        topk: i32,
        num_experts: i32,
        n: i32,
        k: i32,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
