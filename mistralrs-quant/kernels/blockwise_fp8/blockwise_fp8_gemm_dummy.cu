/**
 * @brief Dummy FP8 GEMM kernels for GPUs that don't support FP8 (CC < 8.0).
 *
 * These are stub implementations that will never be called but are needed
 * for linking when FP8 is not supported.
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Dummy FP8 type for compilation
struct __nv_fp8_e4m3_dummy {
  unsigned char __x;
};

extern "C" void launch_fp8_matmul_f16(const __half *input,
                                      const void *weight, // __nv_fp8_e4m3*
                                      const float *weight_scale, __half *output,
                                      int M, int N, int K, int scale_row_stride,
                                      int block_size_y, int block_size_x,
                                      cudaStream_t stream) {
  fprintf(stderr, "FP8 matmul not supported on this GPU (requires compute "
                  "capability >= 8.0)\n");
}

extern "C" void launch_fp8_matmul_bf16(const __nv_bfloat16 *input,
                                       const void *weight, // __nv_fp8_e4m3*
                                       const float *weight_scale,
                                       __nv_bfloat16 *output, int M, int N,
                                       int K, int scale_row_stride,
                                       int block_size_y, int block_size_x,
                                       cudaStream_t stream) {
  fprintf(stderr, "FP8 matmul not supported on this GPU (requires compute "
                  "capability >= 8.0)\n");
}

extern "C" void launch_fp8_indexed_moe_gemm_f16(
    const __half *input,
    const void *weights, // __nv_fp8_e4m3*
    const float *weight_scales, const int32_t *indices, __half *output,
    int num_tokens, int topk, int num_experts, int N, int K,
    int scale_row_stride, int block_size_y, int block_size_x,
    bool input_has_topk_dim, cudaStream_t stream) {
  fprintf(stderr, "FP8 indexed MoE GEMM not supported on this GPU (requires "
                  "compute capability >= 8.0)\n");
}

extern "C" void launch_fp8_indexed_moe_gemm_bf16(
    const __nv_bfloat16 *input,
    const void *weights, // __nv_fp8_e4m3*
    const float *weight_scales, const int32_t *indices, __nv_bfloat16 *output,
    int num_tokens, int topk, int num_experts, int N, int K,
    int scale_row_stride, int block_size_y, int block_size_x,
    bool input_has_topk_dim, cudaStream_t stream) {
  fprintf(stderr, "FP8 indexed MoE GEMM not supported on this GPU (requires "
                  "compute capability >= 8.0)\n");
}
