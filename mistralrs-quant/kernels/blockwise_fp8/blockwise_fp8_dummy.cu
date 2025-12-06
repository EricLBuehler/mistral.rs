#include <cassert>
#include <cstdint>
#include <cuda.h>

extern "C" void launch_dequant_fp8_blockwise_kernel_f32(
    const uint8_t *d_weight, const float *d_scale, float *d_output,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x,
    cudaStream_t stream) {
  assert(false);
}

extern "C" void launch_dequant_fp8_blockwise_kernel_f16(
    const uint8_t *d_weight, const float *d_scale, uint16_t *d_output,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x,
    cudaStream_t stream) {
  assert(false);
}

extern "C" void launch_dequant_fp8_blockwise_kernel_bf16(
    const uint8_t *d_weight, const float *d_scale, uint16_t *d_output,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x,
    cudaStream_t stream) {
  assert(false);
}

extern "C" void launch_quant_fp8_blockwise_kernel_f32(
    const float *d_input, uint8_t *d_weight, float *d_scale, int weight_height,
    int weight_width, int weight_row_stride, int scale_stride,
    int weight_block_size_y, int weight_block_size_x, cudaStream_t stream) {
  assert(false);
}

extern "C" void launch_quant_fp8_blockwise_kernel_f16(
    const uint16_t *d_input, uint8_t *d_weight, float *d_scale,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x,
    cudaStream_t stream) {
  assert(false);
}

extern "C" void launch_quant_fp8_blockwise_kernel_bf16(
    const uint16_t *d_input, uint8_t *d_weight, float *d_scale,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x,
    cudaStream_t stream) {
  assert(false);
}

// GEMM kernels
extern "C" void launch_blockwise_fp8_gemm_f16(
    const uint16_t* input,
    const uint8_t* weight,
    const float* weight_scale,
    uint16_t* output,
    int M, int N, int K,
    int weight_block_size_y,
    int weight_block_size_x,
    int scale_stride,
    cudaStream_t stream
) {
  assert(false);
}

extern "C" void launch_blockwise_fp8_gemm_bf16(
    const uint16_t* input,
    const uint8_t* weight,
    const float* weight_scale,
    uint16_t* output,
    int M, int N, int K,
    int weight_block_size_y,
    int weight_block_size_x,
    int scale_stride,
    cudaStream_t stream
) {
  assert(false);
}

extern "C" void launch_blockwise_fp8_gemm_f32(
    const float* input,
    const uint8_t* weight,
    const float* weight_scale,
    float* output,
    int M, int N, int K,
    int weight_block_size_y,
    int weight_block_size_x,
    int scale_stride,
    cudaStream_t stream
) {
  assert(false);
}

// MoE GEMM kernels
extern "C" void launch_blockwise_fp8_moe_gemm_f16(
    const uint16_t* input,
    const uint8_t* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    uint16_t* output,
    int num_experts, int topk,
    int M, int N, int K,
    int weight_block_size_y, int weight_block_size_x,
    int scale_n_blocks, int scale_k_blocks,
    cudaStream_t stream
) {
  assert(false);
}

extern "C" void launch_blockwise_fp8_moe_gemm_bf16(
    const uint16_t* input,
    const uint8_t* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    uint16_t* output,
    int num_experts, int topk,
    int M, int N, int K,
    int weight_block_size_y, int weight_block_size_x,
    int scale_n_blocks, int scale_k_blocks,
    cudaStream_t stream
) {
  assert(false);
}

extern "C" void launch_blockwise_fp8_moe_gemm_transposed_f16(
    const uint16_t* input,
    const uint8_t* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    uint16_t* output,
    int num_experts, int topk,
    int M, int N, int K,
    int weight_block_size_y, int weight_block_size_x,
    int scale_k_blocks, int scale_n_blocks,
    cudaStream_t stream
) {
  assert(false);
}

extern "C" void launch_blockwise_fp8_moe_gemm_transposed_bf16(
    const uint16_t* input,
    const uint8_t* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    uint16_t* output,
    int num_experts, int topk,
    int M, int N, int K,
    int weight_block_size_y, int weight_block_size_x,
    int scale_k_blocks, int scale_n_blocks,
    cudaStream_t stream
) {
  assert(false);
}

// WMMA MoE GEMM kernels
extern "C" void launch_blockwise_fp8_moe_gemm_wmma_f16(
    const uint16_t* input,
    const uint8_t* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    uint16_t* output,
    int num_experts, int topk,
    int size_m, int size_n, int size_k,
    int weight_block_size_y, int weight_block_size_x,
    int scale_n_blocks, int scale_k_blocks,
    cudaStream_t stream
) {
  assert(false);
}

extern "C" void launch_blockwise_fp8_moe_gemm_wmma_bf16(
    const uint16_t* input,
    const uint8_t* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    uint16_t* output,
    int num_experts, int topk,
    int size_m, int size_n, int size_k,
    int weight_block_size_y, int weight_block_size_x,
    int scale_n_blocks, int scale_k_blocks,
    cudaStream_t stream
) {
  assert(false);
}

extern "C" void launch_blockwise_fp8_moe_gemm_wmma_transposed_f16(
    const uint16_t* input,
    const uint8_t* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    uint16_t* output,
    int num_experts, int topk,
    int size_m, int size_n, int size_k,
    int weight_block_size_y, int weight_block_size_x,
    int scale_k_blocks, int scale_n_blocks,
    cudaStream_t stream
) {
  assert(false);
}

extern "C" void launch_blockwise_fp8_moe_gemm_wmma_transposed_bf16(
    const uint16_t* input,
    const uint8_t* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    uint16_t* output,
    int num_experts, int topk,
    int size_m, int size_n, int size_k,
    int weight_block_size_y, int weight_block_size_x,
    int scale_k_blocks, int scale_n_blocks,
    cudaStream_t stream
) {
  assert(false);
}