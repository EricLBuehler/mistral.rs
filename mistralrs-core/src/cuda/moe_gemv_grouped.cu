/**
 * @brief MOE GEMV kernel optimized for decode phase (small M).
 *
 * Grid: (N / N_PER_BLOCK) - reduced block count
 * Each block processes N_PER_BLOCK output columns for ALL M tokens.
 */

#include "moe_utils.h"
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CEILDIV(x, y) (((x) + (y)-1) / (y))

/**
 * @brief MOE GEMV kernel for standard weight layout [E, N, K].
 *
 * Each block handles N_PER_BLOCK output columns and processes all M tokens.
 * Uses warp-level reduction for efficiency.
 */
template <typename T, int BLOCK_SIZE = 128, int N_PER_BLOCK = 4>
__global__ void moe_gemv_grouped_kernel(
    const T *__restrict__ input,                  // [num_tokens, K]
    const T *__restrict__ weights,                // [num_experts, N, K]
    const int32_t *__restrict__ sorted_token_ids, // [M]
    const int32_t *__restrict__ expert_ids,       // [M]
    const float *__restrict__ topk_weights,       // [M] or nullptr
    T *__restrict__ output,                       // [M, N]
    const int num_experts, const int topk, const int size_m, const int size_n,
    const int size_k) {

  const int n_base = blockIdx.x * N_PER_BLOCK;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  // Each warp handles one output column within N_PER_BLOCK
  const int n_idx = n_base + warp_id;
  if (n_idx >= size_n)
    return;

  // Process each token
  for (int m = 0; m < size_m; m++) {
    const int token_id = sorted_token_ids[m];
    const int expert_id = expert_ids[m];

    // Input row for this token
    const int input_idx = topk_weights ? token_id : (token_id / topk);
    const T *input_row = input + (size_t)input_idx * size_k;

    // Weight row: weights[expert_id, n_idx, :]
    const T *weight_row = weights + (size_t)expert_id * size_n * size_k +
                          (size_t)n_idx * size_k;

    // Each lane computes partial dot product
    float acc = 0.0f;
    for (int k = lane_id; k < size_k; k += 32) {
      float w = vllm::to_float(__ldg(&weight_row[k]));
      float x = vllm::to_float(__ldg(&input_row[k]));
      acc = __fmaf_rn(w, x, acc);
    }

    // Warp reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    // Lane 0 writes result
    if (lane_id == 0) {
      float result = acc;
      if (topk_weights) {
        result *= topk_weights[token_id];
      }
      T out_val;
      vllm::from_float(out_val, result);
      output[(size_t)token_id * size_n + n_idx] = out_val;
    }
  }
}

/**
 * @brief MOE GEMV kernel for transposed weight layout [E, K, N].
 */
template <typename T, int BLOCK_SIZE = 128, int N_PER_BLOCK = 4>
__global__ void moe_gemv_grouped_transposed_kernel(
    const T *__restrict__ input,                  // [num_tokens, K]
    const T *__restrict__ weights,                // [num_experts, K, N]
    const int32_t *__restrict__ sorted_token_ids, // [M]
    const int32_t *__restrict__ expert_ids,       // [M]
    const float *__restrict__ topk_weights,       // [M] or nullptr
    T *__restrict__ output,                       // [M, N]
    const int num_experts, const int topk, const int size_m, const int size_n,
    const int size_k) {

  const int n_base = blockIdx.x * N_PER_BLOCK;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  const int n_idx = n_base + warp_id;
  if (n_idx >= size_n)
    return;

  for (int m = 0; m < size_m; m++) {
    const int token_id = sorted_token_ids[m];
    const int expert_id = expert_ids[m];

    const int input_idx = topk_weights ? token_id : (token_id / topk);
    const T *input_row = input + (size_t)input_idx * size_k;

    // Transposed: weights[expert_id, :, n_idx]
    const T *expert_weights =
        weights + (size_t)expert_id * size_k * size_n + n_idx;

    float acc = 0.0f;
    for (int k = lane_id; k < size_k; k += 32) {
      float w = vllm::to_float(__ldg(&expert_weights[k * size_n]));
      float x = vllm::to_float(__ldg(&input_row[k]));
      acc = __fmaf_rn(w, x, acc);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
      float result = acc;
      if (topk_weights) {
        result *= topk_weights[token_id];
      }
      T out_val;
      vllm::from_float(out_val, result);
      output[(size_t)token_id * size_n + n_idx] = out_val;
    }
  }
}

// ============================================================================
// Launch Functions
// ============================================================================

extern "C" void moe_gemv_grouped(const void *input, const void *weights,
                                 const int32_t *sorted_token_ids,
                                 const int32_t *expert_ids,
                                 const float *topk_weights, void *output,
                                 int num_experts, int topk, int size_m,
                                 int size_n, int size_k, int dtype,
                                 cudaStream_t stream) {
  // 4 warps per block, each handles one N column
  constexpr int N_PER_BLOCK = 4;
  constexpr int BLOCK_SIZE = N_PER_BLOCK * 32; // 128 threads

  dim3 grid(CEILDIV(size_n, N_PER_BLOCK));
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) { // FP16
    moe_gemv_grouped_kernel<half, BLOCK_SIZE, N_PER_BLOCK>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const half *>(input),
            reinterpret_cast<const half *>(weights), sorted_token_ids,
            expert_ids, topk_weights, reinterpret_cast<half *>(output),
            num_experts, topk, size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) { // BF16
    moe_gemv_grouped_kernel<nv_bfloat16, BLOCK_SIZE, N_PER_BLOCK>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const nv_bfloat16 *>(input),
            reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
            expert_ids, topk_weights, reinterpret_cast<nv_bfloat16 *>(output),
            num_experts, topk, size_m, size_n, size_k);
  }
#endif
}

extern "C" void moe_gemv_grouped_transposed(
    const void *input, const void *weights, const int32_t *sorted_token_ids,
    const int32_t *expert_ids, const float *topk_weights, void *output,
    int num_experts, int topk, int size_m, int size_n, int size_k, int dtype,
    cudaStream_t stream) {
  constexpr int N_PER_BLOCK = 4;
  constexpr int BLOCK_SIZE = N_PER_BLOCK * 32;

  dim3 grid(CEILDIV(size_n, N_PER_BLOCK));
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) { // FP16
    moe_gemv_grouped_transposed_kernel<half, BLOCK_SIZE, N_PER_BLOCK>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const half *>(input),
            reinterpret_cast<const half *>(weights), sorted_token_ids,
            expert_ids, topk_weights, reinterpret_cast<half *>(output),
            num_experts, topk, size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) { // BF16
    moe_gemv_grouped_transposed_kernel<nv_bfloat16, BLOCK_SIZE, N_PER_BLOCK>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const nv_bfloat16 *>(input),
            reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
            expert_ids, topk_weights,
            reinterpret_cast<nv_bfloat16 *>(output), num_experts, topk, size_m,
            size_n, size_k);
  }
#endif
}
