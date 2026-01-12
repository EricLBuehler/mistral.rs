/**
 * @brief MOE GEMV kernel optimized for decode phase (small M).
 *
 * Grid: (N) - one block per output column
 * Each block processes ALL M tokens for its output column.
 *
 * This is simpler than the expert-grouped approach and has better
 * block utilization since every block does useful work.
 *
 * For tokens sharing the same expert, weight data will be in L1/L2 cache.
 */

#include "moe_utils.h"
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/**
 * @brief MOE GEMV kernel for standard weight layout [E, N, K].
 *
 * Each block handles one output column (n_idx) and processes all M tokens.
 * Threads cooperate on K-dimension reduction.
 *
 * @param input             [num_tokens, K] - Input activations
 * @param weights           [num_experts, N, K] - Expert weight matrices
 * @param sorted_token_ids  [M] - Token indices (sorted by expert)
 * @param expert_ids        [M] - Expert assignment per token
 * @param topk_weights      [M] (optional) - Per-token gating weights
 * @param output            [M, N] - Output activations
 */
template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_gemv_grouped_kernel(
    const T *__restrict__ input,                  // [num_tokens, K]
    const T *__restrict__ weights,                // [num_experts, N, K]
    const int32_t *__restrict__ sorted_token_ids, // [M]
    const int32_t *__restrict__ expert_ids,       // [M]
    const float *__restrict__ topk_weights,       // [M] or nullptr
    T *__restrict__ output,                       // [M, N]
    const int num_experts, const int topk, const int size_m, const int size_n,
    const int size_k) {

  const int n_idx = blockIdx.x; // Output column this block handles
  if (n_idx >= size_n)
    return;

  const int tid = threadIdx.x;

  __shared__ float s_partial[BLOCK_SIZE];

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

    // Each thread computes partial dot product over K
    float acc = 0.0f;
    for (int k = tid; k < size_k; k += BLOCK_SIZE) {
      float w = vllm::to_float(__ldg(&weight_row[k]));
      float x = vllm::to_float(__ldg(&input_row[k]));
      acc = __fmaf_rn(w, x, acc);
    }

    // Store to shared memory
    s_partial[tid] = acc;
    __syncthreads();

    // Tree reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        s_partial[tid] += s_partial[tid + stride];
      }
      __syncthreads();
    }

    // Thread 0 writes result
    if (tid == 0) {
      float result = s_partial[0];
      if (topk_weights) {
        result *= topk_weights[token_id];
      }
      T out_val;
      vllm::from_float(out_val, result);
      output[(size_t)token_id * size_n + n_idx] = out_val;
    }
    __syncthreads();
  }
}

/**
 * @brief MOE GEMV kernel for transposed weight layout [E, K, N].
 */
template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_gemv_grouped_transposed_kernel(
    const T *__restrict__ input,                  // [num_tokens, K]
    const T *__restrict__ weights,                // [num_experts, K, N]
    const int32_t *__restrict__ sorted_token_ids, // [M]
    const int32_t *__restrict__ expert_ids,       // [M]
    const float *__restrict__ topk_weights,       // [M] or nullptr
    T *__restrict__ output,                       // [M, N]
    const int num_experts, const int topk, const int size_m, const int size_n,
    const int size_k) {

  const int n_idx = blockIdx.x;
  if (n_idx >= size_n)
    return;

  const int tid = threadIdx.x;

  __shared__ float s_partial[BLOCK_SIZE];

  for (int m = 0; m < size_m; m++) {
    const int token_id = sorted_token_ids[m];
    const int expert_id = expert_ids[m];

    const int input_idx = topk_weights ? token_id : (token_id / topk);
    const T *input_row = input + (size_t)input_idx * size_k;

    // Transposed: weights[expert_id, :, n_idx]
    const T *expert_weights =
        weights + (size_t)expert_id * size_k * size_n + n_idx;

    float acc = 0.0f;
    for (int k = tid; k < size_k; k += BLOCK_SIZE) {
      float w = vllm::to_float(__ldg(&expert_weights[k * size_n]));
      float x = vllm::to_float(__ldg(&input_row[k]));
      acc = __fmaf_rn(w, x, acc);
    }

    s_partial[tid] = acc;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        s_partial[tid] += s_partial[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      float result = s_partial[0];
      if (topk_weights) {
        result *= topk_weights[token_id];
      }
      T out_val;
      vllm::from_float(out_val, result);
      output[(size_t)token_id * size_n + n_idx] = out_val;
    }
    __syncthreads();
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
  constexpr int BLOCK_SIZE = 256;

  // Grid: one block per output column
  dim3 grid(size_n);
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) { // FP16
    moe_gemv_grouped_kernel<half, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids, expert_ids,
        topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
        size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) { // BF16
    moe_gemv_grouped_kernel<nv_bfloat16, BLOCK_SIZE><<<grid, block, 0, stream>>>(
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
  constexpr int BLOCK_SIZE = 256;

  dim3 grid(size_n);
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) { // FP16
    moe_gemv_grouped_transposed_kernel<half, BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const half *>(input),
            reinterpret_cast<const half *>(weights), sorted_token_ids,
            expert_ids, topk_weights, reinterpret_cast<half *>(output),
            num_experts, topk, size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) { // BF16
    moe_gemv_grouped_transposed_kernel<nv_bfloat16, BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const nv_bfloat16 *>(input),
            reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
            expert_ids, topk_weights,
            reinterpret_cast<nv_bfloat16 *>(output), num_experts, topk, size_m,
            size_n, size_k);
  }
#endif
}
