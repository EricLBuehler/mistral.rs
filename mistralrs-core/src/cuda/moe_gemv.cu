/**
 * @brief Optimized CUDA kernel for MoE GEMV (General Matrix-Vector Multiplication)
 * for the decode phase.
 *
 * This kernel is optimized for small batch sizes (M <= 8, typically M = 1 for decode).
 * Based on llama.cpp's approach, it uses warp-level reductions instead of tensor cores,
 * which provides better performance for small batches due to lower overhead.
 *
 * @details
 * - Each CUDA block computes ONE output element for ONE token
 * - Grid configuration: (N, M) where N = output dimension, M = num_tokens
 * - Uses warp-level reductions via __shfl_xor_sync
 * - Minimal shared memory usage (32 bytes for 8 warps)
 * - Vectorized loads using half2/bfloat162 for memory bandwidth
 */

#include "moe_utils.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace vllm_rs {

// Warp reduction sum using shuffle instructions
template <int WARP_SIZE = 32>
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE);
  }
  return x;
}

} // namespace vllm_rs

/**
 * @brief MoE GEMV kernel for standard weight layout [E, N, K].
 *
 * @tparam T Data type: half or nv_bfloat16
 * @tparam BLOCK_SIZE Number of threads per block (default 256 = 8 warps)
 *
 * @param input             [M, K] - Input activations for all tokens
 * @param weights           [num_experts, N, K] - Expert weight matrices
 * @param sorted_token_ids  [M] - Indices of tokens sorted by expert assignment
 * @param expert_ids        [M] - Expert ID for each token
 * @param topk_weights      [M] (optional) - Per-token gating weights (nullptr if
 * not used)
 * @param output            [M, N] - Output activations for all tokens
 * @param num_experts       Total number of experts
 * @param topk              Number of experts selected per token
 * @param M                 Number of tokens (work items)
 * @param N                 Output dimension per expert
 * @param K                 Input dimension per expert
 */
template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_gemv_kernel(
    const T *__restrict__ input,                  // [M, K]
    const T *__restrict__ weights,                // [num_experts, N, K]
    const int32_t *__restrict__ sorted_token_ids, // [M]
    const int32_t *__restrict__ expert_ids,       // [M]
    const float *__restrict__ topk_weights, // [M] optional, can be nullptr
    T *__restrict__ output,                 // [M, N]
    const int num_experts, const int topk, const int M, const int N,
    const int K) {
  // blockIdx.x = output row (N dimension)
  // blockIdx.y = token index
  const int row = blockIdx.x;
  const int token_idx = blockIdx.y;

  if (token_idx >= M || row >= N)
    return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts)
    return;

  // Get input and weight pointers
  // If topk_weights is provided, tokens are NOT replicated (one entry per
  // token) If topk_weights is nullptr, tokens are replicated topk times
  const int input_idx = token_id / (topk_weights ? 1 : topk);
  const T *input_row = input + (size_t)input_idx * K;
  const T *weight_row = weights + (size_t)expert * N * K + (size_t)row * K;

  // Vectorized accumulation using half2/bfloat162
  using VecT =
      typename std::conditional<std::is_same<T, half>::value, half2,
                                nv_bfloat162>::type;
  constexpr int VEC_SIZE = 2;
  const int k_vec = K / VEC_SIZE;

  float sum = 0.0f;
  const int tid = threadIdx.x;

  const VecT *in_vec = reinterpret_cast<const VecT *>(input_row);
  const VecT *w_vec = reinterpret_cast<const VecT *>(weight_row);

  // Each thread processes K/BLOCK_SIZE elements
  for (int k = tid; k < k_vec; k += BLOCK_SIZE) {
    VecT in_val = in_vec[k];
    VecT w_val = w_vec[k];
    sum += vllm::to_float(in_val.x) * vllm::to_float(w_val.x);
    sum += vllm::to_float(in_val.y) * vllm::to_float(w_val.y);
  }

  // Handle remainder if K is not divisible by VEC_SIZE
  // (This is rare for typical hidden dimensions which are powers of 2)
  const int remainder_start = k_vec * VEC_SIZE;
  for (int k = remainder_start + tid; k < K; k += BLOCK_SIZE) {
    sum += vllm::to_float(input_row[k]) * vllm::to_float(weight_row[k]);
  }

  // Warp-level reduction
  sum = vllm_rs::warp_reduce_sum(sum);

  // Inter-warp reduction using shared memory
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float smem[NUM_WARPS];
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  if (lane_id == 0) {
    smem[warp_id] = sum;
  }
  __syncthreads();

  // Final reduction in the first warp
  if (warp_id == 0) {
    sum = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;

// Reduce across the first warp
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    // Thread 0 writes the final result
    if (lane_id == 0) {
      if (topk_weights) {
        sum *= topk_weights[token_id];
      }
      T out_val;
      vllm::from_float(out_val, sum);
      output[(size_t)token_id * N + row] = out_val;
    }
  }
}

/**
 * @brief MoE GEMV kernel for transposed weight layout [E, K, N].
 *
 * Same algorithm as moe_gemv_kernel but with different weight access pattern.
 *
 * @param weights [num_experts, K, N] - Expert weight matrices (transposed)
 */
template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_gemv_transposed_kernel(
    const T *__restrict__ input,   // [M, K]
    const T *__restrict__ weights, // [num_experts, K, N] - transposed layout
    const int32_t *__restrict__ sorted_token_ids, // [M]
    const int32_t *__restrict__ expert_ids,       // [M]
    const float *__restrict__ topk_weights, // [M] optional, can be nullptr
    T *__restrict__ output,                 // [M, N]
    const int num_experts, const int topk, const int M, const int N,
    const int K) {
  const int row = blockIdx.x;       // Output N dimension
  const int token_idx = blockIdx.y; // Token index

  if (token_idx >= M || row >= N)
    return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts)
    return;

  const int input_idx = token_id / (topk_weights ? 1 : topk);
  const T *input_row = input + (size_t)input_idx * K;
  // For transposed layout [E, K, N]: weight[k, n] = weights[expert * K * N + k
  // * N + n]
  const T *weight_expert = weights + (size_t)expert * K * N;

  float sum = 0.0f;
  const int tid = threadIdx.x;

  // For transposed layout, weights are accessed with stride N
  // This is less efficient for memory coalescing, but still faster than
  // moe_gemm for small M
  for (int k = tid; k < K; k += BLOCK_SIZE) {
    // weight[k, row] = weight_expert[k * N + row]
    sum += vllm::to_float(input_row[k]) *
           vllm::to_float(weight_expert[(size_t)k * N + row]);
  }

  // Warp-level reduction
  sum = vllm_rs::warp_reduce_sum(sum);

  // Inter-warp reduction
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float smem[NUM_WARPS];
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  if (lane_id == 0) {
    smem[warp_id] = sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    sum = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;

#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
      if (topk_weights) {
        sum *= topk_weights[token_id];
      }
      T out_val;
      vllm::from_float(out_val, sum);
      output[(size_t)token_id * N + row] = out_val;
    }
  }
}

extern "C" void
moe_gemv(const void *input,   // input [size_m or size_m / topk, size_k]
         const void *weights, // weights [num_experts, size_n, size_k]
         const int32_t *sorted_token_ids, const int32_t *expert_ids,
         const float *topk_weights, // device ptr or nullptr
         void *output,              // output [size_m, size_n]
         int num_experts, int topk, int size_m, int size_n, int size_k,
         int dtype, // 0=float16, 1=bf16
         cudaStream_t stream) {

  constexpr int BLOCK_SIZE = 256;

  // Grid: (N, M) - one block per output element per token
  dim3 grid(size_n, size_m);
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) { // FP16
    moe_gemv_kernel<half, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids, expert_ids,
        topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
        size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) { // BF16
    moe_gemv_kernel<nv_bfloat16, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16 *>(input),
        reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
        expert_ids, topk_weights, reinterpret_cast<nv_bfloat16 *>(output),
        num_experts, topk, size_m, size_n, size_k);
  }
#endif
  else {
    fprintf(stderr, "moe_gemv: unsupported dtype.\n");
  }
}

extern "C" void moe_gemv_transposed(
    const void *input, // input [size_m or size_m / topk, size_k]
    const void
        *weights, // weights [num_experts, size_k, size_n] - transposed layout
    const int32_t *sorted_token_ids, const int32_t *expert_ids,
    const float *topk_weights, // device ptr or nullptr
    void *output,              // output [size_m, size_n]
    int num_experts, int topk, int size_m, int size_n, int size_k,
    int dtype, // 0=float16, 1=bf16
    cudaStream_t stream) {

  constexpr int BLOCK_SIZE = 256;

  // Grid: (N, M) - one block per output element per token
  dim3 grid(size_n, size_m);
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) { // FP16
    moe_gemv_transposed_kernel<half, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids, expert_ids,
        topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
        size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) { // BF16
    moe_gemv_transposed_kernel<nv_bfloat16, BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const nv_bfloat16 *>(input),
            reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
            expert_ids, topk_weights, reinterpret_cast<nv_bfloat16 *>(output),
            num_experts, topk, size_m, size_n, size_k);
  }
#endif
  else {
    fprintf(stderr, "moe_gemv_transposed: unsupported dtype.\n");
  }
}
