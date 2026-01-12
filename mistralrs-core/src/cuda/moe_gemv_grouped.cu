/**
 * @brief Grouped MOE GEMV kernel - groups tokens by expert for weight reuse.
 *
 * Unlike moe_gemv which launches (N, M) blocks where each block handles one
 * output element for one token, this kernel launches (num_experts, N_tiles)
 * blocks where each block handles ALL tokens for one expert and one N-tile.
 *
 * This enables weight reuse: load weight row once, compute dot product for
 * all tokens assigned to that expert.
 *
 * Grid: (num_experts, ceil(N/N_TILE))
 * Block: 256 threads
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

#define CEILDIV(x, y) (((x) + (y)-1) / (y))

namespace moe_gemv_grouped {

// Warp reduction sum using shuffle instructions
template <int WARP_SIZE = 32>
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE);
  }
  return x;
}

} // namespace moe_gemv_grouped

/**
 * @brief Grouped MOE GEMV kernel for standard weight layout [E, N, K].
 *
 * Each block processes ONE expert and ONE N-tile (32 output columns).
 * All tokens assigned to that expert are processed together, enabling
 * weight reuse across tokens.
 *
 * @tparam T Data type: half or nv_bfloat16
 * @tparam BLOCK_SIZE Number of threads per block (256 = 8 warps)
 * @tparam N_TILE Number of output columns per block (32 = warp width)
 * @tparam MAX_TOKENS Maximum tokens per expert we handle in registers
 *
 * @param input             [M/topk, K] - Input activations (original token
 * order)
 * @param weights           [num_experts, N, K] - Expert weight matrices
 * @param sorted_token_ids  [M] - Token indices sorted by expert
 * @param expert_offsets    [num_experts + 1] - Segment boundaries per expert
 * @param topk_weights      [M] (optional) - Per-token gating weights
 * @param output            [M, N] - Output activations
 * @param topk              Number of experts per token
 * @param size_n            Output dimension (N)
 * @param size_k            Input dimension (K)
 */
template <typename T, int BLOCK_SIZE = 256, int N_TILE = 32, int MAX_TOKENS = 8>
__global__ void moe_gemv_grouped_kernel(
    const T *__restrict__ input,                   // [M/topk, K]
    const T *__restrict__ weights,                 // [num_experts, N, K]
    const int32_t *__restrict__ sorted_token_ids,  // [M]
    const int32_t *__restrict__ expert_offsets,    // [num_experts + 1]
    const float *__restrict__ topk_weights,        // [M] or nullptr
    T *__restrict__ output,                        // [M, N]
    const int topk, const int size_n, const int size_k) {
  const int expert_id = blockIdx.x;
  const int n_tile_idx = blockIdx.y;

  // Get expert's token segment
  const int segment_start = expert_offsets[expert_id];
  const int segment_end = expert_offsets[expert_id + 1];
  const int num_tokens = segment_end - segment_start;

  // Early exit for empty segments
  if (num_tokens == 0)
    return;

  // Calculate N-dimension bounds
  const int n_base = n_tile_idx * N_TILE;
  if (n_base >= size_n)
    return;

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  // Each lane in the first warp handles one output column
  // Other warps help with K-dimension reduction
  const int n_local = lane_id;
  const int n_global = n_base + n_local;
  const bool valid_n = (n_global < size_n);

  // Expert weight base pointer
  const T *expert_weights = weights + (size_t)expert_id * size_n * size_k;

  // Shared memory for inter-warp reduction
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float s_warp_sums[NUM_WARPS][N_TILE][MAX_TOKENS];

  // Per-token accumulators (in registers for small token counts)
  float acc[MAX_TOKENS];
#pragma unroll
  for (int t = 0; t < MAX_TOKENS; ++t) {
    acc[t] = 0.0f;
  }

  // Process tokens in batches of MAX_TOKENS
  for (int t_base = 0; t_base < num_tokens; t_base += MAX_TOKENS) {
    const int tokens_this_batch = min(MAX_TOKENS, num_tokens - t_base);

// Reset accumulators for this batch
#pragma unroll
    for (int t = 0; t < MAX_TOKENS; ++t) {
      acc[t] = 0.0f;
    }

    // Main K-loop: each thread processes a strided subset of K
    // Threads cooperate to cover the entire K dimension
    for (int k = tid; k < size_k; k += BLOCK_SIZE) {
      // Load weight element for this thread's K position and N column
      float w_val = 0.0f;
      if (valid_n) {
        // weights[expert, n_global, k]
        w_val = vllm::to_float(
            __ldg(&expert_weights[(size_t)n_global * size_k + k]));
      }

// Compute dot product contribution for each token in this batch
#pragma unroll
      for (int t = 0; t < MAX_TOKENS; ++t) {
        if (t < tokens_this_batch) {
          const int token_pair_idx = segment_start + t_base + t;
          const int token_id = sorted_token_ids[token_pair_idx];
          // Input index: token_id / topk if topk_weights provided, else
          // token_id
          const int input_idx = topk_weights ? token_id : (token_id / topk);

          // Load input element
          float in_val = vllm::to_float(__ldg(&input[(size_t)input_idx * size_k + k]));

          // Accumulate
          acc[t] = __fmaf_rn(w_val, in_val, acc[t]);
        }
      }
    }

    // Warp-level reduction for each token
#pragma unroll
    for (int t = 0; t < MAX_TOKENS; ++t) {
      acc[t] = moe_gemv_grouped::warp_reduce_sum(acc[t]);
    }

    // Inter-warp reduction using shared memory
    if (lane_id == 0 && valid_n) {
#pragma unroll
      for (int t = 0; t < MAX_TOKENS; ++t) {
        s_warp_sums[warp_id][n_local][t] = acc[t];
      }
    }
    __syncthreads();

    // First warp aggregates results from all warps
    if (warp_id == 0 && valid_n) {
#pragma unroll
      for (int t = 0; t < MAX_TOKENS; ++t) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
          sum += s_warp_sums[w][n_local][t];
        }

        // Write output for this token
        if (t < tokens_this_batch) {
          const int token_pair_idx = segment_start + t_base + t;
          const int token_id = sorted_token_ids[token_pair_idx];

          // Apply topk weight if provided
          if (topk_weights) {
            sum *= topk_weights[token_id];
          }

          // Write to output
          T out_val;
          vllm::from_float(out_val, sum);
          output[(size_t)token_id * size_n + n_global] = out_val;
        }
      }
    }
    __syncthreads();
  }
}

/**
 * @brief Grouped MOE GEMV kernel for transposed weight layout [E, K, N].
 *
 * Same algorithm but with different weight access pattern.
 * Weights are accessed with stride N instead of stride K.
 */
template <typename T, int BLOCK_SIZE = 256, int N_TILE = 32, int MAX_TOKENS = 8>
__global__ void moe_gemv_grouped_transposed_kernel(
    const T *__restrict__ input,                   // [M/topk, K]
    const T *__restrict__ weights,                 // [num_experts, K, N]
    const int32_t *__restrict__ sorted_token_ids,  // [M]
    const int32_t *__restrict__ expert_offsets,    // [num_experts + 1]
    const float *__restrict__ topk_weights,        // [M] or nullptr
    T *__restrict__ output,                        // [M, N]
    const int topk, const int size_n, const int size_k) {
  const int expert_id = blockIdx.x;
  const int n_tile_idx = blockIdx.y;

  const int segment_start = expert_offsets[expert_id];
  const int segment_end = expert_offsets[expert_id + 1];
  const int num_tokens = segment_end - segment_start;

  if (num_tokens == 0)
    return;

  const int n_base = n_tile_idx * N_TILE;
  if (n_base >= size_n)
    return;

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  const int n_local = lane_id;
  const int n_global = n_base + n_local;
  const bool valid_n = (n_global < size_n);

  // For transposed layout: weights[expert, k, n]
  const T *expert_weights = weights + (size_t)expert_id * size_k * size_n;

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float s_warp_sums[NUM_WARPS][N_TILE][MAX_TOKENS];

  float acc[MAX_TOKENS];
#pragma unroll
  for (int t = 0; t < MAX_TOKENS; ++t) {
    acc[t] = 0.0f;
  }

  for (int t_base = 0; t_base < num_tokens; t_base += MAX_TOKENS) {
    const int tokens_this_batch = min(MAX_TOKENS, num_tokens - t_base);

#pragma unroll
    for (int t = 0; t < MAX_TOKENS; ++t) {
      acc[t] = 0.0f;
    }

    // Main K-loop with transposed weight access
    for (int k = tid; k < size_k; k += BLOCK_SIZE) {
      // For transposed: weights[expert, k, n_global] = expert_weights[k * N +
      // n]
      float w_val = 0.0f;
      if (valid_n) {
        w_val = vllm::to_float(
            __ldg(&expert_weights[(size_t)k * size_n + n_global]));
      }

#pragma unroll
      for (int t = 0; t < MAX_TOKENS; ++t) {
        if (t < tokens_this_batch) {
          const int token_pair_idx = segment_start + t_base + t;
          const int token_id = sorted_token_ids[token_pair_idx];
          const int input_idx = topk_weights ? token_id : (token_id / topk);

          float in_val = vllm::to_float(__ldg(&input[(size_t)input_idx * size_k + k]));
          acc[t] = __fmaf_rn(w_val, in_val, acc[t]);
        }
      }
    }

#pragma unroll
    for (int t = 0; t < MAX_TOKENS; ++t) {
      acc[t] = moe_gemv_grouped::warp_reduce_sum(acc[t]);
    }

    if (lane_id == 0 && valid_n) {
#pragma unroll
      for (int t = 0; t < MAX_TOKENS; ++t) {
        s_warp_sums[warp_id][n_local][t] = acc[t];
      }
    }
    __syncthreads();

    if (warp_id == 0 && valid_n) {
#pragma unroll
      for (int t = 0; t < MAX_TOKENS; ++t) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
          sum += s_warp_sums[w][n_local][t];
        }

        if (t < tokens_this_batch) {
          const int token_pair_idx = segment_start + t_base + t;
          const int token_id = sorted_token_ids[token_pair_idx];

          if (topk_weights) {
            sum *= topk_weights[token_id];
          }

          T out_val;
          vllm::from_float(out_val, sum);
          output[(size_t)token_id * size_n + n_global] = out_val;
        }
      }
    }
    __syncthreads();
  }
}

// ============================================================================
// Launch Functions
// ============================================================================

extern "C" void moe_gemv_grouped(
    const void *input,                   // [size_m/topk, size_k]
    const void *weights,                 // [num_experts, size_n, size_k]
    const int32_t *sorted_token_ids,     // [size_m]
    const int32_t *expert_ids,           // [size_m] - used to compute offsets
    const float *topk_weights,           // [size_m] or nullptr
    void *output,                        // [size_m, size_n]
    int num_experts, int topk, int size_m, int size_n, int size_k, int dtype,
    cudaStream_t stream) {

  // Compute expert offsets
  int32_t *expert_offsets;
  cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int32_t), stream);
  calculate_expert_offsets(expert_ids, size_m, expert_offsets, num_experts,
                           stream);

  constexpr int BLOCK_SIZE = 256;
  constexpr int N_TILE = 32;
  constexpr int MAX_TOKENS = 8;

  dim3 grid(num_experts, CEILDIV(size_n, N_TILE));
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) { // FP16
    moe_gemv_grouped_kernel<half, BLOCK_SIZE, N_TILE, MAX_TOKENS>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const half *>(input),
            reinterpret_cast<const half *>(weights), sorted_token_ids,
            expert_offsets, topk_weights, reinterpret_cast<half *>(output),
            topk, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) { // BF16
    moe_gemv_grouped_kernel<nv_bfloat16, BLOCK_SIZE, N_TILE, MAX_TOKENS>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const nv_bfloat16 *>(input),
            reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
            expert_offsets, topk_weights,
            reinterpret_cast<nv_bfloat16 *>(output), topk, size_n, size_k);
  }
#endif
  else {
    fprintf(stderr, "moe_gemv_grouped: unsupported dtype %d\n", dtype);
  }

  cudaFreeAsync(expert_offsets, stream);
}

extern "C" void moe_gemv_grouped_transposed(
    const void *input,                   // [size_m/topk, size_k]
    const void *weights,                 // [num_experts, size_k, size_n]
    const int32_t *sorted_token_ids,     // [size_m]
    const int32_t *expert_ids,           // [size_m]
    const float *topk_weights,           // [size_m] or nullptr
    void *output,                        // [size_m, size_n]
    int num_experts, int topk, int size_m, int size_n, int size_k, int dtype,
    cudaStream_t stream) {

  int32_t *expert_offsets;
  cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int32_t), stream);
  calculate_expert_offsets(expert_ids, size_m, expert_offsets, num_experts,
                           stream);

  constexpr int BLOCK_SIZE = 256;
  constexpr int N_TILE = 32;
  constexpr int MAX_TOKENS = 8;

  dim3 grid(num_experts, CEILDIV(size_n, N_TILE));
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) { // FP16
    moe_gemv_grouped_transposed_kernel<half, BLOCK_SIZE, N_TILE, MAX_TOKENS>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const half *>(input),
            reinterpret_cast<const half *>(weights), sorted_token_ids,
            expert_offsets, topk_weights, reinterpret_cast<half *>(output),
            topk, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) { // BF16
    moe_gemv_grouped_transposed_kernel<nv_bfloat16, BLOCK_SIZE, N_TILE,
                                       MAX_TOKENS><<<grid, block, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16 *>(input),
        reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
        expert_offsets, topk_weights, reinterpret_cast<nv_bfloat16 *>(output),
        topk, size_n, size_k);
  }
#endif
  else {
    fprintf(stderr, "moe_gemv_grouped_transposed: unsupported dtype %d\n",
            dtype);
  }

  cudaFreeAsync(expert_offsets, stream);
}
