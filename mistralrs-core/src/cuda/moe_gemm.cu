/**
 * @brief CUDA kernel for performing a vectorized Mixture-of-Experts (MoE) GEMM
 * operation. This kernel computes output = MoE_GEMM(input, weights), where each
 * token in `input` is routed to an expert based on `expert_ids`. The kernel
 * supports optional top-k gating weights (`topk_weights`) and leverages shared
 * memory, vectorized loads, and fused multiply-add operations for maximum
 * performance.
 *
 * Original Implementation::
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/moe_gemm.cu
 *
 * @details
 * - Each CUDA block computes one [1, BLOCK_N_TILE] output tile for a single
 * token.
 * - The grid is configured as (CEILDIV(N, BLOCK_N_TILE), M).
 * - Shared memory caches tiles of both input and weight data:
 *    - input: [BLOCK_K_TILE]
 *    - weights: [BLOCK_N_TILE, BLOCK_K_TILE]
 * - Eliminates global atomics and uses vectorized compute for FP16 (__hfma2).
 * - Designed for non-quantized GEMM operations in Mixture-of-Experts layers.
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

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace vllm_rs {

// Load vector type (128-bit for best bandwidth)
template <typename T> struct LoadVecType {
  using type = T;
};
template <> struct LoadVecType<half> {
  using type = float4;
};
template <> struct LoadVecType<nv_bfloat16> {
  using type = float4;
};

template <typename T> struct LoadVecSize {
  static constexpr int value = 1;
};
template <> struct LoadVecSize<half> {
  static constexpr int value = 8;
}; // 8 half values in a float4
template <> struct LoadVecSize<nv_bfloat16> {
  static constexpr int value = 8;
}; // 8 bf16s in a float4

inline __device__ void zero(__nv_bfloat162 &dst) {
  // Use a safe initialization that works on all architectures
  unsigned short zero_bits = 0;
  dst.x = *reinterpret_cast<const __nv_bfloat16*>(&zero_bits);
  dst.y = dst.x;
}
inline __device__ void zero(half2 &dst) {
  dst.x = __half_as_ushort(__float2half(0));
  dst.y = __half_as_ushort(__float2half(0));
}

// Robust FMA helper for different CUDA architectures
template <typename VecT>
__device__ __forceinline__ VecT moe_hfma2(VecT a, VecT b, VecT c) {
  return __hfma2(a, b, c);
}

template <>
__device__ __forceinline__ nv_bfloat162 moe_hfma2(nv_bfloat162 a, nv_bfloat162 b, nv_bfloat162 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hbfma2(a, b, c);
#else
  // Fallback for SM < 8.0: convert to float, compute, and convert back
  // Using explicit element access since vector intrinsics are missing
  float fa_x = __bfloat162float(a.x);
  float fa_y = __bfloat162float(a.y);
  float fb_x = __bfloat162float(b.x);
  float fb_y = __bfloat162float(b.y);
  float fc_x = __bfloat162float(c.x);
  float fc_y = __bfloat162float(c.y);
  
  nv_bfloat162 res;
  res.x = __float2bfloat16(fa_x * fb_x + fc_x);
  res.y = __float2bfloat16(fa_y * fb_y + fc_y);
  return res;
#endif
}

} // namespace vllm_rs

/*
 * @param input             [M, K] - Input activations for all tokens.
 * @param weights           [num_experts, N, K] - Expert weight matrices
 * (expert-major layout).
 * @param sorted_token_ids  [M] - Indices of tokens sorted by expert assignment.
 * @param expert_ids        [M] - Expert ID for each token.
 * @param topk_weights      [M] (optional) - Per-token gating weights (nullptr
 * if not used).
 * @param output            [M, N] - Output activations for all tokens.
 * @param num_experts       Total number of experts.
 * @param topk              Number of experts selected per token (top-k
 * routing).
 * @param M                 Number of tokens.
 * @param N                 Output dimension per expert.
 * @param K                 Input dimension per expert.
 */
template <typename T, typename VecT, int BLOCK_N_TILE, int BLOCK_K_TILE,
          int BLOCK_K_THREADS>
__global__ void moe_gemm_vectorized_kernel(
    const T *__restrict__ input,                  // [M, K]
    const T *__restrict__ weights,                // [num_experts, N, K]
    const int32_t *__restrict__ sorted_token_ids, // [M]
    const int32_t *__restrict__ expert_ids,       // [M]
    const float *__restrict__ topk_weights, // [M] optional, can be nullptr
    T *__restrict__ output,                 // [M, N]
    const int num_experts, const int topk, const int M, const int N,
    const int K) {
  // This block processes token at `token_idx`
  const int token_idx = blockIdx.y;
  if (token_idx >= M)
    return;

  // This block computes a tile of N starting at `n_tile_start`
  const int n_tile_start = blockIdx.x * BLOCK_N_TILE;

  // Thread index for N dimension
  const int tid_n = threadIdx.x;
  // Thread index for K loading helper
  const int tid_k = threadIdx.y;

  // This thread's global N-dimension index
  const int n = n_tile_start + tid_n;
  if (n >= N)
    return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts)
    return;

  const T *input_row =
      input + (size_t)(token_id / (topk_weights ? 1 : topk)) * K;
  const T *weight_expert = weights + (size_t)expert * (size_t)N * (size_t)K;
  const T *weight_row = weight_expert + (size_t)n * K;

  // Vector size for load
  constexpr int LOAD_VEC_SIZE = vllm_rs::LoadVecSize<T>::value;
  using LoadVecT = typename vllm_rs::LoadVecType<T>::type;
  // Vector size for compute
  constexpr int VEC_SIZE = sizeof(T);

  // s_input: Caches the [1, K] input vector tile
  __shared__ T s_input[BLOCK_K_TILE];

  // s_weights: Caches the [N, K] weight matrix tile
  // Layout: [BLOCK_N_TILE][BLOCK_K_TILE] for coalesced compute
  __shared__ T s_weights[BLOCK_N_TILE][BLOCK_K_TILE];

  // This thread's accumulator
  VecT acc;
  vllm_rs::zero(acc);
  LoadVecT zero_vec;
  zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

  const int k_compute_vec_tile_size = BLOCK_K_TILE / VEC_SIZE;
  const int k_vec_tile_size = BLOCK_K_TILE / LOAD_VEC_SIZE;
  const int k_vec_dim_size = K / LOAD_VEC_SIZE;

  // Main K-Loop
  // Loop over the K-dimension in tiles of BLOCK_K_TILE
  for (int k_start_vec = 0; k_start_vec < k_vec_dim_size;
       k_start_vec += k_vec_tile_size) {
    // Load Input Tile to Shared Memory ---
    // Parallel load of s_input using all threads in the block
    int k_loader_idx = tid_k * blockDim.x + tid_n;
    int num_loaders = blockDim.x * blockDim.y;

    for (int i = k_loader_idx; i < k_vec_tile_size; i += num_loaders) {
      if (k_start_vec + i < k_vec_dim_size) {
        reinterpret_cast<LoadVecT *>(s_input)[i] =
            reinterpret_cast<const LoadVecT *>(input_row)[k_start_vec + i];
      } else {
        reinterpret_cast<LoadVecT *>(s_input)[i] = zero_vec;
      }
    }

    // Load Weight Tile to Shared Memory
    // Each thread `tid_n` loads its corresponding row of the weight matrix
    // We parallelize the k-loading using `tid_k`
    for (int k_inner_vec = tid_k; k_inner_vec < k_vec_tile_size;
         k_inner_vec += BLOCK_K_THREADS) {
      if (k_start_vec + k_inner_vec < k_vec_dim_size) {
        reinterpret_cast<LoadVecT *>(s_weights[tid_n])[k_inner_vec] =
            reinterpret_cast<const LoadVecT *>(
                weight_row)[k_start_vec + k_inner_vec];
      } else {
        reinterpret_cast<LoadVecT *>(s_weights[tid_n])[k_inner_vec] = zero_vec;
      }
    }

    __syncthreads(); // Wait for s_input and s_weights to be loaded

    // Compute Partial Dot Product
    VecT *input_vec = reinterpret_cast<VecT *>(s_input);
    VecT *weight_vec = reinterpret_cast<VecT *>(s_weights[tid_n]);
#pragma unroll
    for (int k_vec = 0; k_vec < k_compute_vec_tile_size; ++k_vec) {
      acc = vllm_rs::moe_hfma2(input_vec[k_vec], weight_vec[k_vec], acc);
    }
  }

  __syncthreads();

  // Finalize and Write Output
  if (topk_weights) {
    // Apply top-k weight scaling
    T output_val;
    vllm::from_float(output_val, vllm::to_float(__hadd(acc.x, acc.y)) *
                                     topk_weights[token_id]);
    output[token_id * N + n] = output_val;
  } else {
    output[token_id * N + n] = __hadd(acc.x, acc.y);
  }
}

/*
 * Transposed weight kernel for [num_experts, K, N] layout (stacked format).
 *
 * @param input             [M, K] - Input activations for all tokens.
 * @param weights           [num_experts, K, N] - Expert weight matrices
 * (transposed layout).
 * @param sorted_token_ids  [M] - Indices of tokens sorted by expert assignment.
 * @param expert_ids        [M] - Expert ID for each token.
 * @param topk_weights      [M] (optional) - Per-token gating weights (nullptr
 * if not used).
 * @param output            [M, N] - Output activations for all tokens.
 * @param num_experts       Total number of experts.
 * @param topk              Number of experts selected per token (top-k
 * routing).
 * @param M                 Number of tokens.
 * @param N                 Output dimension per expert.
 * @param K                 Input dimension per expert.
 */
template <typename T, typename VecT, int BLOCK_N_TILE, int BLOCK_K_TILE,
          int BLOCK_K_THREADS>
__global__ void moe_gemm_transposed_kernel(
    const T *__restrict__ input,   // [M, K]
    const T *__restrict__ weights, // [num_experts, K, N] - transposed layout
    const int32_t *__restrict__ sorted_token_ids, // [M]
    const int32_t *__restrict__ expert_ids,       // [M]
    const float *__restrict__ topk_weights, // [M] optional, can be nullptr
    T *__restrict__ output,                 // [M, N]
    const int num_experts, const int topk, const int M, const int N,
    const int K) {
  // This block processes token at `token_idx`
  const int token_idx = blockIdx.y;
  if (token_idx >= M)
    return;

  // This block computes a tile of N starting at `n_tile_start`
  const int n_tile_start = blockIdx.x * BLOCK_N_TILE;

  // Thread index for N dimension
  const int tid_n = threadIdx.x;
  // Thread index for K loading helper
  const int tid_k = threadIdx.y;

  // This thread's global N-dimension index
  const int n = n_tile_start + tid_n;
  if (n >= N)
    return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts)
    return;

  const T *input_row =
      input + (size_t)(token_id / (topk_weights ? 1 : topk)) * K;
  // For transposed layout [E, K, N]: base is expert * K * N
  const T *weight_expert = weights + (size_t)expert * (size_t)K * (size_t)N;

  // Vector size for load
  constexpr int LOAD_VEC_SIZE = vllm_rs::LoadVecSize<T>::value;
  using LoadVecT = typename vllm_rs::LoadVecType<T>::type;
  // Vector size for compute
  constexpr int VEC_SIZE = sizeof(T);

  // s_input: Caches the [1, K] input vector tile
  __shared__ T s_input[BLOCK_K_TILE];

  // s_weights: Caches the [N, K] weight matrix tile (conceptually transposed
  // from global) Layout: [BLOCK_N_TILE][BLOCK_K_TILE] for coalesced compute
  __shared__ T s_weights[BLOCK_N_TILE][BLOCK_K_TILE];

  // This thread's accumulator
  VecT acc;
  vllm_rs::zero(acc);
  LoadVecT zero_vec;
  zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

  const int k_compute_vec_tile_size = BLOCK_K_TILE / VEC_SIZE;
  const int k_vec_tile_size = BLOCK_K_TILE / LOAD_VEC_SIZE;
  const int k_vec_dim_size = K / LOAD_VEC_SIZE;

  // Main K-Loop
  // Loop over the K-dimension in tiles of BLOCK_K_TILE
  for (int k_start_vec = 0; k_start_vec < k_vec_dim_size;
       k_start_vec += k_vec_tile_size) {
    // Load Input Tile to Shared Memory ---
    // Parallel load of s_input using all threads in the block
    int k_loader_idx = tid_k * blockDim.x + tid_n;
    int num_loaders = blockDim.x * blockDim.y;

    for (int i = k_loader_idx; i < k_vec_tile_size; i += num_loaders) {
      if (k_start_vec + i < k_vec_dim_size) {
        reinterpret_cast<LoadVecT *>(s_input)[i] =
            reinterpret_cast<const LoadVecT *>(input_row)[k_start_vec + i];
      } else {
        reinterpret_cast<LoadVecT *>(s_input)[i] = zero_vec;
      }
    }

    // Load Weight Tile to Shared Memory
    // For transposed layout [E, K, N]: weight[e, k, n] is at offset k * N + n
    // We need to load weight[k_start:k_start+BLOCK_K_TILE, n] for each n in our
    // tile Each thread tid_n loads weights for column n, iterating over k
    for (int k_inner = tid_k; k_inner < BLOCK_K_TILE;
         k_inner += BLOCK_K_THREADS) {
      int k_global = k_start_vec * LOAD_VEC_SIZE + k_inner;
      if (k_global < K) {
        // In transposed layout: weight[k, n] = weight_expert[k * N + n]
        s_weights[tid_n][k_inner] = weight_expert[(size_t)k_global * N + n];
      } else {
        s_weights[tid_n][k_inner] = T(0);
      }
    }

    __syncthreads(); // Wait for s_input and s_weights to be loaded

    // Compute Partial Dot Product
    VecT *input_vec = reinterpret_cast<VecT *>(s_input);
    VecT *weight_vec = reinterpret_cast<VecT *>(s_weights[tid_n]);
#pragma unroll
    for (int k_vec = 0; k_vec < k_compute_vec_tile_size; ++k_vec) {
      acc = vllm_rs::moe_hfma2(input_vec[k_vec], weight_vec[k_vec], acc);
    }
  }

  __syncthreads();

  // Finalize and Write Output
  if (topk_weights) {
    // Apply top-k weight scaling
    T output_val;
    vllm::from_float(output_val, vllm::to_float(__hadd(acc.x, acc.y)) *
                                     topk_weights[token_id]);
    output[token_id * N + n] = output_val;
  } else {
    output[token_id * N + n] = __hadd(acc.x, acc.y);
  }
}

extern "C" void
moe_gemm(const void *input,   // input [size_m or size_m / topk, size_k]
         const void *weights, // weights [num_experts, size_n, size_k]
         const int32_t *sorted_token_ids, const int32_t *expert_ids,
         const float *topk_weights, // device ptr or nullptr
         void *output,              // output [size_m, size_n]
         int num_experts, int topk, int size_m, int size_n, int size_k,
         int dtype, // 0=float16, 1=bf16
         cudaStream_t stream) {

  // These tile sizes can be tuned based on GPU architecture and problem size
  constexpr int BLOCK_N_TILE = 64;
  constexpr int BLOCK_K_TILE = 64;
  constexpr int BLOCK_K_THREADS =
      8; // BLOCK_N_TILE * BLOCK_K_THREADS = 512 threads

  dim3 blocks(BLOCK_N_TILE, BLOCK_K_THREADS);
  dim3 grids(CEILDIV(size_n, BLOCK_N_TILE), size_m);

  // Note: No shared memory size needed in launch, as it's statically allocated.
  // If BLOCK_K_TILE or BLOCK_N_TILE were dynamic (template params),
  // we would calculate and pass it here.

  // Vectorization requires K to be divisible by the vector size
  int load_vec_size = (dtype == 2) ? 1 : vllm_rs::LoadVecSize<half>::value;
  ASSERT_THROW(size_k % BLOCK_K_TILE == 0,
               "size_k must be divisible by BLOCK_K_TILE");
  ASSERT_THROW(size_k % load_vec_size == 0,
               "size_k must be divisible by vector size (2 for fp16/bf16)");

  // Output is same type as input, so size is based on dtype
  // size_t output_bytes = (size_t)size_m * size_n * (dtype == 2 ? 4 : 2);
  // cudaMemsetAsync(output, 0, output_bytes, stream);

  if (dtype == 0) {
    moe_gemm_vectorized_kernel<half, half2, BLOCK_N_TILE, BLOCK_K_TILE,
                               BLOCK_K_THREADS><<<grids, blocks, 0, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids, expert_ids,
        topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
        size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) {
    moe_gemm_vectorized_kernel<nv_bfloat16, nv_bfloat162, BLOCK_N_TILE,
                               BLOCK_K_TILE, BLOCK_K_THREADS>
        <<<grids, blocks, 0, stream>>>(
            reinterpret_cast<const nv_bfloat16 *>(input),
            reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
            expert_ids, topk_weights, reinterpret_cast<nv_bfloat16 *>(output),
            num_experts, topk, size_m, size_n, size_k);
  }
#endif
  else {
    fprintf(stderr, "moe_gemm: unsupported dtype.\n");
  }
}

// Transposed weight variant: weights are [num_experts, size_k, size_n] instead
// of [num_experts, size_n, size_k]
extern "C" void moe_gemm_transposed(
    const void *input, // input [size_m or size_m / topk, size_k]
    const void
        *weights, // weights [num_experts, size_k, size_n] - transposed layout
    const int32_t *sorted_token_ids, const int32_t *expert_ids,
    const float *topk_weights, // device ptr or nullptr
    void *output,              // output [size_m, size_n]
    int num_experts, int topk, int size_m, int size_n, int size_k,
    int dtype, // 0=float16, 1=bf16
    cudaStream_t stream) {

  // These tile sizes can be tuned based on GPU architecture and problem size
  constexpr int BLOCK_N_TILE = 64;
  constexpr int BLOCK_K_TILE = 64;
  constexpr int BLOCK_K_THREADS =
      8; // BLOCK_N_TILE * BLOCK_K_THREADS = 512 threads

  dim3 blocks(BLOCK_N_TILE, BLOCK_K_THREADS);
  dim3 grids(CEILDIV(size_n, BLOCK_N_TILE), size_m);

  // Vectorization requires K to be divisible by the vector size
  int load_vec_size = (dtype == 2) ? 1 : vllm_rs::LoadVecSize<half>::value;
  ASSERT_THROW(size_k % BLOCK_K_TILE == 0,
               "size_k must be divisible by BLOCK_K_TILE");
  ASSERT_THROW(size_k % load_vec_size == 0,
               "size_k must be divisible by vector size (2 for fp16/bf16)");

  if (dtype == 0) {
    moe_gemm_transposed_kernel<half, half2, BLOCK_N_TILE, BLOCK_K_TILE,
                               BLOCK_K_THREADS><<<grids, blocks, 0, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids, expert_ids,
        topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
        size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) {
    moe_gemm_transposed_kernel<nv_bfloat16, nv_bfloat162, BLOCK_N_TILE,
                               BLOCK_K_TILE, BLOCK_K_THREADS>
        <<<grids, blocks, 0, stream>>>(
            reinterpret_cast<const nv_bfloat16 *>(input),
            reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
            expert_ids, topk_weights, reinterpret_cast<nv_bfloat16 *>(output),
            num_experts, topk, size_m, size_n, size_k);
  }
#endif
  else {
    fprintf(stderr, "moe_gemm_transposed: unsupported dtype.\n");
  }
}