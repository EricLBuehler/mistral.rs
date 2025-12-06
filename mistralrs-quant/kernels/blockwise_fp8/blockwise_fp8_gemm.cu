/**
 * @brief Blockwise FP8 GEMM kernels for linear layers and MoE experts.
 *
 * This file contains:
 * 1. Blockwise FP8 GEMM kernel for standard linear layers
 * 2. Blockwise FP8 MoE GEMM kernel (non-WMMA variant)
 * 3. Blockwise FP8 MoE GEMM kernel (WMMA variant)
 *
 * The blockwise FP8 format stores weights as FP8 with per-block scaling factors.
 * During GEMM, weights are dequantized on-the-fly before computation.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

// ============================================================================
// Helper functions for FP8 dequantization
// ============================================================================

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 val) {
    return __half2float(__nv_cvt_fp8_to_halfraw(val.__x, __NV_E4M3));
}

__device__ __forceinline__ float dequant_fp8_blockwise(
    __nv_fp8_e4m3 weight_val,
    float scale
) {
    return fp8_to_float(weight_val) * scale;
}

// ============================================================================
// Blockwise FP8 GEMM for standard linear layers
// Computes: output = input @ weight^T where weight is in blockwise FP8 format
// ============================================================================

// Block tile sizes for GEMM
constexpr int GEMM_BLOCK_M = 64;
constexpr int GEMM_BLOCK_N = 64;
constexpr int GEMM_BLOCK_K = 32;
constexpr int GEMM_THREADS = 256;

template<typename T>
__global__ void blockwise_fp8_gemm_kernel(
    const T* __restrict__ input,                    // [M, K]
    const __nv_fp8_e4m3* __restrict__ weight,       // [N, K] in FP8
    const float* __restrict__ weight_scale,         // [N/block_y, K/block_x]
    T* __restrict__ output,                         // [M, N]
    int M, int N, int K,
    int weight_block_size_y,                        // Block size along N dimension
    int weight_block_size_x,                        // Block size along K dimension
    int scale_stride                                // Stride for scale tensor (K/block_x)
) {
    // Block indices
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;

    // Thread indices within block
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Global output row and column this block computes
    const int m_start = block_m * GEMM_BLOCK_M;
    const int n_start = block_n * GEMM_BLOCK_N;

    // Shared memory for tiles
    __shared__ T s_input[GEMM_BLOCK_M][GEMM_BLOCK_K + 1];  // +1 for bank conflict avoidance
    __shared__ T s_weight[GEMM_BLOCK_N][GEMM_BLOCK_K + 1];

    // Accumulator registers
    float acc[4][4] = {{0.0f}};  // Each thread computes a 4x4 output tile

    // Calculate which output elements this thread is responsible for
    const int thread_m = (tid / 16) * 4;  // 4 rows per thread
    const int thread_n = (tid % 16) * 4;  // 4 cols per thread

    // Main K-loop: iterate over K dimension in tiles
    for (int k_start = 0; k_start < K; k_start += GEMM_BLOCK_K) {
        // Cooperative load of input tile [GEMM_BLOCK_M, GEMM_BLOCK_K]
        for (int i = tid; i < GEMM_BLOCK_M * GEMM_BLOCK_K; i += GEMM_THREADS) {
            int local_m = i / GEMM_BLOCK_K;
            int local_k = i % GEMM_BLOCK_K;
            int global_m = m_start + local_m;
            int global_k = k_start + local_k;

            if (global_m < M && global_k < K) {
                s_input[local_m][local_k] = input[global_m * K + global_k];
            } else {
                s_input[local_m][local_k] = T(0);
            }
        }

        // Cooperative load of weight tile with on-the-fly dequantization
        for (int i = tid; i < GEMM_BLOCK_N * GEMM_BLOCK_K; i += GEMM_THREADS) {
            int local_n = i / GEMM_BLOCK_K;
            int local_k = i % GEMM_BLOCK_K;
            int global_n = n_start + local_n;
            int global_k = k_start + local_k;

            if (global_n < N && global_k < K) {
                // Get the scale for this block
                int scale_y = global_n / weight_block_size_y;
                int scale_x = global_k / weight_block_size_x;
                float scale = weight_scale[scale_y * scale_stride + scale_x];

                // Dequantize weight
                __nv_fp8_e4m3 w_fp8 = weight[global_n * K + global_k];
                float w_dequant = dequant_fp8_blockwise(w_fp8, scale);
                s_weight[local_n][local_k] = static_cast<T>(w_dequant);
            } else {
                s_weight[local_n][local_k] = T(0);
            }
        }

        __syncthreads();

        // Compute: each thread accumulates its 4x4 output tile
        #pragma unroll
        for (int k = 0; k < GEMM_BLOCK_K; k++) {
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                float in_val = static_cast<float>(s_input[thread_m + m][k]);
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    float w_val = static_cast<float>(s_weight[thread_n + n][k]);
                    acc[m][n] += in_val * w_val;
                }
            }
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        int global_m = m_start + thread_m + m;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int global_n = n_start + thread_n + n;
            if (global_m < M && global_n < N) {
                output[global_m * N + global_n] = static_cast<T>(acc[m][n]);
            }
        }
    }
}

// ============================================================================
// Blockwise FP8 MoE GEMM kernel (non-WMMA variant)
// Computes: output[token] = input[token] @ expert_weight[expert_id]^T
// ============================================================================

constexpr int MOE_BLOCK_N_TILE = 64;
constexpr int MOE_BLOCK_K_TILE = 64;
constexpr int MOE_BLOCK_K_THREADS = 8;

template<typename T, typename VecT>
__global__ void blockwise_fp8_moe_gemm_kernel(
    const T* __restrict__ input,                    // [M, K]
    const __nv_fp8_e4m3* __restrict__ weights,      // [num_experts, N, K]
    const float* __restrict__ weight_scales,        // [num_experts, N/block_y, K/block_x]
    const int32_t* __restrict__ sorted_token_ids,   // [M]
    const int32_t* __restrict__ expert_ids,         // [M]
    const float* __restrict__ topk_weights,         // [M] optional, can be nullptr
    T* __restrict__ output,                         // [M, N]
    int num_experts, int topk,
    int M, int N, int K,
    int weight_block_size_y,
    int weight_block_size_x,
    int scale_n_blocks,                             // N / weight_block_size_y
    int scale_k_blocks                              // K / weight_block_size_x
) {
    const int token_idx = blockIdx.y;
    if (token_idx >= M) return;

    const int n_tile_start = blockIdx.x * MOE_BLOCK_N_TILE;
    const int tid_n = threadIdx.x;
    const int tid_k = threadIdx.y;
    const int n = n_tile_start + tid_n;
    if (n >= N) return;

    const int token_id = sorted_token_ids[token_idx];
    const int expert = expert_ids[token_idx];
    if (expert < 0 || expert >= num_experts) return;

    const T* input_row = input + (size_t)(token_id / (topk_weights ? 1 : topk)) * K;
    const __nv_fp8_e4m3* weight_expert = weights + (size_t)expert * (size_t)N * (size_t)K;
    const __nv_fp8_e4m3* weight_row = weight_expert + (size_t)n * K;
    const float* scale_expert = weight_scales + (size_t)expert * scale_n_blocks * scale_k_blocks;

    // Shared memory for tiles
    __shared__ T s_input[MOE_BLOCK_K_TILE];
    __shared__ T s_weights[MOE_BLOCK_N_TILE][MOE_BLOCK_K_TILE];

    // Accumulator
    float acc = 0.0f;

    const int k_vec_tile_size = MOE_BLOCK_K_TILE;

    // Main K-loop
    for (int k_start = 0; k_start < K; k_start += MOE_BLOCK_K_TILE) {
        // Load input tile
        int k_loader_idx = tid_k * blockDim.x + tid_n;
        int num_loaders = blockDim.x * blockDim.y;

        for (int i = k_loader_idx; i < k_vec_tile_size; i += num_loaders) {
            int global_k = k_start + i;
            if (global_k < K) {
                s_input[i] = input_row[global_k];
            } else {
                s_input[i] = T(0);
            }
        }

        // Load weight tile with on-the-fly dequantization
        for (int k_inner = tid_k; k_inner < MOE_BLOCK_K_TILE; k_inner += MOE_BLOCK_K_THREADS) {
            int global_k = k_start + k_inner;
            if (global_k < K) {
                // Get scale for this block
                int scale_y = n / weight_block_size_y;
                int scale_x = global_k / weight_block_size_x;
                float scale = scale_expert[scale_y * scale_k_blocks + scale_x];

                __nv_fp8_e4m3 w_fp8 = weight_row[global_k];
                float w_dequant = dequant_fp8_blockwise(w_fp8, scale);
                s_weights[tid_n][k_inner] = static_cast<T>(w_dequant);
            } else {
                s_weights[tid_n][k_inner] = T(0);
            }
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < MOE_BLOCK_K_TILE; k++) {
            acc += static_cast<float>(s_input[k]) * static_cast<float>(s_weights[tid_n][k]);
        }

        __syncthreads();
    }

    // Apply topk weight and write output
    if (topk_weights) {
        acc *= topk_weights[token_id];
    }
    output[token_id * N + n] = static_cast<T>(acc);
}

// Transposed weight variant for stacked format [num_experts, K, N]
template<typename T, typename VecT>
__global__ void blockwise_fp8_moe_gemm_transposed_kernel(
    const T* __restrict__ input,                    // [M, K]
    const __nv_fp8_e4m3* __restrict__ weights,      // [num_experts, K, N] transposed
    const float* __restrict__ weight_scales,        // [num_experts, K/block_y, N/block_x]
    const int32_t* __restrict__ sorted_token_ids,   // [M]
    const int32_t* __restrict__ expert_ids,         // [M]
    const float* __restrict__ topk_weights,         // [M] optional, can be nullptr
    T* __restrict__ output,                         // [M, N]
    int num_experts, int topk,
    int M, int N, int K,
    int weight_block_size_y,                        // Block size along K (first weight dim)
    int weight_block_size_x,                        // Block size along N (second weight dim)
    int scale_k_blocks,                             // K / weight_block_size_y
    int scale_n_blocks                              // N / weight_block_size_x
) {
    const int token_idx = blockIdx.y;
    if (token_idx >= M) return;

    const int n_tile_start = blockIdx.x * MOE_BLOCK_N_TILE;
    const int tid_n = threadIdx.x;
    const int tid_k = threadIdx.y;
    const int n = n_tile_start + tid_n;
    if (n >= N) return;

    const int token_id = sorted_token_ids[token_idx];
    const int expert = expert_ids[token_idx];
    if (expert < 0 || expert >= num_experts) return;

    const T* input_row = input + (size_t)(token_id / (topk_weights ? 1 : topk)) * K;
    // Transposed layout: [E, K, N]
    const __nv_fp8_e4m3* weight_expert = weights + (size_t)expert * (size_t)K * (size_t)N;
    const float* scale_expert = weight_scales + (size_t)expert * scale_k_blocks * scale_n_blocks;

    // Shared memory for tiles
    __shared__ T s_input[MOE_BLOCK_K_TILE];
    __shared__ T s_weights[MOE_BLOCK_N_TILE][MOE_BLOCK_K_TILE];

    // Accumulator
    float acc = 0.0f;

    // Main K-loop
    for (int k_start = 0; k_start < K; k_start += MOE_BLOCK_K_TILE) {
        // Load input tile
        int k_loader_idx = tid_k * blockDim.x + tid_n;
        int num_loaders = blockDim.x * blockDim.y;

        for (int i = k_loader_idx; i < MOE_BLOCK_K_TILE; i += num_loaders) {
            int global_k = k_start + i;
            if (global_k < K) {
                s_input[i] = input_row[global_k];
            } else {
                s_input[i] = T(0);
            }
        }

        // Load weight tile with on-the-fly dequantization (transposed access)
        for (int k_inner = tid_k; k_inner < MOE_BLOCK_K_TILE; k_inner += MOE_BLOCK_K_THREADS) {
            int global_k = k_start + k_inner;
            if (global_k < K) {
                // Transposed: weight[k, n] = weight_expert[k * N + n]
                int scale_y = global_k / weight_block_size_y;
                int scale_x = n / weight_block_size_x;
                float scale = scale_expert[scale_y * scale_n_blocks + scale_x];

                __nv_fp8_e4m3 w_fp8 = weight_expert[(size_t)global_k * N + n];
                float w_dequant = dequant_fp8_blockwise(w_fp8, scale);
                s_weights[tid_n][k_inner] = static_cast<T>(w_dequant);
            } else {
                s_weights[tid_n][k_inner] = T(0);
            }
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < MOE_BLOCK_K_TILE; k++) {
            acc += static_cast<float>(s_input[k]) * static_cast<float>(s_weights[tid_n][k]);
        }

        __syncthreads();
    }

    // Apply topk weight and write output
    if (topk_weights) {
        acc *= topk_weights[token_id];
    }
    output[token_id * N + n] = static_cast<T>(acc);
}

// ============================================================================
// C interface functions
// ============================================================================

extern "C" void launch_blockwise_fp8_gemm_f16(
    const __half* input,
    const __nv_fp8_e4m3* weight,
    const float* weight_scale,
    __half* output,
    int M, int N, int K,
    int weight_block_size_y,
    int weight_block_size_x,
    int scale_stride,
    cudaStream_t stream
) {
    dim3 grid(CEILDIV(N, GEMM_BLOCK_N), CEILDIV(M, GEMM_BLOCK_M));
    dim3 block(GEMM_THREADS);

    blockwise_fp8_gemm_kernel<__half><<<grid, block, 0, stream>>>(
        input, weight, weight_scale, output,
        M, N, K,
        weight_block_size_y, weight_block_size_x, scale_stride
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_blockwise_fp8_gemm_bf16(
    const __nv_bfloat16* input,
    const __nv_fp8_e4m3* weight,
    const float* weight_scale,
    __nv_bfloat16* output,
    int M, int N, int K,
    int weight_block_size_y,
    int weight_block_size_x,
    int scale_stride,
    cudaStream_t stream
) {
    dim3 grid(CEILDIV(N, GEMM_BLOCK_N), CEILDIV(M, GEMM_BLOCK_M));
    dim3 block(GEMM_THREADS);

    blockwise_fp8_gemm_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        input, weight, weight_scale, output,
        M, N, K,
        weight_block_size_y, weight_block_size_x, scale_stride
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_blockwise_fp8_gemm_f32(
    const float* input,
    const __nv_fp8_e4m3* weight,
    const float* weight_scale,
    float* output,
    int M, int N, int K,
    int weight_block_size_y,
    int weight_block_size_x,
    int scale_stride,
    cudaStream_t stream
) {
    dim3 grid(CEILDIV(N, GEMM_BLOCK_N), CEILDIV(M, GEMM_BLOCK_M));
    dim3 block(GEMM_THREADS);

    blockwise_fp8_gemm_kernel<float><<<grid, block, 0, stream>>>(
        input, weight, weight_scale, output,
        M, N, K,
        weight_block_size_y, weight_block_size_x, scale_stride
    );
    CUDA_CHECK(cudaGetLastError());
}

// MoE GEMM launchers
extern "C" void launch_blockwise_fp8_moe_gemm_f16(
    const __half* input,
    const __nv_fp8_e4m3* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    __half* output,
    int num_experts, int topk,
    int M, int N, int K,
    int weight_block_size_y, int weight_block_size_x,
    int scale_n_blocks, int scale_k_blocks,
    cudaStream_t stream
) {
    dim3 blocks(MOE_BLOCK_N_TILE, MOE_BLOCK_K_THREADS);
    dim3 grids(CEILDIV(N, MOE_BLOCK_N_TILE), M);

    blockwise_fp8_moe_gemm_kernel<__half, __half2><<<grids, blocks, 0, stream>>>(
        input, weights, weight_scales,
        sorted_token_ids, expert_ids, topk_weights, output,
        num_experts, topk, M, N, K,
        weight_block_size_y, weight_block_size_x,
        scale_n_blocks, scale_k_blocks
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_blockwise_fp8_moe_gemm_bf16(
    const __nv_bfloat16* input,
    const __nv_fp8_e4m3* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    __nv_bfloat16* output,
    int num_experts, int topk,
    int M, int N, int K,
    int weight_block_size_y, int weight_block_size_x,
    int scale_n_blocks, int scale_k_blocks,
    cudaStream_t stream
) {
    dim3 blocks(MOE_BLOCK_N_TILE, MOE_BLOCK_K_THREADS);
    dim3 grids(CEILDIV(N, MOE_BLOCK_N_TILE), M);

    blockwise_fp8_moe_gemm_kernel<__nv_bfloat16, __nv_bfloat162><<<grids, blocks, 0, stream>>>(
        input, weights, weight_scales,
        sorted_token_ids, expert_ids, topk_weights, output,
        num_experts, topk, M, N, K,
        weight_block_size_y, weight_block_size_x,
        scale_n_blocks, scale_k_blocks
    );
    CUDA_CHECK(cudaGetLastError());
}

// Transposed MoE GEMM launchers
extern "C" void launch_blockwise_fp8_moe_gemm_transposed_f16(
    const __half* input,
    const __nv_fp8_e4m3* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    __half* output,
    int num_experts, int topk,
    int M, int N, int K,
    int weight_block_size_y, int weight_block_size_x,
    int scale_k_blocks, int scale_n_blocks,
    cudaStream_t stream
) {
    dim3 blocks(MOE_BLOCK_N_TILE, MOE_BLOCK_K_THREADS);
    dim3 grids(CEILDIV(N, MOE_BLOCK_N_TILE), M);

    blockwise_fp8_moe_gemm_transposed_kernel<__half, __half2><<<grids, blocks, 0, stream>>>(
        input, weights, weight_scales,
        sorted_token_ids, expert_ids, topk_weights, output,
        num_experts, topk, M, N, K,
        weight_block_size_y, weight_block_size_x,
        scale_k_blocks, scale_n_blocks
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_blockwise_fp8_moe_gemm_transposed_bf16(
    const __nv_bfloat16* input,
    const __nv_fp8_e4m3* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    __nv_bfloat16* output,
    int num_experts, int topk,
    int M, int N, int K,
    int weight_block_size_y, int weight_block_size_x,
    int scale_k_blocks, int scale_n_blocks,
    cudaStream_t stream
) {
    dim3 blocks(MOE_BLOCK_N_TILE, MOE_BLOCK_K_THREADS);
    dim3 grids(CEILDIV(N, MOE_BLOCK_N_TILE), M);

    blockwise_fp8_moe_gemm_transposed_kernel<__nv_bfloat16, __nv_bfloat162><<<grids, blocks, 0, stream>>>(
        input, weights, weight_scales,
        sorted_token_ids, expert_ids, topk_weights, output,
        num_experts, topk, M, N, K,
        weight_block_size_y, weight_block_size_x,
        scale_k_blocks, scale_n_blocks
    );
    CUDA_CHECK(cudaGetLastError());
}
