/**
 * @brief Optimized FP8 GEMM kernels for blockwise quantized weights.
 *
 * Key optimizations:
 * - Vectorized FP8 loads (4 bytes at a time)
 * - Shared memory tiling with fixed sizes for good occupancy
 * - Scale caching to reduce global memory traffic
 * - Coalesced memory access patterns
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace fp8_gemm {

// ============================================================================
// Helper functions
// ============================================================================

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 val) {
    return __half2float(__nv_cvt_fp8_to_halfraw(val.__x, __NV_E4M3));
}

__device__ __forceinline__ float get_block_scale(
    const float* __restrict__ scale,
    int row, int col,
    int scale_row_stride,
    int block_size_y, int block_size_x
) {
    int scale_row = row / block_size_y;
    int scale_col = col / block_size_x;
    return scale[scale_row * scale_row_stride + scale_col];
}

// ============================================================================
// FP8 Matmul Kernel - Optimized with larger tiles and register blocking
// Computes: output = input @ weight.T
// ============================================================================

template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void fp8_matmul_tiled_v2(
    const T* __restrict__ input,           // [M, K]
    const __nv_fp8_e4m3* __restrict__ weight,  // [N, K]
    const float* __restrict__ weight_scale,    // [N/block_y, K/block_x]
    T* __restrict__ output,                // [M, N]
    int M, int N, int K,
    int scale_row_stride,
    int block_size_y, int block_size_x
) {
    // Shared memory - use padding to avoid bank conflicts
    __shared__ float s_input[BLOCK_M][BLOCK_K + 4];
    __shared__ float s_weight[BLOCK_N][BLOCK_K + 4];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * BLOCK_M + ty;
    const int col = bx * BLOCK_N + tx;

    float acc = 0.0f;

    // Number of threads for cooperative loading
    const int num_threads = BLOCK_M * BLOCK_N;
    const int tid = ty * BLOCK_N + tx;

    // Loop over K in tiles
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // Cooperative load of input tile [BLOCK_M, BLOCK_K]
        #pragma unroll
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += num_threads) {
            int load_m = i / BLOCK_K;
            int load_k = i % BLOCK_K;
            int gm = by * BLOCK_M + load_m;
            int gk = k_tile + load_k;

            float val = 0.0f;
            if (gm < M && gk < K) {
                if constexpr (std::is_same_v<T, half>) {
                    val = __half2float(input[gm * K + gk]);
                } else {
                    val = __bfloat162float(input[gm * K + gk]);
                }
            }
            s_input[load_m][load_k] = val;
        }

        // Cooperative load of weight tile [BLOCK_N, BLOCK_K] with dequantization
        #pragma unroll
        for (int i = tid; i < BLOCK_N * BLOCK_K; i += num_threads) {
            int load_n = i / BLOCK_K;
            int load_k = i % BLOCK_K;
            int gn = bx * BLOCK_N + load_n;
            int gk = k_tile + load_k;

            float val = 0.0f;
            if (gn < N && gk < K) {
                __nv_fp8_e4m3 w_fp8 = weight[gn * K + gk];
                float scale = get_block_scale(weight_scale, gn, gk,
                                             scale_row_stride, block_size_y, block_size_x);
                val = fp8_to_float(w_fp8) * scale;
            }
            s_weight[load_n][load_k] = val;
        }

        __syncthreads();

        // Compute partial dot products
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < BLOCK_K; k++) {
                acc += s_input[ty][k] * s_weight[tx][k];
            }
        }

        __syncthreads();
    }

    // Write output
    if (row < M && col < N) {
        if constexpr (std::is_same_v<T, half>) {
            output[row * N + col] = __float2half(acc);
        } else {
            output[row * N + col] = __float2bfloat16(acc);
        }
    }
}

// ============================================================================
// FP8 Indexed MoE GEMM - Optimized for token generation (small batch)
// Each thread computes one output element, vectorized K processing
// ============================================================================

template<typename T, int TILE_K>
__global__ void fp8_indexed_moe_gemm_opt(
    const T* __restrict__ input,               // [num_tokens, K] or [num_tokens, topk, K]
    const __nv_fp8_e4m3* __restrict__ weights, // [num_experts, N, K]
    const float* __restrict__ weight_scales,   // [num_experts, N/block_y, K/block_x]
    const uint32_t* __restrict__ indices,      // [num_tokens, topk]
    T* __restrict__ output,                    // [num_tokens, topk, N]
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    int scale_row_stride,
    int block_size_y, int block_size_x,
    bool input_has_topk_dim
) {
    // Shared memory for input tile - fixed size for good occupancy
    __shared__ float s_input[TILE_K];

    const int token_idx = blockIdx.z;
    const int expert_slot = blockIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= num_tokens || expert_slot >= topk) return;

    const int tid = threadIdx.x;

    // Get expert index
    const uint32_t expert_idx = indices[token_idx * topk + expert_slot];
    if (expert_idx >= (uint32_t)num_experts) return;

    // Pointers
    const __nv_fp8_e4m3* expert_w = weights + (size_t)expert_idx * N * K;
    const int scale_n_dim = CEILDIV(N, block_size_y);
    const int scale_expert_stride = scale_n_dim * scale_row_stride;
    const float* expert_scale = weight_scales + (size_t)expert_idx * scale_expert_stride;

    const T* input_row;
    if (input_has_topk_dim) {
        input_row = input + (size_t)token_idx * topk * K + (size_t)expert_slot * K;
    } else {
        input_row = input + (size_t)token_idx * K;
    }

    float acc = 0.0f;

    // Process K in tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperative load of input tile
        for (int k = tid; k < TILE_K && k_tile + k < K; k += blockDim.x) {
            if constexpr (std::is_same_v<T, half>) {
                s_input[k] = __half2float(input_row[k_tile + k]);
            } else {
                s_input[k] = __bfloat162float(input_row[k_tile + k]);
            }
        }
        __syncthreads();

        // Each thread computes for its output element
        if (n_idx < N) {
            const __nv_fp8_e4m3* weight_row = expert_w + (size_t)n_idx * K + k_tile;

            // Get scale for this block (update when crossing boundaries)
            float scale = get_block_scale(expert_scale, n_idx, k_tile,
                                         scale_row_stride, block_size_y, block_size_x);
            int current_scale_block = k_tile / block_size_x;

            // Process elements in this tile
            int k_end = min(TILE_K, K - k_tile);

            // Unrolled loop for better performance
            int k = 0;

            // Process 4 elements at a time when possible
            for (; k + 3 < k_end; k += 4) {
                int gk = k_tile + k;

                // Check if we need to update scale
                int new_block = gk / block_size_x;
                if (new_block != current_scale_block) {
                    scale = get_block_scale(expert_scale, n_idx, gk,
                                           scale_row_stride, block_size_y, block_size_x);
                    current_scale_block = new_block;
                }

                // Load 4 weights and compute
                float w0 = fp8_to_float(weight_row[k]) * scale;
                float w1 = fp8_to_float(weight_row[k + 1]) * scale;
                float w2 = fp8_to_float(weight_row[k + 2]) * scale;
                float w3 = fp8_to_float(weight_row[k + 3]) * scale;

                acc += s_input[k] * w0;
                acc += s_input[k + 1] * w1;
                acc += s_input[k + 2] * w2;
                acc += s_input[k + 3] * w3;
            }

            // Handle remaining elements
            for (; k < k_end; k++) {
                int gk = k_tile + k;
                int new_block = gk / block_size_x;
                if (new_block != current_scale_block) {
                    scale = get_block_scale(expert_scale, n_idx, gk,
                                           scale_row_stride, block_size_y, block_size_x);
                    current_scale_block = new_block;
                }
                float w = fp8_to_float(weight_row[k]) * scale;
                acc += s_input[k] * w;
            }
        }
        __syncthreads();
    }

    // Write output
    if (n_idx < N) {
        size_t out_idx = (size_t)token_idx * topk * N + (size_t)expert_slot * N + n_idx;
        if constexpr (std::is_same_v<T, half>) {
            output[out_idx] = __float2half(acc);
        } else {
            output[out_idx] = __float2bfloat16(acc);
        }
    }
}

// ============================================================================
// Alternative MoE kernel - one thread per output, no shared memory
// Better for very small batch sizes (token generation)
// ============================================================================

template<typename T>
__global__ void fp8_indexed_moe_gemm_simple(
    const T* __restrict__ input,               // [num_tokens, K] or [num_tokens, topk, K]
    const __nv_fp8_e4m3* __restrict__ weights, // [num_experts, N, K]
    const float* __restrict__ weight_scales,   // [num_experts, N/block_y, K/block_x]
    const uint32_t* __restrict__ indices,      // [num_tokens, topk]
    T* __restrict__ output,                    // [num_tokens, topk, N]
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    int scale_row_stride,
    int block_size_y, int block_size_x,
    bool input_has_topk_dim
) {
    const int token_idx = blockIdx.z;
    const int expert_slot = blockIdx.y;
    const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= num_tokens || expert_slot >= topk || n_idx >= N) return;

    // Get expert index
    const uint32_t expert_idx = indices[token_idx * topk + expert_slot];
    if (expert_idx >= (uint32_t)num_experts) return;

    // Pointers
    const __nv_fp8_e4m3* weight_row = weights + (size_t)expert_idx * N * K + (size_t)n_idx * K;
    const int scale_n_dim = CEILDIV(N, block_size_y);
    const int scale_expert_stride = scale_n_dim * scale_row_stride;
    const float* expert_scale = weight_scales + (size_t)expert_idx * scale_expert_stride;

    const T* input_row;
    if (input_has_topk_dim) {
        input_row = input + (size_t)token_idx * topk * K + (size_t)expert_slot * K;
    } else {
        input_row = input + (size_t)token_idx * K;
    }

    float acc = 0.0f;
    int current_scale_block = -1;
    float scale = 0.0f;

    // Process K dimension with 4-element unrolling
    int k = 0;
    for (; k + 3 < K; k += 4) {
        // Update scale if needed
        int new_block = k / block_size_x;
        if (new_block != current_scale_block) {
            scale = get_block_scale(expert_scale, n_idx, k,
                                   scale_row_stride, block_size_y, block_size_x);
            current_scale_block = new_block;
        }

        // Load input values
        float in0, in1, in2, in3;
        if constexpr (std::is_same_v<T, half>) {
            in0 = __half2float(input_row[k]);
            in1 = __half2float(input_row[k + 1]);
            in2 = __half2float(input_row[k + 2]);
            in3 = __half2float(input_row[k + 3]);
        } else {
            in0 = __bfloat162float(input_row[k]);
            in1 = __bfloat162float(input_row[k + 1]);
            in2 = __bfloat162float(input_row[k + 2]);
            in3 = __bfloat162float(input_row[k + 3]);
        }

        // Load and dequantize weights
        float w0 = fp8_to_float(weight_row[k]) * scale;
        float w1 = fp8_to_float(weight_row[k + 1]) * scale;
        float w2 = fp8_to_float(weight_row[k + 2]) * scale;
        float w3 = fp8_to_float(weight_row[k + 3]) * scale;

        acc += in0 * w0 + in1 * w1 + in2 * w2 + in3 * w3;
    }

    // Handle remaining elements
    for (; k < K; k++) {
        int new_block = k / block_size_x;
        if (new_block != current_scale_block) {
            scale = get_block_scale(expert_scale, n_idx, k,
                                   scale_row_stride, block_size_y, block_size_x);
            current_scale_block = new_block;
        }

        float in_val;
        if constexpr (std::is_same_v<T, half>) {
            in_val = __half2float(input_row[k]);
        } else {
            in_val = __bfloat162float(input_row[k]);
        }
        float w = fp8_to_float(weight_row[k]) * scale;
        acc += in_val * w;
    }

    // Write output
    size_t out_idx = (size_t)token_idx * topk * N + (size_t)expert_slot * N + n_idx;
    if constexpr (std::is_same_v<T, half>) {
        output[out_idx] = __float2half(acc);
    } else {
        output[out_idx] = __float2bfloat16(acc);
    }
}

} // namespace fp8_gemm

// ============================================================================
// C API for FP8 Matmul
// ============================================================================

extern "C" void launch_fp8_matmul_f16(
    const __half* input,
    const __nv_fp8_e4m3* weight,
    const float* weight_scale,
    __half* output,
    int M, int N, int K,
    int scale_row_stride,
    int block_size_y, int block_size_x,
    cudaStream_t stream
) {
    // Use 32x32 tiles with 32x32 thread block
    constexpr int TILE = 32;
    constexpr int TILE_K = 32;

    dim3 block(TILE, TILE);
    dim3 grid(CEILDIV(N, TILE), CEILDIV(M, TILE));

    fp8_gemm::fp8_matmul_tiled_v2<half, TILE, TILE, TILE_K>
        <<<grid, block, 0, stream>>>(
        input, weight, weight_scale, output,
        M, N, K, scale_row_stride, block_size_y, block_size_x
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_fp8_matmul_bf16(
    const __nv_bfloat16* input,
    const __nv_fp8_e4m3* weight,
    const float* weight_scale,
    __nv_bfloat16* output,
    int M, int N, int K,
    int scale_row_stride,
    int block_size_y, int block_size_x,
    cudaStream_t stream
) {
    constexpr int TILE = 32;
    constexpr int TILE_K = 32;

    dim3 block(TILE, TILE);
    dim3 grid(CEILDIV(N, TILE), CEILDIV(M, TILE));

    fp8_gemm::fp8_matmul_tiled_v2<__nv_bfloat16, TILE, TILE, TILE_K>
        <<<grid, block, 0, stream>>>(
        input, weight, weight_scale, output,
        M, N, K, scale_row_stride, block_size_y, block_size_x
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// C API for FP8 Indexed MoE GEMM
// ============================================================================

extern "C" void launch_fp8_indexed_moe_gemm_f16(
    const __half* input,
    const __nv_fp8_e4m3* weights,
    const float* weight_scales,
    const uint32_t* indices,
    __half* output,
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    int scale_row_stride,
    int block_size_y, int block_size_x,
    bool input_has_topk_dim,
    cudaStream_t stream
) {
    constexpr int THREADS = 256;

    dim3 block(THREADS);
    dim3 grid(CEILDIV(N, THREADS), topk, num_tokens);

    // Use simple kernel - no shared memory overhead, good for small batches
    fp8_gemm::fp8_indexed_moe_gemm_simple<half>
        <<<grid, block, 0, stream>>>(
        input, weights, weight_scales, indices, output,
        num_tokens, topk, num_experts, N, K,
        scale_row_stride, block_size_y, block_size_x, input_has_topk_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_fp8_indexed_moe_gemm_bf16(
    const __nv_bfloat16* input,
    const __nv_fp8_e4m3* weights,
    const float* weight_scales,
    const uint32_t* indices,
    __nv_bfloat16* output,
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    int scale_row_stride,
    int block_size_y, int block_size_x,
    bool input_has_topk_dim,
    cudaStream_t stream
) {
    constexpr int THREADS = 256;

    dim3 block(THREADS);
    dim3 grid(CEILDIV(N, THREADS), topk, num_tokens);

    fp8_gemm::fp8_indexed_moe_gemm_simple<__nv_bfloat16>
        <<<grid, block, 0, stream>>>(
        input, weights, weight_scales, indices, output,
        num_tokens, topk, num_experts, N, K,
        scale_row_stride, block_size_y, block_size_x, input_has_topk_dim
    );
    CUDA_CHECK(cudaGetLastError());
}
