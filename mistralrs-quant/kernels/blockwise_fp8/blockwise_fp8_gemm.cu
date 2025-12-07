/**
 * @brief FP8 GEMM kernels for blockwise quantized weights.
 *
 * This file contains:
 * 1. FP8 matmul kernel (for forward method - SLOW backend)
 * 2. FP8 indexed MoE GEMM kernel (for gather_forward method - FAST backend)
 *
 * Both kernels work with blockwise FP8 quantized weights and dequantize on-the-fly
 * during computation for maximum performance.
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
// Helper functions for FP8 dequantization
// ============================================================================

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 val) {
    return __half2float(__nv_cvt_fp8_to_halfraw(val.__x, __NV_E4M3));
}

__device__ __forceinline__ half fp8_to_half(__nv_fp8_e4m3 val) {
    half result;
    result = __nv_cvt_fp8_to_halfraw(val.__x, __NV_E4M3);
    return result;
}

__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_e4m3 val) {
    return __float2bfloat16(fp8_to_float(val));
}

// Get scale for a given position in blockwise quantized tensor
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
// FP8 Matmul Kernel (for forward method - single expert)
// Computes: output = input @ weight.T where weight is FP8 blockwise quantized
// Input: [M, K] in fp16/bf16
// Weight: [N, K] in FP8 with blockwise scales
// Output: [M, N] in fp16/bf16
// ============================================================================

template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void fp8_matmul_kernel(
    const T* __restrict__ input,           // [M, K]
    const __nv_fp8_e4m3* __restrict__ weight,  // [N, K]
    const float* __restrict__ weight_scale,    // [N/block_y, K/block_x]
    T* __restrict__ output,                // [M, N]
    int M, int N, int K,
    int scale_row_stride,
    int block_size_y, int block_size_x
) {
    // Thread block handles BLOCK_M x BLOCK_N output tile
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int global_row = block_row + thread_row;
    int global_col = block_col + thread_col;

    if (global_row >= M || global_col >= N) return;

    // Accumulator
    float acc = 0.0f;

    // Loop over K dimension
    for (int k = 0; k < K; k++) {
        // Load input value
        float input_val;
        if constexpr (std::is_same_v<T, half>) {
            input_val = __half2float(input[global_row * K + k]);
        } else {
            input_val = __bfloat162float(input[global_row * K + k]);
        }

        // Load and dequantize weight
        __nv_fp8_e4m3 weight_fp8 = weight[global_col * K + k];
        float scale = get_block_scale(weight_scale, global_col, k,
                                      scale_row_stride, block_size_y, block_size_x);
        float weight_val = fp8_to_float(weight_fp8) * scale;

        acc += input_val * weight_val;
    }

    // Write output
    if constexpr (std::is_same_v<T, half>) {
        output[global_row * N + global_col] = __float2half(acc);
    } else {
        output[global_row * N + global_col] = __float2bfloat16(acc);
    }
}

// Optimized tiled version with shared memory
template<typename T, int TILE_M, int TILE_N, int TILE_K>
__global__ void fp8_matmul_tiled_kernel(
    const T* __restrict__ input,           // [M, K]
    const __nv_fp8_e4m3* __restrict__ weight,  // [N, K]
    const float* __restrict__ weight_scale,    // [N/block_y, K/block_x]
    T* __restrict__ output,                // [M, N]
    int M, int N, int K,
    int scale_row_stride,
    int block_size_y, int block_size_x
) {
    __shared__ float s_input[TILE_M][TILE_K];
    __shared__ float s_weight[TILE_N][TILE_K];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float acc = 0.0f;

    // Loop over K in tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperative load of input tile
        if (row < M && k_tile + tx < K) {
            float val;
            if constexpr (std::is_same_v<T, half>) {
                val = __half2float(input[row * K + k_tile + tx]);
            } else {
                val = __bfloat162float(input[row * K + k_tile + tx]);
            }
            s_input[ty][tx] = val;
        } else {
            s_input[ty][tx] = 0.0f;
        }

        // Cooperative load of weight tile (with dequantization)
        if (col < N && k_tile + ty < K) {
            __nv_fp8_e4m3 w_fp8 = weight[col * K + k_tile + ty];
            float scale = get_block_scale(weight_scale, col, k_tile + ty,
                                         scale_row_stride, block_size_y, block_size_x);
            s_weight[tx][ty] = fp8_to_float(w_fp8) * scale;
        } else {
            s_weight[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            acc += s_input[ty][k] * s_weight[tx][k];
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
// FP8 Indexed MoE GEMM Kernel (for gather_forward method - FAST backend)
// Computes indexed matmul for MoE where each token selects specific experts
//
// Input: [num_tokens, 1, K] or [num_tokens, topk, K]
// Weights: [num_experts, N, K] in FP8 with blockwise scales
// Indices: [num_tokens, topk]
// Output: [num_tokens, topk, N]
// ============================================================================

template<typename T, int TILE_K>
__global__ void fp8_indexed_moe_gemm_kernel(
    const T* __restrict__ input,               // [num_tokens, K] or [num_tokens, topk, K]
    const __nv_fp8_e4m3* __restrict__ weights, // [num_experts, N, K]
    const float* __restrict__ weight_scales,   // [num_experts, N/block_y, K/block_x]
    const uint32_t* __restrict__ indices,       // [num_tokens, topk]
    T* __restrict__ output,                    // [num_tokens, topk, N]
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    int scale_row_stride,    // K/block_x
    int block_size_y, int block_size_x,
    bool input_has_topk_dim  // true if input is [num_tokens, topk, K], false if [num_tokens, 1, K]
) {
    // Each block handles one (token, expert_slot, n_tile) combination
    int token_idx = blockIdx.z;
    int expert_slot = blockIdx.y;  // 0 to topk-1
    int n_tile = blockIdx.x;

    if (token_idx >= num_tokens || expert_slot >= topk) return;

    int n_start = n_tile * blockDim.x;
    int n_local = threadIdx.x;
    int n_global = n_start + n_local;

    if (n_global >= N) return;

    // Get expert index for this token and slot
    uint32_t expert_idx = indices[token_idx * topk + expert_slot];
    if (expert_idx >= (uint32_t)num_experts) return;

    // Pointer to expert's weights [N, K]
    const __nv_fp8_e4m3* expert_w = weights + (size_t)expert_idx * N * K;
    // Pointer to expert's scales [N/block_y, K/block_x]
    int scale_expert_stride = (N / block_size_y) * scale_row_stride;
    const float* expert_scale = weight_scales + (size_t)expert_idx * scale_expert_stride;

    // Pointer to input row
    const T* input_row;
    if (input_has_topk_dim) {
        input_row = input + (size_t)token_idx * topk * K + (size_t)expert_slot * K;
    } else {
        input_row = input + (size_t)token_idx * K;
    }

    // Accumulate dot product
    float acc = 0.0f;

    // Weight row for this output column
    const __nv_fp8_e4m3* weight_row = expert_w + (size_t)n_global * K;

    for (int k = 0; k < K; k++) {
        // Load input
        float input_val;
        if constexpr (std::is_same_v<T, half>) {
            input_val = __half2float(input_row[k]);
        } else {
            input_val = __bfloat162float(input_row[k]);
        }

        // Load and dequantize weight
        __nv_fp8_e4m3 w_fp8 = weight_row[k];
        float scale = get_block_scale(expert_scale, n_global, k,
                                     scale_row_stride, block_size_y, block_size_x);
        float weight_val = fp8_to_float(w_fp8) * scale;

        acc += input_val * weight_val;
    }

    // Write output
    size_t out_idx = (size_t)token_idx * topk * N + (size_t)expert_slot * N + n_global;
    if constexpr (std::is_same_v<T, half>) {
        output[out_idx] = __float2half(acc);
    } else {
        output[out_idx] = __float2bfloat16(acc);
    }
}

// Optimized version with shared memory tiling
template<typename T, int TILE_N, int TILE_K>
__global__ void fp8_indexed_moe_gemm_tiled_kernel(
    const T* __restrict__ input,               // [num_tokens, K] or [num_tokens, topk, K]
    const __nv_fp8_e4m3* __restrict__ weights, // [num_experts, N, K]
    const float* __restrict__ weight_scales,   // [num_experts, N/block_y, K/block_x]
    const uint32_t* __restrict__ indices,       // [num_tokens, topk]
    T* __restrict__ output,                    // [num_tokens, topk, N]
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    int scale_row_stride,    // K/block_x
    int block_size_y, int block_size_x,
    bool input_has_topk_dim
) {
    __shared__ float s_input[TILE_K];
    __shared__ float s_weight[TILE_N][TILE_K];
    __shared__ float s_scale[TILE_N];

    int token_idx = blockIdx.z;
    int expert_slot = blockIdx.y;
    int n_tile = blockIdx.x;

    if (token_idx >= num_tokens || expert_slot >= topk) return;

    int tid = threadIdx.x;
    int n_start = n_tile * TILE_N;
    int n_global = n_start + tid;

    // Get expert index
    uint32_t expert_idx = indices[token_idx * topk + expert_slot];
    if (expert_idx >= (uint32_t)num_experts) return;

    const __nv_fp8_e4m3* expert_w = weights + (size_t)expert_idx * N * K;
    int scale_expert_stride = CEILDIV(N, block_size_y) * scale_row_stride;
    const float* expert_scale = weight_scales + (size_t)expert_idx * scale_expert_stride;

    const T* input_row;
    if (input_has_topk_dim) {
        input_row = input + (size_t)token_idx * topk * K + (size_t)expert_slot * K;
    } else {
        input_row = input + (size_t)token_idx * K;
    }

    float acc = 0.0f;

    // Loop over K in tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load input tile cooperatively
        if (tid < TILE_K && k_tile + tid < K) {
            if constexpr (std::is_same_v<T, half>) {
                s_input[tid] = __half2float(input_row[k_tile + tid]);
            } else {
                s_input[tid] = __bfloat162float(input_row[k_tile + tid]);
            }
        }

        // Load weight tile with dequantization
        if (n_global < N) {
            // Cache the scale for this k_tile (assumes block_size_x divides TILE_K or handle edge)
            s_scale[tid] = get_block_scale(expert_scale, n_global, k_tile,
                                          scale_row_stride, block_size_y, block_size_x);
        }

        __syncthreads();

        // Load and compute
        if (n_global < N) {
            const __nv_fp8_e4m3* weight_row = expert_w + (size_t)n_global * K;
            float scale = s_scale[tid];

            #pragma unroll
            for (int k = 0; k < TILE_K && k_tile + k < K; k++) {
                __nv_fp8_e4m3 w_fp8 = weight_row[k_tile + k];
                // Update scale if we cross a block boundary
                if (k > 0 && (k_tile + k) % block_size_x == 0) {
                    scale = get_block_scale(expert_scale, n_global, k_tile + k,
                                           scale_row_stride, block_size_y, block_size_x);
                }
                float weight_val = fp8_to_float(w_fp8) * scale;
                acc += s_input[k] * weight_val;
            }
        }

        __syncthreads();
    }

    // Write output
    if (n_global < N) {
        size_t out_idx = (size_t)token_idx * topk * N + (size_t)expert_slot * N + n_global;
        if constexpr (std::is_same_v<T, half>) {
            output[out_idx] = __float2half(acc);
        } else {
            output[out_idx] = __float2bfloat16(acc);
        }
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
    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(CEILDIV(N, TILE_SIZE), CEILDIV(M, TILE_SIZE));

    fp8_gemm::fp8_matmul_tiled_kernel<half, TILE_SIZE, TILE_SIZE, TILE_SIZE>
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
    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(CEILDIV(N, TILE_SIZE), CEILDIV(M, TILE_SIZE));

    fp8_gemm::fp8_matmul_tiled_kernel<__nv_bfloat16, TILE_SIZE, TILE_SIZE, TILE_SIZE>
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
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 64;

    dim3 block(TILE_N);
    dim3 grid(CEILDIV(N, TILE_N), topk, num_tokens);

    fp8_gemm::fp8_indexed_moe_gemm_tiled_kernel<half, TILE_N, TILE_K>
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
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 64;

    dim3 block(TILE_N);
    dim3 grid(CEILDIV(N, TILE_N), topk, num_tokens);

    fp8_gemm::fp8_indexed_moe_gemm_tiled_kernel<__nv_bfloat16, TILE_N, TILE_K>
        <<<grid, block, 0, stream>>>(
        input, weights, weight_scales, indices, output,
        num_tokens, topk, num_experts, N, K,
        scale_row_stride, block_size_y, block_size_x, input_has_topk_dim
    );
    CUDA_CHECK(cudaGetLastError());
}
