/**
 * @brief MXFP4 GEMM kernels with LUT-based dequantization.
 *
 * MXFP4 Format (OCP Microscaling):
 * - FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
 * - Block size: 32 elements
 * - Scale: E8M0 format (8-bit exponent, stored as u8 with bias 127)
 * - 2 FP4 values packed per byte (nibbles)
 *
 * Dequantization: value = ldexp(lut[nibble], scale - 127)
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

// MXFP4 block size (32 elements per scale)
#define MXFP4_BLOCK_SIZE 32

namespace mxfp4_gemm {

// FP4 E2M1 lookup table in constant memory for fast access
// Values: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ============================================================================
// Helper functions
// ============================================================================

// Dequantize a single FP4 nibble with scale
__device__ __forceinline__ float mxfp4_to_float(uint8_t nibble, int8_t scale_exp) {
    float base = FP4_LUT[nibble & 0xF];
    // ldexp(base, scale_exp - 127) = base * 2^(scale_exp - 127)
    return ldexpf(base, (int)scale_exp - 127);
}

// Extract low nibble (bits 0-3)
__device__ __forceinline__ uint8_t low_nibble(uint8_t byte) {
    return byte & 0x0F;
}

// Extract high nibble (bits 4-7)
__device__ __forceinline__ uint8_t high_nibble(uint8_t byte) {
    return (byte >> 4) & 0x0F;
}

// ============================================================================
// MXFP4 Matmul Kernel (for forward method)
// ============================================================================

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void mxfp4_matmul_tiled(
    const T *__restrict__ input,
    const uint8_t *__restrict__ weight,      // Packed FP4 weights [N, K/2]
    const uint8_t *__restrict__ weight_scale, // E8M0 scales [N, K/32]
    const T *__restrict__ bias,               // Optional bias [N]
    T *__restrict__ output,
    int M, int N, int K,
    bool has_bias
) {
    __shared__ float s_input[BLOCK_M][BLOCK_K + 4];
    __shared__ float s_weight[BLOCK_N][BLOCK_K + 4];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * BLOCK_M + ty;
    const int col = bx * BLOCK_N + tx;

    float acc = 0.0f;

    const int num_threads = BLOCK_M * BLOCK_N;
    const int tid = ty * BLOCK_N + tx;

    // Number of scale blocks along K dimension
    const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);

    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // Load input tile
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += num_threads) {
            int lm = i / BLOCK_K;
            int lk = i % BLOCK_K;
            int gm = by * BLOCK_M + lm;
            int gk = k_tile + lk;

            float val = 0.0f;
            if (gm < M && gk < K) {
                if constexpr (std::is_same_v<T, half>) {
                    val = __half2float(__ldg(&input[gm * K + gk]));
                } else {
                    val = __bfloat162float(__ldg(&input[gm * K + gk]));
                }
            }
            s_input[lm][lk] = val;
        }

        // Load weight tile with dequantization
        for (int i = tid; i < BLOCK_N * BLOCK_K; i += num_threads) {
            int ln = i / BLOCK_K;
            int lk = i % BLOCK_K;
            int gn = bx * BLOCK_N + ln;
            int gk = k_tile + lk;

            float val = 0.0f;
            if (gn < N && gk < K) {
                // Get packed byte (2 FP4 values per byte)
                int byte_idx = gk / 2;
                uint8_t packed = __ldg(&weight[gn * (K / 2) + byte_idx]);

                // Get scale for this block
                int scale_idx = gk / MXFP4_BLOCK_SIZE;
                int8_t scale_exp = (int8_t)__ldg(&weight_scale[gn * scale_stride + scale_idx]);

                // Extract correct nibble
                uint8_t nibble = (gk & 1) ? high_nibble(packed) : low_nibble(packed);
                val = mxfp4_to_float(nibble, scale_exp);
            }
            s_weight[ln][lk] = val;
        }

        __syncthreads();

        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < BLOCK_K; k++) {
                acc += s_input[ty][k] * s_weight[tx][k];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        // Add bias if present
        if (has_bias && bias != nullptr) {
            if constexpr (std::is_same_v<T, half>) {
                acc += __half2float(__ldg(&bias[col]));
            } else {
                acc += __bfloat162float(__ldg(&bias[col]));
            }
        }

        if constexpr (std::is_same_v<T, half>) {
            output[row * N + col] = __float2half(acc);
        } else {
            output[row * N + col] = __float2bfloat16(acc);
        }
    }
}

// ============================================================================
// MXFP4 MoE GEMM - Warp-parallel kernel with vectorized loads
// Each warp (32 threads) computes one output element collaboratively
// ============================================================================

template <typename T>
__global__ void mxfp4_moe_gemm(
    const T *__restrict__ input,
    const uint8_t *__restrict__ weights,      // [num_experts, N, K/2]
    const uint8_t *__restrict__ weight_scales, // [num_experts, N, K/32]
    const T *__restrict__ biases,              // [num_experts, N] or nullptr
    const uint32_t *__restrict__ indices,
    T *__restrict__ output,
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    bool has_bias,
    bool input_has_topk_dim
) {
    // Each warp computes one output element
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;

    // Decode warp_id to (token, expert_slot, n_idx)
    const int n_idx = warp_id % N;
    const int temp = warp_id / N;
    const int expert_slot = temp % topk;
    const int token_idx = temp / topk;

    if (token_idx >= num_tokens) return;

    const uint32_t expert_idx = __ldg(&indices[token_idx * topk + expert_slot]);
    if (expert_idx >= (uint32_t)num_experts) return;

    // Weight layout: [num_experts, N, K/2] for packed bytes
    const int weight_row_stride = K / 2;
    const uint8_t *w_row = weights + (size_t)expert_idx * N * weight_row_stride + (size_t)n_idx * weight_row_stride;

    // Scale layout: [num_experts, N, K/32]
    const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);
    const uint8_t *scale_row = weight_scales + (size_t)expert_idx * N * scale_stride + (size_t)n_idx * scale_stride;

    // Input pointer
    const T *in_row;
    if (input_has_topk_dim) {
        in_row = input + (size_t)token_idx * topk * K + (size_t)expert_slot * K;
    } else {
        in_row = input + (size_t)token_idx * K;
    }

    float acc = 0.0f;

    // Process 4 elements per thread per iteration
    // Each warp processes 32*4 = 128 elements per iteration
    const int K_aligned = (K / 128) * 128;

    for (int k_base = 0; k_base < K_aligned; k_base += 128) {
        int k = k_base + lane_id * 4;

        // Load 4 input values
        float i0, i1, i2, i3;
        if constexpr (std::is_same_v<T, half>) {
            half2 h01 = __ldg(reinterpret_cast<const half2 *>(&in_row[k]));
            half2 h23 = __ldg(reinterpret_cast<const half2 *>(&in_row[k + 2]));
            i0 = __half2float(h01.x);
            i1 = __half2float(h01.y);
            i2 = __half2float(h23.x);
            i3 = __half2float(h23.y);
        } else {
            __nv_bfloat162 b01 = __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k]));
            __nv_bfloat162 b23 = __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k + 2]));
            i0 = __bfloat162float(b01.x);
            i1 = __bfloat162float(b01.y);
            i2 = __bfloat162float(b23.x);
            i3 = __bfloat162float(b23.y);
        }

        // Load 2 packed bytes (4 FP4 values) - 16-bit load
        int byte_idx = k / 2;
        uint16_t packed2 = __ldg(reinterpret_cast<const uint16_t *>(&w_row[byte_idx]));
        uint8_t byte0 = packed2 & 0xFF;
        uint8_t byte1 = (packed2 >> 8) & 0xFF;

        // Get scale (all 4 values likely share the same scale within block of 32)
        int scale_idx = k / MXFP4_BLOCK_SIZE;
        int8_t scale_exp = (int8_t)__ldg(&scale_row[scale_idx]);

        // Dequantize 4 values
        float w0 = mxfp4_to_float(low_nibble(byte0), scale_exp);
        float w1 = mxfp4_to_float(high_nibble(byte0), scale_exp);
        float w2 = mxfp4_to_float(low_nibble(byte1), scale_exp);
        float w3 = mxfp4_to_float(high_nibble(byte1), scale_exp);

        acc += i0 * w0 + i1 * w1 + i2 * w2 + i3 * w3;
    }

    // Handle remainder
    for (int k = K_aligned + lane_id; k < K; k += 32) {
        float in_val;
        if constexpr (std::is_same_v<T, half>) {
            in_val = __half2float(__ldg(&in_row[k]));
        } else {
            in_val = __bfloat162float(__ldg(&in_row[k]));
        }

        int byte_idx = k / 2;
        uint8_t packed = __ldg(&w_row[byte_idx]);
        uint8_t nibble = (k & 1) ? high_nibble(packed) : low_nibble(packed);

        int scale_idx = k / MXFP4_BLOCK_SIZE;
        int8_t scale_exp = (int8_t)__ldg(&scale_row[scale_idx]);

        float w_val = mxfp4_to_float(nibble, scale_exp);
        acc += in_val * w_val;
    }

    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 writes the result
    if (lane_id == 0) {
        // Add bias if present
        if (has_bias && biases != nullptr) {
            const T *bias_row = biases + (size_t)expert_idx * N;
            if constexpr (std::is_same_v<T, half>) {
                acc += __half2float(__ldg(&bias_row[n_idx]));
            } else {
                acc += __bfloat162float(__ldg(&bias_row[n_idx]));
            }
        }

        size_t out_idx = (size_t)token_idx * topk * N + (size_t)expert_slot * N + n_idx;
        if constexpr (std::is_same_v<T, half>) {
            output[out_idx] = __float2half(acc);
        } else {
            output[out_idx] = __float2bfloat16(acc);
        }
    }
}

} // namespace mxfp4_gemm

// ============================================================================
// C API
// ============================================================================

extern "C" void launch_mxfp4_matmul_f16(
    const __half *input,
    const uint8_t *weight,
    const uint8_t *weight_scale,
    const __half *bias,
    __half *output,
    int M, int N, int K,
    bool has_bias,
    cudaStream_t stream
) {
    constexpr int TILE = 32;
    constexpr int TILE_K = 32;

    dim3 block(TILE, TILE);
    dim3 grid(CEILDIV(N, TILE), CEILDIV(M, TILE));

    mxfp4_gemm::mxfp4_matmul_tiled<half, TILE, TILE, TILE_K>
        <<<grid, block, 0, stream>>>(
        input, weight, weight_scale, bias, output,
        M, N, K, has_bias
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_mxfp4_matmul_bf16(
    const __nv_bfloat16 *input,
    const uint8_t *weight,
    const uint8_t *weight_scale,
    const __nv_bfloat16 *bias,
    __nv_bfloat16 *output,
    int M, int N, int K,
    bool has_bias,
    cudaStream_t stream
) {
    constexpr int TILE = 32;
    constexpr int TILE_K = 32;

    dim3 block(TILE, TILE);
    dim3 grid(CEILDIV(N, TILE), CEILDIV(M, TILE));

    mxfp4_gemm::mxfp4_matmul_tiled<__nv_bfloat16, TILE, TILE, TILE_K>
        <<<grid, block, 0, stream>>>(
        input, weight, weight_scale, bias, output,
        M, N, K, has_bias
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_mxfp4_indexed_moe_gemm_f16(
    const __half *input,
    const uint8_t *weights,
    const uint8_t *weight_scales,
    const __half *biases,
    const uint32_t *indices,
    __half *output,
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    bool has_bias,
    bool input_has_topk_dim,
    cudaStream_t stream
) {
    constexpr int THREADS_PER_BLOCK = 512;
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;

    int total_outputs = num_tokens * topk * N;
    int total_warps = total_outputs;
    int num_blocks = CEILDIV(total_warps, WARPS_PER_BLOCK);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(num_blocks);

    mxfp4_gemm::mxfp4_moe_gemm<half><<<grid, block, 0, stream>>>(
        input, weights, weight_scales, biases, indices, output,
        num_tokens, topk, num_experts, N, K,
        has_bias, input_has_topk_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_mxfp4_indexed_moe_gemm_bf16(
    const __nv_bfloat16 *input,
    const uint8_t *weights,
    const uint8_t *weight_scales,
    const __nv_bfloat16 *biases,
    const uint32_t *indices,
    __nv_bfloat16 *output,
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    bool has_bias,
    bool input_has_topk_dim,
    cudaStream_t stream
) {
    constexpr int THREADS_PER_BLOCK = 512;
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;

    int total_outputs = num_tokens * topk * N;
    int total_warps = total_outputs;
    int num_blocks = CEILDIV(total_warps, WARPS_PER_BLOCK);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(num_blocks);

    mxfp4_gemm::mxfp4_moe_gemm<__nv_bfloat16><<<grid, block, 0, stream>>>(
        input, weights, weight_scales, biases, indices, output,
        num_tokens, topk, num_experts, N, K,
        has_bias, input_has_topk_dim
    );
    CUDA_CHECK(cudaGetLastError());
}
