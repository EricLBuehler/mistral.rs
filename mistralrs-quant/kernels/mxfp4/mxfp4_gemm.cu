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

// INT8 lookup table for dp4a optimization (values scaled by 2x to allow integer representation)
// Same as llama.cpp: {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
// Final result needs to be multiplied by 0.5 to compensate
__constant__ int8_t FP4_LUT_INT8[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};

// ============================================================================
// Helper functions
// ============================================================================

// Fast E8M0 to float conversion using bit manipulation
// E8M0 format: unsigned 8-bit exponent with bias 127
// Result: 2^(e - 127)
// This is ~2 cycles vs ~20 cycles for ldexpf
__device__ __forceinline__ float e8m0_to_float(uint8_t e) {
    // IEEE 754 float: sign(1) | exponent(8) | mantissa(23)
    // For 2^(e-127), we set exponent field = e, mantissa = 0
    // The IEEE bias is 127, so exponent field e gives 2^(e-127)
    return __uint_as_float((uint32_t)e << 23);
}

// Fast dequantization: LUT lookup + scale multiply (no ldexpf!)
__device__ __forceinline__ float mxfp4_to_float_fast(uint8_t nibble, float scale) {
    return FP4_LUT[nibble & 0xF] * scale;
}

// Legacy function for compatibility - AVOID using this, it's slow!
__device__ __forceinline__ float mxfp4_to_float(uint8_t nibble, uint8_t scale_exp) {
    return FP4_LUT[nibble & 0xF] * e8m0_to_float(scale_exp);
}

// Extract low nibble (bits 0-3)
__device__ __forceinline__ uint8_t low_nibble(uint8_t byte) {
    return byte & 0x0F;
}

// Extract high nibble (bits 4-7)
__device__ __forceinline__ uint8_t high_nibble(uint8_t byte) {
    return (byte >> 4) & 0x0F;
}

// Pack 4 int8 values into int32 for dp4a
__device__ __forceinline__ int pack_int8x4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (int)(uint8_t)a | ((int)(uint8_t)b << 8) | ((int)(uint8_t)c << 16) | ((int)(uint8_t)d << 24);
}

// dp4a: dot product of 4 int8 pairs + accumulate
// result = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + c
__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    // Fallback for older architectures
    int8_t *a8 = (int8_t*)&a;
    int8_t *b8 = (int8_t*)&b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
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

        // Load weight tile with dequantization (using fast e8m0_to_float)
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

                // Get scale for this block and convert to float ONCE (fast bit manipulation)
                int scale_idx = gk / MXFP4_BLOCK_SIZE;
                uint8_t scale_exp = __ldg(&weight_scale[gn * scale_stride + scale_idx]);
                float scale = e8m0_to_float(scale_exp);

                // Extract correct nibble and dequantize (no ldexpf!)
                uint8_t nibble = (gk & 1) ? high_nibble(packed) : low_nibble(packed);
                val = mxfp4_to_float_fast(nibble, scale);
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
// MXFP4 MoE GEMM - Unified Optimized Kernel
//
// Key optimizations:
// 1. Shared memory caching of input row (load ONCE, reuse for all experts)
// 2. Fast e8m0_to_float (bit manipulation instead of ldexpf)
// 3. LUT-based FP4 dequantization
// 4. Adaptive block organization based on input layout:
//    - input_has_topk_dim=false: Process ALL expert_slots per token in one block
//    - input_has_topk_dim=true: Process one expert_slot per block
// 5. Coalesced memory access patterns
// ============================================================================

// Number of N values processed per block (32 warps × 32 threads = 1024 threads max)
#define MOE_BLOCK_N 32

// Unified MXFP4 MoE GEMM Kernel
// When input_has_topk_dim=false: Each block handles ALL topk experts for one (token, n_chunk)
// When input_has_topk_dim=true: Each block handles ONE (token, expert_slot, n_chunk)
template <typename T>
__global__ void mxfp4_moe_gemm(
    const T *__restrict__ input,
    const uint8_t *__restrict__ weights,      // [num_experts, N, K/2]
    const uint8_t *__restrict__ weight_scales, // [num_experts, N, K/32]
    const T *__restrict__ biases,              // [num_experts, N] or nullptr
    const uint32_t *__restrict__ indices,      // [num_tokens, topk]
    T *__restrict__ output,                    // [num_tokens, topk, N]
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    bool has_bias,
    bool input_has_topk_dim
) {
    // Shared memory for cached input row (as floats)
    extern __shared__ float s_input[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int n_chunks = CEILDIV(N, MOE_BLOCK_N);
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int weight_row_stride = K / 2;
    const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);

    // Decode block indices based on input layout
    int token_idx, expert_slot_start, expert_slot_end, n_base;

    if (!input_has_topk_dim) {
        // Block layout: (token_idx, n_chunk) - process ALL experts per block
        n_base = (blockIdx.x % n_chunks) * MOE_BLOCK_N;
        token_idx = blockIdx.x / n_chunks;
        expert_slot_start = 0;
        expert_slot_end = topk;
    } else {
        // Block layout: (token_idx, expert_slot, n_chunk) - one expert per block
        n_base = (blockIdx.x % n_chunks) * MOE_BLOCK_N;
        int temp = blockIdx.x / n_chunks;
        expert_slot_start = temp % topk;
        expert_slot_end = expert_slot_start + 1;
        token_idx = temp / topk;
    }

    if (token_idx >= num_tokens) return;

    const int n_idx = n_base + warp_id;
    if (n_idx >= N) return;

    // Get input row pointer
    const T *in_row;
    if (!input_has_topk_dim) {
        in_row = input + (size_t)token_idx * K;
    } else {
        in_row = input + (size_t)token_idx * topk * K + (size_t)expert_slot_start * K;
    }

    // Load input row to shared memory (coalesced load)
    for (int k = tid; k < K; k += block_size) {
        if constexpr (std::is_same_v<T, half>) {
            s_input[k] = __half2float(__ldg(&in_row[k]));
        } else {
            s_input[k] = __bfloat162float(__ldg(&in_row[k]));
        }
    }
    __syncthreads();

    // Process expert_slots
    for (int expert_slot = expert_slot_start; expert_slot < expert_slot_end; expert_slot++) {
        const uint32_t expert_idx = __ldg(&indices[token_idx * topk + expert_slot]);
        if (expert_idx >= (uint32_t)num_experts) continue;

        const uint8_t *w_row = weights + (size_t)expert_idx * N * weight_row_stride
                             + (size_t)n_idx * weight_row_stride;
        const uint8_t *w_scale_row = weight_scales + (size_t)expert_idx * N * scale_stride
                                   + (size_t)n_idx * scale_stride;

        // Compute dot product: input[K] · weight[K] using warp-level parallelism
        float acc = 0.0f;

        // Each lane processes K/32 elements
        for (int k = lane_id; k < K; k += 32) {
            float in_val = s_input[k];

            // Dequantize MXFP4 weight
            int byte_idx = k / 2;
            uint8_t packed = __ldg(&w_row[byte_idx]);
            uint8_t nibble = (k & 1) ? high_nibble(packed) : low_nibble(packed);

            int scale_idx = k / MXFP4_BLOCK_SIZE;
            float w_scale = e8m0_to_float(__ldg(&w_scale_row[scale_idx]));
            float w_val = FP4_LUT[nibble] * w_scale;

            acc = fmaf(in_val, w_val, acc);
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        // Lane 0 writes result
        if (lane_id == 0) {
            if (has_bias && biases) {
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
    constexpr int THREADS_PER_BLOCK = MOE_BLOCK_N * 32;  // 32 warps = 1024 threads
    int n_chunks = CEILDIV(N, MOE_BLOCK_N);

    // Adaptive grid dimensions based on input layout:
    // - input_has_topk_dim=false: (token, n_chunk) - processes all experts per block
    // - input_has_topk_dim=true: (token, expert_slot, n_chunk) - one expert per block
    int total_blocks = input_has_topk_dim
        ? num_tokens * topk * n_chunks
        : num_tokens * n_chunks;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(total_blocks);

    // Shared memory: K floats for input caching
    size_t shared_mem_size = K * sizeof(float);

    mxfp4_gemm::mxfp4_moe_gemm<half><<<grid, block, shared_mem_size, stream>>>(
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
    constexpr int THREADS_PER_BLOCK = MOE_BLOCK_N * 32;  // 32 warps = 1024 threads
    int n_chunks = CEILDIV(N, MOE_BLOCK_N);

    // Adaptive grid dimensions based on input layout
    int total_blocks = input_has_topk_dim
        ? num_tokens * topk * n_chunks
        : num_tokens * n_chunks;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(total_blocks);

    // Shared memory: K floats for input caching
    size_t shared_mem_size = K * sizeof(float);

    mxfp4_gemm::mxfp4_moe_gemm<__nv_bfloat16><<<grid, block, shared_mem_size, stream>>>(
        input, weights, weight_scales, biases, indices, output,
        num_tokens, topk, num_experts, N, K,
        has_bias, input_has_topk_dim
    );
    CUDA_CHECK(cudaGetLastError());
}
