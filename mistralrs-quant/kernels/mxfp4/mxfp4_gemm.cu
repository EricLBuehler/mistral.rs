/**
 * MXFP4 GEMM kernels with LUT-based dequantization.
 *
 * MXFP4 Format (OCP Microscaling):
 * - FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
 * - Block size: 32 elements per scale
 * - Scale: E8M0 format (8-bit exponent, stored as u8 with bias 127)
 * - 2 FP4 values packed per byte (nibbles)
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define MXFP4_BLOCK_SIZE 32
#define MOE_BLOCK_N 8

namespace mxfp4_gemm {

// FP4 E2M1 lookup table: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// INT8 lookup table for vectorized dequantization (values scaled by 2x)
__constant__ int8_t FP4_LUT_INT8[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};

// E8M0 to float: 2^(e - 127) via IEEE 754 bit manipulation
__device__ __forceinline__ float e8m0_to_float(uint8_t e) {
    return __uint_as_float((uint32_t)e << 23);
}

__device__ __forceinline__ uint8_t low_nibble(uint8_t byte) {
    return byte & 0x0F;
}

__device__ __forceinline__ uint8_t high_nibble(uint8_t byte) {
    return (byte >> 4) & 0x0F;
}

// Fast vectorized LUT lookup using __byte_perm instruction
// Converts 8 FP4 nibbles (packed in int32) to 8 int8 values
__device__ __forceinline__ int2 get_int_from_table_16(const int q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*)table;
    uint32_t tmp[2];
    const uint32_t low_high_selection = 0x32103210 | ((q4 & 0x88888888) >> 1);

    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection >> shift);
    }

    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// ============================================================================
// MXFP4 Matmul Kernel (for linear forward)
// ============================================================================

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void mxfp4_matmul_tiled(
    const T *__restrict__ input,
    const uint8_t *__restrict__ weight,
    const uint8_t *__restrict__ weight_scale,
    const T *__restrict__ bias,
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
    const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);

    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
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

        for (int i = tid; i < BLOCK_N * BLOCK_K; i += num_threads) {
            int ln = i / BLOCK_K;
            int lk = i % BLOCK_K;
            int gn = bx * BLOCK_N + ln;
            int gk = k_tile + lk;

            float val = 0.0f;
            if (gn < N && gk < K) {
                int byte_idx = gk / 2;
                uint8_t packed = __ldg(&weight[gn * (K / 2) + byte_idx]);
                int scale_idx = gk / MXFP4_BLOCK_SIZE;
                float scale = e8m0_to_float(__ldg(&weight_scale[gn * scale_stride + scale_idx]));
                uint8_t nibble = (gk & 1) ? high_nibble(packed) : low_nibble(packed);
                val = FP4_LUT[nibble] * scale;
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
// MXFP4 MoE GEMM Kernel (vectorized with __byte_perm LUT lookup)
// ============================================================================

template <typename T>
__global__ void mxfp4_moe_gemm(
    const T *__restrict__ input,
    const uint8_t *__restrict__ weights,
    const uint8_t *__restrict__ weight_scales,
    const T *__restrict__ biases,
    const uint32_t *__restrict__ indices,
    T *__restrict__ output,
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    bool has_bias,
    bool input_has_topk_dim
) {
    extern __shared__ float s_input[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int n_chunks = CEILDIV(N, MOE_BLOCK_N);
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int weight_row_stride = K / 2;
    const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);

    int token_idx, expert_slot_start, expert_slot_end, n_base;

    if (!input_has_topk_dim) {
        n_base = (blockIdx.x % n_chunks) * MOE_BLOCK_N;
        token_idx = blockIdx.x / n_chunks;
        expert_slot_start = 0;
        expert_slot_end = topk;
    } else {
        n_base = (blockIdx.x % n_chunks) * MOE_BLOCK_N;
        int temp = blockIdx.x / n_chunks;
        expert_slot_start = temp % topk;
        expert_slot_end = expert_slot_start + 1;
        token_idx = temp / topk;
    }

    if (token_idx >= num_tokens) return;

    const int n_idx = n_base + warp_id;
    if (n_idx >= N) return;

    const T *in_row;
    if (!input_has_topk_dim) {
        in_row = input + (size_t)token_idx * K;
    } else {
        in_row = input + (size_t)token_idx * topk * K + (size_t)expert_slot_start * K;
    }

    for (int k = tid; k < K; k += block_size) {
        if constexpr (std::is_same_v<T, half>) {
            s_input[k] = __half2float(in_row[k]);
        } else {
            s_input[k] = __bfloat162float(in_row[k]);
        }
    }
    __syncthreads();

    for (int expert_slot = expert_slot_start; expert_slot < expert_slot_end; expert_slot++) {
        const uint32_t expert_idx = __ldg(&indices[token_idx * topk + expert_slot]);
        if (expert_idx >= (uint32_t)num_experts) continue;

        const uint8_t *w_row = weights + (size_t)expert_idx * N * weight_row_stride
                             + (size_t)n_idx * weight_row_stride;
        const uint8_t *w_scale_row = weight_scales + (size_t)expert_idx * N * scale_stride
                                   + (size_t)n_idx * scale_stride;

        float acc = 0.0f;

        // Process 8 elements per iteration using vectorized LUT lookup
        for (int k = lane_id * 8; k < K; k += 32 * 8) {
            int scale_idx = k / MXFP4_BLOCK_SIZE;
            float w_scale = e8m0_to_float(w_scale_row[scale_idx]) * 0.5f;

            uint32_t w_packed = *((uint32_t*)(w_row + k / 2));
            int2 w_int8 = get_int_from_table_16(w_packed, FP4_LUT_INT8);

            int8_t* w_even = (int8_t*)&w_int8.x;
            int8_t* w_odd = (int8_t*)&w_int8.y;

            acc = fmaf(s_input[k+0], (float)w_even[0] * w_scale, acc);
            acc = fmaf(s_input[k+1], (float)w_odd[0] * w_scale, acc);
            acc = fmaf(s_input[k+2], (float)w_even[1] * w_scale, acc);
            acc = fmaf(s_input[k+3], (float)w_odd[1] * w_scale, acc);
            acc = fmaf(s_input[k+4], (float)w_even[2] * w_scale, acc);
            acc = fmaf(s_input[k+5], (float)w_odd[2] * w_scale, acc);
            acc = fmaf(s_input[k+6], (float)w_even[3] * w_scale, acc);
            acc = fmaf(s_input[k+7], (float)w_odd[3] * w_scale, acc);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

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
    constexpr int THREADS_PER_BLOCK = MOE_BLOCK_N * 32;
    int n_chunks = CEILDIV(N, MOE_BLOCK_N);

    int total_blocks = input_has_topk_dim
        ? num_tokens * topk * n_chunks
        : num_tokens * n_chunks;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(total_blocks);
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
    constexpr int THREADS_PER_BLOCK = MOE_BLOCK_N * 32;
    int n_chunks = CEILDIV(N, MOE_BLOCK_N);

    int total_blocks = input_has_topk_dim
        ? num_tokens * topk * n_chunks
        : num_tokens * n_chunks;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(total_blocks);
    size_t shared_mem_size = K * sizeof(float);

    mxfp4_gemm::mxfp4_moe_gemm<__nv_bfloat16><<<grid, block, shared_mem_size, stream>>>(
        input, weights, weight_scales, biases, indices, output,
        num_tokens, topk, num_experts, N, K,
        has_bias, input_has_topk_dim
    );
    CUDA_CHECK(cudaGetLastError());
}
