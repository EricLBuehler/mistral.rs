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
#include <cuda_fp16.h>
#ifndef NO_BF16_KERNEL
#include <cuda_bf16.h>
#else
#ifndef __CUDA_BF16_TYPES_EXIST__
struct __nv_bfloat16 { uint16_t x; };
#endif
__device__ __forceinline__ float __bfloat162float(__nv_bfloat16 x) { (void)x; return 0.0f; }
__device__ __forceinline__ __nv_bfloat16 __float2bfloat16(float x) { (void)x; __nv_bfloat16 res; res.x = 0; return res; }
#endif
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
#define WARP_SIZE 32

namespace mxfp4_gemm {

// FP4 E2M1 lookup table
__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
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
// table32: LUT packed as 4 uint32 (16 int8 values)
__device__ __forceinline__ int2 get_int_from_table_16(
    const int q4,
    const uint32_t table0, const uint32_t table1,
    const uint32_t table2, const uint32_t table3
) {
    uint32_t tmp[2];
    const uint32_t low_high_selection = 0x32103210 | ((q4 & 0x88888888) >> 1);

    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(table0, table1, q4 >> shift);
        const uint32_t high = __byte_perm(table2, table3, q4 >> shift);
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
// Optimized MXFP4 MoE GEMM Kernel
// Key optimizations:
// 1. Process 32 elements per iteration (matches scale block size)
// 2. Inline LUT in registers (no constant memory latency)
// 3. Dual accumulators to hide FMA latency
// 4. Vectorized uint4 weight loads (16 bytes = 32 weights)
// ============================================================================

template <typename T>
__launch_bounds__(MOE_BLOCK_N * WARP_SIZE)
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
    // Pad shared input by 1 element per 32 to avoid worst-case bank conflicts when each lane
    // reads a contiguous 32-float segment with a stride of 32 between lanes.
    extern __shared__ float s_input_padded[];

    // LUT packed in registers (values scaled by 2x, divide by 2 at end)
    const uint32_t LUT0 = 0x03020100;  // 0, 1, 2, 3
    const uint32_t LUT1 = 0x0C080604;  // 4, 6, 8, 12
    const uint32_t LUT2 = 0xFDFEFF00;  // 0, -1, -2, -3
    const uint32_t LUT3 = 0xF4F8FAFC;  // -4, -6, -8, -12

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

    // Load input to shared memory
    for (int k = tid; k < K; k += block_size) {
        const int smem_idx = k + (k / WARP_SIZE);
        if constexpr (std::is_same_v<T, half>) {
            s_input_padded[smem_idx] = __half2float(__ldg(&in_row[k]));
        } else {
            s_input_padded[smem_idx] = __bfloat162float(__ldg(&in_row[k]));
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

        // Dual accumulators to hide FMA latency
        float acc0 = 0.0f;
        float acc1 = 0.0f;

        // Process 32 elements per iteration (one full scale block)
        // Each lane handles K/32 iterations with stride 1024
        for (int k = lane_id * 32; k < K; k += 32 * 32) {
            // Load scale once for entire 32-element block
            float w_scale =
                e8m0_to_float(__ldg(&w_scale_row[k / MXFP4_BLOCK_SIZE])) * 0.5f;

            // Load 16 bytes = 32 nibbles = 32 weights
            uint4 w_vec = *reinterpret_cast<const uint4*>(w_row + k / 2);
            const float* in = s_input_padded + (k + (k / WARP_SIZE));

            // Process first 8 elements (w_vec.x)
            {
                int2 w_int8 = get_int_from_table_16(w_vec.x, LUT0, LUT1, LUT2, LUT3);
                const int w_even = w_int8.x;
                const int w_odd = w_int8.y;

                acc0 = fmaf(in[0], (float)(int8_t)(w_even) * w_scale, acc0);
                acc1 = fmaf(in[1], (float)(int8_t)(w_odd) * w_scale, acc1);
                acc0 = fmaf(in[2], (float)(int8_t)(w_even >> 8) * w_scale, acc0);
                acc1 = fmaf(in[3], (float)(int8_t)(w_odd >> 8) * w_scale, acc1);
                acc0 = fmaf(in[4], (float)(int8_t)(w_even >> 16) * w_scale, acc0);
                acc1 = fmaf(in[5], (float)(int8_t)(w_odd >> 16) * w_scale, acc1);
                acc0 = fmaf(in[6], (float)(int8_t)(w_even >> 24) * w_scale, acc0);
                acc1 = fmaf(in[7], (float)(int8_t)(w_odd >> 24) * w_scale, acc1);
            }

            // Process second 8 elements (w_vec.y)
            {
                int2 w_int8 = get_int_from_table_16(w_vec.y, LUT0, LUT1, LUT2, LUT3);
                const int w_even = w_int8.x;
                const int w_odd = w_int8.y;

                acc0 = fmaf(in[8], (float)(int8_t)(w_even) * w_scale, acc0);
                acc1 = fmaf(in[9], (float)(int8_t)(w_odd) * w_scale, acc1);
                acc0 = fmaf(in[10], (float)(int8_t)(w_even >> 8) * w_scale, acc0);
                acc1 = fmaf(in[11], (float)(int8_t)(w_odd >> 8) * w_scale, acc1);
                acc0 = fmaf(in[12], (float)(int8_t)(w_even >> 16) * w_scale, acc0);
                acc1 = fmaf(in[13], (float)(int8_t)(w_odd >> 16) * w_scale, acc1);
                acc0 = fmaf(in[14], (float)(int8_t)(w_even >> 24) * w_scale, acc0);
                acc1 = fmaf(in[15], (float)(int8_t)(w_odd >> 24) * w_scale, acc1);
            }

            // Process third 8 elements (w_vec.z)
            {
                int2 w_int8 = get_int_from_table_16(w_vec.z, LUT0, LUT1, LUT2, LUT3);
                const int w_even = w_int8.x;
                const int w_odd = w_int8.y;

                acc0 = fmaf(in[16], (float)(int8_t)(w_even) * w_scale, acc0);
                acc1 = fmaf(in[17], (float)(int8_t)(w_odd) * w_scale, acc1);
                acc0 = fmaf(in[18], (float)(int8_t)(w_even >> 8) * w_scale, acc0);
                acc1 = fmaf(in[19], (float)(int8_t)(w_odd >> 8) * w_scale, acc1);
                acc0 = fmaf(in[20], (float)(int8_t)(w_even >> 16) * w_scale, acc0);
                acc1 = fmaf(in[21], (float)(int8_t)(w_odd >> 16) * w_scale, acc1);
                acc0 = fmaf(in[22], (float)(int8_t)(w_even >> 24) * w_scale, acc0);
                acc1 = fmaf(in[23], (float)(int8_t)(w_odd >> 24) * w_scale, acc1);
            }

            // Process fourth 8 elements (w_vec.w)
            {
                int2 w_int8 = get_int_from_table_16(w_vec.w, LUT0, LUT1, LUT2, LUT3);
                const int w_even = w_int8.x;
                const int w_odd = w_int8.y;

                acc0 = fmaf(in[24], (float)(int8_t)(w_even) * w_scale, acc0);
                acc1 = fmaf(in[25], (float)(int8_t)(w_odd) * w_scale, acc1);
                acc0 = fmaf(in[26], (float)(int8_t)(w_even >> 8) * w_scale, acc0);
                acc1 = fmaf(in[27], (float)(int8_t)(w_odd >> 8) * w_scale, acc1);
                acc0 = fmaf(in[28], (float)(int8_t)(w_even >> 16) * w_scale, acc0);
                acc1 = fmaf(in[29], (float)(int8_t)(w_odd >> 16) * w_scale, acc1);
                acc0 = fmaf(in[30], (float)(int8_t)(w_even >> 24) * w_scale, acc0);
                acc1 = fmaf(in[31], (float)(int8_t)(w_odd >> 24) * w_scale, acc1);
            }
        }

        // Merge accumulators
        float acc = acc0 + acc1;

        // Warp reduction
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

#ifndef NO_BF16_KERNEL
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
#else
extern "C" void launch_mxfp4_matmul_bf16(
    const void *input,
    const uint8_t *weight,
    const uint8_t *weight_scale,
    const void *bias,
    void *output,
    int M, int N, int K,
    bool has_bias,
    cudaStream_t stream
) {
    (void)input; (void)weight; (void)weight_scale; (void)bias; (void)output; (void)M; (void)N; (void)K; (void)has_bias; (void)stream;
    fprintf(stderr, "ERROR: launch_mxfp4_matmul_bf16 requires BF16 support (SM 8.0+)\n");
}
#endif

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
    size_t shared_mem_size = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);

    mxfp4_gemm::mxfp4_moe_gemm<half><<<grid, block, shared_mem_size, stream>>>(
        input, weights, weight_scales, biases, indices, output,
        num_tokens, topk, num_experts, N, K,
        has_bias, input_has_topk_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

#ifndef NO_BF16_KERNEL
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
    size_t shared_mem_size = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);

    mxfp4_gemm::mxfp4_moe_gemm<__nv_bfloat16><<<grid, block, shared_mem_size, stream>>>(
        input, weights, weight_scales, biases, indices, output,
        num_tokens, topk, num_experts, N, K,
        has_bias, input_has_topk_dim
    );
    CUDA_CHECK(cudaGetLastError());
}
#else
extern "C" void launch_mxfp4_indexed_moe_gemm_bf16(
    const void *input,
    const uint8_t *weights,
    const uint8_t *weight_scales,
    const void *biases,
    const uint32_t *indices,
    void *output,
    int num_tokens,
    int topk,
    int num_experts,
    int N, int K,
    bool has_bias,
    bool input_has_topk_dim,
    cudaStream_t stream
) {
    (void)input; (void)weights; (void)weight_scales; (void)biases; (void)indices; (void)output; (void)num_tokens; (void)topk; (void)num_experts; (void)N; (void)K; (void)has_bias; (void)input_has_topk_dim; (void)stream;
    fprintf(stderr, "ERROR: launch_mxfp4_indexed_moe_gemm_bf16 requires BF16 support (SM 8.0+)\n");
}
#endif
