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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
__constant__ float FP4_LUT[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                                  4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                                  -2.0f, -3.0f, -4.0f, -6.0f};

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
__device__ __forceinline__ int2 get_int_from_table_16(const int q4,
                                                      const uint32_t table0,
                                                      const uint32_t table1,
                                                      const uint32_t table2,
                                                      const uint32_t table3) {
  uint32_t tmp[2];
  const uint32_t low_high_selection = 0x32103210 | ((q4 & 0x88888888) >> 1);

#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t shift = 16 * i;
    const uint32_t low = __byte_perm(table0, table1, q4 >> shift);
    const uint32_t high = __byte_perm(table2, table3, q4 >> shift);
    tmp[i] = __byte_perm(low, high, low_high_selection >> shift);
  }

  return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420),
                   __byte_perm(tmp[0], tmp[1], 0x7531));
}

// ============================================================================
// Dequantize 8 FP4 values from one int32 (8 nibbles) using LUT and store
// ============================================================================

__device__ __forceinline__ void dequant_store_8(int q4, float scale,
                                                uint32_t LUT0, uint32_t LUT1,
                                                uint32_t LUT2, uint32_t LUT3,
                                                float *dst) {
  int2 w = get_int_from_table_16(q4, LUT0, LUT1, LUT2, LUT3);
  dst[0] = (float)(int8_t)(w.x) * scale;
  dst[1] = (float)(int8_t)(w.y) * scale;
  dst[2] = (float)(int8_t)(w.x >> 8) * scale;
  dst[3] = (float)(int8_t)(w.y >> 8) * scale;
  dst[4] = (float)(int8_t)(w.x >> 16) * scale;
  dst[5] = (float)(int8_t)(w.y >> 16) * scale;
  dst[6] = (float)(int8_t)(w.x >> 24) * scale;
  dst[7] = (float)(int8_t)(w.y >> 24) * scale;
}

// ============================================================================
// Optimized MXFP4 Matmul Kernel (for linear forward)
// Key optimizations over the original:
// 1. Vectorized uint4 weight loads (32 FP4 values per load, was 1 nibble)
// 2. Register-level tiling: each thread computes TM x TN outputs
// 3. LUT-based dequantization in registers (no constant memory latency)
// 4. fmaf for fused multiply-add
// ============================================================================

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TM, int TN>
__global__ void mxfp4_matmul_tiled(const T *__restrict__ input,
                                   const uint8_t *__restrict__ weight,
                                   const uint8_t *__restrict__ weight_scale,
                                   const T *__restrict__ bias,
                                   T *__restrict__ output, int M, int N, int K,
                                   bool has_bias) {
  constexpr int THREADS_N = BLOCK_N / TN;
  constexpr int THREADS_M = BLOCK_M / TM;
  constexpr int NUM_THREADS = THREADS_N * THREADS_M;
  // +1 padding reduces shared memory bank conflicts
  constexpr int BK_PAD = BLOCK_K + 1;

  __shared__ float s_input[BLOCK_M][BK_PAD];
  __shared__ float s_weight[BLOCK_N][BK_PAD];

  // LUT in registers (values 2x scaled, compensated by 0.5f in scale)
  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int tid = threadIdx.y * THREADS_N + threadIdx.x;
  const int bx = blockIdx.x; // N dimension
  const int by = blockIdx.y; // M dimension
  const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);

  // Initialize TM x TN accumulators per thread
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++)
      acc[i][j] = 0.0f;

  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    // === Load input tile [BLOCK_M x BLOCK_K] to shared memory ===
    for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += NUM_THREADS) {
      const int lm = idx / BLOCK_K;
      const int lk = idx % BLOCK_K;
      const int gm = by * BLOCK_M + lm;
      const int gk = k_tile + lk;

      float val = 0.0f;
      if (gm < M && gk < K) {
        if constexpr (std::is_same_v<T, half>) {
          val = __half2float(__ldg(&input[(size_t)gm * K + gk]));
        } else {
          val = __bfloat162float(__ldg(&input[(size_t)gm * K + gk]));
        }
      }
      s_input[lm][lk] = val;
    }

    // === Load weight tile [BLOCK_N x BLOCK_K] with vectorized dequant ===
    // BLOCK_K = 32 = one MXFP4 block, so one uint4 load per row
    for (int ln = tid; ln < BLOCK_N; ln += NUM_THREADS) {
      const int gn = bx * BLOCK_N + ln;
      if (gn < N) {
        // One uint4 = 16 bytes = 32 packed FP4 values = one full MXFP4 block
        uint4 w_vec = *reinterpret_cast<const uint4 *>(
            &weight[(size_t)gn * (K / 2) + k_tile / 2]);
        float scale =
            e8m0_to_float(__ldg(&weight_scale[(size_t)gn * scale_stride +
                                              k_tile / MXFP4_BLOCK_SIZE])) *
            0.5f;

        dequant_store_8(w_vec.x, scale, LUT0, LUT1, LUT2, LUT3,
                        &s_weight[ln][0]);
        dequant_store_8(w_vec.y, scale, LUT0, LUT1, LUT2, LUT3,
                        &s_weight[ln][8]);
        dequant_store_8(w_vec.z, scale, LUT0, LUT1, LUT2, LUT3,
                        &s_weight[ln][16]);
        dequant_store_8(w_vec.w, scale, LUT0, LUT1, LUT2, LUT3,
                        &s_weight[ln][24]);
      } else {
        // Zero out invalid rows
#pragma unroll
        for (int k = 0; k < BLOCK_K; k++)
          s_weight[ln][k] = 0.0f;
      }
    }

    __syncthreads();

    // === Register-tiled GEMM with outer product ===
#pragma unroll
    for (int k = 0; k < BLOCK_K; k++) {
      // Load input and weight fragments into registers
      float a_frag[TM];
      float b_frag[TN];

#pragma unroll
      for (int i = 0; i < TM; i++)
        a_frag[i] = s_input[threadIdx.y * TM + i][k];

#pragma unroll
      for (int j = 0; j < TN; j++)
        b_frag[j] = s_weight[threadIdx.x * TN + j][k];

      // Outer product accumulation
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
    }

    __syncthreads();
  }

  // === Write output with bias ===
#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int row = by * BLOCK_M + threadIdx.y * TM + i;
    if (row < M) {
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const int col = bx * BLOCK_N + threadIdx.x * TN + j;
        if (col < N) {
          float val = acc[i][j];
          if (has_bias && bias != nullptr) {
            if constexpr (std::is_same_v<T, half>) {
              val += __half2float(__ldg(&bias[col]));
            } else {
              val += __bfloat162float(__ldg(&bias[col]));
            }
          }
          if constexpr (std::is_same_v<T, half>) {
            output[(size_t)row * N + col] = __float2half(val);
          } else {
            output[(size_t)row * N + col] = __float2bfloat16(val);
          }
        }
      }
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
__launch_bounds__(MOE_BLOCK_N *WARP_SIZE) __global__
    void mxfp4_moe_gemm(const T *__restrict__ input,
                        const uint8_t *__restrict__ weights,
                        const uint8_t *__restrict__ weight_scales,
                        const T *__restrict__ biases,
                        const uint32_t *__restrict__ indices,
                        T *__restrict__ output, int num_tokens, int topk,
                        int num_experts, int N, int K, bool has_bias,
                        bool input_has_topk_dim) {
  // Pad shared input by 1 element per 32 to avoid worst-case bank conflicts
  // when each lane reads a contiguous 32-float segment with a stride of 32
  // between lanes.
  extern __shared__ float s_input_padded[];

  // LUT packed in registers (values scaled by 2x, divide by 2 at end)
  const uint32_t LUT0 = 0x03020100; // 0, 1, 2, 3
  const uint32_t LUT1 = 0x0C080604; // 4, 6, 8, 12
  const uint32_t LUT2 = 0xFDFEFF00; // 0, -1, -2, -3
  const uint32_t LUT3 = 0xF4F8FAFC; // -4, -6, -8, -12

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

  if (token_idx >= num_tokens)
    return;

  const int n_idx = n_base + warp_id;
  if (n_idx >= N)
    return;

  const T *in_row;
  if (!input_has_topk_dim) {
    in_row = input + (size_t)token_idx * K;
  } else {
    in_row =
        input + (size_t)token_idx * topk * K + (size_t)expert_slot_start * K;
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

  for (int expert_slot = expert_slot_start; expert_slot < expert_slot_end;
       expert_slot++) {
    const uint32_t expert_idx = __ldg(&indices[token_idx * topk + expert_slot]);
    if (expert_idx >= (uint32_t)num_experts)
      continue;

    const uint8_t *w_row = weights +
                           (size_t)expert_idx * N * weight_row_stride +
                           (size_t)n_idx * weight_row_stride;
    const uint8_t *w_scale_row = weight_scales +
                                 (size_t)expert_idx * N * scale_stride +
                                 (size_t)n_idx * scale_stride;

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
      uint4 w_vec = *reinterpret_cast<const uint4 *>(w_row + k / 2);
      const float *in = s_input_padded + (k + (k / WARP_SIZE));

      // Process first 8 elements (w_vec.x)
      {
        int2 w_int8 = get_int_from_table_16(w_vec.x, LUT0, LUT1, LUT2, LUT3);
        const int w_even = w_int8.x;
        const int w_odd = w_int8.y;

        acc0 = fmaf(in[0], (float)(int8_t)(w_even)*w_scale, acc0);
        acc1 = fmaf(in[1], (float)(int8_t)(w_odd)*w_scale, acc1);
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

        acc0 = fmaf(in[8], (float)(int8_t)(w_even)*w_scale, acc0);
        acc1 = fmaf(in[9], (float)(int8_t)(w_odd)*w_scale, acc1);
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

        acc0 = fmaf(in[16], (float)(int8_t)(w_even)*w_scale, acc0);
        acc1 = fmaf(in[17], (float)(int8_t)(w_odd)*w_scale, acc1);
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

        acc0 = fmaf(in[24], (float)(int8_t)(w_even)*w_scale, acc0);
        acc1 = fmaf(in[25], (float)(int8_t)(w_odd)*w_scale, acc1);
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

      size_t out_idx =
          (size_t)token_idx * topk * N + (size_t)expert_slot * N + n_idx;
      if constexpr (std::is_same_v<T, half>) {
        output[out_idx] = __float2half(acc);
      } else {
        output[out_idx] = __float2bfloat16(acc);
      }
    }
  }
}

// ============================================================================
// Fused MoE Grouped GEMM Kernel
// Discovers tokens per expert on-GPU, then does tiled GEMM with indirect I/O.
// Grid: (CEILDIV(N, BLOCK_N), num_experts)
// Phase 1: all threads scan indices to build per-expert token list in smem
// Phase 2: outer M-tile loop with register-tiled GEMM, indirect reads/writes
// ============================================================================

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TM, int TN>
__global__ void mxfp4_moe_grouped_gemm_tiled(
    const T *__restrict__ input, const uint8_t *__restrict__ weights,
    const uint8_t *__restrict__ weight_scales, const T *__restrict__ biases,
    const uint32_t *__restrict__ indices, T *__restrict__ output,
    int num_tokens, int topk, int num_experts, int N, int K, bool has_bias,
    bool input_has_topk_dim) {
  constexpr int THREADS_N = BLOCK_N / TN;
  constexpr int THREADS_M = BLOCK_M / TM;
  constexpr int NUM_THREADS = THREADS_N * THREADS_M;
  constexpr int BK_PAD = BLOCK_K + 1;

  __shared__ float s_input[BLOCK_M][BK_PAD];
  __shared__ float s_weight[BLOCK_N][BK_PAD];
  __shared__ int num_my_tokens;
  extern __shared__ int s_token_list[];

  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int tid = threadIdx.y * THREADS_N + threadIdx.x;
  const int expert_id = blockIdx.y;
  const int n_base = blockIdx.x * BLOCK_N;
  const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);
  const int total_work = num_tokens * topk;

  const uint8_t *expert_weight = weights + (size_t)expert_id * N * (K / 2);
  const uint8_t *expert_scale =
      weight_scales + (size_t)expert_id * N * scale_stride;

  // === Phase 1: Token discovery ===
  if (tid == 0)
    num_my_tokens = 0;
  __syncthreads();

  for (int i = tid; i < total_work; i += NUM_THREADS) {
    if (__ldg(&indices[i]) == (uint32_t)expert_id) {
      int pos = atomicAdd(&num_my_tokens, 1);
      s_token_list[pos] = i;
    }
  }
  __syncthreads();

  const int M_expert = num_my_tokens;
  if (M_expert == 0)
    return;

  // === Phase 2: Tiled GEMM with indirect I/O ===
  for (int m_tile = 0; m_tile < CEILDIV(M_expert, BLOCK_M); m_tile++) {
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++)
        acc[i][j] = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
      // Load input tile via indirection
      for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += NUM_THREADS) {
        const int lm = idx / BLOCK_K;
        const int lk = idx % BLOCK_K;
        const int work_pos = m_tile * BLOCK_M + lm;
        const int gk = k_tile + lk;

        float val = 0.0f;
        if (work_pos < M_expert && gk < K) {
          const int work_idx = s_token_list[work_pos];
          const int input_row =
              input_has_topk_dim ? work_idx : (work_idx / topk);
          if constexpr (std::is_same_v<T, half>) {
            val = __half2float(__ldg(&input[(size_t)input_row * K + gk]));
          } else {
            val = __bfloat162float(__ldg(&input[(size_t)input_row * K + gk]));
          }
        }
        s_input[lm][lk] = val;
      }

      // Load weight tile with vectorized dequant
      for (int ln = tid; ln < BLOCK_N; ln += NUM_THREADS) {
        const int gn = n_base + ln;
        if (gn < N) {
          uint4 w_vec = *reinterpret_cast<const uint4 *>(
              &expert_weight[(size_t)gn * (K / 2) + k_tile / 2]);
          float scale =
              e8m0_to_float(__ldg(&expert_scale[(size_t)gn * scale_stride +
                                                k_tile / MXFP4_BLOCK_SIZE])) *
              0.5f;
          dequant_store_8(w_vec.x, scale, LUT0, LUT1, LUT2, LUT3,
                          &s_weight[ln][0]);
          dequant_store_8(w_vec.y, scale, LUT0, LUT1, LUT2, LUT3,
                          &s_weight[ln][8]);
          dequant_store_8(w_vec.z, scale, LUT0, LUT1, LUT2, LUT3,
                          &s_weight[ln][16]);
          dequant_store_8(w_vec.w, scale, LUT0, LUT1, LUT2, LUT3,
                          &s_weight[ln][24]);
        } else {
#pragma unroll
          for (int k = 0; k < BLOCK_K; k++)
            s_weight[ln][k] = 0.0f;
        }
      }

      __syncthreads();

#pragma unroll
      for (int k = 0; k < BLOCK_K; k++) {
        float a_frag[TM];
        float b_frag[TN];
#pragma unroll
        for (int i = 0; i < TM; i++)
          a_frag[i] = s_input[threadIdx.y * TM + i][k];
#pragma unroll
        for (int j = 0; j < TN; j++)
          b_frag[j] = s_weight[threadIdx.x * TN + j][k];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
          for (int j = 0; j < TN; j++)
            acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
      }

      __syncthreads();
    } // end k_tile

    // Write output via indirection
#pragma unroll
    for (int i = 0; i < TM; i++) {
      const int work_pos = m_tile * BLOCK_M + threadIdx.y * TM + i;
      if (work_pos < M_expert) {
        const int work_idx = s_token_list[work_pos];
#pragma unroll
        for (int j = 0; j < TN; j++) {
          const int col = n_base + threadIdx.x * TN + j;
          if (col < N) {
            float val = acc[i][j];
            if (has_bias && biases != nullptr) {
              if constexpr (std::is_same_v<T, half>) {
                val +=
                    __half2float(__ldg(&biases[(size_t)expert_id * N + col]));
              } else {
                val += __bfloat162float(
                    __ldg(&biases[(size_t)expert_id * N + col]));
              }
            }
            if constexpr (std::is_same_v<T, half>) {
              output[(size_t)work_idx * N + col] = __float2half(val);
            } else {
              output[(size_t)work_idx * N + col] = __float2bfloat16(val);
            }
          }
        }
      }
    }
  } // end m_tile
}

} // namespace mxfp4_gemm

// ============================================================================
// C API
// ============================================================================

extern "C" void launch_mxfp4_matmul_f16(const __half *input,
                                        const uint8_t *weight,
                                        const uint8_t *weight_scale,
                                        const __half *bias, __half *output,
                                        int M, int N, int K, bool has_bias,
                                        cudaStream_t stream) {
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
  constexpr int THREADS_N = BN / TN; // 16
  constexpr int THREADS_M = BM / TM; // 16

  dim3 block(THREADS_N, THREADS_M);
  dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));

  mxfp4_gemm::mxfp4_matmul_tiled<half, BM, BN, BK, TM, TN>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale, bias, output, M,
                                   N, K, has_bias);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void
launch_mxfp4_matmul_bf16(const __nv_bfloat16 *input, const uint8_t *weight,
                         const uint8_t *weight_scale, const __nv_bfloat16 *bias,
                         __nv_bfloat16 *output, int M, int N, int K,
                         bool has_bias, cudaStream_t stream) {
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
  constexpr int THREADS_N = BN / TN; // 16
  constexpr int THREADS_M = BM / TM; // 16

  dim3 block(THREADS_N, THREADS_M);
  dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));

  mxfp4_gemm::mxfp4_matmul_tiled<__nv_bfloat16, BM, BN, BK, TM, TN>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale, bias, output, M,
                                   N, K, has_bias);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_mxfp4_indexed_moe_gemm_f16(
    const __half *input, const uint8_t *weights, const uint8_t *weight_scales,
    const __half *biases, const uint32_t *indices, __half *output,
    int num_tokens, int topk, int num_experts, int N, int K, bool has_bias,
    bool input_has_topk_dim, cudaStream_t stream) {
  constexpr int THREADS_PER_BLOCK = MOE_BLOCK_N * 32;
  int n_chunks = CEILDIV(N, MOE_BLOCK_N);

  int total_blocks =
      input_has_topk_dim ? num_tokens * topk * n_chunks : num_tokens * n_chunks;

  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(total_blocks);
  size_t shared_mem_size = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);

  mxfp4_gemm::mxfp4_moe_gemm<half><<<grid, block, shared_mem_size, stream>>>(
      input, weights, weight_scales, biases, indices, output, num_tokens, topk,
      num_experts, N, K, has_bias, input_has_topk_dim);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_mxfp4_indexed_moe_gemm_bf16(
    const __nv_bfloat16 *input, const uint8_t *weights,
    const uint8_t *weight_scales, const __nv_bfloat16 *biases,
    const uint32_t *indices, __nv_bfloat16 *output, int num_tokens, int topk,
    int num_experts, int N, int K, bool has_bias, bool input_has_topk_dim,
    cudaStream_t stream) {
  constexpr int THREADS_PER_BLOCK = MOE_BLOCK_N * 32;
  int n_chunks = CEILDIV(N, MOE_BLOCK_N);

  int total_blocks =
      input_has_topk_dim ? num_tokens * topk * n_chunks : num_tokens * n_chunks;

  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(total_blocks);
  size_t shared_mem_size = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);

  mxfp4_gemm::mxfp4_moe_gemm<__nv_bfloat16>
      <<<grid, block, shared_mem_size, stream>>>(
          input, weights, weight_scales, biases, indices, output, num_tokens,
          topk, num_experts, N, K, has_bias, input_has_topk_dim);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" int mxfp4_get_max_smem_optin() {
  int max_smem = 0;
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         dev);
  return max_smem;
}

extern "C" void launch_mxfp4_moe_grouped_gemm_f16(
    const __half *input, const uint8_t *weights, const uint8_t *weight_scales,
    const __half *biases, const uint32_t *indices, __half *output,
    int num_tokens, int topk, int num_experts, int N, int K, bool has_bias,
    bool input_has_topk_dim, cudaStream_t stream) {
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
  dim3 block(BN / TN, BM / TM);
  dim3 grid(CEILDIV(N, BN), num_experts);
  size_t smem = num_tokens * topk * sizeof(int);

  CUDA_CHECK(cudaFuncSetAttribute(
      mxfp4_gemm::mxfp4_moe_grouped_gemm_tiled<half, BM, BN, BK, TM, TN>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  mxfp4_gemm::mxfp4_moe_grouped_gemm_tiled<half, BM, BN, BK, TM, TN>
      <<<grid, block, smem, stream>>>(
          input, weights, weight_scales, biases, indices, output, num_tokens,
          topk, num_experts, N, K, has_bias, input_has_topk_dim);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_mxfp4_moe_grouped_gemm_bf16(
    const __nv_bfloat16 *input, const uint8_t *weights,
    const uint8_t *weight_scales, const __nv_bfloat16 *biases,
    const uint32_t *indices, __nv_bfloat16 *output, int num_tokens, int topk,
    int num_experts, int N, int K, bool has_bias, bool input_has_topk_dim,
    cudaStream_t stream) {
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
  dim3 block(BN / TN, BM / TM);
  dim3 grid(CEILDIV(N, BN), num_experts);
  size_t smem = num_tokens * topk * sizeof(int);

  CUDA_CHECK(cudaFuncSetAttribute(
      mxfp4_gemm::mxfp4_moe_grouped_gemm_tiled<__nv_bfloat16, BM, BN, BK, TM,
                                               TN>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  mxfp4_gemm::mxfp4_moe_grouped_gemm_tiled<__nv_bfloat16, BM, BN, BK, TM, TN>
      <<<grid, block, smem, stream>>>(
          input, weights, weight_scales, biases, indices, output, num_tokens,
          topk, num_experts, N, K, has_bias, input_has_topk_dim);
  CUDA_CHECK(cudaGetLastError());
}
