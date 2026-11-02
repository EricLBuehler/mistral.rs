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

#include "../common/fp_codecs.cuh"

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
#define WARP_SIZE 32

namespace mxfp4_gemm {

// ============================================================================
// Dequantize 8 FP4 values from one int32 (8 nibbles) using LUT and store
// ============================================================================

__device__ __forceinline__ void dequant_store_8(int q4, float scale,
                                                float *dst) {
  float w[8];
  e2m1x8_to_float_doubled(q4, w);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    dst[i] = w[i] * scale;
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

        dequant_store_8(w_vec.x, scale, &s_weight[ln][0]);
        dequant_store_8(w_vec.y, scale, &s_weight[ln][8]);
        dequant_store_8(w_vec.z, scale, &s_weight[ln][16]);
        dequant_store_8(w_vec.w, scale, &s_weight[ln][24]);
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
          dequant_store_8(w_vec.x, scale, &s_weight[ln][0]);
          dequant_store_8(w_vec.y, scale, &s_weight[ln][8]);
          dequant_store_8(w_vec.z, scale, &s_weight[ln][16]);
          dequant_store_8(w_vec.w, scale, &s_weight[ln][24]);
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

// ============================================================================
// Optimized MXFP4 Vector-Matrix Multiply Kernel (for M <= 4, decode case)
//
// For decode (M=1), the tiled GEMM wastes 63/64 of input rows and 15/16 of
// thread rows. This kernel instead parallelizes all threads along the K
// dimension for maximum utilization:
//
//   - Each block handles VECMAT_COLS output columns for one input row
//   - 256 threads (8 warps) collaborate on K-reduction per output column
//   - Vectorized uint4 weight loads + LUT dequant (same as tiled kernel)
//   - Warp shuffle reduction + cross-warp shared memory reduction
// ============================================================================

#define VECMAT_COLS 4
#define VECMAT_THREADS 256
#define VECMAT_WARPS (VECMAT_THREADS / WARP_SIZE) // 8

template <typename T>
__global__ void mxfp4_vecmat(const T *__restrict__ input,
                             const uint8_t *__restrict__ weight,
                             const uint8_t *__restrict__ weight_scale,
                             const T *__restrict__ bias, T *__restrict__ output,
                             int M, int N, int K, bool has_bias) {
  const int row = blockIdx.y;                    // input row (M dim)
  const int col_base = blockIdx.x * VECMAT_COLS; // first output col
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);
  const int k_half = K / 2;

  // Shared memory for cross-warp reduction
  __shared__ float s_reduce[VECMAT_COLS][VECMAT_WARPS];

  // Each thread accumulates over its portion of K for each output column
  float acc[VECMAT_COLS];
#pragma unroll
  for (int c = 0; c < VECMAT_COLS; c++)
    acc[c] = 0.0f;

  // Each thread processes K elements strided by VECMAT_THREADS
  // Since MXFP4 blocks are 32 elements, and we have 256 threads,
  // each thread handles complete 32-element blocks at stride 256*32/32 = 256
  // Actually: thread tid handles elements starting at tid*32, stride
  // VECMAT_THREADS*32 But that's wrong - we want each thread to process
  // contiguous 32-element blocks. Better: stride by number of threads, process
  // blocks.

  // K is divided into blocks of 32. Total blocks = K/32.
  // Each thread processes blocks strided by VECMAT_THREADS.
  const int num_k_blocks = K / MXFP4_BLOCK_SIZE;

  for (int blk = tid; blk < num_k_blocks; blk += VECMAT_THREADS) {
    const int k_start = blk * MXFP4_BLOCK_SIZE;

    // Load 32 input values for this block
    float in_vals[MXFP4_BLOCK_SIZE];
    const T *in_ptr = input + (size_t)row * K + k_start;
#pragma unroll
    for (int i = 0; i < MXFP4_BLOCK_SIZE; i += 4) {
      if constexpr (std::is_same_v<T, half>) {
        in_vals[i + 0] = __half2float(__ldg(&in_ptr[i + 0]));
        in_vals[i + 1] = __half2float(__ldg(&in_ptr[i + 1]));
        in_vals[i + 2] = __half2float(__ldg(&in_ptr[i + 2]));
        in_vals[i + 3] = __half2float(__ldg(&in_ptr[i + 3]));
      } else {
        in_vals[i + 0] = __bfloat162float(__ldg(&in_ptr[i + 0]));
        in_vals[i + 1] = __bfloat162float(__ldg(&in_ptr[i + 1]));
        in_vals[i + 2] = __bfloat162float(__ldg(&in_ptr[i + 2]));
        in_vals[i + 3] = __bfloat162float(__ldg(&in_ptr[i + 3]));
      }
    }

    // For each output column in this block's VECMAT_COLS
#pragma unroll
    for (int c = 0; c < VECMAT_COLS; c++) {
      const int col = col_base + c;
      if (col >= N)
        continue;

      // Load weight: uint4 = 16 bytes = 32 packed FP4 values
      uint4 w_vec = *reinterpret_cast<const uint4 *>(
          &weight[(size_t)col * k_half + k_start / 2]);
      float scale = e8m0_to_float(__ldg(
                        &weight_scale[(size_t)col * scale_stride + blk])) *
                    0.5f;

      // Dequant + dot product
      float w_vals[32];
      dequant_store_8(w_vec.x, scale, &w_vals[0]);
      dequant_store_8(w_vec.y, scale, &w_vals[8]);
      dequant_store_8(w_vec.z, scale, &w_vals[16]);
      dequant_store_8(w_vec.w, scale, &w_vals[24]);

      float dot = 0.0f;
#pragma unroll
      for (int i = 0; i < MXFP4_BLOCK_SIZE; i++) {
        dot = fmaf(in_vals[i], w_vals[i], dot);
      }
      acc[c] += dot;
    }
  }

  // Warp-level reduction (within each warp of 32 threads)
#pragma unroll
  for (int c = 0; c < VECMAT_COLS; c++) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      acc[c] += __shfl_down_sync(0xffffffff, acc[c], offset);
    }
  }

  // Cross-warp reduction via shared memory
  if (lane_id == 0) {
#pragma unroll
    for (int c = 0; c < VECMAT_COLS; c++) {
      s_reduce[c][warp_id] = acc[c];
    }
  }
  __syncthreads();

  // Final reduction by first warp
  if (warp_id == 0) {
#pragma unroll
    for (int c = 0; c < VECMAT_COLS; c++) {
      float val = (lane_id < VECMAT_WARPS) ? s_reduce[c][lane_id] : 0.0f;
#pragma unroll
      for (int offset = VECMAT_WARPS / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
      }

      if (lane_id == 0) {
        const int col = col_base + c;
        if (col < N) {
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
  if (M <= 4) {
    dim3 block(VECMAT_THREADS);
    dim3 grid(CEILDIV(N, VECMAT_COLS), M);
    mxfp4_gemm::mxfp4_vecmat<half><<<grid, block, 0, stream>>>(
        input, weight, weight_scale, bias, output, M, N, K, has_bias);
  } else {
    constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
    constexpr int THREADS_N = BN / TN; // 16
    constexpr int THREADS_M = BM / TM; // 16

    dim3 block(THREADS_N, THREADS_M);
    dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));

    mxfp4_gemm::mxfp4_matmul_tiled<half, BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(input, weight, weight_scale, bias, output,
                                     M, N, K, has_bias);
  }
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void
launch_mxfp4_matmul_bf16(const __nv_bfloat16 *input, const uint8_t *weight,
                         const uint8_t *weight_scale, const __nv_bfloat16 *bias,
                         __nv_bfloat16 *output, int M, int N, int K,
                         bool has_bias, cudaStream_t stream) {
  if (M <= 4) {
    dim3 block(VECMAT_THREADS);
    dim3 grid(CEILDIV(N, VECMAT_COLS), M);
    mxfp4_gemm::mxfp4_vecmat<__nv_bfloat16><<<grid, block, 0, stream>>>(
        input, weight, weight_scale, bias, output, M, N, K, has_bias);
  } else {
    constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
    constexpr int THREADS_N = BN / TN; // 16
    constexpr int THREADS_M = BM / TM; // 16

    dim3 block(THREADS_N, THREADS_M);
    dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));

    mxfp4_gemm::mxfp4_matmul_tiled<__nv_bfloat16, BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(input, weight, weight_scale, bias, output,
                                     M, N, K, has_bias);
  }
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
