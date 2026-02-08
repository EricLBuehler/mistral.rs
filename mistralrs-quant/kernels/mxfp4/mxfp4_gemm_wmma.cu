/**
 * MXFP4 GEMM with WMMA tensor core acceleration (Ampere+ / compute >= 80).
 *
 * Dequantizes MXFP4 weights to FP16/BF16 in shared memory using vectorized
 * uint4 loads + LUT-based dequantization, then uses WMMA fragments for
 * 16x16x16 tensor core matrix multiply-accumulate.
 *
 * Input:  [M, K] in fp16/bf16
 * Weight: [N, K/2] packed MXFP4  +  E8M0 scales [N, K/32]
 * Output: [M, N] in fp16/bf16
 *
 * Block tile: 64x64x32 (M_BLK x N_BLK x K_BLK)
 * K_BLK = 32 matches MXFP4 block size (one scale per 32 elements).
 * 8 warps (4x2), 256 threads. Each warp computes two 16x16 WMMA tiles
 * along N for a total 16x32 sub-tile, giving 4*16 x 2*32 = 64x64.
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda::wmma;

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define MXFP4_BLOCK_SIZE 32

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
    }                                                                          \
  } while (0)

namespace mxfp4_wmma {

// ---------------------------------------------------------------------------
// Helpers (duplicated from mxfp4_gemm.cu to keep this file self-contained)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float e8m0_to_float(uint8_t e) {
  return __uint_as_float((uint32_t)e << 23);
}

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

// Dequantize 8 FP4 nibbles from one int32 and store as fp16
__device__ __forceinline__ void dequant_store_8_f16(int q4, float scale,
                                                    uint32_t L0, uint32_t L1,
                                                    uint32_t L2, uint32_t L3,
                                                    half *dst) {
  int2 w = get_int_from_table_16(q4, L0, L1, L2, L3);
  dst[0] = __float2half((float)(int8_t)(w.x) * scale);
  dst[1] = __float2half((float)(int8_t)(w.y) * scale);
  dst[2] = __float2half((float)(int8_t)(w.x >> 8) * scale);
  dst[3] = __float2half((float)(int8_t)(w.y >> 8) * scale);
  dst[4] = __float2half((float)(int8_t)(w.x >> 16) * scale);
  dst[5] = __float2half((float)(int8_t)(w.y >> 16) * scale);
  dst[6] = __float2half((float)(int8_t)(w.x >> 24) * scale);
  dst[7] = __float2half((float)(int8_t)(w.y >> 24) * scale);
}

// Dequantize 8 FP4 nibbles from one int32 and store as bf16
__device__ __forceinline__ void dequant_store_8_bf16(int q4, float scale,
                                                     uint32_t L0, uint32_t L1,
                                                     uint32_t L2, uint32_t L3,
                                                     __nv_bfloat16 *dst) {
  int2 w = get_int_from_table_16(q4, L0, L1, L2, L3);
  dst[0] = __float2bfloat16((float)(int8_t)(w.x) * scale);
  dst[1] = __float2bfloat16((float)(int8_t)(w.y) * scale);
  dst[2] = __float2bfloat16((float)(int8_t)(w.x >> 8) * scale);
  dst[3] = __float2bfloat16((float)(int8_t)(w.y >> 8) * scale);
  dst[4] = __float2bfloat16((float)(int8_t)(w.x >> 16) * scale);
  dst[5] = __float2bfloat16((float)(int8_t)(w.y >> 16) * scale);
  dst[6] = __float2bfloat16((float)(int8_t)(w.x >> 24) * scale);
  dst[7] = __float2bfloat16((float)(int8_t)(w.y >> 24) * scale);
}

// Dispatch dequant by type
template <typename T>
__device__ __forceinline__ void dequant_store_8(int q4, float scale,
                                                uint32_t L0, uint32_t L1,
                                                uint32_t L2, uint32_t L3,
                                                T *dst);

template <>
__device__ __forceinline__ void
dequant_store_8<half>(int q4, float scale, uint32_t L0, uint32_t L1,
                      uint32_t L2, uint32_t L3, half *dst) {
  dequant_store_8_f16(q4, scale, L0, L1, L2, L3, dst);
}

template <>
__device__ __forceinline__ void
dequant_store_8<__nv_bfloat16>(int q4, float scale, uint32_t L0, uint32_t L1,
                               uint32_t L2, uint32_t L3, __nv_bfloat16 *dst) {
  dequant_store_8_bf16(q4, scale, L0, L1, L2, L3, dst);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr int WMMA_M_DIM = 16;
constexpr int WMMA_N_DIM = 16;
constexpr int WMMA_K_DIM = 16;

// 8 warps (4 along M, 2 along N).
// Each warp handles two WMMA N-tiles (16+16 = 32 along N).
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N; // 8
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * 32; // 256

constexpr int M_BLK = WARPS_M * WMMA_M_DIM;                // 64
constexpr int N_BLK = WARPS_N * 2 * WMMA_N_DIM;            // 64 (2 tiles per warp along N)
constexpr int K_BLK = MXFP4_BLOCK_SIZE;                    // 32
constexpr int WMMA_K_STEPS = K_BLK / WMMA_K_DIM;           // 2

using VecT = float4;
constexpr int VEC_SIZE = 8; // float4 = 16 bytes = 8 fp16/bf16 values

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <typename T>
__launch_bounds__(BLOCK_THREADS) __global__
    void mxfp4_matmul_wmma_kernel(const T *__restrict__ input,
                                  const uint8_t *__restrict__ weight,
                                  const uint8_t *__restrict__ weight_scale,
                                  const T *__restrict__ bias,
                                  T *__restrict__ output, int M, int N, int K,
                                  bool has_bias) {
  // LUT in registers (values 2x scaled, compensated by 0.5f in scale)
  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);

  extern __shared__ uint8_t smem_bytes[];

  // Shared memory layout:
  //   A_sh: [M_BLK, K_BLK] in T   — input tile
  //   B_sh: [N_BLK, K_BLK] in T   — dequantized weight tile
  //   C_sh: [M_BLK, N_BLK] in float — output accumulator for cooperative store
  T *A_sh = reinterpret_cast<T *>(smem_bytes);
  T *B_sh = A_sh + M_BLK * K_BLK;
  uint8_t *C_raw = reinterpret_cast<uint8_t *>(B_sh + N_BLK * K_BLK);
  // Align to float boundary
  size_t align_off = reinterpret_cast<uintptr_t>(C_raw) % alignof(float);
  if (align_off != 0)
    C_raw += (alignof(float) - align_off);
  float *C_sh = reinterpret_cast<float *>(C_raw);

  const int threadId = threadIdx.x;
  const int warpId = threadId / 32;
  const int warp_m_idx = warpId / WARPS_N; // 0..3
  const int warp_n_idx = warpId % WARPS_N; // 0..1

  const int m_base = blockIdx.y * M_BLK;
  const int n_base = blockIdx.x * N_BLK;

  VecT zero_vec;
  zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

  // Two accumulator fragments per warp (for the two N sub-tiles)
  fragment<accumulator, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, float> c_frag[2];
  fill_fragment(c_frag[0], 0.0f);
  fill_fragment(c_frag[1], 0.0f);

  // K-tile loop
  for (int k_base = 0; k_base < K; k_base += K_BLK) {
    // === Load input tile [M_BLK, K_BLK] into A_sh (vectorized) ===
    constexpr int A_VEC_ELEMS = M_BLK * K_BLK / VEC_SIZE; // 64*32/8 = 256
    for (int i = threadId; i < A_VEC_ELEMS; i += BLOCK_THREADS) {
      const int idx = i * VEC_SIZE;
      const int lm = idx / K_BLK;
      const int lk = idx % K_BLK;
      const int gm = m_base + lm;
      const int gk = k_base + lk;

      if (gm < M && gk < K) {
        *reinterpret_cast<VecT *>(&A_sh[lm * K_BLK + lk]) =
            *reinterpret_cast<const VecT *>(&input[(size_t)gm * K + gk]);
      } else {
        *reinterpret_cast<VecT *>(&A_sh[lm * K_BLK + lk]) = zero_vec;
      }
    }

    // === Dequantize weight tile [N_BLK, K_BLK] into B_sh ===
    // Each row is one uint4 (32 packed FP4) + one E8M0 scale
    for (int ln = threadId; ln < N_BLK; ln += BLOCK_THREADS) {
      const int gn = n_base + ln;
      if (gn < N) {
        uint4 w_vec = *reinterpret_cast<const uint4 *>(
            &weight[(size_t)gn * (K / 2) + k_base / 2]);
        float scale =
            e8m0_to_float(
                __ldg(&weight_scale[(size_t)gn * scale_stride +
                                    k_base / MXFP4_BLOCK_SIZE])) *
            0.5f;

        T *dst = &B_sh[ln * K_BLK];
        dequant_store_8<T>(w_vec.x, scale, LUT0, LUT1, LUT2, LUT3, dst);
        dequant_store_8<T>(w_vec.y, scale, LUT0, LUT1, LUT2, LUT3, dst + 8);
        dequant_store_8<T>(w_vec.z, scale, LUT0, LUT1, LUT2, LUT3, dst + 16);
        dequant_store_8<T>(w_vec.w, scale, LUT0, LUT1, LUT2, LUT3, dst + 24);
      } else {
        // Zero-fill invalid rows
        T *dst = &B_sh[ln * K_BLK];
#pragma unroll
        for (int k = 0; k < K_BLK; k++)
          dst[k] = T(0);
      }
    }

    __syncthreads();

    // === WMMA tensor core accumulation ===
    // Each warp handles one 16-row M sub-tile and two 16-col N sub-tiles.
    // Two K-steps per tile (K_BLK=32, WMMA_K=16).
#pragma unroll
    for (int k_step = 0; k_step < WMMA_K_STEPS; k_step++) {
      fragment<matrix_a, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, T, row_major>
          a_frag;
      const T *A_ptr =
          A_sh + warp_m_idx * WMMA_M_DIM * K_BLK + k_step * WMMA_K_DIM;
      load_matrix_sync(a_frag, A_ptr, K_BLK);

      // Two N sub-tiles per warp
#pragma unroll
      for (int n_sub = 0; n_sub < 2; n_sub++) {
        fragment<matrix_b, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, T, col_major>
            b_frag;
        const T *B_ptr = B_sh +
                         (warp_n_idx * 2 + n_sub) * WMMA_N_DIM * K_BLK +
                         k_step * WMMA_K_DIM;
        load_matrix_sync(b_frag, B_ptr, K_BLK);
        mma_sync(c_frag[n_sub], a_frag, b_frag, c_frag[n_sub]);
      }
    }

    __syncthreads();
  } // end k_base loop

  // === Store accumulator fragments to C_sh ===
  for (int n_sub = 0; n_sub < 2; n_sub++) {
    float *C_ptr = C_sh + warp_m_idx * WMMA_M_DIM * N_BLK +
                   (warp_n_idx * 2 + n_sub) * WMMA_N_DIM;
    store_matrix_sync(C_ptr, c_frag[n_sub], N_BLK, mem_row_major);
  }
  __syncthreads();

  // === Cooperative write from C_sh to global output ===
  constexpr int C_ELEMS = M_BLK * N_BLK; // 64*64 = 4096
  for (int i = threadId; i < C_ELEMS; i += BLOCK_THREADS) {
    const int lm = i / N_BLK;
    const int ln = i % N_BLK;
    const int gm = m_base + lm;
    const int gn = n_base + ln;

    if (gm < M && gn < N) {
      float val = C_sh[lm * N_BLK + ln];
      if (has_bias && bias != nullptr) {
        if constexpr (std::is_same_v<T, half>) {
          val += __half2float(__ldg(&bias[gn]));
        } else {
          val += __bfloat162float(__ldg(&bias[gn]));
        }
      }
      if constexpr (std::is_same_v<T, half>) {
        output[(size_t)gm * N + gn] = __float2half(val);
      } else {
        output[(size_t)gm * N + gn] = __float2bfloat16(val);
      }
    }
  }
}

} // namespace mxfp4_wmma

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

static size_t wmma_smem_bytes() {
  using namespace mxfp4_wmma;
  // A_sh + B_sh in fp16/bf16 (2 bytes each) + alignment padding + C_sh in
  // float
  size_t AB = (M_BLK * K_BLK + N_BLK * K_BLK) * 2;
  size_t pad = (16 - (AB % 16)) % 16;
  size_t C = M_BLK * N_BLK * sizeof(float);
  return AB + pad + C;
}

extern "C" void launch_mxfp4_matmul_wmma_f16(
    const __half *input, const uint8_t *weight, const uint8_t *weight_scale,
    const __half *bias, __half *output, int M, int N, int K, bool has_bias,
    cudaStream_t stream) {
  using namespace mxfp4_wmma;

  dim3 grid(CEILDIV(N, N_BLK), CEILDIV(M, M_BLK));
  dim3 block(BLOCK_THREADS);
  size_t smem = wmma_smem_bytes();

  mxfp4_wmma::mxfp4_matmul_wmma_kernel<half>
      <<<grid, block, smem, stream>>>(input, weight, weight_scale, bias, output,
                                      M, N, K, has_bias);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_mxfp4_matmul_wmma_bf16(
    const __nv_bfloat16 *input, const uint8_t *weight,
    const uint8_t *weight_scale, const __nv_bfloat16 *bias,
    __nv_bfloat16 *output, int M, int N, int K, bool has_bias,
    cudaStream_t stream) {
  using namespace mxfp4_wmma;

  dim3 grid(CEILDIV(N, N_BLK), CEILDIV(M, M_BLK));
  dim3 block(BLOCK_THREADS);
  size_t smem = wmma_smem_bytes();

  mxfp4_wmma::mxfp4_matmul_wmma_kernel<__nv_bfloat16>
      <<<grid, block, smem, stream>>>(input, weight, weight_scale, bias, output,
                                      M, N, K, has_bias);
  CUDA_CHECK(cudaGetLastError());
}
