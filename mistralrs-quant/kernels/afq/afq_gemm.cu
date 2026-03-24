/*
 * AFQ (Affine Fast Quantization) Fused GEMM CUDA Kernels
 *
 * Implements fused dequantization + matrix multiplication for optimal
 * performance. These kernels dequantize on-the-fly during the matmul to save
 * memory bandwidth.
 */

#include "afq_utils.cuh"
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// Configuration
// ============================================================================

// QMV (quantized matrix-vector) kernel configuration
#define QMV_BLOCK_SIZE 256
#define QMV_ROWS_PER_BLOCK 4

// QMM (quantized matrix-matrix) kernel configuration
#define QMM_TILE_M 64
#define QMM_TILE_N 64
#define QMM_TILE_K 32

// ============================================================================
// Quantized Matrix-Vector Multiply (QMV)
// Computes: y = x @ W^T where W is quantized
// x: [M, K], W: [N, K] (quantized), y: [M, N]
// Optimized for small M (generation phase)
// ============================================================================

// QMV kernel for power-of-2 bit widths
template <typename T, int bits, int group_size>
__global__ void
afq_qmv_kernel(const T *__restrict__ x, const uint32_t *__restrict__ w_q,
               const T *__restrict__ scales, const T *__restrict__ biases,
               T *__restrict__ y, int M, int N, int K) {
  constexpr int values_per_u32 = 32 / bits;
  const int packed_K = K * bits / 32;
  const int groups_per_row = K / group_size;

  // Each warp processes one output element y[m, n]
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / AFQ_WARP_SIZE;
  int lane = threadIdx.x % AFQ_WARP_SIZE;

  int total_outputs = M * N;
  if (warp_id >= total_outputs)
    return;

  int m = warp_id / N;
  int n = warp_id % N;

  const T *x_row = x + m * K;
  const uint32_t *w_row = w_q + n * packed_K;
  const T *scales_row = scales + n * groups_per_row;
  const T *biases_row = biases + n * groups_per_row;

  float acc = 0.0f;

  // Each lane processes multiple K elements
  for (int k = lane; k < K; k += AFQ_WARP_SIZE) {
    // Get quantized weight
    int packed_idx = k / values_per_u32;
    int value_idx = k % values_per_u32;
    uint32_t packed = w_row[packed_idx];
    uint32_t q = extract_bits<bits>(packed, value_idx);

    // Get scale and bias
    int group_idx = k / group_size;
    float scale, bias;
    if constexpr (std::is_same_v<T, float>) {
      scale = scales_row[group_idx];
      bias = biases_row[group_idx];
    } else if constexpr (std::is_same_v<T, __half>) {
      scale = __half2float(scales_row[group_idx]);
      bias = __half2float(biases_row[group_idx]);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      scale = __bfloat162float(scales_row[group_idx]);
      bias = __bfloat162float(biases_row[group_idx]);
    }
#endif

    // Dequantize: w = q * scale + bias
    float w = (float)q * scale + bias;

    // Get input
    float x_val;
    if constexpr (std::is_same_v<T, float>) {
      x_val = x_row[k];
    } else if constexpr (std::is_same_v<T, __half>) {
      x_val = __half2float(x_row[k]);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      x_val = __bfloat162float(x_row[k]);
    }
#endif

    acc = fmaf(x_val, w, acc);
  }

  // Warp reduce
  acc = warp_reduce_sum(acc);

  // Lane 0 writes result
  if (lane == 0) {
    if constexpr (std::is_same_v<T, float>) {
      y[m * N + n] = acc;
    } else if constexpr (std::is_same_v<T, __half>) {
      y[m * N + n] = __float2half(acc);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      y[m * N + n] = __float2bfloat16(acc);
    }
#endif
  }
}

// QMV kernel for 3-bit (non-power-of-2)
template <typename T, int group_size>
__global__ void
afq_qmv_3bit_kernel(const T *__restrict__ x, const uint8_t *__restrict__ w_q,
                    const T *__restrict__ scales, const T *__restrict__ biases,
                    T *__restrict__ y, int M, int N, int K) {
  const int packed_K = (K * 3 + 7) / 8;
  const int groups_per_row = K / group_size;

  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / AFQ_WARP_SIZE;
  int lane = threadIdx.x % AFQ_WARP_SIZE;

  int total_outputs = M * N;
  if (warp_id >= total_outputs)
    return;

  int m = warp_id / N;
  int n = warp_id % N;

  const T *x_row = x + m * K;
  const uint8_t *w_row = w_q + n * packed_K;
  const T *scales_row = scales + n * groups_per_row;
  const T *biases_row = biases + n * groups_per_row;

  float acc = 0.0f;

  for (int k = lane; k < K; k += AFQ_WARP_SIZE) {
    uint32_t q = extract_3bit(w_row, k);

    int group_idx = k / group_size;
    float scale, bias;
    if constexpr (std::is_same_v<T, float>) {
      scale = scales_row[group_idx];
      bias = biases_row[group_idx];
    } else if constexpr (std::is_same_v<T, __half>) {
      scale = __half2float(scales_row[group_idx]);
      bias = __half2float(biases_row[group_idx]);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      scale = __bfloat162float(scales_row[group_idx]);
      bias = __bfloat162float(biases_row[group_idx]);
    }
#endif

    float w = (float)q * scale + bias;

    float x_val;
    if constexpr (std::is_same_v<T, float>) {
      x_val = x_row[k];
    } else if constexpr (std::is_same_v<T, __half>) {
      x_val = __half2float(x_row[k]);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      x_val = __bfloat162float(x_row[k]);
    }
#endif

    acc = fmaf(x_val, w, acc);
  }

  acc = warp_reduce_sum(acc);

  if (lane == 0) {
    if constexpr (std::is_same_v<T, float>) {
      y[m * N + n] = acc;
    } else if constexpr (std::is_same_v<T, __half>) {
      y[m * N + n] = __float2half(acc);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      y[m * N + n] = __float2bfloat16(acc);
    }
#endif
  }
}

// QMV kernel for 6-bit (non-power-of-2)
template <typename T, int group_size>
__global__ void
afq_qmv_6bit_kernel(const T *__restrict__ x, const uint8_t *__restrict__ w_q,
                    const T *__restrict__ scales, const T *__restrict__ biases,
                    T *__restrict__ y, int M, int N, int K) {
  const int packed_K = (K * 6 + 7) / 8;
  const int groups_per_row = K / group_size;

  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / AFQ_WARP_SIZE;
  int lane = threadIdx.x % AFQ_WARP_SIZE;

  int total_outputs = M * N;
  if (warp_id >= total_outputs)
    return;

  int m = warp_id / N;
  int n = warp_id % N;

  const T *x_row = x + m * K;
  const uint8_t *w_row = w_q + n * packed_K;
  const T *scales_row = scales + n * groups_per_row;
  const T *biases_row = biases + n * groups_per_row;

  float acc = 0.0f;

  for (int k = lane; k < K; k += AFQ_WARP_SIZE) {
    uint32_t q = extract_6bit(w_row, k);

    int group_idx = k / group_size;
    float scale, bias;
    if constexpr (std::is_same_v<T, float>) {
      scale = scales_row[group_idx];
      bias = biases_row[group_idx];
    } else if constexpr (std::is_same_v<T, __half>) {
      scale = __half2float(scales_row[group_idx]);
      bias = __half2float(biases_row[group_idx]);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      scale = __bfloat162float(scales_row[group_idx]);
      bias = __bfloat162float(biases_row[group_idx]);
    }
#endif

    float w = (float)q * scale + bias;

    float x_val;
    if constexpr (std::is_same_v<T, float>) {
      x_val = x_row[k];
    } else if constexpr (std::is_same_v<T, __half>) {
      x_val = __half2float(x_row[k]);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      x_val = __bfloat162float(x_row[k]);
    }
#endif

    acc = fmaf(x_val, w, acc);
  }

  acc = warp_reduce_sum(acc);

  if (lane == 0) {
    if constexpr (std::is_same_v<T, float>) {
      y[m * N + n] = acc;
    } else if constexpr (std::is_same_v<T, __half>) {
      y[m * N + n] = __float2half(acc);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      y[m * N + n] = __float2bfloat16(acc);
    }
#endif
  }
}

// ============================================================================
// Quantized Matrix-Matrix Multiply (QMM) - Tiled version
// Computes: y = x @ W^T where W is quantized
// x: [M, K], W: [N, K] (quantized), y: [M, N]
// Optimized for larger M (prefill phase)
// ============================================================================

// Tiled QMM kernel using shared memory
template <typename T, int bits, int group_size, int TILE_M = 32,
          int TILE_N = 32, int TILE_K = 32>
__global__ void
afq_qmm_kernel(const T *__restrict__ x, const uint32_t *__restrict__ w_q,
               const T *__restrict__ scales, const T *__restrict__ biases,
               T *__restrict__ y, int M, int N, int K) {
  constexpr int values_per_u32 = 32 / bits;
  const int packed_K = K * bits / 32;
  const int groups_per_row = K / group_size;

  // Shared memory for tiles
  __shared__ float x_tile[TILE_M][TILE_K];
  __shared__ float w_tile[TILE_N][TILE_K];

  int bm = blockIdx.y * TILE_M;
  int bn = blockIdx.x * TILE_N;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Accumulator
  float acc = 0.0f;

  // Process K dimension in tiles
  for (int bk = 0; bk < K; bk += TILE_K) {
    // Load x tile
    int m_idx = bm + ty;
    int k_idx = bk + tx;
    if (m_idx < M && k_idx < K) {
      if constexpr (std::is_same_v<T, float>) {
        x_tile[ty][tx] = x[m_idx * K + k_idx];
      } else if constexpr (std::is_same_v<T, __half>) {
        x_tile[ty][tx] = __half2float(x[m_idx * K + k_idx]);
      }
#if __CUDA_ARCH__ >= 800
      else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        x_tile[ty][tx] = __bfloat162float(x[m_idx * K + k_idx]);
      }
#endif
    } else {
      x_tile[ty][tx] = 0.0f;
    }

    // Load and dequantize w tile
    int n_idx = bn + ty;
    if (n_idx < N && k_idx < K) {
      int packed_idx = k_idx / values_per_u32;
      int value_idx = k_idx % values_per_u32;
      uint32_t packed = w_q[n_idx * packed_K + packed_idx];
      uint32_t q = extract_bits<bits>(packed, value_idx);

      int group_idx = k_idx / group_size;
      float scale, bias;
      if constexpr (std::is_same_v<T, float>) {
        scale = scales[n_idx * groups_per_row + group_idx];
        bias = biases[n_idx * groups_per_row + group_idx];
      } else if constexpr (std::is_same_v<T, __half>) {
        scale = __half2float(scales[n_idx * groups_per_row + group_idx]);
        bias = __half2float(biases[n_idx * groups_per_row + group_idx]);
      }
#if __CUDA_ARCH__ >= 800
      else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        scale = __bfloat162float(scales[n_idx * groups_per_row + group_idx]);
        bias = __bfloat162float(biases[n_idx * groups_per_row + group_idx]);
      }
#endif

      w_tile[ty][tx] = (float)q * scale + bias;
    } else {
      w_tile[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot product
#pragma unroll
    for (int k = 0; k < TILE_K; k++) {
      acc = fmaf(x_tile[ty][k], w_tile[tx][k], acc);
    }

    __syncthreads();
  }

  // Write result
  int m_out = bm + ty;
  int n_out = bn + tx;
  if (m_out < M && n_out < N) {
    if constexpr (std::is_same_v<T, float>) {
      y[m_out * N + n_out] = acc;
    } else if constexpr (std::is_same_v<T, __half>) {
      y[m_out * N + n_out] = __float2half(acc);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      y[m_out * N + n_out] = __float2bfloat16(acc);
    }
#endif
  }
}

// ============================================================================
// Extern "C" Launch Functions - QMV
// ============================================================================

#define DEFINE_QMV_LAUNCHER(bits, gs, dtype, dtype_name)                       \
  extern "C" void afq_qmv_##bits##bit_gs##gs##_##dtype_name(                   \
      const dtype *x, const uint32_t *w_q, const dtype *scales,                \
      const dtype *biases, dtype *y, int M, int N, int K) {                    \
    int total_outputs = M * N;                                                 \
    int warps_needed = total_outputs;                                          \
    int threads = warps_needed * AFQ_WARP_SIZE;                                \
    int blocks = cdiv(threads, QMV_BLOCK_SIZE);                                \
    afq_qmv_kernel<dtype, bits, gs>                                            \
        <<<blocks, QMV_BLOCK_SIZE>>>(x, w_q, scales, biases, y, M, N, K);      \
  }

#define DEFINE_QMV_3BIT_LAUNCHER(gs, dtype, dtype_name)                        \
  extern "C" void afq_qmv_3bit_gs##gs##_##dtype_name(                          \
      const dtype *x, const uint8_t *w_q, const dtype *scales,                 \
      const dtype *biases, dtype *y, int M, int N, int K) {                    \
    int total_outputs = M * N;                                                 \
    int warps_needed = total_outputs;                                          \
    int threads = warps_needed * AFQ_WARP_SIZE;                                \
    int blocks = cdiv(threads, QMV_BLOCK_SIZE);                                \
    afq_qmv_3bit_kernel<dtype, gs>                                             \
        <<<blocks, QMV_BLOCK_SIZE>>>(x, w_q, scales, biases, y, M, N, K);      \
  }

#define DEFINE_QMV_6BIT_LAUNCHER(gs, dtype, dtype_name)                        \
  extern "C" void afq_qmv_6bit_gs##gs##_##dtype_name(                          \
      const dtype *x, const uint8_t *w_q, const dtype *scales,                 \
      const dtype *biases, dtype *y, int M, int N, int K) {                    \
    int total_outputs = M * N;                                                 \
    int warps_needed = total_outputs;                                          \
    int threads = warps_needed * AFQ_WARP_SIZE;                                \
    int blocks = cdiv(threads, QMV_BLOCK_SIZE);                                \
    afq_qmv_6bit_kernel<dtype, gs>                                             \
        <<<blocks, QMV_BLOCK_SIZE>>>(x, w_q, scales, biases, y, M, N, K);      \
  }

// 2-bit QMV launchers
DEFINE_QMV_LAUNCHER(2, 32, float, f32)
DEFINE_QMV_LAUNCHER(2, 64, float, f32)
DEFINE_QMV_LAUNCHER(2, 128, float, f32)
DEFINE_QMV_LAUNCHER(2, 32, __half, f16)
DEFINE_QMV_LAUNCHER(2, 64, __half, f16)
DEFINE_QMV_LAUNCHER(2, 128, __half, f16)

// 3-bit QMV launchers
DEFINE_QMV_3BIT_LAUNCHER(32, float, f32)
DEFINE_QMV_3BIT_LAUNCHER(64, float, f32)
DEFINE_QMV_3BIT_LAUNCHER(128, float, f32)
DEFINE_QMV_3BIT_LAUNCHER(32, __half, f16)
DEFINE_QMV_3BIT_LAUNCHER(64, __half, f16)
DEFINE_QMV_3BIT_LAUNCHER(128, __half, f16)

// 4-bit QMV launchers
DEFINE_QMV_LAUNCHER(4, 32, float, f32)
DEFINE_QMV_LAUNCHER(4, 64, float, f32)
DEFINE_QMV_LAUNCHER(4, 128, float, f32)
DEFINE_QMV_LAUNCHER(4, 32, __half, f16)
DEFINE_QMV_LAUNCHER(4, 64, __half, f16)
DEFINE_QMV_LAUNCHER(4, 128, __half, f16)

// 6-bit QMV launchers
DEFINE_QMV_6BIT_LAUNCHER(32, float, f32)
DEFINE_QMV_6BIT_LAUNCHER(64, float, f32)
DEFINE_QMV_6BIT_LAUNCHER(128, float, f32)
DEFINE_QMV_6BIT_LAUNCHER(32, __half, f16)
DEFINE_QMV_6BIT_LAUNCHER(64, __half, f16)
DEFINE_QMV_6BIT_LAUNCHER(128, __half, f16)

// 8-bit QMV launchers
DEFINE_QMV_LAUNCHER(8, 32, float, f32)
DEFINE_QMV_LAUNCHER(8, 64, float, f32)
DEFINE_QMV_LAUNCHER(8, 128, float, f32)
DEFINE_QMV_LAUNCHER(8, 32, __half, f16)
DEFINE_QMV_LAUNCHER(8, 64, __half, f16)
DEFINE_QMV_LAUNCHER(8, 128, __half, f16)

// BFloat16 QMV
DEFINE_QMV_LAUNCHER(2, 32, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(2, 64, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(2, 128, __nv_bfloat16, bf16)
DEFINE_QMV_3BIT_LAUNCHER(32, __nv_bfloat16, bf16)
DEFINE_QMV_3BIT_LAUNCHER(64, __nv_bfloat16, bf16)
DEFINE_QMV_3BIT_LAUNCHER(128, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(4, 32, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(4, 64, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(4, 128, __nv_bfloat16, bf16)
DEFINE_QMV_6BIT_LAUNCHER(32, __nv_bfloat16, bf16)
DEFINE_QMV_6BIT_LAUNCHER(64, __nv_bfloat16, bf16)
DEFINE_QMV_6BIT_LAUNCHER(128, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(8, 32, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(8, 64, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(8, 128, __nv_bfloat16, bf16)

// ============================================================================
// Extern "C" Launch Functions - QMM (for larger batch sizes)
// ============================================================================

#define DEFINE_QMM_LAUNCHER(bits, gs, dtype, dtype_name)                       \
  extern "C" void afq_qmm_##bits##bit_gs##gs##_##dtype_name(                   \
      const dtype *x, const uint32_t *w_q, const dtype *scales,                \
      const dtype *biases, dtype *y, int M, int N, int K) {                    \
    constexpr int TILE_M = 32;                                                 \
    constexpr int TILE_N = 32;                                                 \
    constexpr int TILE_K = 32;                                                 \
    dim3 grid(cdiv(N, TILE_N), cdiv(M, TILE_M));                               \
    dim3 block(TILE_N, TILE_M);                                                \
    afq_qmm_kernel<dtype, bits, gs, TILE_M, TILE_N, TILE_K>                    \
        <<<grid, block>>>(x, w_q, scales, biases, y, M, N, K);                 \
  }

// 4-bit QMM launchers (most common)
DEFINE_QMM_LAUNCHER(4, 32, float, f32)
DEFINE_QMM_LAUNCHER(4, 64, float, f32)
DEFINE_QMM_LAUNCHER(4, 128, float, f32)
DEFINE_QMM_LAUNCHER(4, 32, __half, f16)
DEFINE_QMM_LAUNCHER(4, 64, __half, f16)
DEFINE_QMM_LAUNCHER(4, 128, __half, f16)

// 8-bit QMM launchers
DEFINE_QMM_LAUNCHER(8, 32, float, f32)
DEFINE_QMM_LAUNCHER(8, 64, float, f32)
DEFINE_QMM_LAUNCHER(8, 128, float, f32)
DEFINE_QMM_LAUNCHER(8, 32, __half, f16)
DEFINE_QMM_LAUNCHER(8, 64, __half, f16)
DEFINE_QMM_LAUNCHER(8, 128, __half, f16)

// 2-bit QMM launchers
DEFINE_QMM_LAUNCHER(2, 32, float, f32)
DEFINE_QMM_LAUNCHER(2, 64, float, f32)
DEFINE_QMM_LAUNCHER(2, 128, float, f32)
DEFINE_QMM_LAUNCHER(2, 32, __half, f16)
DEFINE_QMM_LAUNCHER(2, 64, __half, f16)
DEFINE_QMM_LAUNCHER(2, 128, __half, f16)

// BFloat16 QMM
DEFINE_QMM_LAUNCHER(4, 32, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(4, 64, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(4, 128, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(8, 32, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(8, 64, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(8, 128, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(2, 32, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(2, 64, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(2, 128, __nv_bfloat16, bf16)
