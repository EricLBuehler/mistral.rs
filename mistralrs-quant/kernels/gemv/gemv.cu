/*
 * Custom GEMV/GEMM CUDA Kernel for small batch sizes
 *
 * Optimized for LLM decode-phase inference (batch_size=1-8).
 * Computes: Y = X @ A^T + bias where:
 *   - A: [M, K] weight matrix (row-major)
 *   - X: [B, K] input matrix (B = batch size, 1-8)
 *   - bias: [M] optional bias vector
 *   - Y: [B, M] output matrix
 *
 * Design follows llama.cpp mmvf.cu approach:
 *   - Simple loop without heavy unrolling
 *   - Vectorized loads (float2, half2, nv_bfloat162)
 *   - __ldg() for read-only cache path
 *   - Warp-level reduction using XOR shuffle
 *   - Supports batch sizes 1-8 efficiently
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MAX_BATCH_SIZE 8

// Warp-level reduction sum using XOR shuffle (butterfly pattern)
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

// Type conversion helpers
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
#endif

// ============================================================================
// Batched GEMV Kernel - supports batch sizes 1-8
// ============================================================================

template <typename T, typename Vec2, int BLOCK_SIZE, int BATCH_SIZE>
__global__ void
gemv_kernel_batched(const T *__restrict__ A,    // [M, K] weights (row-major)
                    const T *__restrict__ X,    // [B, K] input matrix
                    const T *__restrict__ bias, // [M] optional bias
                    T *__restrict__ Y,          // [B, M] output matrix
                    int M, int K, bool has_bias) {
  const int row = blockIdx.x;
  if (row >= M)
    return;

  const int tid = threadIdx.x;
  const int K2 = K / 2;

  // Static shared memory for block-level reduction
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  __shared__ float warp_sums[NUM_WARPS][BATCH_SIZE];

  // Direct pointer to weight row
  const Vec2 *A_vec = reinterpret_cast<const Vec2 *>(A + row * K);

  // Accumulators for each batch element
  float acc[BATCH_SIZE];
#pragma unroll
  for (int b = 0; b < BATCH_SIZE; b++) {
    acc[b] = 0.0f;
  }

  // Main loop over K dimension
  for (int col2 = tid; col2 < K2; col2 += BLOCK_SIZE) {
    Vec2 a_val = __ldg(A_vec + col2);

// Process each batch element
#pragma unroll
    for (int b = 0; b < BATCH_SIZE; b++) {
      const Vec2 *x_vec = reinterpret_cast<const Vec2 *>(X + b * K);
      Vec2 x_val = __ldg(x_vec + col2);

      if constexpr (std::is_same_v<T, float>) {
        acc[b] = fmaf(a_val.x, x_val.x, acc[b]);
        acc[b] = fmaf(a_val.y, x_val.y, acc[b]);
      } else if constexpr (std::is_same_v<T, __half>) {
        float2 a_f = __half22float2(a_val);
        float2 x_f = __half22float2(x_val);
        acc[b] = fmaf(a_f.x, x_f.x, acc[b]);
        acc[b] = fmaf(a_f.y, x_f.y, acc[b]);
      }
#if __CUDA_ARCH__ >= 800
      else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        acc[b] =
            fmaf(__bfloat162float(a_val.x), __bfloat162float(x_val.x), acc[b]);
        acc[b] =
            fmaf(__bfloat162float(a_val.y), __bfloat162float(x_val.y), acc[b]);
      }
#endif
    }
  }

  // Handle remainder if K is odd
  if (K % 2 != 0 && tid == 0) {
    int last_idx = K - 1;
    float a_last = to_float(__ldg(A + row * K + last_idx));
#pragma unroll
    for (int b = 0; b < BATCH_SIZE; b++) {
      float x_last = to_float(__ldg(X + b * K + last_idx));
      acc[b] = fmaf(a_last, x_last, acc[b]);
    }
  }

// Warp-level reduction for each batch element
#pragma unroll
  for (int b = 0; b < BATCH_SIZE; b++) {
    acc[b] = warp_reduce_sum(acc[b]);
  }

  // Block-level reduction via shared memory
  if constexpr (BLOCK_SIZE > WARP_SIZE) {
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
#pragma unroll
      for (int b = 0; b < BATCH_SIZE; b++) {
        warp_sums[warp_id][b] = acc[b];
      }
    }
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
      for (int b = 0; b < BATCH_SIZE; b++) {
        acc[b] = (lane_id < NUM_WARPS) ? warp_sums[lane_id][b] : 0.0f;
        acc[b] = warp_reduce_sum(acc[b]);
      }
    }
  }

  // Thread 0 writes the final results
  if (tid == 0) {
    float bias_val = has_bias ? to_float(__ldg(bias + row)) : 0.0f;

#pragma unroll
    for (int b = 0; b < BATCH_SIZE; b++) {
      float result = acc[b] + bias_val;

      if constexpr (std::is_same_v<T, float>) {
        Y[b * M + row] = result;
      } else if constexpr (std::is_same_v<T, __half>) {
        Y[b * M + row] = __float2half(result);
      }
#if __CUDA_ARCH__ >= 800
      else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        Y[b * M + row] = __float2bfloat16(result);
      }
#endif
    }
  }
}

// ============================================================================
// Block Size Selection
// ============================================================================

__host__ int get_optimal_block_size(int K) {
  const int K2 = K / 2;
  if (K2 <= 32)
    return 32;
  if (K2 <= 64)
    return 64;
  if (K2 <= 128)
    return 128;
  return 256;
}

// ============================================================================
// Launch macro to reduce code duplication
// ============================================================================

#define LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, BATCH_SIZE)                   \
  gemv_kernel_batched<T, Vec2, BLOCK_SIZE, BATCH_SIZE>                         \
      <<<grid, dim3(BLOCK_SIZE), 0, stream>>>(A, X, bias, Y, M, K, has_bias)

#define DISPATCH_BATCH_SIZE(T, Vec2, BLOCK_SIZE)                               \
  switch (batch_size) {                                                        \
  case 1:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 1);                               \
    break;                                                                     \
  case 2:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 2);                               \
    break;                                                                     \
  case 3:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 3);                               \
    break;                                                                     \
  case 4:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 4);                               \
    break;                                                                     \
  case 5:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 5);                               \
    break;                                                                     \
  case 6:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 6);                               \
    break;                                                                     \
  case 7:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 7);                               \
    break;                                                                     \
  case 8:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 8);                               \
    break;                                                                     \
  default:                                                                     \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 1);                               \
    break;                                                                     \
  }

#define DISPATCH_BLOCK_SIZE(T, Vec2)                                           \
  switch (block_size) {                                                        \
  case 32:                                                                     \
    DISPATCH_BATCH_SIZE(T, Vec2, 32);                                          \
    break;                                                                     \
  case 64:                                                                     \
    DISPATCH_BATCH_SIZE(T, Vec2, 64);                                          \
    break;                                                                     \
  case 128:                                                                    \
    DISPATCH_BATCH_SIZE(T, Vec2, 128);                                         \
    break;                                                                     \
  case 256:                                                                    \
    DISPATCH_BATCH_SIZE(T, Vec2, 256);                                         \
    break;                                                                     \
  default:                                                                     \
    DISPATCH_BATCH_SIZE(T, Vec2, 256);                                         \
    break;                                                                     \
  }

// ============================================================================
// Launch Functions - BF16
// ============================================================================

extern "C" void launch_gemv_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *X,
                                 const __nv_bfloat16 *bias, __nv_bfloat16 *Y,
                                 int M, int K, int batch_size, bool has_bias,
                                 cudaStream_t stream) {
  int block_size = get_optimal_block_size(K);
  dim3 grid(M);

  DISPATCH_BLOCK_SIZE(__nv_bfloat16, __nv_bfloat162);
}

// ============================================================================
// Launch Functions - F16
// ============================================================================

extern "C" void launch_gemv_f16(const __half *A, const __half *X,
                                const __half *bias, __half *Y, int M, int K,
                                int batch_size, bool has_bias,
                                cudaStream_t stream) {
  int block_size = get_optimal_block_size(K);
  dim3 grid(M);

  DISPATCH_BLOCK_SIZE(__half, half2);
}

// ============================================================================
// Launch Functions - F32
// ============================================================================

extern "C" void launch_gemv_f32(const float *A, const float *X,
                                const float *bias, float *Y, int M, int K,
                                int batch_size, bool has_bias,
                                cudaStream_t stream) {
  int block_size = get_optimal_block_size(K);
  dim3 grid(M);

  DISPATCH_BLOCK_SIZE(float, float2);
}
