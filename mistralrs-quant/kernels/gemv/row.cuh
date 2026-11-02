#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace mistralrs_gemv {

constexpr int GEMV_WARP_SIZE = 32;

template <typename T> struct RowArgs {
  const T *weights;
  const T *input;
  const T *bias;
  T *output;
  int rows;
  int columns;
  bool has_bias;
};

__device__ __forceinline__ float as_float(float x) { return x; }
__device__ __forceinline__ float as_float(__half x) { return __half2float(x); }
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ float as_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
#endif

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = GEMV_WARP_SIZE / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename T, typename Vec2, int BLOCK_SIZE, int BATCH_SIZE>
__device__ __forceinline__ void row(RowArgs<T> args, int row, int tid) {
  const T *__restrict__ A = args.weights;
  const T *__restrict__ X = args.input;
  const T *__restrict__ bias = args.bias;
  T *__restrict__ Y = args.output;
  const int M = args.rows;
  const int K = args.columns;
  const bool has_bias = args.has_bias;
  if (row >= M)
    return;
  const int K2 = K / 2;

  constexpr int NUM_WARPS = BLOCK_SIZE / GEMV_WARP_SIZE;
  __shared__ float warp_sums[NUM_WARPS][BATCH_SIZE];

  const Vec2 *A_vec = reinterpret_cast<const Vec2 *>(A + row * K);

  float acc[BATCH_SIZE];
#pragma unroll
  for (int b = 0; b < BATCH_SIZE; b++) {
    acc[b] = 0.0f;
  }

  for (int col2 = tid; col2 < K2; col2 += BLOCK_SIZE) {
    Vec2 a_val = __ldg(A_vec + col2);

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

  if (K % 2 != 0 && tid == 0) {
    int last_idx = K - 1;
    float a_last = as_float(__ldg(A + row * K + last_idx));
#pragma unroll
    for (int b = 0; b < BATCH_SIZE; b++) {
      float x_last = as_float(__ldg(X + b * K + last_idx));
      acc[b] = fmaf(a_last, x_last, acc[b]);
    }
  }

#pragma unroll
  for (int b = 0; b < BATCH_SIZE; b++) {
    acc[b] = warp_reduce_sum(acc[b]);
  }

  if constexpr (BLOCK_SIZE > GEMV_WARP_SIZE) {
    int warp_id = tid / GEMV_WARP_SIZE;
    int lane_id = tid % GEMV_WARP_SIZE;

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

  if (tid == 0) {
    float bias_val = has_bias ? as_float(__ldg(bias + row)) : 0.0f;

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

} // namespace mistralrs_gemv
