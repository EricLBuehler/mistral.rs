#pragma once

#include "../gguf/vecdot.cuh"
#include <cuda_runtime.h>

constexpr int PREFETCH_WARPS = 8;

template <typename T, int qk, int qi, typename Weight, int vdr,
          vec_dot_q_cuda_t dot>
__device__ __forceinline__ void
prefetch_row(const Weight *__restrict__ row,
             const block_q8_1 *__restrict__ input, T *__restrict__ output,
             int columns) {
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int blocks = columns / qk;
  constexpr int threads_per_block = qi / vdr;
  constexpr int blocks_per_iteration =
      PREFETCH_WARPS * WARP_SIZE / threads_per_block;
  float sum = 0.0f;
  for (int block = tid / threads_per_block; block < blocks;
       block += blocks_per_iteration) {
    const int future = block + blocks_per_iteration;
    if (future < blocks) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(row + future));
    }
    const int part = vdr * (tid % threads_per_block);
    sum += dot(row + block, input + block * (qk / QK8_1), part);
  }
  __shared__ float partials[PREFETCH_WARPS - 1][WARP_SIZE];
  if (threadIdx.y > 0) {
    partials[threadIdx.y - 1][threadIdx.x] = sum;
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }
#pragma unroll
  for (int warp = 0; warp < PREFETCH_WARPS - 1; ++warp) {
    sum += partials[warp][threadIdx.x];
  }
  sum = warp_reduce_sum(sum);
  if (threadIdx.x == 0) {
    *output = T(sum);
  }
}
