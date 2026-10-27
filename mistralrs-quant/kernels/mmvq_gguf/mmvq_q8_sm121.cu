// Adapted from llama.cpp MMVQ; see third_party/ggml_mmvq/LICENSE.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int Q8_WARP_SIZE = 32;
constexpr int Q8_BLOCK_SIZE = 32;
constexpr int Q8_WARPS = 8;
constexpr int Q8_VALUES_PER_DOT = 8;
constexpr int Q8_INTS_PER_DOT = 2;
constexpr int Q8_THREADS_PER_QUANT_BLOCK = Q8_BLOCK_SIZE / Q8_VALUES_PER_DOT;
constexpr int Q8_BLOCKS_PER_ITERATION =
    Q8_WARPS * Q8_WARP_SIZE / Q8_THREADS_PER_QUANT_BLOCK;

struct Q8Weight {
  half scale;
  int8_t values[Q8_BLOCK_SIZE];
};

struct Q8Activation {
  half2 scale_sum;
  int8_t values[Q8_BLOCK_SIZE];
};
static_assert(sizeof(Q8Weight) == 34);
static_assert(sizeof(Q8Activation) == 36);

static __device__ __forceinline__ int q8_load_int(const int8_t *values,
                                                  int index) {
  const uint16_t *p =
      reinterpret_cast<const uint16_t *>(values + sizeof(int) * index);
  return int(p[0]) | (int(p[1]) << 16);
}

template <typename T>
__global__ __launch_bounds__(Q8_WARPS *Q8_WARP_SIZE, 1) void mmvq_q8_sm121(
    const Q8Weight *__restrict__ weights,
    const Q8Activation *__restrict__ input, T *__restrict__ output,
    int input_cols) {
  const int tid = threadIdx.y * Q8_WARP_SIZE + threadIdx.x;
  const int blocks = input_cols / Q8_BLOCK_SIZE;
  const Q8Weight *row = weights + (size_t)blockIdx.x * blocks;
  float sum = 0.0f;
  for (int block = tid / Q8_THREADS_PER_QUANT_BLOCK; block < blocks;
       block += Q8_BLOCKS_PER_ITERATION) {
    const int future = block + Q8_BLOCKS_PER_ITERATION;
    if (future < blocks) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(row + future));
    }
    const int index = Q8_INTS_PER_DOT * (tid % Q8_THREADS_PER_QUANT_BLOCK);
    const int *values = reinterpret_cast<const int *>(input[block].values);
    int dot = __dp4a(q8_load_int(row[block].values, index), values[index], 0);
    dot = __dp4a(q8_load_int(row[block].values, index + 1), values[index + 1],
                 dot);
    sum += dot * __half2float(row[block].scale) *
           __low2float(input[block].scale_sum);
  }
  __shared__ float partials[Q8_WARPS - 1][Q8_WARP_SIZE];
  if (threadIdx.y > 0) {
    partials[threadIdx.y - 1][threadIdx.x] = sum;
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }
#pragma unroll
  for (int warp = 0; warp < Q8_WARPS - 1; ++warp) {
    sum += partials[warp][threadIdx.x];
  }
#pragma unroll
  for (int offset = Q8_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  if (threadIdx.x == 0) {
    output[blockIdx.x] = T(sum);
  }
}

#define Q8_SM121_LAUNCHER(SUFFIX, TYPE)                                        \
  extern "C" void launch_mmvq_gguf_q8_0_##SUFFIX##_sm121(                      \
      const void *weights, const void *input, void *output, int input_cols,    \
      int output_rows, int, int, int, void *stream) {                          \
    mmvq_q8_sm121<TYPE><<<output_rows, dim3(Q8_WARP_SIZE, Q8_WARPS), 0,        \
                          static_cast<cudaStream_t>(stream)>>>(                \
        static_cast<const Q8Weight *>(weights),                                \
        static_cast<const Q8Activation *>(input), static_cast<TYPE *>(output), \
        input_cols);                                                           \
  }

Q8_SM121_LAUNCHER(bf16, __nv_bfloat16)
Q8_SM121_LAUNCHER(f16, half)
Q8_SM121_LAUNCHER(f32, float)
