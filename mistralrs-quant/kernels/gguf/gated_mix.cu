#include "vecdot.cuh"
#include <cuda_runtime.h>

constexpr int GATED_MIX_THREADS = 128;
constexpr int GATED_MIX_WARPS = GATED_MIX_THREADS / WARP_SIZE;
constexpr int QUANTIZE_THREADS = 256;

template <typename T> __device__ __forceinline__ float mix_float(T value) {
  return static_cast<float>(value);
}

template <typename T> __device__ __forceinline__ T mix_cast(float value);
template <> __device__ __forceinline__ half mix_cast<half>(float value) {
  return __float2half_rn(value);
}
template <>
__device__ __forceinline__ __nv_bfloat16 mix_cast<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}
template <> __device__ __forceinline__ float mix_cast<float>(float value) {
  return value;
}

template <typename T>
__global__ void quantize_silu_q8_1(const T *__restrict__ input,
                                   block_q8_1 *__restrict__ output, int columns,
                                   int padded_columns, float divisor) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= padded_columns) {
    return;
  }
  float value = 0.0f;
  if (column < columns) {
    const float scaled =
        mix_float(mix_cast<T>(mix_float(input[column]) / divisor));
    value = mix_float(mix_cast<T>(scaled * (1.0f / (1.0f + __expf(-scaled)))));
  }
  float maximum = fabsf(value);
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
  }
  const float sum = warp_reduce_sum(value);
  const float scale = maximum / 127.0f;
  const int block = column / QK8_1;
  const int lane = column % QK8_1;
  output[block].qs[lane] = maximum == 0.0f ? 0 : (int8_t)roundf(value / scale);
  if (lane == 0) {
    output[block].ds = __floats2half2_rn(scale, sum);
  }
}

extern "C" void launch_gguf_quantize_silu_q8_1(const void *input, void *output,
                                               int columns, int padded_columns,
                                               float divisor, int dtype,
                                               cudaStream_t stream) {
  const int blocks = (padded_columns + QUANTIZE_THREADS - 1) / QUANTIZE_THREADS;
#define QUANTIZE(T)                                                            \
  quantize_silu_q8_1<T><<<blocks, QUANTIZE_THREADS, 0, stream>>>(              \
      static_cast<const T *>(input), static_cast<block_q8_1 *>(output),        \
      columns, padded_columns, divisor)
  if (dtype == 0) {
    QUANTIZE(half);
  } else if (dtype == 1) {
    QUANTIZE(__nv_bfloat16);
  } else {
    QUANTIZE(float);
  }
#undef QUANTIZE
}

template <typename T, typename Block, int QK, int QI, int VDR,
          float (*Dot)(const void *, const block_q8_1 *, const int &)>
__global__ void gated_mix_kernel(const Block *__restrict__ weights,
                                 const block_q8_1 *__restrict__ input,
                                 const T *__restrict__ residual,
                                 T *__restrict__ output, int rank, int hidden,
                                 int streams) {
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int column = blockIdx.x * GATED_MIX_WARPS + warp;
  if (column >= hidden) {
    return;
  }
  constexpr int lanes_per_block = QI / VDR;
  constexpr int blocks_per_warp = WARP_SIZE / lanes_per_block;
  const int blocks_per_row = rank / QK;
  const int part = VDR * (lane % lanes_per_block);
  float mixed = 0.0f;
  for (int stream = 0; stream < streams; ++stream) {
    const int row = stream * hidden + column;
    float sum = 0.0f;
    for (int block = lane / lanes_per_block; block < blocks_per_row;
         block += blocks_per_warp) {
      sum = __fadd_rn(sum, Dot(weights + (size_t)row * blocks_per_row + block,
                               input + block * (QK / QK8_1), part));
    }
    const float gate = mix_float(mix_cast<T>(warp_reduce_sum(sum)));
    if (lane == 0) {
      mixed =
          fmaf(mix_float(residual[row]), 1.0f / (1.0f + __expf(-gate)), mixed);
    }
  }
  if (lane == 0) {
    output[column] = mix_cast<T>(mixed / streams);
  }
}

template <typename T>
void launch_gated_mix(const void *weights, const void *input,
                      const void *residual, void *output, int rank, int hidden,
                      int streams, int format, cudaStream_t stream) {
  const int blocks = (hidden + GATED_MIX_WARPS - 1) / GATED_MIX_WARPS;
#define LAUNCH_FORMAT(tag, rust_tag, code, qk, qi, block, vdr, dot)            \
  case code:                                                                   \
    gated_mix_kernel<T, block, qk, qi, vdr, dot>                               \
        <<<blocks, GATED_MIX_THREADS, 0, stream>>>(                            \
            static_cast<const block *>(weights),                               \
            static_cast<const block_q8_1 *>(input),                            \
            static_cast<const T *>(residual), static_cast<T *>(output), rank,  \
            hidden, streams);                                                  \
    break;
  switch (format) { GGUF_VECTOR_FORMATS(LAUNCH_FORMAT) }
#undef LAUNCH_FORMAT
}

extern "C" void launch_gguf_gated_mix(const void *weights, const void *input,
                                      const void *residual, void *output,
                                      int rank, int hidden, int streams,
                                      int format, int dtype,
                                      cudaStream_t stream) {
  if (dtype == 0) {
    launch_gated_mix<half>(weights, input, residual, output, rank, hidden,
                           streams, format, stream);
  } else if (dtype == 1) {
    launch_gated_mix<__nv_bfloat16>(weights, input, residual, output, rank,
                                    hidden, streams, format, stream);
  } else {
    launch_gated_mix<float>(weights, input, residual, output, rank, hidden,
                            streams, format, stream);
  }
}
