#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mistralrs_dynamic_lora {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarpSize;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename T> struct PairOps;

template <> struct PairOps<__half> {
  using Pair = half2;

  __device__ __forceinline__ static float dot(Pair lhs, Pair rhs) {
    const float2 lhs_f = __half22float2(lhs);
    const float2 rhs_f = __half22float2(rhs);
    return lhs_f.x * rhs_f.x + lhs_f.y * rhs_f.y;
  }

  __device__ __forceinline__ static float to_float(__half value) {
    return __half2float(value);
  }

  __device__ __forceinline__ static __half from_float(float value) {
    return __float2half_rn(value);
  }
};

template <> struct PairOps<__nv_bfloat16> {
  using Pair = __nv_bfloat162;

  __device__ __forceinline__ static float dot(Pair lhs, Pair rhs) {
    return __bfloat162float(lhs.x) * __bfloat162float(rhs.x) +
           __bfloat162float(lhs.y) * __bfloat162float(rhs.y);
  }

  __device__ __forceinline__ static float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }

  __device__ __forceinline__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16(value);
  }
};

template <typename T>
__global__ void shrink(const T *__restrict__ input, const T *__restrict__ a,
                       const uint32_t *__restrict__ row_indices,
                       T *__restrict__ hidden, int input_features, int rank) {
  const int block = static_cast<int>(blockIdx.x);
  const int selected_row = block / rank;
  const int rank_row = block - selected_row * rank;
  const uint32_t input_row = row_indices[selected_row];
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  using Pair = typename PairOps<T>::Pair;
  const Pair *input_pairs = reinterpret_cast<const Pair *>(
      input + static_cast<size_t>(input_row) * input_features);
  const Pair *a_pairs = reinterpret_cast<const Pair *>(
      a + static_cast<size_t>(rank_row) * input_features);

  float sum = 0.0f;
  const int pair_count = input_features / 2;
  for (int pair = threadIdx.x; pair < pair_count; pair += kThreads) {
    sum += PairOps<T>::dot(__ldg(a_pairs + pair), __ldg(input_pairs + pair));
  }
  sum = warp_sum(sum);

  __shared__ float warp_sums[kWarps];
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    sum = lane < kWarps ? warp_sums[lane] : 0.0f;
    sum = warp_sum(sum);
    if (lane == 0) {
      hidden[selected_row * rank + rank_row] = PairOps<T>::from_float(sum);
    }
  }
}

template <typename T>
__global__ void
expand_add(const T *__restrict__ b, const uint32_t *__restrict__ row_indices,
           const T *__restrict__ hidden, T *__restrict__ output,
           int output_features, int rank, float scale, int output_blocks) {
  const int selected_row = static_cast<int>(blockIdx.x) / output_blocks;
  const int output_block =
      static_cast<int>(blockIdx.x) - selected_row * output_blocks;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int output_col = output_block * kWarps + warp;
  if (output_col >= output_features) {
    return;
  }

  float sum = 0.0f;
  for (int rank_col = lane; rank_col < rank; rank_col += kWarpSize) {
    const size_t b_index = static_cast<size_t>(output_col) * rank + rank_col;
    sum += PairOps<T>::to_float(b[b_index]) *
           PairOps<T>::to_float(hidden[selected_row * rank + rank_col]);
  }
  sum = warp_sum(sum);
  if (lane == 0) {
    const uint32_t output_row = row_indices[selected_row];
    const size_t output_index =
        static_cast<size_t>(output_row) * output_features + output_col;
    const float base = PairOps<T>::to_float(output[output_index]);
    T delta = PairOps<T>::from_float(sum);
    if (scale != 1.0f) {
      delta = PairOps<T>::from_float(PairOps<T>::to_float(delta) * scale);
    }
    output[output_index] =
        PairOps<T>::from_float(base + PairOps<T>::to_float(delta));
  }
}

template <typename T>
int launch(const T *input, const T *a, const T *b, const uint32_t *row_indices,
           T *hidden, T *output, int input_features, int output_features,
           int rank, int active_rows, float scale, cudaStream_t stream) {
  if (input_features <= 0 || output_features <= 0 || rank <= 0 ||
      active_rows <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const unsigned int shrink_blocks =
      static_cast<unsigned int>(active_rows * rank);
  shrink<T><<<shrink_blocks, kThreads, 0, stream>>>(
      input, a, row_indices, hidden, input_features, rank);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  const int output_blocks = 1 + (output_features - 1) / kWarps;
  const unsigned int expand_blocks =
      static_cast<unsigned int>(active_rows * output_blocks);
  expand_add<T><<<expand_blocks, kThreads, 0, stream>>>(
      b, row_indices, hidden, output, output_features, rank, scale,
      output_blocks);
  return static_cast<int>(cudaGetLastError());
}

} // namespace mistralrs_dynamic_lora

extern "C" int launch_dynamic_lora_f16(const __half *input, const __half *a,
                                       const __half *b,
                                       const uint32_t *row_indices,
                                       __half *hidden, __half *output,
                                       int input_features, int output_features,
                                       int rank, int active_rows, float scale,
                                       cudaStream_t stream) {
  return mistralrs_dynamic_lora::launch(input, a, b, row_indices, hidden,
                                        output, input_features, output_features,
                                        rank, active_rows, scale, stream);
}

extern "C" int
launch_dynamic_lora_bf16(const __nv_bfloat16 *input, const __nv_bfloat16 *a,
                         const __nv_bfloat16 *b, const uint32_t *row_indices,
                         __nv_bfloat16 *hidden, __nv_bfloat16 *output,
                         int input_features, int output_features, int rank,
                         int active_rows, float scale, cudaStream_t stream) {
  return mistralrs_dynamic_lora::launch(input, a, b, row_indices, hidden,
                                        output, input_features, output_features,
                                        rank, active_rows, scale, stream);
}
