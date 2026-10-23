#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mistralrs::fp8 {

constexpr int kFp8GroupSize = 128;
constexpr int kWarpSize = 32;
constexpr int kHalfWarpSize = 16;
constexpr int kMaxBlockThreads = 1024;
constexpr int kQuantThreads = 256;
constexpr float kFp8Max = 448.0f;

struct alignas(16) Bf16x8 {
  __nv_bfloat16 values[8];
};

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ __forceinline__ float half_warp_max(float value) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const unsigned int mask = lane < kHalfWarpSize ? 0x0000ffffU : 0xffff0000U;
#pragma unroll
  for (int offset = kHalfWarpSize / 2; offset > 0; offset /= 2) {
    value =
        fmaxf(value, __shfl_xor_sync(mask, value, offset, kHalfWarpSize));
  }
  return value;
}

__device__ __forceinline__ float block_sum(float value, float *warp_sums) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  const int warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
  value = threadIdx.x < warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      value += __shfl_down_sync(0xffffffffU, value, offset);
    }
  }
  if (threadIdx.x == 0) {
    warp_sums[0] = value;
  }
  __syncthreads();
  return warp_sums[0];
}

__device__ __forceinline__ float normalized_weighted_value(
    __nv_bfloat16 residual, __nv_bfloat16 weight, float inverse_rms) {
  const float normalized = __bfloat162float(residual) * inverse_rms *
                           __bfloat162float(weight);
  return __bfloat162float(__float2bfloat16_rn(normalized));
}

__global__ void add_rms_stat_bf16_aligned_kernel(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ residual_output,
    float *__restrict__ inverse_rms, int columns, float epsilon) {
  const int row = static_cast<int>(blockIdx.x);
  const int vector_columns = columns / 8;
  const uint64_t row_offset = static_cast<uint64_t>(row) * vector_columns;
  const auto *input_vectors = reinterpret_cast<const Bf16x8 *>(input);
  const auto *residual_vectors = reinterpret_cast<const Bf16x8 *>(residual);
  auto *output_vectors = reinterpret_cast<Bf16x8 *>(residual_output);

  float square_sum = 0.0f;
  for (int column = threadIdx.x; column < vector_columns;
       column += blockDim.x) {
    const uint64_t index = row_offset + column;
    const Bf16x8 input_value = input_vectors[index];
    const Bf16x8 residual_value = residual_vectors[index];
    Bf16x8 output_value;
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const float sum = __bfloat162float(input_value.values[element]) +
                        __bfloat162float(residual_value.values[element]);
      const __nv_bfloat16 rounded = __float2bfloat16_rn(sum);
      output_value.values[element] = rounded;
      const float value = __bfloat162float(rounded);
      square_sum += value * value;
    }
    output_vectors[index] = output_value;
  }

  __shared__ float warp_sums[kMaxBlockThreads / kWarpSize];
  square_sum = block_sum(square_sum, warp_sums);
  if (threadIdx.x == 0) {
    inverse_rms[row] =
        rsqrtf(square_sum / static_cast<float>(columns) + epsilon);
  }
}

__global__ void add_rms_stat_bf16_kernel(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ residual_output,
    float *__restrict__ inverse_rms, int columns, float epsilon) {
  const int row = static_cast<int>(blockIdx.x);
  const uint64_t row_offset = static_cast<uint64_t>(row) * columns;

  float square_sum = 0.0f;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const uint64_t index = row_offset + column;
    const float sum = __bfloat162float(input[index]) +
                      __bfloat162float(residual[index]);
    const __nv_bfloat16 rounded = __float2bfloat16_rn(sum);
    residual_output[index] = rounded;
    const float value = __bfloat162float(rounded);
    square_sum += value * value;
  }

  __shared__ float warp_sums[kMaxBlockThreads / kWarpSize];
  square_sum = block_sum(square_sum, warp_sums);
  if (threadIdx.x == 0) {
    inverse_rms[row] =
        rsqrtf(square_sum / static_cast<float>(columns) + epsilon);
  }
}

template <bool StoreNormalized>
__global__ void add_rms_norm_quantize_bf16_row_kernel(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ residual_output,
    __nv_bfloat16 *__restrict__ normalized_output,
    __nv_fp8_e4m3 *__restrict__ quantized_output,
    float *__restrict__ scales, int rows, int columns, int scale_stride_m,
    float epsilon) {
  const int row = static_cast<int>(blockIdx.x);
  const int vector_columns = columns / 8;
  const uint64_t row_offset = static_cast<uint64_t>(row) * vector_columns;
  const auto *input_vectors = reinterpret_cast<const Bf16x8 *>(input);
  const auto *residual_vectors = reinterpret_cast<const Bf16x8 *>(residual);
  const auto *weight_vectors = reinterpret_cast<const Bf16x8 *>(weight);
  auto *residual_output_vectors =
      reinterpret_cast<Bf16x8 *>(residual_output);

  float square_sum = 0.0f;
  if (threadIdx.x < vector_columns) {
    const uint64_t index = row_offset + threadIdx.x;
    const Bf16x8 input_value = input_vectors[index];
    const Bf16x8 residual_value = residual_vectors[index];
    Bf16x8 output_value;
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const float sum = __bfloat162float(input_value.values[element]) +
                        __bfloat162float(residual_value.values[element]);
      const __nv_bfloat16 rounded = __float2bfloat16_rn(sum);
      output_value.values[element] = rounded;
      const float value = __bfloat162float(rounded);
      square_sum += value * value;
    }
    residual_output_vectors[index] = output_value;
  }

  __shared__ float warp_sums[kMaxBlockThreads / kWarpSize];
  square_sum = block_sum(square_sum, warp_sums);
  const float inverse_rms =
      rsqrtf(square_sum / static_cast<float>(columns) + epsilon);

  const int group = threadIdx.x / kHalfWarpSize;
  const int group_lane = threadIdx.x & (kHalfWarpSize - 1);
  const int groups = columns / kFp8GroupSize;
  float values[8] = {};
  if (group < groups) {
    const int vector_column = group * kHalfWarpSize + group_lane;
    const Bf16x8 residual_value =
        residual_output_vectors[row_offset + vector_column];
    const Bf16x8 weight_value = weight_vectors[vector_column];
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      values[element] = normalized_weighted_value(
          residual_value.values[element], weight_value.values[element],
          inverse_rms);
    }
  }

  float maximum = 0.0f;
#pragma unroll
  for (int element = 0; element < 8; ++element) {
    maximum = fmaxf(maximum, fabsf(values[element]));
  }
  maximum = half_warp_max(maximum);

  if (group < groups) {
    const int vector_column = group * kHalfWarpSize + group_lane;
    const uint64_t output_offset =
        (row_offset + vector_column) * static_cast<uint64_t>(8);
    if constexpr (StoreNormalized) {
      Bf16x8 normalized_value;
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        normalized_value.values[element] = __float2bfloat16_rn(values[element]);
      }
      reinterpret_cast<Bf16x8 *>(normalized_output)[row_offset +
                                                     vector_column] =
          normalized_value;
    }

    const float qscale = maximum == 0.0f ? 1.0f : kFp8Max / maximum;
    if (group_lane == 0) {
      scales[static_cast<uint64_t>(group) * scale_stride_m + row] =
          1.0f / qscale;
    }

    auto *output_pairs = reinterpret_cast<__nv_fp8x2_storage_t *>(
        quantized_output + output_offset);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 value = make_float2(values[pair * 2] * qscale,
                                       values[pair * 2 + 1] * qscale);
      output_pairs[pair] =
          __nv_cvt_float2_to_fp8x2(value, __NV_SATFINITE, __NV_E4M3);
    }

    if (row == 0) {
      for (int padding = group_lane; padding < scale_stride_m - rows;
           padding += kHalfWarpSize) {
        scales[static_cast<uint64_t>(group) * scale_stride_m + rows +
               padding] = 0.0f;
      }
    }
  }
}

template <bool Aligned, bool StoreNormalized>
__global__ void rms_norm_quantize_bf16_kernel(
    const __nv_bfloat16 *__restrict__ residual_output,
    const __nv_bfloat16 *__restrict__ weight,
    const float *__restrict__ inverse_rms,
    __nv_bfloat16 *__restrict__ normalized_output,
    __nv_fp8_e4m3 *__restrict__ quantized_output,
    float *__restrict__ scales, int rows, int columns, int scale_stride_m) {
  const int groups = columns / kFp8GroupSize;
  const uint64_t group_count = static_cast<uint64_t>(rows) * groups;
  uint64_t group_index =
      (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      kWarpSize;
  const uint64_t group_stride =
      static_cast<uint64_t>(gridDim.x) * blockDim.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);

  for (; group_index < group_count; group_index += group_stride) {
    const int row = static_cast<int>(group_index / groups);
    const int group = static_cast<int>(group_index % groups);
    const uint64_t group_offset = static_cast<uint64_t>(row) * columns +
                                  static_cast<uint64_t>(group) * kFp8GroupSize;
    const int first_column = group * kFp8GroupSize + lane * 2;
    const int second_column = first_column + 64;
    const float row_inverse_rms = inverse_rms[row];
    float values[4];

    if constexpr (Aligned) {
      const auto *residual_pairs =
          reinterpret_cast<const __nv_bfloat162 *>(residual_output +
                                                   group_offset);
      const auto *weight_pairs = reinterpret_cast<const __nv_bfloat162 *>(
          weight + static_cast<uint64_t>(group) * kFp8GroupSize);
      const __nv_bfloat162 residual_first = residual_pairs[lane];
      const __nv_bfloat162 residual_second = residual_pairs[lane + 32];
      const __nv_bfloat162 weight_first = weight_pairs[lane];
      const __nv_bfloat162 weight_second = weight_pairs[lane + 32];
      values[0] = normalized_weighted_value(
          residual_first.x, weight_first.x, row_inverse_rms);
      values[1] = normalized_weighted_value(
          residual_first.y, weight_first.y, row_inverse_rms);
      values[2] = normalized_weighted_value(
          residual_second.x, weight_second.x, row_inverse_rms);
      values[3] = normalized_weighted_value(
          residual_second.y, weight_second.y, row_inverse_rms);
    } else {
      values[0] = normalized_weighted_value(
          residual_output[group_offset + lane * 2], weight[first_column],
          row_inverse_rms);
      values[1] = normalized_weighted_value(
          residual_output[group_offset + lane * 2 + 1],
          weight[first_column + 1], row_inverse_rms);
      values[2] = normalized_weighted_value(
          residual_output[group_offset + 64 + lane * 2], weight[second_column],
          row_inverse_rms);
      values[3] = normalized_weighted_value(
          residual_output[group_offset + 64 + lane * 2 + 1],
          weight[second_column + 1], row_inverse_rms);
    }

    float maximum = 0.0f;
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      maximum = fmaxf(maximum, fabsf(values[index]));
    }
    maximum = warp_max(maximum);
    const float qscale = maximum == 0.0f ? 1.0f : kFp8Max / maximum;
    if (lane == 0) {
      scales[static_cast<uint64_t>(group) * scale_stride_m + row] =
          1.0f / qscale;
    }

    if constexpr (StoreNormalized) {
      normalized_output[group_offset + lane * 2] =
          __float2bfloat16_rn(values[0]);
      normalized_output[group_offset + lane * 2 + 1] =
          __float2bfloat16_rn(values[1]);
      normalized_output[group_offset + 64 + lane * 2] =
          __float2bfloat16_rn(values[2]);
      normalized_output[group_offset + 64 + lane * 2 + 1] =
          __float2bfloat16_rn(values[3]);
    }

    const float2 first_pair =
        make_float2(values[0] * qscale, values[1] * qscale);
    const float2 second_pair =
        make_float2(values[2] * qscale, values[3] * qscale);
    auto *output_pairs = reinterpret_cast<__nv_fp8x2_storage_t *>(
        quantized_output + group_offset);
    output_pairs[lane] =
        __nv_cvt_float2_to_fp8x2(first_pair, __NV_SATFINITE, __NV_E4M3);
    output_pairs[lane + 32] =
        __nv_cvt_float2_to_fp8x2(second_pair, __NV_SATFINITE, __NV_E4M3);

    if (row == 0) {
      for (int padding = lane; padding < scale_stride_m - rows;
           padding += kWarpSize) {
        scales[static_cast<uint64_t>(group) * scale_stride_m + rows +
               padding] = 0.0f;
      }
    }
  }
}

__host__ __forceinline__ int rounded_block_threads(int elements) {
  const int rounded = (elements + kWarpSize - 1) / kWarpSize * kWarpSize;
  return std::min(std::max(rounded, kWarpSize), kMaxBlockThreads);
}

__host__ int launch_fused_add_rms_norm_quantize_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *residual,
    const __nv_bfloat16 *weight, __nv_bfloat16 *residual_output,
    __nv_bfloat16 *normalized_output, __nv_fp8_e4m3 *quantized_output,
    float *scales, float *inverse_rms, int rows, int columns,
    int scale_stride_m, float epsilon, void *stream) {
  if (input == nullptr || residual == nullptr || weight == nullptr ||
      residual_output == nullptr || quantized_output == nullptr ||
      scales == nullptr || inverse_rms == nullptr || rows <= 0 || columns <= 0 ||
      columns % kFp8GroupSize != 0 || scale_stride_m < rows ||
      !std::isfinite(epsilon) || epsilon < 0.0f ||
      reinterpret_cast<uintptr_t>(residual_output) % alignof(__nv_bfloat16) !=
          0 ||
      (normalized_output != nullptr &&
       reinterpret_cast<uintptr_t>(normalized_output) %
               alignof(__nv_bfloat16) !=
           0) ||
      reinterpret_cast<uintptr_t>(quantized_output) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(scales) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(inverse_rms) % alignof(float) != 0) {
    return -static_cast<int>(cudaErrorInvalidValue);
  }

  const int vector_columns = columns / 8;
  const bool row_aligned =
      vector_columns <= kMaxBlockThreads &&
      reinterpret_cast<uintptr_t>(input) % alignof(Bf16x8) == 0 &&
      reinterpret_cast<uintptr_t>(residual) % alignof(Bf16x8) == 0 &&
      reinterpret_cast<uintptr_t>(weight) % alignof(Bf16x8) == 0 &&
      reinterpret_cast<uintptr_t>(residual_output) % alignof(Bf16x8) == 0 &&
      (normalized_output == nullptr ||
       reinterpret_cast<uintptr_t>(normalized_output) % alignof(Bf16x8) == 0);
  const auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (row_aligned) {
    const int threads = rounded_block_threads(vector_columns);
    if (normalized_output == nullptr) {
      add_rms_norm_quantize_bf16_row_kernel<false>
          <<<rows, threads, 0, cuda_stream>>>(
              input, residual, weight, residual_output, normalized_output,
              quantized_output, scales, rows, columns, scale_stride_m, epsilon);
    } else {
      add_rms_norm_quantize_bf16_row_kernel<true>
          <<<rows, threads, 0, cuda_stream>>>(
              input, residual, weight, residual_output, normalized_output,
              quantized_output, scales, rows, columns, scale_stride_m, epsilon);
    }
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : -static_cast<int>(error);
  }

  const bool stat_aligned =
      reinterpret_cast<uintptr_t>(input) % alignof(Bf16x8) == 0 &&
      reinterpret_cast<uintptr_t>(residual) % alignof(Bf16x8) == 0 &&
      reinterpret_cast<uintptr_t>(residual_output) % alignof(Bf16x8) == 0;
  const bool quant_aligned =
      reinterpret_cast<uintptr_t>(residual_output) %
              alignof(__nv_bfloat162) ==
          0 &&
      reinterpret_cast<uintptr_t>(weight) % alignof(__nv_bfloat162) == 0;
  const int stat_threads =
      rounded_block_threads(stat_aligned ? vector_columns : columns);
  if (stat_aligned) {
    add_rms_stat_bf16_aligned_kernel<<<rows, stat_threads, 0, cuda_stream>>>(
        input, residual, residual_output, inverse_rms, columns, epsilon);
  } else {
    add_rms_stat_bf16_kernel<<<rows, stat_threads, 0, cuda_stream>>>(
        input, residual, residual_output, inverse_rms, columns, epsilon);
  }
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return -static_cast<int>(error);
  }

  constexpr int warps_per_block = kQuantThreads / kWarpSize;
  const uint64_t group_count =
      static_cast<uint64_t>(rows) * columns / kFp8GroupSize;
  const int quant_blocks =
      static_cast<int>((group_count + warps_per_block - 1) / warps_per_block);
  if (quant_aligned) {
    if (normalized_output == nullptr) {
      rms_norm_quantize_bf16_kernel<true, false>
          <<<quant_blocks, kQuantThreads, 0, cuda_stream>>>(
              residual_output, weight, inverse_rms, normalized_output,
              quantized_output, scales, rows, columns, scale_stride_m);
    } else {
      rms_norm_quantize_bf16_kernel<true, true>
          <<<quant_blocks, kQuantThreads, 0, cuda_stream>>>(
              residual_output, weight, inverse_rms, normalized_output,
              quantized_output, scales, rows, columns, scale_stride_m);
    }
  } else if (normalized_output == nullptr) {
    rms_norm_quantize_bf16_kernel<false, false>
        <<<quant_blocks, kQuantThreads, 0, cuda_stream>>>(
            residual_output, weight, inverse_rms, normalized_output,
            quantized_output, scales, rows, columns, scale_stride_m);
  } else {
    rms_norm_quantize_bf16_kernel<false, true>
        <<<quant_blocks, kQuantThreads, 0, cuda_stream>>>(
            residual_output, weight, inverse_rms, normalized_output,
            quantized_output, scales, rows, columns, scale_stride_m);
  }
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

} // namespace mistralrs::fp8

extern "C" const char *mistralrs_fused_rms_norm_fp8_error_string(int status) {
  return cudaGetErrorString(static_cast<cudaError_t>(
      status < 0 ? -static_cast<int64_t>(status) : status));
}

extern "C" int mistralrs_fused_add_rms_norm_quantize_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *residual,
    const __nv_bfloat16 *weight, __nv_bfloat16 *residual_output,
    __nv_bfloat16 *normalized_output, __nv_fp8_e4m3 *quantized_output,
    float *scales, float *inverse_rms, int rows, int columns,
    int scale_stride_m, float epsilon, void *stream) {
  return mistralrs::fp8::launch_fused_add_rms_norm_quantize_bf16(
      input, residual, weight, residual_output, normalized_output,
      quantized_output, scales, inverse_rms, rows, columns, scale_stride_m,
      epsilon, stream);
}
