#include <cfloat>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <type_traits>

#include "../common/fp_codecs.cuh"

namespace {
constexpr int kGroup = 32;
constexpr int kThreads = 128;
constexpr int kGemvThreads = 256;
constexpr int kGemvVector = 4;
constexpr int kGemvMinK = 512;
constexpr int kGemvShortK = 1024;
constexpr int kTileM = 16;
constexpr int kTileN = 64;
constexpr int kMmaK = 16;
constexpr int kMantissaBits = 23;
constexpr unsigned kMantissaMask = 0x007fffffu;
constexpr unsigned kFp8MaxMantissa = 0x00600000u;
constexpr int kFp8ScaleExponentBias = 135;
constexpr int kE8M0ExponentBias = 127;
constexpr unsigned kWarpMask = 0xffffffffu;

__device__ int quantization_exponent(float value) {
  float maximum = fabsf(value);
  for (int offset = kGroup / 2; offset; offset /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(kWarpMask, maximum, offset));
  // Approximate division or log2 loses the exact 448 scale boundary.
  unsigned bits = __float_as_uint(fmaxf(maximum, FLT_MIN));
  int exponent = static_cast<int>(bits >> kMantissaBits) -
                 kFp8ScaleExponentBias +
                 ((bits & kMantissaMask) > kFp8MaxMantissa);
  return min(kE8M0ExponentBias, max(-kE8M0ExponentBias, exponent));
}

template <typename T>
__global__ void quantize(const T *input, T *output, size_t elements) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  float value = index < elements ? static_cast<float>(input[index]) : 0.0f;
  int exponent = quantization_exponent(value);
  float scale = exp2f(exponent);
  float normalized = ldexpf(value, -static_cast<int>(exponent));
  auto packed = __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
  if (index < elements)
    output[index] = static_cast<T>(e4m3_to_float(packed) * scale);
}

template <typename T>
__global__ void quantize_packed(const T *input, uint8_t *output,
                                uint8_t *scales, size_t elements) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  float value = index < elements ? static_cast<float>(input[index]) : 0.0f;
  int exponent = quantization_exponent(value);
  if (index < elements) {
    output[index] = __nv_cvt_float_to_fp8(ldexpf(value, -exponent),
                                          __NV_SATFINITE, __NV_E4M3);
    if (threadIdx.x % kGroup == 0)
      scales[index / kGroup] = exponent + kE8M0ExponentBias;
  }
}

template <typename T>
__global__ void dequantize(const uint8_t *weight, const uint8_t *scales,
                           T *output, size_t elements) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements)
    output[index] = static_cast<T>(e4m3_to_float(weight[index]) *
                                   e8m0_to_float_ocp(scales[index / kGroup]));
}

template <typename T>
__global__ void gemv(const T *input, const uint8_t *weight,
                     const uint8_t *scales, const uint32_t *indices, T *output,
                     int rows, int n, int k, int topk, bool input_per_expert) {
  size_t row =
      (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kGroup;
  size_t output_count = static_cast<size_t>(rows) * topk * n;
  if (row >= output_count)
    return;
  int lane = threadIdx.x % kGroup;
  int column = row % n;
  int slot = (row / n) % topk;
  int token = row / (static_cast<size_t>(n) * topk);
  int expert = indices ? indices[token * topk + slot] : 0;
  int input_row = input_per_expert ? token * topk + slot : token;
  size_t weight_row = (static_cast<size_t>(expert) * n + column) * k;
  float sum = 0.0f;
  for (int offset = lane; offset < k; offset += kGroup) {
    float w = e4m3_to_float(weight[weight_row + offset]);
    float scale = e8m0_to_float_ocp(scales[(weight_row + offset) / kGroup]);
    sum = fmaf(
        static_cast<float>(input[static_cast<size_t>(input_row) * k + offset]),
        w * scale, sum);
  }
  for (int offset = kGroup / 2; offset; offset /= 2)
    sum += __shfl_down_sync(kWarpMask, sum, offset);
  if (lane == 0)
    output[row] = static_cast<T>(sum);
}

template <typename T, int WarpsPerRow>
__global__ void gemv_vector(const T *input, const uint8_t *weight,
                            const uint8_t *scales, const uint32_t *indices,
                            T *output, int n, int k, int topk,
                            bool input_per_expert) {
  constexpr int kWarps = kGemvThreads / kGroup;
  constexpr int kRows = kWarps / WarpsPerRow;
  __shared__ float sums[kWarps];
  int warp = threadIdx.x / kGroup;
  int lane = threadIdx.x % kGroup;
  int row_warp = warp % WarpsPerRow;
  size_t row = static_cast<size_t>(blockIdx.x) * kRows + warp / WarpsPerRow;
  size_t output_count = static_cast<size_t>(topk) * n;
  float sum = 0.0f;
  if (row < output_count) {
    int column = row % n;
    int slot = row / n;
    int expert = indices ? indices[slot] : 0;
    size_t input_row = input_per_expert ? slot : 0;
    size_t weight_row = (static_cast<size_t>(expert) * n + column) * k;
    int offset = (row_warp * kGroup + lane) * kGemvVector;
    for (; offset < k; offset += kGroup * WarpsPerRow * kGemvVector) {
      size_t index = weight_row + offset;
      uint32_t packed = *reinterpret_cast<const uint32_t *>(weight + index);
      float scale = e8m0_to_float_ocp(scales[index / kGroup]);
      __half2 low = __nv_cvt_fp8x2_to_halfraw2(packed & 0xffffu, __NV_E4M3);
      __half2 high = __nv_cvt_fp8x2_to_halfraw2(packed >> 16, __NV_E4M3);
      float2 low_values = __half22float2(low);
      float2 high_values = __half22float2(high);
      const T *x = input + input_row * k + offset;
      sum = fmaf(static_cast<float>(x[0]), low_values.x * scale, sum);
      sum = fmaf(static_cast<float>(x[1]), low_values.y * scale, sum);
      sum = fmaf(static_cast<float>(x[2]), high_values.x * scale, sum);
      sum = fmaf(static_cast<float>(x[3]), high_values.y * scale, sum);
    }
  }
  for (int offset = kGroup / 2; offset; offset /= 2)
    sum += __shfl_down_sync(kWarpMask, sum, offset);
  if (lane == 0)
    sums[warp] = sum;
  __syncthreads();
  if (row_warp == 0) {
    sum = lane < WarpsPerRow ? sums[warp + lane] : 0.0f;
    for (int offset = kGroup / 2; offset; offset /= 2)
      sum += __shfl_down_sync(kWarpMask, sum, offset);
    if (row < output_count && lane == 0)
      output[row] = static_cast<T>(sum);
  }
}

template <typename T>
__global__ void gemm(const T *input, const uint8_t *weight,
                     const uint8_t *scales, T *output, int m, int n, int k) {
#if __CUDA_ARCH__ >= 800
  namespace wmma = nvcuda::wmma;
  // BF16 staging avoids narrowing E8M0-scaled weights to the F16 exponent
  // range.
  __shared__ __align__(32) __nv_bfloat16 a[kTileM * kGroup];
  __shared__ __align__(32) __nv_bfloat16 b[kTileN * kGroup];
  __shared__ __align__(32) float c[kTileN / kTileM][kTileM * kTileM];
  int warp = threadIdx.x / kGroup;
  int row_base = blockIdx.y * kTileM;
  int col_base = blockIdx.x * kTileN;
  wmma::fragment<wmma::accumulator, kTileM, kTileM, kMmaK, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (int group = 0; group < k; group += kGroup) {
    for (int index = threadIdx.x; index < kTileM * kGroup; index += kThreads) {
      int row = row_base + index / kGroup;
      a[index] = static_cast<__nv_bfloat16>(
          row < m ? static_cast<float>(input[static_cast<size_t>(row) * k +
                                             group + index % kGroup])
                  : 0.0f);
    }
    for (int index = threadIdx.x; index < kTileN * kGroup; index += kThreads) {
      int column = col_base + index / kGroup;
      size_t offset = static_cast<size_t>(column) * k + group + index % kGroup;
      b[index] = column < n ? static_cast<__nv_bfloat16>(
                                  e4m3_to_float(weight[offset]) *
                                  e8m0_to_float_ocp(scales[offset / kGroup]))
                            : static_cast<__nv_bfloat16>(0.0f);
    }
    __syncthreads();
    for (int offset = 0; offset < kGroup; offset += kMmaK) {
      wmma::fragment<wmma::matrix_a, kTileM, kTileM, kMmaK, __nv_bfloat16,
                     wmma::row_major>
          af;
      wmma::fragment<wmma::matrix_b, kTileM, kTileM, kMmaK, __nv_bfloat16,
                     wmma::col_major>
          bf;
      wmma::load_matrix_sync(af, a + offset, kGroup);
      wmma::load_matrix_sync(bf, b + warp * kTileM * kGroup + offset, kGroup);
      wmma::mma_sync(accumulator, af, bf, accumulator);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(c[warp], accumulator, kTileM, wmma::mem_row_major);
  __syncthreads();
  for (int index = threadIdx.x; index < kTileM * kTileN; index += kThreads) {
    int row = index / kTileN;
    int column = index % kTileN;
    if (row_base + row < m && col_base + column < n)
      output[static_cast<size_t>(row_base + row) * n + col_base + column] =
          static_cast<T>(c[column / kTileM][row * kTileM + column % kTileM]);
  }
#endif
}

template <typename T>
int launch_quantize(const T *input, T *output, size_t elements,
                    cudaStream_t stream) {
  quantize<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      input, output, elements);
  return static_cast<int>(cudaGetLastError());
}

template <typename T>
int launch_dequantize(const uint8_t *weight, const uint8_t *scales, T *output,
                      size_t elements, cudaStream_t stream) {
  dequantize<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      weight, scales, output, elements);
  return static_cast<int>(cudaGetLastError());
}

template <typename T>
int launch_quantize_packed(const T *input, uint8_t *output, uint8_t *scales,
                           size_t elements, cudaStream_t stream) {
  quantize_packed<<<(elements + kThreads - 1) / kThreads, kThreads, 0,
                    stream>>>(input, output, scales, elements);
  return static_cast<int>(cudaGetLastError());
}

template <typename T>
int launch_matmul(const T *input, const uint8_t *weight, const uint8_t *scales,
                  const uint32_t *indices, T *output, int rows, int n, int k,
                  int topk, bool input_per_expert, bool tensor_cores,
                  cudaStream_t stream) {
  if constexpr (!std::is_same_v<T, float>) {
    if (tensor_cores && !indices && rows > 1) {
      dim3 grid((n + kTileN - 1) / kTileN, (rows + kTileM - 1) / kTileM);
      gemm<<<grid, kThreads, 0, stream>>>(input, weight, scales, output, rows,
                                          n, k);
      return static_cast<int>(cudaGetLastError());
    }
  }
  size_t elements = static_cast<size_t>(rows) * topk * n;
  if (rows == 1 && k >= kGemvMinK && k % kGroup == 0 &&
      reinterpret_cast<uintptr_t>(weight) % alignof(uint32_t) == 0) {
    if (k <= kGemvShortK) {
      constexpr int kWarpsPerRow = 4;
      constexpr int kRows = kGemvThreads / kGroup / kWarpsPerRow;
      gemv_vector<T, kWarpsPerRow>
          <<<(elements + kRows - 1) / kRows, kGemvThreads, 0, stream>>>(
              input, weight, scales, indices, output, n, k, topk,
              input_per_expert);
    } else {
      constexpr int kWarpsPerRow = 8;
      gemv_vector<T, kWarpsPerRow><<<elements, kGemvThreads, 0, stream>>>(
          input, weight, scales, indices, output, n, k, topk, input_per_expert);
    }
    return static_cast<int>(cudaGetLastError());
  }
  gemv<<<(elements * kGroup + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      input, weight, scales, indices, output, rows, n, k, topk,
      input_per_expert);
  return static_cast<int>(cudaGetLastError());
}
} // namespace

#define MXFP8_EXPORT(T, SUFFIX)                                                \
  extern "C" int mxfp8_quantize_packed_##SUFFIX(const T *x, uint8_t *y,        \
                                                uint8_t *s, size_t elements,   \
                                                cudaStream_t stream) {         \
    return launch_quantize_packed(x, y, s, elements, stream);                  \
  }                                                                            \
  extern "C" int mxfp8_quantize_##SUFFIX(const T *x, T *y, size_t elements,    \
                                         cudaStream_t stream) {                \
    return launch_quantize(x, y, elements, stream);                            \
  }                                                                            \
  extern "C" int mxfp8_dequantize_##SUFFIX(const uint8_t *w, const uint8_t *s, \
                                           T *y, size_t elements,              \
                                           cudaStream_t stream) {              \
    return launch_dequantize(w, s, y, elements, stream);                       \
  }                                                                            \
  extern "C" int mxfp8_matmul_##SUFFIX(                                        \
      const T *x, const uint8_t *w, const uint8_t *s, const uint32_t *ids,     \
      T *y, int rows, int n, int k, int topk, bool input_per_expert,           \
      bool tensor_cores, cudaStream_t stream) {                                \
    return launch_matmul(x, w, s, ids, y, rows, n, k, topk, input_per_expert,  \
                         tensor_cores, stream);                                \
  }

MXFP8_EXPORT(float, f32)
MXFP8_EXPORT(half, f16)
MXFP8_EXPORT(__nv_bfloat16, bf16)
