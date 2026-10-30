#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if CUDART_VERSION >= 13000
#include "../common/fp_codecs.cuh"
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#endif

struct RoutedActivation {
  const void *input;
  const float *global;
  const int32_t *sorted_ids;
  const int32_t *expert_ids;
  const int32_t *num_padded;
  uint8_t *packed;
  uint8_t *scales;
  void *aligned;
  int rows;
  int routes;
  int k;
  int input_stride;
  int alignment;
  int a4;
};

struct SourceActivation {
  uint8_t *packed;
  uint8_t *scales;
};

namespace {
constexpr int kThreads = 256;
constexpr int kGatherThreads = 128;
constexpr int kMaxCopyBytes = 16;
constexpr int kGroup = 16;
constexpr int kLanes = kGroup / 2;
constexpr int kWarp = 32;
constexpr unsigned kGroupMask = (1u << kLanes) - 1;
constexpr float kFp4Max = 6.0f;
constexpr float kFp8Max = 448.0f;

#if CUDART_VERSION >= 13000
template <typename T>
__device__ __forceinline__ void quantize_row(RoutedActivation a, int row,
                                             size_t source, float global) {
  const int lane = threadIdx.x % kLanes;
  const unsigned mask = kGroupMask << ((threadIdx.x % kWarp) / kLanes * kLanes);
  const T *input = static_cast<const T *>(a.input);
  for (int group = threadIdx.x / kLanes; group < a.k / kGroup;
       group += blockDim.x / kLanes) {
    const int column = group * kGroup + lane * 2;
    const float first =
        __fdiv_rn(static_cast<float>(input[source + column]), global);
    const float second =
        __fdiv_rn(static_cast<float>(input[source + column + 1]), global);
    float maximum = fmaxf(fabsf(first), fabsf(second));
#pragma unroll
    for (int offset = kLanes / 2; offset; offset /= 2)
      maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, offset, kLanes));
    const uint8_t scale_code = __nv_cvt_float_to_fp8(
        fminf(__fdiv_rn(maximum, kFp4Max), kFp8Max), __NV_SATFINITE, __NV_E4M3);
    const float scale = e4m3_to_float(scale_code);
    const float denominator = scale == 0.0f ? 1.0f : scale;
    const uint8_t packed =
        __nv_cvt_float2_to_fp4x2(make_float2(__fdiv_rn(first, denominator),
                                             __fdiv_rn(second, denominator)),
                                 __NV_E2M1, cudaRoundNearest);
    a.packed[size_t(row) * (a.k / 2) + column / 2] = packed;
    if (lane == 0)
      a.scales[size_t(row) * (a.k / kGroup) + group] = scale_code;
  }
}
template <typename T> __global__ void quantize_routed(RoutedActivation a) {
  const int row = blockIdx.x;
  if (row >= *a.num_padded)
    return;
  const int route = a.sorted_ids[row];
  const bool valid =
      static_cast<unsigned>(route) < static_cast<unsigned>(a.routes);
  if (!valid) {
    for (int byte = threadIdx.x; byte < a.k / 2; byte += blockDim.x)
      a.packed[size_t(row) * (a.k / 2) + byte] = 0;
    for (int group = threadIdx.x; group < a.k / kGroup; group += blockDim.x)
      a.scales[size_t(row) * (a.k / kGroup) + group] = 0;
    return;
  }
  const int expert = a.expert_ids[row / a.alignment];
  quantize_row<T>(a, row, size_t(route / a.input_stride) * a.k,
                  a.global[expert]);
}

template <typename T> __global__ void quantize_source(RoutedActivation a) {
  const int row = blockIdx.x;
  quantize_row<T>(a, row, size_t(row) * a.k, a.global[0]);
}

int copy_width(const void *source, const void *destination, int row_bytes) {
  const uintptr_t alignment = reinterpret_cast<uintptr_t>(source) |
                              reinterpret_cast<uintptr_t>(destination) |
                              uintptr_t(row_bytes);
  for (int width = kMaxCopyBytes; width > 1; width /= 2)
    if ((alignment & uintptr_t(width - 1)) == 0)
      return width;
  return 1;
}

template <typename T>
__device__ __forceinline__ void copy_vector_row(const uint8_t *source,
                                                uint8_t *destination, int bytes,
                                                bool valid) {
  const T *input = reinterpret_cast<const T *>(source);
  T *output = reinterpret_cast<T *>(destination);
  for (int item = threadIdx.x; item < bytes / int(sizeof(T));
       item += blockDim.x)
    output[item] = valid ? input[item] : T{};
}

__device__ __forceinline__ void copy_aligned_row(const uint8_t *source,
                                                 uint8_t *destination,
                                                 int bytes, int width,
                                                 bool valid) {
  if (width == sizeof(uint4))
    copy_vector_row<uint4>(source, destination, bytes, valid);
  else if (width == sizeof(uint64_t))
    copy_vector_row<uint64_t>(source, destination, bytes, valid);
  else if (width == sizeof(uint32_t))
    copy_vector_row<uint32_t>(source, destination, bytes, valid);
  else if (width == sizeof(uint16_t))
    copy_vector_row<uint16_t>(source, destination, bytes, valid);
  else
    copy_vector_row<uint8_t>(source, destination, bytes, valid);
}

__global__ void gather_packed(RoutedActivation a, SourceActivation source,
                              int packed_width, int scale_width) {
  const int row = blockIdx.x;
  if (row >= *a.num_padded)
    return;
  const int route = a.sorted_ids[row];
  const bool valid =
      static_cast<unsigned>(route) < static_cast<unsigned>(a.routes);
  const size_t input_row = valid ? size_t(route / a.input_stride) : 0;
  const int packed_bytes = a.k / 2;
  const int scale_bytes = a.k / kGroup;
  copy_aligned_row(source.packed + input_row * packed_bytes,
                   a.packed + size_t(row) * packed_bytes, packed_bytes,
                   packed_width, valid);
  copy_aligned_row(source.scales + input_row * scale_bytes,
                   a.scales + size_t(row) * scale_bytes, scale_bytes,
                   scale_width, valid);
}

#endif

template <typename T> __global__ void gather_routed(RoutedActivation a) {
  const int row = blockIdx.x;
  if (row >= *a.num_padded)
    return;
  const int route = a.sorted_ids[row];
  const bool valid =
      static_cast<unsigned>(route) < static_cast<unsigned>(a.routes);
  const T *input = static_cast<const T *>(a.input);
  T *output = static_cast<T *>(a.aligned) + size_t(row) * a.k;
  const size_t source = valid ? size_t(route / a.input_stride) * a.k : 0;
  for (int column = threadIdx.x; column < a.k; column += blockDim.x)
    output[column] = valid ? input[source + column] : static_cast<T>(0.0f);
}

template <typename T>
int run(const RoutedActivation *args, cudaStream_t stream) {
  const RoutedActivation &a = *args;
  if (a.a4) {
#if CUDART_VERSION >= 13000
    quantize_routed<T><<<a.rows, kThreads, 0, stream>>>(a);
#else
    return cudaErrorNotSupported;
#endif
  } else {
    gather_routed<T><<<a.rows, kThreads, 0, stream>>>(a);
  }
  return cudaGetLastError();
}
template <typename T>
int run_source(const RoutedActivation *args, const SourceActivation *source,
               cudaStream_t stream) {
#if CUDART_VERSION >= 13000
  const RoutedActivation &a = *args;
  RoutedActivation input = a;
  input.packed = source->packed;
  input.scales = source->scales;
  quantize_source<T><<<a.routes / a.input_stride, kThreads, 0, stream>>>(input);
  const int packed_width = copy_width(source->packed, a.packed, a.k / 2);
  const int scale_width = copy_width(source->scales, a.scales, a.k / kGroup);
  gather_packed<<<a.rows, kGatherThreads, 0, stream>>>(a, *source, packed_width,
                                                       scale_width);
  return cudaGetLastError();
#else
  return cudaErrorNotSupported;
#endif
}
} // namespace

extern "C" int mistralrs_nvfp4_prepare_f16(const RoutedActivation *a,
                                           cudaStream_t s) {
  return run<__half>(a, s);
}
extern "C" int mistralrs_nvfp4_prepare_bf16(const RoutedActivation *a,
                                            cudaStream_t s) {
  return run<__nv_bfloat16>(a, s);
}
extern "C" int mistralrs_nvfp4_prepare_f32(const RoutedActivation *a,
                                           cudaStream_t s) {
  return run<float>(a, s);
}

extern "C" int mistralrs_nvfp4_prepare_source_f16(
    const RoutedActivation *a, const SourceActivation *source, cudaStream_t s) {
  return run_source<__half>(a, source, s);
}

extern "C" int mistralrs_nvfp4_prepare_source_bf16(
    const RoutedActivation *a, const SourceActivation *source, cudaStream_t s) {
  return run_source<__nv_bfloat16>(a, source, s);
}

extern "C" int mistralrs_nvfp4_prepare_source_f32(
    const RoutedActivation *a, const SourceActivation *source, cudaStream_t s) {
  return run_source<float>(a, source, s);
}
