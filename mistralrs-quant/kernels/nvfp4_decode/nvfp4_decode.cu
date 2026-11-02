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

namespace {
constexpr int kWarp = 32;
constexpr int kBlock = 16;
constexpr int kQuantThreads = 128;
constexpr int kWarps = 4;
constexpr int kOutputs = 4;
constexpr float kFp4Max = 6.0f;
constexpr float kFp8Max = 448.0f;
constexpr float kIntegerScale = 0.25f;
constexpr unsigned kMask = 0xffffffffu;

struct Args {
  const void *input;
  const uint8_t *weight;
  const uint8_t *scales;
  const float *weight_global;
  const float *input_global;
  const uint32_t *indices;
  uint8_t *quantized;
  uint8_t *input_scales;
  void *output;
  int routes;
  int n;
  int k;
  int input_stride;
};

#if CUDART_VERSION >= 13000
__device__ int unpack4(uint32_t nibble) {
  const uint32_t magnitude =
      __byte_perm(kE2M1x2Pos0, kE2M1x2Pos1, nibble & 0x7777u);
  const uint32_t negative = __byte_perm(0, kMask, (nibble >> 1) & 0x4444u);
  return static_cast<int>(__vsub4(magnitude ^ negative, negative));
}

__device__ uint2 packed_block(const uint8_t *data) {
  if (reinterpret_cast<uintptr_t>(data) % alignof(uint2) == 0)
    return *reinterpret_cast<const uint2 *>(data);
  uint2 value = {};
#pragma unroll
  for (int i = 0; i < sizeof(uint32_t); ++i) {
    value.x |= uint32_t(data[i]) << (i * 8);
    value.y |= uint32_t(data[i + sizeof(uint32_t)]) << (i * 8);
  }
  return value;
}

template <typename T> __global__ void quantize(Args a) {
  const int route = blockIdx.y;
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= a.k / kBlock)
    return;
  const int expert = a.indices ? a.indices[route] : 0;
  const float global = a.input_global[expert];
  const T *input = static_cast<const T *>(a.input) +
                   (size_t(route) / a.input_stride) * a.k + group * kBlock;
  float values[kBlock];
  float maximum = 0.0f;
#pragma unroll
  for (int i = 0; i < kBlock; ++i) {
    values[i] = __fdiv_rn(static_cast<float>(input[i]), global);
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  const uint8_t scale_byte = __nv_cvt_float_to_fp8(
      fminf(__fdiv_rn(maximum, kFp4Max), kFp8Max), __NV_SATFINITE, __NV_E4M3);
  const float scale = e4m3_to_float(scale_byte);
  const float denominator = scale == 0.0f ? 1.0f : scale;
  uint8_t *output = a.quantized + (size_t(route) * a.k + group * kBlock) / 2;
#pragma unroll
  for (int i = 0; i < kBlock; i += 2) {
    output[i / 2] = __nv_cvt_float2_to_fp4x2(
        make_float2(__fdiv_rn(values[i], denominator),
                    __fdiv_rn(values[i + 1], denominator)),
        __NV_E2M1, cudaRoundNearest);
  }
  a.input_scales[size_t(route) * (a.k / kBlock) + group] = scale_byte;
}

template <typename T> __global__ void gemv(Args a) {
  const int row = (blockIdx.x * kWarps + threadIdx.x / kWarp) * kOutputs;
  const int route = blockIdx.y;
  const int lane = threadIdx.x % kWarp;
  if (row >= a.n)
    return;
  const int expert = a.indices ? a.indices[route] : 0;
  const uint8_t *input = a.quantized + size_t(route) * a.k / 2;
  const uint8_t *input_scales = a.input_scales + size_t(route) * (a.k / kBlock);
  float sums[kOutputs] = {};
  for (int group = lane; group < a.k / kBlock; group += kWarp) {
    const uint2 codes = packed_block(input + group * (kBlock / 2));
    const int4 x = make_int4(unpack4(codes.x), unpack4(codes.x >> 16),
                             unpack4(codes.y), unpack4(codes.y >> 16));
    const float input_scale =
        e4m3_to_float(input_scales[group]) * kIntegerScale;
#pragma unroll
    for (int output = 0; output < kOutputs; ++output) {
      if (row + output >= a.n)
        continue;
      const size_t weight_row = size_t(expert) * a.n + row + output;
      const uint2 w = packed_block(a.weight + weight_row * (a.k / 2) +
                                   group * (kBlock / 2));
      int dot = __dp4a(x.x, unpack4(w.x), 0);
      dot = __dp4a(x.y, unpack4(w.x >> 16), dot);
      dot = __dp4a(x.z, unpack4(w.y), dot);
      dot = __dp4a(x.w, unpack4(w.y >> 16), dot);
      const float scale =
          e4m3_to_float(a.scales[weight_row * (a.k / kBlock) + group]);
      sums[output] = fmaf(float(dot), input_scale * scale, sums[output]);
    }
  }
#pragma unroll
  for (int output = 0; output < kOutputs; ++output) {
#pragma unroll
    for (int offset = kWarp / 2; offset; offset /= 2)
      sums[output] += __shfl_down_sync(kMask, sums[output], offset);
    if (lane == 0 && row + output < a.n) {
      float value = __fmul_rn(
          sums[output], a.weight_global[size_t(expert) * a.n + row + output]);
      value = __fmul_rn(value, a.input_global[expert]);
      static_cast<T *>(a.output)[size_t(route) * a.n + row + output] =
          static_cast<T>(value);
    }
  }
}

template <typename T> int run(const Args *args, cudaStream_t stream) {
  const Args &a = *args;
  quantize<T>
      <<<dim3((a.k / kBlock + kQuantThreads - 1) / kQuantThreads, a.routes),
         kQuantThreads, 0, stream>>>(a);
  gemv<T>
      <<<dim3((a.n + kWarps * kOutputs - 1) / (kWarps * kOutputs), a.routes),
         kWarps * kWarp, 0, stream>>>(a);
  return cudaGetLastError();
}
#else
template <typename T> int run(const Args *, cudaStream_t) {
  return cudaErrorNotSupported;
}
#endif
} // namespace

extern "C" int mistralrs_nvfp4_decode_f16(const Args *a, cudaStream_t s) {
  return run<__half>(a, s);
}
extern "C" int mistralrs_nvfp4_decode_bf16(const Args *a, cudaStream_t s) {
  return run<__nv_bfloat16>(a, s);
}
