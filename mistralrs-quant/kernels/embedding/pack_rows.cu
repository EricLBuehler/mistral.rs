#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
constexpr int kGroup = 32;
constexpr int kWarps = 8;

template <typename T>
__global__ void
quantize_rows(const T *__restrict__ source, unsigned char *__restrict__ data,
              __half *__restrict__ scales, int groups, int bits) {
  const int group = blockIdx.x * kWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (group >= groups) {
    return;
  }
  const float value = static_cast<float>(source[(size_t)group * kGroup + lane]);
  float amax = fabsf(value);
#pragma unroll
  for (int offset = kGroup / 2; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
  }
  const float qmax = bits == 8 ? 127.0f : 7.0f;
  const __half scale = __float2half_rn(__fdiv_rn(amax, qmax));
  if (lane == 0) {
    scales[group] = scale;
  }
  const float d = __half2float(scale);
  const float inv = d > 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
  // Match the host quantizer's ties away from zero and rounded f16 scale.
  const int q = (int)fminf(qmax, fmaxf(-qmax, roundf(__fmul_rn(value, inv))));
  if (bits == 8) {
    data[(size_t)group * kGroup + lane] = (unsigned char)(signed char)q;
  } else {
    const int next = __shfl_down_sync(0xffffffff, q, 1);
    if ((lane & 1) == 0) {
      data[(size_t)group * (kGroup / 2) + lane / 2] =
          (unsigned char)((q + 8) | ((next + 8) << 4));
    }
  }
}

} // namespace

extern "C" int mistralrs_pack_embedding_rows(const void *source,
                                             unsigned char *data, void *scales,
                                             int groups, int bits, int dtype,
                                             cudaStream_t stream) {
  if (groups == 0)
    return 0;
  const int blocks = (groups + kWarps - 1) / kWarps;
  if (dtype == 0) {
    quantize_rows<__half><<<blocks, kWarps * kGroup, 0, stream>>>(
        static_cast<const __half *>(source), data,
        static_cast<__half *>(scales), groups, bits);
  } else if (dtype == 1) {
    quantize_rows<__nv_bfloat16><<<blocks, kWarps * kGroup, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(source), data,
        static_cast<__half *>(scales), groups, bits);
  } else {
    quantize_rows<float><<<blocks, kWarps * kGroup, 0, stream>>>(
        static_cast<const float *>(source), data, static_cast<__half *>(scales),
        groups, bits);
  }
  return static_cast<int>(cudaGetLastError());
}
