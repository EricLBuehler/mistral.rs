#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace routed_gemv {

constexpr int kWarp = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarp * kWarps;
constexpr unsigned kFullMask = 0xffffffffu;
// Units each lane loads before consuming any, so every warp keeps several weight vectors in flight.
constexpr int kDepth = 2;

enum Mode : int { kPlain = 0, kGlu = 1, kReduce = 2 };

// Matches mistralrs_quant::GluActivationType.
enum Activation : int {
  kSilu = 0,
  kGeluTanh = 1,
  kRelu = 2,
  kGeluErf = 3,
  kSigmoid = 4
};

// Codec-interpreted operand: strides are bytes for `data`, elements for the rest.
struct Operand {
  const uint8_t *data;
  const void *scales;
  const void *extra;
  const void *bias;
  long long data_expert;
  long long data_row;
  long long scale_expert;
  long long scale_row;
  long long extra_expert;
  long long extra_row;
  long long bias_expert;
  long long plane;
  int group;
  int block_rows;
};

struct Launch {
  Operand a;
  Operand b;
  const void *input;
  const uint32_t *ids;
  const float *routing;
  void *output;
  long long input_stride;
  int tokens;
  int topk;
  int n;
  int k;
  int input_routes;
  int activation;
  int prefetch;
};

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T> __device__ __forceinline__ T from_float(float x) {
  if constexpr (std::is_same_v<T, float>)
    return x;
  else if constexpr (std::is_same_v<T, __half>)
    return __float2half(x);
  else
    return __float2bfloat16(x);
}

__device__ __forceinline__ float activate(float g, int activation) {
  switch (activation) {
  case kGeluTanh:
    return 0.5f * g *
           (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
  case kRelu:
    return fmaxf(g, 0.0f);
  case kGeluErf:
    return 0.5f * g * (1.0f + erff(g * 0.70710678118f));
  case kSigmoid:
    return 1.0f / (1.0f + expf(-g));
  default:
    return g / (1.0f + expf(-g));
  }
}

__device__ __forceinline__ void prefetch_l2(const void *ptr) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
}

// Pairs of 16-bit values unpack two at a time; f32 rows load four floats per vector.
template <typename T, int N>
__device__ __forceinline__ void load_floats(const T *__restrict__ src,
                                            float *dst) {
  if constexpr (std::is_same_v<T, float>) {
    static_assert(N % 4 == 0);
#pragma unroll
    for (int i = 0; i < N; i += 4) {
      const float4 v = __ldg(reinterpret_cast<const float4 *>(src + i));
      dst[i] = v.x;
      dst[i + 1] = v.y;
      dst[i + 2] = v.z;
      dst[i + 3] = v.w;
    }
  } else {
    static_assert(N % 8 == 0);
#pragma unroll
    for (int i = 0; i < N; i += 8) {
      const uint4 v = __ldg(reinterpret_cast<const uint4 *>(src + i));
      const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if constexpr (std::is_same_v<T, __half>) {
          const float2 f =
              __half22float2(*reinterpret_cast<const __half2 *>(&words[j]));
          dst[i + 2 * j] = f.x;
          dst[i + 2 * j + 1] = f.y;
        } else {
          const float2 f = __bfloat1622float2(
              *reinterpret_cast<const __nv_bfloat162 *>(&words[j]));
          dst[i + 2 * j] = f.x;
          dst[i + 2 * j + 1] = f.y;
        }
      }
    }
  }
}

// Codecs whose activation is the raw float row: one lane owns `Unit` consecutive input columns.
template <int Unit> struct FloatAct {
  static constexpr int kUnit = Unit;
  static constexpr int kUnitLanes = 1;
  struct Act {
    float v[Unit];
  };
  template <typename In>
  __device__ __forceinline__ static Act load_act(const In *__restrict__ x,
                                                 int unit, int) {
    Act act;
    load_floats<In, Unit>(x + static_cast<long long>(unit) * Unit, act.v);
    return act;
  }
};

template <int Unit, int Bytes>
__device__ __forceinline__ void load_bytes(const uint8_t *__restrict__ src,
                                           uint32_t *dst) {
  static_assert(Bytes % 4 == 0);
  if constexpr (Bytes % 16 == 0) {
#pragma unroll
    for (int i = 0; i < Bytes / 16; ++i) {
      const uint4 v = __ldg(reinterpret_cast<const uint4 *>(src) + i);
      dst[4 * i] = v.x;
      dst[4 * i + 1] = v.y;
      dst[4 * i + 2] = v.z;
      dst[4 * i + 3] = v.w;
    }
  } else if constexpr (Bytes % 8 == 0) {
#pragma unroll
    for (int i = 0; i < Bytes / 8; ++i) {
      const uint2 v = __ldg(reinterpret_cast<const uint2 *>(src) + i);
      dst[2 * i] = v.x;
      dst[2 * i + 1] = v.y;
    }
  } else {
#pragma unroll
    for (int i = 0; i < Bytes / 4; ++i)
      dst[i] = __ldg(reinterpret_cast<const uint32_t *>(src) + i);
  }
}

// Each R-row slab of a warp-subgroup streams the routed expert's rows once; Mode picks the epilogue.
template <class C, typename In, typename Out, int L, int R, int M>
__global__ void __launch_bounds__(kThreads) routed_gemv_kernel(Launch a) {
  static_assert(kWarp % L == 0 && L % C::kUnitLanes == 0);
  constexpr int kGroups = kWarp / L;
  constexpr int kPerIter = L / C::kUnitLanes;
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int sub = lane % L;
  const int row0 = ((blockIdx.x * kWarps + warp) * kGroups + lane / L) * R;
  const int units = a.k / C::kUnit;
  const int first_unit = sub / C::kUnitLanes;
  const int part = sub % C::kUnitLanes;
  const int slots = M == kReduce ? a.topk : 1;
  const int token = M == kReduce ? blockIdx.y : blockIdx.y / a.topk;
  float total[R] = {};
  float up_total[R] = {};
  for (int slot = 0; slot < slots; ++slot) {
    const int route = M == kReduce ? token * a.topk + slot : blockIdx.y;
    const int expert = static_cast<int>(a.ids[route]);
    const In *x = static_cast<const In *>(a.input) +
                  (a.input_routes == 1 ? token : route) * a.input_stride;
    typename C::Row rows[R];
    typename C::Row up_rows[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = min(row0 + r, a.n - 1);
      rows[r] = C::row(a.a, expert, row);
      if constexpr (M == kGlu)
        up_rows[r] = C::row(a.b, expert, row);
    }
    float sums[R] = {};
    float up_sums[R] = {};
    for (int base = first_unit; base < units; base += kPerIter * kDepth) {
      typename C::Frag frags[kDepth][R];
      typename C::Frag up_frags[kDepth][R];
#pragma unroll
      for (int d = 0; d < kDepth; ++d) {
        const int unit = min(base + d * kPerIter, units - 1);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          frags[d][r] = C::load(rows[r], unit);
          if constexpr (M == kGlu)
            up_frags[d][r] = C::load(up_rows[r], unit);
        }
      }
      const int ahead = base + kPerIter * kDepth;
      if (a.prefetch && ahead < units) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
          prefetch_l2(C::prefetch(rows[r], ahead));
          if constexpr (M == kGlu)
            prefetch_l2(C::prefetch(up_rows[r], ahead));
        }
      }
#pragma unroll
      for (int d = 0; d < kDepth; ++d) {
        const int unit = base + d * kPerIter;
        if (unit >= units)
          break;
        const auto act = C::template load_act<In>(x, unit, part);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          sums[r] += C::dot(frags[d][r], act);
          if constexpr (M == kGlu)
            up_sums[r] += C::dot(up_frags[d][r], act);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int offset = L / 2; offset > 0; offset /= 2) {
        sums[r] += __shfl_xor_sync(kFullMask, sums[r], offset);
        if constexpr (M == kGlu)
          up_sums[r] += __shfl_xor_sync(kFullMask, up_sums[r], offset);
      }
      const int row = min(row0 + r, a.n - 1);
      float value = C::finish(rows[r], sums[r]);
      if (a.a.bias)
        value += to_float(static_cast<const Out *>(
            a.a.bias)[expert * a.a.bias_expert + row]);
      if constexpr (M == kGlu) {
        float up = C::finish(up_rows[r], up_sums[r]);
        if (a.b.bias)
          up += to_float(static_cast<const Out *>(
              a.b.bias)[expert * a.b.bias_expert + row]);
        up_total[r] = up;
      }
      total[r] = M == kReduce ? fmaf(a.routing[route], value, total[r]) : value;
    }
  }
  if (sub != 0)
    return;
  Out *out = static_cast<Out *>(a.output) +
             static_cast<long long>(blockIdx.y) * a.n;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (row0 + r >= a.n)
      continue;
    float value = total[r];
    if constexpr (M == kGlu)
      value = activate(value, a.activation) * up_total[r];
    out[row0 + r] = from_float<Out>(value);
  }
}

template <class C, typename In, typename Out, int L, int R, int M>
cudaError_t launch_shape(const Launch &a, cudaStream_t stream) {
  constexpr int rows_per_block = kWarps * (kWarp / L) * R;
  const dim3 grid((a.n + rows_per_block - 1) / rows_per_block,
                  M == kReduce ? a.tokens : a.tokens * a.topk);
  routed_gemv_kernel<C, In, Out, L, R, M><<<grid, kThreads, 0, stream>>>(a);
  return cudaGetLastError();
}

// `lanes` is the host's pick for how many lanes share one row; rows per subgroup scale inversely.
template <class C, typename In, typename Out, int M>
cudaError_t launch_lanes(const Launch &a, int lanes, cudaStream_t stream) {
  switch (lanes) {
  case 8:
    if constexpr (8 % C::kUnitLanes == 0)
      return launch_shape<C, In, Out, 8, 1, M>(a, stream);
    break;
  case 16:
    if constexpr (16 % C::kUnitLanes == 0)
      return launch_shape<C, In, Out, 16, 1, M>(a, stream);
    break;
  default:
    return launch_shape<C, In, Out, 32, 2, M>(a, stream);
  }
  return cudaErrorInvalidValue;
}

template <class C, typename In, typename Out>
cudaError_t launch_mode(const Launch &a, int mode, int lanes,
                        cudaStream_t stream) {
  switch (mode) {
  case kGlu:
    return launch_lanes<C, In, Out, kGlu>(a, lanes, stream);
  case kReduce:
    return launch_lanes<C, In, Out, kReduce>(a, lanes, stream);
  default:
    return launch_lanes<C, In, Out, kPlain>(a, lanes, stream);
  }
}

enum DType : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

// Float-activation codecs take the input and output in the activation dtype.
template <class C>
cudaError_t launch_float(const Launch &a, int dtype, int mode, int lanes,
                         cudaStream_t stream) {
  switch (dtype) {
  case kF16:
    return launch_mode<C, __half, __half>(a, mode, lanes, stream);
  case kBF16:
    return launch_mode<C, __nv_bfloat16, __nv_bfloat16>(a, mode, lanes,
                                                        stream);
  default:
    return launch_mode<C, float, float>(a, mode, lanes, stream);
  }
}

} // namespace routed_gemv
