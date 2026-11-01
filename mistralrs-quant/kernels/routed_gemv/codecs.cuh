#pragma once

#include "../common/fp_codecs.cuh"
#include "routed_gemv.cuh"

namespace routed_gemv {

__device__ __forceinline__ float fp4_dot8(uint32_t q, const float *x) {
  float w[8];
  e2m1x8_to_float_doubled(q, w);
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    sum = fmaf(w[i], x[i], sum);
  return sum;
}

__device__ __forceinline__ float fp8_dot4(uint32_t q, const float *x) {
  const float2 lo = __half22float2(__half2(
      __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(q & 0xffffu),
                                 __NV_E4M3)));
  const float2 hi = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(q >> 16), __NV_E4M3)));
  return fmaf(lo.x, x[0], fmaf(lo.y, x[1], fmaf(hi.x, x[2], hi.y * x[3])));
}

// Packed E2M1 [E, N, K/2] (low nibble first), E4M3 per-16 scales [E, N, K/16], f32 [E, N] globals.
template <int Unit> struct Nvfp4 : FloatAct<Unit> {
  static_assert(Unit == 16 || Unit == 32);
  static constexpr int kBlocks = Unit / 16;
  struct Row {
    const uint8_t *w;
    const uint8_t *s;
    float global;
  };
  struct Frag {
    uint32_t q[Unit / 8];
    uint8_t s[kBlocks];
  };
  __device__ __forceinline__ static Row row(const Operand &o, int expert,
                                            int n) {
    return {o.data + expert * o.data_expert + n * o.data_row,
            static_cast<const uint8_t *>(o.scales) + expert * o.scale_expert +
                n * o.scale_row,
            __ldg(static_cast<const float *>(o.extra) + expert * o.extra_expert +
                  n * o.extra_row)};
  }
  __device__ __forceinline__ static const void *prefetch(const Row &r,
                                                         int unit) {
    return r.w + unit * (Unit / 2);
  }
  __device__ __forceinline__ static Frag load(const Row &r, int unit) {
    Frag f;
    load_bytes<Unit, Unit / 2>(r.w + unit * (Unit / 2), f.q);
#pragma unroll
    for (int b = 0; b < kBlocks; ++b)
      f.s[b] = __ldg(r.s + unit * kBlocks + b);
    return f;
  }
  __device__ __forceinline__ static float
  dot(const Frag &f, const typename FloatAct<Unit>::Act &x) {
    float sum = 0.0f;
#pragma unroll
    for (int b = 0; b < kBlocks; ++b)
      sum = fmaf(e4m3_to_float(f.s[b]),
                 fp4_dot8(f.q[2 * b], x.v + 16 * b) +
                     fp4_dot8(f.q[2 * b + 1], x.v + 16 * b + 8),
                 sum);
    return sum;
  }
  __device__ __forceinline__ static float finish(const Row &r, float sum) {
    return sum * r.global * kE2M1x2Half;
  }
};

// Packed E2M1 [E, N, K/2] (low nibble first), E8M0 per-32 scales [E, N, K/32].
struct Mxfp4 : FloatAct<32> {
  struct Row {
    const uint8_t *w;
    const uint8_t *s;
  };
  struct Frag {
    uint32_t q[4];
    uint8_t s;
  };
  __device__ __forceinline__ static Row row(const Operand &o, int expert,
                                            int n) {
    return {o.data + expert * o.data_expert + n * o.data_row,
            static_cast<const uint8_t *>(o.scales) + expert * o.scale_expert +
                n * o.scale_row};
  }
  __device__ __forceinline__ static const void *prefetch(const Row &r,
                                                         int unit) {
    return r.w + unit * 16;
  }
  __device__ __forceinline__ static Frag load(const Row &r, int unit) {
    Frag f;
    load_bytes<32, 16>(r.w + unit * 16, f.q);
    f.s = __ldg(r.s + unit);
    return f;
  }
  __device__ __forceinline__ static float dot(const Frag &f, const Act &x) {
    return e8m0_to_float(f.s) *
           (fp4_dot8(f.q[0], x.v) + fp4_dot8(f.q[1], x.v + 8) +
            fp4_dot8(f.q[2], x.v + 16) + fp4_dot8(f.q[3], x.v + 24));
  }
  __device__ __forceinline__ static float finish(const Row &, float sum) {
    return sum * kE2M1x2Half;
  }
};

// E4M3 [E, N, K] with E8M0 per-32 scales [E, N, K/32].
struct Mxfp8 : FloatAct<32> {
  struct Row {
    const uint8_t *w;
    const uint8_t *s;
  };
  struct Frag {
    uint32_t q[8];
    uint8_t s;
  };
  __device__ __forceinline__ static Row row(const Operand &o, int expert,
                                            int n) {
    return {o.data + expert * o.data_expert + n * o.data_row,
            static_cast<const uint8_t *>(o.scales) + expert * o.scale_expert +
                n * o.scale_row};
  }
  __device__ __forceinline__ static const void *prefetch(const Row &r,
                                                         int unit) {
    return r.w + unit * 32;
  }
  __device__ __forceinline__ static Frag load(const Row &r, int unit) {
    Frag f;
    load_bytes<32, 32>(r.w + unit * 32, f.q);
    f.s = __ldg(r.s + unit);
    return f;
  }
  __device__ __forceinline__ static float dot(const Frag &f, const Act &x) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sum += fp8_dot4(f.q[i], x.v + 4 * i);
    return sum * e8m0_to_float_ocp(f.s);
  }
  __device__ __forceinline__ static float finish(const Row &, float sum) {
    return sum;
  }
};

// E4M3 [E, N, K] with f32 scales per [block_rows, group] block; group 0 means one scale per row.
struct Fp8 : FloatAct<16> {
  struct Row {
    const uint8_t *w;
    const float *s;
    int group;
  };
  struct Frag {
    uint32_t q[4];
    float s;
  };
  __device__ __forceinline__ static Row row(const Operand &o, int expert,
                                            int n) {
    return {o.data + expert * o.data_expert + n * o.data_row,
            static_cast<const float *>(o.scales) + expert * o.scale_expert +
                (n / o.block_rows) * o.scale_row,
            o.group};
  }
  __device__ __forceinline__ static const void *prefetch(const Row &r,
                                                         int unit) {
    return r.w + unit * 16;
  }
  __device__ __forceinline__ static Frag load(const Row &r, int unit) {
    Frag f;
    load_bytes<16, 16>(r.w + unit * 16, f.q);
    f.s = r.group ? __ldg(r.s + unit * 16 / r.group) : 1.0f;
    return f;
  }
  __device__ __forceinline__ static float dot(const Frag &f, const Act &x) {
    return f.s * (fp8_dot4(f.q[0], x.v) + fp8_dot4(f.q[1], x.v + 4) +
                  fp8_dot4(f.q[2], x.v + 8) + fp8_dot4(f.q[3], x.v + 12));
  }
  __device__ __forceinline__ static float finish(const Row &r, float sum) {
    return r.group ? sum : sum * __ldg(r.s);
  }
};

// Little-endian bitstream of `Bits`-bit codes [E, N, K*Bits/8], affine scales/biases [E, N, K/group].
template <int Bits, typename S> struct Afq {
  static constexpr int kUnit = 32;
  static constexpr int kUnitLanes = 1;
  static constexpr int kBytes = kUnit * Bits / 8;
  static constexpr uint32_t kMask = (1u << Bits) - 1;
  struct Act {
    float v[kUnit];
    float sum;
  };
  template <typename In>
  __device__ __forceinline__ static Act load_act(const In *__restrict__ x,
                                                 int unit, int) {
    Act act;
    load_floats<In, kUnit>(x + static_cast<long long>(unit) * kUnit, act.v);
    act.sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kUnit; ++i)
      act.sum += act.v[i];
    return act;
  }
  struct Row {
    const uint8_t *w;
    const S *s;
    const S *b;
    int group;
  };
  struct Frag {
    uint32_t q[kBytes / 4];
    S s;
    S b;
  };
  __device__ __forceinline__ static Row row(const Operand &o, int expert,
                                            int n) {
    const long long scale = expert * o.scale_expert + n * o.scale_row;
    return {o.data + expert * o.data_expert + n * o.data_row,
            static_cast<const S *>(o.scales) + scale,
            static_cast<const S *>(o.extra) + scale, o.group};
  }
  __device__ __forceinline__ static const void *prefetch(const Row &r,
                                                         int unit) {
    return r.w + unit * kBytes;
  }
  __device__ __forceinline__ static Frag load(const Row &r, int unit) {
    Frag f;
    load_bytes<kUnit, kBytes>(r.w + unit * kBytes, f.q);
    const int g = unit * kUnit / r.group;
    f.s = r.s[g];
    f.b = r.b[g];
    return f;
  }
  __device__ __forceinline__ static float dot(const Frag &f, const Act &x) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int bit = i * Bits;
      const int word = bit / 32;
      const int shift = bit % 32;
      uint32_t code = f.q[word] >> shift;
      if (shift + Bits > 32)
        code |= f.q[word + 1] << (32 - shift);
      sum = fmaf(static_cast<float>(code & kMask), x.v[i], sum);
    }
    return fmaf(to_float(f.s), sum, to_float(f.b) * x.sum);
  }
  __device__ __forceinline__ static float finish(const Row &, float sum) {
    return sum;
  }
};

} // namespace routed_gemv
