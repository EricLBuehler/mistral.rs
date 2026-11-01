#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Doubled E2M1 magnitudes 0, 1, 2, 3, 4, 6, 8, 12 and their negations; callers halve the block scale.
constexpr uint32_t kE2M1x2Pos0 = 0x03020100u;
constexpr uint32_t kE2M1x2Pos1 = 0x0C080604u;
constexpr uint32_t kE2M1x2Neg0 = 0xFDFEFF00u;
constexpr uint32_t kE2M1x2Neg1 = 0xF4F8FAFCu;
constexpr float kE2M1x2Half = 0.5f;

constexpr uint32_t kF32Nan = 0x7fc00000u;
constexpr uint32_t kF32Exp2Neg127 = 0x00400000u;
constexpr int kF32MantissaBits = 23;

__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {
  return __half2float(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3));
}

// Raw exponent shift: 0 decodes to 0 and 255 to inf.
__device__ __forceinline__ float e8m0_to_float(uint8_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << kF32MantissaBits);
}

// 0 is 2^-127, the one E8M0 exponent that is a float subnormal; 255 stays inf.
__device__ __forceinline__ float e8m0_to_float_subnormal(uint8_t bits) {
  return bits == 0 ? __uint_as_float(kF32Exp2Neg127) : e8m0_to_float(bits);
}

// OCP MX semantics: 0 is 2^-127 and 255 is NaN.
__device__ __forceinline__ float e8m0_to_float_ocp(uint8_t bits) {
  if (bits == 255)
    return __uint_as_float(kF32Nan);
  return e8m0_to_float_subnormal(bits);
}

// E2M1 built from bits: a local lookup table spills to local memory under dynamic indexing.
__device__ __forceinline__ float e2m1_to_float(uint8_t code) {
  const uint32_t e = (code >> 1) & 3u;
  const uint32_t m = code & 1u;
  const uint32_t magnitude =
      e ? ((e + 126u) << 23) | (m << 22) : m * 0x3f000000u;
  return __uint_as_float(magnitude | (static_cast<uint32_t>(code & 8u) << 28));
}

// Eight low-nibble-first E2M1 codes to doubled values (see kE2M1x2Half).
__device__ __forceinline__ void e2m1x8_to_float_doubled(uint32_t q,
                                                        float *out) {
  uint32_t bytes[2];
  const uint32_t sign = 0x32103210u | ((q & 0x88888888u) >> 1);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t nibbles = q >> (16 * i);
    const uint32_t pos = __byte_perm(kE2M1x2Pos0, kE2M1x2Pos1, nibbles);
    const uint32_t neg = __byte_perm(kE2M1x2Neg0, kE2M1x2Neg1, nibbles);
    bytes[i] = __byte_perm(pos, neg, sign >> (16 * i));
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = static_cast<float>(static_cast<int8_t>(bytes[0] >> (8 * i)));
    out[4 + i] = static_cast<float>(static_cast<int8_t>(bytes[1] >> (8 * i)));
  }
}
