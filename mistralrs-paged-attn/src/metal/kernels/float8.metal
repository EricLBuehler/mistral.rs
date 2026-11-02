#include <metal_stdlib>
using namespace metal;

// Helpers ------------------------------------------------------------
static inline uint as_bits(float x) { return as_type<uint>(x); }
static inline float from_bits(uint b) { return as_type<float>(b); }

// -------------------------------------------------------------------
// FP8 E4M3 (bias = 7)
// -------------------------------------------------------------------
inline float fp8_e4m3_to_float(uchar v) {
  const uint s = v >> 7;
  const uint exp = (v >> 3) & 0xF;
  const uint man = v & 0x7;

  if (exp == 0) { // zero / sub-normal
    if (man == 0)
      return s ? -0.f : 0.f;
    const float m = float(man) / 8.f; // already scaled by 2^-3
    float val = ldexp(m, 1 - 7);      // 2^(1-bias) = 2^-6
    return s ? -val : val;
  }

  // E4M3 has NO infinity - only NaN when exp=15 and mantissa=7
  if (exp == 0xF && man == 0x7) {
    return NAN;
  }

  // Normalized (including exp=0xF with mantissa 0-6, which are valid numbers)
  const float m = 1.f + float(man) / 8.f;
  float val = ldexp(m, int(exp) - 7);
  return s ? -val : val;
}

// -------------------------------------------------------------------
// FP8 E5M2 (bias = 15)
// -------------------------------------------------------------------
inline float fp8_e5m2_to_float(uchar v) {
  const uint s = v >> 7;
  const uint exp = (v >> 2) & 0x1F;
  const uint man = v & 0x3;

  if (exp == 0) {
    if (man == 0)
      return s ? -0.f : 0.f;
    const float m = float(man) / 4.f;
    float val = ldexp(m, 1 - 15); // 2^(1-bias) = 2^-14
    return s ? -val : val;
  }

  if (exp == 0x1F) {
    if (man != 0)
      return NAN;
    return s ? -INFINITY : INFINITY;
  }

  const float m = 1.f + float(man) / 4.f;
  float val = ldexp(m, int(exp) - 15);
  return s ? -val : val;
}

// -------------------------------------------------------------------
// Encoding helpers (round-to-nearest-even, gradual under-flow, sat-to-∞)
// -------------------------------------------------------------------
namespace detail {
template <int EXP_BITS, int MAN_BITS, int BIAS>
inline uchar fp32_to_fp8(float f) {
  const uint bits = as_bits(f);
  const uint s = bits >> 31;
  const uint abs = bits & 0x7FFFFFFF;

  // NaN propagates, Inf saturates
  if (abs >= 0x7F800000u) {
    return uchar((s << 7) | (((1u << EXP_BITS) - 1u) << MAN_BITS) |
                 (abs != 0x7F800000u));
  }

  int e = int((abs >> 23) & 0xFF) - 127;   // unbiased exponent
  uint m = abs & 0x7FFFFFu;                // 23-bit mantissa
  const int EXP_MAX = (1 << EXP_BITS) - 2; // last finite exponent

  // ---------- Normal path -------------------------------------------------
  int e_fp8 = e + BIAS;
  if (e_fp8 >= 1 && e_fp8 <= EXP_MAX) {
    // round-to-nearest-even
    const int shift = 23 - MAN_BITS;
    uint mant = m >> shift;
    const uint lsb = mant & 1u;
    const uint round = (m >> (shift - 1)) & 1u;
    const uint sticky = (m & ((1u << (shift - 1)) - 1u)) != 0u;
    mant += (round & (sticky | lsb));
    if (mant >> MAN_BITS) { // mantissa overflow
      mant = 0;
      ++e_fp8;
      if (e_fp8 > EXP_MAX)
        return uchar((s << 7) | (((1u << EXP_BITS) - 1u) << MAN_BITS)); // ∞
    }
    return uchar((s << 7) | (uint(e_fp8) << MAN_BITS) |
                 (mant & ((1u << MAN_BITS) - 1u)));
  }

  // ---------- Sub-normal / under-flow ------------------------------------
  if (e_fp8 < 1 - MAN_BITS) // too small -> ±0
    return uchar(s << 7);

  // shift so that exponent becomes 1
  int rshift = (1 - e_fp8) + (23 - MAN_BITS);
  uint mant = (0x800000u | m); // implicit 1
  uint rounded = (mant + (1u << (rshift - 1))) >> rshift;
  if (rounded == 0)
    return uchar(s << 7); // rounds to zero

  return uchar((s << 7) | (rounded & ((1u << MAN_BITS) - 1u)));
}
} // namespace detail

constant constexpr uint E4M3_MAX_FINITE = 0x7e;
constant constexpr uint E4M3_NAN = 0x7f;
constant constexpr uint E4M3_MAX_FLOAT_BITS = 0x43e00000;
constant constexpr uint E4M3_MIN_NORMAL_FLOAT_BITS = 0x3c800000;
constant constexpr uint FP32_INFINITY_BITS = 0x7f800000;
constant constexpr uint FP32_MANTISSA_MASK = 0x7fffff;
constant constexpr int FP32_MANTISSA_BITS = 23;
constant constexpr int FP32_EXPONENT_BIAS = 127;
constant constexpr int E4M3_MANTISSA_BITS = 3;
constant constexpr int E4M3_EXPONENT_BIAS = 7;

inline uchar float_to_fp8_e4m3(float f) {
  const uint bits = as_bits(f);
  const uint sign = (bits >> 24) & 0x80;
  const uint magnitude = bits & 0x7fffffff;
  if (magnitude > FP32_INFINITY_BITS) {
    return uchar(E4M3_NAN);
  }
  if (magnitude >= E4M3_MAX_FLOAT_BITS) {
    return uchar(sign | E4M3_MAX_FINITE);
  }

  const int exponent = int(magnitude >> FP32_MANTISSA_BITS) -
                       FP32_EXPONENT_BIAS + E4M3_EXPONENT_BIAS;
  const uint mantissa = magnitude & FP32_MANTISSA_MASK;
  const int shift = FP32_MANTISSA_BITS - E4M3_MANTISSA_BITS;
  uint encoded;
  uint remainder;
  uint halfway;
  if (magnitude < E4M3_MIN_NORMAL_FLOAT_BITS) {
    if (exponent < -E4M3_MANTISSA_BITS) {
      return uchar(sign);
    }
    const int subnormal_shift = shift + 1 - exponent;
    const uint significand = (1u << FP32_MANTISSA_BITS) | mantissa;
    encoded = significand >> subnormal_shift;
    remainder = significand & ((1u << subnormal_shift) - 1u);
    halfway = 1u << (subnormal_shift - 1);
  } else {
    encoded = (uint(exponent) << E4M3_MANTISSA_BITS) | (mantissa >> shift);
    remainder = mantissa & ((1u << shift) - 1u);
    halfway = 1u << (shift - 1);
  }
  // Keep the carry into the normal range and use the encoded LSB for ties.
  encoded += remainder > halfway || (remainder == halfway && (encoded & 1u));
  return uchar(sign | encoded);
}
inline uchar float_to_fp8_e5m2(float f) {
  return detail::fp32_to_fp8<5, 2, 15>(f);
}
