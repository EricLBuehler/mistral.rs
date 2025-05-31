#include <metal_stdlib>

using namespace metal;

// ————————————————————————————————————————————————————————————————
// F8E4M3 (Sign=1, Exponent=4, Mantissa=3; bias=2^(4−1)−1 = 7)
// ————————————————————————————————————————————————————————————————

inline float fp8_e4m3_to_float(uchar v) {
  const uint sign = (v >> 7) & 0x1;
  const uint exp_bits = (v >> 3) & 0xF;
  const uint man_bits = v & 0x7;

  // handle zero / subnormals
  if (exp_bits == 0) {
    if (man_bits == 0) {
      return sign ? -0.0f : 0.0f;
    }
    // subnormal: mantissa / 2^(bias + mantissa_bits)
    float m = float(man_bits) / float(1 << 3);
    float val = ldexp(m, 1 - 7 - 3);
    return sign ? -val : val;
  }
  // handle Inf/NaN
  if (exp_bits == 0xF) {
    return sign ? -INFINITY : INFINITY;
  }
  // normalised
  float mant = 1.0f + float(man_bits) / float(1 << 3);
  int expn = int(exp_bits) - 7;
  float val = ldexp(mant, expn);
  return sign ? -val : val;
}

inline uchar float_to_fp8_e4m3(float f) {
  uint bits = as_type<uint>(f);
  uint sign = bits >> 31;
  int exp = int((bits >> 23) & 0xFF) - 127 + 7; // adjust bias
  uint man = bits & 0x7FFFFF;

  // handle NaN/Inf
  if (exp > 0xE) {
    // map all overflow to Inf
    return uchar((sign << 7) | (0xF << 3));
  }
  // handle zero and subnormals
  if (exp <= 0) {
    // subnormal or underflow → zero
    return uchar(sign << 7);
  }
  // round-to-nearest-even: add half-ULP
  uint mant_rounded = (man + (1 << (23 - 3 - 1))) >> (23 - 3);
  if (mant_rounded == (1 << 3)) {
    // overflow in mantissa → bump exponent
    mant_rounded = 0;
    exp += 1;
    if (exp >= 0xF) {
      // now overflow → Inf
      return uchar((sign << 7) | (0xF << 3));
    }
  }
  return uchar((sign << 7) | (uint(exp) << 3) | (mant_rounded & 0x7));
}

// ————————————————————————————————————————————————————————————————
// F8E5M2 (Sign=1, Exponent=5, Mantissa=2; bias=2^(5−1)−1 = 15)
// ————————————————————————————————————————————————————————————————

inline float fp8_e5m2_to_float(uchar v) {
  const uint sign = (v >> 7) & 0x1;
  const uint exp_bits = (v >> 2) & 0x1F;
  const uint man_bits = v & 0x3;

  if (exp_bits == 0) {
    if (man_bits == 0) {
      return sign ? -0.0f : 0.0f;
    }
    float m = float(man_bits) / float(1 << 2);
    float val = ldexp(m, 1 - 15 - 2);
    return sign ? -val : val;
  }
  if (exp_bits == 0x1F) {
    return sign ? -INFINITY : INFINITY;
  }
  float mant = 1.0f + float(man_bits) / float(1 << 2);
  int expn = int(exp_bits) - 15;
  float val = ldexp(mant, expn);
  return sign ? -val : val;
}

inline uchar float_to_fp8_e5m2(float f) {
  uint bits = as_type<uint>(f);
  uint sign = bits >> 31;
  int exp = int((bits >> 23) & 0xFF) - 127 + 15;
  uint man = bits & 0x7FFFFF;

  if (exp > 0x1D) {
    return uchar((sign << 7) | (0x1F << 2));
  }
  if (exp <= 0) {
    return uchar(sign << 7);
  }
  uint mant_rounded = (man + (1 << (23 - 2 - 1))) >> (23 - 2);
  if (mant_rounded == (1 << 2)) {
    mant_rounded = 0;
    exp += 1;
    if (exp >= 0x1F) {
      return uchar((sign << 7) | (0x1F << 2));
    }
  }
  return uchar((sign << 7) | (uint(exp) << 2) | (mant_rounded & 0x3));
}