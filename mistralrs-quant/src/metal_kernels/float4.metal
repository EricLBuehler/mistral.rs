#include <metal_stdlib>

using namespace metal;

// Convert raw FP4 bits to float.
// Input is expected to be a 4-bit value in the low nibble of the byte.
float fp4_to_float(uchar fp4_bits) {
  // Extract sign bit (bit 3)
  uchar sign = (fp4_bits >> 3) & 1;

  // Extract exponent (bits 1-2)
  uchar exp_bits = (fp4_bits >> 1) & 0x3;

  // Extract mantissa (bit 0)
  uchar mant_bit = fp4_bits & 1;

  // E2M1 format with bias 1
  int fp4_exp_bias = 1;

  if (exp_bits == 0) {
    // Denormal: exponent is 0, so actual exponent is 1 - bias = 0
    // Value = (-1)^sign * 2^0 * (0.mantissa) = (-1)^sign * mantissa/2
    float value = float(mant_bit) * 0.5f;
    return sign != 0 ? -value : value;
  } else {
    // Normal: exponent is exp_bits - bias
    int actual_exp = int(exp_bits) - fp4_exp_bias;
    // Value = (-1)^sign * 2^actual_exp * (1.mantissa)
    float significand = 1.0f + (float(mant_bit) * 0.5f);
    float value = significand * pow(2.0f, float(actual_exp));
    return sign != 0 ? -value : value;
  }
}

float scale_to_float(uchar value) {
  if (value == 0xFF) {
    return NAN;
  } else {
    return pow(2.0f, float(int(value) - 127));
  }
}
