#include <metal_stdlib>
using namespace metal;

// F8E4M3 to float conversion
// Format: 1 sign bit, 4 exponent bits, 3 mantissa bits
// Bias = 7, special values: NaN mapped to 0
inline float f8e4m3_to_float(uint8_t bits) {
  uint sign = (bits >> 7) & 1;
  uint exp = (bits >> 3) & 0xF;
  uint mant = bits & 0x7;

  float val;
  if (exp == 0) {
    // Subnormal: (-1)^sign * 2^(1-bias) * (mant/8) = 2^(-6) * mant/8
    val = ldexp(float(mant) / 8.0f, -6);
  } else if (exp == 0xF && mant == 0x7) {
    // NaN -> 0
    return 0.0f;
  } else {
    // Normal: (-1)^sign * 2^(exp-bias) * (1 + mant/8)
    val = ldexp(1.0f + float(mant) / 8.0f, int(exp) - 7);
  }
  return sign ? -val : val;
}

// F8E4M3::MAX = 448.0
constant float F8E4M3_MAX = 448.0f;

// Dequantize F8Q8 blocks to float32
// Each block: 1 byte F8E4M3 scale + 32 bytes i8 quantized values = 33 bytes
// Dequantized value: qs[i] * (d.to_f32() / F8E4M3_MAX)
kernel void f8q8_dequantize_float(const device uint8_t *w [[buffer(0)]],
                                  device float *out [[buffer(1)]],
                                  constant uint &num_blocks [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]]) {
  if (tid >= num_blocks)
    return;

  const uint block_offset = tid * 33;
  const uint out_offset = tid * 32;

  uint8_t d_byte = w[block_offset];
  float scale = f8e4m3_to_float(d_byte) / F8E4M3_MAX;

  for (int i = 0; i < 32; i++) {
    int8_t q = as_type<int8_t>(w[block_offset + 1 + i]);
    out[out_offset + i] = float(q) * scale;
  }
}

// Dequantize F8Q8 blocks to half precision
kernel void f8q8_dequantize_half(const device uint8_t *w [[buffer(0)]],
                                 device half *out [[buffer(1)]],
                                 constant uint &num_blocks [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]]) {
  if (tid >= num_blocks)
    return;

  const uint block_offset = tid * 33;
  const uint out_offset = tid * 32;

  uint8_t d_byte = w[block_offset];
  float scale = f8e4m3_to_float(d_byte) / F8E4M3_MAX;

  for (int i = 0; i < 32; i++) {
    int8_t q = as_type<int8_t>(w[block_offset + 1 + i]);
    out[out_offset + i] = half(float(q) * scale);
  }
}
