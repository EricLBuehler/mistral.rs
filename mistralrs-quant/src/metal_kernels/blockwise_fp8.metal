#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

/* ----------------------------- parameters ----------------------------- */
extern "C" struct DequantParams {
  uint weight_height;
  uint weight_width;
  uint weight_row_stride; // leading dimension of the weight matrix
  uint scale_stride;      // leading dimension of the scale matrix
  uint block_size_y;      // tile height   ( == weight_block_size_y )
  uint block_size_x;      // tile width    ( == weight_block_size_x )
};

/* ------------------ FP8-E4M3 → float32 helper ------------------ */
inline float fp8_e4m3_to_float(uchar v) {
  const uint sign = v >> 7;
  const uint exponent = (v >> 3) & 0xF;
  const uint mantissa = v & 0x7;

  /* special encodings ------------------------------------------------ */
  if (exponent == 0) { // sub-normal / zero
    if (mantissa == 0)
      return 0.0f * (1.0f - 2.0f * sign); // signed zero
    float m = float(mantissa) / 8.0f;     // 2⁻³ scale for 3-bit frac
    float val = ldexp(m, -6);             // 2^(1-bias-fracbits) : bias=7
    return sign ? -val : val;
  }
  if (exponent == 0xF) { // Inf / NaN
    return sign ? -INFINITY : INFINITY;
  }

  /* normal numbers --------------------------------------------------- */
  float m = 1.0f + float(mantissa) / 8.0f; // implicit leading 1
  int exp = int(exponent) - 7;             // remove bias (bias = 7)
  float val = ldexp(m, exp);
  return sign ? -val : val;
}

/* -------------------------- kernel bodies ---------------------------- */
template <typename OutT>
kernel void
dequant_fp8_blockwise(device const uchar *weight [[buffer(0)]],
                      device const float *scale [[buffer(1)]],
                      device OutT *output [[buffer(2)]],
                      constant DequantParams &p [[buffer(3)]],
                      ushort3 thread_pos_tg [[thread_position_in_threadgroup]],
                      ushort3 tg_id [[threadgroup_position_in_grid]],
                      ushort3 tg_size [[threads_per_threadgroup]]) {
  /* one scale per tile ------------------------------------------------ */
  threadgroup float block_scale;
  if (thread_pos_tg.x == 0 && thread_pos_tg.y == 0) {
    block_scale = scale[tg_id.y * p.scale_stride + tg_id.x];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  /* tile origin ------------------------------------------------------- */
  const uint start_y = tg_id.y * p.block_size_y;
  const uint start_x = tg_id.x * p.block_size_x;

  /* walk over the tile ------------------------------------------------ */
  for (uint local_y = thread_pos_tg.y; local_y < p.block_size_y;
       local_y += tg_size.y) {
    uint weight_y = start_y + local_y;
    if (weight_y >= p.weight_height)
      continue;

    for (uint local_x = thread_pos_tg.x; local_x < p.block_size_x;
         local_x += tg_size.x) {
      uint weight_x = start_x + local_x;
      if (weight_x >= p.weight_width)
        continue;

      uint pos = weight_y * p.weight_row_stride + weight_x;
      float w_val = fp8_e4m3_to_float(weight[pos]);
      float scaled = w_val * block_scale;

      /* write-out with type cast */
      output[pos] = OutT(scaled);
    }
  }
}

/* -------------------------- entry points ----------------------------- */

#define instantiate_dequant_blockwise_fp8(type)                                \
  template [[host_name("dequant_fp8_blockwise_" #type)]] [[kernel]] void       \
  dequant_fp8_blockwise<type>(device const uchar *w [[buffer(0)]],             \
                              device const float *s [[buffer(1)]],             \
                              device type *o [[buffer(2)]],                    \
                              constant DequantParams &p [[buffer(3)]],         \
                              ushort3 tid [[thread_position_in_threadgroup]],  \
                              ushort3 gid [[threadgroup_position_in_grid]],    \
                              ushort3 tgs [[threads_per_threadgroup]]);

instantiate_dequant_blockwise_fp8(float);
instantiate_dequant_blockwise_fp8(half);
instantiate_dequant_blockwise_fp8(bfloat16_t);