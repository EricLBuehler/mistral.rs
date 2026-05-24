#include "utils.metal"
#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void rotary(const device T *src [[buffer(0)]],
                       const device T *cos [[buffer(1)]],
                       const device T *sin [[buffer(2)]],
                       device T *output [[buffer(3)]],
                       constant uint &batch [[buffer(4)]],
                       constant uint &heads [[buffer(5)]],
                       constant uint &seq_len [[buffer(6)]],
                       constant uint &head_dim [[buffer(7)]],
                       constant uint &rot_dim [[buffer(8)]],
                       constant uint &cache_rows [[buffer(9)]],
                       constant bool &is_neox [[buffer(10)]],
                       uint tid [[thread_position_in_grid]]) {
  const uint total = batch * heads * seq_len * head_dim;
  if (tid >= total) {
    return;
  }

  const uint col = tid % head_dim;
  const uint row = tid / head_dim;
  const uint batch_idx = row / (heads * seq_len);
  const uint seq_idx = row % seq_len;
  const uint cache_row =
      cache_rows == batch * seq_len ? batch_idx * seq_len + seq_idx : seq_idx;

  if (col >= rot_dim * 2) {
    output[tid] = src[tid];
    return;
  }

  uint pair_idx;
  uint x_col;
  uint y_col;
  if (is_neox) {
    pair_idx = col < rot_dim ? col : col - rot_dim;
    x_col = pair_idx;
    y_col = pair_idx + rot_dim;
  } else {
    pair_idx = col / 2;
    x_col = pair_idx * 2;
    y_col = x_col + 1;
  }

  const uint base = row * head_dim;
  const float x = float(src[base + x_col]);
  const float y = float(src[base + y_col]);
  const uint cache_idx = cache_row * rot_dim + pair_idx;
  const float c = float(cos[cache_idx]);
  const float s = float(sin[cache_idx]);
  const float out = col == x_col ? x * c - y * s : y * c + x * s;
  output[tid] = T(out);
}

#define instantiate_rotary(type)                                               \
  template [[host_name("rotary_" #type)]] [[kernel]] void rotary<type>(        \
      const device type *src [[buffer(0)]],                                    \
      const device type *cos [[buffer(1)]],                                    \
      const device type *sin [[buffer(2)]], device type *output [[buffer(3)]], \
      constant uint &batch [[buffer(4)]],                                      \
      constant uint &heads [[buffer(5)]],                                      \
      constant uint &seq_len [[buffer(6)]],                                    \
      constant uint &head_dim [[buffer(7)]],                                   \
      constant uint &rot_dim [[buffer(8)]],                                    \
      constant uint &cache_rows [[buffer(9)]],                                 \
      constant bool &is_neox [[buffer(10)]],                                   \
      uint tid [[thread_position_in_grid]]);

instantiate_rotary(float);
instantiate_rotary(half);
#if __METAL_VERSION__ >= 310
instantiate_rotary(bfloat);
#endif
