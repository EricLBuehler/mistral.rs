#include "utils.metal"
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline void apply_rotary_row(const device T *src, device T *output,
                             const device T *cos, const device T *sin,
                             uint row, uint col, uint batch_idx, uint seq_idx,
                             uint heads, uint seq_len, uint head_dim,
                             uint rot_dim, uint cache_row, bool is_neox) {
  const uint src_idx = row * head_dim + col;
  if (col >= rot_dim * 2) {
    output[src_idx] = src[src_idx];
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
  const uint cache_idx = cache_row * rot_dim + pair_idx;
  const float x = float(src[base + x_col]);
  const float y = float(src[base + y_col]);
  const float c = float(cos[cache_idx]);
  const float s = float(sin[cache_idx]);
  const float out = col == x_col ? x * c - y * s : y * c + x * s;
  output[src_idx] = T(out);
}

template <typename T>
inline void apply_rotary_selected(const device T *src, device T *output,
                                  const device T *cos, const device T *sin,
                                  uint tid, uint batch, uint heads,
                                  uint seq_len, uint head_dim, uint rot_dim,
                                  uint cache_rows, bool is_neox) {
  const uint col = tid % head_dim;
  const uint row = tid / head_dim;
  const uint batch_idx = row / (heads * seq_len);
  const uint seq_idx = row % seq_len;
  const uint cache_row =
      cache_rows == batch * seq_len ? batch_idx * seq_len + seq_idx : seq_idx;
  apply_rotary_row(src, output, cos, sin, row, col, batch_idx, seq_idx, heads,
                   seq_len, head_dim, rot_dim, cache_row, is_neox);
}

template <typename T>
inline void apply_rotary_positioned(const device T *src, device T *output,
                                    const device T *cos, const device T *sin,
                                    const device uint *positions, uint tid,
                                    uint heads, uint seq_len, uint head_dim,
                                    uint rot_dim, bool is_neox) {
  const uint col = tid % head_dim;
  const uint row = tid / head_dim;
  const uint batch_idx = row / (heads * seq_len);
  const uint seq_idx = row % seq_len;
  const uint cache_row = positions[batch_idx] + seq_idx;
  apply_rotary_row(src, output, cos, sin, row, col, batch_idx, seq_idx, heads,
                   seq_len, head_dim, rot_dim, cache_row, is_neox);
}

template <typename T>
[[kernel]] void rotary_q(const device T *q [[buffer(0)]],
                         const device T *cos [[buffer(1)]],
                         const device T *sin [[buffer(2)]],
                         device T *q_out [[buffer(3)]],
                         constant uint &batch [[buffer(4)]],
                         constant uint &q_heads [[buffer(5)]],
                         constant uint &seq_len [[buffer(6)]],
                         constant uint &head_dim [[buffer(7)]],
                         constant uint &rot_dim [[buffer(8)]],
                         constant uint &cache_rows [[buffer(9)]],
                         constant bool &is_neox [[buffer(10)]],
                         uint tid [[thread_position_in_grid]]) {
  if (tid >= batch * q_heads * seq_len * head_dim) {
    return;
  }
  apply_rotary_selected(q, q_out, cos, sin, tid, batch, q_heads, seq_len,
                        head_dim, rot_dim, cache_rows, is_neox);
}

template <typename T>
[[kernel]] void rotary_qk(const device T *q [[buffer(0)]],
                          const device T *k [[buffer(1)]],
                          const device T *cos [[buffer(2)]],
                          const device T *sin [[buffer(3)]],
                          device T *q_out [[buffer(4)]],
                          device T *k_out [[buffer(5)]],
                          constant uint &batch [[buffer(6)]],
                          constant uint &q_heads [[buffer(7)]],
                          constant uint &k_heads [[buffer(8)]],
                          constant uint &seq_len [[buffer(9)]],
                          constant uint &head_dim [[buffer(10)]],
                          constant uint &rot_dim [[buffer(11)]],
                          constant uint &cache_rows [[buffer(12)]],
                          constant bool &is_neox [[buffer(13)]],
                          uint tid [[thread_position_in_grid]]) {
  const uint q_total = batch * q_heads * seq_len * head_dim;
  const uint k_total = batch * k_heads * seq_len * head_dim;
  if (tid < q_total) {
    apply_rotary_selected(q, q_out, cos, sin, tid, batch, q_heads, seq_len,
                          head_dim, rot_dim, cache_rows, is_neox);
  } else if (tid < q_total + k_total) {
    apply_rotary_selected(k, k_out, cos, sin, tid - q_total, batch, k_heads,
                          seq_len, head_dim, rot_dim, cache_rows, is_neox);
  }
}

template <typename T>
[[kernel]] void rotary_q_positions(const device T *q [[buffer(0)]],
                                   const device T *cos [[buffer(1)]],
                                   const device T *sin [[buffer(2)]],
                                   const device uint *positions [[buffer(3)]],
                                   device T *q_out [[buffer(4)]],
                                   constant uint &batch [[buffer(5)]],
                                   constant uint &q_heads [[buffer(6)]],
                                   constant uint &seq_len [[buffer(7)]],
                                   constant uint &head_dim [[buffer(8)]],
                                   constant uint &rot_dim [[buffer(9)]],
                                   constant bool &is_neox [[buffer(10)]],
                                   uint tid [[thread_position_in_grid]]) {
  if (tid >= batch * q_heads * seq_len * head_dim) {
    return;
  }
  apply_rotary_positioned(q, q_out, cos, sin, positions, tid, q_heads, seq_len,
                          head_dim, rot_dim, is_neox);
}

template <typename T>
[[kernel]] void rotary_qk_positions(
    const device T *q [[buffer(0)]], const device T *k [[buffer(1)]],
    const device T *cos [[buffer(2)]], const device T *sin [[buffer(3)]],
    const device uint *positions [[buffer(4)]], device T *q_out [[buffer(5)]],
    device T *k_out [[buffer(6)]], constant uint &batch [[buffer(7)]],
    constant uint &q_heads [[buffer(8)]], constant uint &k_heads [[buffer(9)]],
    constant uint &seq_len [[buffer(10)]],
    constant uint &head_dim [[buffer(11)]],
    constant uint &rot_dim [[buffer(12)]],
    constant bool &is_neox [[buffer(13)]],
    uint tid [[thread_position_in_grid]]) {
  const uint q_total = batch * q_heads * seq_len * head_dim;
  const uint k_total = batch * k_heads * seq_len * head_dim;
  if (tid < q_total) {
    apply_rotary_positioned(q, q_out, cos, sin, positions, tid, q_heads,
                            seq_len, head_dim, rot_dim, is_neox);
  } else if (tid < q_total + k_total) {
    apply_rotary_positioned(k, k_out, cos, sin, positions, tid - q_total,
                            k_heads, seq_len, head_dim, rot_dim, is_neox);
  }
}

#define instantiate_rotary(type)                                               \
  template [[host_name("rotary_q_" #type)]] [[kernel]] void rotary_q<type>(    \
      const device type *q [[buffer(0)]],                                      \
      const device type *cos [[buffer(1)]],                                    \
      const device type *sin [[buffer(2)]], device type *q_out [[buffer(3)]],  \
      constant uint &batch [[buffer(4)]],                                      \
      constant uint &q_heads [[buffer(5)]],                                    \
      constant uint &seq_len [[buffer(6)]],                                    \
      constant uint &head_dim [[buffer(7)]],                                   \
      constant uint &rot_dim [[buffer(8)]],                                    \
      constant uint &cache_rows [[buffer(9)]],                                 \
      constant bool &is_neox [[buffer(10)]],                                   \
      uint tid [[thread_position_in_grid]]);                                   \
  template [[host_name("rotary_qk_" #type)]] [[kernel]] void                   \
  rotary_qk<type>(                                                             \
      const device type *q [[buffer(0)]],                                      \
      const device type *k [[buffer(1)]],                                      \
      const device type *cos [[buffer(2)]],                                    \
      const device type *sin [[buffer(3)]],                                    \
      device type *q_out [[buffer(4)]],                                        \
      device type *k_out [[buffer(5)]],                                        \
      constant uint &batch [[buffer(6)]],                                      \
      constant uint &q_heads [[buffer(7)]],                                    \
      constant uint &k_heads [[buffer(8)]],                                    \
      constant uint &seq_len [[buffer(9)]],                                    \
      constant uint &head_dim [[buffer(10)]],                                  \
      constant uint &rot_dim [[buffer(11)]],                                   \
      constant uint &cache_rows [[buffer(12)]],                                \
      constant bool &is_neox [[buffer(13)]],                                   \
      uint tid [[thread_position_in_grid]]);                                   \
  template [[host_name("rotary_q_positions_" #type)]] [[kernel]] void          \
  rotary_q_positions<type>(                                                    \
      const device type *q [[buffer(0)]],                                      \
      const device type *cos [[buffer(1)]],                                    \
      const device type *sin [[buffer(2)]],                                    \
      const device uint *positions [[buffer(3)]],                              \
      device type *q_out [[buffer(4)]],                                        \
      constant uint &batch [[buffer(5)]],                                      \
      constant uint &q_heads [[buffer(6)]],                                    \
      constant uint &seq_len [[buffer(7)]],                                    \
      constant uint &head_dim [[buffer(8)]],                                   \
      constant uint &rot_dim [[buffer(9)]],                                    \
      constant bool &is_neox [[buffer(10)]],                                   \
      uint tid [[thread_position_in_grid]]);                                   \
  template [[host_name("rotary_qk_positions_" #type)]] [[kernel]] void         \
  rotary_qk_positions<type>(                                                   \
      const device type *q [[buffer(0)]],                                      \
      const device type *k [[buffer(1)]],                                      \
      const device type *cos [[buffer(2)]],                                    \
      const device type *sin [[buffer(3)]],                                    \
      const device uint *positions [[buffer(4)]],                              \
      device type *q_out [[buffer(5)]],                                        \
      device type *k_out [[buffer(6)]],                                        \
      constant uint &batch [[buffer(7)]],                                      \
      constant uint &q_heads [[buffer(8)]],                                    \
      constant uint &k_heads [[buffer(9)]],                                    \
      constant uint &seq_len [[buffer(10)]],                                   \
      constant uint &head_dim [[buffer(11)]],                                  \
      constant uint &rot_dim [[buffer(12)]],                                   \
      constant bool &is_neox [[buffer(13)]],                                   \
      uint tid [[thread_position_in_grid]]);

instantiate_rotary(float);
instantiate_rotary(half);
#if __METAL_VERSION__ >= 310
instantiate_rotary(bfloat);
#endif
