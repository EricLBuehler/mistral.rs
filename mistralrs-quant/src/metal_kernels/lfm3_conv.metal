#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

template <typename T>
inline T lfm3_gated_input(const device T *x, constant ulong *p, ulong batch,
                          ulong token, ulong channel) {
  const ulong offset = batch * p[4] + token * p[5] + channel * p[6];
  return T(float(x[offset]) * float(x[offset + 2 * p[2] * p[6]]));
}

template <typename T, ushort WIDTH>
[[kernel]] void
lfm3_conv(const device T *x [[buffer(0)]], const device T *weight [[buffer(1)]],
          const device T *bias [[buffer(2)]],
          const device T *state [[buffer(3)]], device T *output [[buffer(4)]],
          device T *next_state [[buffer(5)]], constant ulong *p [[buffer(6)]],
          uint index [[thread_position_in_grid]]) {
  const ulong seq = p[1], hidden = p[2], width = WIDTH ? WIDTH : p[3];
  if (index >= p[0] * seq * hidden) {
    return;
  }
  const ulong channel = index % hidden;
  const ulong token = index / hidden % seq;
  const ulong batch = index / (hidden * seq);
  const ulong state_base = batch * p[7] + channel * p[8];
  float sum = 0.0f;
  for (ushort tap = 0; tap < width; tap++) {
    T value = T(0.0f);
    if (token + tap + 1 >= width) {
      value = lfm3_gated_input(x, p, batch, token + tap + 1 - width, channel);
    } else if (p[13]) {
      value = state[state_base + (token + tap + 1) * p[9]];
    }
    sum += float(value) * float(weight[channel * p[10] + tap * p[11]]);
  }
  if (p[14]) {
    sum += float(bias[channel * p[12]]);
  }
  // Metal fast-math can remove the dtype rounding before the output gate.
  volatile T convolved = T(sum);
  const ulong gate = batch * p[4] + token * p[5] + (hidden + channel) * p[6];
  output[index] = T(float(convolved) * float(x[gate]));
  if (token + 1 == seq) {
    for (ushort tap = 0; tap < width; tap++) {
      T value = T(0.0f);
      if (seq + tap >= width) {
        value = lfm3_gated_input(x, p, batch, seq + tap - width, channel);
      } else if (p[13]) {
        value = state[state_base + (seq + tap) * p[9]];
      }
      next_state[(batch * hidden + channel) * width + tap] = value;
    }
  }
}

#define instantiate_lfm3_conv(type, width)                                     \
  template [[host_name("lfm3_conv_" #type "_" #width)]] [[kernel]] void        \
  lfm3_conv<type, width>(const device type *x [[buffer(0)]],                   \
                         const device type *weight [[buffer(1)]],              \
                         const device type *bias [[buffer(2)]],                \
                         const device type *state [[buffer(3)]],               \
                         device type *output [[buffer(4)]],                    \
                         device type *next_state [[buffer(5)]],                \
                         constant ulong *p [[buffer(6)]],                      \
                         uint index [[thread_position_in_grid]]);

instantiate_lfm3_conv(float, 0);
instantiate_lfm3_conv(float, 7);
instantiate_lfm3_conv(half, 0);
instantiate_lfm3_conv(half, 7);
instantiate_lfm3_conv(bfloat, 0);
instantiate_lfm3_conv(bfloat, 7);
