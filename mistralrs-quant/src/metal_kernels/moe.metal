#include "utils.metal"
#include <metal_stdlib>
using namespace metal;

// Fused topk-weighted reduce across MoE expert assignments.
// Input rows are in (token, slot)-major order: row `t*topk + s` is token `t`'s
// `s`-th expert contribution. Output row `t` is the topk-weighted sum across
// slots. Accumulation happens in fp32 regardless of T to preserve precision.
template <typename T>
[[kernel]] void moe_weighted_reduce_flat(
    const device T *inputs [[buffer(0)]],
    const device float *topk_weights [[buffer(1)]],
    device T *outputs [[buffer(2)]],
    const constant int &num_tokens [[buffer(3)]],
    const constant int &hidden [[buffer(4)]],
    const constant int &topk [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int h = int(gid.x);
  const int t = int(gid.y);
  if (h >= hidden || t >= num_tokens) {
    return;
  }
  const size_t assignment_base = size_t(t) * size_t(topk);
  float acc = 0.0f;
  for (int s = 0; s < topk; ++s) {
    const size_t assignment = assignment_base + size_t(s);
    acc += static_cast<float>(inputs[assignment * size_t(hidden) + size_t(h)]) *
           topk_weights[assignment];
  }
  outputs[size_t(t) * size_t(hidden) + size_t(h)] = static_cast<T>(acc);
}

#define instantiate_moe_weighted_reduce(type)                                  \
  template [[host_name("moe_weighted_reduce_flat_" #type)]] [[kernel]] void    \
  moe_weighted_reduce_flat<type>(                                              \
      const device type *inputs [[buffer(0)]],                                 \
      const device float *topk_weights [[buffer(1)]],                          \
      device type *outputs [[buffer(2)]],                                      \
      const constant int &num_tokens [[buffer(3)]],                            \
      const constant int &hidden [[buffer(4)]],                                \
      const constant int &topk [[buffer(5)]],                                  \
      uint2 gid [[thread_position_in_grid]]);

instantiate_moe_weighted_reduce(float)
instantiate_moe_weighted_reduce(bfloat)
instantiate_moe_weighted_reduce(half)
