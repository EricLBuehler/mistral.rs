#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

template <typename T>
[[kernel]] void reshape_and_cache(
    const device T *__restrict__ key
    [[buffer(0)]], // [num_tokens, num_heads, head_size]
    const device T *__restrict__ value
    [[buffer(1)]], // [num_tokens, num_heads, head_size]
    device T *__restrict__ key_cache
    [[buffer(2)]], // [num_blocks, num_heads, head_size/x, block_size, x]
    device T *__restrict__ value_cache
    [[buffer(3)]], // [num_blocks, num_heads, head_size, block_size]
    const device int64_t *__restrict__ slot_mapping
    [[buffer(4)]], // [num_tokens]
    device const int &key_stride, device const int &value_stride,
    device const int &num_heads, device const int &head_size,
    device const int &block_size, device const int &x,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  const int64_t token_idx = gid;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = tid; i < n; i += threads_per_threadgroup) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

#define instantiate_reshape_and_cache(type)                                    \
  template [[host_name("reshape_and_cache_" #type)]] [[kernel]] void           \
  reshape_and_cache<type>(                                                     \
      const device type *__restrict__ key [[buffer(0)]],                       \
      const device type *__restrict__ value [[buffer(1)]],                     \
      device type *__restrict__ key_cache [[buffer(2)]],                       \
      device type *__restrict__ value_cache [[buffer(3)]],                     \
      const device int64_t *__restrict__ slot_mapping [[buffer(4)]],           \
      device const int &key_stride, device const int &value_stride,            \
      device const int &num_heads, device const int &head_size,                \
      device const int &block_size, device const int &x,                       \
      uint gid [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint threads_per_threadgroup [[threads_per_threadgroup]]);

instantiate_reshape_and_cache(float);
instantiate_reshape_and_cache(bfloat16_t);
instantiate_reshape_and_cache(half);
