#include <metal_stdlib>
using namespace metal;

// Fused softmax with sinks kernel for GPT-OSS attention.
//
// Each threadgroup processes one row (batch, head, query) of length k_len.
// The sink value for the head is included in the softmax denominator
// but NOT written to the output.
//
// Input:
//   logits: [batch * heads * q_len, k_len] (contiguous)
//   sinks:  [heads] - per-head sink values
// Output:
//   output: [batch * heads * q_len, k_len] - softmax probabilities (sink
//   dropped)

template <typename T>
[[kernel]] void softmax_with_sinks(
    const device T *logits [[buffer(0)]], const device T *sinks [[buffer(1)]],
    device T *output [[buffer(2)]], constant uint &num_heads [[buffer(3)]],
    constant uint &q_len [[buffer(4)]], constant uint &k_len [[buffer(5)]],
    threadgroup float *shared_mem [[threadgroup(0)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint ntg [[threads_per_threadgroup]]) {

  // Each threadgroup handles one row: (batch, head, query)
  const uint row_idx = tgpig;
  const uint h = (row_idx / q_len) % num_heads;

  // Pointers to this row
  const device T *row_logits = logits + row_idx * k_len;
  device T *row_output = output + row_idx * k_len;

  // Get sink value for this head
  const float sink_val = float(sinks[h]);

  // Shared memory layout: [0] = max, [1] = sum
  threadgroup float *s_max = shared_mem;
  threadgroup float *s_sum = shared_mem + 1;

  const uint tid = tpitg;
  constexpr uint SIMD_SIZE = 32;

  // Step 1: Find max (including sink)
  float local_max = -INFINITY;
  for (uint k = tid; k < k_len; k += ntg) {
    float val = float(row_logits[k]);
    local_max = max(local_max, val);
  }
  // Thread 0 includes sink in max
  if (tid == 0) {
    local_max = max(local_max, sink_val);
  }

  // Simdgroup reduction for max
  for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
    local_max = max(local_max, simd_shuffle_xor(local_max, offset));
  }

  // Cross-simdgroup reduction for max via shared memory
  const uint num_simdgroups = (ntg + SIMD_SIZE - 1) / SIMD_SIZE;
  threadgroup float *warp_scratch = shared_mem + 2; // after s_max, s_sum

  if (tiisg == 0) {
    warp_scratch[sgitg] = local_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < SIMD_SIZE) {
    local_max = (tid < num_simdgroups) ? warp_scratch[tid] : -INFINITY;
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
      local_max = max(local_max, simd_shuffle_xor(local_max, offset));
    }
    if (tid == 0) {
      *s_max = local_max;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float row_max = *s_max;

  // Step 2: Compute exp(x - max) and sum (including sink)
  float local_sum = 0.0f;
  for (uint k = tid; k < k_len; k += ntg) {
    float val = float(row_logits[k]);
    local_sum += exp(val - row_max);
  }
  // Thread 0 includes sink in sum
  if (tid == 0) {
    local_sum += exp(sink_val - row_max);
  }

  // Simdgroup reduction for sum
  for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
    local_sum += simd_shuffle_xor(local_sum, offset);
  }

  // Cross-simdgroup reduction for sum
  if (tiisg == 0) {
    warp_scratch[sgitg] = local_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < SIMD_SIZE) {
    local_sum = (tid < num_simdgroups) ? warp_scratch[tid] : 0.0f;
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
      local_sum += simd_shuffle_xor(local_sum, offset);
    }
    if (tid == 0) {
      *s_sum = local_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float row_sum = *s_sum;

  // Step 3: Write normalized outputs (sink is NOT written - it's dropped)
  const float inv_sum = 1.0f / row_sum;
  for (uint k = tid; k < k_len; k += ntg) {
    float val = float(row_logits[k]);
    row_output[k] = T(exp(val - row_max) * inv_sum);
  }
}

#define instantiate_softmax_with_sinks(type)                                   \
  template [[host_name("softmax_with_sinks_" #type)]] [[kernel]] void          \
  softmax_with_sinks<type>(const device type *logits [[buffer(0)]],            \
                           const device type *sinks [[buffer(1)]],             \
                           device type *output [[buffer(2)]],                  \
                           constant uint &num_heads [[buffer(3)]],             \
                           constant uint &q_len [[buffer(4)]],                 \
                           constant uint &k_len [[buffer(5)]],                 \
                           threadgroup float *shared_mem [[threadgroup(0)]],   \
                           uint tgpig [[threadgroup_position_in_grid]],        \
                           uint tpitg [[thread_position_in_threadgroup]],      \
                           uint sgitg [[simdgroup_index_in_threadgroup]],      \
                           uint tiisg [[thread_index_in_simdgroup]],           \
                           uint ntg [[threads_per_threadgroup]]);

instantiate_softmax_with_sinks(float);
instantiate_softmax_with_sinks(half);
#if __METAL_VERSION__ >= 310
instantiate_softmax_with_sinks(bfloat);
#endif
