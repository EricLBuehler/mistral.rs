// Qwen4Exp-specific Metal kernels (Completion Phase 7).
//
// These complement the shared GDN kernels in gdn.metal with fused paths for the
// architecture-owned hyper-connection and PLE operations.

#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_BFLOAT__)
typedef bfloat bfloat16_t;
#endif

// ============================================================================
// Kernel 1: hc_rmsnorm_flatten
//
// Fuses the Qwen4Exp hyper-connection mixer prefix: per-stream RMS
// normalization over the grouped residual [rows, streams, hidden] plus the
// flattened gamma multiply, emitting the projection input in the flattened
// [rows, streams * hidden] layout. The grouped and flattened layouts are
// row-major identical, so input and output share one flat offset scheme and
// the gamma indexes the same absolute element.
//
// One threadgroup per (row, stream) pair; RMS statistics in F32, matching the
// composed Candle path exactly. The flattened gamma is indexed by
// (stream * hidden + i), so the kernel needs the stream count to map the
// threadgroup's (row, stream) pair onto the gamma element.
// ============================================================================

template <typename T>
[[kernel]] void hc_rmsnorm_flatten_kernel(
    const device T *x [[buffer(0)]], const device T *weight [[buffer(1)]],
    device T *output [[buffer(2)]], constant uint &hidden [[buffer(3)]],
    constant float &eps [[buffer(4)]], constant uint &streams [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]]) {
  const size_t base = (size_t)tgpig * hidden;
  const uint stream = tgpig % streams;
  const size_t weight_base = (size_t)stream * hidden;
  const device T *x_row = x + base;
  device T *o_row = output + base;

  float partial = 0.0f;
  for (uint i = gid; i < hidden; i += tg_size) {
    const float v = (float)x_row[i];
    partial = fma(v, v, partial);
  }
  const float warp_total = simd_sum(partial);

  threadgroup float sums[32];
  if (simd_lane == 0) {
    sums[simd_group] = warp_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float sum = 0.0f;
  for (uint s = 0; s < simd_groups; s++) {
    sum += sums[s];
  }

  const float inv_rms = rsqrt(sum / (float)hidden + eps);

  for (uint i = gid; i < hidden; i += tg_size) {
    o_row[i] = (T)((float)x_row[i] * inv_rms * (float)weight[weight_base + i]);
  }
}

#define instantiate_hc_rmsnorm_flatten(type, name)                             \
  template [[host_name("hc_rmsnorm_flatten_" #name)]] [[kernel]]               \
  void hc_rmsnorm_flatten_kernel<type>(                                        \
      const device type *, const device type *, device type *,                 \
      constant uint &, constant float &, constant uint &, uint, uint, uint,    \
      uint, uint, uint);

instantiate_hc_rmsnorm_flatten(float, float);
instantiate_hc_rmsnorm_flatten(half, half);
instantiate_hc_rmsnorm_flatten(bfloat16_t, bfloat16_t);

// ============================================================================
// Kernel 2: ple_conv1d
//
// Fuses the Qwen4Exp PLE causal dilated depthwise convolution over the
// concatenated per-sequence state [history + tokens, channels]: for each output
// token row the kernel reads the configured taps, multiplies with the
// per-channel [kernel_size, channels] tap kernel, accumulates in F32, and
// writes the SiLU-activated result in the activation dtype. Channels never mix
// and the tap order matches the composed Candle path exactly: tap 0 reads the
// oldest visible row, lookback = (kernel_size - 1 - tap) * dilation rows behind
// the current token.
//
// One threadgroup (256 threads) per output token row with a strided loop over
// channels. The caller guarantees (kernel_size - 1) * dilation <= history so
// every tap read stays inside the concatenated state; accumulation in F32 is
// strictly more accurate than the composed path's activation-dtype tap sums.
// ============================================================================

template <typename T>
[[kernel]] void ple_conv1d_kernel(
    const device T *input [[buffer(0)]], const device T *taps [[buffer(1)]],
    device T *output [[buffer(2)]], constant uint &channels [[buffer(3)]],
    constant uint &history [[buffer(4)]],
    constant uint &kernel_size [[buffer(5)]],
    constant uint &dilation [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[thread_position_in_threadgroup]]) {
  const size_t in_base = (size_t)(history + tgpig) * channels;
  device T *o_row = output + (size_t)tgpig * channels;
  for (uint c = gid; c < channels; c += tg_size) {
    float acc = 0.0f;
    for (uint tap = 0; tap < kernel_size; ++tap) {
      const size_t lookback = (size_t)(kernel_size - 1u - tap) * dilation;
      const float v = (float)input[in_base - lookback * channels + c];
      acc = fma((float)taps[(size_t)tap * channels + c], v, acc);
    }
    o_row[c] = (T)(acc / (1.0f + exp(-acc)));
  }
}

#define instantiate_ple_conv1d(type, name)                                     \
  template [[host_name("ple_conv1d_" #name)]] [[kernel]] void                  \
  ple_conv1d_kernel<type>(const device type *, const device type *,            \
                          device type *, constant uint &, constant uint &,     \
                          constant uint &, constant uint &, uint, uint, uint);

instantiate_ple_conv1d(float, float);
instantiate_ple_conv1d(half, half);
instantiate_ple_conv1d(bfloat16_t, bfloat16_t);

// ============================================================================
// Kernel 3: qsa_topk_indices
//
// Deterministic QSA indexer block top-k on Metal. Ranks complete-block scores
// with the exact host `QsaBlockSelector::select` total order: score descending
// under `f32::total_cmp` bit semantics, then the earlier block index on an
// exact tie. This is the gate required by Completion Phase 7 before selection
// may move off the host: candle's shared bitonic argsort has no index
// tie-break, so it cannot reproduce the host order and is not used here.
//
// Implementation: one threadgroup per score row running a bitonic argsort in
// threadgroup memory. Each column owns one index; the comparator is a strict
// total order (key desc, index asc), so the standard bitonic network sorts it
// exactly. Non-finite scores are ordered by the same `f32::total_cmp` bit
// transform instead of being rejected: the host path keeps strict NaN
// rejection, while this device path stays deterministic on already-degenerate
// scores (selection only chooses cache rows; SDPA recomputes attention from
// the gathered K/V, so finite inputs are unaffected).
//
// Columns beyond `ncols` are sentinel indices that sort after every real
// block, so the first `k` entries are the top-k real blocks. The 4096-entry
// threadgroup scratch is 16 KB, within the Apple GPU 32 KB threadgroup limit;
// callers must reject score rows wider than `QSA_TOPK_MAX_COLUMNS` and fall
// back to the host selector.
// ============================================================================

#define QSA_TOPK_SENTINEL 0xFFFFFFFFu
#define QSA_TOPK_MAX_PAD 4096u

inline int qsa_topk_total_key(device const float *x, uint i) {
  const int bits = as_type<int>(x[i]);
  // f32::total_cmp bit transform: arithmetic shift fills the sign mask, then
  // a logical shift halves it, so negatives flip while positives stay put and
  // the signed integer order matches the float total order.
  return bits ^ (int)(((uint)(bits >> 31)) >> 1);
}

inline int qsa_topk_key(device const float *x, uint idx) {
  return idx == QSA_TOPK_SENTINEL ? (-2147483647 - 1) : qsa_topk_total_key(x, idx);
}

// "a is ranked before b": total_cmp key descending, then the earlier index.
inline bool qsa_topk_before(device const float *x, uint a, uint b) {
  const int key_a = qsa_topk_key(x, a);
  const int key_b = qsa_topk_key(x, b);
  return key_a > key_b || (key_a == key_b && a < b);
}

kernel void qsa_topk_indices_kernel(
    const device float *scores [[buffer(0)]], device uint *out [[buffer(1)]],
    constant uint &ncols [[buffer(2)]], constant uint &k [[buffer(3)]],
    constant uint &ncols_pad [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[thread_position_in_threadgroup]]) {
  threadgroup uint shared[QSA_TOPK_MAX_PAD];
  device const float *row = scores + (size_t)tgpig * ncols;

  for (uint c = gid; c < ncols_pad; c += tg_size) {
    shared[c] = c < ncols ? c : QSA_TOPK_SENTINEL;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint kpow = 2u; kpow <= ncols_pad; kpow <<= 1u) {
    for (uint j = kpow >> 1u; j > 0u; j >>= 1u) {
      for (uint c = gid; c < ncols_pad; c += tg_size) {
        const uint ixj = c ^ j;
        if (ixj > c) {
          const bool ascending_half = (c & kpow) == 0u;
          const bool swap_needed =
              ascending_half ? qsa_topk_before(row, shared[ixj], shared[c])
                             : qsa_topk_before(row, shared[c], shared[ixj]);
          if (swap_needed) {
            const uint tmp = shared[c];
            shared[c] = shared[ixj];
            shared[ixj] = tmp;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  for (uint c = gid; c < k; c += tg_size) {
    out[(size_t)tgpig * k + c] = shared[c];
  }
}
