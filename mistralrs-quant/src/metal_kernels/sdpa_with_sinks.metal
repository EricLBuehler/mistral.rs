// sdpa_with_sinks.metal
//
// Fused scaled dot-product attention with per-head attention sinks for Metal.
// Two kernel families:
//   1. sdpa_vector_with_sinks          — decode path (q_len == 1)
//      sdpa_vector_with_sinks_2pass_1/2 — decode with long context (k_len >=
//      1024)
//   2. flash_attn_sinks                — prefill path (q_len > 1), tiled
//
// Sinks: per-head values that contribute to the softmax denominator but
// NOT to the output (virtual probability-mass-absorbing tokens).
// After processing all K/V positions, the sink is integrated into the
// online softmax as: new_max = max(m, sink_h), rescale accumulators,
// l += exp(sink_h - new_max).

#include "bf16.metal"

// ============================================================================
// Decode: sdpa_vector_with_sinks
// Based on candle's sdpa_vector with sink integration after the K/V loop.
// ============================================================================

template <typename T, int D>
[[kernel]] void sdpa_vector_with_sinks(
    const device T *queries [[buffer(0)]], const device T *keys [[buffer(1)]],
    const device T *values [[buffer(2)]],
    const device float *sinks [[buffer(3)]], // [num_heads]
    device T *out [[buffer(4)]], const constant int &gqa_factor,
    const constant int &N, // k_len
    const constant size_t &k_stride, const constant size_t &v_stride,
    const constant float &scale, uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int stride = BN * D;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * elem_per_thread;
  keys += kv_head_idx * k_stride + simd_gid * D + simd_lid * elem_per_thread;
  values += kv_head_idx * v_stride + simd_gid * D + simd_lid * elem_per_thread;
  out += head_idx * D + simd_gid * elem_per_thread;

  // Read the query and scale; zero the output accumulator
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key position
  for (int i = simd_gid; i < N; i += BN) {
    // Read the key
    for (int j = 0; j < elem_per_thread; j++) {
      k[j] = keys[j];
    }

    // Compute the i-th score
    U score = 0;
    for (int j = 0; j < elem_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);

    // Update the accumulators (online softmax)
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Update the output accumulator
    for (int j = 0; j < elem_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * values[j];
    }

    // Move to the next kv
    keys += stride;
    values += stride;
  }

  // Integrate sink into the online softmax for this simdgroup.
  // The sink contributes to the denominator but NOT to the output.
  {
    U sink_val = sinks[head_idx];
    U new_max = max(max_score, sink_val);
    U factor = fast::exp(max_score - new_max);
    U exp_sink = fast::exp(sink_val - new_max);

    for (int j = 0; j < elem_per_thread; j++) {
      o[j] = o[j] * factor;
    }
    // Each simdgroup adds its share: since there are BN simdgroups and the
    // sink should only be counted once globally, we let simdgroup 0 add it
    // in the cross-simdgroup reduction below. Here we just rescale.
    // We add the full exp_sink here; the cross-simdgroup reduction will
    // handle de-duplication by only counting the sink once.
    // Actually, each simdgroup independently tracks its own max/sum.
    // The cross-simdgroup reduction already handles combining them correctly.
    // The sink should appear once in the global softmax. Since only one
    // simdgroup needs to contribute it, we add it to simdgroup 0 only.
    if (simd_gid == 0) {
      sum_exp_score = sum_exp_score * factor + exp_sink;
    } else {
      sum_exp_score = sum_exp_score * factor;
    }
    max_score = new_max;
  }

  // Each simdgroup has a partial output; combine via shared memory.
  // Communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Aggregate all the outputs
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ============================================================================
// Decode 2-pass: sdpa_vector_with_sinks_2pass_1 and _2pass_2
// For long key sequences (k_len >= 1024), split work across 32 blocks.
// Sinks are integrated in pass 2 (reduction).
// ============================================================================

template <typename T, int D>
[[kernel]] void sdpa_vector_with_sinks_2pass_1(
    const device T *queries [[buffer(0)]], const device T *keys [[buffer(1)]],
    const device T *values [[buffer(2)]], device float *out [[buffer(3)]],
    device float *sums [[buffer(4)]], device float *maxs [[buffer(5)]],
    const constant int &gqa_factor, const constant int &N,
    const constant size_t &k_stride, const constant size_t &v_stride,
    const constant float &scale, uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int BN = 8;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int stride = BN * D;
  constexpr int blocks = 32;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores_tg[BN];
  threadgroup U sum_exp_scores_tg[BN];

  // Adjust positions
  const int block_idx = tid.z;
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * elem_per_thread;
  keys += kv_head_idx * k_stride + (block_idx * BN + simd_gid) * D +
          simd_lid * elem_per_thread;
  values += kv_head_idx * v_stride + (block_idx * BN + simd_gid) * D +
            simd_lid * elem_per_thread;
  out += head_idx * blocks * D + block_idx * D + simd_lid * elem_per_thread;
  sums += head_idx * blocks + block_idx;
  maxs += head_idx * blocks + block_idx;

  // Read the query and scale; zero the output accumulator
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -1e9;
  U sum_exp_score = 0;

  // For each key position assigned to this block
  for (int i = block_idx * BN + simd_gid; i < N; i += blocks * BN) {
    // Read the key
    for (int j = 0; j < elem_per_thread; j++) {
      k[j] = keys[j];
    }

    // Compute the score
    U score = 0;
    for (int j = 0; j < elem_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);

    // Update the accumulators (online softmax)
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Update the output accumulator
    for (int j = 0; j < elem_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * values[j];
    }

    // Move to the next kv
    keys += blocks * stride;
    values += blocks * stride;
  }

  // Combine within this block's simdgroups
  if (simd_lid == 0) {
    max_scores_tg[simd_gid] = max_score;
    sum_exp_scores_tg[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = (simd_lid < BN) ? max_scores_tg[simd_lid] : -1e9;
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = (simd_lid < BN) ? sum_exp_scores_tg[simd_lid] : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);

  // Write the sum and new max for this block
  if (simd_gid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Aggregate outputs for this block
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BN + simd_gid] =
        o[i] * fast::exp(max_scores_tg[simd_gid] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0) {
      U output = outputs[simd_lid * BN];
      for (int j = 1; j < BN; j++) {
        output += outputs[simd_lid * BN + j];
      }
      out[i] = static_cast<T>(output);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_with_sinks_2pass_2(
    const device float *partials [[buffer(0)]],
    const device float *sums [[buffer(1)]],
    const device float *maxs [[buffer(2)]],
    const device float *sinks [[buffer(3)]], // [num_heads]
    device T *out [[buffer(4)]], uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int blocks = 32;

  typedef float U;

  thread U o[elem_per_thread];
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.y;
  partials += head_idx * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += head_idx * blocks;
  maxs += head_idx * blocks;
  out += head_idx * D + simd_gid * elem_per_thread;

  // Read the max and sum_exp from each block
  U max_score = maxs[simd_lid];
  U new_max = simd_max(max_score);

  // Integrate sink: treat it as an additional "block" with max=sink,
  // sum=exp(sink-sink)=1
  U sink_val = sinks[head_idx];
  new_max = max(new_max, sink_val);

  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sums[simd_lid] * factor);
  // Add sink contribution to denominator (once)
  sum_exp_score += fast::exp(sink_val - new_max);

  // Read the block into registers and then use shared memory to transpose
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = partials[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ============================================================================
// Prefill: flash_attn_sinks_kernel
// Port of CUDA flash_attn_sinks.cu to Metal.
// Tiled flash attention with online softmax and per-head sinks.
// ============================================================================

// Warp-level sum reduction via simd_sum (Metal equivalent of __shfl_xor_sync)
inline float warp_reduce_sum(float val) { return simd_sum(val); }

template <typename T, int HEAD_DIM, int BR, int BC>
[[kernel]] void flash_attn_sinks_kernel(
    const device T *Q [[buffer(0)]],         // [B, num_heads, S, D]
    const device T *K [[buffer(1)]],         // [B, num_kv_heads, S, D]
    const device T *V [[buffer(2)]],         // [B, num_kv_heads, S, D]
    const device float *sinks [[buffer(3)]], // [num_heads]
    device T *O [[buffer(4)]],               // [B, num_heads, S, D]
    const constant float &scale,
    const constant int &q_len, // number of Q positions
    const constant int &k_len, // number of K/V positions
    const constant int &num_heads, const constant int &num_kv_heads,
    const constant int &window_size, // 0 = no window (full causal)
    threadgroup float *shared_mem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg3 [[thread_position_in_threadgroup]]) {

  const uint tpitg = tpitg3.x;

  // Padded head dim for clean warp division (round up to multiple of 32)
  constexpr int SIMD_SIZE = 32;
  constexpr int D_PAD = ((HEAD_DIM + SIMD_SIZE - 1) / SIMD_SIZE) * SIMD_SIZE;
  constexpr int EPT = D_PAD / SIMD_SIZE; // elements per thread
  constexpr int BLOCK_SIZE = BR * SIMD_SIZE;

  // Shared memory for K and V tiles (float32 for precision)
  threadgroup float *k_smem = shared_mem;              // [BC * D_PAD]
  threadgroup float *v_smem = shared_mem + BC * D_PAD; // [BC * D_PAD]

  const int simd_gid = tpitg / SIMD_SIZE; // which simdgroup (0..BR-1)
  const int simd_lid = tpitg % SIMD_SIZE; // lane within simdgroup

  const int head_idx = tgpig.x;
  const int batch_idx = tgpig.y;
  const int q_tile_idx = tgpig.z;

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;

  // Which query row this simdgroup handles
  const int q_row = q_tile_idx * BR + simd_gid;

  // Offset between Q local row index and absolute KV position
  const int kv_offset = k_len - q_len;

  // Offsets into contiguous [B, H, S, D] layout
  const int q_offset =
      ((batch_idx * num_heads + head_idx) * q_len + q_row) * HEAD_DIM;
  const int kv_base =
      (batch_idx * num_kv_heads + kv_head_idx) * k_len * HEAD_DIM;

  // Load Q row into registers (pre-scaled)
  float q_reg[EPT];
  for (int i = 0; i < EPT; i++) {
    const int d = i * SIMD_SIZE + simd_lid;
    q_reg[i] =
        (q_row < q_len && d < HEAD_DIM) ? float(Q[q_offset + d]) * scale : 0.0f;
  }

  // Output accumulator in registers
  float o_acc[EPT];
  for (int i = 0; i < EPT; i++)
    o_acc[i] = 0.0f;

  float m_i = -INFINITY; // running max
  float l_i = 0.0f;      // running sum of exp

  // Block-level KV bounds (union of all simdgroups' causal windows)
  // Q rows are local (0..q_len), absolute KV position = q_row + kv_offset
  const int block_q_start = q_tile_idx * BR;
  const int block_q_end = min(block_q_start + BR, q_len);
  const int block_kv_start =
      (window_size > 0) ? max(0, block_q_start + kv_offset - window_size + 1)
                        : 0;
  const int block_kv_end = min(k_len, block_q_end + kv_offset);

  // Per-simdgroup causal bounds (using absolute KV position)
  const int my_kv_start = (window_size > 0 && q_row < q_len)
                              ? max(0, q_row + kv_offset - window_size + 1)
                              : 0;
  const int my_kv_end = (q_row < q_len) ? (q_row + kv_offset + 1) : 0;

  // Score buffer for current tile (in registers)
  float scores[BC];

  // Tile loop over K/V
  for (int tile_start = block_kv_start; tile_start < block_kv_end;
       tile_start += BC) {
    const int tile_end = min(tile_start + BC, block_kv_end);
    const int tile_len = tile_end - tile_start;

    // --- Cooperatively load K tile into shared memory ---
    for (int idx = int(tpitg); idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int kj = idx / D_PAD;
      const int kd = idx % D_PAD;
      k_smem[idx] = (kj < tile_len && kd < HEAD_DIM)
                        ? float(K[kv_base + (tile_start + kj) * HEAD_DIM + kd])
                        : 0.0f;
    }

    // --- Cooperatively load V tile into shared memory ---
    for (int idx = int(tpitg); idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int vj = idx / D_PAD;
      const int vd = idx % D_PAD;
      v_smem[idx] = (vj < tile_len && vd < HEAD_DIM)
                        ? float(V[kv_base + (tile_start + vj) * HEAD_DIM + vd])
                        : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (q_row < q_len) {
      // --- Pass 1: Compute scores for this tile ---
      float tile_max = -INFINITY;

      for (int j = 0; j < tile_len; j++) {
        const int kv_pos = tile_start + j;

        // Causal: no valid positions beyond q_row
        if (kv_pos >= my_kv_end) {
          for (int jj = j; jj < tile_len; jj++)
            scores[jj] = -INFINITY;
          break;
        }

        // Sliding window check
        if (kv_pos < my_kv_start) {
          scores[j] = -INFINITY;
          continue;
        }

        // Dot product: q_reg . K[j] from shared memory
        float dot = 0.0f;
        for (int i = 0; i < EPT; i++) {
          dot += q_reg[i] * k_smem[j * D_PAD + i * SIMD_SIZE + simd_lid];
        }
        dot = warp_reduce_sum(dot);

        scores[j] = dot;
        tile_max = max(tile_max, dot);
      }

      // --- Pass 2: Online softmax update + V accumulation ---
      if (tile_max > -INFINITY) {
        const float m_new = max(m_i, tile_max);
        const float rescale = fast::exp(m_i - m_new);

        // Rescale old accumulators (ONCE per tile, not per position)
        for (int i = 0; i < EPT; i++)
          o_acc[i] *= rescale;
        l_i *= rescale;
        m_i = m_new;

        // Accumulate V contributions
        for (int j = 0; j < tile_len; j++) {
          if (scores[j] <= -INFINITY)
            continue; // masked position

          const float p = fast::exp(scores[j] - m_i);
          l_i += p;

          for (int i = 0; i < EPT; i++) {
            o_acc[i] += p * v_smem[j * D_PAD + i * SIMD_SIZE + simd_lid];
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (q_row >= q_len)
    return;

  // Integrate per-head sinks into the softmax denominator.
  // The sink adds exp(sink_val) to the denominator but does NOT
  // contribute any value to the output (virtual token).
  {
    const float sink_val = sinks[head_idx];
    const float m_new = max(m_i, sink_val);
    const float rescale = fast::exp(m_i - m_new);
    for (int i = 0; i < EPT; i++)
      o_acc[i] *= rescale;
    l_i = l_i * rescale + fast::exp(sink_val - m_new);
    m_i = m_new;
  }

  // Normalize and write output
  const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  for (int i = 0; i < EPT; i++) {
    const int d = i * SIMD_SIZE + simd_lid;
    if (d < HEAD_DIM) {
      O[q_offset + d] = static_cast<T>(o_acc[i] * inv_l);
    }
  }
}

// ============================================================================
// Prefill varlen: flash_attn_sinks_varlen_kernel
// Q padded [B, num_heads, max_q_len, D], K/V packed [total_kv, num_kv_heads, D]
// cu_seqlens_q[B+1], cu_seqlens_k[B+1]
// ============================================================================

template <typename T, int HEAD_DIM, int BR, int BC>
[[kernel]] void flash_attn_sinks_varlen_kernel(
    const device T *Q [[buffer(0)]],         // [B, num_heads, max_q_len, D]
    const device T *K [[buffer(1)]],         // [total_kv, num_kv_heads, D]
    const device T *V [[buffer(2)]],         // [total_kv, num_kv_heads, D]
    const device float *sinks [[buffer(3)]], // [num_heads]
    device T *O [[buffer(4)]],               // [B, num_heads, max_q_len, D]
    const device uint *cu_seqlens_q [[buffer(5)]], // [B+1]
    const device uint *cu_seqlens_k [[buffer(6)]], // [B+1]
    const constant float &scale, const constant int &max_q_len,
    const constant int &num_heads, const constant int &num_kv_heads,
    const constant int &window_size,
    threadgroup float *shared_mem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg3 [[thread_position_in_threadgroup]]) {

  const uint tpitg = tpitg3.x;

  constexpr int SIMD_SIZE = 32;
  constexpr int D_PAD = ((HEAD_DIM + SIMD_SIZE - 1) / SIMD_SIZE) * SIMD_SIZE;
  constexpr int EPT = D_PAD / SIMD_SIZE;
  constexpr int BLOCK_SIZE = BR * SIMD_SIZE;

  threadgroup float *k_smem = shared_mem;
  threadgroup float *v_smem = shared_mem + BC * D_PAD;

  const int simd_gid = tpitg / SIMD_SIZE;
  const int simd_lid = tpitg % SIMD_SIZE;

  const int head_idx = tgpig.x;
  const int batch_idx = tgpig.y;
  const int q_tile_idx = tgpig.z;

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;

  // Per-batch-item lengths (derived from cumulative arrays)
  const int my_q_len =
      (int)(cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]);
  const int kv_start = (int)cu_seqlens_k[batch_idx];
  const int my_kv_len =
      (int)(cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx]);

  const int q_row = q_tile_idx * BR + simd_gid;

  const bool q_row_valid = (q_row < my_q_len);

  const int kv_offset = my_kv_len - my_q_len;

  // Q offset: padded [B, H, max_q_len, D]
  const int q_offset =
      ((batch_idx * num_heads + head_idx) * max_q_len + q_row) * HEAD_DIM;

  // Load Q row into registers (pre-scaled)
  float q_reg[EPT];
  for (int i = 0; i < EPT; i++) {
    const int d = i * SIMD_SIZE + simd_lid;
    q_reg[i] =
        (q_row_valid && d < HEAD_DIM) ? float(Q[q_offset + d]) * scale : 0.0f;
  }

  float o_acc[EPT];
  for (int i = 0; i < EPT; i++)
    o_acc[i] = 0.0f;

  float m_i = -INFINITY;
  float l_i = 0.0f;

  // Block-level KV bounds
  const int block_q_start = q_tile_idx * BR;
  const int block_q_end = min(block_q_start + BR, my_q_len);
  const int block_kv_start =
      (window_size > 0) ? max(0, block_q_start + kv_offset - window_size + 1)
                        : 0;
  const int block_kv_end = min(my_kv_len, block_q_end + kv_offset);

  const int my_kv_start_w =
      (window_size > 0) ? max(0, q_row + kv_offset - window_size + 1) : 0;
  const int my_kv_end = q_row + kv_offset + 1;

  float scores[BC];

  for (int tile_start = block_kv_start; tile_start < block_kv_end;
       tile_start += BC) {
    const int tile_end = min(tile_start + BC, block_kv_end);
    const int tile_len = tile_end - tile_start;

    // --- Cooperatively load K tile from packed [total_kv, num_kv_heads, D] ---
    for (int idx = int(tpitg); idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int kj = idx / D_PAD;
      const int kd = idx % D_PAD;
      k_smem[idx] = (kj < tile_len && kd < HEAD_DIM)
                        ? float(K[((kv_start + tile_start + kj) * num_kv_heads +
                                   kv_head_idx) *
                                      HEAD_DIM +
                                  kd])
                        : 0.0f;
    }

    // --- Cooperatively load V tile from packed layout ---
    for (int idx = int(tpitg); idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int vj = idx / D_PAD;
      const int vd = idx % D_PAD;
      v_smem[idx] = (vj < tile_len && vd < HEAD_DIM)
                        ? float(V[((kv_start + tile_start + vj) * num_kv_heads +
                                   kv_head_idx) *
                                      HEAD_DIM +
                                  vd])
                        : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (q_row_valid) {
      // --- Pass 1: Compute scores ---
      float tile_max = -INFINITY;

      for (int j = 0; j < tile_len; j++) {
        const int kv_pos = tile_start + j;

        if (kv_pos >= my_kv_end) {
          for (int jj = j; jj < tile_len; jj++)
            scores[jj] = -INFINITY;
          break;
        }

        if (kv_pos < my_kv_start_w) {
          scores[j] = -INFINITY;
          continue;
        }

        float dot = 0.0f;
        for (int i = 0; i < EPT; i++) {
          dot += q_reg[i] * k_smem[j * D_PAD + i * SIMD_SIZE + simd_lid];
        }
        dot = warp_reduce_sum(dot);

        scores[j] = dot;
        tile_max = max(tile_max, dot);
      }

      // --- Pass 2: Online softmax update + V accumulation ---
      if (tile_max > -INFINITY) {
        const float m_new = max(m_i, tile_max);
        const float rescale = fast::exp(m_i - m_new);

        for (int i = 0; i < EPT; i++)
          o_acc[i] *= rescale;
        l_i *= rescale;
        m_i = m_new;

        for (int j = 0; j < tile_len; j++) {
          if (scores[j] <= -INFINITY)
            continue;

          const float p = fast::exp(scores[j] - m_i);
          l_i += p;

          for (int i = 0; i < EPT; i++) {
            o_acc[i] += p * v_smem[j * D_PAD + i * SIMD_SIZE + simd_lid];
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (!q_row_valid)
    return;

  // Integrate sinks
  {
    const float sink_val = sinks[head_idx];
    const float m_new = max(m_i, sink_val);
    const float rescale = fast::exp(m_i - m_new);
    for (int i = 0; i < EPT; i++)
      o_acc[i] *= rescale;
    l_i = l_i * rescale + fast::exp(sink_val - m_new);
    m_i = m_new;
  }

  // Normalize and write output
  const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  for (int i = 0; i < EPT; i++) {
    const int d = i * SIMD_SIZE + simd_lid;
    if (d < HEAD_DIM) {
      O[q_offset + d] = static_cast<T>(o_acc[i] * inv_l);
    }
  }
}

// ============================================================================
// Instantiation macros
// ============================================================================

// --- Decode: sdpa_vector_with_sinks ---
#define instantiate_sdpa_vector_with_sinks(type, head_dim)                     \
  template [[host_name("sdpa_vector_with_sinks_" #type                         \
                       "_" #head_dim)]] [[kernel]] void                        \
  sdpa_vector_with_sinks<type, head_dim>(                                      \
      const device type *queries [[buffer(0)]],                                \
      const device type *keys [[buffer(1)]],                                   \
      const device type *values [[buffer(2)]],                                 \
      const device float *sinks [[buffer(3)]], device type *out [[buffer(4)]], \
      const constant int &gqa_factor, const constant int &N,                   \
      const constant size_t &k_stride, const constant size_t &v_stride,        \
      const constant float &scale, uint3 tid [[threadgroup_position_in_grid]], \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);                            \
  template [[host_name("sdpa_vector_with_sinks_2pass_1_" #type                 \
                       "_" #head_dim)]] [[kernel]] void                        \
  sdpa_vector_with_sinks_2pass_1<type, head_dim>(                              \
      const device type *queries [[buffer(0)]],                                \
      const device type *keys [[buffer(1)]],                                   \
      const device type *values [[buffer(2)]],                                 \
      device float *out [[buffer(3)]], device float *sums [[buffer(4)]],       \
      device float *maxs [[buffer(5)]], const constant int &gqa_factor,        \
      const constant int &N, const constant size_t &k_stride,                  \
      const constant size_t &v_stride, const constant float &scale,            \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);                            \
  template [[host_name("sdpa_vector_with_sinks_2pass_2_" #type                 \
                       "_" #head_dim)]] [[kernel]] void                        \
  sdpa_vector_with_sinks_2pass_2<type, head_dim>(                              \
      const device float *partials [[buffer(0)]],                              \
      const device float *sums [[buffer(1)]],                                  \
      const device float *maxs [[buffer(2)]],                                  \
      const device float *sinks [[buffer(3)]], device type *out [[buffer(4)]], \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_sdpa_vector_with_sinks_heads(type)                         \
  instantiate_sdpa_vector_with_sinks(type, 64)                                 \
      instantiate_sdpa_vector_with_sinks(type, 80)                             \
          instantiate_sdpa_vector_with_sinks(type, 96)                         \
              instantiate_sdpa_vector_with_sinks(type, 128)                    \
                  instantiate_sdpa_vector_with_sinks(type, 256)

instantiate_sdpa_vector_with_sinks_heads(float)
    instantiate_sdpa_vector_with_sinks_heads(half)
#if __METAL_VERSION__ >= 310
        instantiate_sdpa_vector_with_sinks_heads(bfloat16_t)
#endif

// --- Prefill: flash_attn_sinks_kernel ---
#define instantiate_flash_attn_sinks(type, hd, br, bc)                         \
  template [[host_name("flash_attn_sinks_" #type "_hd" #hd "_br" #br           \
                       "_bc" #bc)]] [[kernel]] void                            \
  flash_attn_sinks_kernel<type, hd, br, bc>(                                   \
      const device type *Q [[buffer(0)]], const device type *K [[buffer(1)]],  \
      const device type *V [[buffer(2)]],                                      \
      const device float *sinks [[buffer(3)]], device type *O [[buffer(4)]],   \
      const constant float &scale, const constant int &q_len,                  \
      const constant int &k_len, const constant int &num_heads,                \
      const constant int &num_kv_heads, const constant int &window_size,       \
      threadgroup float *shared_mem [[threadgroup(0)]],                        \
      uint3 tgpig [[threadgroup_position_in_grid]],                            \
      uint3 tpitg3 [[thread_position_in_threadgroup]]);

// BR=8 (simdgroups per threadgroup), BC chosen by head dim
#define instantiate_flash_attn_sinks_heads(type)                               \
  instantiate_flash_attn_sinks(type, 64, 8, 64)                                \
      instantiate_flash_attn_sinks(type, 80, 8, 32)                            \
          instantiate_flash_attn_sinks(type, 96, 8, 32)                        \
              instantiate_flash_attn_sinks(type, 112, 8, 32)                   \
                  instantiate_flash_attn_sinks(type, 128, 8, 32)               \
                      instantiate_flash_attn_sinks(type, 192, 8, 16)           \
                          instantiate_flash_attn_sinks(type, 256, 8, 16)

            instantiate_flash_attn_sinks_heads(float)
                instantiate_flash_attn_sinks_heads(half)
#if __METAL_VERSION__ >= 310
                    instantiate_flash_attn_sinks_heads(bfloat16_t)
#endif

// --- Prefill varlen: flash_attn_sinks_varlen_kernel ---
#define instantiate_flash_attn_sinks_varlen(type, hd, br, bc)                  \
  template [[host_name("flash_attn_sinks_varlen_" #type "_hd" #hd "_br" #br    \
                       "_bc" #bc)]] [[kernel]] void                            \
  flash_attn_sinks_varlen_kernel<type, hd, br, bc>(                            \
      const device type *Q [[buffer(0)]], const device type *K [[buffer(1)]],  \
      const device type *V [[buffer(2)]],                                      \
      const device float *sinks [[buffer(3)]], device type *O [[buffer(4)]],   \
      const device uint *cu_seqlens_q [[buffer(5)]],                           \
      const device uint *cu_seqlens_k [[buffer(6)]],                           \
      const constant float &scale, const constant int &max_q_len,              \
      const constant int &num_heads, const constant int &num_kv_heads,         \
      const constant int &window_size,                                         \
      threadgroup float *shared_mem [[threadgroup(0)]],                        \
      uint3 tgpig [[threadgroup_position_in_grid]],                            \
      uint3 tpitg3 [[thread_position_in_threadgroup]]);

#define instantiate_flash_attn_sinks_varlen_heads(type)                        \
  instantiate_flash_attn_sinks_varlen(                                         \
      type, 64, 8, 64) instantiate_flash_attn_sinks_varlen(type, 80, 8, 32)    \
      instantiate_flash_attn_sinks_varlen(type, 96, 8, 32)                     \
          instantiate_flash_attn_sinks_varlen(type, 112, 8, 32)                \
              instantiate_flash_attn_sinks_varlen(type, 128, 8, 32)            \
                  instantiate_flash_attn_sinks_varlen(type, 192, 8, 16)        \
                      instantiate_flash_attn_sinks_varlen(type, 256, 8, 16)

                        instantiate_flash_attn_sinks_varlen_heads(float)
                            instantiate_flash_attn_sinks_varlen_heads(half)
#if __METAL_VERSION__ >= 310
                                instantiate_flash_attn_sinks_varlen_heads(
                                    bfloat16_t)
#endif
