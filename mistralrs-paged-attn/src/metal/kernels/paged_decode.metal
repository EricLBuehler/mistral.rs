// Paged decode attention tuned for Apple GPUs.
//
// One threadgroup handles one (partition, kv head, sequence) and computes every query head that shares the kv
// head, so each KV block is read once per group instead of once per query head. Lanes map to the 32 tokens of a
// block for Q.K (coalesced reads of the [D/x, block, x] key layout) and to head-dim rows for P.V (each lane reads a
// row's 32 contiguous tokens). Partial results use the v2 layout so paged_attention_v2_reduce merges partitions;
// partition_size must match a PARTITION_SIZE the reduce kernel is instantiated with.

#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

constant constexpr int PD_BLOCK = 32;
constant constexpr int PD_SIMDS = 4;
constant constexpr int PD_THREADS = PD_SIMDS * 32;
constant constexpr int PD_SHORT_SIMDS = 32;
constant constexpr int PD_SIMD_WIDTH = 32;
constant constexpr int PD_KEY_PACK = 8;

template <typename T> inline void pd_load8(const device T *p, thread float *out);

template <> inline void pd_load8<half>(const device half *p, thread float *out) {
  const half4 a = *(const device half4 *)p;
  const half4 b = *(const device half4 *)(p + 4);
  out[0] = a.x;
  out[1] = a.y;
  out[2] = a.z;
  out[3] = a.w;
  out[4] = b.x;
  out[5] = b.y;
  out[6] = b.z;
  out[7] = b.w;
}

template <>
inline void pd_load8<bfloat16_t>(const device bfloat16_t *p, thread float *out) {
  const uint4 w = *(const device uint4 *)p;
  out[0] = as_type<float>(w.x << 16);
  out[1] = as_type<float>(w.x & 0xFFFF0000u);
  out[2] = as_type<float>(w.y << 16);
  out[3] = as_type<float>(w.y & 0xFFFF0000u);
  out[4] = as_type<float>(w.z << 16);
  out[5] = as_type<float>(w.z & 0xFFFF0000u);
  out[6] = as_type<float>(w.w << 16);
  out[7] = as_type<float>(w.w & 0xFFFF0000u);
}

template <typename T, int HEAD_DIM, int GQA>
[[kernel]] void paged_decode_gqa(
    device T *out [[buffer(0)]],               // [seqs, heads, D], single-partition launches
    device float *exp_sums [[buffer(1)]],      // [seqs, heads, max_parts]
    device float *max_logits [[buffer(2)]],    // [seqs, heads, max_parts]
    device T *tmp_out [[buffer(3)]],           // [seqs, heads, max_parts, D]
    const device T *q [[buffer(4)]],           // [seqs, heads, D] with q_stride
    const device T *k_cache [[buffer(5)]],     // [blocks, kv_heads, D/8, PD_BLOCK, 8]
    const device T *v_cache [[buffer(6)]],     // [blocks, kv_heads, D, PD_BLOCK]
    const device uint32_t *block_tables [[buffer(7)]],
    const device uint32_t *context_lens [[buffer(8)]],
    constant int &max_num_blocks_per_seq [[buffer(9)]],
    constant int &q_stride [[buffer(10)]],
    constant int &kv_block_stride [[buffer(11)]],
    constant int &kv_head_stride [[buffer(12)]],
    constant float &scale [[buffer(13)]],
    constant int &max_num_partitions [[buffer(14)]],
    constant int &partition_size [[buffer(15)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int CHUNKS = HEAD_DIM / 8;
  constexpr int ROWS = HEAD_DIM / 32;

  const int part = tg.x;
  const int kv_head = tg.y;
  const int seq = tg.z;
  const int num_heads = tgs.y * GQA;
  const int context_len = context_lens[seq];
  const int part_start = part * partition_size;
  if (part_start >= context_len) {
    return;
  }
  const int part_end = min(context_len, part_start + partition_size);
  const int start_block = part_start / PD_BLOCK;
  const int end_block = (part_end + PD_BLOCK - 1) / PD_BLOCK;

  threadgroup float q_s[GQA][HEAD_DIM];
  threadgroup float p_s[PD_SIMDS][GQA][PD_BLOCK];
  threadgroup float o_s[GQA][HEAD_DIM];
  threadgroup float red_m[PD_SIMDS][GQA];
  threadgroup float red_l[PD_SIMDS][GQA];

  for (int i = tid; i < GQA * HEAD_DIM; i += PD_THREADS) {
    const int g = i / HEAD_DIM;
    const int d = i % HEAD_DIM;
    q_s[g][d] = float(q[seq * q_stride + (kv_head * GQA + g) * HEAD_DIM + d]) * scale;
    o_s[g][d] = 0.f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m[GQA];
  float l[GQA];
  float o[GQA][ROWS];
  for (int g = 0; g < GQA; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.f;
    for (int r = 0; r < ROWS; ++r) {
      o[g][r] = 0.f;
    }
  }

  for (int b = start_block + int(sg); b < end_block; b += PD_SIMDS) {
    const uint64_t phys = block_tables[seq * max_num_blocks_per_seq + b];
    const device T *kb = k_cache + phys * kv_block_stride + kv_head * kv_head_stride;
    const device T *vb = v_cache + phys * kv_block_stride + kv_head * kv_head_stride;
    const int block_tokens = min(PD_BLOCK, part_end - b * PD_BLOCK);
    const bool valid = int(lane) < block_tokens;

    float s[GQA];
    for (int g = 0; g < GQA; ++g) {
      s[g] = 0.f;
    }
    for (int c = 0; c < CHUNKS; ++c) {
      float k8[8];
      pd_load8<T>(kb + (c * PD_BLOCK + lane) * 8, k8);
      for (int g = 0; g < GQA; ++g) {
        const threadgroup float *qc = &q_s[g][c * 8];
        float acc = 0.f;
        for (int e = 0; e < 8; ++e) {
          acc = fma(qc[e], k8[e], acc);
        }
        s[g] += acc;
      }
    }

    for (int g = 0; g < GQA; ++g) {
      const float sv = valid ? s[g] : -INFINITY;
      const float new_m = max(m[g], simd_max(sv));
      const float alpha = exp(m[g] - new_m);
      const float p = valid ? exp(sv - new_m) : 0.f;
      l[g] = l[g] * alpha + simd_sum(p);
      for (int r = 0; r < ROWS; ++r) {
        o[g][r] *= alpha;
      }
      m[g] = new_m;
      p_s[sg][g][lane] = p;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = 0; r < ROWS; ++r) {
      const device T *vr = vb + (lane + 32 * r) * PD_BLOCK;
      float v[PD_BLOCK];
      for (int t = 0; t < PD_BLOCK; t += 8) {
        pd_load8<T>(vr + t, &v[t]);
      }
      // Slots past the context hold stale cache contents; zero them so 0 * NaN cannot leak in.
      if (block_tokens < PD_BLOCK) {
        for (int t = block_tokens; t < PD_BLOCK; ++t) {
          v[t] = 0.f;
        }
      }
      for (int g = 0; g < GQA; ++g) {
        float acc = 0.f;
        for (int t = 0; t < PD_BLOCK; ++t) {
          acc = fma(p_s[sg][g][t], v[t], acc);
        }
        o[g][r] += acc;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    for (int g = 0; g < GQA; ++g) {
      red_m[sg][g] = m[g];
      red_l[sg][g] = l[g];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float part_m[GQA];
  float part_l[GQA];
  for (int g = 0; g < GQA; ++g) {
    float mx = -INFINITY;
    for (int k = 0; k < PD_SIMDS; ++k) {
      mx = max(mx, red_m[k][g]);
    }
    float sum = 0.f;
    for (int k = 0; k < PD_SIMDS; ++k) {
      sum += red_l[k][g] * exp(red_m[k][g] - mx);
    }
    part_m[g] = mx;
    part_l[g] = sum;
  }

  for (int k = 0; k < PD_SIMDS; ++k) {
    if (int(sg) == k) {
      for (int g = 0; g < GQA; ++g) {
        const float f = exp(m[g] - part_m[g]);
        for (int r = 0; r < ROWS; ++r) {
          o_s[g][lane + 32 * r] += o[g][r] * f;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int i = tid; i < GQA * HEAD_DIM; i += PD_THREADS) {
    const int g = i / HEAD_DIM;
    const int d = i % HEAD_DIM;
    const int head = kv_head * GQA + g;
    const float val = o_s[g][d] / part_l[g];
    if (max_num_partitions == 1) {
      out[(seq * num_heads + head) * HEAD_DIM + d] = T(val);
    } else {
      tmp_out[((seq * num_heads + head) * max_num_partitions + part) * HEAD_DIM + d] = T(val);
    }
  }
  if (max_num_partitions > 1 && tid < uint(GQA)) {
    const int head = kv_head * GQA + int(tid);
    const int idx = (seq * num_heads + head) * max_num_partitions + part;
    max_logits[idx] = part_m[tid];
    exp_sums[idx] = part_l[tid];
  }
}

// Adapted from Candle's MLX-derived sdpa_vector; licensed under Apache-2.0.
// Copyright (c) 2023 Apple Inc. https://github.com/ml-explore/mlx
template <typename T, int HEAD_DIM>
[[kernel]] void
paged_decode_short(device T *out [[buffer(0)]], const device T *q [[buffer(4)]],
                   const device T *k_cache [[buffer(5)]],
                   const device T *v_cache [[buffer(6)]],
                   const device uint32_t *block_tables [[buffer(7)]],
                   const device uint32_t *context_lens [[buffer(8)]],
                   constant int &max_num_blocks_per_seq [[buffer(9)]],
                   constant int &q_stride [[buffer(10)]],
                   constant int &kv_block_stride [[buffer(11)]],
                   constant int &kv_head_stride [[buffer(12)]],
                   constant float &scale [[buffer(13)]],
                   constant int &gqa [[buffer(15)]],
                   uint3 tg [[threadgroup_position_in_grid]],
                   uint3 tgs [[threadgroups_per_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ELEMS = HEAD_DIM / PD_SIMD_WIDTH;
  const int head = tg.y;
  const int seq = tg.z;
  const int kv_head = head / gqa;
  const int context_len = context_lens[seq];
  const int dim_start = lane * ELEMS;

  threadgroup float maxima[PD_SHORT_SIMDS];
  threadgroup float sums[PD_SHORT_SIMDS];
  threadgroup float outputs[PD_SIMD_WIDTH * PD_SHORT_SIMDS];
  float query[ELEMS];
  float o[ELEMS];
  for (int d = 0; d < ELEMS; ++d) {
    query[d] =
        float(q[seq * q_stride + head * HEAD_DIM + dim_start + d]) * scale;
    o[d] = 0.f;
  }

  float m = -INFINITY;
  float l = 0.f;
  for (int token = int(sg); token < context_len; token += PD_SHORT_SIMDS) {
    const int block = token / PD_BLOCK;
    const int slot = token % PD_BLOCK;
    const uint64_t phys = block_tables[seq * max_num_blocks_per_seq + block];
    const device T *kb =
        k_cache + phys * kv_block_stride + kv_head * kv_head_stride;
    const device T *vb =
        v_cache + phys * kv_block_stride + kv_head * kv_head_stride;
    float score = 0.f;
    for (int d = 0; d < ELEMS; ++d) {
      const int dim = dim_start + d;
      const float key =
          float(kb[((dim / PD_KEY_PACK) * PD_BLOCK + slot) * PD_KEY_PACK +
                   dim % PD_KEY_PACK]);
      score = fma(query[d], key, score);
    }
    score = simd_sum(score);
    const float new_m = max(m, score);
    const float alpha = fast::exp(m - new_m);
    const float p = fast::exp(score - new_m);
    l = l * alpha + p;
    m = new_m;
    for (int d = 0; d < ELEMS; ++d) {
      const float value = float(vb[(dim_start + d) * PD_BLOCK + slot]);
      o[d] = fma(p, value, o[d] * alpha);
    }
  }

  if (lane == 0) {
    maxima[sg] = m;
    sums[sg] = l;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float max_logit = simd_max(maxima[lane]);
  const float weight = fast::exp(maxima[lane] - max_logit);
  const float denominator = simd_sum(sums[lane] * weight);

  for (int d = 0; d < ELEMS; ++d) {
    outputs[lane * PD_SHORT_SIMDS + sg] = o[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[d] = simd_sum(outputs[sg * PD_SHORT_SIMDS + lane] * weight) / denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    const int output_start =
        (seq * int(tgs.y) + head) * HEAD_DIM + int(sg) * ELEMS;
    for (int d = 0; d < ELEMS; ++d) {
      out[output_start + d] = T(o[d]);
    }
  }
}

#define instantiate_paged_decode(type, head_dim, gqa)                          \
  template [[host_name("paged_decode_gqa_" #type "_hs" #head_dim "_g" #gqa)]]  \
  [[kernel]] void paged_decode_gqa<type, head_dim, gqa>(                       \
      device type * out [[buffer(0)]], device float *exp_sums [[buffer(1)]],  \
      device float *max_logits [[buffer(2)]],                                  \
      device type *tmp_out [[buffer(3)]], const device type *q [[buffer(4)]],  \
      const device type *k_cache [[buffer(5)]],                                \
      const device type *v_cache [[buffer(6)]],                                \
      const device uint32_t *block_tables [[buffer(7)]],                       \
      const device uint32_t *context_lens [[buffer(8)]],                       \
      constant int &max_num_blocks_per_seq [[buffer(9)]],                      \
      constant int &q_stride [[buffer(10)]],                                   \
      constant int &kv_block_stride [[buffer(11)]],                            \
      constant int &kv_head_stride [[buffer(12)]],                             \
      constant float &scale [[buffer(13)]],                                    \
      constant int &max_num_partitions [[buffer(14)]],                         \
      constant int &partition_size [[buffer(15)]],                             \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 tgs [[threadgroups_per_grid]],                                     \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint tid [[thread_index_in_threadgroup]]);

#define instantiate_paged_decode_gqa(type, head_dim)                           \
  instantiate_paged_decode(type, head_dim, 1)                                  \
  instantiate_paged_decode(type, head_dim, 2)                                  \
  instantiate_paged_decode(type, head_dim, 3)                                  \
  instantiate_paged_decode(type, head_dim, 4)                                  \
  instantiate_paged_decode(type, head_dim, 5)                                  \
  instantiate_paged_decode(type, head_dim, 6)                                  \
  instantiate_paged_decode(type, head_dim, 7)                                  \
  instantiate_paged_decode(type, head_dim, 8)

#define instantiate_paged_decode_heads(type)                                   \
  instantiate_paged_decode_gqa(type, 64)                                       \
  instantiate_paged_decode_gqa(type, 96)                                       \
  instantiate_paged_decode_gqa(type, 128)                                      \
  instantiate_paged_decode_gqa(type, 192)                                      \
  instantiate_paged_decode_gqa(type, 256)

instantiate_paged_decode_heads(bfloat16_t)
instantiate_paged_decode_heads(half)

#define instantiate_paged_decode_short(type, head_dim)                         \
  template [[host_name("paged_decode_short_" #type                             \
                       "_hs" #head_dim)]] [[kernel]] void                      \
  paged_decode_short<type, head_dim>(                                          \
      device type * out [[buffer(0)]], const device type *q [[buffer(4)]],     \
      const device type *k_cache [[buffer(5)]],                                \
      const device type *v_cache [[buffer(6)]],                                \
      const device uint32_t *block_tables [[buffer(7)]],                       \
      const device uint32_t *context_lens [[buffer(8)]],                       \
      constant int &max_num_blocks_per_seq [[buffer(9)]],                      \
      constant int &q_stride [[buffer(10)]],                                   \
      constant int &kv_block_stride [[buffer(11)]],                            \
      constant int &kv_head_stride [[buffer(12)]],                             \
      constant float &scale [[buffer(13)]], constant int &gqa [[buffer(15)]],  \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 tgs [[threadgroups_per_grid]],                                     \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]);

#define instantiate_paged_decode_short_heads(type)                             \
  instantiate_paged_decode_short(type, 128)                                    \
  instantiate_paged_decode_short(type, 256)

instantiate_paged_decode_short_heads(bfloat16_t)
instantiate_paged_decode_short_heads(half)
