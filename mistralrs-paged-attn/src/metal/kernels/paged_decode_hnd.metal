#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

constant constexpr int PD_BLOCK = 32;
constant constexpr int PD_SIMDS = 4;
constant constexpr int PD_THREADS = PD_SIMDS * 32;
constant constexpr int PD_SHORT_SIMDS = 32;
constant constexpr int PD_SIMD_WIDTH = 32;
constant constexpr int PD_SPLIT_SIMDS = 8;
constant constexpr int PD_SPLIT_PARTITIONS = 32;

template <typename T>
inline void pd_load8(const device T *p, thread float *out);

template <>
inline void pd_load8<half>(const device half *p, thread float *out) {
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
inline void pd_load8<bfloat16_t>(const device bfloat16_t *p,
                                 thread float *out) {
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
[[kernel]] void paged_decode_hnd_gqa(
    device T *out [[buffer(0)]], // [seqs, heads, D], single-partition launches
    device float *exp_sums [[buffer(1)]],   // [seqs, heads, max_parts]
    device float *max_logits [[buffer(2)]], // [seqs, heads, max_parts]
    device T *tmp_out [[buffer(3)]],        // [seqs, heads, max_parts, D]
    const device T *q [[buffer(4)]],        // [seqs, heads, D] with q_stride
    const device T *k_cache [[buffer(5)]],  // [blocks, kv_heads, PD_BLOCK, D]
    const device T *v_cache [[buffer(6)]],  // [blocks, kv_heads, PD_BLOCK, D]
    const device uint32_t *block_tables [[buffer(7)]],
    const device uint32_t *context_lens [[buffer(8)]],
    constant int &max_num_blocks_per_seq [[buffer(9)]],
    constant int &q_stride [[buffer(10)]],
    constant int &kv_block_stride [[buffer(11)]],
    constant int &kv_head_stride [[buffer(12)]],
    constant float &scale [[buffer(13)]],
    constant int &max_num_partitions [[buffer(14)]],
    constant int &partition_size [[buffer(15)]],
    constant int &sliding_window [[buffer(16)]],
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
  const int first_token =
      sliding_window > 0 ? max(0, context_len - sliding_window) : 0;
  if (part_end <= first_token) {
    for (int g = 0; g < GQA; ++g) {
      const int index =
          (seq * num_heads + kv_head * GQA + g) * max_num_partitions + part;
      if (tid == 0) {
        exp_sums[index] = 0.f;
        max_logits[index] = -FLT_MAX;
      }
      for (int d = tid; d < HEAD_DIM; d += PD_THREADS) {
        tmp_out[index * HEAD_DIM + d] = T(0);
      }
    }
    return;
  }
  const int start_block = max(part_start, first_token) / PD_BLOCK;
  const int end_block = (part_end + PD_BLOCK - 1) / PD_BLOCK;

  threadgroup float q_s[GQA][HEAD_DIM];
  threadgroup float p_s[PD_SIMDS][GQA][PD_BLOCK];
  threadgroup float o_s[GQA][HEAD_DIM];
  threadgroup float red_m[PD_SIMDS][GQA];
  threadgroup float red_l[PD_SIMDS][GQA];

  for (int i = tid; i < GQA * HEAD_DIM; i += PD_THREADS) {
    const int g = i / HEAD_DIM;
    const int d = i % HEAD_DIM;
    q_s[g][d] =
        float(q[seq * q_stride + (kv_head * GQA + g) * HEAD_DIM + d]) * scale;
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
    const device T *kb =
        k_cache + phys * kv_block_stride + kv_head * kv_head_stride;
    const device T *vb =
        v_cache + phys * kv_block_stride + kv_head * kv_head_stride;
    const int block_tokens = min(PD_BLOCK, part_end - b * PD_BLOCK);
    const int block_first = max(0, first_token - b * PD_BLOCK);
    const bool valid = int(lane) >= block_first && int(lane) < block_tokens;

    float s[GQA];
    for (int g = 0; g < GQA; ++g) {
      s[g] = 0.f;
    }
    for (int c = 0; c < CHUNKS; ++c) {
      float k8[8];
      pd_load8<T>(kb + lane * HEAD_DIM + c * 8, k8);
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
      float v[PD_BLOCK];
      for (int t = 0; t < PD_BLOCK; ++t) {
        v[t] = t >= block_first && t < block_tokens
                   ? float(vb[t * HEAD_DIM + lane + 32 * r])
                   : 0.f;
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
      tmp_out[((seq * num_heads + head) * max_num_partitions + part) *
                  HEAD_DIM +
              d] = T(val);
    }
  }
  if (max_num_partitions > 1 && tid < uint(GQA)) {
    const int head = kv_head * GQA + int(tid);
    const int idx = (seq * num_heads + head) * max_num_partitions + part;
    max_logits[idx] = part_m[tid];
    exp_sums[idx] = part_l[tid];
  }
}

#define instantiate_paged_decode(type, head_dim, gqa)                          \
  template [[host_name("paged_decode_hnd_gqa_" #type "_hs" #head_dim           \
                       "_g" #gqa)]] [[kernel]] void                            \
  paged_decode_hnd_gqa<type, head_dim, gqa>(                                   \
      device type * out [[buffer(0)]], device float *exp_sums [[buffer(1)]],   \
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
      constant int &sliding_window [[buffer(16)]],                             \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 tgs [[threadgroups_per_grid]],                                     \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint tid [[thread_index_in_threadgroup]]);

#define instantiate_paged_decode_hnd_gqa(type, head_dim)                       \
  instantiate_paged_decode(type, head_dim, 1)                                  \
      instantiate_paged_decode(type, head_dim, 2)                              \
          instantiate_paged_decode(type, head_dim, 3)                          \
              instantiate_paged_decode(type, head_dim, 4)                      \
                  instantiate_paged_decode(type, head_dim, 5)                  \
                      instantiate_paged_decode(type, head_dim, 6)              \
                          instantiate_paged_decode(type, head_dim, 7)          \
                              instantiate_paged_decode(type, head_dim, 8)

#define instantiate_paged_decode_heads(type)                                   \
  instantiate_paged_decode_hnd_gqa(type, 64)                                   \
      instantiate_paged_decode_hnd_gqa(type, 96)                               \
          instantiate_paged_decode_hnd_gqa(type, 128)                          \
              instantiate_paged_decode_hnd_gqa(type, 192)                      \
                  instantiate_paged_decode_hnd_gqa(type, 256)

instantiate_paged_decode_heads(bfloat16_t) instantiate_paged_decode_heads(half)

    instantiate_paged_decode(half, 512, 1)
        instantiate_paged_decode(half, 512, 2)
            instantiate_paged_decode(half, 512, 3)
                instantiate_paged_decode(half, 512, 4)
                    instantiate_paged_decode(bfloat16_t, 512, 1)
                        instantiate_paged_decode(bfloat16_t, 512, 2)
                            instantiate_paged_decode(bfloat16_t, 512, 3)
                                instantiate_paged_decode(bfloat16_t, 512, 4)

    // Adapted from Candle's MLX-derived sdpa_vector; licensed under Apache-2.0.
    // Copyright (c) 2023 Apple Inc. https://github.com/ml-explore/mlx
    template <typename T, int HEAD_DIM>
    [[kernel]] void paged_decode_hnd_short(
        device T *out [[buffer(0)]], const device T *q [[buffer(4)]],
        const device T *k_cache [[buffer(5)]],
        const device T *v_cache [[buffer(6)]],
        const device uint32_t *block_tables [[buffer(7)]],
        const device uint32_t *context_lens [[buffer(8)]],
        constant int &max_num_blocks_per_seq [[buffer(9)]],
        constant int &q_stride [[buffer(10)]],
        constant int &kv_block_stride [[buffer(11)]],
        constant int &kv_head_stride [[buffer(12)]],
        constant float &scale [[buffer(13)]], constant int &gqa [[buffer(15)]],
        constant int &sliding_window [[buffer(16)]],
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
  const int first_token =
      sliding_window > 0 ? max(0, context_len - sliding_window) : 0;

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
  for (int token = first_token + int(sg); token < context_len;
       token += PD_SHORT_SIMDS) {
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
      const float key = float(kb[slot * HEAD_DIM + dim]);
      score = fma(query[d], key, score);
    }
    score = simd_sum(score);
    const float new_m = max(m, score);
    const float alpha = fast::exp(m - new_m);
    const float p = fast::exp(score - new_m);
    l = l * alpha + p;
    m = new_m;
    for (int d = 0; d < ELEMS; ++d) {
      const float value = float(vb[slot * HEAD_DIM + dim_start + d]);
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

#define instantiate_paged_decode_hnd_short(type, head_dim)                     \
  template [[host_name("paged_decode_hnd_short_" #type                         \
                       "_hs" #head_dim)]] [[kernel]] void                      \
  paged_decode_hnd_short<type, head_dim>(                                      \
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
      constant int &sliding_window [[buffer(16)]],                             \
      uint3 tg [[threadgroup_position_in_grid]],                               \
      uint3 tgs [[threadgroups_per_grid]],                                     \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_paged_decode_hnd_short(half, 64) instantiate_paged_decode_hnd_short(
    half, 96) instantiate_paged_decode_hnd_short(half, 128)
    instantiate_paged_decode_hnd_short(
        half, 192) instantiate_paged_decode_hnd_short(half, 256)
        instantiate_paged_decode_hnd_short(half, 512)
            instantiate_paged_decode_hnd_short(bfloat16_t, 64)
                instantiate_paged_decode_hnd_short(bfloat16_t, 96)
                    instantiate_paged_decode_hnd_short(bfloat16_t, 128)
                        instantiate_paged_decode_hnd_short(bfloat16_t, 192)
                            instantiate_paged_decode_hnd_short(bfloat16_t, 256)
                                instantiate_paged_decode_hnd_short(bfloat16_t,
                                                                   512)

template <typename T, int HEAD_DIM>
[[kernel]] void paged_decode_hnd_split(
    device float *exp_sums [[buffer(1)]],
    device float *max_logits [[buffer(2)]],
    device float *partials [[buffer(3)]], const device T *q [[buffer(4)]],
    const device T *k_cache [[buffer(5)]],
    const device T *v_cache [[buffer(6)]],
    const device uint32_t *block_tables [[buffer(7)]],
    const device uint32_t *context_lens [[buffer(8)]],
    constant int &max_num_blocks_per_seq [[buffer(9)]],
    constant int &q_stride [[buffer(10)]],
    constant int &kv_block_stride [[buffer(11)]],
    constant int &kv_head_stride [[buffer(12)]],
    constant float &scale [[buffer(13)]], constant int &gqa [[buffer(15)]],
    constant int &sliding_window [[buffer(16)]],
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
  const int visible_first =
      sliding_window > 0 ? max(0, context_len - sliding_window) : 0;
  const int part_size =
      (context_len + PD_SPLIT_PARTITIONS - 1) / PD_SPLIT_PARTITIONS;
  const int first_token = max(visible_first, int(tg.x) * part_size);
  const int end_token = min(context_len, (int(tg.x) + 1) * part_size);
  const int index = (seq * int(tgs.y) + head) * PD_SPLIT_PARTITIONS + int(tg.x);
  if (first_token >= end_token) {
    if (sg == 0 && lane == 0) {
      exp_sums[index] = 0.f;
      max_logits[index] = -INFINITY;
    }
    for (int d = int(sg) * PD_SIMD_WIDTH + int(lane); d < HEAD_DIM;
         d += PD_SPLIT_SIMDS * PD_SIMD_WIDTH) {
      partials[index * HEAD_DIM + d] = 0.f;
    }
    return;
  }

  threadgroup float maxima[PD_SPLIT_SIMDS];
  threadgroup float sums[PD_SPLIT_SIMDS];
  threadgroup float outputs[PD_SIMD_WIDTH * PD_SPLIT_SIMDS];
  float query[ELEMS];
  float o[ELEMS];
  for (int d = 0; d < ELEMS; ++d) {
    query[d] =
        float(q[seq * q_stride + head * HEAD_DIM + dim_start + d]) * scale;
    o[d] = 0.f;
  }

  float m = -INFINITY;
  float l = 0.f;
  for (int token = first_token + int(sg); token < end_token;
       token += PD_SPLIT_SIMDS) {
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
      const float key = float(kb[slot * HEAD_DIM + dim]);
      score = fma(query[d], key, score);
    }
    score = simd_sum(score);
    const float new_m = max(m, score);
    const float alpha = fast::exp(m - new_m);
    const float p = fast::exp(score - new_m);
    l = l * alpha + p;
    m = new_m;
    for (int d = 0; d < ELEMS; ++d) {
      const float value = float(vb[slot * HEAD_DIM + dim_start + d]);
      o[d] = fma(p, value, o[d] * alpha);
    }
  }

  if (lane == 0) {
    maxima[sg] = m;
    sums[sg] = l;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float local_max = lane < PD_SPLIT_SIMDS ? maxima[lane] : -INFINITY;
  const float max_logit = simd_max(local_max);
  const float weight = fast::exp(local_max - max_logit);
  const float denominator =
      simd_sum((lane < PD_SPLIT_SIMDS ? sums[lane] : 0.f) * weight);
  if (sg == 0 && lane == 0) {
    exp_sums[index] = denominator;
    max_logits[index] = max_logit;
  }
  for (int d = 0; d < ELEMS; ++d) {
    outputs[lane * PD_SPLIT_SIMDS + sg] =
        o[d] * fast::exp(maxima[sg] - max_logit);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float value = outputs[lane * PD_SPLIT_SIMDS];
      for (int s = 1; s < PD_SPLIT_SIMDS; ++s)
        value += outputs[lane * PD_SPLIT_SIMDS + s];
      partials[index * HEAD_DIM + int(lane) * ELEMS + d] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, int D>
[[kernel]] void
paged_decode_hnd_split_reduce(device T *out [[buffer(0)]],
                              const device float *sums [[buffer(1)]],
                              const device float *maxs [[buffer(2)]],
                              const device float *partials [[buffer(3)]],
                              uint3 tid [[threadgroup_position_in_grid]],
                              uint3 tgs [[threadgroups_per_grid]],
                              uint simd_gid [[simdgroup_index_in_threadgroup]],
                              uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = PD_SPLIT_PARTITIONS;
  constexpr int BD = PD_SIMD_WIDTH;
  constexpr int elem_per_thread = D / BD;
  constexpr int blocks = PD_SPLIT_PARTITIONS;

  typedef float U;

  thread U o[elem_per_thread];
  threadgroup U outputs[BN * BD];

  const int head_idx = tid.y * tgs.x + tid.x;
  partials += head_idx * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += head_idx * blocks;
  maxs += head_idx * blocks;
  out += head_idx * D + simd_gid * elem_per_thread;

  U max_score = maxs[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sums[simd_lid] * factor);

  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = partials[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template [[host_name("paged_decode_hnd_split_half_hs512")]] [[kernel]] void
paged_decode_hnd_split<half, 512>(
    device float *exp_sums [[buffer(1)]],
    device float *max_logits [[buffer(2)]],
    device float *partials [[buffer(3)]], const device half *q [[buffer(4)]],
    const device half *k_cache [[buffer(5)]],
    const device half *v_cache [[buffer(6)]],
    const device uint32_t *block_tables [[buffer(7)]],
    const device uint32_t *context_lens [[buffer(8)]],
    constant int &max_num_blocks_per_seq [[buffer(9)]],
    constant int &q_stride [[buffer(10)]],
    constant int &kv_block_stride [[buffer(11)]],
    constant int &kv_head_stride [[buffer(12)]],
    constant float &scale [[buffer(13)]], constant int &gqa [[buffer(15)]],
    constant int &sliding_window [[buffer(16)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]);
template
    [[host_name("paged_decode_hnd_split_reduce_half_hs512")]] [[kernel]] void
    paged_decode_hnd_split_reduce<half, 512>(
        device half *out [[buffer(0)]], const device float *sums [[buffer(1)]],
        const device float *maxs [[buffer(2)]],
        const device float *partials [[buffer(3)]],
        uint3 tid [[threadgroup_position_in_grid]],
        uint3 tgs [[threadgroups_per_grid]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]]);
template
    [[host_name("paged_decode_hnd_split_bfloat16_t_hs512")]] [[kernel]] void
    paged_decode_hnd_split<bfloat16_t, 512>(
        device float *exp_sums [[buffer(1)]],
        device float *max_logits [[buffer(2)]],
        device float *partials [[buffer(3)]],
        const device bfloat16_t *q [[buffer(4)]],
        const device bfloat16_t *k_cache [[buffer(5)]],
        const device bfloat16_t *v_cache [[buffer(6)]],
        const device uint32_t *block_tables [[buffer(7)]],
        const device uint32_t *context_lens [[buffer(8)]],
        constant int &max_num_blocks_per_seq [[buffer(9)]],
        constant int &q_stride [[buffer(10)]],
        constant int &kv_block_stride [[buffer(11)]],
        constant int &kv_head_stride [[buffer(12)]],
        constant float &scale [[buffer(13)]], constant int &gqa [[buffer(15)]],
        constant int &sliding_window [[buffer(16)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint3 tgs [[threadgroups_per_grid]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]);
template [[host_name(
    "paged_decode_hnd_split_reduce_bfloat16_t_hs512")]] [[kernel]] void
paged_decode_hnd_split_reduce<bfloat16_t, 512>(
    device bfloat16_t *out [[buffer(0)]],
    const device float *sums [[buffer(1)]],
    const device float *maxs [[buffer(2)]],
    const device float *partials [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tgs [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);
