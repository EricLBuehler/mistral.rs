#include "float8.metal"
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

constant ushort PP_WIDTH = 32;
constant ushort PP_TILE = 8;
constant ushort PP_MAX_VALUE = 4096;
constant ushort PP_MATRIX_VALUE = 512;
constant ulong PP_ABSENT = ulong(-1);

constant ulong PP_HEADS [[function_constant(100)]];
constant ulong PP_KV_HEADS [[function_constant(101)]];
constant ulong PP_KEY_DIM [[function_constant(102)]];
constant ulong PP_VALUE_DIM [[function_constant(103)]];
constant ulong PP_BLOCK_SIZE [[function_constant(104)]];
constant ulong PP_PACKED_X [[function_constant(105)]];
constant ulong PP_HND [[function_constant(106)]];
constant ulong PP_PACKED_QUERIES [[function_constant(107)]];
constant ulong PP_CAUSAL [[function_constant(108)]];
constant ulong PP_WINDOW_LEFT [[function_constant(109)]];
constant ulong PP_QUERY_DTYPE [[function_constant(110)]];
constant ulong PP_CACHE_DTYPE [[function_constant(111)]];
constant ulong PP_MASK_DTYPE [[function_constant(112)]];
constant ulong PP_ALIBI_DTYPE [[function_constant(113)]];
constant ulong PP_SINK_DTYPE [[function_constant(114)]];
constant ulong PP_PACKED_MASK [[function_constant(115)]];
constant ulong PP_HAS_SCALES [[function_constant(116)]];
constant ulong PP_WINDOW_RIGHT [[function_constant(117)]];
constant ulong PP_ATTENTION_CHUNK_SIZE [[function_constant(118)]];
constant bool PP_HAS_SOFTCAP [[function_constant(119)]];
constant bool PP_HAS_RANGES [[function_constant(120)]];
constant ushort PP_BLOCK_SHIFT [[function_constant(121)]];

struct PrefillParams {
  ulong query_strides[4];
  ulong key_strides[5];
  ulong value_strides[4];
  ulong table_strides[2];
  ulong mask_strides[4];
  ulong mask_shape[4];
  ulong heads;
  ulong kv_heads;
  ulong key_dim;
  ulong value_dim;
  ulong block_size;
  ulong packed_x;
  ulong query_axis;
  ulong hnd;
  ulong packed_queries;
  ulong causal;
  ulong window_left;
  ulong query_dtype;
  ulong cache_dtype;
  ulong mask_dtype;
  ulong alibi_dtype;
  ulong sink_dtype;
  ulong packed_mask;
  ulong batch;
  ulong blocks;
  ulong has_scales;
  ulong alibi_stride;
  ulong sink_stride;
  ulong window_right;
  ulong attention_chunk_size;
};

inline __attribute__((always_inline)) float pp_load(const device uchar *data,
                                                    ulong offset, ulong dtype) {
  if (dtype == 0)
    return float(((const device half *)data)[offset]);
  if (dtype == 1)
    return as_type<float>(uint(((const device ushort *)data)[offset]) << 16);
  if (dtype == 2)
    return ((const device float *)data)[offset];
  return fp8_e4m3_to_float(data[offset]);
}

inline __attribute__((always_inline)) void
pp_store(device uchar *data, ulong offset, ulong dtype, float value) {
  if (dtype == 0)
    ((device half *)data)[offset] = half(value);
  else if (dtype == 1) {
    uint bits = as_type<uint>(value);
    bits = isnan(value) ? 0x7fc00000u : bits + 0x7fffu + ((bits >> 16) & 1u);
    ((device ushort *)data)[offset] = ushort(bits >> 16);
  } else
    ((device float *)data)[offset] = value;
}

inline __attribute__((always_inline)) ulong pp_block_index(ulong token) {
  return PP_BLOCK_SHIFT < 64 ? token >> PP_BLOCK_SHIFT : token / PP_BLOCK_SIZE;
}

inline __attribute__((always_inline)) ulong pp_block_offset(ulong token) {
  return PP_BLOCK_SHIFT < 64 ? token & (PP_BLOCK_SIZE - 1)
                             : token % PP_BLOCK_SIZE;
}

inline __attribute__((always_inline)) ulong
pp_key_offset(constant PrefillParams &p, ulong physical, ulong head, ulong slot,
              ulong dim) {
  ulong base = physical * p.key_strides[0] + head * p.key_strides[1];
  return PP_HND ? base + slot * p.key_strides[2] + dim * p.key_strides[3]
                : base + (dim / PP_PACKED_X) * p.key_strides[2] +
                      slot * p.key_strides[3] +
                      (dim % PP_PACKED_X) * p.key_strides[4];
}

inline __attribute__((always_inline)) ulong
pp_value_offset(constant PrefillParams &p, ulong physical, ulong head,
                ulong slot, ulong dim) {
  ulong base = physical * p.value_strides[0] + head * p.value_strides[1];
  return PP_HND ? base + slot * p.value_strides[2] + dim * p.value_strides[3]
                : base + dim * p.value_strides[2] + slot * p.value_strides[3];
}

inline __attribute__((always_inline)) bool pp_allowed(constant PrefillParams &p,
                                                      const device ulong *meta,
                                                      const device ulong *row,
                                                      ulong query, ulong key) {
  ulong qp = row[2] + query;
  ulong kp = row[3] + key;
  if (PP_ATTENTION_CHUNK_SIZE &&
      qp / PP_ATTENTION_CHUNK_SIZE != kp / PP_ATTENTION_CHUNK_SIZE)
    return false;
  bool allowed = (!PP_CAUSAL || kp <= qp) &&
                 (!PP_WINDOW_LEFT || kp > qp || qp - kp < PP_WINDOW_LEFT) &&
                 (!PP_WINDOW_RIGHT || kp < qp || kp - qp < PP_WINDOW_RIGHT);
  if (allowed)
    return true;
  for (ulong i = 0; PP_HAS_RANGES && i < row[7]; ++i) {
    ulong start = meta[row[6] + 2 * i];
    ulong end = meta[row[6] + 2 * i + 1];
    if (qp >= start && qp < end && kp >= start && kp < end)
      return true;
  }
  return false;
}

inline __attribute__((always_inline)) float
pp_score(constant PrefillParams &p, constant float *scalars,
         const device ulong *meta, const device ulong *row,
         const device uchar *mask, const device uchar *alibi, ulong seq,
         ulong head, ulong query, ulong key, float dot) {
  if (query >= row[0] || key >= row[1] || !pp_allowed(p, meta, row, query, key))
    return -INFINITY;
  float value = dot * scalars[0];
  if (PP_HAS_SOFTCAP)
    value = tanh(value / scalars[1]) * scalars[1];
  if (PP_ALIBI_DTYPE != PP_ABSENT) {
    ulong key_position = row[3] + key;
    ulong query_position = row[2] + query;
    float distance = key_position >= query_position
                         ? float(key_position - query_position)
                         : -float(query_position - key_position);
    value += pp_load(alibi, head * p.alibi_stride, PP_ALIBI_DTYPE) * distance;
  }
  if (PP_MASK_DTYPE != PP_ABSENT) {
    ulong mb = PP_PACKED_MASK ? 0 : seq;
    ulong mq = query + (PP_PACKED_MASK ? row[4] : 0);
    ulong mk = key + row[8];
    ulong index = (p.mask_shape[0] == 1 ? 0 : mb) * p.mask_strides[0] +
                  (p.mask_shape[1] == 1 ? 0 : head) * p.mask_strides[1] +
                  (p.mask_shape[2] == 1 ? 0 : mq) * p.mask_strides[2] +
                  (p.mask_shape[3] == 1 ? 0 : mk) * p.mask_strides[3];
    value += pp_load(mask, index, PP_MASK_DTYPE);
  }
  return value;
}

inline __attribute__((always_inline)) ulong
pp_query_base(constant PrefillParams &p, const device ulong *row, ulong seq,
              ulong head, ulong query) {
  return (PP_PACKED_QUERIES ? 0 : seq) * p.query_strides[0] +
         head * p.query_strides[1] +
         (query + (PP_PACKED_QUERIES ? row[4] : 0)) * p.query_strides[2];
}

inline __attribute__((always_inline)) ulong
pp_output_base(constant PrefillParams &p, const device ulong *row, ulong seq,
               ulong head, ulong query) {
  return (((PP_PACKED_QUERIES ? 0 : seq) * PP_HEADS + head) * p.query_axis +
          query + (PP_PACKED_QUERIES ? row[4] : 0)) *
         PP_VALUE_DIM;
}

inline __attribute__((always_inline)) ulong2
pp_key_bounds(constant PrefillParams &p, const device ulong *meta,
              const device ulong *row, ulong query, ulong count) {
  ulong first = 0;
  ulong last = row[1];
  ulong query_start = row[2] + query;
  ulong query_end = query_start + min(count, row[0] - query);
  if (PP_CAUSAL)
    last = query_end <= row[3] ? 0 : min(last, query_end - row[3]);
  if (PP_WINDOW_RIGHT) {
    ulong exclusive = query_end - 1 + PP_WINDOW_RIGHT;
    last = exclusive <= row[3] ? 0 : min(last, exclusive - row[3]);
  }
  if (PP_WINDOW_LEFT && query_start >= PP_WINDOW_LEFT) {
    ulong first_position = query_start - PP_WINDOW_LEFT + 1;
    first = first_position > row[3] ? min(last, first_position - row[3]) : 0;
  }
  for (ulong i = 0; PP_HAS_RANGES && i < row[7]; ++i) {
    ulong start = meta[row[6] + 2 * i];
    ulong end = meta[row[6] + 2 * i + 1];
    if (start >= query_end || end <= query_start)
      continue;
    ulong range_first = start > row[3] ? min(row[1], start - row[3]) : 0;
    ulong range_last = end > row[3] ? min(row[1], end - row[3]) : 0;
    if (range_first >= range_last)
      continue;
    if (first < last) {
      first = min(first, range_first);
      last = max(last, range_last);
    } else {
      first = range_first;
      last = range_last;
    }
  }
  return ulong2(first, last);
}

kernel void paged_prefill_metadata_copy(constant ulong *source [[buffer(0)]],
                                        device ulong *target [[buffer(1)]],
                                        constant ulong &count [[buffer(2)]],
                                        uint index
                                        [[thread_position_in_grid]]) {
  if (index < count)
    target[index] = source[index];
}

#define PP_ARGUMENTS                                                           \
  constant PrefillParams &p [[buffer(0)]],                                     \
      constant float *scalars [[buffer(1)]],                                   \
      const device uchar *q [[buffer(2)]],                                     \
      const device uchar *k [[buffer(3)]],                                     \
      const device uchar *v [[buffer(4)]],                                     \
      const device uint *table [[buffer(5)]],                                  \
      const device uchar *mask [[buffer(6)]],                                  \
      const device uchar *alibi [[buffer(7)]],                                 \
      const device uchar *sinks [[buffer(8)]],                                 \
      const device float *k_scale [[buffer(9)]],                               \
      const device float *v_scale [[buffer(10)]],                              \
      const device ulong *meta [[buffer(11)]],                                 \
      device uchar *out [[buffer(12)]],                                        \
      uint3 group [[threadgroup_position_in_grid]],                            \
      ushort lane [[thread_index_in_simdgroup]],                               \
      ushort tid [[thread_index_in_threadgroup]]

[[max_total_threads_per_threadgroup(32)]]
kernel void paged_prefill_vector(PP_ARGUMENTS) {
  ulong query = group.x, head = group.y, seq = group.z;
  const device ulong *row = meta + seq * 9;
  if (query >= row[0]) {
    if (!PP_PACKED_QUERIES)
      for (ulong d = lane; d < PP_VALUE_DIM; d += PP_WIDTH)
        pp_store(out, pp_output_base(p, row, seq, head, query) + d,
                 PP_QUERY_DTYPE, 0);
    return;
  }
  float accum[PP_MAX_VALUE / PP_WIDTH];
  for (ushort i = 0; i < PP_MAX_VALUE / PP_WIDTH; ++i)
    accum[i] = 0;
  float maximum = PP_SINK_DTYPE == PP_ABSENT
                      ? -INFINITY
                      : pp_load(sinks, head * p.sink_stride, PP_SINK_DTYPE);
  float denominator = PP_SINK_DTYPE == PP_ABSENT ? 0 : 1;
  ulong qbase = pp_query_base(p, row, seq, head, query);
  ulong kvhead = head / (PP_HEADS / PP_KV_HEADS);
  float ks = PP_HAS_SCALES ? *k_scale : 1;
  float vs = PP_HAS_SCALES ? *v_scale : 1;
  ulong2 bounds = pp_key_bounds(p, meta, row, query, 1);
  for (ulong key = bounds.x; key < bounds.y; ++key) {
    if (!pp_allowed(p, meta, row, query, key))
      continue;
    ulong physical = table[seq * p.table_strides[0] +
                           pp_block_index(key) * p.table_strides[1]];
    if (physical >= p.blocks)
      continue;
    ulong slot = pp_block_offset(key);
    float dot = 0;
    for (ulong d = lane; d < PP_KEY_DIM; d += PP_WIDTH)
      dot = fma(pp_load(q, qbase + d * p.query_strides[3], PP_QUERY_DTYPE),
                pp_load(k, pp_key_offset(p, physical, kvhead, slot, d),
                        PP_CACHE_DTYPE) *
                    ks,
                dot);
    float score = pp_score(p, scalars, meta, row, mask, alibi, seq, head, query,
                           key, simd_sum(dot));
    if (score == -INFINITY)
      continue;
    float next = max(maximum, score);
    float alpha = maximum == -INFINITY ? 0 : exp(maximum - next);
    float weight = exp(score - next);
    denominator = denominator * alpha + weight;
    maximum = next;
    for (ulong d = lane; d < PP_VALUE_DIM; d += PP_WIDTH)
      accum[d / PP_WIDTH] =
          fma(weight,
              pp_load(v, pp_value_offset(p, physical, kvhead, slot, d),
                      PP_CACHE_DTYPE) *
                  vs,
              accum[d / PP_WIDTH] * alpha);
  }
  ulong obase = pp_output_base(p, row, seq, head, query);
  for (ulong d = lane; d < PP_VALUE_DIM; d += PP_WIDTH)
    pp_store(out, obase + d, PP_QUERY_DTYPE,
             denominator > 0 ? accum[d / PP_WIDTH] / denominator : 0);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void paged_prefill_matrix(PP_ARGUMENTS) {
  constexpr ushort SIMDS = 4;
  constexpr ushort KEYS = PP_TILE * SIMDS;
  ushort sg = tid / PP_WIDTH;
  ulong query = ulong(group.x) * PP_TILE, head = group.y, seq = group.z;
  const device ulong *row = meta + seq * 9;
  ulong dv = ((PP_VALUE_DIM + PP_TILE - 1) / PP_TILE) * PP_TILE;
  ulong kvhead = head / (PP_HEADS / PP_KV_HEADS);
  threadgroup float qs[SIMDS * PP_TILE * PP_TILE];
  threadgroup float ks[SIMDS * PP_TILE * PP_TILE];
  threadgroup float scores[PP_TILE * KEYS];
  threadgroup float probabilities[PP_TILE * KEYS];
  threadgroup float values[SIMDS * PP_TILE * PP_TILE];
  threadgroup float output[PP_TILE * PP_MATRIX_VALUE];
  threadgroup float maxima[PP_TILE], denominators[PP_TILE], alphas[PP_TILE];
  threadgroup float *qstage = qs + sg * PP_TILE * PP_TILE;
  threadgroup float *kstage = ks + sg * PP_TILE * PP_TILE;
  threadgroup float *vstage = values + sg * PP_TILE * PP_TILE;
  for (ulong index = tid; index < PP_TILE * dv; index += PP_WIDTH * SIMDS)
    output[index] = 0;
  if (tid < PP_TILE) {
    maxima[tid] = PP_SINK_DTYPE == PP_ABSENT
                      ? -INFINITY
                      : pp_load(sinks, head * p.sink_stride, PP_SINK_DTYPE);
    denominators[tid] = PP_SINK_DTYPE == PP_ABSENT ? 0 : 1;
  }
  float key_scale = PP_HAS_SCALES ? *k_scale : 1;
  float value_scale = PP_HAS_SCALES ? *v_scale : 1;
  ulong2 bounds =
      query < row[0] ? pp_key_bounds(p, meta, row, query, PP_TILE) : ulong2(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ulong key = (bounds.x / KEYS) * KEYS; key < bounds.y; key += KEYS) {
    simdgroup_float8x8 dot = make_filled_simdgroup_matrix<float, 8>(0);
    for (ulong dim = 0; dim < PP_KEY_DIM; dim += PP_TILE) {
      for (ushort index = lane; index < PP_TILE * PP_TILE; index += PP_WIDTH) {
        ulong local = index / PP_TILE, d = dim + index % PP_TILE;
        qstage[index] =
            query + local < row[0] && d < PP_KEY_DIM
                ? pp_load(q,
                          pp_query_base(p, row, seq, head, query + local) +
                              d * p.query_strides[3],
                          PP_QUERY_DTYPE)
                : 0;
        ulong token = key + sg * PP_TILE + local;
        float value = 0;
        if (token < bounds.y && token >= bounds.x && d < PP_KEY_DIM) {
          ulong physical = table[seq * p.table_strides[0] +
                                 pp_block_index(token) * p.table_strides[1]];
          if (physical < p.blocks)
            value = pp_load(k,
                            pp_key_offset(p, physical, kvhead,
                                          pp_block_offset(token), d),
                            PP_CACHE_DTYPE) *
                    key_scale;
        }
        kstage[index] = value;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_float8x8 qm, km;
      simdgroup_load(qm, qstage, PP_TILE);
      simdgroup_load(km, kstage, PP_TILE, ulong2(0), true);
      simdgroup_multiply_accumulate(dot, qm, km, dot);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(dot, scores + sg * PP_TILE, KEYS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort local = sg; local < PP_TILE; local += SIMDS) {
      ulong token = key + lane;
      bool valid = token >= bounds.x && token < bounds.y;
      if (valid)
        valid = table[seq * p.table_strides[0] +
                      pp_block_index(token) * p.table_strides[1]] < p.blocks;
      float score =
          valid ? pp_score(p, scalars, meta, row, mask, alibi, seq, head,
                           query + local, token, scores[local * KEYS + lane])
                : -INFINITY;
      float maximum = max(maxima[local], simd_max(score));
      float alpha =
          maxima[local] == -INFINITY ? 0 : exp(maxima[local] - maximum);
      float probability = score == -INFINITY ? 0 : exp(score - maximum);
      float sum = denominators[local] * alpha + simd_sum(probability);
      probabilities[local * KEYS + lane] = probability;
      if (lane == 0) {
        maxima[local] = maximum;
        denominators[local] = sum;
        alphas[local] = alpha;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ulong index = tid; index < PP_TILE * dv; index += PP_WIDTH * SIMDS)
      output[index] *= alphas[index / dv];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ulong dim = sg * PP_TILE; dim < dv; dim += PP_TILE * SIMDS) {
      simdgroup_float8x8 om;
      simdgroup_load(om, output + dim, dv);
      for (ushort key_tile = 0; key_tile < KEYS; key_tile += PP_TILE) {
        for (ushort index = lane; index < PP_TILE * PP_TILE;
             index += PP_WIDTH) {
          ulong token = key + key_tile + index / PP_TILE,
                d = dim + index % PP_TILE;
          float value = 0;
          if (token < bounds.y && token >= bounds.x && d < PP_VALUE_DIM) {
            ulong physical = table[seq * p.table_strides[0] +
                                   pp_block_index(token) * p.table_strides[1]];
            if (physical < p.blocks)
              value = pp_load(v,
                              pp_value_offset(p, physical, kvhead,
                                              pp_block_offset(token), d),
                              PP_CACHE_DTYPE) *
                      value_scale;
          }
          vstage[index] = value;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 pm, vm;
        simdgroup_load(pm, probabilities + key_tile, KEYS);
        simdgroup_load(vm, vstage, PP_TILE);
        simdgroup_multiply_accumulate(om, pm, vm, om);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
      simdgroup_store(om, output + dim, dv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ulong index = tid; index < PP_TILE * dv; index += PP_WIDTH * SIMDS) {
    ulong local = index / dv, d = index % dv;
    if (d < PP_VALUE_DIM &&
        query + local < (PP_PACKED_QUERIES ? row[0] : p.query_axis)) {
      float result =
          denominators[local] > 0 ? output[index] / denominators[local] : 0;
      pp_store(out, pp_output_base(p, row, seq, head, query + local) + d,
               PP_QUERY_DTYPE, result);
    }
  }
}

template <typename A, typename B>
inline __attribute__((always_inline)) float2 pp_mma(vec<A, 2> a, vec<B, 2> b,
                                                    float2 c) {
  simdgroup_matrix<A, 8, 8> am;
  simdgroup_matrix<B, 8, 8> bm;
  simdgroup_float8x8 cm;
  reinterpret_cast<thread vec<A, 2> &>(am.thread_elements()) = a;
  reinterpret_cast<thread vec<B, 2> &>(bm.thread_elements()) = b;
  reinterpret_cast<thread float2 &>(cm.thread_elements()) = c;
  simdgroup_multiply_accumulate(cm, am, bm, cm);
  return reinterpret_cast<thread float2 &>(cm.thread_elements());
}

template <typename T, ushort D, bool BF16_ALIAS = false>
[[max_total_threads_per_threadgroup(128)]]
kernel void paged_prefill_typed(PP_ARGUMENTS) {
  constexpr ushort SIMDS = 4;
  constexpr ushort KEYS = PP_TILE * SIMDS;
  constexpr ushort FRAGMENTS = D / KEYS;
  ushort sg = tid / PP_WIDTH;
  ushort fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
  ushort fragment_col = (((lane / 4) & 2) * 2) + (lane % 2) * 2;
  ulong query = ulong(group.x) * PP_TILE, head = group.y, seq = group.z;
  const device ulong *row = meta + seq * 9;
  ulong kvhead = head / (PP_HEADS / PP_KV_HEADS);
  threadgroup T qs[PP_TILE * D];
  threadgroup T ks[SIMDS * PP_TILE * PP_TILE];
  threadgroup T values[SIMDS * PP_TILE * PP_TILE];
  threadgroup float scores[PP_TILE * KEYS];
  threadgroup T probabilities[PP_TILE * KEYS];
  threadgroup float maxima[PP_TILE], denominators[PP_TILE], alphas[PP_TILE];
  threadgroup T *kstage = ks + sg * PP_TILE * PP_TILE;
  threadgroup T *vstage = values + sg * PP_TILE * PP_TILE;
  float2 output[FRAGMENTS];
  for (ushort i = 0; i < FRAGMENTS; ++i)
    output[i] = float2(0);
  for (ushort index = tid; index < PP_TILE * D; index += PP_WIDTH * SIMDS) {
    ulong local = index / D, d = index % D;
    qs[index] =
        query + local < row[0] && d < PP_KEY_DIM
            ? T(pp_load(q,
                        pp_query_base(p, row, seq, head, query + local) +
                            d * p.query_strides[3],
                        PP_QUERY_DTYPE))
            : T(0);
  }
  if (tid < PP_TILE) {
    maxima[tid] = PP_SINK_DTYPE == PP_ABSENT
                      ? -INFINITY
                      : pp_load(sinks, head * p.sink_stride, PP_SINK_DTYPE);
    denominators[tid] = PP_SINK_DTYPE == PP_ABSENT ? 0 : 1;
  }
  float key_scale = PP_HAS_SCALES ? *k_scale : 1;
  float value_scale = PP_HAS_SCALES ? *v_scale : 1;
  ulong2 bounds =
      query < row[0] ? pp_key_bounds(p, meta, row, query, PP_TILE) : ulong2(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ulong key = (bounds.x / KEYS) * KEYS; key < bounds.y; key += KEYS) {
    simdgroup_float8x8 dot = make_filled_simdgroup_matrix<float, 8>(0);
#pragma clang loop unroll(full)
    for (ushort dim = 0; dim < D; dim += PP_TILE) {
      for (ushort index = lane; index < PP_TILE * PP_TILE; index += PP_WIDTH) {
        ulong token = key + sg * PP_TILE + index / PP_TILE;
        ulong d = dim + index % PP_TILE;
        T value = T(0);
        if (token < bounds.y && token >= bounds.x && d < PP_KEY_DIM) {
          ulong physical = table[seq * p.table_strides[0] +
                                 pp_block_index(token) * p.table_strides[1]];
          if (physical < p.blocks)
            value = T(pp_load(k,
                              pp_key_offset(p, physical, kvhead,
                                            pp_block_offset(token), d),
                              PP_CACHE_DTYPE) *
                      key_scale);
        }
        kstage[index] = value;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_matrix<T, 8, 8> qm, km;
      simdgroup_load(qm, qs + dim, D);
      simdgroup_load(km, kstage, PP_TILE, ulong2(0), true);
      simdgroup_multiply_accumulate(dot, qm, km, dot);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(dot, scores + sg * PP_TILE, KEYS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort local = sg; local < PP_TILE; local += SIMDS) {
      ulong token = key + lane;
      bool valid = token >= bounds.x && token < bounds.y;
      if (valid)
        valid = table[seq * p.table_strides[0] +
                      pp_block_index(token) * p.table_strides[1]] < p.blocks;
      float score =
          valid ? pp_score(p, scalars, meta, row, mask, alibi, seq, head,
                           query + local, token, scores[local * KEYS + lane])
                : -INFINITY;
      float maximum = max(maxima[local], simd_max(score));
      float alpha =
          maxima[local] == -INFINITY ? 0 : exp(maxima[local] - maximum);
      float probability = score == -INFINITY ? 0 : exp(score - maximum);
      probabilities[local * KEYS + lane] = T(probability);
      float denominator = denominators[local] * alpha + simd_sum(probability);
      if (lane == 0) {
        maxima[local] = maximum;
        denominators[local] = denominator;
        alphas[local] = alpha;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (ushort frag = 0; frag < FRAGMENTS; ++frag) {
      ulong dim = sg * PP_TILE + frag * KEYS;
      output[frag] *= alphas[fragment_row];
#pragma clang loop unroll(full)
      for (ushort key_tile = 0; key_tile < KEYS; key_tile += PP_TILE) {
        for (ushort index = lane; index < PP_TILE * PP_TILE;
             index += PP_WIDTH) {
          ulong token = key + key_tile + index / PP_TILE,
                d = dim + index % PP_TILE;
          T value = T(0);
          if (token < bounds.y && token >= bounds.x && d < PP_VALUE_DIM) {
            ulong physical = table[seq * p.table_strides[0] +
                                   pp_block_index(token) * p.table_strides[1]];
            if (physical < p.blocks)
              value = T(pp_load(v,
                                pp_value_offset(p, physical, kvhead,
                                                pp_block_offset(token), d),
                                PP_CACHE_DTYPE) *
                        value_scale);
          }
          vstage[index] = value;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<T, 8, 8> pm;
        simdgroup_matrix<T, 8, 8> vm;
        simdgroup_load(pm, probabilities + key_tile, KEYS);
        simdgroup_load(vm, vstage, PP_TILE);
        output[frag] =
            pp_mma(reinterpret_cast<thread vec<T, 2> &>(pm.thread_elements()),
                   reinterpret_cast<thread vec<T, 2> &>(vm.thread_elements()),
                   output[frag]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma clang loop unroll(full)
  for (ushort frag = 0; frag < FRAGMENTS; ++frag) {
    ulong d = sg * PP_TILE + frag * KEYS + fragment_col;
    ulong local = fragment_row;
    if (query + local < (PP_PACKED_QUERIES ? row[0] : p.query_axis)) {
      float denominator = denominators[local];
      ulong base = pp_output_base(p, row, seq, head, query + local) + d;
      if (d < PP_VALUE_DIM)
        pp_store(out, base, PP_QUERY_DTYPE,
                 denominator > 0 ? output[frag][0] / denominator : 0);
      if (d + 1 < PP_VALUE_DIM)
        pp_store(out, base + 1, PP_QUERY_DTYPE,
                 denominator > 0 ? output[frag][1] / denominator : 0);
    }
  }
}
template [[host_name("paged_prefill_typed_f16_64")]] kernel void
    paged_prefill_typed<half, 64>(PP_ARGUMENTS);
template [[host_name("paged_prefill_typed_f16_128")]] kernel void
    paged_prefill_typed<half, 128>(PP_ARGUMENTS);
template [[host_name("paged_prefill_typed_f16_256")]] kernel void
    paged_prefill_typed<half, 256>(PP_ARGUMENTS);
template [[host_name("paged_prefill_typed_f32_64")]] kernel void
    paged_prefill_typed<float, 64>(PP_ARGUMENTS);
template [[host_name("paged_prefill_typed_f32_128")]] kernel void
    paged_prefill_typed<float, 128>(PP_ARGUMENTS);
template [[host_name("paged_prefill_typed_f32_256")]] kernel void
    paged_prefill_typed<float, 256>(PP_ARGUMENTS);
template [[host_name("paged_prefill_typed_f32_512")]] kernel void
    paged_prefill_typed<float, 512>(PP_ARGUMENTS);
#if defined(__HAVE_BFLOAT__)
template [[host_name("paged_prefill_typed_bf16_64")]] kernel void
    paged_prefill_typed<bfloat, 64>(PP_ARGUMENTS);
template [[host_name("paged_prefill_typed_bf16_128")]] kernel void
    paged_prefill_typed<bfloat, 128>(PP_ARGUMENTS);
template [[host_name("paged_prefill_typed_bf16_256")]] kernel void
    paged_prefill_typed<bfloat, 256>(PP_ARGUMENTS);
#else
template [[host_name("paged_prefill_typed_bf16_64")]] kernel void
    paged_prefill_typed<float, 64, true>(PP_ARGUMENTS);
template [[host_name("paged_prefill_typed_bf16_128")]] kernel void
    paged_prefill_typed<float, 128, true>(PP_ARGUMENTS);
template [[host_name("paged_prefill_typed_bf16_256")]] kernel void
    paged_prefill_typed<float, 256, true>(PP_ARGUMENTS);
#endif

template <typename T>
inline __attribute__((always_inline))
float2 pp_float_fragment(const threadgroup T *data, ulong stride,
                         bool transpose, ushort lane) {
  ushort row = ((lane / 4) & 4) + ((lane / 2) % 4);
  ushort col = (((lane / 4) & 2) * 2) + (lane % 2) * 2;
  float2 result;
  result[0] = float(data[transpose ? col * stride + row : row * stride + col]);
  result[1] = float(
      data[transpose ? (col + 1) * stride + row : row * stride + col + 1]);
  return result;
}

template <typename T, ushort D, bool BF16_ALIAS = false>
[[max_total_threads_per_threadgroup(128)]]
kernel void paged_prefill_q32(PP_ARGUMENTS) {
  constexpr ushort SIMDS = 4;
  constexpr ushort QUERIES = PP_TILE * SIMDS;
  constexpr ushort KEYS = 16;
  constexpr ushort STRIDE = D + 16 / sizeof(T);
  constexpr ushort KEY_STRIDE = KEYS + 16 / sizeof(T);
  constexpr ushort KV_ELEMENTS =
      D * KEY_STRIDE > KEYS * STRIDE ? D * KEY_STRIDE : KEYS * STRIDE;
  constexpr ushort FRAGMENTS = D / PP_TILE;
  constexpr bool SHARED_QUERY = D <= 128;
  constexpr ushort QUERY_READS = QUERIES * D / (PP_WIDTH * SIMDS);
  constexpr ushort CACHE_READS = KEYS * D / (PP_WIDTH * SIMDS);
  ushort sg = tid / PP_WIDTH;
  ushort fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
  ushort fragment_col = (((lane / 4) & 2) * 2) + (lane % 2) * 2;
  ulong query = ulong(group.x) * QUERIES, head = group.y, seq = group.z;
  const device ulong *row = meta + seq * 9;
  ulong local = sg * PP_TILE + fragment_row;
  ulong kvhead = head / (PP_HEADS / PP_KV_HEADS);
  threadgroup T kvs[KV_ELEMENTS];
  threadgroup uchar valid_keys[KEYS];
  threadgroup T queries[SHARED_QUERY ? QUERIES * STRIDE : 1];
  float2 query_fragments[SHARED_QUERY ? 1 : FRAGMENTS];
  float2 output[FRAGMENTS];
  if constexpr (SHARED_QUERY) {
    ushort query_row = tid / (D / QUERY_READS);
    ushort query_col = (tid % (D / QUERY_READS)) * QUERY_READS;
    ulong qr = query + query_row;
    ulong qbase = pp_query_base(p, row, seq, head, qr);
    if (p.query_strides[3] == 1 && PP_KEY_DIM == D) {
      if (qr < row[0]) {
        const device T *src = ((const device T *)q) + qbase + query_col;
#pragma clang loop unroll(full)
        for (ushort j = 0; j < QUERY_READS; ++j) {
          T value;
          if constexpr (sizeof(T) == 2 && !BF16_ALIAS)
            value = src[j];
          else
            value = T(pp_load(q, qbase + query_col + j, PP_QUERY_DTYPE));
          queries[query_row * STRIDE + query_col + j] = value;
        }
      } else {
#pragma clang loop unroll(full)
        for (ushort j = 0; j < QUERY_READS; ++j)
          queries[query_row * STRIDE + query_col + j] = T(0);
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort j = 0; j < QUERY_READS; ++j) {
        ulong d = query_col + j;
        T value = T(0);
        if (qr < row[0] && d < PP_KEY_DIM) {
          if constexpr (sizeof(T) == 2 && !BF16_ALIAS)
            value = ((const device T *)q)[qbase + d * p.query_strides[3]];
          else
            value =
                T(pp_load(q, qbase + d * p.query_strides[3], PP_QUERY_DTYPE));
        }
        queries[query_row * STRIDE + d] = value;
      }
    }
  } else {
    ulong qbase = pp_query_base(p, row, seq, head, query + local);
#pragma clang loop unroll(full)
    for (ushort frag = 0; frag < FRAGMENTS; ++frag) {
#pragma clang loop unroll(full)
      for (ushort element = 0; element < 2; ++element) {
        ulong d = frag * PP_TILE + fragment_col + element;
        query_fragments[frag][element] =
            query + local < row[0] && d < PP_KEY_DIM
                ? pp_load(q, qbase + d * p.query_strides[3], PP_QUERY_DTYPE)
                : 0;
      }
    }
  }
  ushort cache_row = tid / (D / CACHE_READS);
  ushort cache_col = (tid % (D / CACHE_READS)) * CACHE_READS;
#pragma clang loop unroll(full)
  for (ushort frag = 0; frag < FRAGMENTS; ++frag)
    output[frag] = float2(0);
  float maximum = PP_SINK_DTYPE == PP_ABSENT
                      ? -INFINITY
                      : pp_load(sinks, head * p.sink_stride, PP_SINK_DTYPE);
  float denominator = PP_SINK_DTYPE == PP_ABSENT ? 0 : 1;
  float key_scale = PP_HAS_SCALES ? *k_scale : 1;
  float value_scale = PP_HAS_SCALES ? *v_scale : 1;
  ulong2 bounds =
      query < row[0] ? pp_key_bounds(p, meta, row, query, QUERIES) : ulong2(0);
  for (ulong key = (bounds.x / KEYS) * KEYS; key < bounds.y; key += KEYS) {
    ulong token = key + cache_row;
    bool valid = token < bounds.y && token >= bounds.x;
    ulong physical = valid ? table[seq * p.table_strides[0] +
                                   pp_block_index(token) * p.table_strides[1]]
                           : p.blocks;
    valid = valid && physical < p.blocks;
    if (cache_col == 0)
      valid_keys[cache_row] = valid;
    ulong slot = pp_block_offset(token);
    ulong kbase = physical * p.key_strides[0] + kvhead * p.key_strides[1] +
                  slot * (PP_HND ? p.key_strides[2] : p.key_strides[3]);
    if (PP_HND && p.key_strides[3] == 1 && PP_KEY_DIM == D) {
      if (valid) {
        const device T *src = ((const device T *)k) + kbase + cache_col;
#pragma clang loop unroll(full)
        for (ushort j = 0; j < CACHE_READS; ++j) {
          ushort d = cache_col + j;
          T value;
          if constexpr (sizeof(T) == 2 && !BF16_ALIAS)
            value = T(float(src[j]) * key_scale);
          else
            value = T(pp_load(k, kbase + cache_col + j, PP_CACHE_DTYPE) *
                      key_scale);
          kvs[d * KEY_STRIDE + cache_row] = value;
        }
      } else {
#pragma clang loop unroll(full)
        for (ushort j = 0; j < CACHE_READS; ++j) {
          ushort d = cache_col + j;
          kvs[d * KEY_STRIDE + cache_row] = T(0);
        }
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort j = 0; j < CACHE_READS; ++j) {
        ulong d = cache_col + j;
        T value = T(0);
        if (valid && d < PP_KEY_DIM) {
          ulong offset =
              kbase + (PP_HND ? d * p.key_strides[3]
                              : (d / PP_PACKED_X) * p.key_strides[2] +
                                    (d % PP_PACKED_X) * p.key_strides[4]);
          if constexpr (sizeof(T) == 2 && !BF16_ALIAS)
            value = T(float(((const device T *)k)[offset]) * key_scale);
          else
            value = T(pp_load(k, offset, PP_CACHE_DTYPE) * key_scale);
        }
        kvs[d * KEY_STRIDE + cache_row] = value;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 dots[KEYS / PP_TILE];
#pragma clang loop unroll(full)
    for (ushort kt = 0; kt < KEYS / PP_TILE; ++kt)
      dots[kt] = float2(0);
#pragma clang loop unroll(full)
    for (ushort dim = 0; dim < D; dim += PP_TILE) {
      simdgroup_barrier(mem_flags::mem_none);
      float2 qm;
      if constexpr (SHARED_QUERY)
        qm = pp_float_fragment(queries + sg * PP_TILE * STRIDE + dim, STRIDE,
                               false, lane);
      else
        qm = query_fragments[dim / PP_TILE];
      float2 km[KEYS / PP_TILE];
#pragma clang loop unroll(full)
      for (ushort kt = 0; kt < KEYS / PP_TILE; ++kt)
        km[kt] = pp_float_fragment(kvs + dim * KEY_STRIDE + kt * PP_TILE,
                                   KEY_STRIDE, false, lane);
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (ushort kt = 0; kt < KEYS / PP_TILE; ++kt)
        dots[kt] = pp_mma(qm, km[kt], dots[kt]);
    }
    float tile_maximum = -INFINITY;
    bool plain_tile = false;
    if (PP_MASK_DTYPE == PP_ABSENT && PP_ALIBI_DTYPE == PP_ABSENT &&
        !PP_HAS_SOFTCAP) {
      ulong query_first = row[2] + query;
      ulong query_last = query_first + QUERIES - 1;
      ulong key_first = row[3] + key;
      ulong key_last = key_first + KEYS - 1;
      plain_tile = query + QUERIES <= row[0] && key >= bounds.x &&
                   key + KEYS <= bounds.y &&
                   (!PP_CAUSAL || key_last <= query_first) &&
                   (!PP_WINDOW_LEFT || key_first > query_last ||
                    query_last - key_first < PP_WINDOW_LEFT) &&
                   (!PP_WINDOW_RIGHT || key_last < query_first ||
                    key_last - query_first < PP_WINDOW_RIGHT);
      if (PP_ATTENTION_CHUNK_SIZE)
        plain_tile = plain_tile &&
                     query_first / PP_ATTENTION_CHUNK_SIZE ==
                         query_last / PP_ATTENTION_CHUNK_SIZE &&
                     query_first / PP_ATTENTION_CHUNK_SIZE ==
                         key_first / PP_ATTENTION_CHUNK_SIZE &&
                     query_first / PP_ATTENTION_CHUNK_SIZE ==
                         key_last / PP_ATTENTION_CHUNK_SIZE;
    }
    if (plain_tile) {
#pragma clang loop unroll(full)
      for (ushort kt = 0; kt < KEYS / PP_TILE; ++kt) {
#pragma clang loop unroll(full)
        for (ushort element = 0; element < 2; ++element) {
          bool valid = valid_keys[kt * PP_TILE + fragment_col + element] != 0;
          float score = valid ? dots[kt][element] * scalars[0] : -INFINITY;
          dots[kt][element] = score;
          tile_maximum = max(tile_maximum, score);
        }
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort kt = 0; kt < KEYS / PP_TILE; ++kt) {
#pragma clang loop unroll(full)
        for (ushort element = 0; element < 2; ++element) {
          ulong token = key + kt * PP_TILE + fragment_col + element;
          bool valid = valid_keys[kt * PP_TILE + fragment_col + element] != 0;
          float score =
              valid ? pp_score(p, scalars, meta, row, mask, alibi, seq, head,
                               query + local, token, dots[kt][element])
                    : -INFINITY;
          dots[kt][element] = score;
          tile_maximum = max(tile_maximum, score);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong vbase = physical * p.value_strides[0] + kvhead * p.value_strides[1] +
                  slot * (PP_HND ? p.value_strides[2] : p.value_strides[3]);
    if (PP_HND && p.value_strides[3] == 1 && PP_VALUE_DIM == D) {
      if (valid) {
        const device T *src = ((const device T *)v) + vbase + cache_col;
#pragma clang loop unroll(full)
        for (ushort j = 0; j < CACHE_READS; ++j) {
          ushort d = cache_col + j;
          T value;
          if constexpr (sizeof(T) == 2 && !BF16_ALIAS)
            value = T(float(src[j]) * value_scale);
          else
            value = T(pp_load(v, vbase + cache_col + j, PP_CACHE_DTYPE) *
                      value_scale);
          kvs[cache_row * STRIDE + d] = value;
        }
      } else {
#pragma clang loop unroll(full)
        for (ushort j = 0; j < CACHE_READS; ++j) {
          ushort d = cache_col + j;
          kvs[cache_row * STRIDE + d] = T(0);
        }
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort j = 0; j < CACHE_READS; ++j) {
        ulong d = cache_col + j;
        T value = T(0);
        if (valid && d < PP_VALUE_DIM) {
          ulong offset =
              vbase + d * (PP_HND ? p.value_strides[3] : p.value_strides[2]);
          if constexpr (sizeof(T) == 2 && !BF16_ALIAS)
            value = T(float(((const device T *)v)[offset]) * value_scale);
          else
            value = T(pp_load(v, offset, PP_CACHE_DTYPE) * value_scale);
        }
        kvs[cache_row * STRIDE + d] = value;
      }
    }
    // Lanes differing by 1 or 8 hold the same matrix row.
    tile_maximum = max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(1)));
    tile_maximum = max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(8)));
    float next_maximum = max(maximum, tile_maximum);
    float alpha = maximum == -INFINITY ? 0 : exp(maximum - next_maximum);
    float tile_sum = 0;
#pragma clang loop unroll(full)
    for (ushort kt = 0; kt < KEYS / PP_TILE; ++kt) {
#pragma clang loop unroll(full)
      for (ushort element = 0; element < 2; ++element) {
        float score = dots[kt][element];
        float probability = score == -INFINITY ? 0 : exp(score - next_maximum);
        dots[kt][element] = probability;
        tile_sum += probability;
      }
    }
    tile_sum += simd_shuffle_xor(tile_sum, ushort(1));
    tile_sum += simd_shuffle_xor(tile_sum, ushort(8));
    maximum = next_maximum;
    denominator = denominator * alpha + tile_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (ushort frag = 0; frag < FRAGMENTS; ++frag) {
      output[frag][0] *= alpha;
      output[frag][1] *= alpha;
#pragma clang loop unroll(full)
      for (ushort kt = 0; kt < KEYS / PP_TILE; ++kt) {
        simdgroup_barrier(mem_flags::mem_none);
        float2 vm = pp_float_fragment(
            kvs + kt * PP_TILE * STRIDE + frag * PP_TILE, STRIDE, false, lane);
        simdgroup_barrier(mem_flags::mem_none);
        output[frag] = pp_mma(dots[kt], vm, output[frag]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma clang loop unroll(full)
  for (ushort frag = 0; frag < FRAGMENTS; ++frag) {
    ulong d = frag * PP_TILE + fragment_col;
    if (query + local < (PP_PACKED_QUERIES ? row[0] : p.query_axis)) {
      ulong base = pp_output_base(p, row, seq, head, query + local) + d;
      if (d < PP_VALUE_DIM)
        pp_store(out, base, PP_QUERY_DTYPE,
                 denominator > 0 ? output[frag][0] / denominator : 0);
      if (d + 1 < PP_VALUE_DIM)
        pp_store(out, base + 1, PP_QUERY_DTYPE,
                 denominator > 0 ? output[frag][1] / denominator : 0);
    }
  }
}

template [[host_name("paged_prefill_q32_f16_64")]] kernel void
    paged_prefill_q32<half, 64>(PP_ARGUMENTS);
template [[host_name("paged_prefill_q32_f16_128")]] kernel void
    paged_prefill_q32<half, 128>(PP_ARGUMENTS);
template [[host_name("paged_prefill_q32_f16_256")]] kernel void
    paged_prefill_q32<half, 256>(PP_ARGUMENTS);
template [[host_name("paged_prefill_q32_f32_64")]] kernel void
    paged_prefill_q32<float, 64>(PP_ARGUMENTS);
template [[host_name("paged_prefill_q32_f32_128")]] kernel void
    paged_prefill_q32<float, 128>(PP_ARGUMENTS);
#if defined(__HAVE_BFLOAT__)
template [[host_name("paged_prefill_q32_bf16_64")]] kernel void
    paged_prefill_q32<bfloat, 64>(PP_ARGUMENTS);
template [[host_name("paged_prefill_q32_bf16_128")]] kernel void
    paged_prefill_q32<bfloat, 128>(PP_ARGUMENTS);
template [[host_name("paged_prefill_q32_bf16_256")]] kernel void
    paged_prefill_q32<bfloat, 256>(PP_ARGUMENTS);
#else
template [[host_name("paged_prefill_q32_bf16_64")]] kernel void
    paged_prefill_q32<float, 64, true>(PP_ARGUMENTS);
template [[host_name("paged_prefill_q32_bf16_128")]] kernel void
    paged_prefill_q32<float, 128, true>(PP_ARGUMENTS);
template [[host_name("paged_prefill_q32_bf16_256")]] kernel void
    paged_prefill_q32<float, 256, true>(PP_ARGUMENTS);
#endif

// Q8/K64: mistralrs-quant/src/metal_kernels/flash_attn.metal (ggml-org, MIT).
template <typename T, ushort D>
[[kernel, max_total_threads_per_threadgroup(256)]] void
paged_prefill_direct(PP_ARGUMENTS) {
  constexpr ushort SIMDS = 8;
  constexpr ushort QUERIES = 8;
  constexpr ushort KEYS = PP_TILE * SIMDS;
  constexpr ushort QUERY_STRIDE = D + 16 / sizeof(T);
  constexpr ushort QUERY_READS = QUERIES * D / (PP_WIDTH * SIMDS);
  constexpr ushort QUERY_COLUMN_THREADS = D / QUERY_READS;
  constexpr ushort OUTPUT_STRIDE = PP_TILE * SIMDS;
  constexpr ushort OUTPUT_FRAGMENTS = D / OUTPUT_STRIDE;
  ushort sg = tid / PP_WIDTH;
  ushort fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
  ushort fragment_col = (((lane / 4) & 2) * 2) + (lane % 2) * 2;
  ulong query = ulong(group.x) * QUERIES, head = group.y, seq = group.z;
  const device ulong *row = meta + seq * 9;
  ulong kvhead = head / (PP_HEADS / PP_KV_HEADS);
  threadgroup T queries[QUERIES * QUERY_STRIDE];
  threadgroup float scores[QUERIES * KEYS];
  threadgroup float alphas[QUERIES], denominators[QUERIES];
  threadgroup uchar valid_keys[KEYS];
  float2 output[OUTPUT_FRAGMENTS];
#pragma clang loop unroll(full)
  for (ushort i = 0; i < OUTPUT_FRAGMENTS; ++i)
    output[i] = float2(0);
  ushort query_local = tid / QUERY_COLUMN_THREADS;
  ushort query_column = (tid % QUERY_COLUMN_THREADS) * QUERY_READS;
  bool query_valid = query + query_local < row[0];
  ulong query_base = pp_query_base(p, row, seq, head, query + query_local);
  if (sizeof(T) == 2 && p.query_strides[3] == 1) {
#pragma clang loop unroll(full)
    for (ushort j = 0; j < QUERY_READS; ++j) {
      ushort d = query_column + j;
      queries[query_local * QUERY_STRIDE + d] =
          query_valid && d < PP_KEY_DIM ? ((const device T *)q)[query_base + d]
                                        : T(0);
    }
  } else {
#pragma clang loop unroll(full)
    for (ushort j = 0; j < QUERY_READS; ++j) {
      ushort d = query_column + j;
      queries[query_local * QUERY_STRIDE + d] =
          query_valid && d < PP_KEY_DIM
              ? T(pp_load(q, query_base + d * p.query_strides[3],
                          PP_QUERY_DTYPE))
              : T(0);
    }
  }
  float maximum = PP_SINK_DTYPE == PP_ABSENT
                      ? -INFINITY
                      : pp_load(sinks, head * p.sink_stride, PP_SINK_DTYPE);
  float denominator = PP_SINK_DTYPE == PP_ABSENT ? 0 : 1;
  if (lane == 0)
    denominators[sg] = denominator;
  ulong2 bounds =
      query < row[0] ? pp_key_bounds(p, meta, row, query, QUERIES) : ulong2(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ulong key = (bounds.x / KEYS) * KEYS; key < bounds.y; key += KEYS) {
    ulong2 tokens(key + sg * PP_TILE + fragment_col,
                  key + sg * PP_TILE + fragment_col + 1);
    bool2 valid(tokens[0] >= bounds.x && tokens[0] < bounds.y,
                tokens[1] >= bounds.x && tokens[1] < bounds.y);
    ulong2 key_bases(0);
#pragma clang loop unroll(full)
    for (ushort element = 0; element < 2; ++element) {
      ulong physical =
          valid[element]
              ? table[seq * p.table_strides[0] +
                      pp_block_index(tokens[element]) * p.table_strides[1]]
              : p.blocks;
      valid[element] = valid[element] && physical < p.blocks;
      if (valid[element])
        key_bases[element] = physical * p.key_strides[0] +
                             kvhead * p.key_strides[1] +
                             pp_block_offset(tokens[element]) *
                                 (PP_HND ? p.key_strides[2] : p.key_strides[3]);
      if (fragment_row == 0)
        valid_keys[sg * PP_TILE + fragment_col + element] = valid[element];
    }
    float2 dot(0);
    ulong first_key = key + sg * PP_TILE;
    bool collective_keys = sizeof(T) == 2 && PP_HND && PP_KEY_DIM == D &&
                           PP_BLOCK_SIZE % PP_TILE == 0 &&
                           p.key_strides[3] == 1 && p.key_strides[2] == D &&
                           first_key >= bounds.x &&
                           first_key + PP_TILE <= bounds.y;
    ulong key_physical =
        collective_keys ? table[seq * p.table_strides[0] +
                                pp_block_index(first_key) * p.table_strides[1]]
                        : p.blocks;
    collective_keys = collective_keys && key_physical < p.blocks;
    if (collective_keys) {
      ulong base = key_physical * p.key_strides[0] + kvhead * p.key_strides[1] +
                   pp_block_offset(first_key) * p.key_strides[2];
#pragma clang loop unroll_count(4)
      for (ushort dim = 0; dim < D; dim += PP_TILE * 2) {
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_matrix<T, PP_TILE, PP_TILE> loaded;
        simdgroup_load(loaded, queries + dim, QUERY_STRIDE);
        vec<T, 2> qm0 =
            reinterpret_cast<thread vec<T, 2> &>(loaded.thread_elements());
        simdgroup_load(loaded, ((const device T *)k) + base + dim, D, ulong2(0),
                       true);
        vec<T, 2> km0 =
            reinterpret_cast<thread vec<T, 2> &>(loaded.thread_elements());
        simdgroup_load(loaded, queries + dim + PP_TILE, QUERY_STRIDE);
        vec<T, 2> qm1 =
            reinterpret_cast<thread vec<T, 2> &>(loaded.thread_elements());
        simdgroup_load(loaded, ((const device T *)k) + base + dim + PP_TILE, D,
                       ulong2(0), true);
        vec<T, 2> km1 =
            reinterpret_cast<thread vec<T, 2> &>(loaded.thread_elements());
        simdgroup_barrier(mem_flags::mem_none);
        dot = pp_mma(qm0, km0, dot);
        dot = pp_mma(qm1, km1, dot);
      }
    } else {
#pragma clang loop unroll_count(4)
      for (ushort dim = 0; dim < D; dim += PP_TILE * 2) {
        simdgroup_barrier(mem_flags::mem_none);
        ulong d0 = dim + fragment_row;
        ulong d1 = d0 + PP_TILE;
        ulong offset0 = PP_HND ? d0 * p.key_strides[3]
                               : (d0 / PP_PACKED_X) * p.key_strides[2] +
                                     (d0 % PP_PACKED_X) * p.key_strides[4];
        ulong offset1 = PP_HND ? d1 * p.key_strides[3]
                               : (d1 / PP_PACKED_X) * p.key_strides[2] +
                                     (d1 % PP_PACKED_X) * p.key_strides[4];
        vec<T, 2> km0(T(0)), km1(T(0));
#pragma clang loop unroll(full)
        for (ushort element = 0; element < 2; ++element) {
          if (valid[element] && d0 < PP_KEY_DIM) {
            if constexpr (sizeof(T) == 2)
              km0[element] =
                  ((const device T *)k)[key_bases[element] + offset0];
            else
              km0[element] =
                  T(pp_load(k, key_bases[element] + offset0, PP_CACHE_DTYPE));
          }
          if (valid[element] && d1 < PP_KEY_DIM) {
            if constexpr (sizeof(T) == 2)
              km1[element] =
                  ((const device T *)k)[key_bases[element] + offset1];
            else
              km1[element] =
                  T(pp_load(k, key_bases[element] + offset1, PP_CACHE_DTYPE));
          }
        }
        ulong query_offset = fragment_row * QUERY_STRIDE + dim + fragment_col;
        vec<T, 2> qm0(queries[query_offset], queries[query_offset + 1]);
        vec<T, 2> qm1(queries[query_offset + PP_TILE],
                      queries[query_offset + PP_TILE + 1]);
        simdgroup_barrier(mem_flags::mem_none);
        dot = pp_mma(qm0, km0, dot);
        dot = pp_mma(qm1, km1, dot);
      }
    }
    ulong score_offset = fragment_row * KEYS + sg * PP_TILE + fragment_col;
    scores[score_offset] = dot[0];
    scores[score_offset + 1] = dot[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool plain_tile = false;
    if (PP_MASK_DTYPE == PP_ABSENT && PP_ALIBI_DTYPE == PP_ABSENT &&
        !PP_HAS_SOFTCAP) {
      ulong query_first = row[2] + query;
      ulong query_last = query_first + QUERIES - 1;
      ulong key_first = row[3] + key;
      ulong key_last = key_first + KEYS - 1;
      plain_tile = query + QUERIES <= row[0] && key >= bounds.x &&
                   key + KEYS <= bounds.y &&
                   (!PP_CAUSAL || key_last <= query_first) &&
                   (!PP_WINDOW_LEFT || key_first > query_last ||
                    query_last - key_first < PP_WINDOW_LEFT) &&
                   (!PP_WINDOW_RIGHT || key_last < query_first ||
                    key_last - query_first < PP_WINDOW_RIGHT);
      if (PP_ATTENTION_CHUNK_SIZE)
        plain_tile = plain_tile &&
                     query_first / PP_ATTENTION_CHUNK_SIZE ==
                         query_last / PP_ATTENTION_CHUNK_SIZE &&
                     query_first / PP_ATTENTION_CHUNK_SIZE ==
                         key_first / PP_ATTENTION_CHUNK_SIZE &&
                     query_first / PP_ATTENTION_CHUNK_SIZE ==
                         key_last / PP_ATTENTION_CHUNK_SIZE;
    }
    float2 score;
    if (plain_tile) {
#pragma clang loop unroll(full)
      for (ushort element = 0; element < 2; ++element) {
        ushort key_local = lane * 2 + element;
        score[element] = valid_keys[key_local]
                             ? scores[sg * KEYS + key_local] * scalars[0]
                             : -INFINITY;
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort element = 0; element < 2; ++element) {
        ushort key_local = lane * 2 + element;
        score[element] = valid_keys[key_local]
                             ? pp_score(p, scalars, meta, row, mask, alibi, seq,
                                        head, query + sg, key + key_local,
                                        scores[sg * KEYS + key_local])
                             : -INFINITY;
      }
    }
    float next_maximum = max(maximum, simd_max(max(score[0], score[1])));
    float alpha = maximum == -INFINITY ? 0 : exp(maximum - next_maximum);
    float2 probability;
#pragma clang loop unroll(full)
    for (ushort element = 0; element < 2; ++element) {
      probability[element] =
          score[element] == -INFINITY ? 0 : exp(score[element] - next_maximum);
      scores[sg * KEYS + lane * 2 + element] = probability[element];
    }
    denominator =
        denominator * alpha + simd_sum(probability[0] + probability[1]);
    maximum = next_maximum;
    if (lane == 0) {
      denominators[sg] = denominator;
      alphas[sg] = alpha;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (ushort frag = 0; frag < OUTPUT_FRAGMENTS; ++frag)
      output[frag] *= alphas[fragment_row];
#pragma clang loop unroll(full)
    for (ushort key_tile = 0; key_tile < KEYS; key_tile += PP_TILE) {
      ulong token = key + key_tile + fragment_row;
      bool value_valid = valid_keys[key_tile + fragment_row] != 0;
      ulong physical = value_valid
                           ? table[seq * p.table_strides[0] +
                                   pp_block_index(token) * p.table_strides[1]]
                           : p.blocks;
      ulong value_base =
          value_valid
              ? physical * p.value_strides[0] + kvhead * p.value_strides[1] +
                    pp_block_offset(token) *
                        (PP_HND ? p.value_strides[2] : p.value_strides[3])
              : 0;
      ulong first_value = key + key_tile;
      bool collective_values =
          sizeof(T) == 2 && PP_HND && PP_VALUE_DIM == D &&
          PP_BLOCK_SIZE % PP_TILE == 0 && p.value_strides[3] == 1 &&
          p.value_strides[2] == D && first_value >= bounds.x &&
          first_value + PP_TILE <= bounds.y;
      ulong value_physical =
          collective_values
              ? table[seq * p.table_strides[0] +
                      pp_block_index(first_value) * p.table_strides[1]]
              : p.blocks;
      collective_values = collective_values && value_physical < p.blocks;
      if (collective_values) {
        ulong base = value_physical * p.value_strides[0] +
                     kvhead * p.value_strides[1] +
                     pp_block_offset(first_value) * p.value_strides[2];
        simdgroup_float8x8 probability_matrix;
        simdgroup_load(probability_matrix, scores + key_tile, KEYS);
        float2 pm = reinterpret_cast<thread float2 &>(
            probability_matrix.thread_elements());
#pragma clang loop unroll(full)
        for (ushort frag = 0; frag < OUTPUT_FRAGMENTS; ++frag) {
          ulong d = sg * PP_TILE + frag * OUTPUT_STRIDE;
          simdgroup_matrix<T, PP_TILE, PP_TILE> value_matrix;
          simdgroup_load(value_matrix, ((const device T *)v) + base + d, D);
          vec<T, 2> vm = reinterpret_cast<thread vec<T, 2> &>(
              value_matrix.thread_elements());
          output[frag] = pp_mma(pm, vm, output[frag]);
        }
      } else {
        ulong probability_offset =
            fragment_row * KEYS + key_tile + fragment_col;
        float2 pm(scores[probability_offset], scores[probability_offset + 1]);
#pragma clang loop unroll(full)
        for (ushort frag = 0; frag < OUTPUT_FRAGMENTS; ++frag) {
          ulong d = sg * PP_TILE + frag * OUTPUT_STRIDE + fragment_col;
          vec<T, 2> vm(T(0));
#pragma clang loop unroll(full)
          for (ushort element = 0; element < 2; ++element) {
            if (value_valid && d + element < PP_VALUE_DIM) {
              ulong offset =
                  value_base + (d + element) * (PP_HND ? p.value_strides[3]
                                                       : p.value_strides[2]);
              if constexpr (sizeof(T) == 2)
                vm[element] = ((const device T *)v)[offset];
              else
                vm[element] = T(pp_load(v, offset, PP_CACHE_DTYPE));
            }
          }
          output[frag] = pp_mma(pm, vm, output[frag]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float sum = denominators[fragment_row];
#pragma clang loop unroll(full)
  for (ushort frag = 0; frag < OUTPUT_FRAGMENTS; ++frag) {
    ulong d = sg * PP_TILE + frag * OUTPUT_STRIDE + fragment_col;
    if (query + fragment_row < (PP_PACKED_QUERIES ? row[0] : p.query_axis)) {
      ulong base = pp_output_base(p, row, seq, head, query + fragment_row) + d;
      if (d < PP_VALUE_DIM)
        pp_store(out, base, PP_QUERY_DTYPE,
                 sum > 0 ? output[frag][0] / sum : 0);
      if (d + 1 < PP_VALUE_DIM)
        pp_store(out, base + 1, PP_QUERY_DTYPE,
                 sum > 0 ? output[frag][1] / sum : 0);
    }
  }
}

template [[host_name("paged_prefill_direct_f16_512")]] kernel void
    paged_prefill_direct<half, 512>(PP_ARGUMENTS);
#if defined(__HAVE_BFLOAT__)
template [[host_name("paged_prefill_direct_bf16_512")]] kernel void
    paged_prefill_direct<bfloat, 512>(PP_ARGUMENTS);
#else
template [[host_name("paged_prefill_direct_bf16_512")]] kernel void
    paged_prefill_direct<float, 512>(PP_ARGUMENTS);
#endif
