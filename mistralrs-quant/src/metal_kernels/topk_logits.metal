// Two-stage top-k over a large vocab + softmax stats, packed for the
// sampler's CPU-side post-processing.

#include <metal_stdlib>
using namespace metal;

// Simdgroup-wide max with corresponding index. Returns max value via the
// builtin, and writes the matching index into `out_idx` for all lanes.
inline float simd_max_with_idx(float v, int idx, thread int &out_idx) {
    float m = simd_max(v);
    int winner = (v == m) ? idx : INT_MIN;
    out_idx = simd_max(winner);
    return m;
}

kernel void copy_f32(
        device const float *src [[buffer(0)]],
        device       float *dst [[buffer(1)]],
        constant int       &n   [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (int(gid) < n) {
        dst[gid] = src[gid];
    }
}

kernel void apply_sparse_penalties_f32(
        device float       *logits          [[buffer(0)]],
        device const uint  *token_ids       [[buffer(1)]],
        device const float *counts          [[buffer(2)]],
        constant int       &n               [[buffer(3)]],
        constant int       &n_tokens        [[buffer(4)]],
        constant float     &frequency_penalty [[buffer(5)]],
        constant float     &presence_penalty  [[buffer(6)]],
        constant float     &repetition_penalty[[buffer(7)]],
        uint gid [[thread_position_in_grid]]) {
    if (int(gid) >= n_tokens) return;
    uint token_id = token_ids[gid];
    if (int(token_id) >= n) return;
    float count = counts[gid];
    if (count <= 0.0f) return;
    float v = logits[token_id];
    v -= count * frequency_penalty + presence_penalty;
    if (repetition_penalty != 1.0f) {
        v = (v > 0.0f) ? (v / repetition_penalty) : (v * repetition_penalty);
    }
    logits[token_id] = v;
}

kernel void topk_logits_stage1_f32(
        device const float *input          [[buffer(0)]],
        device float       *block_values   [[buffer(1)]],
        device uint        *block_indices  [[buffer(2)]],
        device float       *block_maxes    [[buffer(3)]],
        device float       *block_sums     [[buffer(4)]],
        constant int       &ncols          [[buffer(5)]],
        constant int       &k              [[buffer(6)]],
        constant int       &chunk_size     [[buffer(7)]],
        constant float     &inv_temperature[[buffer(8)]],
        threadgroup char   *s_used         [[threadgroup(0)]],
        uint tid     [[thread_index_in_threadgroup]],
        uint chunk   [[threadgroup_position_in_grid]],
        uint tg_size [[threads_per_threadgroup]],
        uint simd_lid[[thread_index_in_simdgroup]],
        uint simd_gid[[simdgroup_index_in_threadgroup]]) {
    const int start = int(chunk) * chunk_size;
    const int end = min(start + chunk_size, ncols);
    const int width = max(0, end - start);

    for (uint i = tid; i < uint(chunk_size); i += tg_size) {
        s_used[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float warp_maxes[32];
    threadgroup int   warp_indices[32];
    const uint num_warps = (tg_size + 31) / 32;

    for (int ki = 0; ki < k; ++ki) {
        float local_max = -INFINITY;
        int local_idx = -1;
        for (int local = int(tid); local < width; local += int(tg_size)) {
            float c = input[start + local];
            if (!s_used[local] && !isnan(c) && c > local_max) {
                local_max = c;
                local_idx = start + local;
            }
        }

        int warp_max_idx;
        float warp_max = simd_max_with_idx(local_max, local_idx, warp_max_idx);
        if (simd_lid == 0) {
            warp_maxes[simd_gid] = warp_max;
            warp_indices[simd_gid] = warp_max_idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0) {
            float v = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
            int idx = (tid < num_warps) ? warp_indices[tid] : INT_MIN;
            int fin_idx;
            float fin_max = simd_max_with_idx(v, idx, fin_idx);
            if (simd_lid == 0) {
                block_values[chunk * k + ki] = fin_max;
                block_indices[chunk * k + ki] = (fin_idx >= 0) ? uint(fin_idx) : 0u;
                if (fin_idx >= start && fin_idx < end) {
                    s_used[fin_idx - start] = 1;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Local softmax stats: max + sumexp(x - max), with temperature applied to
    // x before the max so the consumer sees a fully temperature-scaled softmax.
    const float block_max = (width > 0)
        ? block_values[chunk * k] * inv_temperature
        : -INFINITY;
    float local_sum = 0.0f;
    if (block_max != -INFINITY) {
        for (int local = int(tid); local < width; local += int(tg_size)) {
            float c = input[start + local];
            if (!isnan(c)) {
                local_sum += metal::exp(c * inv_temperature - block_max);
            }
        }
    }

    // Threadgroup sum reduction (one float per thread).
    threadgroup float sums[32];
    float warp_sum = simd_sum(local_sum);
    if (simd_lid == 0) {
        sums[simd_gid] = warp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        float v = (tid < num_warps) ? sums[tid] : 0.0f;
        float total = simd_sum(v);
        if (simd_lid == 0) {
            block_maxes[chunk] = block_max;
            block_sums[chunk] = total;
        }
    }
}

kernel void topk_logits_stage2_packed_f32(
        device const float *block_values  [[buffer(0)]],
        device const uint  *block_indices [[buffer(1)]],
        device const float *block_maxes   [[buffer(2)]],
        device const float *block_sums    [[buffer(3)]],
        device float       *packed_out    [[buffer(4)]],
        constant int       &nblocks       [[buffer(5)]],
        constant int       &k             [[buffer(6)]],
        threadgroup char   *s_used        [[threadgroup(0)]],
        uint tid     [[thread_index_in_threadgroup]],
        uint tg_size [[threads_per_threadgroup]],
        uint simd_lid[[thread_index_in_simdgroup]],
        uint simd_gid[[simdgroup_index_in_threadgroup]]) {
    const int n_candidates = nblocks * k;
    const uint num_warps = (tg_size + 31) / 32;

    for (int i = int(tid); i < n_candidates; i += int(tg_size)) {
        s_used[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global softmax max + denominator.
    float local_global_max = -INFINITY;
    for (int b = int(tid); b < nblocks; b += int(tg_size)) {
        local_global_max = metal::max(local_global_max, block_maxes[b]);
    }
    int dummy;
    float warp_g_max = simd_max_with_idx(local_global_max, 0, dummy);

    threadgroup float warp_maxes[32];
    if (simd_lid == 0) {
        warp_maxes[simd_gid] = warp_g_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float s_global_max;
    if (simd_gid == 0) {
        float v = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
        float fin_max = simd_max(v);
        if (simd_lid == 0) {
            s_global_max = fin_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_denom = 0.0f;
    if (s_global_max != -INFINITY) {
        for (int b = int(tid); b < nblocks; b += int(tg_size)) {
            local_denom += block_sums[b] * metal::exp(block_maxes[b] - s_global_max);
        }
    }
    threadgroup float sums[32];
    float warp_sum = simd_sum(local_denom);
    if (simd_lid == 0) {
        sums[simd_gid] = warp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        float v = (tid < num_warps) ? sums[tid] : 0.0f;
        float denom = simd_sum(v);
        if (simd_lid == 0) {
            packed_out[2 * k] = denom;
            packed_out[2 * k + 1] = s_global_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup int merge_warp_indices[32];
    for (int ki = 0; ki < k; ++ki) {
        float local_max = -INFINITY;
        int local_pos = -1;
        for (int pos = int(tid); pos < n_candidates; pos += int(tg_size)) {
            float c = block_values[pos];
            if (!s_used[pos] && !isnan(c) && c > local_max) {
                local_max = c;
                local_pos = pos;
            }
        }
        int warp_max_pos;
        float warp_max = simd_max_with_idx(local_max, local_pos, warp_max_pos);
        if (simd_lid == 0) {
            warp_maxes[simd_gid] = warp_max;
            merge_warp_indices[simd_gid] = warp_max_pos;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_gid == 0) {
            float v = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
            int idx = (tid < num_warps) ? merge_warp_indices[tid] : INT_MIN;
            int fin_pos;
            float fin_max = simd_max_with_idx(v, idx, fin_pos);
            if (simd_lid == 0) {
                packed_out[ki] = fin_max;
                packed_out[k + ki] =
                    (fin_pos >= 0) ? float(block_indices[fin_pos]) : 0.0f;
                if (fin_pos >= 0) {
                    s_used[fin_pos] = 1;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
