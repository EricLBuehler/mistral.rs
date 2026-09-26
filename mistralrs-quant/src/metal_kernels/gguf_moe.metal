// Bounded indexed (routed MoE) GEMV over stacked GGUF expert weights.
//
// The weight buffer holds `[experts, n, k]` GGML-quantized expert matrices in
// the packed block layout produced by the GGUF converter. For every
// (token, expert) pair the routed MoE needs one dot product between each of
// the selected expert's `n` rows and the token's activation row. These kernels
// compute exactly those dot products and nothing else: only the selected
// experts' blocks are ever read, and the stack is never dequantized or
// materialized densely.
//
// Dispatch: one threadgroup (exactly one simdgroup, 32 threads) per output
// element `out[pair, row]`. Every lane dot-products a strided subset of the
// row's quantized sub-blocks against the activation row in F32, and the lane
// results are combined with `simd_sum`. The grid is a single flattened
// dimension (`pair = gid / n`, `row = gid % n`) because Metal requires the
// thread-position attributes to be uniformly scalar or uniformly vector.
//
// Dequantization formulas match the GGML reference decoders (and candle's CPU
// `GgmlType` implementations) exactly; only the F32 accumulation order differs
// from the CPU paths.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint QK_K = 256;

// GGML f16 fields sit at byte offsets that are not 2-byte aligned for every
// block (e.g. Q8_0's 34-byte stride), so decode them from bytes explicitly
// instead of performing an aligned uint16 load.
static inline float half_from_bytes(device const uint8_t *p) {
    const uint16_t bits = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    return as_type<half>(bits);
}

// ---- Q2_K -----------------------------------------------------------------
//
// Block (84 bytes): scales[16] u8, qs[64] u8, d f16, dmin f16.
// 16 sub-blocks of 16 elements per 256-element block. Element `e` of a block:
//   is    = e / 16                       (scale byte index)
//   o     = e / 128                      (qs half selection)
//   s     = (e % 128) / 32               (2-bit field selector)
//   l     = e % 32                       (byte within the qs half)
//   q     = (qs[o * 32 + l] >> (s * 2)) & 3
//   value = d * (scales[is] & 0xF) * q - dmin * (scales[is] >> 4)
[[kernel]] void indexed_moe_gemv_q2_k(
    device const uint8_t *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const uint32_t *ids [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int32_t &n [[buffer(4)]],
    constant int32_t &k [[buffer(5)]],
    constant int32_t &topk [[buffer(6)]],
    constant int32_t &x_per_pair [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint tgsz [[threads_per_threadgroup]]) {

    const uint pair = gid / (uint)n;
    const uint row = gid % (uint)n;
    const uint expert = ids[pair];
    const uint token = x_per_pair != 0 ? pair : pair / (uint)topk;
    const uint nb = (uint)k / QK_K;
    const uint sub_blocks = nb * 16;

    device const uint8_t *row_ptr =
        weights + ((uint64_t)expert * (uint)n + row) * ((uint64_t)nb * 84);
    device const float *x_row = x + (uint64_t)token * (uint)k;

    float acc = 0.0f;
    for (uint si = tiisg; si < sub_blocks; si += tgsz) {
        const uint blk = si / 16;
        const uint sub = si % 16;
        device const uint8_t *b = row_ptr + (uint64_t)blk * 84;
        const float d = half_from_bytes(b + 80);
        const float dmin = half_from_bytes(b + 82);
        const uint8_t scb = b[sub];
        const float dsc = d * (float)(scb & 0xF);
        const float dm = dmin * (float)(scb >> 4);
        device const uint8_t *qs = b + 16;
        const uint o = sub >> 3;
        device const float *y = x_row + (uint64_t)blk * QK_K + sub * 16;
        float dsum = 0.0f;
        float msum = 0.0f;
        for (uint i = 0; i < 16; ++i) {
            const uint r = ((sub & 7) << 4) + i;
            const uint shift = (r >> 5) << 1;
            const int qi = (int)((qs[o * 32 + (r & 31)] >> shift) & 3);
            dsum += y[i] * (float)qi;
            msum += y[i];
        }
        acc += dsc * dsum - dm * msum;
    }

    out[(uint64_t)pair * (uint)n + row] = simd_sum(acc);
}

// ---- Q4_K -----------------------------------------------------------------
//
// Block (144 bytes): d f16, dmin f16, scales[12] u8, qs[128] u8.
// 8 sub-blocks of 32 elements per 256-element block. Element `e` of a block:
//   sub   = e / 32
//   c     = (sub / 2) * 32               (qs byte chunk)
//   nib   = sub odd ? qs[c + e%32] >> 4 : qs[c + e%32] & 0xF
//   value = d * sc - dmin * m
// with the 6-bit scale pair `(sc, m)` decoded by the reference
// `get_scale_min_k4(sub, scales)` scheme.
[[kernel]] void indexed_moe_gemv_q4_k(
    device const uint8_t *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const uint32_t *ids [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int32_t &n [[buffer(4)]],
    constant int32_t &k [[buffer(5)]],
    constant int32_t &topk [[buffer(6)]],
    constant int32_t &x_per_pair [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint tgsz [[threads_per_threadgroup]]) {

    const uint pair = gid / (uint)n;
    const uint row = gid % (uint)n;
    const uint expert = ids[pair];
    const uint token = x_per_pair != 0 ? pair : pair / (uint)topk;
    const uint nb = (uint)k / QK_K;
    const uint sub_blocks = nb * 8;

    device const uint8_t *row_ptr =
        weights + ((uint64_t)expert * (uint)n + row) * ((uint64_t)nb * 144);
    device const float *x_row = x + (uint64_t)token * (uint)k;

    float acc = 0.0f;
    for (uint si = tiisg; si < sub_blocks; si += tgsz) {
        const uint blk = si >> 3;
        const uint sub = si & 7;
        device const uint8_t *b = row_ptr + (uint64_t)blk * 144;
        const float d = half_from_bytes(b);
        const float dmin = half_from_bytes(b + 2);
        device const uint8_t *scales = b + 4;
        device const uint8_t *qs = b + 16;
        uint8_t sc, m;
        if (sub < 4) {
            sc = scales[sub] & 63;
            m = scales[sub + 4] & 63;
        } else {
            sc = (scales[sub + 4] & 0xF) | ((scales[sub - 4] >> 6) << 4);
            m = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
        }
        const float dsc = d * (float)sc;
        const float dm = dmin * (float)m;
        const uint c = (sub >> 1) * 32;
        const uint nib_shift = (sub & 1) << 2;
        device const float *y = x_row + (uint64_t)blk * QK_K + sub * 32;
        float dsum = 0.0f;
        float msum = 0.0f;
        for (uint i = 0; i < 32; ++i) {
            const int nib = (int)((qs[c + i] >> nib_shift) & 0xF);
            dsum += y[i] * (float)nib;
            msum += y[i];
        }
        acc += dsc * dsum - dm * msum;
    }

    out[(uint64_t)pair * (uint)n + row] = simd_sum(acc);
}

// ---- Q6_K -----------------------------------------------------------------
//
// Block (210 bytes): ql[128] u8, qh[64] u8, scales[16] i8, d f16.
// 16 sub-blocks of 16 elements per 256-element block. Element `e` of a block:
//   n     = e / 128                      (128-element chunk)
//   g     = (e % 128) / 32               (quant group 0..3)
//   l     = e % 32
//   q     = ((ql nibble) | (2 qh bits << 4)) - 32
//   value = d * scales[n * 8 + g * 2 + (l / 16)] * q
[[kernel]] void indexed_moe_gemv_q6_k(
    device const uint8_t *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const uint32_t *ids [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int32_t &n [[buffer(4)]],
    constant int32_t &k [[buffer(5)]],
    constant int32_t &topk [[buffer(6)]],
    constant int32_t &x_per_pair [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint tgsz [[threads_per_threadgroup]]) {

    const uint pair = gid / (uint)n;
    const uint row = gid % (uint)n;
    const uint expert = ids[pair];
    const uint token = x_per_pair != 0 ? pair : pair / (uint)topk;
    const uint nb = (uint)k / QK_K;
    const uint sub_blocks = nb * 16;

    device const uint8_t *row_ptr =
        weights + ((uint64_t)expert * (uint)n + row) * ((uint64_t)nb * 210);
    device const float *x_row = x + (uint64_t)token * (uint)k;

    float acc = 0.0f;
    for (uint si = tiisg; si < sub_blocks; si += tgsz) {
        const uint blk = si / 16;
        const uint sub = si % 16;
        device const uint8_t *b = row_ptr + (uint64_t)blk * 210;
        const float d = half_from_bytes(b + 208);
        device const int8_t *scales = (device const int8_t *)(b + 192);
        device const uint8_t *ql = b;
        device const uint8_t *qh = b + 128;
        const uint r = sub << 4;
        const uint nn = r >> 7;
        const uint rem = r & 127;
        const uint g = rem >> 5;
        const uint l0 = rem & 31;
        const uint ql_idx = nn * 64 + ((g & 1) << 5) + l0;
        const uint qh_idx = nn * 32 + l0;
        const int8_t scale = scales[nn * 8 + (l0 >> 4) + g * 2];
        device const float *y = x_row + (uint64_t)blk * QK_K + r;
        float dsum = 0.0f;
        for (uint i = 0; i < 16; ++i) {
            // Element l = l0 + i: the ql low/high nibble and the 2 qh bits both
            // come from byte `l` of this chunk's ql/qh slices.
            const uint8_t qlb = ql[ql_idx + i];
            const int nib = (g < 2) ? (int)(qlb & 0xF) : (int)(qlb >> 4);
            const int bits = (int)((qh[qh_idx + i] >> (g << 1)) & 3);
            const int q = ((nib | (bits << 4)) - 32);
            dsum += y[i] * (float)q;
        }
        acc += d * (float)scale * dsum;
    }

    out[(uint64_t)pair * (uint)n + row] = simd_sum(acc);
}

// ---- Q8_0 -----------------------------------------------------------------
//
// Block (34 bytes): d f16, qs[32] i8. 32 elements per block.
[[kernel]] void indexed_moe_gemv_q8_0(
    device const uint8_t *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const uint32_t *ids [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int32_t &n [[buffer(4)]],
    constant int32_t &k [[buffer(5)]],
    constant int32_t &topk [[buffer(6)]],
    constant int32_t &x_per_pair [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint tgsz [[threads_per_threadgroup]]) {

    const uint pair = gid / (uint)n;
    const uint row = gid % (uint)n;
    const uint expert = ids[pair];
    const uint token = x_per_pair != 0 ? pair : pair / (uint)topk;
    const uint nb = (uint)k / 32;

    device const uint8_t *row_ptr =
        weights + ((uint64_t)expert * (uint)n + row) * ((uint64_t)nb * 34);
    device const float *x_row = x + (uint64_t)token * (uint)k;

    float acc = 0.0f;
    for (uint blk = tiisg; blk < nb; blk += tgsz) {
        device const uint8_t *b = row_ptr + (uint64_t)blk * 34;
        const float d = half_from_bytes(b);
        device const int8_t *qs = (device const int8_t *)(b + 2);
        device const float *y = x_row + (uint64_t)blk * 32;
        float dsum = 0.0f;
        for (uint i = 0; i < 32; ++i) {
            dsum += y[i] * (float)qs[i];
        }
        acc += d * dsum;
    }

    out[(uint64_t)pair * (uint)n + row] = simd_sum(acc);
}

// ---- Q4_0 -----------------------------------------------------------------
//
// Block (18 bytes): d f16, qs[16] u8 of nibbles. 32 elements per block.
// Element `i < 16` is the low nibble of byte `i`, element `i + 16` is the high
// nibble of byte `i` (candle/GGML `block_q4_0` convention). The value
// `d * (q - 8)` is accumulated as `d * (sum(q*y) - 8 * sum(y))`, matching the
// K-quant kernels' separated sum/correction form.
[[kernel]] void indexed_moe_gemv_q4_0(
    device const uint8_t *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const uint32_t *ids [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int32_t &n [[buffer(4)]],
    constant int32_t &k [[buffer(5)]],
    constant int32_t &topk [[buffer(6)]],
    constant int32_t &x_per_pair [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint tgsz [[threads_per_threadgroup]]) {

    const uint pair = gid / (uint)n;
    const uint row = gid % (uint)n;
    const uint expert = ids[pair];
    const uint token = x_per_pair != 0 ? pair : pair / (uint)topk;
    const uint nb = (uint)k / 32;

    device const uint8_t *row_ptr =
        weights + ((uint64_t)expert * (uint)n + row) * ((uint64_t)nb * 18);
    device const float *x_row = x + (uint64_t)token * (uint)k;

    float acc = 0.0f;
    for (uint blk = tiisg; blk < nb; blk += tgsz) {
        device const uint8_t *b = row_ptr + (uint64_t)blk * 18;
        const float d = half_from_bytes(b);
        device const uint8_t *qs = b + 2;
        device const float *y = x_row + (uint64_t)blk * 32;
        float dsum = 0.0f;
        float msum = 0.0f;
        for (uint i = 0; i < 16; ++i) {
            const float y0 = y[i];
            const float y1 = y[i + 16];
            dsum += y0 * (float)(qs[i] & 0xF) + y1 * (float)(qs[i] >> 4);
            msum += y0 + y1;
        }
        acc += d * (dsum - 8.0f * msum);
    }

    out[(uint64_t)pair * (uint)n + row] = simd_sum(acc);
}
