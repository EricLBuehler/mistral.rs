// Routed MoE tiled matmul kernel for Metal.
// Based on candle's kernel_mul_mm_id but reads routing table from
// DEVICE MEMORY instead of threadgroup memory — no 32KB limit.
//
// Rust builds the routing table (rowids) as a device buffer.
// This kernel is identical to candle's kernel_mul_mm_id_impl
// except rowids is `device const` not `threadgroup`.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── Block types ──
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    half d;
    half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

// ── Tiling constants (same as candle/llama.cpp) ──
#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4
#define THREAD_MAT_N 2
#define THREAD_PER_ROW 2
#define THREAD_PER_COL 4
#define SG_MAT_SIZE 64
#define SG_MAT_ROW 8
#define QK_NL 16

// ── Scale/min extraction (from candle) ──
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                           uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// ── Q4_K dequantization (from candle, exact copy) ──
void dequantize_q4_K(device const block_q4_K *xb, short il, thread half4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il<2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

// ── Routed MoE kernel ──
// Identical to candle's kernel_mul_mm_id_impl but:
// - rowids is `device const` (pre-built by Rust) not `threadgroup` (built in-kernel)
// - expert routing is pre-computed: rowids[i] = (expert_slot, token_idx)
//
// Grid: (ceil(n_tokens_for_expert/32), ceil(n_out/64), n_experts)
// Threads: 128
kernel void moe_mm_q4k_routed(
    device const  uchar   * all_weights    [[buffer(0)]],  // [n_experts, n_out, n_in] Q4_K
    device const  float   * all_inputs     [[buffer(1)]],  // [total_input_rows, n_in] f32
    device        float   * all_outputs    [[buffer(2)]],  // output
    device const  uint    * route_counts   [[buffer(3)]],  // [n_experts] count per expert
    device const  ushort2 * route_rowids   [[buffer(4)]],  // [total_pairs] (slot, token) per pair
    device const  uint    * route_offsets  [[buffer(5)]],  // [n_experts] cumulative offset
    constant      int64_t & ne00           [[buffer(6)]],  // n_in (K)
    constant      int64_t & ne0            [[buffer(7)]],  // n_out (N)
    constant      uint64_t& nb01           [[buffer(8)]],  // bytes per weight row
    constant      int64_t & ne11           [[buffer(9)]],  // input rows per token group (1 or topk)
    constant      uint64_t& nb10           [[buffer(10)]], // bytes per input element (4 for f32)
    constant      uint64_t& nb11           [[buffer(11)]], // bytes per input row
    constant      uint64_t& nb12           [[buffer(12)]], // bytes per input token group
    constant      int64_t & ne0ne1         [[buffer(13)]], // ne0 * output_stride
    threadgroup   uchar   * shared_memory  [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiitg [[thread_index_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint expert_id = tgpig.z;
    const uint ne1 = route_counts[expert_id];  // tokens for this expert
    if (ne1 == 0) return;

    const uint route_off = route_offsets[expert_id];

    // rowids for this expert — device memory, no 32KB limit
    device const ushort2 * rowids = route_rowids + route_off;

    // Weight pointer for this expert
    const uint64_t nb02 = (uint64_t)ne0 * nb01;
    device const uchar * src0 = all_weights + expert_id * nb02;

    threadgroup half  * sa = (threadgroup half  *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;  // output tile row
    const uint r1 = tgpig.x;  // token tile col

    if (r1 * BLOCK_SIZE_N >= ne1) return;

    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    short thread_row = ((short)tiitg / THREAD_PER_ROW) < n_rows ? ((short)tiitg / THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg / THREAD_PER_COL) < n_cols ? ((short)tiitg / THREAD_PER_COL) : n_cols - 1;

    simdgroup_half8x8     ma[4];
    simdgroup_float8x8    mb[2];
    simdgroup_float8x8    c_res[8];
    for (short i = 0; i < 8; i++) {
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);
    constexpr short nl = QK_NL;
    ushort offset1 = il / nl;

    device const ushort2 & id = rowids[r1 * BLOCK_SIZE_N + thread_col];

    device const block_q4_K * x = (device const block_q4_K *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01) + offset1;
    device const float      * y = (device const float      *)((device const uchar *)all_inputs
        + nb12 * id[1]
        + nb11 * (id[0] % ne11)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        half4x4 temp_a;
        dequantize_q4_K(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8)
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8)
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL))
            = *((device float2x4 *)y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1) / nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            for (int i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    // Write results — scatter to correct output positions via rowids
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shared_memory)
                                      + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i % 4) + 8 * BLOCK_SIZE_M * (i / 4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * C = all_outputs + (BLOCK_SIZE_M * r0);
        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                device const ushort2 & jid = rowids[r1 * BLOCK_SIZE_N + j];
                int joff = jid[0] * ne0 + jid[1] * ne0ne1;
                for (int i = 0; i < n_rows; i++) {
                    *(C + i + joff) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}
