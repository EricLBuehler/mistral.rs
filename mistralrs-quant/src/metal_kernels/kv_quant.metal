#include <metal_stdlib>

using namespace metal;

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*********************************/
/************* 8-bit *************/
/*********************************/

#define QK8_0 32

struct BlockQ8_0 {
    half d;
    char qs[QK8_0];
};

static_assert(sizeof(BlockQ8_0) == sizeof(half) + QK8_0, "wrong q8_0 block size/padding");


void quantize_row_q8_0_ref(const device float * x, device BlockQ8_0 * y) {
    const int qk = QK8_0;

    float amax = 0.0f; // absolute max

    for (int j = 0; j < qk; j++) {
        const float v = x[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = half(d);

    for (int j = 0; j < qk; ++j) {
        const float x0 = x[j]*id;

        y->qs[j] = round(x0);
    }
}

template <typename T>
[[kernel]] void quantize_8bit_kv(
    const device T* k [[buffer(0)]],
    const device T* v [[buffer(1)]],
    device BlockQ8_0* out_k [[buffer(2)]],
    device BlockQ8_0* out_v [[buffer(3)]],
    uint tid [[ thread_position_in_grid ]]
) {
    const device T* k_block = k + tid*QK8_0;
    const device T* v_block = v + tid*QK8_0;
    const device BlockQ8_0* o_k_block = k + tid;
    const device BlockQ8_0* o_v_block = v + tid;

    quantize_row_q8_0_ref(k_block, o_k_block);
    quantize_row_q8_0_ref(v_block, o_v_block);
}

/*********************************/
/************* 4-bit *************/
/*********************************/

#define QK4_0 32

struct BlockQ4_0 {
    half d;
    char qs[QK4_0 / 2];
};

static_assert(sizeof(BlockQ4_0) == sizeof(half) + QK4_0 / 2, "wrong q4_0 block size/padding");


void quantize_row_q4_0_ref(const device float * x, device BlockQ4_0 * y) {
    const int qk = QK4_0;

    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < qk; j++) {
        const float v = x[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d  = max / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = half(d);

    for (int j = 0; j < qk/2; ++j) {
        const float x0 = x[0    + j]*id;
        const float x1 = x[qk/2 + j]*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

template <typename T>
[[kernel]] void quantize_4bit_kv(
    const device T* k [[buffer(0)]],
    const device T* v [[buffer(1)]],
    device BlockQ4_0* out_k [[buffer(2)]],
    device BlockQ4_0* out_v [[buffer(3)]],
    uint tid [[ thread_position_in_grid ]]
) {
    const device T* k_block = k + tid*QK4_0;
    const device T* v_block = v + tid*QK4_0;
    const device BlockQ4_0* o_k_block = k + tid;
    const device BlockQ4_0* o_v_block = v + tid;

    quantize_row_q4_0_ref(k_block, o_k_block);
    quantize_row_q4_0_ref(v_block, o_v_block);
}