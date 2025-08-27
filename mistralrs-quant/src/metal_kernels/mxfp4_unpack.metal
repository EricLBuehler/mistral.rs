#include <metal_stdlib>
#include "bf16.metal"

using namespace metal;

template<typename T>
[[kernel]]void unpack_mxfp4(
        const device uint8_t *blocks      [[buffer(0)]],
        const device uint8_t *scales_u8   [[buffer(1)]],   // original uint8 exponent bias
              device T       *out         [[buffer(2)]],
        device const uint &rows_total, device const uint &B,
        uint gid                          [[thread_position_in_grid]])
{
    const uint elements   = rows_total * B;   // total bytes

    if (gid >= elements) return;

    /* -------- look-up table in constant memory -------- */
    constexpr float LUT[16] = {
        +0.0f, +0.5f, +1.0f, +1.5f,
        +2.0f, +3.0f, +4.0f, +6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f,
        -2.0f, -3.0f, -4.0f, -6.0f
    };

    uint   row     = gid / B;
    uint   col     = gid % B;
    uint8_t packed = blocks[gid];

    /* de-pack 4-bit nibbles */
    uint idx_lo =  packed        & 0x0Fu;
    uint idx_hi = (packed >> 4u) & 0x0Fu;

    /* broadcast scale:  2.0^(scale_int32)  with bias-127 removal */
    uint8_t  s_u8        = scales_u8[row];
    int      s_int       = int(s_u8) - 127;       // subtract bias
    float    scale_factor = exp2(float(s_int));   // fast power-of-two

    /* convert, scale, and write (2 values per byte) */
    uint out_base = gid * 2u;   // each byte expands to two outputs
    out[out_base]     = T(scale_factor * LUT[idx_lo]);
    out[out_base + 1] = T(scale_factor * LUT[idx_hi]);
}

#define INSTANTIATE_UNPACK_MXFP4(T) \
    template [[host_name("unpack_mxfp4_" #T)]] \
    kernel void unpack_mxfp4<T>( \
        const device uint8_t *blocks      [[buffer(0)]], \
        const device uint8_t *scales_u8   [[buffer(1)]], \
              device T       *out         [[buffer(2)]], \
        device const uint &rows_total, device const uint &B, \
        uint gid                          [[thread_position_in_grid]]);

INSTANTIATE_UNPACK_MXFP4(float)
INSTANTIATE_UNPACK_MXFP4(half)
INSTANTIATE_UNPACK_MXFP4(bfloat)
