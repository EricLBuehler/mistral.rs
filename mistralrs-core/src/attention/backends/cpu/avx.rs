//! x86 micro-kernels for the tiled CPU attention paths: AVX512 (with AVX2 fallback)
//! dot products, fused multiply-accumulate rows, and a vectorized Cephes exp for the
//! online-softmax hot loop. All accumulate in f32; runtime feature detection at entry.

use std::sync::OnceLock;

pub(super) fn avx512() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| !force_avx2() && is_x86_feature_detected!("avx512f"))
}

#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let len = a.len();
    let main = len & !15;
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i < main {
        let x = _mm512_loadu_ps(a.as_ptr().add(i));
        let y = _mm512_loadu_ps(b.as_ptr().add(i));
        acc = _mm512_fmadd_ps(x, y, acc);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    for j in main..len {
        sum += a[j] * b[j];
    }
    sum
}

#[target_feature(enable = "avx512f")]
unsafe fn dot4_f32_avx512(q: &[f32], k0: &[f32], k1: &[f32], k2: &[f32], k3: &[f32]) -> [f32; 4] {
    use core::arch::x86_64::*;
    let len = q.len();
    let main = len & !15;
    let mut a0 = _mm512_setzero_ps();
    let mut a1 = _mm512_setzero_ps();
    let mut a2 = _mm512_setzero_ps();
    let mut a3 = _mm512_setzero_ps();
    let mut i = 0;
    while i < main {
        let x = _mm512_loadu_ps(q.as_ptr().add(i));
        a0 = _mm512_fmadd_ps(x, _mm512_loadu_ps(k0.as_ptr().add(i)), a0);
        a1 = _mm512_fmadd_ps(x, _mm512_loadu_ps(k1.as_ptr().add(i)), a1);
        a2 = _mm512_fmadd_ps(x, _mm512_loadu_ps(k2.as_ptr().add(i)), a2);
        a3 = _mm512_fmadd_ps(x, _mm512_loadu_ps(k3.as_ptr().add(i)), a3);
        i += 16;
    }
    let mut out = [
        _mm512_reduce_add_ps(a0),
        _mm512_reduce_add_ps(a1),
        _mm512_reduce_add_ps(a2),
        _mm512_reduce_add_ps(a3),
    ];
    for j in main..len {
        let qj = q[j];
        out[0] += qj * k0[j];
        out[1] += qj * k1[j];
        out[2] += qj * k2[j];
        out[3] += qj * k3[j];
    }
    out
}

#[target_feature(enable = "avx512f")]
unsafe fn mad_f32_avx512(acc: &mut [f32], values: &[f32], scale: f32) {
    use core::arch::x86_64::*;
    let len = acc.len();
    let main = len & !15;
    let s = _mm512_set1_ps(scale);
    let mut i = 0;
    while i < main {
        let v = _mm512_loadu_ps(values.as_ptr().add(i));
        let a = _mm512_loadu_ps(acc.as_ptr().add(i));
        _mm512_storeu_ps(acc.as_mut_ptr().add(i), _mm512_fmadd_ps(v, s, a));
        i += 16;
    }
    for j in main..len {
        acc[j] += values[j] * scale;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn scale_f32_avx512(xs: &mut [f32], scale: f32) {
    use core::arch::x86_64::*;
    let len = xs.len();
    let main = len & !15;
    let s = _mm512_set1_ps(scale);
    let mut i = 0;
    while i < main {
        let v = _mm512_loadu_ps(xs.as_ptr().add(i));
        _mm512_storeu_ps(xs.as_mut_ptr().add(i), _mm512_mul_ps(v, s));
        i += 16;
    }
    for x in &mut xs[main..] {
        *x *= scale;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn max_f32_avx512(xs: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let len = xs.len();
    let main = len & !15;
    let mut acc = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i < main {
        acc = _mm512_max_ps(acc, _mm512_loadu_ps(xs.as_ptr().add(i)));
        i += 16;
    }
    let mut m = _mm512_reduce_max_ps(acc);
    for &x in &xs[main..] {
        m = m.max(x);
    }
    m
}

// Cephes exp on 16 lanes; same polynomial as the scalar fast_exp.
#[allow(clippy::excessive_precision)]
#[target_feature(enable = "avx512f")]
unsafe fn softmax_row_f32_avx512(row: &mut [f32], m: f32) -> f32 {
    use core::arch::x86_64::*;
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const C0: f32 = 0.693_359_375;
    const C1: f32 = -2.121_944_4e-4;
    let len = row.len();
    let main = len & !15;
    let mv = _mm512_set1_ps(m);
    let log2e = _mm512_set1_ps(LOG2E);
    let c0 = _mm512_set1_ps(C0);
    let c1 = _mm512_set1_ps(C1);
    let p0 = _mm512_set1_ps(0.5);
    let p1 = _mm512_set1_ps(0.166_665_46);
    let p2 = _mm512_set1_ps(0.041_665_795);
    let p3 = _mm512_set1_ps(0.008_333_45);
    let p4 = _mm512_set1_ps(0.001_392_034_5);
    let one = _mm512_set1_ps(1.0);
    let mut sumv = _mm512_setzero_ps();
    let mut i = 0;
    while i < main {
        let x = _mm512_sub_ps(_mm512_loadu_ps(row.as_ptr().add(i)), mv);
        let x = _mm512_max_ps(
            _mm512_set1_ps(-87.0),
            _mm512_min_ps(_mm512_set1_ps(87.0), x),
        );
        let z = _mm512_roundscale_ps::<0>(_mm512_mul_ps(x, log2e));
        let r = _mm512_fnmadd_ps(z, c1, _mm512_fnmadd_ps(z, c0, x));
        let r2 = _mm512_mul_ps(r, r);
        let p = _mm512_fmadd_ps(
            r,
            _mm512_fmadd_ps(r, _mm512_fmadd_ps(r, _mm512_fmadd_ps(r, p4, p3), p2), p1),
            p0,
        );
        let p = _mm512_add_ps(r, _mm512_mul_ps(r2, p));
        let zi = _mm512_cvtps_epi32(z);
        let e = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_add_epi32(zi, _mm512_set1_epi32(127)),
            23,
        ));
        let v = _mm512_mul_ps(e, _mm512_add_ps(one, p));
        _mm512_storeu_ps(row.as_mut_ptr().add(i), v);
        sumv = _mm512_add_ps(sumv, v);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(sumv);
    for v in &mut row[main..] {
        let e = super::elem::fast_exp(*v - m);
        *v = e;
        sum += e;
    }
    sum
}

#[inline(always)]
pub(super) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    if avx512() {
        return unsafe { dot_f32_avx512(a, b) };
    }
    if avx2_f16c() {
        return unsafe { dot_f32_avx2(a, b) };
    }
    let mut sum = 0f32;
    for (x, y) in a.iter().zip(b) {
        sum += x * y;
    }
    sum
}

#[inline(always)]
pub(super) fn dot4_f32(q: &[f32], k0: &[f32], k1: &[f32], k2: &[f32], k3: &[f32]) -> [f32; 4] {
    if avx512() {
        return unsafe { dot4_f32_avx512(q, k0, k1, k2, k3) };
    }
    [
        dot_f32(q, k0),
        dot_f32(q, k1),
        dot_f32(q, k2),
        dot_f32(q, k3),
    ]
}

#[inline(always)]
pub(super) fn mad_f32(acc: &mut [f32], values: &[f32], scale: f32) {
    if avx512() {
        return unsafe { mad_f32_avx512(acc, values, scale) };
    }
    for (a, v) in acc.iter_mut().zip(values) {
        *a += v * scale;
    }
}

#[inline(always)]
pub(super) fn scale_f32(xs: &mut [f32], scale: f32) {
    if avx512() {
        return unsafe { scale_f32_avx512(xs, scale) };
    }
    for v in xs.iter_mut() {
        *v *= scale;
    }
}

#[inline(always)]
pub(super) fn max_f32(xs: &[f32]) -> f32 {
    if avx512() {
        return unsafe { max_f32_avx512(xs) };
    }
    xs.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

#[inline(always)]
pub(super) fn softmax_row_f32(row: &mut [f32], m: f32) -> f32 {
    if avx512() {
        return unsafe { softmax_row_f32_avx512(row, m) };
    }
    if avx2_f16c() {
        return unsafe { softmax_row_f32_avx2(row, m) };
    }
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = super::elem::fast_exp(*v - m);
        sum += *v;
    }
    sum
}

// f16 K/V kernels: hardware vcvtph2ps widening, f32 accumulation. Halves the KV
// stream against f32 at one convert per 16 lanes.
#[target_feature(enable = "avx512f")]
unsafe fn dot_f16_avx512(a: &[half::f16], b: &[half::f16]) -> f32 {
    use core::arch::x86_64::*;
    let len = a.len();
    let main = len & !15;
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i < main {
        let x = _mm512_cvtph_ps(_mm256_loadu_si256(a.as_ptr().add(i) as *const _));
        let y = _mm512_cvtph_ps(_mm256_loadu_si256(b.as_ptr().add(i) as *const _));
        acc = _mm512_fmadd_ps(x, y, acc);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    for j in main..len {
        sum += a[j].to_f32() * b[j].to_f32();
    }
    sum
}

#[target_feature(enable = "avx512f")]
unsafe fn dot4_f16_avx512(
    q: &[half::f16],
    k0: &[half::f16],
    k1: &[half::f16],
    k2: &[half::f16],
    k3: &[half::f16],
) -> [f32; 4] {
    use core::arch::x86_64::*;
    let len = q.len();
    let main = len & !15;
    let mut a0 = _mm512_setzero_ps();
    let mut a1 = _mm512_setzero_ps();
    let mut a2 = _mm512_setzero_ps();
    let mut a3 = _mm512_setzero_ps();
    let mut i = 0;
    while i < main {
        let x = _mm512_cvtph_ps(_mm256_loadu_si256(q.as_ptr().add(i) as *const _));
        a0 = _mm512_fmadd_ps(
            x,
            _mm512_cvtph_ps(_mm256_loadu_si256(k0.as_ptr().add(i) as *const _)),
            a0,
        );
        a1 = _mm512_fmadd_ps(
            x,
            _mm512_cvtph_ps(_mm256_loadu_si256(k1.as_ptr().add(i) as *const _)),
            a1,
        );
        a2 = _mm512_fmadd_ps(
            x,
            _mm512_cvtph_ps(_mm256_loadu_si256(k2.as_ptr().add(i) as *const _)),
            a2,
        );
        a3 = _mm512_fmadd_ps(
            x,
            _mm512_cvtph_ps(_mm256_loadu_si256(k3.as_ptr().add(i) as *const _)),
            a3,
        );
        i += 16;
    }
    let mut out = [
        _mm512_reduce_add_ps(a0),
        _mm512_reduce_add_ps(a1),
        _mm512_reduce_add_ps(a2),
        _mm512_reduce_add_ps(a3),
    ];
    for j in main..len {
        let qj = q[j].to_f32();
        out[0] += qj * k0[j].to_f32();
        out[1] += qj * k1[j].to_f32();
        out[2] += qj * k2[j].to_f32();
        out[3] += qj * k3[j].to_f32();
    }
    out
}

#[target_feature(enable = "avx512f")]
unsafe fn mad_f16_avx512(acc: &mut [f32], values: &[half::f16], scale: f32) {
    use core::arch::x86_64::*;
    let len = acc.len();
    let main = len & !15;
    let s = _mm512_set1_ps(scale);
    let mut i = 0;
    while i < main {
        let v = _mm512_cvtph_ps(_mm256_loadu_si256(values.as_ptr().add(i) as *const _));
        let a = _mm512_loadu_ps(acc.as_ptr().add(i));
        _mm512_storeu_ps(acc.as_mut_ptr().add(i), _mm512_fmadd_ps(v, s, a));
        i += 16;
    }
    for j in main..len {
        acc[j] += values[j].to_f32() * scale;
    }
}

#[inline(always)]
pub(super) fn dot_f16(a: &[half::f16], b: &[half::f16]) -> Option<f32> {
    if avx512() {
        return Some(unsafe { dot_f16_avx512(a, b) });
    }
    if avx2_f16c() {
        let r = unsafe { dot4_f16_avx2(a, b, b, b, b) };
        return Some(r[0]);
    }
    None
}

#[inline(always)]
pub(super) fn dot4_f16(
    q: &[half::f16],
    k0: &[half::f16],
    k1: &[half::f16],
    k2: &[half::f16],
    k3: &[half::f16],
) -> Option<[f32; 4]> {
    if avx512() {
        return Some(unsafe { dot4_f16_avx512(q, k0, k1, k2, k3) });
    }
    if avx2_f16c() {
        return Some(unsafe { dot4_f16_avx2(q, k0, k1, k2, k3) });
    }
    None
}

#[inline(always)]
pub(super) fn mad_f16(acc: &mut [f32], values: &[half::f16], scale: f32) -> bool {
    if avx512() {
        unsafe { mad_f16_avx512(acc, values, scale) };
        return true;
    }
    if avx2_f16c() {
        unsafe { mad_f16_avx2(acc, values, scale) };
        return true;
    }
    false
}

// P.V accumulation over a kv tile with register-held accumulators: v is converted once
// per position and up to 4 grouped q-rows' accumulators (4 x 8 zmm = full register file)
// never touch memory inside the tile.
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn pv_tile_f16_avx512(
    vkq: *mut f32,
    vkq_stride: usize,
    group: usize,
    dv: usize,
    v_data: &[half::f16],
    v_row_of: &dyn Fn(usize) -> usize,
    bs: usize,
    be: usize,
    p_tile: &[f32],
    tile_stride: usize,
) {
    use core::arch::x86_64::*;
    debug_assert!(dv <= 128 && dv.is_multiple_of(16) && group <= 4);
    let nv = dv / 16;
    let mut acc = [[_mm512_setzero_ps(); 8]; 4];
    for (j, accj) in acc.iter_mut().enumerate().take(group) {
        for (x, a) in accj.iter_mut().enumerate().take(nv) {
            *a = _mm512_loadu_ps(vkq.add(j * vkq_stride + x * 16));
        }
    }
    for kv in bs..be {
        let base = v_row_of(kv);
        let vp = v_data.as_ptr().add(base);
        let mut v16 = [_mm512_setzero_ps(); 8];
        for (x, v) in v16.iter_mut().enumerate().take(nv) {
            *v = _mm512_cvtph_ps(_mm256_loadu_si256(vp.add(x * 16) as *const _));
        }
        for (j, accj) in acc.iter_mut().enumerate().take(group) {
            let p = _mm512_set1_ps(p_tile[j * tile_stride + (kv - bs)]);
            for (x, a) in accj.iter_mut().enumerate().take(nv) {
                *a = _mm512_fmadd_ps(v16[x], p, *a);
            }
        }
    }
    for (j, accj) in acc.iter().enumerate().take(group) {
        for (x, a) in accj.iter().enumerate().take(nv) {
            _mm512_storeu_ps(vkq.add(j * vkq_stride + x * 16), *a);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn pv_tile_f16(
    vkq: *mut f32,
    vkq_stride: usize,
    group: usize,
    dv: usize,
    v_data: &[half::f16],
    v_row_of: &dyn Fn(usize) -> usize,
    bs: usize,
    be: usize,
    p_tile: &[f32],
    tile_stride: usize,
) -> bool {
    if avx512() && dv <= 128 && dv.is_multiple_of(16) && group <= 4 {
        unsafe {
            pv_tile_f16_avx512(
                vkq,
                vkq_stride,
                group,
                dv,
                v_data,
                v_row_of,
                bs,
                be,
                p_tile,
                tile_stride,
            )
        };
        return true;
    }
    false
}

#[target_feature(enable = "avx512f")]
unsafe fn expand_f16_avx512(dst: &mut [f32], src: &[half::f16]) {
    use core::arch::x86_64::*;
    let len = dst.len();
    let main = len & !15;
    let mut i = 0;
    while i < main {
        let v = _mm512_cvtph_ps(_mm256_loadu_si256(src.as_ptr().add(i) as *const _));
        _mm512_storeu_ps(dst.as_mut_ptr().add(i), v);
        i += 16;
    }
    for j in main..len {
        dst[j] = src[j].to_f32();
    }
}

#[inline(always)]
pub(super) fn expand_f16(dst: &mut [f32], src: &[half::f16]) -> bool {
    if avx512() {
        unsafe { expand_f16_avx512(dst, src) };
        return true;
    }
    if avx2_f16c() {
        unsafe { expand_f16_avx2(dst, src) };
        return true;
    }
    false
}

// 16x16 f32 transpose network (unpack + 4x4 shuffle + 128-bit permute stages).
#[target_feature(enable = "avx512f")]
unsafe fn transpose16(v: &mut [core::arch::x86_64::__m512; 16]) {
    use core::arch::x86_64::*;
    let mut t = [_mm512_setzero_ps(); 16];
    for i in 0..8 {
        t[2 * i] = _mm512_unpacklo_ps(v[2 * i], v[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_ps(v[2 * i], v[2 * i + 1]);
    }
    for i in 0..4 {
        v[4 * i] = _mm512_shuffle_ps::<0x44>(t[4 * i], t[4 * i + 2]);
        v[4 * i + 1] = _mm512_shuffle_ps::<0xEE>(t[4 * i], t[4 * i + 2]);
        v[4 * i + 2] = _mm512_shuffle_ps::<0x44>(t[4 * i + 1], t[4 * i + 3]);
        v[4 * i + 3] = _mm512_shuffle_ps::<0xEE>(t[4 * i + 1], t[4 * i + 3]);
    }
    for i in 0..2 {
        for j in 0..4 {
            t[8 * i + j] = _mm512_shuffle_f32x4::<0x88>(v[8 * i + j], v[8 * i + j + 4]);
            t[8 * i + j + 4] = _mm512_shuffle_f32x4::<0xDD>(v[8 * i + j], v[8 * i + j + 4]);
        }
    }
    for j in 0..8 {
        v[j] = _mm512_shuffle_f32x4::<0x88>(t[j], t[j + 8]);
        v[j + 8] = _mm512_shuffle_f32x4::<0xDD>(t[j], t[j + 8]);
    }
}

// Expand an f16 K tile into transposed f32 scratch: kt[d][kv] from rows k[kv][d].
#[target_feature(enable = "avx512f")]
unsafe fn expand_transpose_f16_avx512(
    k_data: &[half::f16],
    row_of: &dyn Fn(usize) -> usize,
    bs: usize,
    bn: usize,
    d: usize,
    kt: &mut [f32],
    kt_stride: usize,
) {
    use core::arch::x86_64::*;
    let mut kv0 = 0;
    while kv0 < bn {
        let kvn = (bn - kv0).min(16);
        let mut dd = 0;
        while dd < d {
            let mut v = [_mm512_setzero_ps(); 16];
            for (r, vr) in v.iter_mut().enumerate().take(kvn) {
                let base = row_of(bs + kv0 + r) + dd;
                *vr = _mm512_cvtph_ps(_mm256_loadu_si256(k_data.as_ptr().add(base) as *const _));
            }
            transpose16(&mut v);
            for (j, vr) in v.iter().enumerate().take((d - dd).min(16)) {
                _mm512_storeu_ps(kt.as_mut_ptr().add((dd + j) * kt_stride + kv0), *vr);
            }
            dd += 16;
        }
        kv0 += 16;
    }
}

// GEMM-structured tile scoring: 16 kv scores per fma against broadcast q elements,
// no horizontal reductions, K streamed once per tile for all rows.
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_block_avx512(
    q_rows: &[f32],
    nq: usize,
    d: usize,
    kt: &[f32],
    kt_stride: usize,
    bn: usize,
    scale: f32,
    s_tile: &mut [f32],
    tile_stride: usize,
) {
    use core::arch::x86_64::*;
    let sv = _mm512_set1_ps(scale);
    let mut kv0 = 0;
    while kv0 < bn {
        // 8 q rows x 16 kv accumulators
        let mut acc = [_mm512_setzero_ps(); 8];
        for dd in 0..d {
            let kvec = _mm512_loadu_ps(kt.as_ptr().add(dd * kt_stride + kv0));
            for (j, a) in acc.iter_mut().enumerate().take(nq) {
                let qb = _mm512_set1_ps(*q_rows.get_unchecked(j * d + dd));
                *a = _mm512_fmadd_ps(qb, kvec, *a);
            }
        }
        for (j, a) in acc.iter().enumerate().take(nq) {
            _mm512_storeu_ps(
                s_tile.as_mut_ptr().add(j * tile_stride + kv0),
                _mm512_mul_ps(*a, sv),
            );
        }
        kv0 += 16;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn score_block_f16(
    q_rows: &[f32],
    nq: usize,
    d: usize,
    k_data: &[half::f16],
    row_of: &dyn Fn(usize) -> usize,
    bs: usize,
    bn: usize,
    kt: &mut [f32],
    scale: f32,
    s_tile: &mut [f32],
    tile_stride: usize,
) -> bool {
    if !avx512() || nq > 8 || !d.is_multiple_of(16) {
        return false;
    }
    let kt_stride = (bn + 15) & !15;
    unsafe {
        expand_transpose_f16_avx512(k_data, row_of, bs, bn, d, kt, kt_stride);
        score_block_avx512(q_rows, nq, d, kt, kt_stride, bn, scale, s_tile, tile_stride);
    }
    true
}

// 256-bit tier for CPUs without AVX512 (everything Haswell/Zen1+): same ops, ymm
// width, f16 through F16C. pv_tile stays 512-only; the generic mad loop covers it.
pub(super) fn avx2_f16c() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
    })
}

fn force_avx2() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("MISTRALRS_FORCE_AVX2").as_deref() == Ok("1"))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let len = a.len();
    let main = len & !7;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < main {
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc,
        );
        i += 8;
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    let mut sum = _mm_cvtss_f32(s);
    for j in main..len {
        sum += a[j] * b[j];
    }
    sum
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot4_f16_avx2(
    q: &[half::f16],
    k0: &[half::f16],
    k1: &[half::f16],
    k2: &[half::f16],
    k3: &[half::f16],
) -> [f32; 4] {
    use core::arch::x86_64::*;
    let len = q.len();
    let main = len & !7;
    let mut a = [_mm256_setzero_ps(); 4];
    let ks = [k0, k1, k2, k3];
    let mut i = 0;
    while i < main {
        let x = _mm256_cvtph_ps(_mm_loadu_si128(q.as_ptr().add(i) as *const _));
        for (acc, k) in a.iter_mut().zip(ks.iter()) {
            *acc = _mm256_fmadd_ps(
                x,
                _mm256_cvtph_ps(_mm_loadu_si128(k.as_ptr().add(i) as *const _)),
                *acc,
            );
        }
        i += 8;
    }
    let mut out = [0f32; 4];
    for (o, acc) in out.iter_mut().zip(a.iter()) {
        let hi = _mm256_extractf128_ps(*acc, 1);
        let lo = _mm256_castps256_ps128(*acc);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        *o = _mm_cvtss_f32(s);
    }
    for j in main..len {
        let qj = q[j].to_f32();
        for (o, k) in out.iter_mut().zip(ks.iter()) {
            *o += qj * k[j].to_f32();
        }
    }
    out
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn mad_f16_avx2(acc: &mut [f32], values: &[half::f16], scale: f32) {
    use core::arch::x86_64::*;
    let len = acc.len();
    let main = len & !7;
    let s = _mm256_set1_ps(scale);
    let mut i = 0;
    while i < main {
        let v = _mm256_cvtph_ps(_mm_loadu_si128(values.as_ptr().add(i) as *const _));
        let a = _mm256_loadu_ps(acc.as_ptr().add(i));
        _mm256_storeu_ps(acc.as_mut_ptr().add(i), _mm256_fmadd_ps(v, s, a));
        i += 8;
    }
    for j in main..len {
        acc[j] += values[j].to_f32() * scale;
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn expand_f16_avx2(dst: &mut [f32], src: &[half::f16]) {
    use core::arch::x86_64::*;
    let len = dst.len();
    let main = len & !7;
    let mut i = 0;
    while i < main {
        let v = _mm256_cvtph_ps(_mm_loadu_si128(src.as_ptr().add(i) as *const _));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), v);
        i += 8;
    }
    for j in main..len {
        dst[j] = src[j].to_f32();
    }
}

#[allow(clippy::excessive_precision)]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_row_f32_avx2(row: &mut [f32], m: f32) -> f32 {
    use core::arch::x86_64::*;
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const C0: f32 = 0.693_359_375;
    const C1: f32 = -2.121_944_4e-4;
    let len = row.len();
    let main = len & !7;
    let mv = _mm256_set1_ps(m);
    let mut sumv = _mm256_setzero_ps();
    let mut i = 0;
    while i < main {
        let x = _mm256_sub_ps(_mm256_loadu_ps(row.as_ptr().add(i)), mv);
        let x = _mm256_max_ps(
            _mm256_set1_ps(-87.0),
            _mm256_min_ps(_mm256_set1_ps(87.0), x),
        );
        let z = _mm256_round_ps::<0>(_mm256_mul_ps(x, _mm256_set1_ps(LOG2E)));
        let r = _mm256_fnmadd_ps(
            z,
            _mm256_set1_ps(C1),
            _mm256_fnmadd_ps(z, _mm256_set1_ps(C0), x),
        );
        let r2 = _mm256_mul_ps(r, r);
        let p = _mm256_fmadd_ps(
            r,
            _mm256_fmadd_ps(
                r,
                _mm256_fmadd_ps(
                    r,
                    _mm256_fmadd_ps(
                        r,
                        _mm256_set1_ps(0.001_392_034_5),
                        _mm256_set1_ps(0.008_333_45),
                    ),
                    _mm256_set1_ps(0.041_665_795),
                ),
                _mm256_set1_ps(0.166_665_46),
            ),
            _mm256_set1_ps(0.5),
        );
        let p = _mm256_add_ps(r, _mm256_mul_ps(r2, p));
        let zi = _mm256_cvtps_epi32(z);
        let e = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(zi, _mm256_set1_epi32(127)),
            23,
        ));
        let v = _mm256_mul_ps(e, _mm256_add_ps(_mm256_set1_ps(1.0), p));
        _mm256_storeu_ps(row.as_mut_ptr().add(i), v);
        sumv = _mm256_add_ps(sumv, v);
        i += 8;
    }
    let hi = _mm256_extractf128_ps(sumv, 1);
    let lo = _mm256_castps256_ps128(sumv);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    let mut sum = _mm_cvtss_f32(s);
    for v in &mut row[main..] {
        let e = super::elem::fast_exp(*v - m);
        *v = e;
        sum += e;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx_kernels_match_reference() {
        for len in [16usize, 64, 128, 131] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.13).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).cos()).collect();
            let refdot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot_f32(&a, &b) - refdot).abs() < 1e-3 * len as f32);
            let d4 = dot4_f32(&a, &b, &a, &b, &a);
            assert!((d4[0] - refdot).abs() < 1e-3 * len as f32);
            let mut acc = vec![0.5f32; len];
            let mut acc_ref = acc.clone();
            mad_f32(&mut acc, &a, 1.7);
            for (r, v) in acc_ref.iter_mut().zip(&a) {
                *r += v * 1.7;
            }
            for (g, w) in acc.iter().zip(&acc_ref) {
                assert!((g - w).abs() < 1e-4);
            }
            let mut row = a.clone();
            let m = max_f32(&row);
            let s = softmax_row_f32(&mut row, m);
            let want: f32 = a.iter().map(|v| (v - m).exp()).sum();
            assert!((s - want).abs() / want < 1e-3, "softmax sum {s} vs {want}");
        }
    }
}
