use half::{bf16, f16};

#[cfg(target_arch = "x86_64")]
use super::avx::{dot_f32, mad_f32, scale_f32};
#[cfg(target_arch = "aarch64")]
use super::neon::{dot_f32, mad_f32, scale_f32};

#[inline(always)]
pub(super) fn simd_max_f32(xs: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    return super::neon::max_f32(xs);
    #[cfg(target_arch = "x86_64")]
    return super::avx::max_f32(xs);
    #[allow(unreachable_code)]
    xs.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

#[inline(always)]
pub(super) fn simd_softmax_row_f32(row: &mut [f32], m: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    return super::neon::softmax_row_f32(row, m);
    #[cfg(target_arch = "x86_64")]
    return super::avx::softmax_row_f32(row, m);
    #[allow(unreachable_code)]
    {
        let mut sum = 0f32;
        for v in row.iter_mut() {
            *v = fast_exp(*v - m);
            sum += *v;
        }
        sum
    }
}

const DOT_CHUNK: usize = 4;

pub(in crate::attention) trait DowncastF32 {
    fn cast(x: f32) -> Self;
}

impl DowncastF32 for f32 {
    #[inline(always)]
    fn cast(x: f32) -> Self {
        x
    }
}

impl DowncastF32 for f16 {
    #[inline(always)]
    fn cast(x: f32) -> Self {
        f16::from_f32(x)
    }
}

impl DowncastF32 for bf16 {
    #[inline(always)]
    fn cast(x: f32) -> Self {
        bf16::from_f32(x)
    }
}

pub(in crate::attention) trait ElemOps: Copy + DowncastF32 {
    const USE_BARRIER_POOL: bool = false;
    // score via a once-expanded f32 K tile instead of converting per q-row dot
    const EXPAND_SCORE: bool = false;

    fn to_f32(self) -> f32;
    fn dot(a: &[Self], b: &[Self]) -> f32;

    fn dot4(q: &[Self], k0: &[Self], k1: &[Self], k2: &[Self], k3: &[Self]) -> [f32; 4] {
        [
            Self::dot(q, k0),
            Self::dot(q, k1),
            Self::dot(q, k2),
            Self::dot(q, k3),
        ]
    }

    #[inline(always)]
    fn scale_acc(xs: &mut [f32], scale: f32) {
        for v in xs {
            *v *= scale;
        }
    }

    #[inline(always)]
    fn mad(acc: &mut [f32], values: &[Self], scale: f32) {
        for (acc, value) in acc.iter_mut().zip(values.iter()) {
            *acc += value.to_f32() * scale;
        }
    }

    // widen a row into f32 scratch (vectorized where the arch provides it)
    #[inline(always)]
    fn expand_row(dst: &mut [f32], src: &[Self]) {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = s.to_f32();
        }
    }

    // gemm-structured tile scoring over transposed K scratch; false = per-row dot path
    #[allow(clippy::too_many_arguments)]
    fn score_block(
        _q_rows: &[f32],
        _nq: usize,
        _d: usize,
        _k_data: &[Self],
        _row_of: &dyn Fn(usize) -> usize,
        _bs: usize,
        _bn: usize,
        _kt: &mut [f32],
        _scale: f32,
        _s_tile: &mut [f32],
        _tile_stride: usize,
    ) -> bool {
        false
    }

    // P.V over a kv tile for `group` q-rows with evenly strided accumulators; returns
    // false to fall back to the per-row mad loop.
    #[allow(clippy::too_many_arguments)]
    fn pv_tile(
        _rows: &mut [f32],
        _vkq_stride: usize,
        _group: usize,
        _dv: usize,
        _v_data: &[Self],
        _v_row_of: &dyn Fn(usize) -> usize,
        _bs: usize,
        _be: usize,
        _p_tile: &[f32],
        _tile_stride: usize,
    ) -> bool {
        false
    }
}

impl ElemOps for f32 {
    const USE_BARRIER_POOL: bool = true;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn dot(a: &[Self], b: &[Self]) -> f32 {
        dot_f32(a, b)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn dot4(q: &[Self], k0: &[Self], k1: &[Self], k2: &[Self], k3: &[Self]) -> [f32; 4] {
        super::neon::dot4_f32(q, k0, k1, k2, k3)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn dot4(q: &[Self], k0: &[Self], k1: &[Self], k2: &[Self], k3: &[Self]) -> [f32; 4] {
        super::avx::dot4_f32(q, k0, k1, k2, k3)
    }

    #[inline(always)]
    fn scale_acc(xs: &mut [f32], scale: f32) {
        scale_f32(xs, scale)
    }

    #[inline(always)]
    fn mad(acc: &mut [f32], values: &[Self], scale: f32) {
        mad_f32(acc, values, scale)
    }
}

impl ElemOps for f16 {
    // f32 accumulators everywhere; only the K/V streams are half precision
    const USE_BARRIER_POOL: bool = cfg!(any(target_arch = "aarch64", target_arch = "x86_64"));
    const EXPAND_SCORE: bool = cfg!(target_arch = "x86_64");

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline(always)]
    fn dot(a: &[Self], b: &[Self]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        if super::neon::fp16_fast() {
            return super::neon::dot_f16(a, b);
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(v) = super::avx::dot_f16(a, b) {
            return v;
        }
        dot_cast(a, b)
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[inline(always)]
    fn dot4(q: &[Self], k0: &[Self], k1: &[Self], k2: &[Self], k3: &[Self]) -> [f32; 4] {
        #[cfg(target_arch = "aarch64")]
        if super::neon::fp16_fast() {
            return super::neon::dot4_f16(q, k0, k1, k2, k3);
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(v) = super::avx::dot4_f16(q, k0, k1, k2, k3) {
            return v;
        }
        [
            dot_cast(q, k0),
            dot_cast(q, k1),
            dot_cast(q, k2),
            dot_cast(q, k3),
        ]
    }

    #[inline(always)]
    fn scale_acc(xs: &mut [f32], scale: f32) {
        scale_f32(xs, scale)
    }

    #[inline(always)]
    fn mad(acc: &mut [f32], values: &[Self], scale: f32) {
        #[cfg(target_arch = "aarch64")]
        if super::neon::fp16_fast() {
            return super::neon::mad_f16(acc, values, scale);
        }
        #[cfg(target_arch = "x86_64")]
        if super::avx::mad_f16(acc, values, scale) {
            return;
        }
        for (acc, value) in acc.iter_mut().zip(values.iter()) {
            *acc += value.to_f32() * scale;
        }
    }

    #[inline(always)]
    fn expand_row(dst: &mut [f32], src: &[Self]) {
        #[cfg(target_arch = "x86_64")]
        if super::avx::expand_f16(dst, src) {
            return;
        }
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = s.to_f32();
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::too_many_arguments)]
    fn score_block(
        q_rows: &[f32],
        nq: usize,
        d: usize,
        k_data: &[Self],
        row_of: &dyn Fn(usize) -> usize,
        bs: usize,
        bn: usize,
        kt: &mut [f32],
        scale: f32,
        s_tile: &mut [f32],
        tile_stride: usize,
    ) -> bool {
        super::avx::score_block_f16(
            q_rows,
            nq,
            d,
            k_data,
            row_of,
            bs,
            bn,
            kt,
            scale,
            s_tile,
            tile_stride,
        )
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[allow(clippy::too_many_arguments)]
    fn pv_tile(
        rows: &mut [f32],
        vkq_stride: usize,
        group: usize,
        dv: usize,
        v_data: &[Self],
        v_row_of: &dyn Fn(usize) -> usize,
        bs: usize,
        be: usize,
        p_tile: &[f32],
        tile_stride: usize,
    ) -> bool {
        #[cfg(target_arch = "aarch64")]
        return super::neon::pv_tile_f16(
            rows.as_mut_ptr(),
            vkq_stride,
            group,
            dv,
            v_data,
            v_row_of,
            bs,
            be,
            p_tile,
            tile_stride,
        );
        #[cfg(target_arch = "x86_64")]
        super::avx::pv_tile_f16(
            rows.as_mut_ptr(),
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
    }
}

impl ElemOps for bf16 {
    const USE_BARRIER_POOL: bool = cfg!(target_arch = "aarch64");

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline(always)]
    fn dot(a: &[Self], b: &[Self]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        if super::neon::bf16_fast() {
            return super::neon::dot_bf16(a, b);
        }
        dot_cast(a, b)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn dot4(q: &[Self], k0: &[Self], k1: &[Self], k2: &[Self], k3: &[Self]) -> [f32; 4] {
        if super::neon::bf16_fast() {
            return super::neon::dot4_bf16(q, k0, k1, k2, k3);
        }
        [
            dot_cast(q, k0),
            dot_cast(q, k1),
            dot_cast(q, k2),
            dot_cast(q, k3),
        ]
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn scale_acc(xs: &mut [f32], scale: f32) {
        scale_f32(xs, scale)
    }

    #[inline(always)]
    fn mad(acc: &mut [f32], values: &[Self], scale: f32) {
        #[cfg(target_arch = "aarch64")]
        if super::neon::bf16_fast() {
            return super::neon::mad_bf16(acc, values, scale);
        }
        for (acc, value) in acc.iter_mut().zip(values.iter()) {
            *acc += value.to_f32() * scale;
        }
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0f32;
    let chunks = a.len() / DOT_CHUNK;
    for i in 0..chunks {
        let i = i * DOT_CHUNK;
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
    }
    for i in (chunks * DOT_CHUNK)..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn scale_f32(xs: &mut [f32], scale: f32) {
    for v in xs {
        *v *= scale;
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn mad_f32(acc: &mut [f32], values: &[f32], scale: f32) {
    for (acc, value) in acc.iter_mut().zip(values.iter()) {
        *acc += *value * scale;
    }
}

#[inline(always)]
fn dot_cast<T: ElemOps>(a: &[T], b: &[T]) -> f32 {
    let mut sum = 0f32;
    let chunks = a.len() / DOT_CHUNK;
    for i in 0..chunks {
        let i = i * DOT_CHUNK;
        sum += a[i].to_f32() * b[i].to_f32()
            + a[i + 1].to_f32() * b[i + 1].to_f32()
            + a[i + 2].to_f32() * b[i + 2].to_f32()
            + a[i + 3].to_f32() * b[i + 3].to_f32();
    }
    for i in (chunks * DOT_CHUNK)..a.len() {
        sum += a[i].to_f32() * b[i].to_f32();
    }
    sum
}

// Cephes-style exp for the online-softmax hot path; inputs are <= 0 after max
// subtraction, ~1e-7 rel error, ~4x faster than libm expf.
#[inline(always)]
pub(crate) fn fast_exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const C0: f32 = 0.693_359_4;
    const C1: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.0, 87.0);
    let z = (x * LOG2E).round();
    let r = x - z * C0 - z * C1;
    let r2 = r * r;
    let p = r + r2
        * (0.5
            + r * (0.166_665_46 + r * (0.041_665_795 + r * (0.008_333_45 + r * 0.001_392_034_5))));
    let e = f32::from_bits((((z as i32) + 127) << 23) as u32);
    e * (1.0 + p)
}
