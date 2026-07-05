#[inline(always)]
pub(super) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 16 <= a.len() {
            sum0 = vfmaq_f32(
                sum0,
                vld1q_f32(a.as_ptr().add(i)),
                vld1q_f32(b.as_ptr().add(i)),
            );
            sum1 = vfmaq_f32(
                sum1,
                vld1q_f32(a.as_ptr().add(i + 4)),
                vld1q_f32(b.as_ptr().add(i + 4)),
            );
            sum2 = vfmaq_f32(
                sum2,
                vld1q_f32(a.as_ptr().add(i + 8)),
                vld1q_f32(b.as_ptr().add(i + 8)),
            );
            sum3 = vfmaq_f32(
                sum3,
                vld1q_f32(a.as_ptr().add(i + 12)),
                vld1q_f32(b.as_ptr().add(i + 12)),
            );
            i += 16;
        }
        let mut sum = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        while i + 4 <= a.len() {
            sum = vfmaq_f32(
                sum,
                vld1q_f32(a.as_ptr().add(i)),
                vld1q_f32(b.as_ptr().add(i)),
            );
            i += 4;
        }
        let mut out = vaddvq_f32(sum);
        while i < a.len() {
            out += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        out
    }
}

#[inline(always)]
pub(super) fn scale_f32(xs: &mut [f32], scale: f32) {
    use core::arch::aarch64::*;

    unsafe {
        let scale_v = vdupq_n_f32(scale);
        let mut i = 0usize;
        while i + 16 <= xs.len() {
            let p = xs.as_mut_ptr().add(i);
            vst1q_f32(p, vmulq_f32(vld1q_f32(p), scale_v));
            vst1q_f32(p.add(4), vmulq_f32(vld1q_f32(p.add(4)), scale_v));
            vst1q_f32(p.add(8), vmulq_f32(vld1q_f32(p.add(8)), scale_v));
            vst1q_f32(p.add(12), vmulq_f32(vld1q_f32(p.add(12)), scale_v));
            i += 16;
        }
        while i + 4 <= xs.len() {
            let p = xs.as_mut_ptr().add(i);
            vst1q_f32(p, vmulq_f32(vld1q_f32(p), scale_v));
            i += 4;
        }
        while i < xs.len() {
            *xs.get_unchecked_mut(i) *= scale;
            i += 1;
        }
    }
}

#[inline(always)]
pub(super) fn mad_f32(acc: &mut [f32], values: &[f32], scale: f32) {
    use core::arch::aarch64::*;

    unsafe {
        let scale_v = vdupq_n_f32(scale);
        let mut i = 0usize;
        while i + 16 <= acc.len() {
            let p = acc.as_mut_ptr().add(i);
            vst1q_f32(
                p,
                vfmaq_f32(vld1q_f32(p), vld1q_f32(values.as_ptr().add(i)), scale_v),
            );
            vst1q_f32(
                p.add(4),
                vfmaq_f32(
                    vld1q_f32(p.add(4)),
                    vld1q_f32(values.as_ptr().add(i + 4)),
                    scale_v,
                ),
            );
            vst1q_f32(
                p.add(8),
                vfmaq_f32(
                    vld1q_f32(p.add(8)),
                    vld1q_f32(values.as_ptr().add(i + 8)),
                    scale_v,
                ),
            );
            vst1q_f32(
                p.add(12),
                vfmaq_f32(
                    vld1q_f32(p.add(12)),
                    vld1q_f32(values.as_ptr().add(i + 12)),
                    scale_v,
                ),
            );
            i += 16;
        }
        while i + 4 <= acc.len() {
            let p = acc.as_mut_ptr().add(i);
            vst1q_f32(
                p,
                vfmaq_f32(vld1q_f32(p), vld1q_f32(values.as_ptr().add(i)), scale_v),
            );
            i += 4;
        }
        while i < acc.len() {
            *acc.get_unchecked_mut(i) += *values.get_unchecked(i) * scale;
            i += 1;
        }
    }
}

// Four dots sharing one pass over q: q vectors load once per 4 k rows and the four
// independent fma chains keep the pipes full (a lone 128-dot is latency-bound).
#[inline(always)]
pub(super) fn dot4_f32(q: &[f32], k0: &[f32], k1: &[f32], k2: &[f32], k3: &[f32]) -> [f32; 4] {
    use core::arch::aarch64::*;

    unsafe {
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= q.len() {
            let qv = vld1q_f32(q.as_ptr().add(i));
            a0 = vfmaq_f32(a0, qv, vld1q_f32(k0.as_ptr().add(i)));
            a1 = vfmaq_f32(a1, qv, vld1q_f32(k1.as_ptr().add(i)));
            a2 = vfmaq_f32(a2, qv, vld1q_f32(k2.as_ptr().add(i)));
            a3 = vfmaq_f32(a3, qv, vld1q_f32(k3.as_ptr().add(i)));
            i += 4;
        }
        let mut out = [
            vaddvq_f32(a0),
            vaddvq_f32(a1),
            vaddvq_f32(a2),
            vaddvq_f32(a3),
        ];
        while i < q.len() {
            let qi = *q.get_unchecked(i);
            out[0] += qi * *k0.get_unchecked(i);
            out[1] += qi * *k1.get_unchecked(i);
            out[2] += qi * *k2.get_unchecked(i);
            out[3] += qi * *k3.get_unchecked(i);
            i += 1;
        }
        out
    }
}

// In-place exp(x - m) over a tile row with the Cephes-style polynomial; returns the sum.
#[inline(always)]
pub(super) fn softmax_row_f32(row: &mut [f32], m: f32) -> f32 {
    use core::arch::aarch64::*;

    const C0: f32 = 0.693_359_375;
    const C1: f32 = -2.121_944_4e-4;
    unsafe {
        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let c0 = vdupq_n_f32(C0);
        let c1 = vdupq_n_f32(C1);
        let p0 = vdupq_n_f32(0.5);
        let p1 = vdupq_n_f32(0.166_665_46);
        let p2 = vdupq_n_f32(0.041_665_795);
        let p3 = vdupq_n_f32(0.008_333_45);
        let p4 = vdupq_n_f32(0.001_392_034_5);
        let one = vdupq_n_f32(1.0);
        let mv = vdupq_n_f32(m);
        let lo_clamp = vdupq_n_f32(-87.0);
        let hi_clamp = vdupq_n_f32(87.0);
        let bias = vdupq_n_s32(127);

        let mut sum_v = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= row.len() {
            let ptr = row.as_mut_ptr().add(i);
            let x = vminq_f32(vmaxq_f32(vsubq_f32(vld1q_f32(ptr), mv), lo_clamp), hi_clamp);
            let zi = vcvtaq_s32_f32(vmulq_f32(x, log2e));
            let zf = vcvtq_f32_s32(zi);
            let r = vfmsq_f32(vfmsq_f32(x, zf, c0), zf, c1);
            let mut p = vfmaq_f32(p3, r, p4);
            p = vfmaq_f32(p2, r, p);
            p = vfmaq_f32(p1, r, p);
            p = vfmaq_f32(p0, r, p);
            let poly = vfmaq_f32(r, vmulq_f32(r, r), p);
            let e = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(zi, bias), 23));
            let result = vmulq_f32(e, vaddq_f32(one, poly));
            vst1q_f32(ptr, result);
            sum_v = vaddq_f32(sum_v, result);
            i += 4;
        }
        let mut sum = vaddvq_f32(sum_v);
        while i < row.len() {
            let v = super::elem::fast_exp(*row.get_unchecked(i) - m);
            *row.get_unchecked_mut(i) = v;
            sum += v;
            i += 1;
        }
        sum
    }
}

#[inline(always)]
pub(super) fn max_f32(row: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    unsafe {
        let mut i = 0usize;
        let mut mv = vdupq_n_f32(f32::NEG_INFINITY);
        while i + 4 <= row.len() {
            mv = vmaxq_f32(mv, vld1q_f32(row.as_ptr().add(i)));
            i += 4;
        }
        let mut m = vmaxvq_f32(mv);
        while i < row.len() {
            m = m.max(*row.get_unchecked(i));
            i += 1;
        }
        m
    }
}
