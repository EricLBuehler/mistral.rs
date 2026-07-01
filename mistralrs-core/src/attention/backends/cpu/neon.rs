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
