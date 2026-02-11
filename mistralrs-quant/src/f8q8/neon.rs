use super::{BlockF8Q8, BlockQ8_0, QK8_0};
use candle_core::Result;

#[allow(unused_imports)]
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[allow(unused_imports)]
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(feature = "arm-nightly-feat")]
use std::arch::is_aarch64_feature_detected;

#[inline(always)]
#[cfg(feature = "arm-nightly-feat")]
unsafe fn vdotq_s32_local(vz: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    if is_aarch64_feature_detected!("dotprod") {
        vdotq_s32(vz, a, b)
    } else {
        unreachable!();
    }
}

#[inline(always)]
#[cfg(not(feature = "arm-nightly-feat"))]
unsafe fn vdotq_s32_local(vz: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(vz, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)))
}

#[inline(always)]
pub(crate) fn vec_dot_f8q8_q8_0(n: usize, xs: &[BlockF8Q8], ys: &[BlockQ8_0]) -> Result<f32> {
    let qk = QK8_0;
    if !n.is_multiple_of(QK8_0) {
        candle_core::bail!("vec_dot_f8q8_q8_0: {n} is not divisible by {qk}")
    }
    let nb = n / QK8_0;
    unsafe {
        let mut sumv0 = vdupq_n_f32(0.0f32);
        for i in 0..nb {
            let x0 = &xs[i];
            let y0 = &ys[i];

            let x0_0 = vld1q_s8(x0.qs.as_ptr());
            let x0_1 = vld1q_s8(x0.qs.as_ptr().add(16));

            // load y
            let y0_0 = vld1q_s8(y0.qs.as_ptr());
            let y0_1 = vld1q_s8(y0.qs.as_ptr().add(16));

            let p0 = vdotq_s32_local(vdupq_n_s32(0), x0_0, y0_0);
            let p1 = vdotq_s32_local(vdupq_n_s32(0), x0_1, y0_1);

            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(vaddq_s32(p0, p1)),
                x0.dq_d() * y0.d.to_f32(),
            );
        }
        Ok(vaddvq_f32(sumv0))
    }
}

#[cfg(feature = "arm-nightly-feat")]
struct I8mmParams {
    x0: int8x16_t,
    x1: int8x16_t,
    x2: int8x16_t,
    x3: int8x16_t,
    y0: int8x16_t,
    y1: int8x16_t,
    y2: int8x16_t,
    y3: int8x16_t,
}

#[cfg(feature = "arm-nightly-feat")]
impl I8mmParams {
    #[inline(always)]
    unsafe fn new(
        xv0_0: int8x16_t,
        xv0_1: int8x16_t,
        xv1_0: int8x16_t,
        xv1_1: int8x16_t,
        yv0_0: int8x16_t,
        yv0_1: int8x16_t,
        yv1_0: int8x16_t,
        yv1_1: int8x16_t,
    ) -> Self {
        // 1. 16xi8 -> 2xi64
        let xv0_0 = vreinterpretq_s64_s8(xv0_0);
        let xv0_1 = vreinterpretq_s64_s8(xv0_1);
        let xv1_0 = vreinterpretq_s64_s8(xv1_0);
        let xv1_1 = vreinterpretq_s64_s8(xv1_1);

        let yv0_0 = vreinterpretq_s64_s8(yv0_0);
        let yv0_1 = vreinterpretq_s64_s8(yv0_1);
        let yv1_0 = vreinterpretq_s64_s8(yv1_0);
        let yv1_1 = vreinterpretq_s64_s8(yv1_1);

        // 2. ZIP
        let x0_0 = vzip1q_s64(xv0_0, xv1_0);
        let x0_1 = vzip2q_s64(xv0_0, xv1_0);
        let x1_0 = vzip1q_s64(xv0_1, xv1_1);
        let x1_1 = vzip2q_s64(xv0_1, xv1_1);

        let y0_0 = vzip1q_s64(yv0_0, yv1_0);
        let y0_1 = vzip2q_s64(yv0_0, yv1_0);
        let y1_0 = vzip1q_s64(yv0_1, yv1_1);
        let y1_1 = vzip2q_s64(yv0_1, yv1_1);

        // 3. interpret back
        let x0_0 = vreinterpretq_s8_s64(x0_0);
        let x0_1 = vreinterpretq_s8_s64(x0_1);
        let x1_0 = vreinterpretq_s8_s64(x1_0);
        let x1_1 = vreinterpretq_s8_s64(x1_1);

        let y0_0 = vreinterpretq_s8_s64(y0_0);
        let y0_1 = vreinterpretq_s8_s64(y0_1);
        let y1_0 = vreinterpretq_s8_s64(y1_0);
        let y1_1 = vreinterpretq_s8_s64(y1_1);

        I8mmParams {
            x0: x0_0,
            x1: x0_1,
            x2: x1_0,
            x3: x1_1,
            y0: y0_0,
            y1: y0_1,
            y2: y1_0,
            y3: y1_1,
        }
    }

    #[inline(always)]
    unsafe fn calculate(&self, acc: int32x4_t) -> int32x4_t {
        if is_aarch64_feature_detected!("i8mm") {
            self.impl_calc(acc)
        } else {
            unreachable!();
        }
    }

    unsafe fn impl_calc(&self, acc: int32x4_t) -> int32x4_t {
        let mut a = acc;
        a = vmmlaq_s32(a, self.y0, self.x0);
        a = vmmlaq_s32(a, self.y1, self.x1);
        a = vmmlaq_s32(a, self.y2, self.x2);
        vmmlaq_s32(a, self.y3, self.x3)
    }
}

#[inline(always)]
#[cfg(feature = "arm-nightly-feat")]
pub(crate) fn i8mm_f8q8_q8_0(
    n: usize,
    xs_0: &[BlockF8Q8],
    xs_1: &[BlockF8Q8],
    ys_0: &[BlockQ8_0],
    ys_1: &[BlockQ8_0],
) -> Result<[f32; 4]> {
    assert_eq!(xs_0.len(), xs_1.len());
    assert_eq!(ys_0.len(), ys_1.len());
    assert_eq!(xs_0.len(), ys_0.len());
    let qk = QK8_0;
    if !n.is_multiple_of(QK8_0) {
        candle_core::bail!("i8mm_f8q8_q8_0: {n} is not divisible by {qk}")
    }
    let nb = n / QK8_0;
    unsafe {
        let mut sum_f32 = vdupq_n_f32(0.0);

        for i in 0..nb {
            let x0 = &xs_0[i];
            let x1 = &xs_1[i];
            let y0 = &ys_0[i];
            let y1 = &ys_1[i];

            let factor_00: f32 = x0.dq_d() * y0.d.to_f32();
            let factor_01: f32 = x1.dq_d() * y0.d.to_f32();
            let factor_10: f32 = x0.dq_d() * y1.d.to_f32();
            let factor_11: f32 = x1.dq_d() * y1.d.to_f32();

            let xv0_0 = vld1q_s8(x0.qs.as_ptr());
            let xv0_1 = vld1q_s8(x0.qs.as_ptr().add(16));
            let xv1_0 = vld1q_s8(x1.qs.as_ptr());
            let xv1_1 = vld1q_s8(x1.qs.as_ptr().add(16));

            let yv0_0 = vld1q_s8(y0.qs.as_ptr());
            let yv0_1 = vld1q_s8(y0.qs.as_ptr().add(16));
            let yv1_0 = vld1q_s8(y1.qs.as_ptr());
            let yv1_1 = vld1q_s8(y1.qs.as_ptr().add(16));

            let i8mm = I8mmParams::new(xv0_0, xv0_1, xv1_0, xv1_1, yv0_0, yv0_1, yv1_0, yv1_1);
            let loop_sum_s32 = i8mm.calculate(vdupq_n_s32(0));

            // scaling
            let factor_elems: [f32; 4] = [factor_00, factor_01, factor_10, factor_11];
            let rawptr = &factor_elems as *const f32;
            let factor: float32x4_t = vld1q_f32(rawptr);
            let loop_sum_f32 = vcvtq_f32_s32(loop_sum_s32);

            sum_f32 = vmlaq_f32(sum_f32, loop_sum_f32, factor);
        }
        // extract elements of the vector register
        let f0 = vgetq_lane_f32(sum_f32, 0);
        let f1 = vgetq_lane_f32(sum_f32, 1);
        let f2 = vgetq_lane_f32(sum_f32, 2);
        let f3 = vgetq_lane_f32(sum_f32, 3);
        let res: [f32; 4] = [f0, f1, f2, f3];
        Ok(res)
    }
}
