//! Dot products shared by the CPU decode and MoE kernels.
//!
//! Both kernels need the same property from a dot: four independent accumulators over a fixed
//! lane assignment, so one output element's reduction order does not depend on how outputs were
//! partitioned across threads. That is what lets those kernels claim bit-exact parallelism.
//!
//! bf16 is widened as it loads rather than upcast into a temporary, which is the whole point on
//! a bandwidth-bound decode: the weight is streamed once and never materialised as f32.

/// `a . b` with four independent accumulators.
#[cfg(target_arch = "aarch64")]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= a.len() {
            s0 = vfmaq_f32(
                s0,
                vld1q_f32(a.as_ptr().add(i)),
                vld1q_f32(b.as_ptr().add(i)),
            );
            s1 = vfmaq_f32(
                s1,
                vld1q_f32(a.as_ptr().add(i + 4)),
                vld1q_f32(b.as_ptr().add(i + 4)),
            );
            s2 = vfmaq_f32(
                s2,
                vld1q_f32(a.as_ptr().add(i + 8)),
                vld1q_f32(b.as_ptr().add(i + 8)),
            );
            s3 = vfmaq_f32(
                s3,
                vld1q_f32(a.as_ptr().add(i + 12)),
                vld1q_f32(b.as_ptr().add(i + 12)),
            );
            i += 16;
        }
        let mut sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
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

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (ai, bi) in a.iter().zip(b) {
        acc += ai * bi;
    }
    acc
}

/// `a . bf16(b)`, widening the bf16 lanes as they load.
/// Widen four adjacent bf16 lanes into one `float32x4_t`.
///
/// bf16 is the top half of an f32, so this is a 16-bit left shift after widening u16 -> u32.
/// A 16-bit immediate shift does not exist for `u16`, hence the `vmovl_u16` first.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn widen(ptr: *const u16) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;
    unsafe { vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vld1_u16(ptr)))) }
}

/// `a . b` in bf16. Four accumulators, widened as the weights load.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn dot_bf16(a: &[f32], b: &[u16]) -> f32 {
    use core::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= a.len() {
            s0 = vfmaq_f32(s0, vld1q_f32(a.as_ptr().add(i)), widen(b.as_ptr().add(i)));
            s1 = vfmaq_f32(
                s1,
                vld1q_f32(a.as_ptr().add(i + 4)),
                widen(b.as_ptr().add(i + 4)),
            );
            s2 = vfmaq_f32(
                s2,
                vld1q_f32(a.as_ptr().add(i + 8)),
                widen(b.as_ptr().add(i + 8)),
            );
            s3 = vfmaq_f32(
                s3,
                vld1q_f32(a.as_ptr().add(i + 12)),
                widen(b.as_ptr().add(i + 12)),
            );
            i += 16;
        }
        let mut sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        while i + 4 <= a.len() {
            sum = vfmaq_f32(sum, vld1q_f32(a.as_ptr().add(i)), widen(b.as_ptr().add(i)));
            i += 4;
        }
        let mut out = vaddvq_f32(sum);
        while i < a.len() {
            out += *a.get_unchecked(i) * f32::from_bits((*b.get_unchecked(i) as u32) << 16);
            i += 1;
        }
        out
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn dot_bf16(a: &[f32], b: &[u16]) -> f32 {
    let mut acc = 0f32;
    for (ai, bi) in a.iter().zip(b) {
        acc += ai * f32::from_bits((*bi as u32) << 16);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serial_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0f32;
        for (x, w) in a.iter().zip(b) {
            acc += x * w;
        }
        acc
    }

    /// The vectorised dot reassociates, so it is not the serial order. What must hold is that the
    /// order is a property of the lane assignment and not of anything the caller varies, which
    /// is what lets the decode and MoE kernels claim bit-exact parallelism.
    #[test]
    fn dots_agree_with_the_serial_order_in_scale_and_sign() {
        let k = 4096;
        let x: Vec<f32> = (0..k).map(|i| (i % 31) as f32 * 0.125 - 1.0).collect();
        let w: Vec<f32> = (0..k).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
        let wb: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();

        let serial = serial_f32(&x, &w);
        let wide = dot_f32(&x, &w);
        assert!(
            (wide - serial).abs() <= 1e-3 * serial.abs().max(1.0),
            "f32 {wide} vs {serial}"
        );
        let bf = dot_bf16(&x, &wb);
        assert!(
            (bf - serial).abs() <= 1e-2 * serial.abs().max(1.0),
            "bf16 {bf} vs {serial}"
        );
        // Repeated calls are identical: nothing here depends on thread scheduling or on where
        // in the buffer the operands happen to sit.
        assert_eq!(dot_f32(&x, &w), dot_f32(&x, &w));
        assert_eq!(dot_bf16(&x, &wb), dot_bf16(&x, &wb));
    }

    /// Lengths that do not fill the 16-wide main loop, so the 4-wide and scalar tails run.
    #[test]
    fn tails_match_the_vectorised_prefix() {
        for k in [0usize, 1, 3, 4, 5, 15, 16, 17, 31, 33] {
            let x: Vec<f32> = (0..k).map(|i| i as f32).collect();
            let w: Vec<f32> = (0..k).map(|i| (i * 2) as f32).collect();
            let wb: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let serial = serial_f32(&x, &w);
            assert!((dot_f32(&x, &w) - serial).abs() < 1e-3, "f32 k={k}");
            assert!((dot_bf16(&x, &wb) - serial).abs() < 1e-2, "bf16 k={k}");
        }
    }
}
