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

/// One BFMMLA step: `acc += a_block * b_block` over four bf16 lanes.
///
/// Both rows of both operands carry the same four lanes, so lane 0 of the 2x2 fp32 accumulator
/// holds the whole partial product and lanes 1-3 duplicate it. That waste is inherent to a GEMV
/// rather than to this formulation; the win is that the bf16 weights are consumed at their
/// native width instead of being widened to f32 on every load.
///
/// The activation is narrowed to bf16 with a 16-bit right shift. Exact for activations that
/// arrived as bf16 - which is what the model produces - but it would truncate a genuine f32.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "bf16")]
#[inline]
unsafe fn bfmmla_block(
    acc: core::arch::aarch64::float32x4_t,
    av: core::arch::aarch64::float32x4_t,
    bv: core::arch::aarch64::uint16x4_t,
) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;
    unsafe {
        let a = vshrn_n_u32::<16>(vreinterpretq_u32_f32(av));
        let n = vcombine_u16(a, a);
        let m = vcombine_u16(bv, bv);
        let mut o = acc;
        // The arrangement specifiers have to be literal template text after the placeholder:
        // `bfmmla` requires `.4s` on the accumulator and `.8h` on the operands, and Rust's asm
        // template parser has no syntax for spelling those inside `{}`.
        core::arch::asm!(
            "bfmmla {o:v}.4s, {n:v}.8h, {m:v}.8h",
            o = inout(vreg) o,
            n = in(vreg) n,
            m = in(vreg) m
        );
        o
    }
}

/// `a . b` reduced through BFMMLA instead of [`dot_bf16`]'s widen-then-FMA loop.
///
/// Selected by `MISTRALRS_DOT_BFMMLA=1`. The point is matching the reference oracle's reduction
/// order, not speed: the vLLM CPU GEMM that produces the goldens is bf16 x bf16 -> fp32 through
/// BFMMLA, so its partial sums are ordered by this instruction. The two orders differ by about
/// 0.15% relative on outputs of magnitude ~15, far more than rounding, which is enough to move
/// the Kolibri-1 router near a knife-edge.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "bf16")]
pub(crate) unsafe fn dot_bf16_bfmmla(a: &[f32], b: &[u16]) -> f32 {
    use core::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= a.len() {
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            s0 = bfmmla_block(s0, vld1q_f32(ap.add(i)), vld1_u16(bp.add(i)));
            s1 = bfmmla_block(s1, vld1q_f32(ap.add(i + 4)), vld1_u16(bp.add(i + 4)));
            s2 = bfmmla_block(s2, vld1q_f32(ap.add(i + 8)), vld1_u16(bp.add(i + 8)));
            s3 = bfmmla_block(s3, vld1q_f32(ap.add(i + 12)), vld1_u16(bp.add(i + 12)));
            i += 16;
        }
        // Only lane 0 of each accumulator holds a partial product; lanes 1-3 are the duplicate
        // rows. Sum lane 0 across the four accumulators rather than across the lanes.
        let mut out = vgetq_lane_f32(s0, 0)
            + vgetq_lane_f32(s1, 0)
            + vgetq_lane_f32(s2, 0)
            + vgetq_lane_f32(s3, 0);
        while i + 4 <= a.len() {
            let acc = bfmmla_block(
                vdupq_n_f32(0.0),
                vld1q_f32(a.as_ptr().add(i)),
                vld1_u16(b.as_ptr().add(i)),
            );
            out += vgetq_lane_f32(acc, 0);
            i += 4;
        }
        while i < a.len() {
            out += *a.get_unchecked(i) * f32::from_bits((*b.get_unchecked(i) as u32) << 16);
            i += 1;
        }
        out
    }
}

/// Whether bf16 dots reduce through BFMMLA.
///
/// `MISTRALRS_DOT_BFMMLA=1` selects it, on hosts that have the instruction. Resolved once, so
/// the per-dot cost is one perfectly-predicted branch against a 2560-element reduction.
fn bfmmla_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(target_arch = "aarch64")]
        {
            std::env::var("MISTRALRS_DOT_BFMMLA")
                .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                && std::arch::is_aarch64_feature_detected!("bf16")
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    })
}

/// `a . b` in bf16.
///
/// The reduction order is [`dot_bf16`]'s four-accumulator FMA one, unless `MISTRALRS_DOT_BFMMLA=1`,
/// in which case it is BFMMLA's - matching the reference oracle's CPU GEMM.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn dot_bf16(a: &[f32], b: &[u16]) -> f32 {
    if bfmmla_selected() {
        // Safety: `bfmmla_selected` has just checked the bf16 feature at runtime.
        unsafe { dot_bf16_bfmmla(a, b) }
    } else {
        dot_bf16_fma(a, b)
    }
}

#[cfg(target_arch = "aarch64")]
fn dot_bf16_fma(a: &[f32], b: &[u16]) -> f32 {
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

    /// The BFMMLA reduction must agree with the FMA one to within bf16 rounding, at every tail
    /// length. The two orders are deliberately not bit-equal - that difference is the whole
    /// reason the kernel exists - so this bounds it rather than asserting equality.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn bfmmla_agrees_with_the_fma_order() {
        if !std::arch::is_aarch64_feature_detected!("bf16") {
            return;
        }
        for k in [0usize, 1, 3, 4, 5, 15, 16, 17, 31, 33, 2560] {
            let x: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
            let wb: Vec<u16> = (0..k)
                .map(|i| (((i * 7) % 23) as f32 - 11.0).to_bits() >> 16)
                .map(|b| b as u16)
                .collect();
            let fma = dot_bf16_fma(&x, &wb);
            let ml = unsafe { dot_bf16_bfmmla(&x, &wb) };
            let scale = fma.abs().max(1.0);
            assert!(
                (fma - ml).abs() / scale < 1e-3,
                "k={k}: fma={fma} bfmmla={ml}"
            );
        }
    }

    /// Same inputs, same answer, every time: the flag must not introduce run-to-run variation.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn bfmmla_is_deterministic() {
        if !std::arch::is_aarch64_feature_detected!("bf16") {
            return;
        }
        let k = 2560;
        let x: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
        let wb: Vec<u16> = (0..k)
            .map(|i| (((i * 7) % 23) as f32 - 11.0).to_bits() >> 16)
            .map(|b| b as u16)
            .collect();
        let first = unsafe { dot_bf16_bfmmla(&x, &wb) };
        for _ in 0..64 {
            assert_eq!(first, unsafe { dot_bf16_bfmmla(&x, &wb) });
        }
    }
}
