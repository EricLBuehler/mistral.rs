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

    const C0: f32 = 0.693_359_4;
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

// f16 NEON intrinsics are unstable on stable rustc, so the fp16 kernels use inline asm:
// fmlal/fmlal2 do f16 x f16 multiplies with f32 accumulation (FEAT_FP16FML).
pub(super) fn fp16_fast() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        std::arch::is_aarch64_feature_detected!("fp16")
            && std::arch::is_aarch64_feature_detected!("fhm")
    })
}

#[target_feature(enable = "fp16,fhm")]
unsafe fn dot_f16_main(ap: *const u16, bp: *const u16, n: usize) -> f32 {
    use core::arch::aarch64::*;
    let acc0: float32x4_t;
    let acc1: float32x4_t;
    core::arch::asm!(
        "movi {a0:v}.4s, #0",
        "movi {a1:v}.4s, #0",
        "2:",
        "ldr {t0:q}, [{ap}], #16",
        "ldr {t1:q}, [{bp}], #16",
        "fmlal {a0:v}.4s, {t0:v}.4h, {t1:v}.4h",
        "fmlal2 {a1:v}.4s, {t0:v}.4h, {t1:v}.4h",
        "subs {n}, {n}, #8",
        "b.gt 2b",
        ap = inout(reg) ap => _,
        bp = inout(reg) bp => _,
        n = inout(reg) n => _,
        a0 = out(vreg) acc0,
        a1 = out(vreg) acc1,
        t0 = out(vreg) _,
        t1 = out(vreg) _,
        options(nostack, readonly),
    );
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

#[inline(always)]
pub(super) fn dot_f16(a: &[half::f16], b: &[half::f16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let main = len & !7;
    let mut sum = if main >= 8 {
        unsafe { dot_f16_main(a.as_ptr() as *const u16, b.as_ptr() as *const u16, main) }
    } else {
        0.0
    };
    for i in main..len {
        sum += a[i].to_f32() * b[i].to_f32();
    }
    sum
}

#[target_feature(enable = "fp16,fhm")]
#[allow(clippy::too_many_arguments)]
unsafe fn dot4_f16_main(
    qp: *const u16,
    p0: *const u16,
    p1: *const u16,
    p2: *const u16,
    p3: *const u16,
    n: usize,
) -> [f32; 4] {
    use core::arch::aarch64::*;
    let a0: float32x4_t;
    let a1: float32x4_t;
    let a2: float32x4_t;
    let a3: float32x4_t;
    core::arch::asm!(
        "movi {a0:v}.4s, #0",
        "movi {a1:v}.4s, #0",
        "movi {a2:v}.4s, #0",
        "movi {a3:v}.4s, #0",
        "2:",
        "ldr {tq:q}, [{qp}], #16",
        "ldr {t0:q}, [{p0}], #16",
        "ldr {t1:q}, [{p1}], #16",
        "ldr {t2:q}, [{p2}], #16",
        "ldr {t3:q}, [{p3}], #16",
        "fmlal {a0:v}.4s, {tq:v}.4h, {t0:v}.4h",
        "fmlal2 {a0:v}.4s, {tq:v}.4h, {t0:v}.4h",
        "fmlal {a1:v}.4s, {tq:v}.4h, {t1:v}.4h",
        "fmlal2 {a1:v}.4s, {tq:v}.4h, {t1:v}.4h",
        "fmlal {a2:v}.4s, {tq:v}.4h, {t2:v}.4h",
        "fmlal2 {a2:v}.4s, {tq:v}.4h, {t2:v}.4h",
        "fmlal {a3:v}.4s, {tq:v}.4h, {t3:v}.4h",
        "fmlal2 {a3:v}.4s, {tq:v}.4h, {t3:v}.4h",
        "subs {n}, {n}, #8",
        "b.gt 2b",
        qp = inout(reg) qp => _,
        p0 = inout(reg) p0 => _,
        p1 = inout(reg) p1 => _,
        p2 = inout(reg) p2 => _,
        p3 = inout(reg) p3 => _,
        n = inout(reg) n => _,
        a0 = out(vreg) a0,
        a1 = out(vreg) a1,
        a2 = out(vreg) a2,
        a3 = out(vreg) a3,
        tq = out(vreg) _,
        t0 = out(vreg) _,
        t1 = out(vreg) _,
        t2 = out(vreg) _,
        t3 = out(vreg) _,
        options(nostack, readonly),
    );
    [
        vaddvq_f32(a0),
        vaddvq_f32(a1),
        vaddvq_f32(a2),
        vaddvq_f32(a3),
    ]
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn dot4_f16(
    q: &[half::f16],
    k0: &[half::f16],
    k1: &[half::f16],
    k2: &[half::f16],
    k3: &[half::f16],
) -> [f32; 4] {
    let len = q.len();
    let main = len & !7;
    let mut out = if main >= 8 {
        unsafe {
            dot4_f16_main(
                q.as_ptr() as *const u16,
                k0.as_ptr() as *const u16,
                k1.as_ptr() as *const u16,
                k2.as_ptr() as *const u16,
                k3.as_ptr() as *const u16,
                main,
            )
        }
    } else {
        [0.0; 4]
    };
    for i in main..len {
        let qi = q[i].to_f32();
        out[0] += qi * k0[i].to_f32();
        out[1] += qi * k1[i].to_f32();
        out[2] += qi * k2[i].to_f32();
        out[3] += qi * k3[i].to_f32();
    }
    out
}

// acc (f32) += values (f16) * scale
#[target_feature(enable = "fp16")]
unsafe fn mad_f16_main(ap: *mut f32, vp: *const u16, scale: f32, n: usize) {
    core::arch::asm!(
        "dup {s:v}.4s, {scale:v}.s[0]",
        "2:",
        "ldr {tv:q}, [{vp}], #16",
        "ldp {t0:q}, {t1:q}, [{ap}]",
        "fcvtl {lo:v}.4s, {tv:v}.4h",
        "fcvtl2 {hi:v}.4s, {tv:v}.8h",
        "fmla {t0:v}.4s, {lo:v}.4s, {s:v}.4s",
        "fmla {t1:v}.4s, {hi:v}.4s, {s:v}.4s",
        "stp {t0:q}, {t1:q}, [{ap}], #32",
        "subs {n}, {n}, #8",
        "b.gt 2b",
        ap = inout(reg) ap => _,
        vp = inout(reg) vp => _,
        n = inout(reg) n => _,
        scale = in(vreg) scale,
        s = out(vreg) _,
        tv = out(vreg) _,
        t0 = out(vreg) _,
        t1 = out(vreg) _,
        lo = out(vreg) _,
        hi = out(vreg) _,
        options(nostack),
    );
}

#[inline(always)]
pub(super) fn mad_f16(acc: &mut [f32], values: &[half::f16], scale: f32) {
    let len = acc.len();
    let main = len & !7;
    if main >= 8 {
        unsafe { mad_f16_main(acc.as_mut_ptr(), values.as_ptr() as *const u16, scale, main) };
    }
    for i in main..len {
        acc[i] += values[i].to_f32() * scale;
    }
}

#[cfg(test)]
mod f16_kernel_tests {
    use super::*;
    use half::f16;

    #[test]
    fn f16_asm_kernels_match_reference() {
        if !fp16_fast() {
            return;
        }
        for len in [8usize, 16, 64, 128, 131] {
            let a: Vec<f16> = (0..len)
                .map(|i| f16::from_f32((i as f32 * 0.13).sin()))
                .collect();
            let b: Vec<f16> = (0..len)
                .map(|i| f16::from_f32((i as f32 * 0.07).cos()))
                .collect();
            let c: Vec<f16> = (0..len)
                .map(|i| f16::from_f32(0.3 - i as f32 * 0.001))
                .collect();
            let d: Vec<f16> = (0..len)
                .map(|i| f16::from_f32(0.9 - i as f32 * 0.002))
                .collect();
            let e: Vec<f16> = (0..len)
                .map(|i| f16::from_f32((i % 7) as f32 * 0.1))
                .collect();
            let refdot = |x: &[f16], y: &[f16]| -> f32 {
                x.iter().zip(y).map(|(p, q)| p.to_f32() * q.to_f32()).sum()
            };
            assert!((dot_f16(&a, &b) - refdot(&a, &b)).abs() < 1e-2 * len as f32);
            let d4 = dot4_f16(&a, &b, &c, &d, &e);
            for (got, want) in d4.iter().zip([
                refdot(&a, &b),
                refdot(&a, &c),
                refdot(&a, &d),
                refdot(&a, &e),
            ]) {
                assert!(
                    (got - want).abs() < 1e-2 * len as f32,
                    "{got} vs {want} len={len}"
                );
            }
            let mut acc = vec![0.5f32; len];
            let mut acc_ref = acc.clone();
            mad_f16(&mut acc, &a, 1.7);
            for (r, v) in acc_ref.iter_mut().zip(&a) {
                *r += v.to_f32() * 1.7;
            }
            for (g, w) in acc.iter().zip(&acc_ref) {
                assert!((g - w).abs() < 1e-3, "mad {g} vs {w} len={len}");
            }
        }
    }
}

// bf16 twins of the fp16 kernels: bfdot pairs bf16 products into f32 lanes (FEAT_BF16),
// and bf16 -> f32 widening is a plain left shift by 16.
pub(super) fn bf16_fast() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| std::arch::is_aarch64_feature_detected!("bf16"))
}

#[target_feature(enable = "bf16")]
unsafe fn dot_bf16_inner(ap: *const u16, bp: *const u16, n: usize) -> f32 {
    use core::arch::aarch64::*;
    let acc0: float32x4_t;
    let acc1: float32x4_t;
    core::arch::asm!(
        "movi {a0:v}.4s, #0",
        "movi {a1:v}.4s, #0",
        "2:",
        "ldp {t0:q}, {t2:q}, [{ap}], #32",
        "ldp {t1:q}, {t3:q}, [{bp}], #32",
        "bfdot {a0:v}.4s, {t0:v}.8h, {t1:v}.8h",
        "bfdot {a1:v}.4s, {t2:v}.8h, {t3:v}.8h",
        "subs {n}, {n}, #16",
        "b.gt 2b",
        ap = inout(reg) ap => _,
        bp = inout(reg) bp => _,
        n = inout(reg) n => _,
        a0 = out(vreg) acc0,
        a1 = out(vreg) acc1,
        t0 = out(vreg) _,
        t1 = out(vreg) _,
        t2 = out(vreg) _,
        t3 = out(vreg) _,
        options(nostack, readonly),
    );
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

#[inline(always)]
pub(super) fn dot_bf16(a: &[half::bf16], b: &[half::bf16]) -> f32 {
    let len = a.len();
    let main = len & !15;
    let mut sum = if main >= 16 {
        unsafe { dot_bf16_inner(a.as_ptr() as *const u16, b.as_ptr() as *const u16, main) }
    } else {
        0.0
    };
    for i in main..len {
        sum += a[i].to_f32() * b[i].to_f32();
    }
    sum
}

#[target_feature(enable = "bf16")]
#[allow(clippy::too_many_arguments)]
unsafe fn dot4_bf16_inner(
    qp: *const u16,
    p0: *const u16,
    p1: *const u16,
    p2: *const u16,
    p3: *const u16,
    n: usize,
) -> [f32; 4] {
    use core::arch::aarch64::*;
    let a0: float32x4_t;
    let a1: float32x4_t;
    let a2: float32x4_t;
    let a3: float32x4_t;
    core::arch::asm!(
        "movi {a0:v}.4s, #0",
        "movi {a1:v}.4s, #0",
        "movi {a2:v}.4s, #0",
        "movi {a3:v}.4s, #0",
        "2:",
        "ldr {tq:q}, [{qp}], #16",
        "ldr {t0:q}, [{p0}], #16",
        "ldr {t1:q}, [{p1}], #16",
        "ldr {t2:q}, [{p2}], #16",
        "ldr {t3:q}, [{p3}], #16",
        "bfdot {a0:v}.4s, {tq:v}.8h, {t0:v}.8h",
        "bfdot {a1:v}.4s, {tq:v}.8h, {t1:v}.8h",
        "bfdot {a2:v}.4s, {tq:v}.8h, {t2:v}.8h",
        "bfdot {a3:v}.4s, {tq:v}.8h, {t3:v}.8h",
        "subs {n}, {n}, #8",
        "b.gt 2b",
        qp = inout(reg) qp => _,
        p0 = inout(reg) p0 => _,
        p1 = inout(reg) p1 => _,
        p2 = inout(reg) p2 => _,
        p3 = inout(reg) p3 => _,
        n = inout(reg) n => _,
        a0 = out(vreg) a0,
        a1 = out(vreg) a1,
        a2 = out(vreg) a2,
        a3 = out(vreg) a3,
        tq = out(vreg) _,
        t0 = out(vreg) _,
        t1 = out(vreg) _,
        t2 = out(vreg) _,
        t3 = out(vreg) _,
        options(nostack, readonly),
    );
    [
        vaddvq_f32(a0),
        vaddvq_f32(a1),
        vaddvq_f32(a2),
        vaddvq_f32(a3),
    ]
}

#[inline(always)]
pub(super) fn dot4_bf16(
    q: &[half::bf16],
    k0: &[half::bf16],
    k1: &[half::bf16],
    k2: &[half::bf16],
    k3: &[half::bf16],
) -> [f32; 4] {
    let len = q.len();
    let main = len & !7;
    let mut out = if main >= 8 {
        unsafe {
            dot4_bf16_inner(
                q.as_ptr() as *const u16,
                k0.as_ptr() as *const u16,
                k1.as_ptr() as *const u16,
                k2.as_ptr() as *const u16,
                k3.as_ptr() as *const u16,
                main,
            )
        }
    } else {
        [0.0; 4]
    };
    for i in main..len {
        let qi = q[i].to_f32();
        out[0] += qi * k0[i].to_f32();
        out[1] += qi * k1[i].to_f32();
        out[2] += qi * k2[i].to_f32();
        out[3] += qi * k3[i].to_f32();
    }
    out
}

unsafe fn mad_bf16_inner(ap: *mut f32, vp: *const u16, scale: f32, n: usize) {
    core::arch::asm!(
        "dup {s:v}.4s, {scale:v}.s[0]",
        "2:",
        "ldr {tv:q}, [{vp}], #16",
        "ldp {t0:q}, {t1:q}, [{ap}]",
        "shll {lo:v}.4s, {tv:v}.4h, #16",
        "shll2 {hi:v}.4s, {tv:v}.8h, #16",
        "fmla {t0:v}.4s, {lo:v}.4s, {s:v}.4s",
        "fmla {t1:v}.4s, {hi:v}.4s, {s:v}.4s",
        "stp {t0:q}, {t1:q}, [{ap}], #32",
        "subs {n}, {n}, #8",
        "b.gt 2b",
        ap = inout(reg) ap => _,
        vp = inout(reg) vp => _,
        n = inout(reg) n => _,
        scale = in(vreg) scale,
        s = out(vreg) _,
        tv = out(vreg) _,
        t0 = out(vreg) _,
        t1 = out(vreg) _,
        lo = out(vreg) _,
        hi = out(vreg) _,
        options(nostack),
    );
}

// acc (f32) += values (bf16) * scale
#[inline(always)]
pub(super) fn mad_bf16(acc: &mut [f32], values: &[half::bf16], scale: f32) {
    let len = acc.len();
    let main = len & !7;
    if main >= 8 {
        unsafe { mad_bf16_inner(acc.as_mut_ptr(), values.as_ptr() as *const u16, scale, main) };
    }
    for i in main..len {
        acc[i] += values[i].to_f32() * scale;
    }
}

#[cfg(test)]
mod bf16_kernel_tests {
    use super::*;
    use half::bf16;

    #[test]
    fn bf16_asm_kernels_match_reference() {
        if !bf16_fast() {
            return;
        }
        for len in [16usize, 64, 128, 131] {
            let a: Vec<bf16> = (0..len)
                .map(|i| bf16::from_f32((i as f32 * 0.13).sin()))
                .collect();
            let b: Vec<bf16> = (0..len)
                .map(|i| bf16::from_f32((i as f32 * 0.07).cos()))
                .collect();
            let c: Vec<bf16> = (0..len)
                .map(|i| bf16::from_f32(0.3 - i as f32 * 0.001))
                .collect();
            let d: Vec<bf16> = (0..len)
                .map(|i| bf16::from_f32(0.9 - i as f32 * 0.002))
                .collect();
            let e: Vec<bf16> = (0..len)
                .map(|i| bf16::from_f32((i % 7) as f32 * 0.1))
                .collect();
            let refdot = |x: &[bf16], y: &[bf16]| -> f32 {
                x.iter().zip(y).map(|(p, q)| p.to_f32() * q.to_f32()).sum()
            };
            assert!((dot_bf16(&a, &b) - refdot(&a, &b)).abs() < 3e-2 * len as f32);
            let d4 = dot4_bf16(&a, &b, &c, &d, &e);
            for (got, want) in d4.iter().zip([
                refdot(&a, &b),
                refdot(&a, &c),
                refdot(&a, &d),
                refdot(&a, &e),
            ]) {
                assert!(
                    (got - want).abs() < 3e-2 * len as f32,
                    "{got} vs {want} len={len}"
                );
            }
            let mut acc = vec![0.5f32; len];
            let mut acc_ref = acc.clone();
            mad_bf16(&mut acc, &a, 1.7);
            for (r, v) in acc_ref.iter_mut().zip(&a) {
                *r += v.to_f32() * 1.7;
            }
            for (g, w) in acc.iter().zip(&acc_ref) {
                assert!((g - w).abs() < 1e-2, "mad {g} vs {w} len={len}");
            }
        }
    }
}

// P.V with register-held accumulator chunks: the mad loop's acc ldp/stp per 8 values
// is ~30% of decode attention cycles. dv is processed in 64-f32 chunks (16 vregs of
// accumulator) held across the whole kv span; v re-reads are L1-resident tile hits.
#[target_feature(enable = "fp16")]
#[allow(clippy::too_many_arguments)]
unsafe fn pv_chunk_f16(
    acc: *mut f32,
    cn: usize,
    v_base: *const u16,
    v_stride: usize,
    p: *const f32,
    kv_n: usize,
) {
    use core::arch::aarch64::*;
    let nv = cn / 4;
    let mut a = [vdupq_n_f32(0.0); 16];
    for (x, ax) in a.iter_mut().enumerate().take(nv) {
        *ax = vld1q_f32(acc.add(x * 4));
    }
    for kv in 0..kv_n {
        let pv = vdupq_n_f32(*p.add(kv));
        let vp = v_base.add(kv * v_stride);
        let mut x = 0;
        while x + 2 <= nv {
            let vh = vreinterpretq_f16_u16(vld1q_u16(vp.add(x * 4)));
            a[x] = vfmaq_f32(a[x], vcvt_f32_f16(vget_low_f16(vh)), pv);
            a[x + 1] = vfmaq_f32(a[x + 1], vcvt_high_f32_f16(vh), pv);
            x += 2;
        }
    }
    for (x, ax) in a.iter().enumerate().take(nv) {
        vst1q_f32(acc.add(x * 4), *ax);
    }
}

#[inline(always)]
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
    if !fp16_fast() || !dv.is_multiple_of(8) || dv > 256 || be <= bs {
        return false;
    }
    // rows must be contiguous with a fixed stride for the chunk kernel's v walk
    let v0 = v_row_of(bs);
    let v_stride = if be > bs + 1 {
        v_row_of(bs + 1) - v0
    } else {
        dv
    };
    for kv in bs + 1..be {
        if v_row_of(kv) != v0 + (kv - bs) * v_stride {
            return false;
        }
    }
    let kv_n = be - bs;
    unsafe {
        for j in 0..group {
            let row_acc = vkq.add(j * vkq_stride);
            let p = p_tile.as_ptr().add(j * tile_stride);
            let mut c0 = 0;
            while c0 < dv {
                let cn = (dv - c0).min(64);
                pv_chunk_f16(
                    row_acc.add(c0),
                    cn,
                    v_data.as_ptr().add(v0 + c0) as *const u16,
                    v_stride,
                    p,
                    kv_n,
                );
                c0 += cn;
            }
        }
    }
    true
}
