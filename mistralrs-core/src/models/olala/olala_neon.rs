//! aarch64 NEON inner loops for the Olala Mamba3-MIMO recurrence.
//!
//! The state update and readout dominate the per-head cost. These routines use
//! `vfmaq_f32`/`vmulq_f32` over four-wide f32 vectors, mirroring the patterns in
//! `attention/backends/cpu/neon.rs` but in f32.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

/// Per-value-index `p` recurrence: for every state index `d`,
///   kv[d,p] = sum_r k[r,d] * v[p,r]
///   s'[p,d] = alpha*s[p,d] + beta*kv[d,p] + gamma*curr_kv[d,p]
///   qs[r,p] = sum_d s'[p,d] * q[r,d]
/// `s` is a head slice of `(D_V, D_QK)`; `k`/`q` are `(R,D_QK)`; `v_state` is
/// `(R,D_V)`; `v_proj` is `(D_V,R)`.
///
/// Returns the per-rank readout sum for this `p`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mamba_state_scan_p(
    s: &mut [f32],
    k_state: &[f32],
    v_state: &[f32],
    k_b: &[f32],
    v_proj: &[f32],
    q_b: &[f32],
    alpha: f32,
    beta: f32,
    gamma: f32,
    p: usize,
    d_v: usize,
    d_qk: usize,
    r: usize,
) -> [f32; 4] {
    let alpha_v = vdupq_n_f32(alpha);
    let beta_v = vdupq_n_f32(beta);
    let gamma_v = vdupq_n_f32(gamma);
    let mut qs = [0f32; 4];
    let mut acc_scratch = [[0f32; 128]; 4];
    let s_row = &mut s[p * d_qk..(p + 1) * d_qk];
    // v_state is (r, d_v): element (r, p) lives at r*d_v + p.
    let lanes = [
        v_state[p],
        v_state[d_v + p],
        v_state[2 * d_v + p],
        v_state[3 * d_v + p],
    ];
    let vp = &v_proj[p * r..(p + 1) * r];
    let vp_lanes = [vp[0], vp[1], vp[2], vp[3]];
    let mut d = 0;
    while d + 4 <= d_qk {
        let mut prev = vdupq_n_f32(0.0);
        let mut curr = vdupq_n_f32(0.0);
        for rr in 0..r {
            let kb = vld1q_f32(k_b.as_ptr().add(rr * d_qk + d));
            let ks = vld1q_f32(k_state.as_ptr().add(rr * d_qk + d));
            curr = vaddq_f32(curr, vmulq_n_f32(kb, vp_lanes[rr]));
            prev = vaddq_f32(prev, vmulq_n_f32(ks, lanes[rr]));
        }
        let old = vld1q_f32(s_row.as_ptr().add(d));
        let sv = vaddq_f32(
            vaddq_f32(vmulq_f32(old, alpha_v), vmulq_f32(prev, beta_v)),
            vmulq_f32(curr, gamma_v),
        );
        vst1q_f32(s_row.as_mut_ptr().add(d), sv);
        let q0 = vld1q_f32(q_b.as_ptr().add(d));
        let q1 = vld1q_f32(q_b.as_ptr().add(d_qk + d));
        let q2 = vld1q_f32(q_b.as_ptr().add(2 * d_qk + d));
        let q3 = vld1q_f32(q_b.as_ptr().add(3 * d_qk + d));
        vst1q_f32(acc_scratch[0].as_mut_ptr().add(d), vmulq_f32(sv, q0));
        vst1q_f32(acc_scratch[1].as_mut_ptr().add(d), vmulq_f32(sv, q1));
        vst1q_f32(acc_scratch[2].as_mut_ptr().add(d), vmulq_f32(sv, q2));
        vst1q_f32(acc_scratch[3].as_mut_ptr().add(d), vmulq_f32(sv, q3));
        d += 4;
    }
    // Sequential reduction matches the per-step scalar order closely.
    for dd in 0..d_qk {
        qs[0] += acc_scratch[0][dd];
        qs[1] += acc_scratch[1][dd];
        qs[2] += acc_scratch[2][dd];
        qs[3] += acc_scratch[3][dd];
    }
    qs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_p_matches_scalar() {
        let (d_v, d_qk, r) = (8usize, 16usize, 4usize);
        let mk = |len: usize, off: f32| -> Vec<f32> {
            (0..len).map(|i| ((i as f32 + off) * 0.37).sin()).collect()
        };
        let s = mk(d_v * d_qk, 8.0);
        let k_state = mk(r * d_qk, 9.0);
        let v_state = mk(r * d_v, 10.0);
        let k_b = mk(r * d_qk, 3.0);
        let v_proj = mk(d_v * r, 4.0);
        let q_b = mk(r * d_qk, 5.0);
        let (alpha, beta, gamma) = (0.9f32, 0.05, 0.1);
        let mut s_neon = s.clone();
        let mut s_scalar = s.clone();
        let p = 0usize;
        let got = unsafe {
            mamba_state_scan_p(
                &mut s_neon,
                &k_state,
                &v_state,
                &k_b,
                &v_proj,
                &q_b,
                alpha,
                beta,
                gamma,
                p,
                d_v,
                d_qk,
                r,
            )
        };
        let mut want = [0f32; 4];
        for d in 0..d_qk {
            let mut prev = 0f32;
            let mut curr = 0f32;
            for rr in 0..r {
                prev += k_state[rr * d_qk + d] * v_state[rr * d_v + p];
                curr += k_b[rr * d_qk + d] * v_proj[p * r + rr];
            }
            let sv = alpha * s_scalar[p * d_qk + d] + beta * prev + gamma * curr;
            s_scalar[p * d_qk + d] = sv;
            for rr in 0..r {
                want[rr] += sv * q_b[rr * d_qk + d];
            }
        }
        assert_eq!(&s_neon[..d_qk], &s_scalar[..d_qk]);
        for rr in 0..r {
            assert!(
                (got[rr] - want[rr]).abs() < 1e-4,
                "q{rr}: {} vs {}",
                got[rr],
                want[rr]
            );
        }
    }
}
