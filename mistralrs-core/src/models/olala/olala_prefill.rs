//! Closed-form (chunked) Mamba3-MIMO prefill for Olala.
//!
//! Implements SPEC-mamba3-mimo.md section 5: instead of T serial recurrent steps,
//! project all tokens at once and evaluate the linear recurrence as a lower-triangular
//! decay-weighted sum. Equivalent to the iterative decode path (unit-tested below).

use candle_core::{Result, Tensor, D};

const PI: f32 = std::f32::consts::PI;
const A_FLOOR: f32 = 1e-4;

fn softplus(x: &Tensor) -> Result<Tensor> {
    let neg_abs = x.abs()?.neg()?;
    let soft = neg_abs.exp()?.affine(1.0, 1.0)?.log()?;
    x.clamp(0.0, f32::MAX)?.add(&soft)
}
fn rms_inv_last(x: &Tensor, eps: f32) -> Result<Tensor> {
    x.sqr()?
        .mean(D::Minus1)?
        .affine(1.0, f64::from(eps))?
        .sqrt()?
        .recip()
}

/// Closed-form Mamba step for a single prefill chunk.
///
/// Inputs are laid out token-major:
///   ps: (T, H, 2*D_v+3), pd: (T, 2*G*R*D_qk + n_ang)
/// Weight/state tensors use the same layouts as `olala.rs`.
/// Returns y: (T, H*D_v) plus the closing states.
#[allow(clippy::too_many_arguments)]
pub(crate) fn closed_form_mamba_prefill(
    ps: &Tensor,
    pd: &Tensor,
    norm_weight: &Tensor,
    bias: &Tensor,
    mxz: &Tensor,
    mimo_o: &Tensor,
    dt_bias: &Tensor,
    d_skip: &Tensor,
    s_in: &Tensor,
    k_in: &Tensor,
    v_in: &Tensor,
    ang_in: &Tensor,
    h: usize,
    d_v: usize,
    d_qk: usize,
    r: usize,
    n_ang: usize,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let t = ps.dim(0)?;
    let per_group = r * d_qk;
    let dev = ps.device();

    // --- token-wise coefficients (T,H) ---
    let dt_raw = ps.narrow(2, 2 * d_v, 1)?.squeeze(D::Minus1)?; // (T,H)
    let a_raw = ps.narrow(2, 2 * d_v + 1, 1)?.squeeze(D::Minus1)?;
    let trap_raw = ps.narrow(2, 2 * d_v + 2, 1)?.squeeze(D::Minus1)?;
    let dt_bias_t = dt_bias.t()?; // (1,H)
    let dt = softplus(&dt_raw.broadcast_add(&dt_bias_t)?)?; // (T,H)
    let a_neg = softplus(&a_raw)?.neg()?.clamp(f32::MIN, -A_FLOOR)?;
    let alpha = a_neg.mul(&dt)?.exp()?; // (T,H)
    let trap = sigmoid_1(&trap_raw)?;
    let gamma = trap.mul(&dt)?; // (T,H)
    let cprev = trap.affine(-1.0, 1.0)?.mul(&dt)?;

    // gamma_shifted[h,t] = cprev[h,t+1], zero at the last token
    let mut gs = Tensor::zeros((t, h), candle_core::DType::F32, dev)?;
    if t > 1 {
        gs = gs.slice_assign(&[0..t - 1, 0..h], &cprev.narrow(0, 1, t - 1)?)?;
    }
    let factor = gamma.add(&gs)?; // (T,H)

    // --- angle cumsum (T,n_ang) shared across heads, then per-head phase ---
    let ang_raw = pd.narrow(1, 2 * per_group, n_ang)?; // (T,n_ang)
    let contrib3 = ang_raw
        .tanh()?
        .affine(f64::from(PI), 0.0)?
        .unsqueeze(D::Minus1)?
        .broadcast_mul(&dt.unsqueeze(1)?)?; // (T,n_ang,H)
    let cum = contrib3.cumsum(0)?; // (T,n_ang,H)
    let ang_in3 = ang_in.permute((1, 0))?.unsqueeze(0)?; // (1,n_ang,H)
    let cum = cum.add(&ang_in3.broadcast_as((t, n_ang, h))?)?;
    let cum = cum.sub(
        &cum.affine(1.0 / (2.0 * f64::from(PI)), 0.0)?
            .round()?
            .affine(2.0 * f64::from(PI), 0.0)?,
    )?;
    let cos_a = cum.cos()?; // (T,n_ang,H)
    let sin_a = cum.sin()?;

    // --- B/C groups, norm, bias, rotary -> k_rot/q_rot (T,H,R,DQ) ---
    let bc = pd.narrow(1, 0, 2 * per_group)?.reshape((t, 2 * r, d_qk))?;
    let inv = rms_inv_last(&bc, eps)?.unsqueeze(D::Minus1)?; // (T,2r,1)
    let normed = bc
        .broadcast_mul(&inv)?
        .broadcast_mul(&norm_weight.unsqueeze(0)?)? // (1,2r,dq)
        .reshape((t, 2 * r, d_qk))?;
    let bias3 = bias.unsqueeze(0)?; // (1,H,2r,dq)
    let normed_h = normed.unsqueeze(1)?.broadcast_as((t, h, 2 * r, d_qk))?;
    let k_biased = normed_h
        .narrow(2, 0, r)?
        .broadcast_add(&bias3.narrow(2, 0, r)?)?; // (T,H,r,dq)
    let q_biased = normed_h
        .narrow(2, r, r)?
        .broadcast_add(&bias3.narrow(2, r, r)?)?;
    let rotary = |x: &Tensor| -> Result<Tensor> {
        let q = n_ang;
        let half = d_qk / 2;
        let g0 = x.narrow(3, 0, q)?;
        let g1 = x.narrow(3, q, q)?;
        let g2 = x.narrow(3, half, q)?;
        let g3 = x.narrow(3, half + q, d_qk - half - q)?;
        let c = cos_a
            .permute((0, 2, 1))?
            .unsqueeze(2)?
            .broadcast_as((t, h, r, q))?;
        let s = sin_a
            .permute((0, 2, 1))?
            .unsqueeze(2)?
            .broadcast_as((t, h, r, q))?;
        let n0 = g0.broadcast_mul(&c)?.sub(&g2.broadcast_mul(&s)?)?;
        let n2 = g0.broadcast_mul(&s)?.add(&g2.broadcast_mul(&c)?)?;
        Tensor::cat(&[n0, g1, n2, g3], 3)
    };
    // x is (T,2r,dq) but rotary needs (T,H,r,dq); broadcast bias already gave (T,H,r,dq).
    let k_rot = rotary(&k_biased)?; // (T,H,r,dq)
    let q_rot = rotary(&q_biased)?;

    // --- v/z projections (T,H,dv,r) ---
    let z_in = ps.narrow(2, 0, d_v)?; // (T,H,dv)
    let x_in = ps.narrow(2, d_v, d_v)?;
    let mxz3 = mxz.unsqueeze(0)?; // (1,H,dv,2r)
    let v = x_in
        .unsqueeze(D::Minus1)?
        .broadcast_mul(&mxz3.narrow(3, 0, r)?)?;
    let z = z_in
        .unsqueeze(D::Minus1)?
        .broadcast_mul(&mxz3.narrow(3, r, r)?)?;

    // --- trapezoid state-in correction ---
    let prev_kv_in = k_in
        .permute((0, 2, 1))?
        .contiguous()?
        .matmul(&v_in.contiguous()?)? // (H,dq,dv)
        .permute((0, 2, 1))?
        .contiguous()?; // (H,dv,dq)
    let gs0 = cprev.narrow(0, 0, 1)?.reshape((h,))?; // coeff_prev at first token
    let s_eff = s_in.add(&prev_kv_in.broadcast_mul(&gs0.reshape((h, 1, 1))?)?)?;

    // --- decay matrix W[u,t,h] = exp(cum_a[t]-cum_a[u]) for u<=t ---
    let log_alpha = alpha.log()?; // (T,H)
    let cum_a = log_alpha.cumsum(0)?; // (T,H)
                                      // (T,T,H) lower triangular
    let cu = cum_a.unsqueeze(0)?; // (1,T,H)
    let ct = cum_a.unsqueeze(1)?; // (T,1,H)
    let mut w = ct.broadcast_sub(&cu)?.exp()?; // (T,T,H)
    let tri = Tensor::tril2(t, candle_core::DType::F32, dev)?
        .reshape((t, t, 1))?
        .broadcast_as((t, t, h))?;
    w = w.mul(&tri)?;
    let wf = w.mul(&factor.unsqueeze(0)?.broadcast_as((t, t, h))?)?;

    // --- kv contraction (T,H,dv,dq) ---
    let kr = k_rot.permute((0, 1, 3, 2))?.contiguous()?; // (T,H,dq,r)
    let vv = v.permute((0, 1, 3, 2))?.contiguous()?; // (T,H,r,dv)
    let kv = kr.matmul(&vv)?.permute((0, 1, 3, 2))?.contiguous()?; // (T,H,dv,dq)

    // --- state: state[t] = sum_u wf[u,t]*kv[u] + exp(cum_a[t])*s_eff ---
    let kv_u = kv.unsqueeze(0)?; // (1,T,H,dv,dq)
    let wf4 = wf.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?; // (t,u,H,1,1)
    let state = kv_u.broadcast_mul(&wf4)?.sum(1)?; // (t,H,dv,dq)
    let decay = cum_a.exp()?; // (T,H)
    let s_eff_b = s_eff.unsqueeze(0)?;
    let state =
        state.add(&s_eff_b.broadcast_mul(&decay.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)?)?; // (T,H,dv,dq)

    // --- readout with diagonal correction ---
    let q_for = q_rot.permute((0, 1, 3, 2))?.contiguous()?; // (T,H,dq,r)
    let mut qstate = state.matmul(&q_for)?; // (T,H,dv,r)
    let kv_diag = kv.matmul(&q_for)?; // (T,H,dv,r)
    qstate =
        qstate.sub(&kv_diag.broadcast_mul(&gs.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)?)?;

    // --- output ---
    let d = d_skip.unsqueeze(0)?.unsqueeze(D::Minus1)?; // (1,H,1,1)
    let o = qstate.broadcast_add(&v.broadcast_mul(&d)?)?;
    let o = o.mul(&z.silu()?)?;
    let mo = mimo_o.unsqueeze(0)?; // (1,H,dv,r)
    let y = o
        .broadcast_mul(&mo)?
        .sum(D::Minus1)? // (T,H,dv)
        .reshape((t, h * d_v))?;

    // closing state: last token
    let s_last = state.narrow(0, t - 1, 1)?.squeeze(0)?;
    let k_last = k_rot.narrow(0, t - 1, 1)?.squeeze(0)?;
    let v_last = v
        .narrow(0, t - 1, 1)?
        .squeeze(0)?
        .permute((0, 2, 1))?
        .contiguous()?;
    let ang_last = cum
        .narrow(0, t - 1, 1)?
        .squeeze(0)?
        .permute((1, 0))?
        .contiguous()?; // (H,n_ang)
    Ok((s_last, k_last, v_last, ang_last, y))
}

fn sigmoid_1(x: &Tensor) -> Result<Tensor> {
    x.neg()?.exp()?.affine(1.0, 1.0)?.recip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::olala::olala_cpu::tensor_mamba_step;
    use candle_core::{DType, Device, Tensor};

    fn gen(len: usize, off: f32, dev: &Device) -> Result<Tensor> {
        let v = (0..len)
            .map(|i| ((i as f32 + off) * 0.37).sin() * 0.5)
            .collect::<Vec<_>>();
        Tensor::from_vec(v, len, dev)
    }
    fn flat(t: &Tensor) -> Result<Vec<f32>> {
        t.reshape((t.elem_count(),))?.contiguous()?.to_vec1::<f32>()
    }
    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn closed_form_matches_iterative() -> Result<()> {
        let dev = &Device::Cpu;
        let (t, h, d_v, d_qk, r, n_ang) = (4usize, 2usize, 8usize, 16usize, 2usize, 4usize);
        let eps = 1e-6f32;
        let per = 2 * d_v + 3;
        let ps = gen(t * h * per, 0.0, dev)?.reshape((t, h, per))?;
        let pd = gen(t * (2 * r * d_qk + n_ang), 1.0, dev)?.reshape((t, 2 * r * d_qk + n_ang))?;
        let norm_weight = gen(2 * r * d_qk, 2.0, dev)?.reshape((2 * r, d_qk))?;
        let bias = gen(h * 2 * r * d_qk, 3.0, dev)?.reshape((h, 2 * r, d_qk))?;
        let mxz = gen(h * d_v * 2 * r, 4.0, dev)?.reshape((h, d_v, 2 * r))?;
        let mimo_o = gen(h * d_v * r, 5.0, dev)?.reshape((h, d_v, r))?;
        let dt_bias = gen(h, 6.0, dev)?.reshape((h, 1))?;
        let d_skip = gen(h, 7.0, dev)?.reshape((h, 1))?;
        let s = gen(h * d_v * d_qk, 8.0, dev)?.reshape((h, d_v, d_qk))?;
        let k = gen(h * r * d_qk, 9.0, dev)?.reshape((h, r, d_qk))?;
        let v = gen(h * r * d_v, 10.0, dev)?.reshape((h, r, d_v))?;
        let ang = gen(h * n_ang, 11.0, dev)?.reshape((h, n_ang))?;

        // iterative reference
        let (mut rs, mut rk, mut rv, mut ra) = (s.clone(), k.clone(), v.clone(), ang.clone());
        let mut ys = Vec::with_capacity(t);
        for i in 0..t {
            let ps_i = ps.narrow(0, i, 1)?.squeeze(0)?;
            let pd_i = pd.narrow(0, i, 1)?;
            let (ns, nk, nv, na, y) = tensor_mamba_step(
                &ps_i,
                &pd_i,
                &norm_weight,
                &bias,
                &mxz,
                &mimo_o,
                &dt_bias,
                &d_skip,
                &rs,
                &rk,
                &rv,
                &ra,
                h,
                d_v,
                d_qk,
                r,
                n_ang,
                eps,
            )?;
            rs = ns;
            rk = nk;
            rv = nv;
            ra = na;
            ys.push(y);
        }
        let ref_y = Tensor::cat(&ys, 0)?;

        // closed form
        let (cs, ck, cv, ca, cy) = closed_form_mamba_prefill(
            &ps,
            &pd,
            &norm_weight,
            &bias,
            &mxz,
            &mimo_o,
            &dt_bias,
            &d_skip,
            &s,
            &k,
            &v,
            &ang,
            h,
            d_v,
            d_qk,
            r,
            n_ang,
            eps,
        )?;

        let diffs: Vec<_> = [
            ("y", flat(&ref_y)?, flat(&cy)?),
            ("s", flat(&rs)?, flat(&cs)?),
            ("k", flat(&rk)?, flat(&ck)?),
            ("v", flat(&rv)?, flat(&cv)?),
            ("ang", flat(&ra)?, flat(&ca)?),
        ]
        .into_iter()
        .map(|(name, a, b)| (name, max_diff(&a, &b)))
        .collect();
        for (name, d) in &diffs {
            println!("{name}: max diff {d:e}");
        }
        for (name, d) in diffs {
            assert!(d < 1e-4, "{name} max diff {d}");
        }
        Ok(())
    }
}
