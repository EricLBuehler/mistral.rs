//! Fused scalar CPU kernel for the Olala Mamba3-MIMO recurrence.
//!
//! Mirrors the tensor-op recurrence in `olala.rs` exactly, operating on flat f32
//! slices with `barrier_pool` parallelisation over heads. Kept in a separate
//! module so it can be unit-tested against the tensor path without loading a model.

use candle_core::{Device, Result, Tensor};

const PI: f32 = std::f32::consts::PI;
const A_FLOOR: f32 = 1e-4;

#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_mamba_step(
    ps: &[f32],
    pd: &[f32],
    norm_weight: &[f32],
    bias: &[f32],
    mxz: &[f32],
    mimo_o: &[f32],
    dt_bias: &[f32],
    d_skip: &[f32],
    s_state: &[f32],
    k_state: &[f32],
    v_state: &[f32],
    ang_state: &[f32],
    h: usize,
    d_v: usize,
    d_qk: usize,
    r: usize,
    n_ang: usize,
    eps: f32,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let per_head = 2 * d_v + 3;
    let per_group = r * d_qk;
    let mut s_new = vec![0f32; s_state.len()];
    let mut k_new = vec![0f32; k_state.len()];
    let mut v_new = vec![0f32; v_state.len()];
    let mut ang_new = vec![0f32; ang_state.len()];
    let mut y = vec![0f32; h * d_v];
    let (sp, kp, vp, ap, yp) = (
        s_new.as_mut_ptr() as usize,
        k_new.as_mut_ptr() as usize,
        v_new.as_mut_ptr() as usize,
        ang_new.as_mut_ptr() as usize,
        y.as_mut_ptr() as usize,
    );
    let (s_len, k_len, v_len, a_len, y_len) = (
        s_new.len(),
        k_new.len(),
        v_new.len(),
        ang_new.len(),
        y.len(),
    );
    let (ps_s, pd_s, nw_s, b_s, mx_s, mo_s, dtb_s, dsk_s, ss, ks, vs, as_) = (
        ps,
        pd,
        norm_weight,
        bias,
        mxz,
        mimo_o,
        dt_bias,
        d_skip,
        s_state,
        k_state,
        v_state,
        ang_state,
    );

    candle_core::utils::barrier_pool().execute_chunked(h, |range| {
        for head in range {
            let s_new = unsafe { std::slice::from_raw_parts_mut(sp as *mut f32, s_len) };
            let k_new = unsafe { std::slice::from_raw_parts_mut(kp as *mut f32, k_len) };
            let v_new = unsafe { std::slice::from_raw_parts_mut(vp as *mut f32, v_len) };
            let ang_new = unsafe { std::slice::from_raw_parts_mut(ap as *mut f32, a_len) };
            let y = unsafe { std::slice::from_raw_parts_mut(yp as *mut f32, y_len) };

            let ps_h = &ps_s[head * per_head..(head + 1) * per_head];
            let softplus = |v: f32| -> f32 { v.max(0.0) + (1.0 + (-v.abs()).exp()).ln() };
            let dt = softplus(ps_h[2 * d_v] + dtb_s[head]);
            let a_neg = (-softplus(ps_h[2 * d_v + 1])).min(-A_FLOOR);
            let alpha = (a_neg * dt).exp();
            let trap = 1.0 / (1.0 + (-ps_h[2 * d_v + 2]).exp());
            let beta = (1.0 - trap) * dt * alpha;
            let gamma = trap * dt;

            let a_h = &as_[head * n_ang..(head + 1) * n_ang];
            let ang_o = &mut ang_new[head * n_ang..(head + 1) * n_ang];
            let mut cs = [0f32; 32];
            let mut sn = [0f32; 32];
            for i in 0..n_ang {
                let raw = pd_s[2 * per_group + i];
                let a = a_h[i] + raw.tanh() * PI * dt;
                let a = a - (a / (2.0 * PI)).round() * (2.0 * PI);
                ang_o[i] = a;
                cs[i] = a.cos();
                sn[i] = a.sin();
            }

            let mut kb = vec![0f32; r * d_qk];
            let mut qb = vec![0f32; r * d_qk];
            for g in 0..2 * r {
                let src = &pd_s[g * d_qk..(g + 1) * d_qk];
                let ssq: f32 = src.iter().map(|v| v * v).sum();
                let scale = 1.0 / (ssq / d_qk as f32 + eps).sqrt();
                let dst = if g < r { &mut kb } else { &mut qb };
                let gr = if g < r { g } else { g - r };
                for d in 0..d_qk {
                    let v = src[d] * scale * nw_s[g * d_qk + d]
                        + b_s[head * 2 * r * d_qk + g * d_qk + d];
                    if d < n_ang {
                        let d2 = d + d_qk / 2;
                        let v2 = src[d2] * scale * nw_s[g * d_qk + d2]
                            + b_s[head * 2 * r * d_qk + g * d_qk + d2];
                        dst[gr * d_qk + d] = v * cs[d] - v2 * sn[d];
                        dst[gr * d_qk + d2] = v * sn[d] + v2 * cs[d];
                        continue;
                    }
                    if d >= d_qk / 2 && d < d_qk / 2 + n_ang {
                        continue;
                    }
                    dst[gr * d_qk + d] = v;
                }
            }

            // mxz is stored (h, d_v, 2r); mimo_o is (h, d_v, r).
            let mut vp = vec![0f32; d_v * r];
            let mut zp = vec![0f32; d_v * r];
            for rr in 0..r {
                for pp in 0..d_v {
                    vp[pp * r + rr] = ps_h[d_v + pp] * mx_s[head * d_v * 2 * r + pp * 2 * r + rr];
                    zp[pp * r + rr] = ps_h[pp] * mx_s[head * d_v * 2 * r + pp * 2 * r + r + rr];
                }
            }

            let s_h = &ss[head * d_v * d_qk..(head + 1) * d_v * d_qk];
            let ks_h = &ks[head * r * d_qk..(head + 1) * r * d_qk];
            let vs_h = &vs[head * r * d_v..(head + 1) * r * d_v];
            let so = &mut s_new[head * d_v * d_qk..(head + 1) * d_v * d_qk];
            let ko = &mut k_new[head * r * d_qk..(head + 1) * r * d_qk];
            let vo = &mut v_new[head * r * d_v..(head + 1) * r * d_v];

            let mut qstate = [[0f32; 4]; 64];
            #[cfg(target_arch = "aarch64")]
            so.copy_from_slice(s_h);
            for pp in 0..d_v {
                #[cfg(target_arch = "aarch64")]
                let qs = unsafe {
                    crate::models::olala::olala_neon::mamba_state_scan_p(
                        so, ks_h, vs_h, &kb, &vp, &qb, alpha, beta, gamma, pp, d_v, d_qk, r,
                    )
                };
                #[cfg(not(target_arch = "aarch64"))]
                let qs = {
                    let mut acc = [0f32; 4];
                    for d in 0..d_qk {
                        let mut prev = 0f32;
                        let mut curr = 0f32;
                        for rr in 0..r {
                            prev += ks_h[rr * d_qk + d] * vs_h[rr * d_v + pp];
                            curr += kb[rr * d_qk + d] * vp[pp * r + rr];
                        }
                        let sv = alpha * s_h[pp * d_qk + d] + beta * prev + gamma * curr;
                        so[pp * d_qk + d] = sv;
                        for rr in 0..r {
                            acc[rr] += sv * qb[rr * d_qk + d];
                        }
                    }
                    acc
                };
                qstate[pp] = qs;
            }
            for pp in 0..d_v {
                let qs = qstate[pp];
                let mut acc = 0f32;
                for rr in 0..r {
                    let o = qs[rr] + dsk_s[head] * vp[pp * r + rr];
                    let z = zp[pp * r + rr];
                    let silu = z / (1.0 + (-z).exp());
                    acc += o * silu * mo_s[head * d_v * r + pp * r + rr];
                }
                y[head * d_v + pp] = acc;
            }
            ko.copy_from_slice(&kb);
            for rr in 0..r {
                for pp in 0..d_v {
                    vo[rr * d_v + pp] = vp[pp * r + rr];
                }
            }
        }
    });

    Ok((
        Tensor::from_vec(s_new, (h, d_v, d_qk), device)?,
        Tensor::from_vec(k_new, (h, r, d_qk), device)?,
        Tensor::from_vec(v_new, (h, r, d_v), device)?,
        Tensor::from_vec(ang_new, (h, n_ang), device)?,
        Tensor::from_vec(y, (h * d_v,), device)?,
    ))
}

use candle_core::D;

#[allow(clippy::too_many_arguments)]
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.neg()?.exp()?.affine(1.0, 1.0)?.recip()
}
fn softplus(x: &Tensor) -> Result<Tensor> {
    let neg_abs = x.abs()?.neg()?;
    let soft = neg_abs.exp()?.affine(1.0, 1.0)?.log()?;
    x.clamp(0.0, f32::MAX)?.add(&soft)
}
fn rms_inv(x: &Tensor, eps: f32) -> Result<Tensor> {
    x.sqr()?
        .mean(D::Minus1)?
        .affine(1.0, f64::from(eps))?
        .sqrt()?
        .recip()
}
fn wrap_angle(a: &Tensor) -> Result<Tensor> {
    let turns = a.affine(1.0 / (2.0 * f64::from(PI)), 0.0)?.round()?;
    a.sub(&turns.affine(2.0 * f64::from(PI), 0.0)?)
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

pub(crate) fn tensor_mamba_step(
    ps: &Tensor,
    pd: &Tensor,
    norm_weight: &Tensor,
    bias: &Tensor,
    mxz: &Tensor,
    mimo_o: &Tensor,
    dt_bias: &Tensor,
    d_skip: &Tensor,
    s: &Tensor,
    k: &Tensor,
    v: &Tensor,
    ang: &Tensor,
    h: usize,
    d_v: usize,
    d_qk: usize,
    r: usize,
    n_ang: usize,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let z_in = ps.narrow(1, 0, d_v)?;
    let x_in = ps.narrow(1, d_v, d_v)?;
    let dt_raw = ps.narrow(1, 2 * d_v, 1)?;
    let a_raw = ps.narrow(1, 2 * d_v + 1, 1)?;
    let trap_raw = ps.narrow(1, 2 * d_v + 2, 1)?;
    let bc = pd.narrow(1, 0, 2 * r * d_qk)?.reshape((2 * r, d_qk))?;
    let ang_raw = pd.narrow(1, 2 * r * d_qk, n_ang)?.reshape((n_ang,))?;

    let inv = rms_inv(&bc, eps)?.reshape((2 * r, 1))?;
    let normed = bc
        .broadcast_mul(&inv)?
        .mul(norm_weight)?
        .reshape((1, 2 * r, d_qk))?;
    let dt = softplus(&dt_raw.add(dt_bias)?)?;
    let a_neg = softplus(&a_raw)?.neg()?.clamp(f32::MIN, -A_FLOOR)?;
    let alpha = a_neg.mul(&dt)?.exp()?;
    let trap = sigmoid(&trap_raw)?;
    let beta = trap.affine(-1.0, 1.0)?.mul(&dt)?.mul(&alpha)?;
    let gamma = trap.mul(&dt)?;
    let delta_ang = ang_raw
        .reshape((1, n_ang))?
        .tanh()?
        .affine(f64::from(PI), 0.0)?
        .broadcast_mul(&dt)?;
    let ang_new = wrap_angle(&ang.add(&delta_ang)?)?;
    let cos_a = ang_new.cos()?.reshape((h, 1, n_ang))?;
    let sin_a = ang_new.sin()?.reshape((h, 1, n_ang))?;

    let mut k_b = normed
        .narrow(1, 0, r)?
        .broadcast_add(&bias.narrow(1, 0, r)?)?;
    let mut q_b = normed
        .narrow(1, r, r)?
        .broadcast_add(&bias.narrow(1, r, r)?)?;
    let quarter = n_ang;
    let half = d_qk / 2;
    for mat in [&mut k_b, &mut q_b] {
        let g0 = mat.narrow(2, 0, quarter)?;
        let g1 = mat.narrow(2, quarter, quarter)?;
        let g2 = mat.narrow(2, half, quarter)?;
        let g3 = mat.narrow(2, half + quarter, d_qk - half - quarter)?;
        let n0 = g0.broadcast_mul(&cos_a)?.sub(&g2.broadcast_mul(&sin_a)?)?;
        let n2 = g0.broadcast_mul(&sin_a)?.add(&g2.broadcast_mul(&cos_a)?)?;
        *mat = Tensor::cat(&[n0, g1, n2, g3], 2)?.contiguous()?;
    }
    let v_proj = x_in
        .reshape((h, d_v, 1))?
        .broadcast_mul(&mxz.narrow(2, 0, r)?)?;
    let z_proj = z_in
        .reshape((h, d_v, 1))?
        .broadcast_mul(&mxz.narrow(2, r, r)?)?;
    let prev_kv = k
        .permute((0, 2, 1))?
        .contiguous()?
        .matmul(&v.contiguous()?)?
        .permute((0, 2, 1))?
        .contiguous()?;
    let curr_kv = k_b
        .permute((0, 2, 1))?
        .contiguous()?
        .matmul(&v_proj.permute((0, 2, 1))?.contiguous()?)?
        .permute((0, 2, 1))?
        .contiguous()?;
    let alpha_b = alpha.reshape((h, 1, 1))?;
    let beta_b = beta.reshape((h, 1, 1))?;
    let gamma_b = gamma.reshape((h, 1, 1))?;
    let s_new = s
        .broadcast_mul(&alpha_b)?
        .add(&prev_kv.broadcast_mul(&beta_b)?)?
        .add(&curr_kv.broadcast_mul(&gamma_b)?)?
        .contiguous()?;
    let qstate = s_new.matmul(&q_b.permute((0, 2, 1))?.contiguous()?)?;
    let o = qstate.add(&v_proj.broadcast_mul(&d_skip.reshape((h, 1, 1))?)?)?;
    let o = o.mul(&z_proj.silu()?)?;
    let y = o.mul(mimo_o)?.sum(D::Minus1)?.reshape((h * d_v,))?;
    Ok((
        s_new,
        k_b,
        v_proj.permute((0, 2, 1))?.contiguous()?,
        ang_new,
        y,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};

    #[test]
    fn fused_mamba_step_matches_tensor_path() -> Result<()> {
        let dev = &Device::Cpu;
        let (h, d_v, d_qk, r, n_ang) = (48usize, 64usize, 128usize, 4usize, 32usize);
        let eps = 1e-6f32;
        let per = 2 * d_v + 3;
        let gen = |len: usize, off: f32| -> Result<Tensor> {
            let v = (0..len)
                .map(|i| ((i as f32 + off) * 0.37).sin() * 0.5)
                .collect::<Vec<_>>();
            Tensor::from_vec(v, len, dev)
        };
        let ps = gen(h * per, 0.0)?.reshape((h, per))?;
        let pd = gen(2 * r * d_qk + n_ang, 1.0)?.reshape((1, 2 * r * d_qk + n_ang))?;
        let norm_weight = gen(2 * r * d_qk, 2.0)?.reshape((2 * r, d_qk))?;
        let bias = gen(h * 2 * r * d_qk, 3.0)?.reshape((h, 2 * r, d_qk))?;
        // mxz is (h, d_v, 2r), mimo_o is (h, d_v, r)
        let mxz = gen(h * d_v * 2 * r, 4.0)?.reshape((h, d_v, 2 * r))?;
        let mimo_o = gen(h * d_v * r, 5.0)?.reshape((h, d_v, r))?;
        let dt_bias = gen(h, 6.0)?.reshape((h, 1))?;
        let d_skip = gen(h, 7.0)?.reshape((h, 1))?;
        let s = gen(h * d_v * d_qk, 8.0)?.reshape((h, d_v, d_qk))?;
        let k = gen(h * r * d_qk, 9.0)?.reshape((h, r, d_qk))?;
        let v = gen(h * r * d_v, 10.0)?.reshape((h, r, d_v))?;
        let ang = gen(h * n_ang, 11.0)?.reshape((h, n_ang))?;

        let (ts, tk, tv, ta, ty) = tensor_mamba_step(
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
        let (fs, fk, fv, fa, fy) = fused_mamba_step(
            &flat(&ps)?,
            &flat(&pd)?,
            &flat(&norm_weight)?,
            &flat(&bias)?,
            &flat(&mxz)?,
            &flat(&mimo_o)?,
            &flat(&dt_bias)?,
            &flat(&d_skip)?,
            &flat(&s)?,
            &flat(&k)?,
            &flat(&v)?,
            &flat(&ang)?,
            h,
            d_v,
            d_qk,
            r,
            n_ang,
            eps,
            dev,
        )?;

        for (name, a, b) in [
            ("y", flat(&ty)?, flat(&fy)?),
            ("s", flat(&ts)?, flat(&fs)?),
            ("k", flat(&tk)?, flat(&fk)?),
            ("v", flat(&tv)?, flat(&fv)?),
            ("ang", flat(&ta)?, flat(&fa)?),
        ] {
            let d = max_diff(&a, &b);
            println!("{name}: max diff {d:e}");
            assert!(d < 1e-4, "{name} max diff {d}");
        }
        Ok(())
    }
}
