#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Context, Device, Result, Storage, Tensor, WithDType};

use half::bf16;
use half::f16;
use rayon::prelude::*;
use rayon::ThreadPool;

use std::sync::LazyLock;
use std::{f32, iter::Sum};

// Optional hardware prefetch for ping‑pong style scheduling.
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm_prefetch;

// --- portable wrapper around architecture‑specific prefetch -------------
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch(addr: *const u8) {
    _mm_prefetch(addr as *const i8, core::arch::x86_64::_MM_HINT_T0);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch(addr: *const u8) {
    core::arch::asm!(
        "prfm pldl1keep, [{0}]",
        in(reg) addr,
        options(readonly, nostack, preserves_flags)
    );
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn prefetch(_addr: *const u8) {}
// -----------------------------------------------------------------------

use crate::attention::SdpaParams;

#[cfg(target_os = "macos")]
/// Elevate the thread QoS so macOS prefers running it on Performance (P) cores.
unsafe fn set_thread_affinity() {
    // USER_INTERACTIVE has the highest scheduling priority that user code
    // can request and is most likely to be scheduled on P‑cores.
    use libc::{pthread_set_qos_class_self_np, qos_class_t::QOS_CLASS_USER_INTERACTIVE};
    // The second argument is a relative priority within the QoS class (0 = default).
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

#[cfg(not(target_os = "macos"))]
#[inline(always)]
unsafe fn set_thread_affinity() {
    // On non‑macOS platforms we currently leave affinity untouched.
}

/// Rayon pool used by the flash‑attention CPU kernels, with a per‑thread
/// start handler that applies our affinity hint exactly once.
static FLASH_ATTN_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .start_handler(|_| unsafe {
            set_thread_affinity();
        })
        .build()
        .expect("Failed to build custom Rayon thread‑pool for flash‑attention")
});

const DOT_CHUNK: usize = 4;

/// Size (in KV positions) processed by each inner‑tile job.
const TILE_KV: usize = 16;

/// Unrolled dot‑product that works purely in `f32`.
#[inline]
fn vec_dot_f32<T: UpcastF32 + Copy>(a: &[T], b: &[T]) -> f32 {
    let mut sum = 0f32;
    let chunks = a.len() / DOT_CHUNK;
    for i in 0..chunks {
        let i_chunk = i * DOT_CHUNK;
        sum += a[i_chunk].to_f32() * b[i_chunk].to_f32()
            + a[i_chunk + 1].to_f32() * b[i_chunk + 1].to_f32()
            + a[i_chunk + 2].to_f32() * b[i_chunk + 2].to_f32()
            + a[i_chunk + 3].to_f32() * b[i_chunk + 3].to_f32();
    }
    for i in (chunks * DOT_CHUNK)..a.len() {
        sum += a[i].to_f32() * b[i].to_f32();
    }
    sum
}

pub(super) trait DowncastF32 {
    fn cast(x: f32) -> Self;
}

impl DowncastF32 for f32 {
    fn cast(x: f32) -> Self {
        x
    }
}

impl DowncastF32 for f16 {
    fn cast(x: f32) -> Self {
        f16::from_f32(x)
    }
}

impl DowncastF32 for bf16 {
    fn cast(x: f32) -> Self {
        bf16::from_f32(x)
    }
}

/// Up‑cast helper: convert any supported element type to `f32` once so that the
/// hot kernels can run entirely in `f32` and only down‑cast when writing out.
pub(super) trait UpcastF32 {
    fn to_f32(self) -> f32;
}

impl UpcastF32 for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

impl UpcastF32 for f16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

impl UpcastF32 for bf16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

pub fn run_flash_attn_cpu<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real + DowncastF32 + UpcastF32,
{
    // Inline CPU slice extraction for q, k, v, and optional mask
    let (q_guard, q_layout) = q.storage_and_layout();
    let q_data: &[T] = if let Storage::Cpu(cpu) = &*q_guard {
        let data = cpu.as_slice::<T>().context("Expected CPU storage for q")?;
        &data[q_layout.start_offset()..]
    } else {
        return Err(candle_core::Error::Msg("Expected CPU storage for q".into()));
    };
    let (k_guard, k_layout) = k.storage_and_layout();
    let k_data: &[T] = if let Storage::Cpu(cpu) = &*k_guard {
        let data = cpu.as_slice::<T>().context("Expected CPU storage for k")?;
        &data[k_layout.start_offset()..]
    } else {
        return Err(candle_core::Error::Msg("Expected CPU storage for k".into()));
    };
    let (v_guard, v_layout) = v.storage_and_layout();
    let v_data: &[T] = if let Storage::Cpu(cpu) = &*v_guard {
        let data = cpu.as_slice::<T>().context("Expected CPU storage for v")?;
        &data[v_layout.start_offset()..]
    } else {
        return Err(candle_core::Error::Msg("Expected CPU storage for v".into()));
    };
    let mask_guard = mask.map(|mask| mask.storage_and_layout().0);
    let mask_data: Option<&[T]> = if let Some(mask_guard) = &mask_guard {
        let mask = mask.as_ref().unwrap();

        if let Storage::Cpu(cpu) = &**mask_guard {
            let data = cpu
                .as_slice::<T>()
                .context("Expected CPU storage for mask")?;
            Some(&data[mask.layout().start_offset()..])
        } else {
            return Err(candle_core::Error::Msg(
                "Expected CPU storage for mask".into(),
            ));
        }
    } else {
        None
    };
    // q_guard, k_guard, v_guard, and m_guard (if any) are kept in scope to hold storage alive

    let q_stride = q.stride();
    let k_stride = k.stride();
    let v_stride = v.stride();

    // Fast path for decode: q_len == 1
    if q.shape().dims()[1] == 1 {
        return flash_attn_cpu_single_q(
            q_data,
            k_data,
            v_data,
            mask_data,
            q.shape().dims(),
            k.shape().dims(),
            v.shape().dims(),
            q_stride,
            k_stride,
            v_stride,
            sdpa_params.softmax_scale,
            0.0,
            sdpa_params.softcap.unwrap_or(0.0),
        );
    }

    flash_attn_cpu(
        q_data,
        k_data,
        v_data,
        mask_data,
        q.shape().dims(),
        k.shape().dims(),
        v.shape().dims(),
        q_stride,
        k_stride,
        v_stride,
        sdpa_params.softmax_scale,
        0.0,
        sdpa_params.softcap.unwrap_or(0.0),
    )
}

/// Optimised path for the common decode case: q_len == 1 but kv_len ≫ 1.
/// All intermediate maths is done in `f32`; we down‑cast to `T` exactly once
/// when writing the output buffer.
#[allow(clippy::too_many_arguments)]
fn flash_attn_cpu_single_q<T>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    mask_vec: Option<&[T]>,
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor>
where
    T: WithDType + Copy + UpcastF32 + DowncastF32,
{
    let (b, _qlen, h, d) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let kv_tiles = kv_len.div_ceil(TILE_KV);
    let dv = d;

    // --- head tiling factors and softmax helper -----------------------
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    // final output
    let mut out = vec![T::cast(0.0); b * h * dv];
    // scratch: one tile  ×  heads ×  dv  (flattened)
    let mut scratch = vec![0f32; kv_tiles * h * dv];
    let mut scratch_s = vec![0f32; kv_tiles * h]; // softmax S per tile
    let mut scratch_m = vec![f32::NEG_INFINITY; kv_tiles * h];

    // Store as plain usize so the capture is Sync (raw *mut T is not Sync)
    let scratch_ptr = scratch.as_mut_ptr() as usize;
    let scratch_s_ptr = scratch_s.as_mut_ptr() as usize;
    let scratch_m_ptr = scratch_m.as_mut_ptr() as usize;

    FLASH_ATTN_POOL.install(|| {
        (0..kv_tiles).into_par_iter().for_each(|tile_idx| {
            let start = tile_idx * TILE_KV;
            let end = (start + TILE_KV).min(kv_len);

            for b_i in 0..b {
                for h_i in 0..h {
                    // Per‑head slope for ALiBi / NTK bias.
                    let slope = if max_bias > 0.0 {
                        2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                    } else {
                        1.0
                    };
                    let k_head = h_i / rk2;
                    let v_head = h_i / rv2;

                    // pointers into scratch
                    let base_vk = ((tile_idx * h + h_i) * dv) as isize;
                    let base_s = (tile_idx * h + h_i) as isize;

                    let q_base = b_i * qstride[0] + h_i * qstride[2];
                    let q_row = &q_data[q_base..q_base + d];

                    // Re‑materialise the pointer from the captured usize.
                    let vkq: &mut [f32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            (scratch_ptr as *mut f32).add(base_vk as usize),
                            dv,
                        )
                    };
                    let mut s = 0f32;
                    let mut m = f32::NEG_INFINITY;

                    for kv_pos in start..end {
                        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                        {
                            // One‑step look‑ahead prefetch of the next K and V rows.
                            let next_pos = kv_pos + 1;
                            if next_pos < end {
                                let next_k_base =
                                    b_i * kstride[0] + next_pos * kstride[1] + k_head * kstride[2];
                                let next_v_base =
                                    b_i * vstride[0] + next_pos * vstride[1] + v_head * vstride[2];
                                unsafe {
                                    prefetch(k_data.as_ptr().add(next_k_base) as *const u8);
                                    prefetch(v_data.as_ptr().add(next_v_base) as *const u8);
                                }
                            }
                        }
                        // ---------- dot(Q,K) ----------
                        let k_base = b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
                        let k_row = &k_data[k_base..k_base + d];

                        let mut s_val = vec_dot_f32(q_row, k_row) * scale;

                        if logit_softcap != 0.0 {
                            s_val = logit_softcap * (s_val / logit_softcap).tanh();
                        }
                        if let Some(mv_vec) = mask_vec {
                            s_val += slope * mv_vec[(b_i * kv_len) + kv_pos].to_f32();
                        }

                        // -------- tile-local softmax ----------
                        let m_old = m;
                        let mut ms = 1.0;
                        let mut vs = 1.0;
                        if s_val > m {
                            m = s_val;
                            ms = (m_old - m).exp();
                            for v in vkq.iter_mut() {
                                *v *= ms;
                            }
                        } else {
                            vs = (s_val - m).exp();
                        }

                        // -------- add V ----------
                        let v_base = b_i * vstride[0] + kv_pos * vstride[1] + v_head * vstride[2];
                        for d_i in 0..dv {
                            vkq[d_i] += v_data[v_base + d_i].to_f32() * vs;
                        }

                        s = s * ms + vs;
                    }

                    unsafe {
                        *((scratch_s_ptr as *mut f32).add(base_s as usize)) = s;
                        *((scratch_m_ptr as *mut f32).add(base_s as usize)) = m;
                    }
                }
            }
        });
    });

    // -------- serial reduction over tiles (per head) --------
    for b_i in 0..b {
        for h_i in 0..h {
            let out_off = (b_i * h + h_i) * dv;

            // start with tile 0
            let mut vkq = vec![0f32; dv];
            vkq.copy_from_slice(&scratch[(h_i * dv)..(h_i * dv + dv)]);
            let mut s = scratch_s[h_i];
            let mut m = scratch_m[h_i];

            for tile_idx in 1..kv_tiles {
                let base = (tile_idx * h + h_i) * dv;
                let base_s = tile_idx * h + h_i;

                let m_b = scratch_m[base_s];
                let s_b = scratch_s[base_s];
                let vkq_b = &scratch[base..base + dv];

                if m >= m_b {
                    let factor = (m_b - m).exp();
                    for (v, v_b) in vkq.iter_mut().zip(vkq_b) {
                        *v += v_b * factor;
                    }
                    s += s_b * factor;
                } else {
                    let factor = (m - m_b).exp();
                    for (v, v_a) in vkq.iter_mut().zip(vkq_b) {
                        *v = v_a + *v * factor;
                    }
                    s = s_b + s * factor;
                    m = m_b;
                }
            }

            let inv_s = 1.0 / s;
            for (o, v) in out[out_off..][..dv].iter_mut().zip(vkq.iter()) {
                *o = T::cast(*v * inv_s);
            }
        }
    }

    Tensor::from_vec(out, (b, h, 1, dv), &Device::Cpu)
}

/// Main forward flash-attention CPU routine.
/// Shapes follow Candle convention: (B, S, H, D)
#[allow(clippy::too_many_arguments)]
fn flash_attn_cpu<T>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    mask_vec: Option<&[T]>,
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor>
where
    T: WithDType + Copy + UpcastF32 + DowncastF32,
{
    let (b, q_len, h, d) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;

    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    let mut out = vec![T::cast(0.0); b * q_len * h * dv];

    assert_eq!(qstride[3], 1, "q must have contiguous rows");
    assert_eq!(kstride[3], 1, "k must have contiguous rows");
    assert_eq!(vstride[3], 1, "v must have contiguous rows");

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(dv)
            .with_min_len(64)
            .enumerate()
            .for_each(|(row_idx, out_chunk)| {
                // Decode flat index back to (batch, head, q_pos)
                let rows_per_batch = h * q_len;
                let b_i = row_idx / rows_per_batch;
                let rem = row_idx % rows_per_batch;
                let h_i = rem / q_len;
                let q_pos = rem % q_len;

                let slope = if max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    1.0
                };

                let k_head = h_i / rk2;
                let v_head = h_i / rv2;

                let mut vkq = vec![0f32; dv];
                let mut s = 0.0f32;
                let mut m = f32::NEG_INFINITY;

                // Q row
                let q_base = b_i * qstride[0] + q_pos * qstride[1] + h_i * qstride[2];
                let q_row = &q_data[q_base..q_base + d];

                for kv_pos in 0..kv_len {
                    // -------- ping‑pong prefetch of next KV row --------
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    {
                        let next_pos = kv_pos + 1;
                        if next_pos < kv_len {
                            let next_k_base =
                                b_i * kstride[0] + next_pos * kstride[1] + k_head * kstride[2];
                            let next_v_base =
                                b_i * vstride[0] + next_pos * vstride[1] + v_head * vstride[2];
                            unsafe {
                                prefetch(k_data.as_ptr().add(next_k_base) as *const u8);
                                prefetch(v_data.as_ptr().add(next_v_base) as *const u8);
                            }
                        }
                    }
                    // ----------------------------------------------------
                    // Mask (optional)
                    let mv = if let Some(mv_vec) = mask_vec {
                        slope * mv_vec[((b_i * q_len + q_pos) * kv_len) + kv_pos].to_f32()
                    } else {
                        0.0
                    };
                    if mv == f32::NEG_INFINITY {
                        continue;
                    }

                    // K row
                    let k_base = b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
                    let k_row = &k_data[k_base..k_base + d];

                    // dot(Q, K)
                    let mut s_val = vec_dot_f32(q_row, k_row);
                    let mut scale_applied = scale;
                    if logit_softcap != 0.0 {
                        scale_applied /= logit_softcap;
                    }
                    s_val *= scale_applied;
                    if logit_softcap != 0.0 {
                        s_val = logit_softcap * s_val.tanh();
                    }
                    s_val += mv;

                    // online softmax
                    let m_old = m;
                    let mut ms = 1.0f32;
                    let mut vs = 1.0f32;
                    if s_val > m {
                        m = s_val;
                        ms = (m_old - m).exp();
                        for v in vkq.iter_mut() {
                            *v *= ms;
                        }
                    } else {
                        vs = (s_val - m).exp();
                    }

                    // V row (strided) as f32
                    let v_base = b_i * vstride[0] + kv_pos * vstride[1] + v_head * vstride[2];
                    for d_i in 0..dv {
                        vkq[d_i] += v_data[v_base + d_i].to_f32() * vs;
                    }

                    s = s * ms + vs;
                }

                // final normalisation + downcast
                let inv_s = 1.0 / s;
                for (o, v) in out_chunk.iter_mut().zip(vkq.iter()) {
                    *o = T::cast(*v * inv_s);
                }
            });
    });

    let out_shape = (b, h, q_len, dv);
    Tensor::from_vec(out, out_shape, &Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::SdpaParams;
    use candle_core::{Device, Tensor, D};
    use candle_nn::ops::softmax;

    use candle_core::Result as CandleResult;

    const EPS: f32 = 1e-4;

    #[test]
    fn test_flash_attn_cpu_single_q() -> CandleResult<()> {
        // Test for q_len == 1 (single Q)
        let b = 1;
        let h = 2;
        let d = 4;
        let kv_len = 2;

        // Create q with shape (b, 1, h, d) filled with ones
        let q_vec = vec![1.0f32; b * h * d];
        let q = Tensor::from_vec(q_vec.clone(), (b, 1, h, d), &Device::Cpu)?;

        // Create k and v with shape (b, kv_len, h, d) filled with ones
        let k_vec = vec![1.0f32; b * kv_len * h * d];
        let v_vec = vec![1.0f32; b * kv_len * h * d];
        let k = Tensor::from_vec(k_vec.clone(), (b, kv_len, h, d), &Device::Cpu)?;
        let v = Tensor::from_vec(v_vec.clone(), (b, kv_len, h, d), &Device::Cpu)?;

        // SDPA parameters: scale = 1.0, no softcap
        let sdpa = SdpaParams {
            softmax_scale: 1.0,
            softcap: None,
            n_kv_groups: 1,
            sliding_window: None,
        };
        let scale = sdpa.softmax_scale;

        // Run the flash attention CPU function
        let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa)?;
        // Expect output shape (b, h, 1, d)
        assert_eq!(out.shape().dims(), &[b, h, 1, d]);

        // Naive attention
        let q_perm = q.clone().permute((0, 2, 1, 3))?;
        let k_perm = k.clone().permute((0, 2, 1, 3))?;
        let v_perm = v.clone().permute((0, 2, 1, 3))?;

        let q_resh = q_perm.reshape(&[b * h, 1, d])?;
        let k_resh = k_perm.reshape(&[b * h, kv_len, d])?;
        let v_resh = v_perm.reshape(&[b * h, kv_len, d])?;

        let logits = (q_resh.matmul(&k_resh.transpose(1, 2)?)? * scale as f64)?;
        let weights = softmax(&logits, D::Minus1)?;

        let attn_out = weights.matmul(&v_resh)?;

        let naive_out = attn_out.reshape(&[b, h, 1, d])?;

        let flash_data = out.flatten_all()?.to_vec1::<f32>()?;
        let naive_data = naive_out.flatten_all()?.to_vec1::<f32>()?;
        for (f_val, n_val) in flash_data.iter().zip(naive_data.iter()) {
            assert!((f_val - n_val).abs() < EPS);
        }

        Ok(())
    }

    #[test]
    fn test_flash_attn_cpu_full_q() -> CandleResult<()> {
        // Test for q_len > 1 (full Q)
        let b = 1;
        let q_len = 2;
        let h = 2;
        let d = 4;
        let kv_len = 2;

        // Create q with shape (b, q_len, h, d) filled with ones
        let q_vec = vec![1.0f32; b * q_len * h * d];
        let q = Tensor::from_vec(q_vec.clone(), (b, q_len, h, d), &Device::Cpu)?;

        // Create k and v with shape (b, kv_len, h, d) filled with ones
        let k_vec = vec![1.0f32; b * kv_len * h * d];
        let v_vec = vec![1.0f32; b * kv_len * h * d];
        let k = Tensor::from_vec(k_vec.clone(), (b, kv_len, h, d), &Device::Cpu)?;
        let v = Tensor::from_vec(v_vec.clone(), (b, kv_len, h, d), &Device::Cpu)?;

        // SDPA parameters: scale = 1.0, no softcap
        let sdpa = SdpaParams {
            softmax_scale: 1.0,
            softcap: None,
            n_kv_groups: 1,
            sliding_window: None,
        };
        let scale = sdpa.softmax_scale;

        // Run the flash attention CPU function
        let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa)?;
        // Expect output shape (b, h, q_len, d)
        assert_eq!(out.shape().dims(), &[b, h, q_len, d]);

        // Naive attention
        let q_perm = q.clone().permute((0, 2, 1, 3))?;
        let k_perm = k.clone().permute((0, 2, 1, 3))?;
        let v_perm = v.clone().permute((0, 2, 1, 3))?;

        let q_resh = q_perm.reshape(&[b * h, q_len, d])?;
        let k_resh = k_perm.reshape(&[b * h, kv_len, d])?;
        let v_resh = v_perm.reshape(&[b * h, kv_len, d])?;

        let logits = (q_resh.matmul(&k_resh.transpose(1, 2)?)? * scale as f64)?;
        let weights = softmax(&logits, D::Minus1)?;

        let attn_out = weights.matmul(&v_resh)?;

        let naive_out = attn_out.reshape(&[b, h, q_len, d])?;

        // 7) Compare element‐by‐element
        let flash_data = out.flatten_all()?.to_vec1::<f32>()?;
        let naive_data = naive_out.flatten_all()?.to_vec1::<f32>()?;
        for (f_val, n_val) in flash_data.iter().zip(naive_data.iter()) {
            assert!((f_val - n_val).abs() < EPS);
        }

        Ok(())
    }

    #[test]
    fn test_flash_attn_cpu_single_q_softcap() -> CandleResult<()> {
        // Test for q_len == 1 (single Q) with softcap
        let b = 1;
        let h = 2;
        let d = 4;
        let kv_len = 2;

        // Create q with shape (b, 1, h, d) filled with ones
        let q_vec = vec![1.0f32; b * h * d];
        let q = Tensor::from_vec(q_vec.clone(), (b, 1, h, d), &Device::Cpu)?;

        // Create k and v with shape (b, kv_len, h, d) filled with ones
        let k_vec = vec![1.0f32; b * kv_len * h * d];
        let v_vec = vec![1.0f32; b * kv_len * h * d];
        let k = Tensor::from_vec(k_vec.clone(), (b, kv_len, h, d), &Device::Cpu)?;
        let v = Tensor::from_vec(v_vec.clone(), (b, kv_len, h, d), &Device::Cpu)?;

        // SDPA parameters: scale = 1.0, softcap = Some(0.5)
        let sdpa = SdpaParams {
            softmax_scale: 1.0,
            softcap: Some(0.5),
            n_kv_groups: 1,
            sliding_window: None,
        };
        let scale = sdpa.softmax_scale;
        let sc = sdpa.softcap.unwrap();

        // Run the flash attention CPU function
        let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa)?;
        // Expect output shape (b, h, 1, d)
        assert_eq!(out.shape().dims(), &[b, h, 1, d]);

        // Naive attention with softcap
        let q_perm = q.clone().permute((0, 2, 1, 3))?;
        let k_perm = k.clone().permute((0, 2, 1, 3))?;
        let v_perm = v.clone().permute((0, 2, 1, 3))?;

        let q_resh = q_perm.reshape(&[b * h, 1, d])?;
        let k_resh = k_perm.reshape(&[b * h, kv_len, d])?;
        let v_resh = v_perm.reshape(&[b * h, kv_len, d])?;

        let logits = (q_resh.matmul(&k_resh.transpose(1, 2)?)? * scale as f64)?;

        let logits = (logits / sc as f64)?;
        let logits = logits.tanh()?;
        let logits = (logits * sc as f64)?;

        let weights = softmax(&logits, D::Minus1)?;
        let attn_out = weights.matmul(&v_resh)?;
        let naive_out = attn_out.reshape(&[b, h, 1, d])?;

        let flash_data = out.flatten_all()?.to_vec1::<f32>()?;
        let naive_data = naive_out.flatten_all()?.to_vec1::<f32>()?;
        for (f_val, n_val) in flash_data.iter().zip(naive_data.iter()) {
            assert!((f_val - n_val).abs() < EPS);
        }
        Ok(())
    }

    #[test]
    fn test_flash_attn_cpu_full_q_softcap() -> CandleResult<()> {
        // Test for q_len > 1 (full Q) with softcap
        let b = 1;
        let q_len = 2;
        let h = 2;
        let d = 4;
        let kv_len = 2;

        // Create q with shape (b, q_len, h, d) filled with ones
        let q_vec = vec![1.0f32; b * q_len * h * d];
        let q = Tensor::from_vec(q_vec.clone(), (b, q_len, h, d), &Device::Cpu)?;

        // Create k and v with shape (b, kv_len, h, d) filled with ones
        let k_vec = vec![1.0f32; b * kv_len * h * d];
        let v_vec = vec![1.0f32; b * kv_len * h * d];
        let k = Tensor::from_vec(k_vec.clone(), (b, kv_len, h, d), &Device::Cpu)?;
        let v = Tensor::from_vec(v_vec.clone(), (b, kv_len, h, d), &Device::Cpu)?;

        // SDPA parameters: scale = 1.0, softcap = Some(0.5)
        let sdpa = SdpaParams {
            softmax_scale: 1.0,
            softcap: Some(0.5),
            n_kv_groups: 1,
            sliding_window: None,
        };
        let scale = sdpa.softmax_scale;
        let sc = sdpa.softcap.unwrap();

        // Run the flash attention CPU function
        let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa)?;
        // Expect output shape (b, h, q_len, d)
        assert_eq!(out.shape().dims(), &[b, h, q_len, d]);

        // Naive attention with softcap
        let q_perm = q.clone().permute((0, 2, 1, 3))?;
        let k_perm = k.clone().permute((0, 2, 1, 3))?;
        let v_perm = v.clone().permute((0, 2, 1, 3))?;

        let q_resh = q_perm.reshape(&[b * h, q_len, d])?;
        let k_resh = k_perm.reshape(&[b * h, kv_len, d])?;
        let v_resh = v_perm.reshape(&[b * h, kv_len, d])?;

        let logits = (q_resh.matmul(&k_resh.transpose(1, 2)?)? * scale as f64)?;

        let logits = (logits / sc as f64)?;
        let logits = logits.tanh()?;
        let logits = (logits * sc as f64)?;

        let weights = softmax(&logits, D::Minus1)?;
        let attn_out = weights.matmul(&v_resh)?;
        let naive_out = attn_out.reshape(&[b, h, q_len, d])?;

        let flash_data = out.flatten_all()?.to_vec1::<f32>()?;
        let naive_data = naive_out.flatten_all()?.to_vec1::<f32>()?;
        for (f_val, n_val) in flash_data.iter().zip(naive_data.iter()) {
            assert!((f_val - n_val).abs() < EPS);
        }
        Ok(())
    }
}
