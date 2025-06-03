#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Context, Device, Result, Storage, Tensor, WithDType};

use half::bf16;
use half::f16;
use rayon::prelude::*;
use rayon::ThreadPool;

use std::sync::LazyLock;
use std::{f32, iter::Sum};

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
    // Shapes: (B, 1, H, D)
    let (b, _q_len, h, d) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;

    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    // Final output buffer (B, H, 1, D) – already typed as `T`
    let mut out = vec![T::cast(0.0); b * h * dv];

    // Split the KV axis into cache‑friendly tiles.
    let kv_tiles = kv_len.div_ceil(TILE_KV);

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(dv)
            .with_min_len(64)
            .enumerate()
            .for_each(|(row_idx, out_chunk)| {
                let b_i = row_idx / h;
                let h_i = row_idx % h;

                let slope = if max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    1.0
                };

                let k_head = h_i / rk2;
                let v_head = h_i / rv2;

                // ---------------- map‑reduce over KV tiles -----------------
                let (vkq, s_tot, _m_tot) = (0..kv_tiles)
                    .into_par_iter()
                    .map(|tile_idx| {
                        let start = tile_idx * TILE_KV;
                        let end = (start + TILE_KV).min(kv_len);

                        let mut vkq = vec![0f32; dv];
                        let mut s = 0.0f32;
                        let mut m = f32::NEG_INFINITY;

                        // Q row is already contiguous in memory for q_len == 1
                        let q_base = b_i * qstride[0] + h_i * qstride[2];
                        let q_row: Vec<f32> =
                            (0..d).map(|di| q_data[q_base + di].to_f32()).collect();

                        for kv_pos in start..end {
                            // ---- mask -------------------------------------------------------
                            let mv = if let Some(mv_vec) = mask_vec {
                                slope * mv_vec[(b_i * kv_len) + kv_pos].to_f32()
                            } else {
                                0.0
                            };
                            if mv == f32::NEG_INFINITY {
                                continue;
                            }

                            // ---- K row -------------------------------------------------------
                            let k_base =
                                b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
                            let k_row: Vec<f32> =
                                (0..d).map(|di| k_data[k_base + di].to_f32()).collect();

                            // dot(Q, K)
                            let mut s_val = vec_dot_f32(&q_row, &k_row);

                            let mut scale_applied = scale;
                            if logit_softcap != 0.0 {
                                scale_applied /= logit_softcap;
                            }
                            s_val *= scale_applied;
                            if logit_softcap != 0.0 {
                                s_val = logit_softcap * s_val.tanh();
                            }
                            s_val += mv;

                            // ---- tile‑local online soft‑max -------------------------------
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

                            // ---- V row -------------------------------------------------------
                            let v_base =
                                b_i * vstride[0] + kv_pos * vstride[1] + v_head * vstride[2];
                            for d_i in 0..dv {
                                vkq[d_i] += v_data[v_base + d_i * vstride[3]].to_f32() * vs;
                            }

                            s = s * ms + vs;
                        }

                        (vkq, s, m)
                    })
                    // --------------- reduce two tiles --------------------------------------
                    .reduce(
                        || (vec![0f32; dv], 0.0f32, f32::NEG_INFINITY),
                        |mut a, b| {
                            let (ref mut vkq_a, mut s_a, m_a) = a;
                            let (vkq_b, s_b, m_b) = b;
                            if m_a >= m_b {
                                let factor = (m_b - m_a).exp();
                                for (va, vb) in vkq_a.iter_mut().zip(vkq_b) {
                                    *va += vb * factor;
                                }
                                s_a += s_b * factor;
                                (vkq_a.clone(), s_a, m_a)
                            } else {
                                let factor = (m_a - m_b).exp();
                                let mut vkq_new = vkq_b;
                                for (vb, va) in vkq_new.iter_mut().zip(vkq_a) {
                                    *vb += *va * factor;
                                }
                                (vkq_new, s_b + s_a * factor, m_b)
                            }
                        },
                    );

                // ---------------- final normalisation + down‑cast -------------
                let inv_s = 1.0 / s_tot;
                for (o, v) in out_chunk.iter_mut().zip(vkq.iter()) {
                    *o = T::cast(*v * inv_s);
                }
            });
    });

    let out_shape = (b, h, 1usize, dv);
    Tensor::from_vec(out, out_shape, &Device::Cpu)
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

                // Gather Q row (strided) as f32
                let q_base = b_i * qstride[0] + q_pos * qstride[1] + h_i * qstride[2];
                let q_row: Vec<f32> = (0..d)
                    .map(|di| q_data[q_base + di * qstride[3]].to_f32())
                    .collect();

                for kv_pos in 0..kv_len {
                    // Mask (optional)
                    let mv = if let Some(mv_vec) = mask_vec {
                        slope * mv_vec[((b_i * q_len + q_pos) * kv_len) + kv_pos].to_f32()
                    } else {
                        0.0
                    };
                    if mv == f32::NEG_INFINITY {
                        continue;
                    }

                    // K row (strided) as f32
                    let k_base = b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
                    let k_row: Vec<f32> = (0..d)
                        .map(|di| k_data[k_base + di * kstride[3]].to_f32())
                        .collect();

                    // dot(Q, K)
                    let mut s_val = vec_dot_f32(&q_row, &k_row);
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
                        vkq[d_i] += v_data[v_base + d_i * vstride[3]].to_f32() * vs;
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
