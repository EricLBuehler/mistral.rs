#![allow(unused)]

use candle_core::{
    backend::BackendStorage, from_storage_no_op, DType, MetalStorage, Result, Shape, Storage,
    Tensor, D,
};

use super::{AfqBits, AfqGroupSize};

/// Returns (w_q, scales, biases)
pub(crate) fn afq_quantize_op(
    w: &Tensor,
    group_size: AfqGroupSize,
    bits: AfqBits,
) -> Result<(Tensor, Tensor, Tensor)> {
    let group_size = group_size as usize;
    let bits = bits as usize;

    if w.rank() < 2 {
        candle_core::bail!("AFQ quantize expects weight matrix of at least rank 2");
    }
    if w.dim(D::Minus1)? % group_size != 0 {
        candle_core::bail!(
            "Last dim of weight matrix ({:?}) must be divisible by group size {group_size}.",
            w.dims()
        );
    }

    #[cfg(feature = "metal")]
    {
        let w_s = w.storage_and_layout().0;
        let Storage::Metal(w_s) = &*w_s else {
            candle_core::bail!("expected metal")
        };
        let device = w_s.device();

        // This is necessary to avoid the errors of "A command encoder is already encoding to this command buffer"
        let command_buffer = device.new_command_buffer()?;
        command_buffer.set_label("afq-quantize");

        let mut wq_shape = w.dims().to_vec();
        *wq_shape.last_mut().unwrap() = w.dim(D::Minus1)? * bits / 32;
        let mut s_shape = w.dims().to_vec();
        *s_shape.last_mut().unwrap() = w.dim(D::Minus1)? / group_size;

        let output =
            device.new_buffer(wq_shape.iter().product(), DType::U32, "afq-quantize-output")?;
        let scales =
            device.new_buffer(s_shape.iter().product(), w.dtype(), "afq-quantize-scales")?;
        let biases =
            device.new_buffer(s_shape.iter().product(), w.dtype(), "afq-quantize-biases")?;

        crate::metal_kernels::call_affine_quantize(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            w.dtype(),
            w_s.buffer(),
            w.layout().start_offset() * w_s.dtype().size_in_bytes(),
            w.dims(),
            w.stride(),
            &output,
            &wq_shape,
            &scales,
            &biases,
            false,
            group_size,
            bits,
        )
        .map_err(candle_core::Error::wrap)?;

        // This is necessary to avoid the errors of "A command encoder is already encoding to this command buffer"
        command_buffer.commit();

        let output = from_storage_no_op(
            Storage::Metal(MetalStorage::new(
                output,
                device.clone(),
                wq_shape.iter().product(),
                DType::U32,
            )),
            wq_shape,
            false,
        );
        let scales = from_storage_no_op(
            Storage::Metal(MetalStorage::new(
                scales,
                device.clone(),
                s_shape.iter().product(),
                w.dtype(),
            )),
            s_shape.clone(),
            false,
        );
        let biases = from_storage_no_op(
            Storage::Metal(MetalStorage::new(
                biases,
                device.clone(),
                s_shape.iter().product(),
                w.dtype(),
            )),
            s_shape,
            false,
        );

        Ok((output, scales, biases))
    }
    #[cfg(not(feature = "metal"))]
    {
        return cpu_backend::afq_quantize_op(w, group_size, bits);
    }
}

pub(crate) fn afq_dequantize_op(
    w_q: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: AfqGroupSize,
    bits: AfqBits,
) -> Result<Tensor> {
    let group_size = group_size as usize;
    let bits = bits as usize;

    if w_q.rank() < 2 || scales.rank() < 2 || biases.rank() < 2 {
        candle_core::bail!("AFQ dequantize expects all matrices of at least rank 2");
    }

    #[cfg(feature = "metal")]
    {
        let wq_s = w_q.storage_and_layout().0;
        let Storage::Metal(wq_s) = &*wq_s else {
            candle_core::bail!("expected metal")
        };
        let s_s = scales.storage_and_layout().0;
        let Storage::Metal(s_s) = &*s_s else {
            candle_core::bail!("expected metal")
        };
        let b_s = biases.storage_and_layout().0;
        let Storage::Metal(b_s) = &*b_s else {
            candle_core::bail!("expected metal")
        };

        let device = wq_s.device();

        let command_buffer = device.command_buffer()?;
        command_buffer.set_label("afq-dequantize");

        let out_size = w_q.dim(D::Minus1)? * 32 / bits;
        let mut w_shape = w_q.dims().to_vec();
        *w_shape.last_mut().unwrap() = out_size;

        if out_size != scales.dim(D::Minus1)? * group_size
            || out_size != biases.dim(D::Minus1)? * group_size
        {
            candle_core::bail!(
                "Scales and biases do not match the matrix given dequantization parameters."
            );
        }

        let output = device.new_buffer(
            w_shape.iter().product(),
            scales.dtype(),
            "afq-dequantize-output",
        )?;

        assert_eq!(w_q.layout().start_offset(), 0);
        assert_eq!(scales.layout().start_offset(), 0);
        assert_eq!(biases.layout().start_offset(), 0);
        crate::metal_kernels::call_affine_quantize(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            scales.dtype(),
            wq_s.buffer(),
            w_q.layout().start_offset() * wq_s.dtype().size_in_bytes(),
            w_q.dims(),
            w_q.stride(),
            &output,
            &w_shape,
            s_s.buffer(),
            b_s.buffer(),
            true,
            group_size,
            bits,
        )
        .map_err(candle_core::Error::wrap)?;

        let output = from_storage_no_op(
            Storage::Metal(MetalStorage::new(
                output,
                device.clone(),
                w_shape.iter().product(),
                scales.dtype(),
            )),
            w_shape,
            false,
        );

        Ok(output)
    }
    #[cfg(not(feature = "metal"))]
    {
        return cpu_backend::afq_dequantize_op(w_q, scales, biases, group_size, bits);
    }
}

fn make_dummy_indices(x: &Tensor) -> Result<Tensor> {
    let x_batches = x
        .dims()
        .iter()
        .take(x.rank() - 2)
        .copied()
        .collect::<Vec<_>>();

    (Tensor::ones(x_batches.iter().product::<usize>(), DType::F32, x.device())?
        .cumsum(0)?
        .to_dtype(DType::U32)?
        - 1.)?
        .reshape(x_batches)
}

/// The indices lhs_indices and rhs_indices contain flat indices along the batch dimensions (i.e. all but the last two dimensions) of a and b respectively.
#[allow(clippy::too_many_arguments)]
pub(crate) fn afq_mm_op(
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    lhs_indices: Option<&Tensor>,
    rhs_indices: Option<&Tensor>,
    group_size: AfqGroupSize,
    bits: AfqBits,
    transpose: bool,
) -> Result<Tensor> {
    let group_size = group_size as usize;
    let bits = bits as usize;

    let w_outer_dims = {
        if w.dtype() != DType::U32 {
            candle_core::bail!("AFQ weight matrix must be u32");
        }
        if scales.dims() != biases.dims() {
            candle_core::bail!("Scales and biases should have the same shapes");
        }
        if w.dim(D::Minus1)? * 32 / bits != scales.dim(D::Minus1)? * group_size {
            candle_core::bail!("Last dims of w and scales must be compatible.");
        }

        let x_inner_dims = x.dim(D::Minus1)?;

        // Calculate transpose w dims
        let w_inner_dims = if transpose {
            w.dim(D::Minus1)? * 32 / bits
        } else {
            w.dim(D::Minus2)?
        };
        let w_outer_dims = if transpose {
            w.dim(D::Minus2)?
        } else {
            w.dim(D::Minus1)? * 32 / bits
        };

        if w_inner_dims != x_inner_dims {
            candle_core::bail!(
                "w inner dims ({:?}) must match x inner dims ({:?}). transpose={transpose}",
                w.dims(),
                x.dims()
            );
        }

        w_outer_dims
    };

    #[cfg(feature = "metal")]
    {
        let x_s = x.storage_and_layout().0;
        let Storage::Metal(x_s) = &*x_s else {
            candle_core::bail!("expected metal")
        };
        let w_s = w.storage_and_layout().0;
        let Storage::Metal(w_s) = &*w_s else {
            candle_core::bail!("expected metal")
        };
        let s_s = scales.storage_and_layout().0;
        let Storage::Metal(s_s) = &*s_s else {
            candle_core::bail!("expected metal")
        };
        let b_s = biases.storage_and_layout().0;
        let Storage::Metal(b_s) = &*b_s else {
            candle_core::bail!("expected metal")
        };

        let device = w_s.device();

        let command_buffer = device.command_buffer()?;
        command_buffer.set_label("afq-qmm");

        assert_eq!(x.layout().start_offset(), 0);
        assert_eq!(w.layout().start_offset(), 0);
        assert_eq!(scales.layout().start_offset(), 0);
        assert_eq!(biases.layout().start_offset(), 0);

        let (output, out_shape) = if lhs_indices.is_some() || rhs_indices.is_some() {
            let mut lhs_indices = match lhs_indices {
                Some(lhs_indices) => lhs_indices.clone(),
                None => make_dummy_indices(x)?,
            };
            let mut rhs_indices = match rhs_indices {
                Some(rhs_indices) => rhs_indices.clone(),
                None => make_dummy_indices(w)?,
            };
            if lhs_indices.dtype() != DType::U32 || rhs_indices.dtype() != DType::U32 {
                candle_core::bail!("lhs and rhs indices must be u32.")
            }
            // Broadcast the indices if applicable.
            {
                let mut shape = lhs_indices.shape().clone();
                shape = rhs_indices
                    .shape()
                    .broadcast_shape_binary_op(rhs_indices.shape(), "afq-qmm")?;
                lhs_indices = lhs_indices.broadcast_as(shape.clone())?;
                rhs_indices = rhs_indices.broadcast_as(shape)?;
            }

            let li_s = lhs_indices.storage_and_layout().0;
            let Storage::Metal(li_s) = &*li_s else {
                candle_core::bail!("expected metal")
            };
            let ri_s = rhs_indices.storage_and_layout().0;
            let Storage::Metal(ri_s) = &*ri_s else {
                candle_core::bail!("expected metal")
            };

            let mut out_shape = lhs_indices.dims().to_vec();
            out_shape.push(x.dim(D::Minus2)?);
            out_shape.push(w_outer_dims);

            let output =
                device.new_buffer(out_shape.iter().product(), scales.dtype(), "afq-qmm-output")?;

            crate::metal_kernels::call_afq_qmm(
                device.device(),
                &command_buffer,
                &crate::metal_kernels::Kernels::new(),
                scales.dtype(),
                x_s.buffer(),
                x.dims(),
                x.stride(),
                w_s.buffer(),
                w.dims(),
                w.stride(),
                s_s.buffer(),
                scales.stride(),
                b_s.buffer(),
                biases.stride(),
                &output,
                &out_shape,
                Some((li_s.buffer(), ri_s.buffer())),
                Some(lhs_indices.dims()),
                Some((lhs_indices.stride(), rhs_indices.stride())),
                transpose,
                bits,
                group_size,
            )
            .map_err(candle_core::Error::wrap)?;

            (output, out_shape)
        } else {
            let mut out_shape = x.dims().to_vec();
            *out_shape.last_mut().unwrap() = w_outer_dims;

            let output =
                device.new_buffer(out_shape.iter().product(), scales.dtype(), "afq-qmm-output")?;

            crate::metal_kernels::call_afq_qmm(
                device.device(),
                &command_buffer,
                &crate::metal_kernels::Kernels::new(),
                scales.dtype(),
                x_s.buffer(),
                x.dims(),
                x.stride(),
                w_s.buffer(),
                w.dims(),
                w.stride(),
                s_s.buffer(),
                scales.stride(),
                b_s.buffer(),
                biases.stride(),
                &output,
                &out_shape,
                None,
                None,
                None,
                transpose,
                bits,
                group_size,
            )
            .map_err(candle_core::Error::wrap)?;

            (output, out_shape)
        };

        let output = from_storage_no_op(
            Storage::Metal(MetalStorage::new(
                output,
                device.clone(),
                out_shape.iter().product(),
                scales.dtype(),
            )),
            out_shape,
            false,
        );

        Ok(output)
    }
    #[cfg(not(feature = "metal"))]
    {
        return cpu_backend::afq_mm_op(
            x,
            w,
            scales,
            biases,
            lhs_indices,
            rhs_indices,
            group_size,
            bits,
            transpose,
        );
    }
}

#[cfg(feature = "metal")]
#[cfg(test)]
mod metal_tests {
    use candle_core::{DType, Device, Result, Tensor, D};

    use crate::{afq::ops::afq_dequantize_op, AfqBits, AfqGroupSize};

    use super::afq_quantize_op;

    fn run_afq_roundtrip(bits: AfqBits) -> Result<f32> {
        let device = Device::new_metal(0)?;
        let group_size = AfqGroupSize::Low;

        let xs = Tensor::randn(0f32, 1f32, (32, 32), &device)?;

        let (w_q, scales, biases) = afq_quantize_op(&xs, group_size, bits)?;

        // println!("w_q = {w_q}");
        // println!("scales = {scales}");
        // println!("biases = {biases}");

        let ys = afq_dequantize_op(&w_q, &scales, &biases, group_size, bits)?;

        // println!("xs = {xs}");
        // println!("ys = {ys}");
        // println!("delta = {}", (xs - ys)?);

        let rmse = (xs - ys)?
            .sqr()?
            .mean(D::Minus1)?
            .sqrt()?
            .mean_all()?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;

        Ok(rmse)
    }

    #[test]
    fn test_afq_eight() -> Result<()> {
        let rmse = run_afq_roundtrip(AfqBits::Eight)?;
        assert!(rmse < 0.005, "{rmse}");
        Ok(())
    }

    #[test]
    fn test_afq_six() -> Result<()> {
        let rmse = run_afq_roundtrip(AfqBits::Six)?;
        assert!(rmse < 0.02, "{rmse}");
        Ok(())
    }

    #[test]
    fn test_afq_four() -> Result<()> {
        let rmse = run_afq_roundtrip(AfqBits::Four)?;
        assert!(rmse < 0.078, "{rmse}");
        Ok(())
    }

    #[test]
    fn test_afq_three() -> Result<()> {
        let rmse = run_afq_roundtrip(AfqBits::Three)?;
        assert!(rmse < 0.17, "{rmse}");
        Ok(())
    }

    #[test]
    fn test_afq_two() -> Result<()> {
        let rmse = run_afq_roundtrip(AfqBits::Two)?;
        assert!(rmse < 0.35, "{rmse}");
        Ok(())
    }
}

// ============================================================
//                    Portable CPU back‑end
// ============================================================
#[cfg(not(feature = "metal"))]
mod cpu_backend {
    use super::*;
    use candle_core::{DType, Device, Result, Tensor, D};
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    use core::arch::aarch64::*;
    use half::f16;

    /// Simple scalar (reference) quantiser: per‑`group_size` affine.
    pub(crate) fn afq_quantize_op(
        w: &Tensor,
        group_size: usize,
        bits: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let device = Device::Cpu;
        let levels = ((1u32 << bits) - 1) as f32;

        // Flatten everything except the last dim.
        let w_vec = w.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let outer: usize = w_vec.len() / w.dim(D::Minus1)?;
        let inner = w.dim(D::Minus1)?;

        let packed_row = inner * bits / 32;
        let groups_per_row = inner / group_size;

        let mut q_codes = vec![0u32; outer * packed_row];
        let mut scales = vec![0f32; outer * groups_per_row];
        let mut biases = vec![0f32; outer * groups_per_row];

        for row in 0..outer {
            for g in 0..groups_per_row {
                let base = row * inner + g * group_size;
                let slice = &w_vec[base..base + group_size];
                let (min_v, max_v) = slice
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                let scale = if (max_v - min_v).abs() < 1e-12 {
                    1.0
                } else {
                    (max_v - min_v) / levels
                };
                let bias = min_v;
                scales[row * groups_per_row + g] = scale;
                biases[row * groups_per_row + g] = bias;

                for i in 0..group_size {
                    let j = g * group_size + i; // position in this row
                    let bit_off = j * bits; // overall bit offset
                    let word_id = bit_off / 32; // u32 index
                    let shift = bit_off % 32; // intra‑word shift

                    let q_mask = ((1u32 << bits) - 1) as u32;
                    let q_val = ((w_vec[base + i] - bias) / scale)
                        .round()
                        .clamp(0.0, levels) as u32
                        & q_mask;

                    let row_base = row * packed_row;
                    q_codes[row_base + word_id] |= q_val << shift;
                    if shift + bits > 32 {
                        q_codes[row_base + word_id + 1] |= q_val >> (32 - shift);
                    }
                }
            }
        }

        let w_q = Tensor::from_vec(
            q_codes,
            {
                let mut d = w.dims().to_vec();
                *d.last_mut().unwrap() = packed_row;
                d
            },
            &device,
        )?
        .to_dtype(DType::U32)?;
        let sc = Tensor::from_vec(
            scales,
            {
                let mut d = w.dims().to_vec();
                *d.last_mut().unwrap() = groups_per_row;
                d
            },
            &device,
        )?;
        let bs = Tensor::from_vec(
            biases,
            {
                let mut d = w.dims().to_vec();
                *d.last_mut().unwrap() = groups_per_row;
                d
            },
            &device,
        )?;
        Ok((w_q, sc, bs))
    }

    /// Scalar de‑quantiser (inverse of the above).
    pub(crate) fn afq_dequantize_op(
        w_q: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        group_size: usize,
        _bits: usize,
    ) -> Result<Tensor> {
        let device = Device::Cpu;
        let codes = w_q.flatten_all()?.to_vec1::<u32>()?;
        let sc = scales.flatten_all()?.to_vec1::<f32>()?;
        let bs = biases.flatten_all()?.to_vec1::<f32>()?;

        let packed_row = w_q.dim(D::Minus1)?;
        let outer = codes.len() / packed_row;
        let inner = packed_row * 32 / _bits;
        let groups_per_row = inner / group_size;

        let mut out = vec![0f32; outer * inner];
        for row in 0..outer {
            for g in 0..groups_per_row {
                let scale = sc[row * groups_per_row + g];
                let bias = bs[row * groups_per_row + g];
                for i in 0..group_size {
                    let j = g * group_size + i;
                    let bit_off = j * _bits;
                    let word_id = bit_off / 32;
                    let shift = bit_off % 32;

                    let row_base = row * packed_row;
                    let mut q = (codes[row_base + word_id] >> shift) & ((1u32 << _bits) - 1);
                    if shift + _bits > 32 {
                        q |=
                            (codes[row_base + word_id + 1] << (32 - shift)) & ((1u32 << _bits) - 1);
                    }

                    let idx = row * inner + j;
                    out[idx] = bias + q as f32 * scale;
                }
            }
        }
        Tensor::from_vec(
            out,
            {
                let mut d = w_q.dims().to_vec();
                *d.last_mut().unwrap() = inner;
                d
            },
            &device,
        )
    }

    /// Fast NEON implementation for 2‑D matrix multiply of F16 × (packed‑u32) with affine groups.
    ///
    /// Falls back to scalar path if:
    /// * `transpose == true`
    /// * `lhs_indices` / `rhs_indices` are supplied
    /// * `bits > 8`
    /// * the compiled target isn’t aarch64+neon
    #[inline]
    #[allow(clippy::too_many_arguments)]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn afq_mm_fast_neon(
        x: &Tensor,
        w: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        group_size: usize,
        bits: usize,
    ) -> Result<Tensor> {
        if x.rank() != 2 || w.rank() != 2 || scales.rank() != 2 || biases.rank() != 2 {
            candle_core::bail!("afq_mm_fast_neon currently supports only 2‑D tensors");
        }
        // Layout: w is [K, packed_N]
        let m = x.dim(D::Minus2)?; // batch
        let k = x.dim(D::Minus1)?; // inner
        let packed_n = w.dim(D::Minus1)?;
        let n = packed_n * 32 / bits; // real output dim
        let groups_per_row = n / group_size;
        debug_assert_eq!(w.dim(D::Minus2)?, k);
        debug_assert_eq!(scales.dim(D::Minus2)?, k);
        debug_assert_eq!(biases.dim(D::Minus2)?, k);
        debug_assert_eq!(scales.dim(D::Minus1)?, groups_per_row);
        debug_assert_eq!(biases.dim(D::Minus1)?, groups_per_row);

        // Flatten all tensors
        let x_vec = x.flatten_all()?.to_dtype(DType::F16)?.to_vec1::<f16>()?;
        let w_codes = w.flatten_all()?.to_vec1::<u32>()?;
        let sc_vec = scales.flatten_all()?.to_vec1::<f32>()?;
        let bs_vec = biases.flatten_all()?.to_vec1::<f32>()?;

        let mut y = vec![0f32; m * n];

        let mask = (1u32 << bits) - 1;
        let codes_per_word = 32 / bits;
        let words_per_group = group_size / codes_per_word;

        for row_idx in 0..m {
            let x_row = &x_vec[row_idx * k..(row_idx + 1) * k];
            // Pointer to this output row
            let y_row = &mut y[row_idx * n..(row_idx + 1) * n];

            // zero‑init the row
            for v in y_row.iter_mut() {
                *v = 0.0;
            }

            for kk in 0..k {
                let x_scalar = x_row[kk].to_f32();

                // pre‑compute offset in w/scales/biases for this kk row
                let w_row_base = kk * packed_n;
                let sb_base = kk * groups_per_row;

                for g in 0..groups_per_row {
                    let scale = sc_vec[sb_base + g];
                    let bias = bs_vec[sb_base + g];

                    let x_scale = x_scalar * scale;
                    let x_bias = x_scalar * bias;

                    let y_group = &mut y_row[g * group_size..(g + 1) * group_size];

                    let word_offset = w_row_base + g * words_per_group;

                    // Process group_size elements (<=32). Work in 8‑value (u32) blocks (exact for bits 2‑8).
                    let mut local_codes = [0u32; 32];
                    for w_idx in 0..words_per_group {
                        let word = w_codes[word_offset + w_idx];
                        for c in 0..codes_per_word {
                            local_codes[w_idx * codes_per_word + c] = (word >> (c * bits)) & mask;
                        }
                    }

                    // NEON accumulate in 4‑wide lanes.
                    unsafe {
                        let bias_vec = vdupq_n_f32(x_bias);
                        let mul_const = x_scale;

                        for chunk in 0..(group_size / 4) {
                            let q_u32 = vld1q_u32(local_codes.as_ptr().add(chunk * 4));
                            let q_f32 = vcvtq_f32_u32(q_u32);
                            let acc_ptr = y_group.as_mut_ptr().add(chunk * 4);
                            let acc_old = vld1q_f32(acc_ptr);
                            let acc_new = vmlaq_n_f32(bias_vec, q_f32, mul_const);
                            // acc_old + acc_new
                            let acc_sum = vaddq_f32(acc_old, acc_new);
                            vst1q_f32(acc_ptr, acc_sum);
                        }
                    }
                }
            }
        }

        Tensor::from_vec(y, vec![m, n], &Device::Cpu)?.to_dtype(x.dtype())
    }

    /// NEON fast‑path where `transpose == true`.
    ///
    /// `w` is stored with shape **[packed_N, K]** (i.e., the affine‑quantised
    /// matrix is already “row‑major” with respect to the *output*
    /// dimension).  We multiply:
    ///
    ///    Y[M,K] = X[M,K] · Wᵀ[K,N]   (where N = packed_N * 32 / bits)
    ///
    /// so every output column in `Wᵀ` corresponds to one original
    /// *group‑size* block in the packed rows.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn afq_mm_fast_neon_t(
        x: &Tensor,
        w: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        group_size: usize,
        bits: usize,
    ) -> Result<Tensor> {
        // Shapes check: X[M,K],  W[packed_N, K]  (packed along dim‑0)
        if x.rank() != 2 || w.rank() != 2 || scales.rank() != 2 || biases.rank() != 2 {
            candle_core::bail!("afq_mm_fast_neon_t supports only 2‑D tensors");
        }

        let m = x.dim(D::Minus2)?; // batch‑rows
        let k = x.dim(D::Minus1)?; // shared dim
        let packed_n = w.dim(D::Minus1)?; // packed columns (input dimension, packed)
        let k_out = w.dim(D::Minus2)?; // number of output rows
        let n = packed_n * 32 / bits; // real input dim (unpacked input)
        let groups_per_row = n / group_size;

        // Ensure X’s inner dimension matches the unpacked input dimension of W
        if k != n {
            candle_core::bail!("afq_mm_fast_neon_t: X inner dim {k} must equal unpacked dim {n}");
        }
        // Ensure scales / biases have one row per output channel
        debug_assert_eq!(scales.dim(D::Minus2)?, k_out);
        debug_assert_eq!(biases.dim(D::Minus2)?, k_out);
        debug_assert_eq!(scales.dim(D::Minus1)?, groups_per_row);
        debug_assert_eq!(biases.dim(D::Minus1)?, groups_per_row);

        // Flatten everything
        let x_vec = x.flatten_all()?.to_dtype(DType::F16)?.to_vec1::<f16>()?;
        let w_codes = w.flatten_all()?.to_vec1::<u32>()?;
        let sc_vec = scales.flatten_all()?.to_vec1::<f32>()?;
        let bs_vec = biases.flatten_all()?.to_vec1::<f32>()?;

        let mut y = vec![0f32; m * k_out];

        let mask = (1u32 << bits) - 1;
        let codes_per_word = 32 / bits;
        let words_per_group = group_size / codes_per_word;

        dbg!(m, k_out, packed_n, codes_per_word);
        use rayon::prelude::*;
        for row_idx in 0..m {
            let x_row = &x_vec[row_idx * k..(row_idx + 1) * k];
            let y_row = &mut y[row_idx * k_out..(row_idx + 1) * k_out];

            // zero‑init the row
            for v in y_row.iter_mut() {
                *v = 0.0;
            }

            // Parallel over kk (output channels) for this row
            y_row.par_iter_mut().enumerate().for_each(|(kk, y_slot)| {
                let mut acc_vec = unsafe { vdupq_n_f32(0.0) };
                let sb_base = kk * groups_per_row;

                for packed_row_idx in 0..packed_n {
                    // Row‑major access: row = kk (output), col = packed_row_idx (input packed)
                    let word = w_codes[kk * packed_n + packed_row_idx];

                    // Unpack codes_per_word values
                    let mut local_codes = [0u32; 8]; // supports up to 8 codes per word
                    for c in 0..codes_per_word {
                        local_codes[c] = (word >> (c * bits)) & mask;
                    }

                    // Identify which group this packed_row_idx belongs to
                    let g = packed_row_idx * codes_per_word / group_size;
                    let scale = sc_vec[sb_base + g];
                    let bias = bs_vec[sb_base + g];

                    // For each code in this word (each input position)
                    for c in 0..codes_per_word {
                        let input_idx = packed_row_idx * codes_per_word + c;
                        if input_idx >= n {
                            break;
                        }
                        let x_scalar = x_row[input_idx].to_f32();
                        let q_val = local_codes[c] as f32;
                        let val = x_scalar * (q_val * scale + bias);

                        // Accumulate val into the correct lane
                        let lane = c % 4;
                        let mut tmp = [0.0f32; 4];
                        unsafe {
                            vst1q_f32(tmp.as_mut_ptr(), acc_vec);
                        }
                        tmp[lane] += val;
                        acc_vec = unsafe { vld1q_f32(tmp.as_ptr()) };
                    }
                }

                // Horizontal sum -> scalar and store
                unsafe {
                    let pair_sum = vadd_f32(vget_high_f32(acc_vec), vget_low_f32(acc_vec));
                    let scalar = vget_lane_f32(pair_sum, 0) + vget_lane_f32(pair_sum, 1);
                    *y_slot = scalar;
                }
            });
        }

        Tensor::from_vec(y, vec![m, k_out], &Device::Cpu)?.to_dtype(x.dtype())
    }

    /// Very simple (and slow) matmul after full de‑quantisation.  Handles 2‑D tensors.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn afq_mm_op(
        x: &Tensor,
        w: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        _lhs_indices: Option<&Tensor>,
        _rhs_indices: Option<&Tensor>,
        group_size: usize,
        bits: usize,
        transpose: bool,
    ) -> Result<Tensor> {
        // --- Try fast NEON path -------------------------------------------------
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if _lhs_indices.is_none() && _rhs_indices.is_none() && bits <= 8 {
                // dbg!(transpose);
                // dbg!(&x.squeeze(0)?);
                // dbg!(&w);
                // dbg!(&scales);
                // dbg!(&biases);
                let try_fast = if !transpose {
                    afq_mm_fast_neon(&x.squeeze(0)?, w, scales, biases, group_size, bits)
                } else {
                    afq_mm_fast_neon_t(&x.squeeze(0)?, w, scales, biases, group_size, bits)
                };
                // dbg!(&try_fast);
                // dbg!(&try_fast);
                if let Ok(t) = try_fast {
                    return Ok(t.unsqueeze(0)?);
                }
            }
            // Otherwise, fall through to the scalar fallback.
        }
        // ------------------------------------------------------------------------
        let w_f32 = afq_dequantize_op(w, scales, biases, group_size, bits)?.to_dtype(x.dtype())?;
        if transpose {
            x.broadcast_matmul(&w_f32.t()?)
        } else {
            x.broadcast_matmul(&w_f32)
        }
    }
}
