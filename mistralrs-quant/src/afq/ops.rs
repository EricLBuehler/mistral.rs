#![allow(unused)]

use candle_core::{backend::BackendStorage, DType, Result, Shape, Storage, Tensor, D};

#[cfg(feature = "metal")]
use candle_core::MetalStorage;

#[cfg(feature = "cuda")]
use candle_core::{
    cuda::{cudarc::driver::DevicePtr, CudaStorageSlice},
    CudaStorage,
};

use super::{AfqBits, AfqGroupSize};

#[cfg(feature = "cuda")]
use crate::utils::get_cuda_device;

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

        let encoder = device.command_encoder()?;
        encoder.set_label("afq-quantize");

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
            &encoder,
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

        let output = Tensor::from((
            Storage::Metal(MetalStorage::new(
                output,
                device.clone(),
                wq_shape.iter().product(),
                DType::U32,
            )),
            Shape::from(wq_shape),
        ));
        let scales = Tensor::from((
            Storage::Metal(MetalStorage::new(
                scales,
                device.clone(),
                s_shape.iter().product(),
                w.dtype(),
            )),
            Shape::from(s_shape.clone()),
        ));
        let biases = Tensor::from((
            Storage::Metal(MetalStorage::new(
                biases,
                device.clone(),
                s_shape.iter().product(),
                w.dtype(),
            )),
            Shape::from(s_shape),
        ));

        return Ok((output, scales, biases));
    }
    #[cfg(feature = "cuda")]
    if w.device().is_cuda() {
        return cuda_backend::afq_quantize_op(w, group_size, bits);
    }

    // CPU fallback for non-accelerated devices
    cpu_backend::afq_quantize_op(w, group_size, bits)
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

        let encoder = device.command_encoder()?;
        encoder.set_label("afq-dequantize");

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
            &encoder,
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

        let output = Tensor::from((
            Storage::Metal(MetalStorage::new(
                output,
                device.clone(),
                w_shape.iter().product(),
                scales.dtype(),
            )),
            Shape::from(w_shape),
        ));

        return Ok(output);
    }
    #[cfg(feature = "cuda")]
    if w_q.device().is_cuda() {
        return cuda_backend::afq_dequantize_op(w_q, scales, biases, group_size, bits);
    }

    // CPU fallback for non-accelerated devices
    cpu_backend::afq_dequantize_op(w_q, scales, biases, group_size, bits)
}

fn make_dummy_indices(x: &Tensor) -> Result<Tensor> {
    let x_batches = x
        .dims()
        .iter()
        .take(x.rank() - 2)
        .copied()
        .collect::<Vec<_>>();

    Tensor::arange(0u32, x_batches.iter().product::<usize>() as u32, x.device())?.reshape(x_batches)

    // (Tensor::ones(x_batches.iter().product::<usize>(), DType::F32, x.device())?
    //     .cumsum(0)?
    //     .to_dtype(DType::U32)?
    //     - 1.)?
    //     .reshape(x_batches)
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
            assert_eq!(lhs_indices.layout().start_offset(), 0);
            assert_eq!(rhs_indices.layout().start_offset(), 0);
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

            let encoder = device.command_encoder()?;
            encoder.set_label("afq-qmm");

            let output =
                device.new_buffer(out_shape.iter().product(), scales.dtype(), "afq-qmm-output")?;

            crate::metal_kernels::call_afq_qmm(
                device.device(),
                &encoder,
                &crate::metal_kernels::Kernels::new(),
                scales.dtype(),
                x_s.buffer(),
                x.layout().start_offset() * x.dtype().size_in_bytes(),
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

            let encoder = device.command_encoder()?;
            encoder.set_label("afq-qmm");

            let output =
                device.new_buffer(out_shape.iter().product(), scales.dtype(), "afq-qmm-output")?;

            crate::metal_kernels::call_afq_qmm(
                device.device(),
                &encoder,
                &crate::metal_kernels::Kernels::new(),
                scales.dtype(),
                x_s.buffer(),
                x.layout().start_offset() * x.dtype().size_in_bytes(),
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

        let output = Tensor::from((
            Storage::Metal(MetalStorage::new(
                output,
                device.clone(),
                out_shape.iter().product(),
                scales.dtype(),
            )),
            Shape::from(out_shape),
        ));

        return Ok(output);
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return cuda_backend::afq_mm_op(
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

    // CPU fallback for non-accelerated devices
    cpu_backend::afq_mm_op(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        group_size,
        bits,
        transpose,
    )
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

        println!("w_q shape = {:?}, dtype = {:?}", w_q.shape(), w_q.dtype());
        println!(
            "scales shape = {:?}, dtype = {:?}",
            scales.shape(),
            scales.dtype()
        );
        println!(
            "biases shape = {:?}, dtype = {:?}",
            biases.shape(),
            biases.dtype()
        );
        println!(
            "First few w_q values: {:?}",
            w_q.flatten_all()?
                .to_vec1::<u32>()?
                .iter()
                .take(10)
                .collect::<Vec<_>>()
        );
        println!(
            "First few scales: {:?}",
            scales
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .take(5)
                .collect::<Vec<_>>()
        );
        println!(
            "First few biases: {:?}",
            biases
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .take(5)
                .collect::<Vec<_>>()
        );

        let ys = afq_dequantize_op(&w_q, &scales, &biases, group_size, bits)?;

        println!(
            "xs min/max: {:?}/{:?}",
            xs.min(D::Minus1)?.min_all()?.to_scalar::<f32>()?,
            xs.max(D::Minus1)?.max_all()?.to_scalar::<f32>()?
        );
        println!(
            "ys min/max: {:?}/{:?}",
            ys.min(D::Minus1)?.min_all()?.to_scalar::<f32>()?,
            ys.max(D::Minus1)?.max_all()?.to_scalar::<f32>()?
        );
        println!(
            "First few xs values: {:?}",
            xs.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .take(5)
                .collect::<Vec<_>>()
        );
        println!(
            "First few ys values: {:?}",
            ys.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .take(5)
                .collect::<Vec<_>>()
        );

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
mod cpu_backend {
    use super::*;
    use candle_core::{DType, Device, Result, Tensor, D};

    /// Simple scalar (reference) quantiser: per‑`group_size` affine.
    pub(crate) fn afq_quantize_op(
        w: &Tensor,
        group_size: usize,
        bits: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if bits == 40 {
            // mxfp4 is not supported in CPU backend
            candle_core::bail!("mxfp4 quantization is only supported on Metal backend");
        }
        let device = w.device().clone();
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

                    let q_mask = (1u32 << bits) - 1;
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
        )?
        .to_dtype(w.dtype())?;
        let bs = Tensor::from_vec(
            biases,
            {
                let mut d = w.dims().to_vec();
                *d.last_mut().unwrap() = groups_per_row;
                d
            },
            &device,
        )?
        .to_dtype(w.dtype())?;
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
        if _bits == 40 {
            // mxfp4 is not supported in CPU backend
            candle_core::bail!("mxfp4 dequantization is only supported on Metal backend");
        }
        let device = w_q.device().clone();
        let codes = w_q.flatten_all()?.to_vec1::<u32>()?;
        let sc = scales
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let bs = biases
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

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
        )?
        .to_dtype(scales.dtype())
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
        if bits == 40 {
            // mxfp4 is not supported in CPU backend
            candle_core::bail!("mxfp4 matmul is only supported on Metal backend");
        }
        let w_f32 = afq_dequantize_op(w, scales, biases, group_size, bits)?.to_dtype(x.dtype())?;
        if transpose {
            x.broadcast_matmul(&w_f32.t()?)
        } else {
            x.broadcast_matmul(&w_f32)
        }
    }
}

// ============================================================
//                    CUDA backend
// ============================================================
#[cfg(feature = "cuda")]
mod cuda_backend {
    use super::*;
    use crate::afq::ffi;
    use candle_core::{cuda::cudarc::driver::DevicePtr, CudaStorage, DType, Result, Tensor, D};
    use half::{bf16, f16};

    /// CUDA-accelerated AFQ quantization
    pub(crate) fn afq_quantize_op(
        w: &Tensor,
        group_size: usize,
        bits: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if bits == 40 {
            candle_core::bail!("mxfp4 quantization is not supported on CUDA backend");
        }
        if bits == 3 || bits == 6 {
            // Non-power-of-2 bit widths fall back to CPU for quantization
            return super::cpu_backend::afq_quantize_op(w, group_size, bits);
        }

        let dev = crate::utils::get_cuda_device(w)?;

        let (rows, cols) = (
            w.dims().iter().take(w.rank() - 1).product::<usize>(),
            w.dim(D::Minus1)?,
        );

        let packed_cols = cols * bits / 32;
        let groups_per_row = cols / group_size;

        // Allocate output tensors
        let w_q_shape: Vec<usize> = {
            let mut s = w.dims().to_vec();
            *s.last_mut().unwrap() = packed_cols;
            s
        };
        let s_shape: Vec<usize> = {
            let mut s = w.dims().to_vec();
            *s.last_mut().unwrap() = groups_per_row;
            s
        };

        // Dispatch based on dtype and bits/group_size
        // Each arm returns the final tensors directly
        match w.dtype() {
            DType::F16 => {
                let w_q_buf = unsafe { dev.alloc::<u32>(rows * packed_cols)? };
                let scales_buf = unsafe { dev.alloc::<f16>(rows * groups_per_row)? };
                let biases_buf = unsafe { dev.alloc::<f16>(rows * groups_per_row)? };

                let (w_s, _) = w.storage_and_layout();
                let Storage::Cuda(w_s) = &*w_s else {
                    candle_core::bail!("Expected CUDA storage");
                };
                let (w_ptr, _w_guard) =
                    crate::utils::slice_ptr(w_s.as_cuda_slice::<f16>()?, w.layout().start_offset());
                let (wq_ptr, wq_guard) = w_q_buf.device_ptr(w_q_buf.stream());
                let (s_ptr, s_guard) = scales_buf.device_ptr(scales_buf.stream());
                let (b_ptr, b_guard) = biases_buf.device_ptr(biases_buf.stream());

                unsafe {
                    match (bits, group_size) {
                        (2, 32) => ffi::afq_quantize_2bit_gs32_f16(
                            w_ptr as *const f16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f16,
                            b_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 64) => ffi::afq_quantize_2bit_gs64_f16(
                            w_ptr as *const f16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f16,
                            b_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 128) => ffi::afq_quantize_2bit_gs128_f16(
                            w_ptr as *const f16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f16,
                            b_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 32) => ffi::afq_quantize_4bit_gs32_f16(
                            w_ptr as *const f16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f16,
                            b_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 64) => ffi::afq_quantize_4bit_gs64_f16(
                            w_ptr as *const f16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f16,
                            b_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 128) => ffi::afq_quantize_4bit_gs128_f16(
                            w_ptr as *const f16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f16,
                            b_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 32) => ffi::afq_quantize_8bit_gs32_f16(
                            w_ptr as *const f16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f16,
                            b_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 64) => ffi::afq_quantize_8bit_gs64_f16(
                            w_ptr as *const f16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f16,
                            b_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 128) => ffi::afq_quantize_8bit_gs128_f16(
                            w_ptr as *const f16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f16,
                            b_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        _ => candle_core::bail!(
                            "Unsupported bits/group_size combination: {bits}/{group_size}"
                        ),
                    }
                }
                drop(wq_guard);
                drop(s_guard);
                drop(b_guard);

                let w_q_storage = CudaStorage::wrap_cuda_slice(w_q_buf, dev.clone());
                let w_q = Tensor::from((
                    Storage::Cuda(w_q_storage),
                    candle_core::Shape::from(w_q_shape),
                ));

                let scales_storage = CudaStorage::wrap_cuda_slice(scales_buf, dev.clone());
                let scales = Tensor::from((
                    Storage::Cuda(scales_storage),
                    candle_core::Shape::from(s_shape.clone()),
                ));

                let biases_storage = CudaStorage::wrap_cuda_slice(biases_buf, dev.clone());
                let biases = Tensor::from((
                    Storage::Cuda(biases_storage),
                    candle_core::Shape::from(s_shape),
                ));

                Ok((w_q, scales, biases))
            }
            DType::F32 => {
                let w_q_buf = unsafe { dev.alloc::<u32>(rows * packed_cols)? };
                let scales_buf = unsafe { dev.alloc::<f32>(rows * groups_per_row)? };
                let biases_buf = unsafe { dev.alloc::<f32>(rows * groups_per_row)? };

                let (w_s, _) = w.storage_and_layout();
                let Storage::Cuda(w_s) = &*w_s else {
                    candle_core::bail!("Expected CUDA storage");
                };
                let (w_ptr, _w_guard) =
                    crate::utils::slice_ptr(w_s.as_cuda_slice::<f32>()?, w.layout().start_offset());
                let (wq_ptr, wq_guard) = w_q_buf.device_ptr(w_q_buf.stream());
                let (s_ptr, s_guard) = scales_buf.device_ptr(scales_buf.stream());
                let (b_ptr, b_guard) = biases_buf.device_ptr(biases_buf.stream());

                unsafe {
                    match (bits, group_size) {
                        (2, 32) => ffi::afq_quantize_2bit_gs32_f32(
                            w_ptr as *const f32,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f32,
                            b_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 64) => ffi::afq_quantize_2bit_gs64_f32(
                            w_ptr as *const f32,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f32,
                            b_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 128) => ffi::afq_quantize_2bit_gs128_f32(
                            w_ptr as *const f32,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f32,
                            b_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 32) => ffi::afq_quantize_4bit_gs32_f32(
                            w_ptr as *const f32,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f32,
                            b_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 64) => ffi::afq_quantize_4bit_gs64_f32(
                            w_ptr as *const f32,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f32,
                            b_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 128) => ffi::afq_quantize_4bit_gs128_f32(
                            w_ptr as *const f32,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f32,
                            b_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 32) => ffi::afq_quantize_8bit_gs32_f32(
                            w_ptr as *const f32,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f32,
                            b_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 64) => ffi::afq_quantize_8bit_gs64_f32(
                            w_ptr as *const f32,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f32,
                            b_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 128) => ffi::afq_quantize_8bit_gs128_f32(
                            w_ptr as *const f32,
                            wq_ptr as *mut u32,
                            s_ptr as *mut f32,
                            b_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        _ => candle_core::bail!(
                            "Unsupported bits/group_size combination: {bits}/{group_size}"
                        ),
                    }
                }
                drop(wq_guard);
                drop(s_guard);
                drop(b_guard);

                let w_q_storage = CudaStorage::wrap_cuda_slice(w_q_buf, dev.clone());
                let w_q = Tensor::from((
                    Storage::Cuda(w_q_storage),
                    candle_core::Shape::from(w_q_shape),
                ));

                let scales_storage = CudaStorage::wrap_cuda_slice(scales_buf, dev.clone());
                let scales = Tensor::from((
                    Storage::Cuda(scales_storage),
                    candle_core::Shape::from(s_shape.clone()),
                ));

                let biases_storage = CudaStorage::wrap_cuda_slice(biases_buf, dev.clone());
                let biases = Tensor::from((
                    Storage::Cuda(biases_storage),
                    candle_core::Shape::from(s_shape),
                ));

                Ok((w_q, scales, biases))
            }
            DType::BF16 => {
                let w_q_buf = unsafe { dev.alloc::<u32>(rows * packed_cols)? };
                let scales_buf = unsafe { dev.alloc::<bf16>(rows * groups_per_row)? };
                let biases_buf = unsafe { dev.alloc::<bf16>(rows * groups_per_row)? };

                let (w_s, _) = w.storage_and_layout();
                let Storage::Cuda(w_s) = &*w_s else {
                    candle_core::bail!("Expected CUDA storage");
                };
                let (w_ptr, _w_guard) = crate::utils::slice_ptr(
                    w_s.as_cuda_slice::<bf16>()?,
                    w.layout().start_offset(),
                );
                let (wq_ptr, wq_guard) = w_q_buf.device_ptr(w_q_buf.stream());
                let (s_ptr, s_guard) = scales_buf.device_ptr(scales_buf.stream());
                let (b_ptr, b_guard) = biases_buf.device_ptr(biases_buf.stream());

                unsafe {
                    match (bits, group_size) {
                        (2, 32) => ffi::afq_quantize_2bit_gs32_bf16(
                            w_ptr as *const bf16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut bf16,
                            b_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 64) => ffi::afq_quantize_2bit_gs64_bf16(
                            w_ptr as *const bf16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut bf16,
                            b_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 128) => ffi::afq_quantize_2bit_gs128_bf16(
                            w_ptr as *const bf16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut bf16,
                            b_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 32) => ffi::afq_quantize_4bit_gs32_bf16(
                            w_ptr as *const bf16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut bf16,
                            b_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 64) => ffi::afq_quantize_4bit_gs64_bf16(
                            w_ptr as *const bf16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut bf16,
                            b_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 128) => ffi::afq_quantize_4bit_gs128_bf16(
                            w_ptr as *const bf16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut bf16,
                            b_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 32) => ffi::afq_quantize_8bit_gs32_bf16(
                            w_ptr as *const bf16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut bf16,
                            b_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 64) => ffi::afq_quantize_8bit_gs64_bf16(
                            w_ptr as *const bf16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut bf16,
                            b_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 128) => ffi::afq_quantize_8bit_gs128_bf16(
                            w_ptr as *const bf16,
                            wq_ptr as *mut u32,
                            s_ptr as *mut bf16,
                            b_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        _ => candle_core::bail!(
                            "Unsupported bits/group_size combination: {bits}/{group_size}"
                        ),
                    }
                }
                drop(wq_guard);
                drop(s_guard);
                drop(b_guard);

                let w_q_storage = CudaStorage::wrap_cuda_slice(w_q_buf, dev.clone());
                let w_q = Tensor::from((
                    Storage::Cuda(w_q_storage),
                    candle_core::Shape::from(w_q_shape),
                ));

                let scales_storage = CudaStorage::wrap_cuda_slice(scales_buf, dev.clone());
                let scales = Tensor::from((
                    Storage::Cuda(scales_storage),
                    candle_core::Shape::from(s_shape.clone()),
                ));

                let biases_storage = CudaStorage::wrap_cuda_slice(biases_buf, dev.clone());
                let biases = Tensor::from((
                    Storage::Cuda(biases_storage),
                    candle_core::Shape::from(s_shape),
                ));

                Ok((w_q, scales, biases))
            }
            other => candle_core::bail!("Unsupported dtype for AFQ CUDA quantization: {other:?}"),
        }
    }

    /// CUDA-accelerated AFQ dequantization
    pub(crate) fn afq_dequantize_op(
        w_q: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        group_size: usize,
        bits: usize,
    ) -> Result<Tensor> {
        if bits == 40 {
            candle_core::bail!("mxfp4 dequantization is not supported on CUDA backend");
        }

        let dev = crate::utils::get_cuda_device(w_q)?;

        let rows = w_q.dims().iter().take(w_q.rank() - 1).product::<usize>();
        // Calculate cols from scales tensor (works for all bit widths)
        let groups_per_row = scales.dim(D::Minus1)?;
        let cols = groups_per_row * group_size;

        let out_shape: Vec<usize> = {
            let mut s = w_q.dims().to_vec();
            *s.last_mut().unwrap() = cols;
            s
        };

        let (wq_s, _) = w_q.storage_and_layout();
        let Storage::Cuda(wq_s) = &*wq_s else {
            candle_core::bail!("Expected CUDA storage");
        };
        let (s_s, _) = scales.storage_and_layout();
        let Storage::Cuda(s_s) = &*s_s else {
            candle_core::bail!("Expected CUDA storage");
        };
        let (b_s, _) = biases.storage_and_layout();
        let Storage::Cuda(b_s) = &*b_s else {
            candle_core::bail!("Expected CUDA storage");
        };

        let (wq_ptr, _wq_guard) =
            crate::utils::slice_ptr(wq_s.as_cuda_slice::<u32>()?, w_q.layout().start_offset());

        match scales.dtype() {
            DType::F16 => {
                let output_buf = unsafe { dev.alloc::<f16>(rows * cols)? };
                let (s_ptr, _s_guard) = crate::utils::slice_ptr(
                    s_s.as_cuda_slice::<f16>()?,
                    scales.layout().start_offset(),
                );
                let (b_ptr, _b_guard) = crate::utils::slice_ptr(
                    b_s.as_cuda_slice::<f16>()?,
                    biases.layout().start_offset(),
                );
                let (out_ptr, out_guard) = output_buf.device_ptr(output_buf.stream());

                unsafe {
                    match (bits, group_size) {
                        (2, 32) => ffi::afq_dequantize_2bit_gs32_f16(
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 64) => ffi::afq_dequantize_2bit_gs64_f16(
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 128) => ffi::afq_dequantize_2bit_gs128_f16(
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (3, 32) => ffi::afq_dequantize_3bit_gs32_f16(
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (3, 64) => ffi::afq_dequantize_3bit_gs64_f16(
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (3, 128) => ffi::afq_dequantize_3bit_gs128_f16(
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 32) => ffi::afq_dequantize_4bit_gs32_f16(
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 64) => ffi::afq_dequantize_4bit_gs64_f16(
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 128) => ffi::afq_dequantize_4bit_gs128_f16(
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (6, 32) => ffi::afq_dequantize_6bit_gs32_f16(
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (6, 64) => ffi::afq_dequantize_6bit_gs64_f16(
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (6, 128) => ffi::afq_dequantize_6bit_gs128_f16(
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 32) => ffi::afq_dequantize_8bit_gs32_f16(
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 64) => ffi::afq_dequantize_8bit_gs64_f16(
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 128) => ffi::afq_dequantize_8bit_gs128_f16(
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            rows as i32,
                            cols as i32,
                        ),
                        _ => candle_core::bail!(
                            "Unsupported bits/group_size combination: {bits}/{group_size}"
                        ),
                    }
                }
                drop(out_guard);

                let output_storage = CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
                let output = Tensor::from((
                    Storage::Cuda(output_storage),
                    candle_core::Shape::from(out_shape),
                ));
                Ok(output)
            }
            DType::F32 => {
                let output_buf = unsafe { dev.alloc::<f32>(rows * cols)? };
                let (s_ptr, _s_guard) = crate::utils::slice_ptr(
                    s_s.as_cuda_slice::<f32>()?,
                    scales.layout().start_offset(),
                );
                let (b_ptr, _b_guard) = crate::utils::slice_ptr(
                    b_s.as_cuda_slice::<f32>()?,
                    biases.layout().start_offset(),
                );
                let (out_ptr, out_guard) = output_buf.device_ptr(output_buf.stream());

                unsafe {
                    match (bits, group_size) {
                        (2, 32) => ffi::afq_dequantize_2bit_gs32_f32(
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 64) => ffi::afq_dequantize_2bit_gs64_f32(
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 128) => ffi::afq_dequantize_2bit_gs128_f32(
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (3, 32) => ffi::afq_dequantize_3bit_gs32_f32(
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (3, 64) => ffi::afq_dequantize_3bit_gs64_f32(
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (3, 128) => ffi::afq_dequantize_3bit_gs128_f32(
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 32) => ffi::afq_dequantize_4bit_gs32_f32(
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 64) => ffi::afq_dequantize_4bit_gs64_f32(
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 128) => ffi::afq_dequantize_4bit_gs128_f32(
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (6, 32) => ffi::afq_dequantize_6bit_gs32_f32(
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (6, 64) => ffi::afq_dequantize_6bit_gs64_f32(
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (6, 128) => ffi::afq_dequantize_6bit_gs128_f32(
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 32) => ffi::afq_dequantize_8bit_gs32_f32(
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 64) => ffi::afq_dequantize_8bit_gs64_f32(
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 128) => ffi::afq_dequantize_8bit_gs128_f32(
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            rows as i32,
                            cols as i32,
                        ),
                        _ => candle_core::bail!(
                            "Unsupported bits/group_size combination: {bits}/{group_size}"
                        ),
                    }
                }
                drop(out_guard);

                let output_storage = CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
                let output = Tensor::from((
                    Storage::Cuda(output_storage),
                    candle_core::Shape::from(out_shape),
                ));
                Ok(output)
            }
            DType::BF16 => {
                let output_buf = unsafe { dev.alloc::<bf16>(rows * cols)? };
                let (s_ptr, _s_guard) = crate::utils::slice_ptr(
                    s_s.as_cuda_slice::<bf16>()?,
                    scales.layout().start_offset(),
                );
                let (b_ptr, _b_guard) = crate::utils::slice_ptr(
                    b_s.as_cuda_slice::<bf16>()?,
                    biases.layout().start_offset(),
                );
                let (out_ptr, out_guard) = output_buf.device_ptr(output_buf.stream());

                unsafe {
                    match (bits, group_size) {
                        (2, 32) => ffi::afq_dequantize_2bit_gs32_bf16(
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 64) => ffi::afq_dequantize_2bit_gs64_bf16(
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (2, 128) => ffi::afq_dequantize_2bit_gs128_bf16(
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (3, 32) => ffi::afq_dequantize_3bit_gs32_bf16(
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (3, 64) => ffi::afq_dequantize_3bit_gs64_bf16(
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (3, 128) => ffi::afq_dequantize_3bit_gs128_bf16(
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 32) => ffi::afq_dequantize_4bit_gs32_bf16(
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 64) => ffi::afq_dequantize_4bit_gs64_bf16(
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (4, 128) => ffi::afq_dequantize_4bit_gs128_bf16(
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (6, 32) => ffi::afq_dequantize_6bit_gs32_bf16(
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (6, 64) => ffi::afq_dequantize_6bit_gs64_bf16(
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (6, 128) => ffi::afq_dequantize_6bit_gs128_bf16(
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 32) => ffi::afq_dequantize_8bit_gs32_bf16(
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 64) => ffi::afq_dequantize_8bit_gs64_bf16(
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        (8, 128) => ffi::afq_dequantize_8bit_gs128_bf16(
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            rows as i32,
                            cols as i32,
                        ),
                        _ => candle_core::bail!(
                            "Unsupported bits/group_size combination: {bits}/{group_size}"
                        ),
                    }
                }
                drop(out_guard);

                let output_storage = CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
                let output = Tensor::from((
                    Storage::Cuda(output_storage),
                    candle_core::Shape::from(out_shape),
                ));
                Ok(output)
            }
            other => candle_core::bail!("Unsupported dtype for AFQ CUDA dequantization: {other:?}"),
        }
    }

    /// CUDA-accelerated AFQ fused matmul
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
        if bits == 40 {
            candle_core::bail!("mxfp4 matmul is not supported on CUDA backend");
        }

        // For indexed matmul, fall back to dequantize + matmul for now
        if _lhs_indices.is_some() || _rhs_indices.is_some() {
            let w_dequant =
                afq_dequantize_op(w, scales, biases, group_size, bits)?.to_dtype(x.dtype())?;
            return if transpose {
                x.broadcast_matmul(&w_dequant.t()?)
            } else {
                x.broadcast_matmul(&w_dequant)
            };
        }

        if !transpose {
            // Non-transposed matmul: fall back to dequantize + matmul
            let w_dequant =
                afq_dequantize_op(w, scales, biases, group_size, bits)?.to_dtype(x.dtype())?;
            return x.broadcast_matmul(&w_dequant);
        }

        // Transposed case: y = x @ W^T
        // x: [M, K], W: [N, K], y: [M, N]
        let dev = crate::utils::get_cuda_device(x)?;

        let x_rank = x.rank();
        let (m, k) = (
            x.dims().iter().take(x_rank - 1).product::<usize>(),
            x.dim(D::Minus1)?,
        );
        let n = w.dim(D::Minus2)?;
        // Calculate actual_k from scales tensor (works for all bit widths)
        let groups_per_row = scales.dim(D::Minus1)?;
        let actual_k = groups_per_row * group_size;

        if k != actual_k {
            candle_core::bail!(
                "x inner dim ({k}) does not match w inner dim ({actual_k}) for transposed matmul"
            );
        }

        let out_shape: Vec<usize> = {
            let mut s = x.dims().to_vec();
            *s.last_mut().unwrap() = n;
            s
        };

        let (x_s, _) = x.storage_and_layout();
        let Storage::Cuda(x_s) = &*x_s else {
            candle_core::bail!("Expected CUDA storage");
        };
        let (w_s, _) = w.storage_and_layout();
        let Storage::Cuda(w_s) = &*w_s else {
            candle_core::bail!("Expected CUDA storage");
        };
        let (s_s, _) = scales.storage_and_layout();
        let Storage::Cuda(s_s) = &*s_s else {
            candle_core::bail!("Expected CUDA storage");
        };
        let (b_s, _) = biases.storage_and_layout();
        let Storage::Cuda(b_s) = &*b_s else {
            candle_core::bail!("Expected CUDA storage");
        };

        let (wq_ptr, _wq_guard) =
            crate::utils::slice_ptr(w_s.as_cuda_slice::<u32>()?, w.layout().start_offset());

        match x.dtype() {
            DType::F16 => {
                let output_buf = unsafe { dev.alloc::<f16>(m * n)? };
                let (x_ptr, _x_guard) =
                    crate::utils::slice_ptr(x_s.as_cuda_slice::<f16>()?, x.layout().start_offset());
                let (s_ptr, _s_guard) = crate::utils::slice_ptr(
                    s_s.as_cuda_slice::<f16>()?,
                    scales.layout().start_offset(),
                );
                let (b_ptr, _b_guard) = crate::utils::slice_ptr(
                    b_s.as_cuda_slice::<f16>()?,
                    biases.layout().start_offset(),
                );
                let (out_ptr, out_guard) = output_buf.device_ptr(output_buf.stream());

                // Use QMV kernel for fused quantized matmul
                unsafe {
                    match (bits, group_size) {
                        (2, 32) => ffi::afq_qmv_2bit_gs32_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (2, 64) => ffi::afq_qmv_2bit_gs64_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (2, 128) => ffi::afq_qmv_2bit_gs128_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (3, 32) => ffi::afq_qmv_3bit_gs32_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (3, 64) => ffi::afq_qmv_3bit_gs64_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (3, 128) => ffi::afq_qmv_3bit_gs128_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (4, 32) => ffi::afq_qmv_4bit_gs32_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (4, 64) => ffi::afq_qmv_4bit_gs64_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (4, 128) => ffi::afq_qmv_4bit_gs128_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (6, 32) => ffi::afq_qmv_6bit_gs32_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (6, 64) => ffi::afq_qmv_6bit_gs64_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (6, 128) => ffi::afq_qmv_6bit_gs128_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u8,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (8, 32) => ffi::afq_qmv_8bit_gs32_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (8, 64) => ffi::afq_qmv_8bit_gs64_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (8, 128) => ffi::afq_qmv_8bit_gs128_f16(
                            x_ptr as *const f16,
                            wq_ptr as *const u32,
                            s_ptr as *const f16,
                            b_ptr as *const f16,
                            out_ptr as *mut f16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        _ => candle_core::bail!(
                            "Unsupported bits/group_size combination: {bits}/{group_size}"
                        ),
                    }
                }
                drop(out_guard);

                let output_storage = CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
                let output = Tensor::from((
                    Storage::Cuda(output_storage),
                    candle_core::Shape::from(out_shape),
                ));
                Ok(output)
            }
            DType::F32 => {
                let output_buf = unsafe { dev.alloc::<f32>(m * n)? };
                let (x_ptr, _x_guard) =
                    crate::utils::slice_ptr(x_s.as_cuda_slice::<f32>()?, x.layout().start_offset());
                let (s_ptr, _s_guard) = crate::utils::slice_ptr(
                    s_s.as_cuda_slice::<f32>()?,
                    scales.layout().start_offset(),
                );
                let (b_ptr, _b_guard) = crate::utils::slice_ptr(
                    b_s.as_cuda_slice::<f32>()?,
                    biases.layout().start_offset(),
                );
                let (out_ptr, out_guard) = output_buf.device_ptr(output_buf.stream());

                unsafe {
                    match (bits, group_size) {
                        (2, 32) => ffi::afq_qmv_2bit_gs32_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (2, 64) => ffi::afq_qmv_2bit_gs64_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (2, 128) => ffi::afq_qmv_2bit_gs128_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (3, 32) => ffi::afq_qmv_3bit_gs32_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (3, 64) => ffi::afq_qmv_3bit_gs64_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (3, 128) => ffi::afq_qmv_3bit_gs128_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (4, 32) => ffi::afq_qmv_4bit_gs32_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (4, 64) => ffi::afq_qmv_4bit_gs64_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (4, 128) => ffi::afq_qmv_4bit_gs128_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (6, 32) => ffi::afq_qmv_6bit_gs32_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (6, 64) => ffi::afq_qmv_6bit_gs64_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (6, 128) => ffi::afq_qmv_6bit_gs128_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u8,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (8, 32) => ffi::afq_qmv_8bit_gs32_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (8, 64) => ffi::afq_qmv_8bit_gs64_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (8, 128) => ffi::afq_qmv_8bit_gs128_f32(
                            x_ptr as *const f32,
                            wq_ptr as *const u32,
                            s_ptr as *const f32,
                            b_ptr as *const f32,
                            out_ptr as *mut f32,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        _ => candle_core::bail!(
                            "Unsupported bits/group_size combination: {bits}/{group_size}"
                        ),
                    }
                }
                drop(out_guard);

                let output_storage = CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
                let output = Tensor::from((
                    Storage::Cuda(output_storage),
                    candle_core::Shape::from(out_shape),
                ));
                Ok(output)
            }
            DType::BF16 => {
                let output_buf = unsafe { dev.alloc::<bf16>(m * n)? };
                let (x_ptr, _x_guard) = crate::utils::slice_ptr(
                    x_s.as_cuda_slice::<bf16>()?,
                    x.layout().start_offset(),
                );
                let (s_ptr, _s_guard) = crate::utils::slice_ptr(
                    s_s.as_cuda_slice::<bf16>()?,
                    scales.layout().start_offset(),
                );
                let (b_ptr, _b_guard) = crate::utils::slice_ptr(
                    b_s.as_cuda_slice::<bf16>()?,
                    biases.layout().start_offset(),
                );
                let (out_ptr, out_guard) = output_buf.device_ptr(output_buf.stream());

                // Use QMV kernel for fused quantized matmul
                unsafe {
                    match (bits, group_size) {
                        (2, 32) => ffi::afq_qmv_2bit_gs32_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (2, 64) => ffi::afq_qmv_2bit_gs64_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (2, 128) => ffi::afq_qmv_2bit_gs128_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (3, 32) => ffi::afq_qmv_3bit_gs32_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (3, 64) => ffi::afq_qmv_3bit_gs64_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (3, 128) => ffi::afq_qmv_3bit_gs128_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (4, 32) => ffi::afq_qmv_4bit_gs32_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (4, 64) => ffi::afq_qmv_4bit_gs64_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (4, 128) => ffi::afq_qmv_4bit_gs128_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (6, 32) => ffi::afq_qmv_6bit_gs32_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (6, 64) => ffi::afq_qmv_6bit_gs64_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (6, 128) => ffi::afq_qmv_6bit_gs128_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u8,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (8, 32) => ffi::afq_qmv_8bit_gs32_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (8, 64) => ffi::afq_qmv_8bit_gs64_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        (8, 128) => ffi::afq_qmv_8bit_gs128_bf16(
                            x_ptr as *const bf16,
                            wq_ptr as *const u32,
                            s_ptr as *const bf16,
                            b_ptr as *const bf16,
                            out_ptr as *mut bf16,
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                        _ => candle_core::bail!(
                            "Unsupported bits/group_size combination: {bits}/{group_size}"
                        ),
                    }
                }
                drop(out_guard);

                let output_storage = CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
                let output = Tensor::from((
                    Storage::Cuda(output_storage),
                    candle_core::Shape::from(out_shape),
                ));
                Ok(output)
            }
            other => candle_core::bail!("Unsupported dtype for AFQ CUDA matmul: {other:?}"),
        }
    }
}
