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

        let command_buffer = device.command_buffer()?;
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

        assert_eq!(w.layout().start_offset(), 0);
        crate::metal_kernels::call_affine_quantize(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            w.dtype(),
            w_s.buffer(),
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
        candle_core::bail!("`afq_quantize_op` only works on Metal.")
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
        candle_core::bail!("`afq_dequantize_op` only works on Metal.")
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
        candle_core::bail!("`afq_mm_op` only works on Metal.")
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
