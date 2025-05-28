#![allow(unused)]

use candle_core::{
    backend::BackendStorage, from_storage_no_op, DType, MetalStorage, Result, Shape, Storage,
    Tensor, D,
};

use candle_core::quantized::{GgmlDType, GgmlType};

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
        cpu_backend::afq_quantize_op(w, group_size, bits)
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
        cpu_backend::afq_dequantize_op(w_q, scales, biases, group_size, bits)
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
mod cpu_backend {
    #[derive(Clone, Copy, Debug)]
    pub struct BlockAfq<const BITS: usize, const GROUP: usize, const NQ: usize> {
        /// Packed little‑endian codes.  One u32 holds `32 / BITS` values.
        pub qs: [u32; NQ],
        pub d: [f32; GROUP],
        pub m: [f32; GROUP],
    }

    impl<const BITS: usize, const GROUP: usize, const NQ: usize> BlockAfq<BITS, GROUP, NQ> {
        /// De‑quantise one block into `dst` (len == GROUP)
        fn dequantize_row(&self, dst: &mut [f32]) {
            let mask = (1u32 << BITS) - 1;
            let vals_per_word = 32 / BITS;

            for i in 0..GROUP {
                let word_idx = i / vals_per_word;
                let shift = (i % vals_per_word) * BITS;
                let q = (self.qs[word_idx] >> shift) & mask;
                dst[i] = self.m[i] + (q as f32) * self.d[i];
            }
        }

        /// Quantise `src` (len == GROUP) into `self`.
        fn quantize_row(src: &[f32], dst: &mut Self) {
            assert_eq!(src.len(), GROUP);

            let levels = ((1u32 << BITS) - 1) as f32;
            dst.qs.fill(0);

            for i in 0..GROUP {
                let bias = *src
                    .iter()
                    .take(GROUP)
                    .fold(&f32::MAX, |a, b| if b < a { b } else { a });
                let max = *src
                    .iter()
                    .take(GROUP)
                    .fold(&f32::MIN, |a, b| if b > a { b } else { a });
                let scale = if (max - bias).abs() < 1e-12 {
                    1.0
                } else {
                    (max - bias) / levels
                };

                dst.d[i] = scale;
                dst.m[i] = bias;

                // Quantise
                let q_val = ((src[i] - bias) / scale).round().clamp(0.0, levels) as u32;
                let vals_per_word = 32 / BITS;
                let word_idx = i / vals_per_word;
                let shift = (i % vals_per_word) * BITS;
                dst.qs[word_idx] |= q_val << shift;
            }
        }
    }

    impl<const BITS: usize, const GROUP: usize, const NQ: usize> GgmlType
        for BlockAfq<BITS, GROUP, NQ>
    {
        // Candle expects these two associated constants
        const BLCK_SIZE: usize = GROUP;
        const DTYPE: GgmlDType = match BITS {
            2 => GgmlDType::Q2K,
            3 => GgmlDType::Q3K,
            4 => GgmlDType::Q4K,
            6 => GgmlDType::Q6K,
            8 => GgmlDType::Q8K,
            _ => unreachable!(),
        };

        type VecDotType = Self; // dot product works block-vs-block

        fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
            if ys.len() != xs.len() * GROUP {
                candle_core::bail!(
                    "to_float: dst len {} must equal {}×{}",
                    ys.len(),
                    xs.len(),
                    GROUP
                );
            }
            for (blk_idx, blk) in xs.iter().enumerate() {
                let off = blk_idx * GROUP;
                blk.dequantize_row(&mut ys[off..off + GROUP]);
            }
            Ok(())
        }

        fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
            if xs.len() != ys.len() * GROUP {
                candle_core::bail!(
                    "from_float: src len {} must equal {}×{}",
                    xs.len(),
                    ys.len(),
                    GROUP
                );
            }
            for (blk_idx, blk) in ys.iter_mut().enumerate() {
                let off = blk_idx * GROUP;
                Self::quantize_row(&xs[off..off + GROUP], blk);
            }
            Ok(())
        }

        // ------------------------------------------------------------------
        //      Dot-product stubs – provide specialised kernels later
        // ------------------------------------------------------------------
        fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
            unimplemented!("vec_dot is not yet implemented for BlockAfq");
        }

        fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
            unimplemented!("vec_dot_unopt is not yet implemented for BlockAfq");
        }
    }

    use super::*;
    use candle_core::{
        quantized::{k_quants::BlockQ3K, GgmlDType},
        DType, Device, Result, Tensor, D,
    };
    use half::f16;
    use itertools::izip;

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

        // Grouped: one Vec per row, each containing one Vec<u32> per group
        let mut row_group_q: Vec<Vec<Vec<u32>>> = Vec::with_capacity(outer);
        // Group‑level scales (one f32 per group)
        let mut row_group_scales: Vec<Vec<f32>> = Vec::with_capacity(outer);
        let mut row_group_biases: Vec<Vec<f32>> = Vec::with_capacity(outer);
        // ----------------------------------------------------------------------
        for row in 0..outer {
            let mut qs_row: Vec<u32> = Vec::with_capacity(inner);
            let mut sc_row: Vec<f32> = Vec::with_capacity(inner);
            let mut bs_row: Vec<f32> = Vec::with_capacity(inner);
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
                sc_row.push(scale); // same scale for all `group_size` elements
                bs_row.push(bias); // same bias for all `group_size` elements

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
                    qs_row.push(q_val);
                }
            }
            // --- build per‑group buckets for this row ------------------------------
            let groups_q: Vec<Vec<u32>> = qs_row
                .chunks(group_size)
                .map(|chunk| chunk.to_vec())
                .collect();
            let groups_sc: Vec<f32> = sc_row;
            let groups_bs: Vec<f32> = bs_row;

            row_group_q.push(groups_q);
            row_group_scales.push(groups_sc);
            row_group_biases.push(groups_bs);
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
        // Show dimensions of the first row’s grouped structure for sanity.
        dbg!(
            row_group_q[0].len(),    // number of groups
            row_group_q[0][0].len(), // codes per group
            row_group_scales[0][0],  // first scale
            row_group_biases[0][0]   // first bias
        );

        // ------------------------------------------------------------------
        //  Build typed AFQ blocks (BlockAfq<4, 64>) for every row
        // ------------------------------------------------------------------
        if group_size != 64 || bits != 4 {
            panic!("BlockAfq helper currently supports only group_size = 64 and bits = 4");
        }

        // One Vec<Vec<BlockAfq<4, 64>>> : outer rows → groups
        let mut row_blocks: Vec<Vec<BlockAfq<4, 64, 16>>> = Vec::with_capacity(outer);

        for (groups_q, groups_sc, groups_bs) in
            izip!(&row_group_q, &row_group_scales, &row_group_biases)
        {
            let mut blocks: Vec<BlockAfq<4, 64, 16>> = Vec::with_capacity(groups_q.len());

            for (q_codes, &scale, &bias) in
                itertools::izip!(groups_q, groups_sc.iter(), groups_bs.iter())
            {
                // Pack 64 q-codes (4-bit) into 16 u32 words.
                let mut blk = BlockAfq::<4, 64, 16> {
                    qs: [0u32; 16],
                    d: [scale; 64], // broadcast per-group scale
                    m: [bias; 64],  // broadcast per-group bias
                };

                for (i, &q) in q_codes.iter().enumerate() {
                    let word = i / 8; // 8 values per u32 when bits = 4
                    let shift = (i % 8) * 4;
                    blk.qs[word] |= (q & 0xF) << shift;
                }

                blocks.push(blk);
            }

            row_blocks.push(blocks);
        }

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
        dbg!(&w, &scales, &biases);
        let x_vec = x.flatten_all()?.to_dtype(DType::F16)?.to_vec1::<f16>()?;
        let w_codes = w.flatten_all()?.to_vec1::<u32>()?;
        let sc_vec = scales.flatten_all()?.to_vec1::<f32>()?;
        let bs_vec = biases.flatten_all()?.to_vec1::<f32>()?;
        todo!()
    }
}

// ============================================================
//                    Portable CPU back‑end
// ============================================================
#[cfg(not(feature = "metal"))]
mod cpu_backend {
    use super::*;
    use candle_core::{DType, Device, Result, Tensor, D};

    /// Simple scalar (reference) quantiser: per‑`group_size` affine.
    pub(crate) fn afq_quantize_op(
        w: &Tensor,
        group_size: usize,
        bits: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
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
        let w_f32 = afq_dequantize_op(w, scales, biases, group_size, bits)?.to_dtype(x.dtype())?;
        if transpose {
            x.broadcast_matmul(&w_f32.t()?)
        } else {
            x.broadcast_matmul(&w_f32)
        }
    }
}
