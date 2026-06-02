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

            // Split-K (rank-2 only): partition K across grid.z when the
            // output tile count alone would leave the GPU underutilized.
            if transpose && x.rank() == 2 {
                let m_outer = x.dim(0)?;
                let n_out = w_outer_dims;
                let k_in = x.dim(1)?;
                const BM: usize = 32;
                const BN: usize = 32;
                const TARGET_TGS: usize = 512;
                let n_tiles = n_out.div_ceil(BN);
                let m_tiles = m_outer.div_ceil(BM);
                let current_tgs = n_tiles * m_tiles;
                let mut split_k = (TARGET_TGS / current_tgs.max(1)).max(1);
                if k_in == 0 || group_size == 0 {
                    split_k = 1;
                } else {
                    split_k = split_k.min(k_in / group_size);
                    while split_k > 1 && k_in % (split_k * group_size) != 0 {
                        split_k -= 1;
                    }
                }
                if split_k >= 2 {
                    let intermediate_elems = split_k * m_outer * n_out;
                    let intermediate = device.new_buffer(
                        intermediate_elems,
                        scales.dtype(),
                        "afq-qmm-splitk-intermediate",
                    )?;
                    let encoder = device.command_encoder()?;
                    encoder.set_label("afq-qmm-splitk");
                    crate::metal_kernels::call_afq_qmm_splitk(
                        device.device(),
                        &encoder,
                        &crate::metal_kernels::Kernels::new(),
                        scales.dtype(),
                        x_s.buffer(),
                        x.layout().start_offset() * x.dtype().size_in_bytes(),
                        w_s.buffer(),
                        s_s.buffer(),
                        b_s.buffer(),
                        &intermediate,
                        m_outer,
                        n_out,
                        k_in,
                        split_k,
                        bits,
                        group_size,
                    )
                    .map_err(candle_core::Error::wrap)?;

                    let intermediate_tensor = Tensor::from((
                        Storage::Metal(MetalStorage::new(
                            intermediate,
                            device.clone(),
                            intermediate_elems,
                            scales.dtype(),
                        )),
                        Shape::from(vec![split_k, m_outer, n_out]),
                    ));
                    return intermediate_tensor.sum(0);
                }
            }

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

/// Stable wrapper around candle's `call_mlx_arg_sort` for u32 keys. Candle's
/// `Tensor::arg_sort_last_dim` uses a single-threadgroup bitonic sort that
/// silently returns garbage for n > 1024 on Metal; this routes around it by
/// calling the multi-block sort directly. Returns u32 perm of shape [n].
/// `Kernels` is cached process-wide so the metallib only compiles once.
#[cfg(feature = "metal")]
pub fn metal_arg_sort_u32_1d(keys: &Tensor) -> Result<Tensor> {
    use std::sync::OnceLock;
    static KERNELS: OnceLock<candle_metal_kernels::Kernels> = OnceLock::new();

    if keys.rank() != 1 {
        candle_core::bail!(
            "metal_arg_sort_u32_1d expects rank 1; got {:?}",
            keys.dims()
        );
    }
    if keys.dtype() != DType::U32 {
        candle_core::bail!("metal_arg_sort_u32_1d expects u32; got {:?}", keys.dtype());
    }
    if !keys.is_contiguous() {
        candle_core::bail!("metal_arg_sort_u32_1d expects contiguous input");
    }

    let n = keys.dim(0)?;
    let storage = keys.storage_and_layout().0;
    let Storage::Metal(s) = &*storage else {
        candle_core::bail!("expected metal storage");
    };
    let device = s.device();
    let dst = device.new_buffer(n, DType::U32, "argsort-perm")?;

    let cmk_device: &candle_metal_kernels::metal::Device = device.device();
    let kernels = KERNELS.get_or_init(candle_metal_kernels::Kernels::new);
    let encoder = device.command_encoder()?;
    encoder.set_label("mlx-argsort");
    let src_offset = keys.layout().start_offset() * DType::U32.size_in_bytes();
    let src = candle_metal_kernels::BufferOffset {
        buffer: s.buffer(),
        offset_in_bytes: src_offset,
    };
    candle_metal_kernels::call_mlx_arg_sort(
        cmk_device,
        &encoder,
        kernels,
        candle_metal_kernels::DType::U32,
        /* nrows */ 1,
        /* ncols */ n,
        src,
        &dst,
    )
    .map_err(candle_core::Error::wrap)?;

    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(dst, device.clone(), n, DType::U32)),
        Shape::from(vec![n]),
    )))
}

/// Fused topk-weighted reduce: collapses `[num_tokens * topk, hidden]`
/// per-assignment outputs into `[num_tokens, hidden]` by topk-weighted sum.
/// One Metal launch replaces the `reshape -> to_dtype(F32) -> broadcast_mul ->
/// sum -> to_dtype` tail. Accepts BF16/F16/F32 input, f32 weights.
#[cfg(feature = "metal")]
pub fn metal_moe_weighted_reduce_flat(
    inputs: &Tensor,
    topk_weights: &Tensor,
    num_tokens: usize,
    topk: usize,
) -> Result<Tensor> {
    if inputs.rank() != 2 {
        candle_core::bail!(
            "metal_moe_weighted_reduce_flat: inputs must be rank 2 [M,H], got {:?}",
            inputs.dims()
        );
    }
    let total_assignments = inputs.dim(0)?;
    let hidden = inputs.dim(1)?;
    if total_assignments != num_tokens * topk {
        candle_core::bail!(
            "metal_moe_weighted_reduce_flat: input rows {total_assignments} != num_tokens {num_tokens} * topk {topk}"
        );
    }
    if topk_weights.elem_count() != total_assignments {
        candle_core::bail!(
            "metal_moe_weighted_reduce_flat: topk_weights must have {total_assignments} elements, got {}",
            topk_weights.elem_count()
        );
    }
    if !matches!(inputs.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!(
            "metal_moe_weighted_reduce_flat: unsupported input dtype {:?}",
            inputs.dtype()
        );
    }

    let inputs = inputs.contiguous()?;
    let topk_weights = topk_weights
        .flatten_all()?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let (in_storage, in_layout) = inputs.storage_and_layout();
    let Storage::Metal(in_s) = &*in_storage else {
        candle_core::bail!("metal_moe_weighted_reduce_flat: inputs must live on Metal");
    };
    let (tw_storage, tw_layout) = topk_weights.storage_and_layout();
    let Storage::Metal(tw_s) = &*tw_storage else {
        candle_core::bail!("metal_moe_weighted_reduce_flat: topk_weights must live on Metal");
    };

    let device = in_s.device();
    let out_elems = num_tokens * hidden;
    let dtype = inputs.dtype();
    let output = device.new_buffer(out_elems, dtype, "moe-weighted-reduce")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("moe-weighted-reduce");
    crate::metal_kernels::call_moe_weighted_reduce_flat(
        device.device(),
        &encoder,
        &crate::metal_kernels::Kernels::new(),
        dtype,
        in_s.buffer(),
        in_layout.start_offset() * dtype.size_in_bytes(),
        tw_s.buffer(),
        tw_layout.start_offset() * DType::F32.size_in_bytes(),
        &output,
        num_tokens,
        hidden,
        topk,
    )
    .map_err(candle_core::Error::wrap)?;

    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(output, device.clone(), out_elems, dtype)),
        Shape::from(vec![num_tokens, hidden]),
    )))
}

/// Sorted-MoE tiled grouped GEMM via the MLX-ported `affine_gather_qmm_rhs`
/// kernel. `x_sorted` is `[M, K]` row-contiguous with rows pre-sorted so that
/// rows mapped to the same expert are contiguous. `sorted_expert_ids` is `[M]`
/// u32 of expert ids in the same order. `w` is `[E, N, K]` (transpose=true).
/// Output is `[M, N]` in the sorted row order; caller is responsible for the
/// inverse permutation back to the natural token-expert order.
#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
pub fn afq_gather_qmm_rhs_sorted(
    x_sorted: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    sorted_expert_ids: &Tensor,
    group_size: AfqGroupSize,
    bits: AfqBits,
) -> Result<Tensor> {
    let group_size = group_size as usize;
    let bits = bits as usize;

    if w.dtype() != DType::U32 {
        candle_core::bail!("AFQ weight matrix must be u32");
    }
    if scales.dims() != biases.dims() {
        candle_core::bail!("Scales and biases must share shape");
    }
    if x_sorted.rank() != 2 {
        candle_core::bail!(
            "afq_gather_qmm_rhs_sorted expects x_sorted rank 2 [M,K]; got {:?}",
            x_sorted.dims()
        );
    }
    if w.rank() != 3 {
        candle_core::bail!(
            "afq_gather_qmm_rhs_sorted expects w rank 3 [E,N,K] (transpose=true); got {:?}",
            w.dims()
        );
    }
    if sorted_expert_ids.dtype() != DType::U32 || sorted_expert_ids.rank() != 1 {
        candle_core::bail!(
            "sorted_expert_ids must be u32 rank-1; got dtype={:?} rank={}",
            sorted_expert_ids.dtype(),
            sorted_expert_ids.rank()
        );
    }

    let m = x_sorted.dim(0)?;
    let k = x_sorted.dim(1)?;
    let n = w.dim(1)?;
    let k_w = w.dim(2)? * 32 / bits;
    if k != k_w {
        candle_core::bail!("x_sorted K ({k}) must match w K ({k_w})");
    }
    if sorted_expert_ids.dim(0)? != m {
        candle_core::bail!(
            "sorted_expert_ids len ({}) must match M ({m})",
            sorted_expert_ids.dim(0)?
        );
    }
    assert_eq!(x_sorted.layout().start_offset(), 0);
    assert_eq!(w.layout().start_offset(), 0);
    assert_eq!(scales.layout().start_offset(), 0);
    assert_eq!(biases.layout().start_offset(), 0);
    assert_eq!(sorted_expert_ids.layout().start_offset(), 0);

    let x_s = x_sorted.storage_and_layout().0;
    let Storage::Metal(x_s) = &*x_s else {
        candle_core::bail!("expected metal x_sorted")
    };
    let w_s = w.storage_and_layout().0;
    let Storage::Metal(w_s) = &*w_s else {
        candle_core::bail!("expected metal w")
    };
    let s_s = scales.storage_and_layout().0;
    let Storage::Metal(s_s) = &*s_s else {
        candle_core::bail!("expected metal scales")
    };
    let b_s = biases.storage_and_layout().0;
    let Storage::Metal(b_s) = &*b_s else {
        candle_core::bail!("expected metal biases")
    };
    let i_s = sorted_expert_ids.storage_and_layout().0;
    let Storage::Metal(i_s) = &*i_s else {
        candle_core::bail!("expected metal sorted_expert_ids")
    };

    let device = w_s.device();
    let out_shape = vec![m, n];
    let output = device.new_buffer(out_shape.iter().product(), scales.dtype(), "afq-gather-rhs")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("afq-gather-rhs");

    crate::metal_kernels::call_afq_gather_qmm_rhs(
        device.device(),
        &encoder,
        &crate::metal_kernels::Kernels::new(),
        scales.dtype(),
        x_s.buffer(),
        x_sorted.layout().start_offset() * x_sorted.dtype().size_in_bytes(),
        w_s.buffer(),
        s_s.buffer(),
        b_s.buffer(),
        i_s.buffer(),
        &output,
        m,
        n,
        k,
        bits,
        group_size,
    )
    .map_err(candle_core::Error::wrap)?;

    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            out_shape.iter().product(),
            scales.dtype(),
        )),
        Shape::from(out_shape),
    )))
}

/// Fused gate+up sorted-MoE kernel: computes `y = activation(gate(x)) * up(x)`
/// in a single launch. Saves one matmul + the intermediate gate buffer + the
/// extra x global reads. `act_idx` matches the codes used by qmm_t_gate_up:
/// 0=Silu/Swish, 1=Gelu(tanh approx), 2=Gelu(erf approx), 3=Relu.
#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
pub fn afq_gather_qmm_rhs_sorted_gate_up(
    x_sorted: &Tensor,
    w_gate: &Tensor,
    scales_gate: &Tensor,
    biases_gate: &Tensor,
    w_up: &Tensor,
    scales_up: &Tensor,
    biases_up: &Tensor,
    sorted_expert_ids: &Tensor,
    group_size: AfqGroupSize,
    bits: AfqBits,
    act_idx: usize,
) -> Result<Tensor> {
    let group_size = group_size as usize;
    let bits = bits as usize;

    if x_sorted.rank() != 2 {
        candle_core::bail!("expects x_sorted rank 2 [M,K]; got {:?}", x_sorted.dims());
    }
    for (name, w) in [("w_gate", w_gate), ("w_up", w_up)] {
        if w.dtype() != DType::U32 {
            candle_core::bail!("{name} must be u32");
        }
        if w.rank() != 3 {
            candle_core::bail!("{name} expects rank 3 [E,N,K]; got {:?}", w.dims());
        }
    }
    if scales_gate.dims() != biases_gate.dims() || scales_up.dims() != biases_up.dims() {
        candle_core::bail!("Scales/biases shape mismatch");
    }
    if w_gate.dims() != w_up.dims() || scales_gate.dims() != scales_up.dims() {
        candle_core::bail!("Gate and up weight shapes must match");
    }
    if sorted_expert_ids.dtype() != DType::U32 || sorted_expert_ids.rank() != 1 {
        candle_core::bail!("sorted_expert_ids must be u32 rank-1");
    }

    let m = x_sorted.dim(0)?;
    let k = x_sorted.dim(1)?;
    let n = w_gate.dim(1)?;
    let k_w = w_gate.dim(2)? * 32 / bits;
    if k != k_w {
        candle_core::bail!("x K ({k}) must match w K ({k_w})");
    }
    if sorted_expert_ids.dim(0)? != m {
        candle_core::bail!("sorted_expert_ids len must be M");
    }
    for t in [
        x_sorted,
        w_gate,
        scales_gate,
        biases_gate,
        w_up,
        scales_up,
        biases_up,
        sorted_expert_ids,
    ] {
        assert_eq!(t.layout().start_offset(), 0);
    }

    let extract = |t: &Tensor, lbl: &str| -> Result<_> {
        let s = t.storage_and_layout().0;
        let Storage::Metal(s) = &*s else {
            candle_core::bail!("expected metal {lbl}")
        };
        Ok(s.clone())
    };
    let x_s = extract(x_sorted, "x_sorted")?;
    let wg_s = extract(w_gate, "w_gate")?;
    let sg_s = extract(scales_gate, "scales_gate")?;
    let bg_s = extract(biases_gate, "biases_gate")?;
    let wu_s = extract(w_up, "w_up")?;
    let su_s = extract(scales_up, "scales_up")?;
    let bu_s = extract(biases_up, "biases_up")?;
    let i_s = extract(sorted_expert_ids, "sorted_expert_ids")?;

    let device = wg_s.device();
    let out_shape = vec![m, n];
    let output = device.new_buffer(
        out_shape.iter().product(),
        scales_gate.dtype(),
        "afq-gather-rhs-gate-up",
    )?;
    let encoder = device.command_encoder()?;
    encoder.set_label("afq-gather-rhs-gate-up");

    crate::metal_kernels::call_afq_gather_qmm_rhs_gate_up(
        device.device(),
        &encoder,
        &crate::metal_kernels::Kernels::new(),
        scales_gate.dtype(),
        x_s.buffer(),
        x_sorted.layout().start_offset() * x_sorted.dtype().size_in_bytes(),
        wg_s.buffer(),
        sg_s.buffer(),
        bg_s.buffer(),
        wu_s.buffer(),
        su_s.buffer(),
        bu_s.buffer(),
        i_s.buffer(),
        &output,
        m,
        n,
        k,
        bits,
        group_size,
        act_idx,
    )
    .map_err(candle_core::Error::wrap)?;

    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            out_shape.iter().product(),
            scales_gate.dtype(),
        )),
        Shape::from(out_shape),
    )))
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
        assert!(rmse < 0.09, "{rmse}");
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
        assert!(rmse < 0.40, "{rmse}");
        Ok(())
    }

    #[test]
    fn test_metal_arg_sort_u32_1d() -> Result<()> {
        use crate::afq::ops::metal_arg_sort_u32_1d;
        let device = Device::new_metal(0)?;
        for n in [1024usize, 4096, 8192, 32768] {
            let data: Vec<u32> = (0..n as u32).rev().collect();
            let t = Tensor::from_vec(data.clone(), (n,), &device)?;
            let perm = metal_arg_sort_u32_1d(&t)?;
            let p = perm.to_vec1::<u32>()?;
            for i in 0..n {
                assert_eq!(
                    p[i],
                    (n - 1 - i) as u32,
                    "n={n} idx={i}: perm[{i}]={} expected {}",
                    p[i],
                    n - 1 - i
                );
            }
            let sorted_keys = t.gather(&perm, 0)?.to_vec1::<u32>()?;
            for i in 0..n {
                assert_eq!(
                    sorted_keys[i], i as u32,
                    "n={n} sorted[{i}]={} expected {i}",
                    sorted_keys[i]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_argsort_large() -> Result<()> {
        let device = Device::new_metal(0)?;
        for n in [1024usize, 4096, 8192, 16384, 32768] {
            let data: Vec<u32> = (0..n as u32).rev().collect();
            let t = Tensor::from_vec(data, (n,), &device)?;
            let perm = t.arg_sort_last_dim(true)?;
            let p = perm.to_vec1::<u32>()?;
            let ok = p[0] == (n - 1) as u32 && p[n - 1] == 0;
            println!("n={n} perm[0]={} perm[-1]={} ok={ok}", p[0], p[n - 1]);
        }
        Ok(())
    }

    // Cross-check the MLX-ported sorted-MoE kernel against a dequantized
    // reference. Catches RoPE-style layout / dispatch / unalignment bugs
    // before they hit the benchmark.
    #[test]
    fn test_afq_gather_qmm_rhs_sorted_matches_dequant_ref() -> Result<()> {
        use crate::afq::ops::{afq_dequantize_op, afq_gather_qmm_rhs_sorted, afq_quantize_op};

        let device = Device::new_metal(0)?;
        let group_size = AfqGroupSize::Med;
        let bits = AfqBits::Eight;
        let bits_usize = bits as usize;
        let gs_usize = group_size as usize;

        // Includes M < 128 (BM=16 path), 128 <= M < 512 (BM=32), M >= 512 (BM=64),
        // and an unaligned-M case for the unaligned branches.
        let cases: &[(usize, usize, usize, usize)] = &[
            (4, 64, 64, 64),
            (4, 17, 64, 64),
            (4, 64, 96, 64),
            (4, 64, 64, 128),
            (8, 128, 704, 2816),
            (8, 200, 704, 2816),
            (8, 512, 704, 2816),
            (8, 600, 704, 2816),
            (8, 2048, 704, 2816),
        ];
        for &(num_experts, m, n, k) in cases {
            assert!(k % gs_usize == 0);
            let w = Tensor::randn(0f32, 0.02f32, (num_experts, n, k), &device)?;
            let (w_q, scales, biases) = afq_quantize_op(&w, group_size, bits)?;
            let w_dequant = afq_dequantize_op(&w_q, &scales, &biases, group_size, bits)?;

            let x = Tensor::randn(0f32, 1f32, (m, k), &device)?;

            let ids_vec: Vec<u32> = (0..m)
                .map(|i| ((i * 7 + 13) % num_experts) as u32)
                .collect();
            let mut sorted: Vec<u32> = ids_vec.clone();
            sorted.sort();
            let sorted_ids = Tensor::from_vec(sorted.clone(), (m,), &device)?;

            let y_new = afq_gather_qmm_rhs_sorted(
                &x,
                &w_q,
                &scales,
                &biases,
                &sorted_ids,
                group_size,
                bits,
            )?;

            let w_sel = w_dequant.index_select(&sorted_ids, 0)?;
            let y_ref = x
                .unsqueeze(1)?
                .matmul(&w_sel.transpose(1, 2)?)?
                .squeeze(1)?;
            let diff = (y_new - y_ref.clone())?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let scale = y_ref.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                diff <= scale * 1e-2,
                "case e={num_experts} m={m} n={n} k={k}: max diff {diff} exceeds 1e-2 * |ref| ({})",
                scale * 1e-2
            );
            let _ = bits_usize;
        }
        Ok(())
    }

    #[test]
    fn test_metal_moe_weighted_reduce_flat_matches_ref() -> Result<()> {
        use crate::afq::ops::metal_moe_weighted_reduce_flat;
        let device = Device::new_metal(0)?;

        // Mix of small (cache-friendly) and Gemma-4-26B-A4B-shape cases.
        let cases: &[(usize, usize, usize)] = &[(2, 2, 8), (4, 8, 64), (4096, 8, 2816)];
        for &(num_tokens, topk, hidden) in cases {
            let m = num_tokens * topk;
            let inputs = Tensor::randn(0f32, 1f32, (m, hidden), &device)?.to_dtype(DType::BF16)?;
            let topk_weights = Tensor::randn(0f32, 0.5f32, (num_tokens, topk), &device)?;

            let y_new = metal_moe_weighted_reduce_flat(&inputs, &topk_weights, num_tokens, topk)?;

            let y_ref = inputs
                .reshape((num_tokens, topk, hidden))?
                .to_dtype(DType::F32)?
                .broadcast_mul(&topk_weights.unsqueeze(D::Minus1)?)?
                .sum(D::Minus2)?
                .to_dtype(DType::BF16)?;
            let diff = (y_new - y_ref.clone())?
                .abs()?
                .max_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?;
            let scale = y_ref
                .abs()?
                .max_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?
                .max(1e-3);
            assert!(
                diff <= scale * 5e-2,
                "num_tokens={num_tokens} topk={topk} hidden={hidden}: diff {diff} > 5e-2 * |ref| ({})",
                scale * 5e-2
            );
        }
        Ok(())
    }

    #[test]
    fn test_afq_gather_qmm_rhs_sorted_gate_up_matches_dequant_ref() -> Result<()> {
        use crate::afq::ops::{
            afq_dequantize_op, afq_gather_qmm_rhs_sorted_gate_up, afq_quantize_op,
        };

        let device = Device::new_metal(0)?;
        let group_size = AfqGroupSize::Med;
        let bits = AfqBits::Eight;

        // Test with act=0 (Silu), since it's simple to reference.
        let cases: &[(usize, usize, usize, usize)] =
            &[(4, 64, 64, 64), (8, 128, 704, 2816), (8, 600, 704, 2816)];
        for &(num_experts, m, n, k) in cases {
            let wg = Tensor::randn(0f32, 0.02f32, (num_experts, n, k), &device)?;
            let wu = Tensor::randn(0f32, 0.02f32, (num_experts, n, k), &device)?;
            let (wg_q, sg, bg) = afq_quantize_op(&wg, group_size, bits)?;
            let (wu_q, su, bu) = afq_quantize_op(&wu, group_size, bits)?;
            let wg_d = afq_dequantize_op(&wg_q, &sg, &bg, group_size, bits)?;
            let wu_d = afq_dequantize_op(&wu_q, &su, &bu, group_size, bits)?;

            let x = Tensor::randn(0f32, 1f32, (m, k), &device)?;

            let ids_vec: Vec<u32> = (0..m)
                .map(|i| ((i * 7 + 13) % num_experts) as u32)
                .collect();
            let mut sorted: Vec<u32> = ids_vec.clone();
            sorted.sort();
            let sorted_ids = Tensor::from_vec(sorted, (m,), &device)?;

            let y_new = afq_gather_qmm_rhs_sorted_gate_up(
                &x,
                &wg_q,
                &sg,
                &bg,
                &wu_q,
                &su,
                &bu,
                &sorted_ids,
                group_size,
                bits,
                /* act=Silu */ 0,
            )?;

            let wg_sel = wg_d.index_select(&sorted_ids, 0)?;
            let wu_sel = wu_d.index_select(&sorted_ids, 0)?;
            let gate = x
                .unsqueeze(1)?
                .matmul(&wg_sel.transpose(1, 2)?)?
                .squeeze(1)?;
            let up = x
                .unsqueeze(1)?
                .matmul(&wu_sel.transpose(1, 2)?)?
                .squeeze(1)?;
            let y_ref = (gate.silu()? * up)?;
            let diff = (y_new - y_ref.clone())?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let scale = y_ref.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
            assert!(
                diff <= scale * 5e-2,
                "case e={num_experts} m={m} n={n} k={k}: gate_up diff {diff} > 5e-2 * {scale}"
            );
        }
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
