#[cfg(feature = "cuda")]
mod ffi;

use candle_core::{
    backend::BackendStorage, CpuStorage, CustomOp3, Layout, Result, Shape, Storage, Tensor,
    WithDType,
};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct RotaryEmb {
    is_neox: bool,
}

impl RotaryEmb {
    fn cache_dims(&self, l_src: &Layout, l_cos: &Layout, l_sin: &Layout) -> Result<(usize, usize)> {
        let (batch, _, seq_len, head_dim) = l_src.shape().dims4()?;
        let (cos_rows, rot_dim) = match l_cos.shape().dims() {
            [rows, dim] => (*rows, *dim),
            [cos_batch, cos_seq, dim] if *cos_batch == batch && *cos_seq == seq_len => {
                (batch * seq_len, *dim)
            }
            _ => candle_core::bail!("invalid RoPE cos shape {:?}", l_cos.shape()),
        };
        let (sin_rows, sin_dim) = match l_sin.shape().dims() {
            [rows, dim] => (*rows, *dim),
            [sin_batch, sin_seq, dim] if *sin_batch == batch && *sin_seq == seq_len => {
                (batch * seq_len, *dim)
            }
            _ => candle_core::bail!("invalid RoPE sin shape {:?}", l_sin.shape()),
        };
        if (cos_rows, rot_dim) != (sin_rows, sin_dim) {
            candle_core::bail!(
                "RoPE cos/sin shape mismatch {:?} {:?}",
                l_cos.shape(),
                l_sin.shape()
            );
        }
        if cos_rows != seq_len && cos_rows != batch * seq_len {
            candle_core::bail!(
                "RoPE cache rows {cos_rows} are incompatible with batch {batch} and seq {seq_len}"
            );
        }
        if rot_dim == 0 || rot_dim * 2 > head_dim {
            candle_core::bail!(
                "RoPE rot dim {} is incompatible with head dim {head_dim}",
                rot_dim * 2
            );
        }
        Ok((cos_rows, rot_dim))
    }
}

impl CustomOp3 for RotaryEmb {
    fn name(&self) -> &'static str {
        "mistralrs-rotary"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
            is_neox: bool,
        ) -> Result<(CpuStorage, Shape)>
        where
            T: WithDType
                + Copy
                + Send
                + Sync
                + std::ops::Add<Output = T>
                + std::ops::Sub<Output = T>
                + std::ops::Mul<Output = T>,
        {
            let src = match l_src.contiguous_offsets() {
                Some((o1, o2)) => &src[o1..o2],
                None => candle_core::bail!("RoPE input must be contiguous"),
            };
            let cos = match l_cos.contiguous_offsets() {
                Some((o1, o2)) => &cos[o1..o2],
                None => candle_core::bail!("RoPE cos must be contiguous"),
            };
            let sin = match l_sin.contiguous_offsets() {
                Some((o1, o2)) => &sin[o1..o2],
                None => candle_core::bail!("RoPE sin must be contiguous"),
            };
            let (batch, heads, seq_len, head_dim) = l_src.shape().dims4()?;
            let (cache_rows, rot_dim) = RotaryEmb { is_neox }.cache_dims(l_src, l_cos, l_sin)?;
            let mut dst = src.to_vec();
            dst.par_chunks_mut(head_dim)
                .enumerate()
                .for_each(|(row, dst)| {
                    let batch_idx = row / (heads * seq_len);
                    let seq_idx = row % seq_len;
                    let cache_row = if cache_rows == batch * seq_len {
                        batch_idx * seq_len + seq_idx
                    } else {
                        seq_idx
                    };
                    let cache_offset = cache_row * rot_dim;
                    for pair_idx in 0..rot_dim {
                        let (x_idx, y_idx) = if is_neox {
                            (pair_idx, pair_idx + rot_dim)
                        } else {
                            (pair_idx * 2, pair_idx * 2 + 1)
                        };
                        let x = dst[x_idx];
                        let y = dst[y_idx];
                        let cos = cos[cache_offset + pair_idx];
                        let sin = sin[cache_offset + pair_idx];
                        dst[x_idx] = x * cos - y * sin;
                        dst[y_idx] = y * cos + x * sin;
                    }
                });
            Ok((T::to_cpu_storage_owned(dst), l_src.shape().clone()))
        }

        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3, self.is_neox),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3, self.is_neox),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3, self.is_neox),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3, self.is_neox),
            _ => candle_core::bail!(
                "unsupported RoPE dtype {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let (batch, heads, seq_len, head_dim) = l1.shape().dims4()?;
        let (cache_rows, rot_dim) = self.cache_dims(l1, l2, l3)?;
        let dtype = s1.dtype();
        if s2.dtype() != dtype || s3.dtype() != dtype {
            candle_core::bail!(
                "RoPE dtype mismatch {:?} {:?} {:?}",
                dtype,
                s2.dtype(),
                s3.dtype()
            );
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rotary");
        let elem_count = l1.shape().elem_count();
        let output = device.new_buffer(elem_count, dtype, "rotary-output")?;

        crate::metal_kernels::call_rotary(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            dtype,
            s1.buffer(),
            s2.buffer(),
            s3.buffer(),
            l1.start_offset() * dtype.size_in_bytes(),
            l2.start_offset() * dtype.size_in_bytes(),
            l3.start_offset() * dtype.size_in_bytes(),
            batch,
            heads,
            seq_len,
            head_dim,
            rot_dim,
            cache_rows,
            self.is_neox,
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        let storage = candle_core::MetalStorage::new(output, device.clone(), elem_count, dtype);
        Ok((storage, l1.shape().clone()))
    }
}

pub fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor, is_neox: bool) -> Result<Tensor> {
    let x = x.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    x.apply_op3_no_bwd(&cos, &sin, &RotaryEmb { is_neox })
}

#[cfg(feature = "metal")]
#[derive(Clone, Copy)]
struct RotaryDims {
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    rot_dim: usize,
    cache_rows: usize,
}

#[cfg(feature = "metal")]
fn rotary_dims(x: &Tensor, cos: &Tensor, sin: &Tensor, positioned: bool) -> Result<RotaryDims> {
    let (batch, heads, seq_len, head_dim) = x.dims4()?;
    let (cache_rows, rot_dim) = if positioned {
        cos.shape().dims2()?
    } else {
        match cos.dims() {
            [rows, dim] => (*rows, *dim),
            [cos_batch, cos_seq, dim] if *cos_batch == batch && *cos_seq == seq_len => {
                (batch * seq_len, *dim)
            }
            _ => candle_core::bail!("invalid RoPE cos shape {:?}", cos.shape()),
        }
    };
    let (sin_rows, sin_dim) = if positioned {
        sin.shape().dims2()?
    } else {
        match sin.dims() {
            [rows, dim] => (*rows, *dim),
            [sin_batch, sin_seq, dim] if *sin_batch == batch && *sin_seq == seq_len => {
                (batch * seq_len, *dim)
            }
            _ => candle_core::bail!("invalid RoPE sin shape {:?}", sin.shape()),
        }
    };
    if (cache_rows, rot_dim) != (sin_rows, sin_dim) {
        candle_core::bail!(
            "RoPE cos/sin shape mismatch {:?} {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if !positioned && cache_rows != seq_len && cache_rows != batch * seq_len {
        candle_core::bail!(
            "RoPE cache rows {cache_rows} are incompatible with batch {batch} and seq {seq_len}"
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        candle_core::bail!(
            "RoPE rot dim {} is incompatible with head dim {head_dim}",
            rot_dim * 2
        );
    }
    Ok(RotaryDims {
        batch,
        heads,
        seq_len,
        head_dim,
        rot_dim,
        cache_rows,
    })
}

fn check_qk_shape(q: &Tensor, k: &Tensor) -> Result<usize> {
    let (batch, _, seq_len, head_dim) = q.dims4()?;
    let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
        candle_core::bail!("q/k RoPE shape mismatch {:?} {:?}", q.shape(), k.shape());
    }
    Ok(k_heads)
}

fn typed_slice<'a, T>(xs: &'a [T], layout: &Layout, name: &'static str) -> Result<&'a [T]> {
    match layout.contiguous_offsets() {
        Some((start, end)) => Ok(&xs[start..end]),
        None => candle_core::bail!("{name} must be contiguous for RoPE"),
    }
}

fn cpu_positions<'a>(
    storage_and_layout: &'a Option<(std::sync::RwLockReadGuard<'a, Storage>, &'a Layout)>,
) -> Result<Option<&'a [u32]>> {
    let Some((storage, layout)) = storage_and_layout else {
        return Ok(None);
    };
    let Storage::Cpu(CpuStorage::U32(positions)) = &**storage else {
        candle_core::bail!("RoPE positions must be CPU u32");
    };
    Ok(Some(typed_slice(positions, layout, "positions")?))
}

struct CpuRotaryInput<'a, T> {
    src: &'a [T],
    src_l: &'a Layout,
    cos: &'a [T],
    cos_l: &'a Layout,
    sin: &'a [T],
    sin_l: &'a Layout,
    positions: Option<&'a [u32]>,
    is_neox: bool,
}

fn apply_rotary_cpu_inner<T>(input: CpuRotaryInput<'_, T>) -> Result<Tensor>
where
    T: WithDType
        + Copy
        + Send
        + Sync
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>,
{
    let CpuRotaryInput {
        src,
        src_l,
        cos,
        cos_l,
        sin,
        sin_l,
        positions,
        is_neox,
    } = input;
    let src = typed_slice(src, src_l, "RoPE input")?;
    let cos = typed_slice(cos, cos_l, "RoPE cos")?;
    let sin = typed_slice(sin, sin_l, "RoPE sin")?;
    let (batch, heads, seq_len, head_dim) = src_l.shape().dims4()?;
    let positioned = positions.is_some();
    let (cache_rows, rot_dim) = {
        let (cache_rows, rot_dim) = if positioned {
            cos_l.shape().dims2()?
        } else {
            RotaryEmb { is_neox }.cache_dims(src_l, cos_l, sin_l)?
        };
        if positioned && sin_l.shape().dims2()? != (cache_rows, rot_dim) {
            candle_core::bail!(
                "RoPE cos/sin shape mismatch {:?} {:?}",
                cos_l.shape(),
                sin_l.shape()
            );
        }
        if rot_dim == 0 || rot_dim * 2 > head_dim {
            candle_core::bail!(
                "RoPE rot dim {} is incompatible with head dim {head_dim}",
                rot_dim * 2
            );
        }
        (cache_rows, rot_dim)
    };
    if let Some(positions) = positions {
        if positions.len() != batch {
            candle_core::bail!(
                "RoPE positions length {} does not match batch {batch}",
                positions.len()
            );
        }
        for position in positions {
            if *position as usize + seq_len > cache_rows {
                candle_core::bail!(
                    "RoPE position {} with seq {seq_len} exceeds cache rows {}",
                    position,
                    cache_rows
                );
            }
        }
    }
    let mut dst = src.to_vec();
    dst.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(row, dst)| {
            let batch_idx = row / (heads * seq_len);
            let seq_idx = row % seq_len;
            let cache_row = if let Some(positions) = positions {
                positions[batch_idx] as usize + seq_idx
            } else if cache_rows == batch * seq_len {
                batch_idx * seq_len + seq_idx
            } else {
                seq_idx
            };
            let cache_offset = cache_row * rot_dim;
            for pair_idx in 0..rot_dim {
                let (x_idx, y_idx) = if is_neox {
                    (pair_idx, pair_idx + rot_dim)
                } else {
                    (pair_idx * 2, pair_idx * 2 + 1)
                };
                let x = dst[x_idx];
                let y = dst[y_idx];
                let cos = cos[cache_offset + pair_idx];
                let sin = sin[cache_offset + pair_idx];
                dst[x_idx] = x * cos - y * sin;
                dst[y_idx] = y * cos + x * sin;
            }
        });
    Tensor::from_vec(dst, src_l.shape().clone(), &candle_core::Device::Cpu)
}

fn cpu_apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.map(Tensor::contiguous).transpose()?;
    let position_storage_and_layout = positions.as_ref().map(Tensor::storage_and_layout);
    let positions = cpu_positions(&position_storage_and_layout)?;

    let (q_s, q_l) = q.storage_and_layout();
    let (cos_s, cos_l) = cos.storage_and_layout();
    let (sin_s, sin_l) = sin.storage_and_layout();
    match (&*q_s, &*cos_s, &*sin_s) {
        (
            Storage::Cpu(CpuStorage::BF16(q)),
            Storage::Cpu(CpuStorage::BF16(cos)),
            Storage::Cpu(CpuStorage::BF16(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        (
            Storage::Cpu(CpuStorage::F16(q)),
            Storage::Cpu(CpuStorage::F16(cos)),
            Storage::Cpu(CpuStorage::F16(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        (
            Storage::Cpu(CpuStorage::F32(q)),
            Storage::Cpu(CpuStorage::F32(cos)),
            Storage::Cpu(CpuStorage::F32(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        (
            Storage::Cpu(CpuStorage::F64(q)),
            Storage::Cpu(CpuStorage::F64(cos)),
            Storage::Cpu(CpuStorage::F64(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        _ => candle_core::bail!(
            "unsupported CPU RoPE dtype {:?} {:?} {:?}",
            q.dtype(),
            cos.dtype(),
            sin.dtype()
        ),
    }
}

fn cpu_apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    if q.dtype() != k.dtype() {
        candle_core::bail!("q/k dtype mismatch {:?} {:?}", q.dtype(), k.dtype());
    }
    check_qk_shape(q, k)?;
    let (q_out, k_out) = rayon::join(
        || cpu_apply_rotary_q(q, cos, sin, positions, is_neox),
        || cpu_apply_rotary_q(k, cos, sin, positions, is_neox),
    );
    Ok((q_out?, k_out?))
}

#[cfg(feature = "cuda")]
fn restore_cuda_rope_layout(
    x: Tensor,
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Tensor> {
    x.reshape((batch, seq_len, heads, head_dim))?
        .transpose(1, 2)
}

#[cfg(feature = "cuda")]
fn cuda_apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    let (batch, heads, seq_len, head_dim) = q.dims4()?;
    let q_embed = q.transpose(1, 2)?.flatten(0, 1)?;
    if let Some(positions) = positions {
        apply_rotary_inplace_q_positions(&q_embed, cos, sin, positions, is_neox)?;
    } else {
        apply_rotary_inplace_q(&q_embed, cos, sin, is_neox)?;
    }
    restore_cuda_rope_layout(q_embed, batch, heads, seq_len, head_dim)
}

#[cfg(feature = "cuda")]
fn cuda_apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
        candle_core::bail!("q/k RoPE shape mismatch {:?} {:?}", q.shape(), k.shape());
    }
    let q_embed = q.transpose(1, 2)?.flatten(0, 1)?;
    let k_embed = k.transpose(1, 2)?.flatten(0, 1)?;
    if let Some(positions) = positions {
        apply_rotary_inplace_positions(&q_embed, &k_embed, cos, sin, positions, is_neox)?;
    } else {
        apply_rotary_inplace(&q_embed, &k_embed, cos, sin, is_neox)?;
    }
    Ok((
        restore_cuda_rope_layout(q_embed, batch, q_heads, seq_len, head_dim)?,
        restore_cuda_rope_layout(k_embed, batch, k_heads, seq_len, head_dim)?,
    ))
}

#[cfg(feature = "metal")]
fn metal_tensor(storage: Storage, shape: Shape) -> Tensor {
    Tensor::from((storage, shape))
}

#[cfg(feature = "metal")]
fn metal_apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    use candle_core::MetalStorage;

    let q = q.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.map(Tensor::contiguous).transpose()?;
    let dims = rotary_dims(&q, &cos, &sin, positions.is_some())?;
    if let Some(positions) = positions.as_ref() {
        if positions.dtype() != candle_core::DType::U32 || positions.dims1()? != dims.batch {
            candle_core::bail!("RoPE positions must be u32 with length {}", dims.batch);
        }
    }

    let (q_s, q_l) = q.storage_and_layout();
    let (cos_s, cos_l) = cos.storage_and_layout();
    let (sin_s, sin_l) = sin.storage_and_layout();
    let q_s = match &*q_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("q must be a Metal tensor"),
    };
    let cos_s = match &*cos_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("cos must be a Metal tensor"),
    };
    let sin_s = match &*sin_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("sin must be a Metal tensor"),
    };
    let device = q_s.device();
    let output = device.new_buffer(q_l.shape().elem_count(), q_s.dtype(), "rotary-q")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("rotary-q");

    if let Some(positions) = positions.as_ref() {
        let (positions_s, positions_l) = positions.storage_and_layout();
        let positions_s = match &*positions_s {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("positions must be a Metal tensor"),
        };
        crate::metal_kernels::call_rotary_q_positions(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            q_s.dtype(),
            q_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            positions_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            positions_l.start_offset() * positions_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            is_neox,
            &output,
        )
    } else {
        crate::metal_kernels::call_rotary_q(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            q_s.dtype(),
            q_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            dims.cache_rows,
            is_neox,
            &output,
        )
    }
    .map_err(candle_core::Error::wrap)?;

    Ok(metal_tensor(
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            q_l.shape().elem_count(),
            q_s.dtype(),
        )),
        q_l.shape().clone(),
    ))
}

#[cfg(feature = "metal")]
fn metal_apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    use candle_core::MetalStorage;

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.map(Tensor::contiguous).transpose()?;
    let dims = rotary_dims(&q, &cos, &sin, positions.is_some())?;
    let k_heads = check_qk_shape(&q, &k)?;
    if let Some(positions) = positions.as_ref() {
        if positions.dtype() != candle_core::DType::U32 || positions.dims1()? != dims.batch {
            candle_core::bail!("RoPE positions must be u32 with length {}", dims.batch);
        }
    }

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (cos_s, cos_l) = cos.storage_and_layout();
    let (sin_s, sin_l) = sin.storage_and_layout();
    let q_s = match &*q_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("q must be a Metal tensor"),
    };
    let k_s = match &*k_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("k must be a Metal tensor"),
    };
    let cos_s = match &*cos_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("cos must be a Metal tensor"),
    };
    let sin_s = match &*sin_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("sin must be a Metal tensor"),
    };
    let device = q_s.device();
    let q_out = device.new_buffer(q_l.shape().elem_count(), q_s.dtype(), "rotary-q")?;
    let k_out = device.new_buffer(k_l.shape().elem_count(), k_s.dtype(), "rotary-k")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("rotary-qk");

    if let Some(positions) = positions.as_ref() {
        let (positions_s, positions_l) = positions.storage_and_layout();
        let positions_s = match &*positions_s {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("positions must be a Metal tensor"),
        };
        crate::metal_kernels::call_rotary_qk_positions(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            q_s.dtype(),
            q_s.buffer(),
            k_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            positions_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            k_l.start_offset() * k_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            positions_l.start_offset() * positions_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            k_heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            is_neox,
            &q_out,
            &k_out,
        )
    } else {
        crate::metal_kernels::call_rotary_qk(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            q_s.dtype(),
            q_s.buffer(),
            k_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            k_l.start_offset() * k_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            k_heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            dims.cache_rows,
            is_neox,
            &q_out,
            &k_out,
        )
    }
    .map_err(candle_core::Error::wrap)?;

    Ok((
        metal_tensor(
            Storage::Metal(MetalStorage::new(
                q_out,
                device.clone(),
                q_l.shape().elem_count(),
                q_s.dtype(),
            )),
            q_l.shape().clone(),
        ),
        metal_tensor(
            Storage::Metal(MetalStorage::new(
                k_out,
                device.clone(),
                k_l.shape().elem_count(),
                k_s.dtype(),
            )),
            k_l.shape().clone(),
        ),
    ))
}

pub fn apply_rotary_q(q: &Tensor, cos: &Tensor, sin: &Tensor, is_neox: bool) -> Result<Tensor> {
    apply_rotary_q_inner(q, cos, sin, None, is_neox)
}

pub fn apply_rotary_q_positions(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<Tensor> {
    apply_rotary_q_inner(q, cos, sin, Some(positions), is_neox)
}

fn apply_rotary_q_inner(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        return cuda_apply_rotary_q(q, cos, sin, positions, is_neox);
    }
    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        return metal_apply_rotary_q(q, cos, sin, positions, is_neox);
    }
    cpu_apply_rotary_q(q, cos, sin, positions, is_neox)
}

pub fn apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    apply_rotary_qk_inner(q, k, cos, sin, None, is_neox)
}

pub fn apply_rotary_qk_positions(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    apply_rotary_qk_inner(q, k, cos, sin, Some(positions), is_neox)
}

fn apply_rotary_qk_inner(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    if q.dtype() != k.dtype() {
        candle_core::bail!("q/k dtype mismatch {:?} {:?}", q.dtype(), k.dtype());
    }
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        return cuda_apply_rotary_qk(q, k, cos, sin, positions, is_neox);
    }
    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        return metal_apply_rotary_qk(q, k, cos, sin, positions, is_neox);
    }
    cpu_apply_rotary_qk(q, k, cos, sin, positions, is_neox)
}

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::{
        backend::{BackendDevice, BackendStorage},
        cuda_backend::{CudaDType, CudaStorage, CudaStorageSlice},
        CpuStorage, DType, InplaceOp3, Layout, MetalStorage, Result, Storage, Tensor,
    };
    use half::{bf16, f16};
    use std::ffi::{c_int, c_long};

    use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

    fn rotary_dtype(dtype: DType) -> Result<u32> {
        Ok(match dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        })
    }

    struct RotaryLaunch<'a> {
        query: &'a mut CudaStorage,
        query_l: &'a Layout,
        cos_cache: &'a CudaStorage,
        cos_l: &'a Layout,
        sin_cache: &'a CudaStorage,
        sin_l: &'a Layout,
        positions: Option<(&'a CudaStorage, &'a Layout)>,
        is_neox: bool,
    }

    fn launch_rotary<T>(args: RotaryLaunch<'_>) -> Result<()>
    where
        T: CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    {
        let RotaryLaunch {
            query,
            query_l,
            cos_cache,
            cos_l,
            sin_cache,
            sin_l,
            positions,
            is_neox,
        } = args;

        if cos_cache.dtype() != query.dtype() || sin_cache.dtype() != query.dtype() {
            candle_core::bail!("apply-rotary expects all tensors to have the same dtype");
        }

        let dev = query.device().clone();
        if !cos_cache.device().same_device(&dev) || !sin_cache.device().same_device(&dev) {
            candle_core::bail!("apply-rotary tensors must be on the same cuda device");
        }

        if query_l.stride().len() != 3 {
            candle_core::bail!("apply-rotary expects query rank 3 ({query_l:?})")
        }
        if cos_l.stride().len() != 2 || sin_l.stride().len() != 2 {
            candle_core::bail!("apply-rotary expects rank 2 caches")
        }

        let (num_tokens, num_heads, head_size) = query_l.shape().dims3()?;
        let rot_dim = cos_l.dims()[1];
        if sin_l.shape().dims2()? != (cos_l.dims()[0], rot_dim) {
            candle_core::bail!(
                "shape mismatch cos_cache {:?} and sin_cache {:?}",
                cos_l.shape(),
                sin_l.shape()
            )
        }
        if positions.is_none() && (num_tokens, rot_dim) != cos_l.shape().dims2()? {
            candle_core::bail!(
                "shape mismatch cos_cache {:?}, expected {:?}",
                cos_l.shape(),
                (num_tokens, rot_dim)
            )
        }
        if rot_dim == 0 || rot_dim * 2 > head_size {
            candle_core::bail!(
                "rotary dimension {rot_dim} is incompatible with head size {head_size}"
            )
        }

        let query_dtype = query.dtype();
        let stream = dev.cuda_stream();
        let query = query.as_cuda_slice_mut::<T>()?;
        let cos_cache = cos_cache.as_cuda_slice::<T>()?;
        let sin_cache = sin_cache.as_cuda_slice::<T>()?;
        let (query, _query_guard) = slice_ptr_mut_on_stream(query, query_l.start_offset(), &stream);
        let (cos_cache, _cos_guard) = slice_ptr_on_stream(cos_cache, cos_l.start_offset(), &stream);
        let (sin_cache, _sin_guard) = slice_ptr_on_stream(sin_cache, sin_l.start_offset(), &stream);

        let positions = if let Some((positions, positions_l)) = positions {
            if positions.dtype() != DType::U32 {
                candle_core::bail!("apply-rotary-positions expects positions to be u32");
            }
            if !positions.device().same_device(&dev) {
                candle_core::bail!("positions must be on the same cuda device as query");
            }
            if positions_l.stride().len() != 1 {
                candle_core::bail!("apply-rotary-positions expects rank 1 positions")
            }
            let positions_len = positions_l.shape().dims1()?;
            if positions_len == 0 || num_tokens % positions_len != 0 {
                candle_core::bail!(
                    "positions length {positions_len} is incompatible with token count {num_tokens}"
                );
            }
            let positions = match &positions.slice {
                CudaStorageSlice::U32(positions) => positions,
                _ => candle_core::bail!("positions dtype mismatch"),
            };
            let (positions, guard) =
                slice_ptr_on_stream(positions, positions_l.start_offset(), &stream);
            Some((positions, num_tokens / positions_len, guard))
        } else {
            None
        };

        let neox = if is_neox { 1 } else { 0 };
        let stream = stream.cu_stream() as c_long;
        let internal_type = rotary_dtype(query_dtype)?;
        match positions {
            None => unsafe {
                super::ffi::rotary_embedding(
                    query as *const core::ffi::c_void,
                    std::ptr::null(),
                    cos_cache as *const core::ffi::c_void,
                    sin_cache as *const core::ffi::c_void,
                    neox,
                    head_size as c_int,
                    num_tokens as c_long,
                    rot_dim as c_int,
                    num_heads as c_int,
                    0,
                    query_l.stride()[0] as c_long,
                    0,
                    internal_type,
                    stream,
                )
            },
            Some((positions, seq_len, _positions_guard)) => unsafe {
                super::ffi::rotary_embedding_positions(
                    query as *const core::ffi::c_void,
                    std::ptr::null(),
                    cos_cache as *const core::ffi::c_void,
                    sin_cache as *const core::ffi::c_void,
                    positions as *const core::ffi::c_void,
                    neox,
                    head_size as c_int,
                    num_tokens as c_long,
                    rot_dim as c_int,
                    seq_len as c_int,
                    num_heads as c_int,
                    0,
                    query_l.stride()[0] as c_long,
                    0,
                    internal_type,
                    stream,
                )
            },
        }
        Ok(())
    }

    struct RotaryInplace {
        is_neox: bool,
    }

    impl InplaceOp3 for RotaryInplace {
        fn name(&self) -> &'static str {
            "mistralrs-rotary-inplace"
        }

        fn cpu_fwd(
            &self,
            _: &mut CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<()> {
            candle_core::bail!("apply-rotary-inplace is only supported for cuda")
        }

        fn cuda_fwd(
            &self,
            query: &mut CudaStorage,
            query_l: &Layout,
            cos_cache: &CudaStorage,
            cos_l: &Layout,
            sin_cache: &CudaStorage,
            sin_l: &Layout,
        ) -> Result<()> {
            match query.dtype() {
                DType::F16 => launch_rotary::<f16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                DType::BF16 => launch_rotary::<bf16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                DType::F32 => launch_rotary::<f32>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                dt => {
                    candle_core::bail!(
                        "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                    )
                }
            }
        }

        fn metal_fwd(
            &self,
            _: &mut MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
        ) -> Result<()> {
            candle_core::bail!("apply-rotary-inplace is only supported for cuda")
        }
    }

    struct RotaryPositionsInplace<'a> {
        is_neox: bool,
        positions: &'a Tensor,
    }

    impl InplaceOp3 for RotaryPositionsInplace<'_> {
        fn name(&self) -> &'static str {
            "mistralrs-rotary-positions-inplace"
        }

        fn cpu_fwd(
            &self,
            _: &mut CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<()> {
            candle_core::bail!("apply-rotary-positions-inplace is only supported for cuda")
        }

        fn cuda_fwd(
            &self,
            query: &mut CudaStorage,
            query_l: &Layout,
            cos_cache: &CudaStorage,
            cos_l: &Layout,
            sin_cache: &CudaStorage,
            sin_l: &Layout,
        ) -> Result<()> {
            let (positions_storage, positions_l) = self.positions.storage_and_layout();
            let positions = match &*positions_storage {
                Storage::Cuda(positions) => positions,
                _ => candle_core::bail!("positions must be a cuda tensor"),
            };
            match query.dtype() {
                DType::F16 => launch_rotary::<f16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                DType::BF16 => launch_rotary::<bf16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                DType::F32 => launch_rotary::<f32>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                dt => {
                    candle_core::bail!(
                        "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                    )
                }
            }
        }

        fn metal_fwd(
            &self,
            _: &mut MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
        ) -> Result<()> {
            candle_core::bail!("apply-rotary-positions-inplace is only supported for cuda")
        }
    }

    fn apply_rotary_(
        query: &Tensor,
        key: Option<&Tensor>,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        let dtype = query.dtype();
        if key.is_some_and(|key| key.dtype() != dtype)
            || cos_cache.dtype() != dtype
            || sin_cache.dtype() != dtype
        {
            candle_core::bail!("apply-rotary expects all tensors to have the same dtype");
        }
        let op = RotaryInplace { is_neox };
        query.inplace_op3(cos_cache, sin_cache, &op)?;
        if let Some(key) = key {
            key.inplace_op3(cos_cache, sin_cache, &op)?;
        }
        Ok(())
    }

    fn apply_rotary_positions_(
        query: &Tensor,
        key: Option<&Tensor>,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        positions: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        let dtype = query.dtype();
        if key.is_some_and(|key| key.dtype() != dtype)
            || cos_cache.dtype() != dtype
            || sin_cache.dtype() != dtype
            || positions.dtype() != DType::U32
        {
            candle_core::bail!(
                "apply-rotary-positions expects q/k/caches to share dtype and positions to be u32"
            );
        }

        let cos_cache = cos_cache.contiguous()?;
        let sin_cache = sin_cache.contiguous()?;
        let positions = positions.contiguous()?;
        let op = RotaryPositionsInplace {
            is_neox,
            positions: &positions,
        };
        query.inplace_op3(&cos_cache, &sin_cache, &op)?;
        if let Some(key) = key {
            key.inplace_op3(&cos_cache, &sin_cache, &op)?;
        }
        Ok(())
    }

    /// Apply Rotary position encoding inplace
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor of shape `(num_tokens, num_heads, head_size)`.
    /// * `key` - Key tensor of shape `(num_tokens, num_kv_heads, head_size)`.
    /// * `cos_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
    /// * `sin_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
    /// * `is_neox` - Use neox encoding instead of gpt-j style rotary
    pub fn apply_rotary_inplace(
        query: &Tensor,
        key: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match key.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_(query, Some(key), cos_cache, sin_cache, is_neox)
            }
            dt => {
                candle_core::bail!("apply_rotary is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }

    pub fn apply_rotary_inplace_q(
        query: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match query.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_(query, None, cos_cache, sin_cache, is_neox)
            }
            dt => {
                candle_core::bail!("apply_rotary is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }

    pub fn apply_rotary_inplace_positions(
        query: &Tensor,
        key: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        positions: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match key.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_positions_(query, Some(key), cos_cache, sin_cache, positions, is_neox)
            }
            dt => {
                candle_core::bail!("apply_rotary is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }

    pub fn apply_rotary_inplace_q_positions(
        query: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        positions: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match query.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_positions_(query, None, cos_cache, sin_cache, positions, is_neox)
            }
            dt => {
                candle_core::bail!("apply_rotary is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::*;

/// Apply Rotary position encoding inplace
///
/// # Arguments
///
/// * `query` - Query tensor of shape `(num_tokens, num_heads, head_size)`.
/// * `key` - Key tensor of shape `(num_tokens, num_kv_heads, head_size)`.
/// * `cos_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
/// * `sin_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
/// * `is_neox` - Use neox encoding instead of gpt-j style rotary
#[cfg(not(feature = "cuda"))]
pub fn apply_rotary_inplace(
    _query: &candle_core::Tensor,
    _key: &candle_core::Tensor,
    _cos_cache: &candle_core::Tensor,
    _sin_cache: &candle_core::Tensor,
    _is_neox: bool,
) -> candle_core::Result<()> {
    candle_core::bail!("apply_rotary is only supported for cuda");
}

#[cfg(not(feature = "cuda"))]
pub fn apply_rotary_inplace_q(
    _query: &candle_core::Tensor,
    _cos_cache: &candle_core::Tensor,
    _sin_cache: &candle_core::Tensor,
    _is_neox: bool,
) -> candle_core::Result<()> {
    candle_core::bail!("apply_rotary is only supported for cuda");
}

#[cfg(not(feature = "cuda"))]
pub fn apply_rotary_inplace_positions(
    _query: &candle_core::Tensor,
    _key: &candle_core::Tensor,
    _cos_cache: &candle_core::Tensor,
    _sin_cache: &candle_core::Tensor,
    _positions: &candle_core::Tensor,
    _is_neox: bool,
) -> candle_core::Result<()> {
    candle_core::bail!("apply_rotary is only supported for cuda");
}

#[cfg(not(feature = "cuda"))]
pub fn apply_rotary_inplace_q_positions(
    _query: &candle_core::Tensor,
    _cos_cache: &candle_core::Tensor,
    _sin_cache: &candle_core::Tensor,
    _positions: &candle_core::Tensor,
    _is_neox: bool,
) -> candle_core::Result<()> {
    candle_core::bail!("apply_rotary is only supported for cuda");
}
