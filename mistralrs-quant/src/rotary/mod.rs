#[cfg(feature = "cuda")]
mod ffi;

use candle_core::{
    backend::BackendStorage, CpuStorage, CustomOp3, Layout, Result, Shape, Tensor, WithDType,
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

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::{
        backend::{BackendDevice, BackendStorage},
        DType, Result, Storage, Tensor,
    };
    use half::{bf16, f16};
    use std::ffi::{c_int, c_long};

    use crate::utils::slice_ptr;

    fn apply_rotary_<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
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

        let internal_type = match dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        };

        let (q, q_l) = query.storage_and_layout();
        let q = match &*q {
            Storage::Cuda(q) => q,
            _ => candle_core::bail!("query must be a cuda tensor"),
        };
        let dev = q.device().clone();

        let k_storage_and_layout = key.map(Tensor::storage_and_layout);

        let (cc, cc_l) = cos_cache.storage_and_layout();
        let cc = match &*cc {
            Storage::Cuda(cc) => cc,
            _ => candle_core::bail!("cos_cache must be a cuda tensor"),
        };

        let (sc, sc_l) = sin_cache.storage_and_layout();
        let sc = match &*sc {
            Storage::Cuda(sc) => sc,
            _ => candle_core::bail!("sin_cache must be a cuda tensor"),
        };

        let q_rank = q_l.stride().len();
        let cc_rank = cc_l.stride().len();
        let sc_rank = sc_l.stride().len();

        if q_rank != 3 {
            candle_core::bail!("apply-rotary expects query rank 3 ({q_l:?})")
        }

        if cc_rank != 2 || sc_rank != 2 {
            candle_core::bail!(
                "apply-rotary expects cache tensors of rank 2 (k: {cc_l:?}, v: {sc_l:?})"
            )
        }

        // Get cuda slices for all tensors
        let q = q.as_cuda_slice::<T>()?;
        let cc = cc.as_cuda_slice::<T>()?;
        let sc = sc.as_cuda_slice::<T>()?;

        // Get cuda views for all tensors
        let (q, _q_guard) = slice_ptr(q, q_l.start_offset());
        let (cc, _cc_guard) = slice_ptr(cc, cc_l.start_offset());
        let (sc, _sc_guard) = slice_ptr(sc, sc_l.start_offset());

        let (num_tokens, num_heads, head_size) = q_l.shape().dims3()?;
        let (num_kv_heads, key_stride, k, _k_guard) =
            if let Some((k_storage, k_l)) = k_storage_and_layout.as_ref() {
                let k_storage = match &**k_storage {
                    Storage::Cuda(k) => k,
                    _ => candle_core::bail!("key must be a cuda tensor"),
                };
                if !k_storage.device().same_device(&dev) {
                    candle_core::bail!("key must be on the same cuda device as query");
                }
                if k_l.stride().len() != 3 {
                    candle_core::bail!("apply-rotary expects key rank 3 ({k_l:?})")
                }
                let (num_tokens_kv, num_kv_heads, head_size_kv) = k_l.shape().dims3()?;
                if (num_tokens, head_size) != (num_tokens_kv, head_size_kv) {
                    candle_core::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
                }
                let k = k_storage.as_cuda_slice::<T>()?;
                let (k, k_guard) = slice_ptr(k, k_l.start_offset());
                (num_kv_heads, k_l.stride()[0], k, Some(k_guard))
            } else {
                (0, 0, 0, None)
            };

        let rot_dim = cc_l.dims()[1];
        if (num_tokens, rot_dim) != cc_l.shape().dims2()? {
            candle_core::bail!(
                "shape mismatch cos_cache {:?}, expected {:?}",
                cc_l.shape(),
                (num_tokens, rot_dim)
            )
        }

        if (num_tokens, rot_dim) != sc_l.shape().dims2()? {
            candle_core::bail!(
                "shape mismatch sin_cache {:?}, expected {:?}",
                sc_l.shape(),
                (num_tokens, rot_dim)
            )
        }
        if rot_dim == 0 || rot_dim * 2 > head_size {
            candle_core::bail!(
                "rotary dimension {rot_dim} is incompatible with head size {head_size}"
            )
        }

        let query_stride = q_l.stride()[0];

        let neox = if is_neox { 1 } else { 0 };
        let stream = dev.cuda_stream().cu_stream() as c_long;

        unsafe {
            super::ffi::rotary_embedding(
                q as *const core::ffi::c_void,
                k as *const core::ffi::c_void,
                cc as *const core::ffi::c_void,
                sc as *const core::ffi::c_void,
                neox,
                head_size as c_int,
                num_tokens as c_long,
                rot_dim as c_int,
                num_heads as c_int,
                num_kv_heads as c_int,
                query_stride as c_long,
                key_stride as c_long,
                internal_type,
                stream,
            )
        }
        Ok(())
    }

    fn apply_rotary_positions_<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
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

        let internal_type = match dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        };

        let cos_cache = cos_cache.contiguous()?;
        let sin_cache = sin_cache.contiguous()?;
        let positions = positions.contiguous()?;

        let (q, q_l) = query.storage_and_layout();
        let q = match &*q {
            Storage::Cuda(q) => q,
            _ => candle_core::bail!("query must be a cuda tensor"),
        };

        let k_storage_and_layout = key.map(Tensor::storage_and_layout);

        let (cc, cc_l) = cos_cache.storage_and_layout();
        let cc = match &*cc {
            Storage::Cuda(cc) => cc,
            _ => candle_core::bail!("cos_cache must be a cuda tensor"),
        };

        let (sc, sc_l) = sin_cache.storage_and_layout();
        let sc = match &*sc {
            Storage::Cuda(sc) => sc,
            _ => candle_core::bail!("sin_cache must be a cuda tensor"),
        };

        let (pos, pos_l) = positions.storage_and_layout();
        let pos = match &*pos {
            Storage::Cuda(pos) => pos,
            _ => candle_core::bail!("positions must be a cuda tensor"),
        };

        if !cc.device().same_device(q.device())
            || !sc.device().same_device(q.device())
            || !pos.device().same_device(q.device())
        {
            candle_core::bail!("apply-rotary-positions tensors must be on the same cuda device");
        }
        let dev = q.device().clone();

        let q_rank = q_l.stride().len();
        let cc_rank = cc_l.stride().len();
        let sc_rank = sc_l.stride().len();
        let pos_rank = pos_l.stride().len();

        if q_rank != 3 {
            candle_core::bail!("apply-rotary-positions expects query rank 3 ({q_l:?})")
        }

        if cc_rank != 2 || sc_rank != 2 || pos_rank != 1 {
            candle_core::bail!("apply-rotary-positions expects rank 2 caches and rank 1 positions")
        }

        let q = q.as_cuda_slice::<T>()?;
        let cc = cc.as_cuda_slice::<T>()?;
        let sc = sc.as_cuda_slice::<T>()?;
        let pos = match &pos.slice {
            candle_core::cuda_backend::CudaStorageSlice::U32(pos) => pos,
            _ => candle_core::bail!("positions dtype mismatch"),
        };

        let (q, _q_guard) = slice_ptr(q, q_l.start_offset());
        let (cc, _cc_guard) = slice_ptr(cc, cc_l.start_offset());
        let (sc, _sc_guard) = slice_ptr(sc, sc_l.start_offset());
        let (pos, _pos_guard) = slice_ptr(pos, pos_l.start_offset());

        let (num_tokens, num_heads, head_size) = q_l.shape().dims3()?;
        let positions_len = pos_l.shape().dims1()?;
        if positions_len == 0 || num_tokens % positions_len != 0 {
            candle_core::bail!(
                "positions length {positions_len} is incompatible with token count {num_tokens}"
            );
        }
        let seq_len = num_tokens / positions_len;

        let (num_kv_heads, key_stride, k, _k_guard) =
            if let Some((k_storage, k_l)) = k_storage_and_layout.as_ref() {
                let k_storage = match &**k_storage {
                    Storage::Cuda(k) => k,
                    _ => candle_core::bail!("key must be a cuda tensor"),
                };
                if !k_storage.device().same_device(&dev) {
                    candle_core::bail!("key must be on the same cuda device as query");
                }
                if k_l.stride().len() != 3 {
                    candle_core::bail!("apply-rotary-positions expects key rank 3 ({k_l:?})")
                }
                let (num_tokens_kv, num_kv_heads, head_size_kv) = k_l.shape().dims3()?;
                if (num_tokens, head_size) != (num_tokens_kv, head_size_kv) {
                    candle_core::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
                }
                let k = k_storage.as_cuda_slice::<T>()?;
                let (k, k_guard) = slice_ptr(k, k_l.start_offset());
                (num_kv_heads, k_l.stride()[0], k, Some(k_guard))
            } else {
                (0, 0, 0, None)
            };

        let rot_dim = cc_l.dims()[1];
        if sc_l.shape().dims2()? != (cc_l.dims()[0], rot_dim) {
            candle_core::bail!(
                "shape mismatch cos_cache {:?} and sin_cache {:?}",
                cc_l.shape(),
                sc_l.shape()
            )
        }
        if rot_dim == 0 || rot_dim * 2 > head_size {
            candle_core::bail!(
                "rotary dimension {rot_dim} is incompatible with head size {head_size}"
            )
        }

        let query_stride = q_l.stride()[0];
        let neox = if is_neox { 1 } else { 0 };
        let stream = dev.cuda_stream().cu_stream() as c_long;

        unsafe {
            super::ffi::rotary_embedding_positions(
                q as *const core::ffi::c_void,
                k as *const core::ffi::c_void,
                cc as *const core::ffi::c_void,
                sc as *const core::ffi::c_void,
                pos as *const core::ffi::c_void,
                neox,
                head_size as c_int,
                num_tokens as c_long,
                rot_dim as c_int,
                seq_len as c_int,
                num_heads as c_int,
                num_kv_heads as c_int,
                query_stride as c_long,
                key_stride as c_long,
                internal_type,
                stream,
            )
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
            DType::F16 => apply_rotary_::<f16>(query, Some(key), cos_cache, sin_cache, is_neox),
            DType::BF16 => apply_rotary_::<bf16>(query, Some(key), cos_cache, sin_cache, is_neox),
            DType::F32 => apply_rotary_::<f32>(query, Some(key), cos_cache, sin_cache, is_neox),
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
            DType::F16 => apply_rotary_::<f16>(query, None, cos_cache, sin_cache, is_neox),
            DType::BF16 => apply_rotary_::<bf16>(query, None, cos_cache, sin_cache, is_neox),
            DType::F32 => apply_rotary_::<f32>(query, None, cos_cache, sin_cache, is_neox),
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
            DType::F16 => apply_rotary_positions_::<f16>(
                query,
                Some(key),
                cos_cache,
                sin_cache,
                positions,
                is_neox,
            ),
            DType::BF16 => apply_rotary_positions_::<bf16>(
                query,
                Some(key),
                cos_cache,
                sin_cache,
                positions,
                is_neox,
            ),
            DType::F32 => apply_rotary_positions_::<f32>(
                query,
                Some(key),
                cos_cache,
                sin_cache,
                positions,
                is_neox,
            ),
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
            DType::F16 => apply_rotary_positions_::<f16>(
                query, None, cos_cache, sin_cache, positions, is_neox,
            ),
            DType::BF16 => apply_rotary_positions_::<bf16>(
                query, None, cos_cache, sin_cache, positions, is_neox,
            ),
            DType::F32 => apply_rotary_positions_::<f32>(
                query, None, cos_cache, sin_cache, positions, is_neox,
            ),
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
