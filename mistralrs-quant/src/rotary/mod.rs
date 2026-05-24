#[cfg(feature = "cuda")]
mod ffi;

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
        key: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        let dtype = query.dtype();
        if key.dtype() != dtype || cos_cache.dtype() != dtype || sin_cache.dtype() != dtype {
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

        let (k, k_l) = key.storage_and_layout();
        let k = match &*k {
            Storage::Cuda(k) => k,
            _ => candle_core::bail!("key must be a cuda tensor"),
        };

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
        let k_rank = k_l.stride().len();
        let cc_rank = cc_l.stride().len();
        let sc_rank = sc_l.stride().len();

        if q_rank != 3 || k_rank != 3 {
            candle_core::bail!(
                "apply-rotary expects input tensors of rank 3 (k: {q_l:?}, v: {k_l:?})"
            )
        }

        if cc_rank != 2 || sc_rank != 2 {
            candle_core::bail!(
                "apply-rotary expects cache tensors of rank 2 (k: {cc_l:?}, v: {sc_l:?})"
            )
        }

        // Get cuda slices for all tensors
        let q = q.as_cuda_slice::<T>()?;
        let k = k.as_cuda_slice::<T>()?;
        let cc = cc.as_cuda_slice::<T>()?;
        let sc = sc.as_cuda_slice::<T>()?;

        // Get cuda views for all tensors
        let (q, _q_guard) = slice_ptr(q, q_l.start_offset());
        let (k, _k_guard) = slice_ptr(k, k_l.start_offset());
        let (cc, _cc_guard) = slice_ptr(cc, cc_l.start_offset());
        let (sc, _sc_guard) = slice_ptr(sc, sc_l.start_offset());

        let (num_tokens, num_heads, head_size) = q_l.shape().dims3()?;
        let (num_tokens_kv, num_kv_heads, head_size_kv) = k_l.shape().dims3()?;

        if (num_tokens, head_size) != (num_tokens_kv, head_size_kv) {
            candle_core::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
        }

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

        let query_stride = q_l.stride()[0];
        let key_stride = k_l.stride()[0];

        let neox = if is_neox { 1 } else { 0 };

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
            DType::F16 => apply_rotary_::<f16>(query, key, cos_cache, sin_cache, is_neox),
            DType::BF16 => apply_rotary_::<bf16>(query, key, cos_cache, sin_cache, is_neox),
            DType::F32 => apply_rotary_::<f32>(query, key, cos_cache, sin_cache, is_neox),
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
