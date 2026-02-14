use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi::gather_kv_cache as ffi_gather_kv_cache;
use candle_core::backend::BackendStorage;
use candle_core::{DType, IndexOp, Result, Storage, Tensor};
use float8::F8E4M3;

pub fn gather_kv_cache(
    key_cache: &Tensor,   // [num_blocks, kv_heads, head_size/x, block_size, x]
    value_cache: &Tensor, // [num_blocks, kv_heads, head_size, block_size]
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    block_table: &Tensor, // [batch, max_blocks]
    cu_seq_lens: &Tensor, // [batch + 1]
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let cache_dtype = key_cache.dtype();
    if value_cache.dtype() != cache_dtype {
        candle_core::bail!(
            "gather_kv_cache expects matching cache dtypes, got {:?} and {:?}",
            cache_dtype,
            value_cache.dtype()
        );
    }

    let block_table = block_table.contiguous()?;
    let cu_seq_lens = cu_seq_lens.contiguous()?;

    if !matches!(block_table.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "gather_kv_cache expects i32/u32 block_table (got {:?})",
            block_table.dtype()
        );
    }
    if !matches!(cu_seq_lens.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "gather_kv_cache expects i32/u32 cu_seq_lens (got {:?})",
            cu_seq_lens.dtype()
        );
    }

    // Extract dimensions from cache shapes
    let k_dims = key_cache.dims5()?;
    let num_kv_heads = k_dims.1;
    let head_size_over_x = k_dims.2;
    let block_size = k_dims.3;
    let x = k_dims.4;
    let head_size = head_size_over_x * x;

    // num_tokens = cu_seq_lens[-1], num_seqs = len(cu_seq_lens) - 1
    let cu_seq_lens_len = cu_seq_lens.dims1()?;
    let num_seqs = cu_seq_lens_len - 1;
    let num_tokens = if cu_seq_lens.dtype() == DType::I32 {
        cu_seq_lens.i(cu_seq_lens_len - 1)?.to_scalar::<i32>()? as usize
    } else {
        cu_seq_lens.i(cu_seq_lens_len - 1)?.to_scalar::<u32>()? as usize
    };

    if num_tokens == 0 {
        let k_out = Tensor::zeros((0, num_kv_heads, head_size), out_dtype, key_cache.device())?;
        let v_out = Tensor::zeros((0, num_kv_heads, head_size), out_dtype, key_cache.device())?;
        return Ok((k_out, v_out));
    }

    let k_out = Tensor::zeros(
        (num_tokens, num_kv_heads, head_size),
        out_dtype,
        key_cache.device(),
    )?;
    let v_out = Tensor::zeros(
        (num_tokens, num_kv_heads, head_size),
        out_dtype,
        key_cache.device(),
    )?;

    let out_dtype_code: u32 = match out_dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        other => candle_core::bail!(
            "gather_kv_cache only supports f16, bf16, f32 output (got {other:?})"
        ),
    };
    let cache_dtype_code: u32 = match cache_dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        DType::F8E4M3 => 3,
        other => candle_core::bail!(
            "gather_kv_cache only supports f16, bf16, f32, f8e4m3 cache (got {other:?})"
        ),
    };

    // Scope all storage borrows so k_out/v_out can be moved in the return.
    {
        let (kc_s, kc_l) = key_cache.storage_and_layout();
        let kc_s = match &*kc_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("key_cache must be a cuda tensor"),
        };
        let (vc_s, vc_l) = value_cache.storage_and_layout();
        let vc_s = match &*vc_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("value_cache must be a cuda tensor"),
        };

        // Get cache pointers - handle FP8 vs regular dtype
        let (kc_ptr, _kc_guard) = if cache_dtype_code == 3 {
            slice_ptr(kc_s.as_cuda_slice::<F8E4M3>()?, kc_l.start_offset())
        } else {
            match cache_dtype {
                DType::F16 => slice_ptr(kc_s.as_cuda_slice::<half::f16>()?, kc_l.start_offset()),
                DType::BF16 => slice_ptr(kc_s.as_cuda_slice::<half::bf16>()?, kc_l.start_offset()),
                DType::F32 => slice_ptr(kc_s.as_cuda_slice::<f32>()?, kc_l.start_offset()),
                _ => unreachable!(),
            }
        };
        let (vc_ptr, _vc_guard) = if cache_dtype_code == 3 {
            slice_ptr(vc_s.as_cuda_slice::<F8E4M3>()?, vc_l.start_offset())
        } else {
            match cache_dtype {
                DType::F16 => slice_ptr(vc_s.as_cuda_slice::<half::f16>()?, vc_l.start_offset()),
                DType::BF16 => slice_ptr(vc_s.as_cuda_slice::<half::bf16>()?, vc_l.start_offset()),
                DType::F32 => slice_ptr(vc_s.as_cuda_slice::<f32>()?, vc_l.start_offset()),
                _ => unreachable!(),
            }
        };

        // Get output pointers
        let (ko_s, ko_l) = k_out.storage_and_layout();
        let ko_s = match &*ko_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("k_out must be a cuda tensor"),
        };
        let (vo_s, vo_l) = v_out.storage_and_layout();
        let vo_s = match &*vo_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("v_out must be a cuda tensor"),
        };
        let (ko_ptr, _ko_guard) = match out_dtype {
            DType::F16 => slice_ptr(ko_s.as_cuda_slice::<half::f16>()?, ko_l.start_offset()),
            DType::BF16 => slice_ptr(ko_s.as_cuda_slice::<half::bf16>()?, ko_l.start_offset()),
            DType::F32 => slice_ptr(ko_s.as_cuda_slice::<f32>()?, ko_l.start_offset()),
            _ => unreachable!(),
        };
        let (vo_ptr, _vo_guard) = match out_dtype {
            DType::F16 => slice_ptr(vo_s.as_cuda_slice::<half::f16>()?, vo_l.start_offset()),
            DType::BF16 => slice_ptr(vo_s.as_cuda_slice::<half::bf16>()?, vo_l.start_offset()),
            DType::F32 => slice_ptr(vo_s.as_cuda_slice::<f32>()?, vo_l.start_offset()),
            _ => unreachable!(),
        };

        // Block table and cu_seq_lens
        let (bt_s, bt_l) = block_table.storage_and_layout();
        let bt_s = match &*bt_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("block_table must be a cuda tensor"),
        };
        let (bt_ptr, _bt_guard) = slice_ptr(bt_s.as_cuda_slice::<u32>()?, bt_l.start_offset());

        let (cu_s, cu_l) = cu_seq_lens.storage_and_layout();
        let cu_s = match &*cu_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("cu_seq_lens must be a cuda tensor"),
        };
        let (cu_ptr, _cu_guard) = if cu_seq_lens.dtype() == DType::I32 {
            slice_ptr(cu_s.as_cuda_slice::<i32>()?, cu_l.start_offset())
        } else {
            slice_ptr(cu_s.as_cuda_slice::<u32>()?, cu_l.start_offset())
        };

        // Scale pointers — hoist storage guards so they outlive the pointers
        let _ks_storage = k_scale.map(|ks| ks.storage_and_layout());
        let (k_scale_ptr, _ks_guard) = if let Some((ref s, l)) = _ks_storage {
            let s = match &**s {
                Storage::Cuda(s) => s,
                _ => candle_core::bail!("k_scale must be a cuda tensor"),
            };
            let (ptr, guard) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
            (ptr as *const f32, Some(guard))
        } else {
            (std::ptr::null(), None)
        };
        let _vs_storage = v_scale.map(|vs| vs.storage_and_layout());
        let (v_scale_ptr, _vs_guard) = if let Some((ref s, l)) = _vs_storage {
            let s = match &**s {
                Storage::Cuda(s) => s,
                _ => candle_core::bail!("v_scale must be a cuda tensor"),
            };
            let (ptr, guard) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
            (ptr as *const f32, Some(guard))
        } else {
            (std::ptr::null(), None)
        };

        let (_, block_table_stride) = bt_l.shape().dims2()?;

        let dev = kc_s.device();

        unsafe {
            ffi_gather_kv_cache(
                kc_ptr as *const core::ffi::c_void,
                vc_ptr as *const core::ffi::c_void,
                ko_ptr as *const core::ffi::c_void,
                vo_ptr as *const core::ffi::c_void,
                k_scale_ptr,
                v_scale_ptr,
                bt_ptr as *const i32,
                cu_ptr as *const i32,
                num_tokens as i32,
                num_seqs as i32,
                block_size as i32,
                block_table_stride as i32,
                num_kv_heads as i32,
                head_size as i32,
                x as i32,
                dev.cuda_stream().cu_stream(),
                out_dtype_code,
                cache_dtype_code,
            );
        }
    }

    Ok((k_out, v_out))
}
