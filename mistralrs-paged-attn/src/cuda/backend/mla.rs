use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi::{
    concat_and_cache_mla as ffi_concat_and_cache_mla,
    flashinfer_mla_decode as ffi_flashinfer_mla_decode, gather_mla_cache as ffi_gather_mla_cache,
};
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::CudaStorageSlice;
use candle_core::{DType, Result, Storage, Tensor};

pub fn concat_and_cache_mla(
    ckv: &Tensor,
    k_pe: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let dtype = ckv.dtype();
    if k_pe.dtype() != dtype {
        candle_core::bail!(
            "concat_and_cache_mla requires matching dtypes for ckv and k_pe, got {:?} and {:?}",
            dtype,
            k_pe.dtype()
        );
    }
    if ckv_cache.dtype() != dtype || kpe_cache.dtype() != dtype {
        candle_core::bail!(
            "concat_and_cache_mla requires matching cache dtype, got {:?} and {:?}",
            ckv_cache.dtype(),
            kpe_cache.dtype()
        );
    }

    let (ckv_s, ckv_l) = ckv.storage_and_layout();
    let (kpe_s, kpe_l) = k_pe.storage_and_layout();
    let (ckv_cache_s, ckv_cache_l) = ckv_cache.storage_and_layout();
    let (kpe_cache_s, kpe_cache_l) = kpe_cache.storage_and_layout();
    let (slot_s, slot_l) = slot_mapping.storage_and_layout();

    let ckv_s = match &*ckv_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("ckv must be a cuda tensor"),
    };
    let kpe_s = match &*kpe_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("k_pe must be a cuda tensor"),
    };
    let ckv_cache_s = match &*ckv_cache_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("ckv_cache must be a cuda tensor"),
    };
    let kpe_cache_s = match &*kpe_cache_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kpe_cache must be a cuda tensor"),
    };
    let slot_s = match &*slot_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("slot_mapping must be a cuda tensor"),
    };

    let (num_tokens, kv_lora_rank) = ckv_l.shape().dims2()?;
    let (num_tokens_kpe, kpe_head_dim) = kpe_l.shape().dims2()?;
    if num_tokens != num_tokens_kpe {
        candle_core::bail!("ckv and k_pe batch mismatch: {num_tokens} vs {num_tokens_kpe}");
    }

    let (num_blocks, block_size, cache_kv_lora_rank) = ckv_cache_l.shape().dims3()?;
    let (num_blocks_kpe, block_size_kpe, cache_kpe_head_dim) = kpe_cache_l.shape().dims3()?;
    if num_blocks != num_blocks_kpe || block_size != block_size_kpe {
        candle_core::bail!(
            "ckv_cache and kpe_cache block shape mismatch: {:?} vs {:?}",
            ckv_cache_l.shape(),
            kpe_cache_l.shape()
        );
    }
    if cache_kv_lora_rank != kv_lora_rank {
        candle_core::bail!(
            "ckv_cache last dim mismatch: expected {kv_lora_rank}, got {cache_kv_lora_rank}"
        );
    }
    if cache_kpe_head_dim != kpe_head_dim {
        candle_core::bail!(
            "kpe_cache last dim mismatch: expected {kpe_head_dim}, got {cache_kpe_head_dim}"
        );
    }

    let slot_len = slot_l.shape().dims1()?;
    if slot_len != num_tokens {
        candle_core::bail!("slot_mapping length mismatch: expected {num_tokens}, got {slot_len}");
    }

    let dtype_code = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        other => {
            candle_core::bail!("concat_and_cache_mla only supports f16, bf16, f32 (got {other:?})")
        }
    };

    let ckv_stride = ckv_l.stride()[0];
    let kpe_stride = kpe_l.stride()[0];

    let dev = ckv_s.device();

    let (
        ckv_ptr,
        _ckv_guard,
        kpe_ptr,
        _kpe_guard,
        ckv_cache_ptr,
        _ckv_cache_guard,
        kpe_cache_ptr,
        _kpe_cache_guard,
        slot_ptr,
        _slot_guard,
    ) = match (
        &ckv_s.slice,
        &kpe_s.slice,
        &ckv_cache_s.slice,
        &kpe_cache_s.slice,
        &slot_s.slice,
    ) {
        (
            CudaStorageSlice::F16(ckv),
            CudaStorageSlice::F16(kpe),
            CudaStorageSlice::F16(ckv_cache),
            CudaStorageSlice::F16(kpe_cache),
            CudaStorageSlice::I64(slot),
        ) => {
            let (ckv_ptr, ckv_guard) = slice_ptr(ckv, ckv_l.start_offset());
            let (kpe_ptr, kpe_guard) = slice_ptr(kpe, kpe_l.start_offset());
            let (ckv_cache_ptr, ckv_cache_guard) = slice_ptr(ckv_cache, ckv_cache_l.start_offset());
            let (kpe_cache_ptr, kpe_cache_guard) = slice_ptr(kpe_cache, kpe_cache_l.start_offset());
            let (slot_ptr, slot_guard) = slice_ptr(slot, slot_l.start_offset());
            (
                ckv_ptr,
                ckv_guard,
                kpe_ptr,
                kpe_guard,
                ckv_cache_ptr,
                ckv_cache_guard,
                kpe_cache_ptr,
                kpe_cache_guard,
                slot_ptr,
                slot_guard,
            )
        }
        (
            CudaStorageSlice::BF16(ckv),
            CudaStorageSlice::BF16(kpe),
            CudaStorageSlice::BF16(ckv_cache),
            CudaStorageSlice::BF16(kpe_cache),
            CudaStorageSlice::I64(slot),
        ) => {
            let (ckv_ptr, ckv_guard) = slice_ptr(ckv, ckv_l.start_offset());
            let (kpe_ptr, kpe_guard) = slice_ptr(kpe, kpe_l.start_offset());
            let (ckv_cache_ptr, ckv_cache_guard) = slice_ptr(ckv_cache, ckv_cache_l.start_offset());
            let (kpe_cache_ptr, kpe_cache_guard) = slice_ptr(kpe_cache, kpe_cache_l.start_offset());
            let (slot_ptr, slot_guard) = slice_ptr(slot, slot_l.start_offset());
            (
                ckv_ptr,
                ckv_guard,
                kpe_ptr,
                kpe_guard,
                ckv_cache_ptr,
                ckv_cache_guard,
                kpe_cache_ptr,
                kpe_cache_guard,
                slot_ptr,
                slot_guard,
            )
        }
        (
            CudaStorageSlice::F32(ckv),
            CudaStorageSlice::F32(kpe),
            CudaStorageSlice::F32(ckv_cache),
            CudaStorageSlice::F32(kpe_cache),
            CudaStorageSlice::I64(slot),
        ) => {
            let (ckv_ptr, ckv_guard) = slice_ptr(ckv, ckv_l.start_offset());
            let (kpe_ptr, kpe_guard) = slice_ptr(kpe, kpe_l.start_offset());
            let (ckv_cache_ptr, ckv_cache_guard) = slice_ptr(ckv_cache, ckv_cache_l.start_offset());
            let (kpe_cache_ptr, kpe_cache_guard) = slice_ptr(kpe_cache, kpe_cache_l.start_offset());
            let (slot_ptr, slot_guard) = slice_ptr(slot, slot_l.start_offset());
            (
                ckv_ptr,
                ckv_guard,
                kpe_ptr,
                kpe_guard,
                ckv_cache_ptr,
                ckv_cache_guard,
                kpe_cache_ptr,
                kpe_cache_guard,
                slot_ptr,
                slot_guard,
            )
        }
        _ => {
            candle_core::bail!(
                    "concat_and_cache_mla expects matching dtype for ckv/k_pe/caches and i64 slot_mapping"
                )
        }
    };

    unsafe {
        ffi_concat_and_cache_mla(
            ckv_ptr as *const core::ffi::c_void,
            kpe_ptr as *const core::ffi::c_void,
            ckv_cache_ptr as *const core::ffi::c_void,
            kpe_cache_ptr as *const core::ffi::c_void,
            slot_ptr as *const core::ffi::c_long,
            num_tokens as i32,
            kv_lora_rank as i32,
            kpe_head_dim as i32,
            block_size as i32,
            ckv_stride as i32,
            kpe_stride as i32,
            dev.cuda_stream().cu_stream(),
            dtype_code,
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn flashinfer_mla_decode(
    q_nope: &Tensor,
    q_pe: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    paged_kv_indptr: &Tensor,
    paged_kv_indices: &Tensor,
    paged_kv_last_page_len: &Tensor,
    request_indices: &Tensor,
    kv_tile_indices: &Tensor,
    o_indptr: &Tensor,
    kv_chunk_size: &Tensor,
    sm_scale: f32,
) -> Result<Tensor> {
    let dtype = q_nope.dtype();
    if q_pe.dtype() != dtype {
        candle_core::bail!(
            "flashinfer_mla_decode expects matching q_nope/q_pe dtype, got {:?} and {:?}",
            dtype,
            q_pe.dtype()
        );
    }
    if ckv_cache.dtype() != dtype || kpe_cache.dtype() != dtype {
        candle_core::bail!(
            "flashinfer_mla_decode expects matching cache dtype, got {:?} and {:?}",
            ckv_cache.dtype(),
            kpe_cache.dtype()
        );
    }

    let (batch_size, num_heads, head_dim_ckv) = q_nope.dims3()?;
    let (batch_size_pe, num_heads_pe, head_dim_kpe) = q_pe.dims3()?;
    if batch_size != batch_size_pe || num_heads != num_heads_pe {
        candle_core::bail!("flashinfer_mla_decode expects matching q_nope/q_pe batch/head dims");
    }
    if head_dim_ckv != 512 || head_dim_kpe != 64 {
        candle_core::bail!(
            "flashinfer_mla_decode is compiled for head dims 512/64, got {head_dim_ckv}/{head_dim_kpe}"
        );
    }

    let (num_blocks, block_size, cache_head_dim_ckv) = ckv_cache.dims3()?;
    let (num_blocks_kpe, block_size_kpe, cache_head_dim_kpe) = kpe_cache.dims3()?;
    if num_blocks != num_blocks_kpe || block_size != block_size_kpe {
        candle_core::bail!("ckv_cache and kpe_cache block shape mismatch");
    }
    if cache_head_dim_ckv != head_dim_ckv {
        candle_core::bail!(
            "ckv_cache head dim mismatch: expected {head_dim_ckv}, got {cache_head_dim_ckv}"
        );
    }
    if cache_head_dim_kpe != head_dim_kpe {
        candle_core::bail!(
            "kpe_cache head dim mismatch: expected {head_dim_kpe}, got {cache_head_dim_kpe}"
        );
    }
    if paged_kv_indptr.dims1()? != batch_size + 1 {
        candle_core::bail!(
            "paged_kv_indptr length mismatch: expected {}, got {}",
            batch_size + 1,
            paged_kv_indptr.dims1()?
        );
    }
    if paged_kv_last_page_len.dims1()? != batch_size {
        candle_core::bail!(
            "paged_kv_last_page_len length mismatch: expected {}, got {}",
            batch_size,
            paged_kv_last_page_len.dims1()?
        );
    }
    if request_indices.dims1()? != batch_size
        || kv_tile_indices.dims1()? != batch_size
        || o_indptr.dims1()? != batch_size + 1
        || kv_chunk_size.dims1()? != 1
    {
        candle_core::bail!("flashinfer_mla_decode metadata tensor shapes are invalid");
    }

    let dtype_code = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        other => {
            candle_core::bail!("flashinfer_mla_decode only supports f16, bf16, f32 (got {other:?})")
        }
    };

    let (q_nope_s, q_nope_l) = q_nope.storage_and_layout();
    let (q_pe_s, q_pe_l) = q_pe.storage_and_layout();
    let (ckv_cache_s, ckv_cache_l) = ckv_cache.storage_and_layout();
    let (kpe_cache_s, kpe_cache_l) = kpe_cache.storage_and_layout();
    let (kv_indptr_s, kv_indptr_l) = paged_kv_indptr.storage_and_layout();
    let (kv_indices_s, kv_indices_l) = paged_kv_indices.storage_and_layout();
    let (kv_last_s, kv_last_l) = paged_kv_last_page_len.storage_and_layout();
    let (request_indices_s, request_indices_l) = request_indices.storage_and_layout();
    let (kv_tile_indices_s, kv_tile_indices_l) = kv_tile_indices.storage_and_layout();
    let (o_indptr_s, o_indptr_l) = o_indptr.storage_and_layout();
    let (kv_chunk_s, kv_chunk_l) = kv_chunk_size.storage_and_layout();

    let q_nope_s = match &*q_nope_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("q_nope must be a cuda tensor"),
    };
    let q_pe_s = match &*q_pe_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("q_pe must be a cuda tensor"),
    };
    let ckv_cache_s = match &*ckv_cache_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("ckv_cache must be a cuda tensor"),
    };
    let kpe_cache_s = match &*kpe_cache_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kpe_cache must be a cuda tensor"),
    };
    let kv_indptr_s = match &*kv_indptr_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("paged_kv_indptr must be a cuda tensor"),
    };
    let kv_indices_s = match &*kv_indices_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("paged_kv_indices must be a cuda tensor"),
    };
    let kv_last_s = match &*kv_last_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("paged_kv_last_page_len must be a cuda tensor"),
    };
    let request_indices_s = match &*request_indices_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("request_indices must be a cuda tensor"),
    };
    let kv_tile_indices_s = match &*kv_tile_indices_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kv_tile_indices must be a cuda tensor"),
    };
    let o_indptr_s = match &*o_indptr_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("o_indptr must be a cuda tensor"),
    };
    let kv_chunk_s = match &*kv_chunk_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kv_chunk_size must be a cuda tensor"),
    };

    let output = Tensor::zeros(
        (batch_size, num_heads, head_dim_ckv),
        dtype,
        q_nope.device(),
    )?;

    {
        let (output_s, output_l) = output.storage_and_layout();
        let output_s = match &*output_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("output must be a cuda tensor"),
        };

        let dev = q_nope_s.device();

        let (
            q_nope_ptr,
            _q_nope_guard,
            q_pe_ptr,
            _q_pe_guard,
            ckv_cache_ptr,
            _ckv_cache_guard,
            kpe_cache_ptr,
            _kpe_cache_guard,
            kv_indptr_ptr,
            _kv_indptr_guard,
            kv_indices_ptr,
            _kv_indices_guard,
            kv_last_ptr,
            _kv_last_guard,
            request_indices_ptr,
            _request_indices_guard,
            kv_tile_indices_ptr,
            _kv_tile_indices_guard,
            o_indptr_ptr,
            _o_indptr_guard,
            kv_chunk_ptr,
            _kv_chunk_guard,
            output_ptr,
            _output_guard,
        ) = match (
            &q_nope_s.slice,
            &q_pe_s.slice,
            &ckv_cache_s.slice,
            &kpe_cache_s.slice,
            &kv_indptr_s.slice,
            &kv_indices_s.slice,
            &kv_last_s.slice,
            &request_indices_s.slice,
            &kv_tile_indices_s.slice,
            &o_indptr_s.slice,
            &kv_chunk_s.slice,
            &output_s.slice,
        ) {
            (
                CudaStorageSlice::F16(q_nope),
                CudaStorageSlice::F16(q_pe),
                CudaStorageSlice::F16(ckv_cache),
                CudaStorageSlice::F16(kpe_cache),
                CudaStorageSlice::I32(kv_indptr),
                CudaStorageSlice::I32(kv_indices),
                CudaStorageSlice::I32(kv_last),
                CudaStorageSlice::I32(request_indices),
                CudaStorageSlice::I32(kv_tile_indices),
                CudaStorageSlice::I32(o_indptr),
                CudaStorageSlice::I32(kv_chunk),
                CudaStorageSlice::F16(output),
            ) => {
                let (q_nope_ptr, q_nope_guard) = slice_ptr(q_nope, q_nope_l.start_offset());
                let (q_pe_ptr, q_pe_guard) = slice_ptr(q_pe, q_pe_l.start_offset());
                let (ckv_cache_ptr, ckv_cache_guard) =
                    slice_ptr(ckv_cache, ckv_cache_l.start_offset());
                let (kpe_cache_ptr, kpe_cache_guard) =
                    slice_ptr(kpe_cache, kpe_cache_l.start_offset());
                let (kv_indptr_ptr, kv_indptr_guard) =
                    slice_ptr(kv_indptr, kv_indptr_l.start_offset());
                let (kv_indices_ptr, kv_indices_guard) =
                    slice_ptr(kv_indices, kv_indices_l.start_offset());
                let (kv_last_ptr, kv_last_guard) = slice_ptr(kv_last, kv_last_l.start_offset());
                let (request_indices_ptr, request_indices_guard) =
                    slice_ptr(request_indices, request_indices_l.start_offset());
                let (kv_tile_indices_ptr, kv_tile_indices_guard) =
                    slice_ptr(kv_tile_indices, kv_tile_indices_l.start_offset());
                let (o_indptr_ptr, o_indptr_guard) = slice_ptr(o_indptr, o_indptr_l.start_offset());
                let (kv_chunk_ptr, kv_chunk_guard) = slice_ptr(kv_chunk, kv_chunk_l.start_offset());
                let (output_ptr, output_guard) = slice_ptr(output, output_l.start_offset());
                (
                    q_nope_ptr,
                    q_nope_guard,
                    q_pe_ptr,
                    q_pe_guard,
                    ckv_cache_ptr,
                    ckv_cache_guard,
                    kpe_cache_ptr,
                    kpe_cache_guard,
                    kv_indptr_ptr,
                    kv_indptr_guard,
                    kv_indices_ptr,
                    kv_indices_guard,
                    kv_last_ptr,
                    kv_last_guard,
                    request_indices_ptr,
                    request_indices_guard,
                    kv_tile_indices_ptr,
                    kv_tile_indices_guard,
                    o_indptr_ptr,
                    o_indptr_guard,
                    kv_chunk_ptr,
                    kv_chunk_guard,
                    output_ptr,
                    output_guard,
                )
            }
            (
                CudaStorageSlice::BF16(q_nope),
                CudaStorageSlice::BF16(q_pe),
                CudaStorageSlice::BF16(ckv_cache),
                CudaStorageSlice::BF16(kpe_cache),
                CudaStorageSlice::I32(kv_indptr),
                CudaStorageSlice::I32(kv_indices),
                CudaStorageSlice::I32(kv_last),
                CudaStorageSlice::I32(request_indices),
                CudaStorageSlice::I32(kv_tile_indices),
                CudaStorageSlice::I32(o_indptr),
                CudaStorageSlice::I32(kv_chunk),
                CudaStorageSlice::BF16(output),
            ) => {
                let (q_nope_ptr, q_nope_guard) = slice_ptr(q_nope, q_nope_l.start_offset());
                let (q_pe_ptr, q_pe_guard) = slice_ptr(q_pe, q_pe_l.start_offset());
                let (ckv_cache_ptr, ckv_cache_guard) =
                    slice_ptr(ckv_cache, ckv_cache_l.start_offset());
                let (kpe_cache_ptr, kpe_cache_guard) =
                    slice_ptr(kpe_cache, kpe_cache_l.start_offset());
                let (kv_indptr_ptr, kv_indptr_guard) =
                    slice_ptr(kv_indptr, kv_indptr_l.start_offset());
                let (kv_indices_ptr, kv_indices_guard) =
                    slice_ptr(kv_indices, kv_indices_l.start_offset());
                let (kv_last_ptr, kv_last_guard) = slice_ptr(kv_last, kv_last_l.start_offset());
                let (request_indices_ptr, request_indices_guard) =
                    slice_ptr(request_indices, request_indices_l.start_offset());
                let (kv_tile_indices_ptr, kv_tile_indices_guard) =
                    slice_ptr(kv_tile_indices, kv_tile_indices_l.start_offset());
                let (o_indptr_ptr, o_indptr_guard) = slice_ptr(o_indptr, o_indptr_l.start_offset());
                let (kv_chunk_ptr, kv_chunk_guard) = slice_ptr(kv_chunk, kv_chunk_l.start_offset());
                let (output_ptr, output_guard) = slice_ptr(output, output_l.start_offset());
                (
                    q_nope_ptr,
                    q_nope_guard,
                    q_pe_ptr,
                    q_pe_guard,
                    ckv_cache_ptr,
                    ckv_cache_guard,
                    kpe_cache_ptr,
                    kpe_cache_guard,
                    kv_indptr_ptr,
                    kv_indptr_guard,
                    kv_indices_ptr,
                    kv_indices_guard,
                    kv_last_ptr,
                    kv_last_guard,
                    request_indices_ptr,
                    request_indices_guard,
                    kv_tile_indices_ptr,
                    kv_tile_indices_guard,
                    o_indptr_ptr,
                    o_indptr_guard,
                    kv_chunk_ptr,
                    kv_chunk_guard,
                    output_ptr,
                    output_guard,
                )
            }
            (
                CudaStorageSlice::F32(q_nope),
                CudaStorageSlice::F32(q_pe),
                CudaStorageSlice::F32(ckv_cache),
                CudaStorageSlice::F32(kpe_cache),
                CudaStorageSlice::I32(kv_indptr),
                CudaStorageSlice::I32(kv_indices),
                CudaStorageSlice::I32(kv_last),
                CudaStorageSlice::I32(request_indices),
                CudaStorageSlice::I32(kv_tile_indices),
                CudaStorageSlice::I32(o_indptr),
                CudaStorageSlice::I32(kv_chunk),
                CudaStorageSlice::F32(output),
            ) => {
                let (q_nope_ptr, q_nope_guard) = slice_ptr(q_nope, q_nope_l.start_offset());
                let (q_pe_ptr, q_pe_guard) = slice_ptr(q_pe, q_pe_l.start_offset());
                let (ckv_cache_ptr, ckv_cache_guard) =
                    slice_ptr(ckv_cache, ckv_cache_l.start_offset());
                let (kpe_cache_ptr, kpe_cache_guard) =
                    slice_ptr(kpe_cache, kpe_cache_l.start_offset());
                let (kv_indptr_ptr, kv_indptr_guard) =
                    slice_ptr(kv_indptr, kv_indptr_l.start_offset());
                let (kv_indices_ptr, kv_indices_guard) =
                    slice_ptr(kv_indices, kv_indices_l.start_offset());
                let (kv_last_ptr, kv_last_guard) = slice_ptr(kv_last, kv_last_l.start_offset());
                let (request_indices_ptr, request_indices_guard) =
                    slice_ptr(request_indices, request_indices_l.start_offset());
                let (kv_tile_indices_ptr, kv_tile_indices_guard) =
                    slice_ptr(kv_tile_indices, kv_tile_indices_l.start_offset());
                let (o_indptr_ptr, o_indptr_guard) = slice_ptr(o_indptr, o_indptr_l.start_offset());
                let (kv_chunk_ptr, kv_chunk_guard) = slice_ptr(kv_chunk, kv_chunk_l.start_offset());
                let (output_ptr, output_guard) = slice_ptr(output, output_l.start_offset());
                (
                    q_nope_ptr,
                    q_nope_guard,
                    q_pe_ptr,
                    q_pe_guard,
                    ckv_cache_ptr,
                    ckv_cache_guard,
                    kpe_cache_ptr,
                    kpe_cache_guard,
                    kv_indptr_ptr,
                    kv_indptr_guard,
                    kv_indices_ptr,
                    kv_indices_guard,
                    kv_last_ptr,
                    kv_last_guard,
                    request_indices_ptr,
                    request_indices_guard,
                    kv_tile_indices_ptr,
                    kv_tile_indices_guard,
                    o_indptr_ptr,
                    o_indptr_guard,
                    kv_chunk_ptr,
                    kv_chunk_guard,
                    output_ptr,
                    output_guard,
                )
            }
            _ => {
                candle_core::bail!(
                    "flashinfer_mla_decode expects q/cache dtype to match and indices to be i32"
                )
            }
        };

        unsafe {
            ffi_flashinfer_mla_decode(
                q_nope_ptr as *const core::ffi::c_void,
                q_pe_ptr as *const core::ffi::c_void,
                ckv_cache_ptr as *const core::ffi::c_void,
                kpe_cache_ptr as *const core::ffi::c_void,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                kv_last_ptr as *const i32,
                output_ptr as *const core::ffi::c_void,
                batch_size as i32,
                num_heads as i32,
                block_size as i32,
                sm_scale,
                -1,
                0.0,
                1.0,
                1.0,
                request_indices_ptr as *const i32,
                kv_tile_indices_ptr as *const i32,
                o_indptr_ptr as *const i32,
                kv_chunk_ptr as *const i32,
                dtype_code,
                dev.cuda_stream().cu_stream(),
            );
        }
    }

    Ok(output)
}

pub fn gather_mla_cache(
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    token_to_seq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dtype = ckv_cache.dtype();
    if kpe_cache.dtype() != dtype {
        candle_core::bail!(
            "gather_mla_cache expects matching cache dtypes, got {:?} and {:?}",
            dtype,
            kpe_cache.dtype()
        );
    }

    let (num_blocks, block_size, kv_lora_rank) = ckv_cache.dims3()?;
    let (num_blocks_kpe, block_size_kpe, kpe_head_dim) = kpe_cache.dims3()?;
    if num_blocks != num_blocks_kpe || block_size != block_size_kpe {
        candle_core::bail!("ckv_cache and kpe_cache block shape mismatch");
    }

    let block_table = block_table.contiguous()?;
    let cu_seq_lens = cu_seq_lens.contiguous()?;
    let token_to_seq = token_to_seq.contiguous()?;

    if block_table.dtype() != DType::I32
        || cu_seq_lens.dtype() != DType::I32
        || token_to_seq.dtype() != DType::I32
    {
        candle_core::bail!("gather_mla_cache expects i32 metadata tensors");
    }

    let num_tokens = token_to_seq.dims1()?;
    let ckv_out = Tensor::zeros((num_tokens, kv_lora_rank), dtype, ckv_cache.device())?;
    let kpe_out = Tensor::zeros((num_tokens, kpe_head_dim), dtype, kpe_cache.device())?;

    {
        let (ckv_cache_s, ckv_cache_l) = ckv_cache.storage_and_layout();
        let (kpe_cache_s, kpe_cache_l) = kpe_cache.storage_and_layout();
        let (block_table_s, block_table_l) = block_table.storage_and_layout();
        let (cu_seq_lens_s, cu_seq_lens_l) = cu_seq_lens.storage_and_layout();
        let (token_to_seq_s, token_to_seq_l) = token_to_seq.storage_and_layout();
        let (ckv_out_s, ckv_out_l) = ckv_out.storage_and_layout();
        let (kpe_out_s, kpe_out_l) = kpe_out.storage_and_layout();

        let ckv_cache_s = match &*ckv_cache_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("ckv_cache must be a cuda tensor"),
        };
        let kpe_cache_s = match &*kpe_cache_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("kpe_cache must be a cuda tensor"),
        };
        let block_table_s = match &*block_table_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("block_table must be a cuda tensor"),
        };
        let cu_seq_lens_s = match &*cu_seq_lens_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("cu_seq_lens must be a cuda tensor"),
        };
        let token_to_seq_s = match &*token_to_seq_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("token_to_seq must be a cuda tensor"),
        };
        let ckv_out_s = match &*ckv_out_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("ckv_out must be a cuda tensor"),
        };
        let kpe_out_s = match &*kpe_out_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("kpe_out must be a cuda tensor"),
        };

        let dtype_code = match dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            other => {
                candle_core::bail!("gather_mla_cache only supports f16, bf16, f32 (got {other:?})")
            }
        };

        let dev = ckv_cache_s.device();

        let (
            ckv_cache_ptr,
            _ckv_cache_guard,
            kpe_cache_ptr,
            _kpe_cache_guard,
            block_table_ptr,
            _block_table_guard,
            cu_seq_lens_ptr,
            _cu_seq_lens_guard,
            token_to_seq_ptr,
            _token_to_seq_guard,
            ckv_out_ptr,
            _ckv_out_guard,
            kpe_out_ptr,
            _kpe_out_guard,
        ) = match (
            &ckv_cache_s.slice,
            &kpe_cache_s.slice,
            &block_table_s.slice,
            &cu_seq_lens_s.slice,
            &token_to_seq_s.slice,
            &ckv_out_s.slice,
            &kpe_out_s.slice,
        ) {
            (
                CudaStorageSlice::F16(ckv_cache),
                CudaStorageSlice::F16(kpe_cache),
                CudaStorageSlice::I32(block_table),
                CudaStorageSlice::I32(cu_seq_lens),
                CudaStorageSlice::I32(token_to_seq),
                CudaStorageSlice::F16(ckv_out),
                CudaStorageSlice::F16(kpe_out),
            ) => {
                let (ckv_cache_ptr, ckv_cache_guard) =
                    slice_ptr(ckv_cache, ckv_cache_l.start_offset());
                let (kpe_cache_ptr, kpe_cache_guard) =
                    slice_ptr(kpe_cache, kpe_cache_l.start_offset());
                let (block_table_ptr, block_table_guard) =
                    slice_ptr(block_table, block_table_l.start_offset());
                let (cu_seq_lens_ptr, cu_seq_lens_guard) =
                    slice_ptr(cu_seq_lens, cu_seq_lens_l.start_offset());
                let (token_to_seq_ptr, token_to_seq_guard) =
                    slice_ptr(token_to_seq, token_to_seq_l.start_offset());
                let (ckv_out_ptr, ckv_out_guard) = slice_ptr(ckv_out, ckv_out_l.start_offset());
                let (kpe_out_ptr, kpe_out_guard) = slice_ptr(kpe_out, kpe_out_l.start_offset());
                (
                    ckv_cache_ptr,
                    ckv_cache_guard,
                    kpe_cache_ptr,
                    kpe_cache_guard,
                    block_table_ptr,
                    block_table_guard,
                    cu_seq_lens_ptr,
                    cu_seq_lens_guard,
                    token_to_seq_ptr,
                    token_to_seq_guard,
                    ckv_out_ptr,
                    ckv_out_guard,
                    kpe_out_ptr,
                    kpe_out_guard,
                )
            }
            (
                CudaStorageSlice::BF16(ckv_cache),
                CudaStorageSlice::BF16(kpe_cache),
                CudaStorageSlice::I32(block_table),
                CudaStorageSlice::I32(cu_seq_lens),
                CudaStorageSlice::I32(token_to_seq),
                CudaStorageSlice::BF16(ckv_out),
                CudaStorageSlice::BF16(kpe_out),
            ) => {
                let (ckv_cache_ptr, ckv_cache_guard) =
                    slice_ptr(ckv_cache, ckv_cache_l.start_offset());
                let (kpe_cache_ptr, kpe_cache_guard) =
                    slice_ptr(kpe_cache, kpe_cache_l.start_offset());
                let (block_table_ptr, block_table_guard) =
                    slice_ptr(block_table, block_table_l.start_offset());
                let (cu_seq_lens_ptr, cu_seq_lens_guard) =
                    slice_ptr(cu_seq_lens, cu_seq_lens_l.start_offset());
                let (token_to_seq_ptr, token_to_seq_guard) =
                    slice_ptr(token_to_seq, token_to_seq_l.start_offset());
                let (ckv_out_ptr, ckv_out_guard) = slice_ptr(ckv_out, ckv_out_l.start_offset());
                let (kpe_out_ptr, kpe_out_guard) = slice_ptr(kpe_out, kpe_out_l.start_offset());
                (
                    ckv_cache_ptr,
                    ckv_cache_guard,
                    kpe_cache_ptr,
                    kpe_cache_guard,
                    block_table_ptr,
                    block_table_guard,
                    cu_seq_lens_ptr,
                    cu_seq_lens_guard,
                    token_to_seq_ptr,
                    token_to_seq_guard,
                    ckv_out_ptr,
                    ckv_out_guard,
                    kpe_out_ptr,
                    kpe_out_guard,
                )
            }
            (
                CudaStorageSlice::F32(ckv_cache),
                CudaStorageSlice::F32(kpe_cache),
                CudaStorageSlice::I32(block_table),
                CudaStorageSlice::I32(cu_seq_lens),
                CudaStorageSlice::I32(token_to_seq),
                CudaStorageSlice::F32(ckv_out),
                CudaStorageSlice::F32(kpe_out),
            ) => {
                let (ckv_cache_ptr, ckv_cache_guard) =
                    slice_ptr(ckv_cache, ckv_cache_l.start_offset());
                let (kpe_cache_ptr, kpe_cache_guard) =
                    slice_ptr(kpe_cache, kpe_cache_l.start_offset());
                let (block_table_ptr, block_table_guard) =
                    slice_ptr(block_table, block_table_l.start_offset());
                let (cu_seq_lens_ptr, cu_seq_lens_guard) =
                    slice_ptr(cu_seq_lens, cu_seq_lens_l.start_offset());
                let (token_to_seq_ptr, token_to_seq_guard) =
                    slice_ptr(token_to_seq, token_to_seq_l.start_offset());
                let (ckv_out_ptr, ckv_out_guard) = slice_ptr(ckv_out, ckv_out_l.start_offset());
                let (kpe_out_ptr, kpe_out_guard) = slice_ptr(kpe_out, kpe_out_l.start_offset());
                (
                    ckv_cache_ptr,
                    ckv_cache_guard,
                    kpe_cache_ptr,
                    kpe_cache_guard,
                    block_table_ptr,
                    block_table_guard,
                    cu_seq_lens_ptr,
                    cu_seq_lens_guard,
                    token_to_seq_ptr,
                    token_to_seq_guard,
                    ckv_out_ptr,
                    ckv_out_guard,
                    kpe_out_ptr,
                    kpe_out_guard,
                )
            }
            _ => {
                candle_core::bail!(
                    "gather_mla_cache expects cache dtypes to match and metadata tensors to be i32"
                )
            }
        };

        let (_, block_table_stride) = block_table_l.shape().dims2()?;

        unsafe {
            ffi_gather_mla_cache(
                ckv_cache_ptr as *const core::ffi::c_void,
                kpe_cache_ptr as *const core::ffi::c_void,
                ckv_out_ptr as *const core::ffi::c_void,
                kpe_out_ptr as *const core::ffi::c_void,
                block_table_ptr as *const i32,
                cu_seq_lens_ptr as *const i32,
                token_to_seq_ptr as *const i32,
                num_tokens as i32,
                block_size as i32,
                block_table_stride as i32,
                kv_lora_rank as i32,
                kpe_head_dim as i32,
                dev.cuda_stream().cu_stream(),
                dtype_code,
            );
        }
    }

    Ok((ckv_out, kpe_out))
}
