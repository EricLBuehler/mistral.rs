use candle_core::{DType, Result, Tensor};

pub const USE_FA3_FP8_PAGED: bool = cfg!(has_fa3_fp8_paged);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fa3DecodeSchedule {
    pub batch_size: usize,
    pub total_q: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
    pub max_seqlen_k: usize,
    pub num_splits: usize,
    pub num_sm: usize,
    pub device_id: usize,
}

#[derive(Clone, Copy)]
pub struct Fa3DecodeMetadata<'a> {
    pub paged_kv_indptr: &'a Tensor,
    pub paged_kv_indices: &'a Tensor,
    pub paged_kv_last_page_len: &'a Tensor,
    pub page_table: &'a Tensor,
    pub seqused_k: &'a Tensor,
    pub cu_seqlens_q: &'a Tensor,
    pub scheduler_metadata: &'a Tensor,
}

#[derive(Clone, Copy)]
pub struct Fa3DecodeParams<'a> {
    pub query: &'a Tensor,
    pub quantized_query: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub page_table: &'a Tensor,
    pub seqused_k: &'a Tensor,
    pub cu_seqlens_q: &'a Tensor,
    pub scheduler_metadata: &'a Tensor,
    pub output_accum: &'a Tensor,
    pub lse_accum: &'a Tensor,
    pub output_lse: &'a Tensor,
    pub q_descale: &'a Tensor,
    pub k_descale: &'a Tensor,
    pub v_descale: &'a Tensor,
    pub schedule: Fa3DecodeSchedule,
    pub softmax_scale: f32,
}

#[cfg(has_fa3_fp8_paged)]
fn as_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| candle_core::Error::msg(format!("FA3 {name} does not fit in i32")))
}

#[cfg(has_fa3_fp8_paged)]
pub fn fa3_prepare_decode_metadata(
    metadata: Fa3DecodeMetadata<'_>,
    schedule: Fa3DecodeSchedule,
) -> Result<()> {
    use crate::cuda::backend::slice_ptr_on_stream;
    use crate::cuda::ffi::{
        fa3_fp8_decode_materialize_paged_metadata, fa3_fp8_decode_prepare,
        Fa3Fp8DecodeScheduleParams,
    };
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::CudaStorageSlice;
    use candle_core::Storage;

    if schedule.batch_size == 0
        || schedule.total_q == 0
        || schedule.total_q > schedule.batch_size
        || schedule.q_heads == 0
        || schedule.kv_heads == 0
        || !schedule.q_heads.is_multiple_of(schedule.kv_heads)
        || schedule.head_dim != 256
        || schedule.page_size == 0
        || schedule.max_seqlen_k == 0
        || schedule.num_splits <= 1
        || schedule.num_splits > 256
        || schedule.num_sm == 0
    {
        candle_core::bail!("invalid FA3 decode schedule: {schedule:?}");
    }

    let Fa3DecodeMetadata {
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        page_table,
        seqused_k,
        cu_seqlens_q,
        scheduler_metadata,
    } = metadata;
    for (name, tensor) in [
        ("paged_kv_indptr", paged_kv_indptr),
        ("paged_kv_indices", paged_kv_indices),
        ("paged_kv_last_page_len", paged_kv_last_page_len),
        ("page_table", page_table),
        ("seqused_k", seqused_k),
        ("cu_seqlens_q", cu_seqlens_q),
        ("scheduler_metadata", scheduler_metadata),
    ] {
        if tensor.dtype() != DType::I32 || !tensor.is_contiguous() {
            candle_core::bail!("FA3 expects contiguous i32 {name}");
        }
        if tensor.device().location() != paged_kv_indptr.device().location() {
            candle_core::bail!("FA3 metadata tensors must be on one CUDA device");
        }
    }
    let max_pages_per_sequence = page_table.dims2()?.1;
    let scheduler_len = 2 * schedule.batch_size.div_ceil(4) * 4 + 1;
    if paged_kv_indptr.dims1()? != schedule.batch_size + 1
        || paged_kv_last_page_len.dims1()? != schedule.batch_size
        || page_table.dims2()?.0 != schedule.batch_size
        || seqused_k.dims1()? != schedule.batch_size
        || cu_seqlens_q.dims1()? != schedule.batch_size + 1
        || scheduler_metadata.dims1()? < scheduler_len
        || max_pages_per_sequence == 0
        || schedule.max_seqlen_k > max_pages_per_sequence.saturating_mul(schedule.page_size)
    {
        candle_core::bail!("FA3 metadata shapes do not match schedule {schedule:?}");
    }

    let (indptr_storage, indptr_layout) = paged_kv_indptr.storage_and_layout();
    let (indices_storage, indices_layout) = paged_kv_indices.storage_and_layout();
    let (last_storage, last_layout) = paged_kv_last_page_len.storage_and_layout();
    let (page_table_storage, page_table_layout) = page_table.storage_and_layout();
    let (seqused_storage, seqused_layout) = seqused_k.storage_and_layout();
    let (cu_q_storage, cu_q_layout) = cu_seqlens_q.storage_and_layout();
    let (scheduler_storage, scheduler_layout) = scheduler_metadata.storage_and_layout();

    let indptr_storage = match &*indptr_storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("FA3 metadata must be on CUDA"),
    };
    let indices_storage = match &*indices_storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("FA3 metadata must be on CUDA"),
    };
    let last_storage = match &*last_storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("FA3 metadata must be on CUDA"),
    };
    let page_table_storage = match &*page_table_storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("FA3 metadata must be on CUDA"),
    };
    let seqused_storage = match &*seqused_storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("FA3 metadata must be on CUDA"),
    };
    let cu_q_storage = match &*cu_q_storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("FA3 metadata must be on CUDA"),
    };
    let scheduler_storage = match &*scheduler_storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("FA3 metadata must be on CUDA"),
    };
    let (
        CudaStorageSlice::I32(indptr),
        CudaStorageSlice::I32(indices),
        CudaStorageSlice::I32(last),
        CudaStorageSlice::I32(page_table),
        CudaStorageSlice::I32(seqused),
        CudaStorageSlice::I32(cu_q),
        CudaStorageSlice::I32(scheduler),
    ) = (
        &indptr_storage.slice,
        &indices_storage.slice,
        &last_storage.slice,
        &page_table_storage.slice,
        &seqused_storage.slice,
        &cu_q_storage.slice,
        &scheduler_storage.slice,
    )
    else {
        unreachable!()
    };
    let stream = indptr_storage.device().cuda_stream();
    let (indptr_ptr, _indptr_guard) =
        slice_ptr_on_stream(indptr, indptr_layout.start_offset(), &stream);
    let (indices_ptr, _indices_guard) =
        slice_ptr_on_stream(indices, indices_layout.start_offset(), &stream);
    let (last_ptr, _last_guard) = slice_ptr_on_stream(last, last_layout.start_offset(), &stream);
    let (page_table_ptr, _page_table_guard) =
        slice_ptr_on_stream(page_table, page_table_layout.start_offset(), &stream);
    let (seqused_ptr, _seqused_guard) =
        slice_ptr_on_stream(seqused, seqused_layout.start_offset(), &stream);
    let (cu_q_ptr, _cu_q_guard) = slice_ptr_on_stream(cu_q, cu_q_layout.start_offset(), &stream);
    let (scheduler_ptr, _scheduler_guard) =
        slice_ptr_on_stream(scheduler, scheduler_layout.start_offset(), &stream);

    let status = unsafe {
        fa3_fp8_decode_materialize_paged_metadata(
            indptr_ptr as *const i32,
            indices_ptr as *const i32,
            last_ptr as *const i32,
            page_table_ptr as *mut i32,
            seqused_ptr as *mut i32,
            as_i32(schedule.batch_size, "batch size")?,
            as_i32(max_pages_per_sequence, "page table stride")?,
            as_i32(schedule.page_size, "page size")?,
            stream.cu_stream(),
        )
    };
    if status != 0 {
        candle_core::bail!("FA3 paged metadata materialization failed with status {status}");
    }

    let params = Fa3Fp8DecodeScheduleParams {
        cu_seqlens_q: cu_q_ptr as *const i32,
        seqused_k: seqused_ptr as *const i32,
        scheduler_metadata: scheduler_ptr as *mut i32,
        batch_size: as_i32(schedule.batch_size, "batch size")?,
        total_q: as_i32(schedule.total_q, "query count")?,
        num_q_heads: as_i32(schedule.q_heads, "query head count")?,
        num_kv_heads: as_i32(schedule.kv_heads, "KV head count")?,
        head_dim: as_i32(schedule.head_dim, "head dimension")?,
        page_size: as_i32(schedule.page_size, "page size")?,
        max_seqlen_k: as_i32(schedule.max_seqlen_k, "maximum KV length")?,
        num_splits: as_i32(schedule.num_splits, "split count")?,
        num_sm: as_i32(schedule.num_sm, "SM count")?,
        device_id: as_i32(schedule.device_id, "device ordinal")?,
    };
    let status = unsafe { fa3_fp8_decode_prepare(&params, stream.cu_stream()) };
    if status != 0 {
        candle_core::bail!("FA3 scheduler preparation failed with status {status}");
    }
    Ok(())
}

#[cfg(not(has_fa3_fp8_paged))]
pub fn fa3_prepare_decode_metadata(
    _metadata: Fa3DecodeMetadata<'_>,
    _schedule: Fa3DecodeSchedule,
) -> Result<()> {
    candle_core::bail!("FA3 FP8 paged attention was not built for this CUDA target")
}

#[cfg(has_fa3_fp8_paged)]
pub fn fa3_fp8_decode(params: Fa3DecodeParams<'_>) -> Result<Tensor> {
    use crate::cuda::backend::slice_ptr_on_stream;
    use crate::cuda::ffi::{
        fa3_bf16_to_e4m3_static, fa3_fp8_decode_run, Fa3Fp8DecodeParams, Fa3Fp8DecodeScheduleParams,
    };
    use candle_core::backend::BackendStorage;
    use candle_core::Storage;
    use float8::F8E4M3;

    let Fa3DecodeParams {
        query,
        quantized_query,
        key_cache,
        value_cache,
        page_table,
        seqused_k,
        cu_seqlens_q,
        scheduler_metadata,
        output_accum,
        lse_accum,
        output_lse,
        q_descale,
        k_descale,
        v_descale,
        schedule,
        softmax_scale,
    } = params;
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        candle_core::bail!("FA3 softmax scale must be finite and positive");
    }
    if query.dtype() != DType::BF16
        || quantized_query.dtype() != DType::F8E4M3
        || key_cache.dtype() != DType::F8E4M3
        || value_cache.dtype() != DType::F8E4M3
        || output_accum.dtype() != DType::F32
        || lse_accum.dtype() != DType::F32
        || output_lse.dtype() != DType::F32
        || q_descale.dtype() != DType::F32
        || k_descale.dtype() != DType::F32
        || v_descale.dtype() != DType::F32
        || page_table.dtype() != DType::I32
        || seqused_k.dtype() != DType::I32
        || cu_seqlens_q.dtype() != DType::I32
        || scheduler_metadata.dtype() != DType::I32
    {
        candle_core::bail!("FA3 decode tensor dtypes do not match the FP8/BF16 contract");
    }
    let (num_pages, kv_heads, page_size, head_dim) = key_cache.dims4()?;
    let (batch_size, q_heads, query_head_dim) = query.dims3()?;
    let max_pages_per_sequence = page_table.dims2()?.1;
    if schedule.batch_size != batch_size
        || schedule.total_q != batch_size
        || schedule.q_heads != q_heads
        || schedule.kv_heads != kv_heads
        || schedule.head_dim != head_dim
        || schedule.page_size != page_size
        || query_head_dim != head_dim
        || head_dim != 256
        || value_cache.dims4()? != key_cache.dims4()?
        || quantized_query.dims3()? != query.dims3()?
        || page_table.dims2()?.0 != batch_size
        || seqused_k.dims1()? != batch_size
        || cu_seqlens_q.dims1()? != batch_size + 1
        || output_accum.dims4()? != (schedule.num_splits, q_heads, batch_size, head_dim)
        || lse_accum.dims3()? != (schedule.num_splits, q_heads, batch_size)
        || output_lse.dims2()? != (q_heads, batch_size)
        || q_descale.elem_count() != 1
        || k_descale.elem_count() != 1
        || v_descale.elem_count() != 1
        || schedule.max_seqlen_k > max_pages_per_sequence.saturating_mul(page_size)
    {
        candle_core::bail!("FA3 decode tensor shapes do not match schedule {schedule:?}");
    }
    for (name, tensor) in [
        ("query", query),
        ("quantized_query", quantized_query),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("page_table", page_table),
        ("seqused_k", seqused_k),
        ("cu_seqlens_q", cu_seqlens_q),
        ("scheduler_metadata", scheduler_metadata),
        ("output_accum", output_accum),
        ("lse_accum", lse_accum),
        ("output_lse", output_lse),
        ("q_descale", q_descale),
        ("k_descale", k_descale),
        ("v_descale", v_descale),
    ] {
        if !tensor.is_contiguous() {
            candle_core::bail!("FA3 expects contiguous {name}");
        }
        if tensor.device().location() != query.device().location() {
            candle_core::bail!("FA3 decode tensors must be on one CUDA device");
        }
    }
    let output = unsafe { Tensor::empty(query.shape().clone(), DType::BF16, query.device())? };

    {
        let (query_storage, query_layout) = query.storage_and_layout();
        let (quantized_storage, quantized_layout) = quantized_query.storage_and_layout();
        let (key_storage, key_layout) = key_cache.storage_and_layout();
        let (value_storage, value_layout) = value_cache.storage_and_layout();
        let (page_table_storage, page_table_layout) = page_table.storage_and_layout();
        let (seqused_storage, seqused_layout) = seqused_k.storage_and_layout();
        let (cu_q_storage, cu_q_layout) = cu_seqlens_q.storage_and_layout();
        let (scheduler_storage, scheduler_layout) = scheduler_metadata.storage_and_layout();
        let (out_accum_storage, out_accum_layout) = output_accum.storage_and_layout();
        let (lse_accum_storage, lse_accum_layout) = lse_accum.storage_and_layout();
        let (output_lse_storage, output_lse_layout) = output_lse.storage_and_layout();
        let (q_descale_storage, q_descale_layout) = q_descale.storage_and_layout();
        let (k_descale_storage, k_descale_layout) = k_descale.storage_and_layout();
        let (v_descale_storage, v_descale_layout) = v_descale.storage_and_layout();
        let (output_storage, output_layout) = output.storage_and_layout();

        macro_rules! cuda_storage {
            ($storage:ident, $name:literal) => {
                match &*$storage {
                    Storage::Cuda(storage) => storage,
                    _ => candle_core::bail!(concat!("FA3 ", $name, " must be on CUDA")),
                }
            };
        }
        let query_storage = cuda_storage!(query_storage, "query");
        let quantized_storage = cuda_storage!(quantized_storage, "quantized query");
        let key_storage = cuda_storage!(key_storage, "key cache");
        let value_storage = cuda_storage!(value_storage, "value cache");
        let page_table_storage = cuda_storage!(page_table_storage, "page table");
        let seqused_storage = cuda_storage!(seqused_storage, "sequence lengths");
        let cu_q_storage = cuda_storage!(cu_q_storage, "query offsets");
        let scheduler_storage = cuda_storage!(scheduler_storage, "scheduler metadata");
        let out_accum_storage = cuda_storage!(out_accum_storage, "output accumulator");
        let lse_accum_storage = cuda_storage!(lse_accum_storage, "LSE accumulator");
        let output_lse_storage = cuda_storage!(output_lse_storage, "output LSE");
        let q_descale_storage = cuda_storage!(q_descale_storage, "query descale");
        let k_descale_storage = cuda_storage!(k_descale_storage, "key descale");
        let v_descale_storage = cuda_storage!(v_descale_storage, "value descale");
        let output_storage = cuda_storage!(output_storage, "output");

        let stream = query_storage.device().cuda_stream();
        let (query_ptr, _query_guard) = slice_ptr_on_stream(
            query_storage.as_cuda_slice::<half::bf16>()?,
            query_layout.start_offset(),
            &stream,
        );
        let (quantized_ptr, _quantized_guard) = slice_ptr_on_stream(
            quantized_storage.as_cuda_slice::<F8E4M3>()?,
            quantized_layout.start_offset(),
            &stream,
        );
        let (key_ptr, _key_guard) = slice_ptr_on_stream(
            key_storage.as_cuda_slice::<F8E4M3>()?,
            key_layout.start_offset(),
            &stream,
        );
        let (value_ptr, _value_guard) = slice_ptr_on_stream(
            value_storage.as_cuda_slice::<F8E4M3>()?,
            value_layout.start_offset(),
            &stream,
        );
        let (page_table_ptr, _page_table_guard) = slice_ptr_on_stream(
            page_table_storage.as_cuda_slice::<i32>()?,
            page_table_layout.start_offset(),
            &stream,
        );
        let (seqused_ptr, _seqused_guard) = slice_ptr_on_stream(
            seqused_storage.as_cuda_slice::<i32>()?,
            seqused_layout.start_offset(),
            &stream,
        );
        let (cu_q_ptr, _cu_q_guard) = slice_ptr_on_stream(
            cu_q_storage.as_cuda_slice::<i32>()?,
            cu_q_layout.start_offset(),
            &stream,
        );
        let (scheduler_ptr, _scheduler_guard) = slice_ptr_on_stream(
            scheduler_storage.as_cuda_slice::<i32>()?,
            scheduler_layout.start_offset(),
            &stream,
        );
        let (out_accum_ptr, _out_accum_guard) = slice_ptr_on_stream(
            out_accum_storage.as_cuda_slice::<f32>()?,
            out_accum_layout.start_offset(),
            &stream,
        );
        let (lse_accum_ptr, _lse_accum_guard) = slice_ptr_on_stream(
            lse_accum_storage.as_cuda_slice::<f32>()?,
            lse_accum_layout.start_offset(),
            &stream,
        );
        let (output_lse_ptr, _output_lse_guard) = slice_ptr_on_stream(
            output_lse_storage.as_cuda_slice::<f32>()?,
            output_lse_layout.start_offset(),
            &stream,
        );
        let (q_descale_ptr, _q_descale_guard) = slice_ptr_on_stream(
            q_descale_storage.as_cuda_slice::<f32>()?,
            q_descale_layout.start_offset(),
            &stream,
        );
        let (k_descale_ptr, _k_descale_guard) = slice_ptr_on_stream(
            k_descale_storage.as_cuda_slice::<f32>()?,
            k_descale_layout.start_offset(),
            &stream,
        );
        let (v_descale_ptr, _v_descale_guard) = slice_ptr_on_stream(
            v_descale_storage.as_cuda_slice::<f32>()?,
            v_descale_layout.start_offset(),
            &stream,
        );
        let (output_ptr, _output_guard) = slice_ptr_on_stream(
            output_storage.as_cuda_slice::<half::bf16>()?,
            output_layout.start_offset(),
            &stream,
        );

        let status = unsafe {
            fa3_bf16_to_e4m3_static(
                query_ptr as *const core::ffi::c_void,
                quantized_ptr as *mut core::ffi::c_void,
                as_i32(batch_size, "batch size")?,
                as_i32(q_heads.saturating_mul(head_dim), "query row width")?,
                i64::try_from(query_layout.stride()[0]).map_err(candle_core::Error::wrap)?,
                i64::try_from(quantized_layout.stride()[0]).map_err(candle_core::Error::wrap)?,
                q_descale_ptr as *const f32,
                stream.cu_stream(),
            )
        };
        if status != 0 {
            candle_core::bail!("FA3 query quantization failed with status {status}");
        }

        let schedule_params = Fa3Fp8DecodeScheduleParams {
            cu_seqlens_q: cu_q_ptr as *const i32,
            seqused_k: seqused_ptr as *const i32,
            scheduler_metadata: scheduler_ptr as *mut i32,
            batch_size: as_i32(schedule.batch_size, "batch size")?,
            total_q: as_i32(schedule.total_q, "query count")?,
            num_q_heads: as_i32(schedule.q_heads, "query head count")?,
            num_kv_heads: as_i32(schedule.kv_heads, "KV head count")?,
            head_dim: as_i32(schedule.head_dim, "head dimension")?,
            page_size: as_i32(schedule.page_size, "page size")?,
            max_seqlen_k: as_i32(schedule.max_seqlen_k, "maximum KV length")?,
            num_splits: as_i32(schedule.num_splits, "split count")?,
            num_sm: as_i32(schedule.num_sm, "SM count")?,
            device_id: as_i32(schedule.device_id, "device ordinal")?,
        };
        let ffi_params = Fa3Fp8DecodeParams {
            schedule: schedule_params,
            q: quantized_ptr as *const core::ffi::c_void,
            k: key_ptr as *const core::ffi::c_void,
            v: value_ptr as *const core::ffi::c_void,
            out: output_ptr as *mut core::ffi::c_void,
            softmax_lse: output_lse_ptr as *mut f32,
            out_accum: out_accum_ptr as *mut f32,
            softmax_lse_accum: lse_accum_ptr as *mut f32,
            page_table: page_table_ptr as *const i32,
            q_descale: q_descale_ptr as *const f32,
            k_descale: k_descale_ptr as *const f32,
            v_descale: v_descale_ptr as *const f32,
            q_row_stride: i64::try_from(quantized_layout.stride()[0])
                .map_err(candle_core::Error::wrap)?,
            q_head_stride: i64::try_from(quantized_layout.stride()[1])
                .map_err(candle_core::Error::wrap)?,
            k_token_stride: i64::try_from(key_layout.stride()[2])
                .map_err(candle_core::Error::wrap)?,
            k_head_stride: i64::try_from(key_layout.stride()[1])
                .map_err(candle_core::Error::wrap)?,
            k_page_stride: i64::try_from(key_layout.stride()[0])
                .map_err(candle_core::Error::wrap)?,
            v_token_stride: i64::try_from(value_layout.stride()[2])
                .map_err(candle_core::Error::wrap)?,
            v_head_stride: i64::try_from(value_layout.stride()[1])
                .map_err(candle_core::Error::wrap)?,
            v_page_stride: i64::try_from(value_layout.stride()[0])
                .map_err(candle_core::Error::wrap)?,
            out_row_stride: i64::try_from(output_layout.stride()[0])
                .map_err(candle_core::Error::wrap)?,
            out_head_stride: i64::try_from(output_layout.stride()[1])
                .map_err(candle_core::Error::wrap)?,
            page_table_batch_stride: i64::try_from(page_table_layout.stride()[0])
                .map_err(candle_core::Error::wrap)?,
            q_descale_batch_stride: 0,
            q_descale_head_stride: 0,
            k_descale_batch_stride: 0,
            k_descale_head_stride: 0,
            v_descale_batch_stride: 0,
            v_descale_head_stride: 0,
            num_pages: as_i32(num_pages, "page count")?,
            max_pages_per_sequence: as_i32(max_pages_per_sequence, "maximum pages")?,
            softmax_scale,
            scheduler_metadata_prepared: 1,
        };
        let status = unsafe { fa3_fp8_decode_run(&ffi_params, stream.cu_stream()) };
        if status != 0 {
            candle_core::bail!("FA3 FP8 decode failed with status {status}");
        }
    }
    Ok(output)
}

#[cfg(not(has_fa3_fp8_paged))]
pub fn fa3_fp8_decode(_params: Fa3DecodeParams<'_>) -> Result<Tensor> {
    candle_core::bail!("FA3 FP8 paged attention was not built for this CUDA target")
}
