#[cfg(has_fa3_fp8_paged)]
use candle_core::DType;
use candle_core::{Result, Tensor};

pub const USE_FA3_FP8_PAGED: bool = cfg!(has_fa3_fp8_paged);
pub const FA3_DECODE_MAX_QUERY_LEN: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fa3DecodeSchedule {
    pub batch_size: usize,
    pub query_len: usize,
    pub total_q: usize,
    pub causal: bool,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fa3PagedMetadataLayout {
    pub source_rows_per_sequence: usize,
    pub source_row_offset: usize,
}

impl Fa3PagedMetadataLayout {
    pub const fn per_sequence() -> Self {
        Self {
            source_rows_per_sequence: 1,
            source_row_offset: 0,
        }
    }

    pub const fn decode(query_len: usize) -> Self {
        Self {
            source_rows_per_sequence: query_len,
            source_row_offset: query_len.saturating_sub(1),
        }
    }

    #[cfg(any(test, has_fa3_fp8_paged))]
    fn source_rows(self, batch_size: usize) -> Option<usize> {
        batch_size.checked_mul(self.source_rows_per_sequence)
    }

    #[cfg(any(test, has_fa3_fp8_paged))]
    fn valid(self) -> bool {
        self.source_rows_per_sequence > 0
            && self.source_rows_per_sequence <= FA3_DECODE_MAX_QUERY_LEN
            && self.source_row_offset < self.source_rows_per_sequence
    }
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
    fa3_prepare_paged_metadata(
        metadata,
        schedule,
        Fa3PagedMetadataLayout::decode(schedule.query_len),
    )
}

#[cfg(has_fa3_fp8_paged)]
pub fn fa3_prepare_paged_metadata(
    metadata: Fa3DecodeMetadata<'_>,
    schedule: Fa3DecodeSchedule,
    metadata_layout: Fa3PagedMetadataLayout,
) -> Result<()> {
    use crate::cuda::backend::slice_ptr_on_stream;
    use crate::cuda::ffi::{
        fa3_fp8_decode_prepare, fa3_fp8_paged_materialize_metadata, Fa3Fp8DecodeScheduleParams,
    };
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::CudaStorageSlice;
    use candle_core::Storage;

    if schedule.batch_size == 0
        || schedule.query_len == 0
        || schedule.query_len > FA3_DECODE_MAX_QUERY_LEN
        || schedule.total_q == 0
        || schedule.batch_size.checked_mul(schedule.query_len) != Some(schedule.total_q)
        || schedule.q_heads == 0
        || schedule.kv_heads == 0
        || !schedule.q_heads.is_multiple_of(schedule.kv_heads)
        || schedule.head_dim != 256
        || schedule.page_size == 0
        || schedule.max_seqlen_k == 0
        || schedule.num_splits <= 1
        || schedule.num_splits > 256
        || schedule.num_sm == 0
        || !metadata_layout.valid()
        || metadata_layout.source_rows(schedule.batch_size).is_none()
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
    let source_rows = metadata_layout
        .source_rows(schedule.batch_size)
        .ok_or_else(|| candle_core::Error::msg("FA3 metadata row count overflow"))?;
    let scheduler_vectors = 2 + usize::from(schedule.causal);
    let scheduler_len = scheduler_vectors * schedule.batch_size.div_ceil(4) * 4 + 1;
    if paged_kv_indptr.dims1()? != source_rows + 1
        || paged_kv_last_page_len.dims1()? != source_rows
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
    let stream = page_table_storage.device().cuda_stream();
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
        fa3_fp8_paged_materialize_metadata(
            indptr_ptr as *const i32,
            indices_ptr as *const i32,
            last_ptr as *const i32,
            page_table_ptr as *mut i32,
            seqused_ptr as *mut i32,
            as_i32(schedule.batch_size, "batch size")?,
            as_i32(
                metadata_layout.source_rows_per_sequence,
                "metadata rows per sequence",
            )?,
            as_i32(metadata_layout.source_row_offset, "metadata row offset")?,
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
        query_len: as_i32(schedule.query_len, "query length")?,
        total_q: as_i32(schedule.total_q, "query count")?,
        causal: i32::from(schedule.causal),
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

#[cfg(not(has_fa3_fp8_paged))]
pub fn fa3_prepare_paged_metadata(
    _metadata: Fa3DecodeMetadata<'_>,
    _schedule: Fa3DecodeSchedule,
    _metadata_layout: Fa3PagedMetadataLayout,
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
    let (total_q, q_heads, query_head_dim) = query.dims3()?;
    let max_pages_per_sequence = page_table.dims2()?.1;
    if schedule.total_q != total_q
        || schedule.batch_size.checked_mul(schedule.query_len) != Some(total_q)
        || schedule.q_heads != q_heads
        || schedule.kv_heads != kv_heads
        || schedule.head_dim != head_dim
        || schedule.page_size != page_size
        || query_head_dim != head_dim
        || head_dim != 256
        || value_cache.dims4()? != key_cache.dims4()?
        || quantized_query.dims3()? != query.dims3()?
        || page_table.dims2()?.0 != schedule.batch_size
        || seqused_k.dims1()? != schedule.batch_size
        || cu_seqlens_q.dims1()? != schedule.batch_size + 1
        || output_accum.dims4()? != (schedule.num_splits, q_heads, total_q, head_dim)
        || lse_accum.dims3()? != (schedule.num_splits, q_heads, total_q)
        || output_lse.dims2()? != (q_heads, total_q)
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
                as_i32(total_q, "query count")?,
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
            query_len: as_i32(schedule.query_len, "query length")?,
            total_q: as_i32(schedule.total_q, "query count")?,
            causal: i32::from(schedule.causal),
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

#[cfg(test)]
mod tests {
    use super::Fa3PagedMetadataLayout;

    #[cfg(has_fa3_fp8_paged)]
    use super::{
        fa3_fp8_decode, fa3_prepare_decode_metadata, fa3_prepare_paged_metadata, Fa3DecodeMetadata,
        Fa3DecodeParams, Fa3DecodeSchedule,
    };
    #[cfg(has_fa3_fp8_paged)]
    use candle_core::{DType, Device, Result, Tensor};

    #[cfg(has_fa3_fp8_paged)]
    const TEST_BATCH_SIZE: usize = 2;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_QUERY_LEN: usize = 3;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_Q_HEADS: usize = 4;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_KV_HEADS: usize = 1;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_HEAD_DIM: usize = 256;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_PAGE_SIZE: usize = 32;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_NUM_PAGES: usize = 4;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_MAX_PAGES_PER_SEQUENCE: usize = 2;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_MAX_SEQUENCE_LEN: usize = 40;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_NUM_SPLITS: usize = 2;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_OUTPUT_TOLERANCE: f32 = 0.0;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_REFERENCE_MAX_TOLERANCE: f32 = 0.01;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_REFERENCE_MEAN_TOLERANCE: f32 = 0.003;
    #[cfg(has_fa3_fp8_paged)]
    const TEST_SCALES: TestScales = TestScales {
        q: 0.5,
        k: 0.25,
        v: 0.75,
    };

    #[cfg(has_fa3_fp8_paged)]
    struct TestMetadata {
        paged_kv_indptr: Tensor,
        paged_kv_indices: Tensor,
        paged_kv_last_page_len: Tensor,
        page_table: Tensor,
        seqused_k: Tensor,
        cu_seqlens_q: Tensor,
        scheduler_metadata: Tensor,
    }

    #[cfg(has_fa3_fp8_paged)]
    impl TestMetadata {
        fn new(
            indptr: Vec<i32>,
            indices: Vec<i32>,
            last_page_len: Vec<i32>,
            schedule: Fa3DecodeSchedule,
            device: &Device,
        ) -> Result<Self> {
            let indptr_len = indptr.len();
            let indices_len = indices.len();
            let last_page_len_len = last_page_len.len();
            let scheduler_vectors = 2 + usize::from(schedule.causal);
            let scheduler_len = scheduler_vectors * schedule.batch_size.div_ceil(4) * 4 + 1;
            Ok(Self {
                paged_kv_indptr: Tensor::from_vec(indptr, indptr_len, device)?,
                paged_kv_indices: Tensor::from_vec(indices, indices_len, device)?,
                paged_kv_last_page_len: Tensor::from_vec(last_page_len, last_page_len_len, device)?,
                page_table: unsafe {
                    Tensor::empty(
                        (schedule.batch_size, TEST_MAX_PAGES_PER_SEQUENCE),
                        DType::I32,
                        device,
                    )?
                },
                seqused_k: unsafe { Tensor::empty(schedule.batch_size, DType::I32, device)? },
                cu_seqlens_q: Tensor::from_vec(
                    (0..=schedule.batch_size)
                        .map(|batch| (batch * schedule.query_len) as i32)
                        .collect::<Vec<_>>(),
                    schedule.batch_size + 1,
                    device,
                )?,
                scheduler_metadata: unsafe { Tensor::empty(scheduler_len, DType::I32, device)? },
            })
        }

        fn as_decode_metadata(&self) -> Fa3DecodeMetadata<'_> {
            Fa3DecodeMetadata {
                paged_kv_indptr: &self.paged_kv_indptr,
                paged_kv_indices: &self.paged_kv_indices,
                paged_kv_last_page_len: &self.paged_kv_last_page_len,
                page_table: &self.page_table,
                seqused_k: &self.seqused_k,
                cu_seqlens_q: &self.cu_seqlens_q,
                scheduler_metadata: &self.scheduler_metadata,
            }
        }
    }

    #[cfg(has_fa3_fp8_paged)]
    struct TestRun {
        output: Tensor,
        quantized_query: Tensor,
    }

    #[cfg(has_fa3_fp8_paged)]
    #[derive(Clone, Copy)]
    struct TestScales {
        q: f32,
        k: f32,
        v: f32,
    }

    #[cfg(has_fa3_fp8_paged)]
    impl TestScales {
        const UNIT: Self = Self {
            q: 1.0,
            k: 1.0,
            v: 1.0,
        };
    }

    #[cfg(has_fa3_fp8_paged)]
    struct TestFa3Params<'a> {
        query: &'a Tensor,
        key_cache: &'a Tensor,
        value_cache: &'a Tensor,
        metadata: &'a TestMetadata,
        schedule: Fa3DecodeSchedule,
        scales: TestScales,
    }

    #[cfg(has_fa3_fp8_paged)]
    fn run_fa3(params: TestFa3Params<'_>) -> Result<TestRun> {
        let TestFa3Params {
            query,
            key_cache,
            value_cache,
            metadata,
            schedule,
            scales,
        } = params;
        let quantized_query =
            unsafe { Tensor::empty(query.shape().clone(), DType::F8E4M3, query.device())? };
        let output_accum = unsafe {
            Tensor::empty(
                (
                    schedule.num_splits,
                    schedule.q_heads,
                    schedule.total_q,
                    schedule.head_dim,
                ),
                DType::F32,
                query.device(),
            )?
        };
        let lse_accum = unsafe {
            Tensor::empty(
                (schedule.num_splits, schedule.q_heads, schedule.total_q),
                DType::F32,
                query.device(),
            )?
        };
        let output_lse = unsafe {
            Tensor::empty(
                (schedule.q_heads, schedule.total_q),
                DType::F32,
                query.device(),
            )?
        };
        let q_descale = Tensor::new(&[scales.q], query.device())?;
        let k_descale = Tensor::new(&[scales.k], query.device())?;
        let v_descale = Tensor::new(&[scales.v], query.device())?;
        let output = fa3_fp8_decode(Fa3DecodeParams {
            query,
            quantized_query: &quantized_query,
            key_cache,
            value_cache,
            page_table: &metadata.page_table,
            seqused_k: &metadata.seqused_k,
            cu_seqlens_q: &metadata.cu_seqlens_q,
            scheduler_metadata: &metadata.scheduler_metadata,
            output_accum: &output_accum,
            lse_accum: &lse_accum,
            output_lse: &output_lse,
            q_descale: &q_descale,
            k_descale: &k_descale,
            v_descale: &v_descale,
            schedule,
            softmax_scale: 1.0 / (schedule.head_dim as f32).sqrt(),
        })?;
        Ok(TestRun {
            output,
            quantized_query,
        })
    }

    #[cfg(has_fa3_fp8_paged)]
    fn max_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
        let lhs = lhs
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rhs = rhs
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut max_diff = 0.0f32;
        for (lhs, rhs) in lhs.into_iter().zip(rhs) {
            assert!(lhs.is_finite() && rhs.is_finite());
            max_diff = max_diff.max((lhs - rhs).abs());
        }
        Ok(max_diff)
    }

    #[cfg(has_fa3_fp8_paged)]
    fn quantize_bf16_to_e4m3(value: f32, descale: f32) -> f32 {
        float8::F8E4M3::from_f32(half::bf16::from_f32(value).to_f32() / descale).to_f32()
    }

    #[cfg(has_fa3_fp8_paged)]
    struct ScalarPagedAttention<'a> {
        query: &'a [f32],
        key: &'a [f32],
        value: &'a [f32],
        page_table: &'a [Vec<i32>],
        sequence_lengths: &'a [usize],
        schedule: Fa3DecodeSchedule,
        scales: TestScales,
    }

    #[cfg(has_fa3_fp8_paged)]
    impl ScalarPagedAttention<'_> {
        fn run(&self) -> Vec<f32> {
            let schedule = self.schedule;
            let mut output = vec![0.0f32; schedule.total_q * schedule.q_heads * schedule.head_dim];
            let queries_per_kv_head = schedule.q_heads / schedule.kv_heads;
            let softmax_scale = 1.0 / (schedule.head_dim as f32).sqrt();
            for batch in 0..schedule.batch_size {
                let sequence_len = self.sequence_lengths[batch];
                assert!(sequence_len >= schedule.query_len);
                for query_row in 0..schedule.query_len {
                    let total_query_row = batch * schedule.query_len + query_row;
                    let attended_tokens = if schedule.causal {
                        sequence_len - schedule.query_len + query_row + 1
                    } else {
                        sequence_len
                    };
                    for query_head in 0..schedule.q_heads {
                        let kv_head = query_head / queries_per_kv_head;
                        let mut scores = Vec::with_capacity(attended_tokens);
                        for token in 0..attended_tokens {
                            let page = self.page_table[batch][token / schedule.page_size] as usize;
                            let physical_token =
                                page * schedule.page_size + token % schedule.page_size;
                            let mut dot = 0.0f32;
                            for dim in 0..schedule.head_dim {
                                let query_idx = (total_query_row * schedule.q_heads + query_head)
                                    * schedule.head_dim
                                    + dim;
                                let key_idx = (physical_token * schedule.kv_heads + kv_head)
                                    * schedule.head_dim
                                    + dim;
                                let q = quantize_bf16_to_e4m3(self.query[query_idx], self.scales.q)
                                    * self.scales.q;
                                let k = quantize_bf16_to_e4m3(self.key[key_idx], self.scales.k)
                                    * self.scales.k;
                                dot += q * k;
                            }
                            scores.push(dot * softmax_scale);
                        }
                        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let denominator = scores
                            .iter_mut()
                            .map(|score| {
                                *score = (*score - max_score).exp();
                                *score
                            })
                            .sum::<f32>();
                        for dim in 0..schedule.head_dim {
                            let mut sum = 0.0f32;
                            for (token, weight) in scores.iter().enumerate() {
                                let page =
                                    self.page_table[batch][token / schedule.page_size] as usize;
                                let physical_token =
                                    page * schedule.page_size + token % schedule.page_size;
                                let value_idx = (physical_token * schedule.kv_heads + kv_head)
                                    * schedule.head_dim
                                    + dim;
                                let v = quantize_bf16_to_e4m3(self.value[value_idx], self.scales.v)
                                    * self.scales.v;
                                sum += weight / denominator * v;
                            }
                            let output_idx = (total_query_row * schedule.q_heads + query_head)
                                * schedule.head_dim
                                + dim;
                            output[output_idx] = sum;
                        }
                    }
                }
            }
            output
        }
    }

    #[cfg(has_fa3_fp8_paged)]
    fn error_stats(actual: &[f32], expected: &[f32]) -> (f32, f32) {
        assert_eq!(actual.len(), expected.len());
        let mut max = 0.0f32;
        let mut total = 0.0f32;
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite() && expected.is_finite());
            let error = (actual - expected).abs();
            max = max.max(error);
            total += error;
        }
        (max, total / actual.len() as f32)
    }

    #[test]
    fn metadata_layout_selects_decode_tail_or_per_sequence_row() {
        let decode = Fa3PagedMetadataLayout::decode(7);
        assert!(decode.valid());
        assert_eq!(decode.source_rows(3), Some(21));
        assert_eq!(decode.source_row_offset, 6);

        let prefill = Fa3PagedMetadataLayout::per_sequence();
        assert!(prefill.valid());
        assert_eq!(prefill.source_rows(3), Some(3));
        assert_eq!(prefill.source_row_offset, 0);
    }

    #[test]
    fn metadata_layout_rejects_empty_and_out_of_range_rows() {
        assert!(!Fa3PagedMetadataLayout::decode(0).valid());
        assert!(!Fa3PagedMetadataLayout {
            source_rows_per_sequence: 2,
            source_row_offset: 2,
        }
        .valid());
    }

    #[cfg(has_fa3_fp8_paged)]
    #[test]
    fn per_sequence_metadata_matches_decode_tail_attention() -> Result<()> {
        use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;

        if !crate::cuda::USE_FP8 {
            return Ok(());
        }
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let Device::Cuda(cuda_device) = &device else {
            unreachable!()
        };
        let stream = cuda_device.cuda_stream();
        let context = stream.context();
        let Ok(compute_major) =
            context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        else {
            return Ok(());
        };
        let Ok(num_sm) =
            context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        else {
            return Ok(());
        };
        if compute_major != 9 || num_sm <= 0 {
            return Ok(());
        }

        let total_q = TEST_BATCH_SIZE * TEST_QUERY_LEN;
        let schedule = Fa3DecodeSchedule {
            batch_size: TEST_BATCH_SIZE,
            query_len: TEST_QUERY_LEN,
            total_q,
            causal: true,
            q_heads: TEST_Q_HEADS,
            kv_heads: TEST_KV_HEADS,
            head_dim: TEST_HEAD_DIM,
            page_size: TEST_PAGE_SIZE,
            max_seqlen_k: TEST_MAX_SEQUENCE_LEN,
            num_splits: TEST_NUM_SPLITS,
            num_sm: num_sm as usize,
            device_id: context.ordinal(),
        };

        let cache_tokens = TEST_NUM_PAGES * TEST_PAGE_SIZE;
        let key = Tensor::from_vec(
            (0..cache_tokens * TEST_KV_HEADS * TEST_HEAD_DIM)
                .map(|idx| ((idx % 31) as f32 - 15.0) / 16.0)
                .collect::<Vec<_>>(),
            (cache_tokens, TEST_KV_HEADS, TEST_HEAD_DIM),
            &Device::Cpu,
        )?
        .to_dtype(DType::BF16)?
        .to_device(&device)?;
        let value = Tensor::from_vec(
            (0..cache_tokens * TEST_KV_HEADS * TEST_HEAD_DIM)
                .map(|idx| ((idx % 23) as f32 - 11.0) / 16.0)
                .collect::<Vec<_>>(),
            (cache_tokens, TEST_KV_HEADS, TEST_HEAD_DIM),
            &Device::Cpu,
        )?
        .to_dtype(DType::BF16)?
        .to_device(&device)?;
        let key_cache = unsafe {
            Tensor::empty(
                (TEST_NUM_PAGES, TEST_KV_HEADS, TEST_PAGE_SIZE, TEST_HEAD_DIM),
                DType::F8E4M3,
                &device,
            )?
        };
        let value_cache = unsafe { Tensor::empty_like(&key_cache)? };
        let slot_mapping = Tensor::from_vec(
            (0..cache_tokens)
                .map(|slot| slot as i64)
                .collect::<Vec<_>>(),
            cache_tokens,
            &device,
        )?;
        crate::cuda::backend::reshape_and_cache_flashinfer(
            &key,
            &value,
            &key_cache,
            &value_cache,
            &slot_mapping,
            crate::KvCacheScales { k: 1.0, v: 1.0 },
        )?;
        let query = Tensor::from_vec(
            (0..total_q * TEST_Q_HEADS * TEST_HEAD_DIM)
                .map(|idx| ((idx % 19) as f32 - 9.0) / 16.0)
                .collect::<Vec<_>>(),
            (total_q, TEST_Q_HEADS, TEST_HEAD_DIM),
            &Device::Cpu,
        )?
        .to_dtype(DType::BF16)?
        .to_device(&device)?;

        let per_sequence = TestMetadata::new(
            vec![0, 2, 4],
            vec![0, 1, 2, 3],
            vec![3, 8],
            schedule,
            &device,
        )?;
        let decode_tail = TestMetadata::new(
            vec![0, 1, 2, 4, 5, 6, 8],
            vec![3, 2, 0, 1, 1, 0, 2, 3],
            vec![1, 2, 3, 1, 2, 8],
            schedule,
            &device,
        )?;
        fa3_prepare_paged_metadata(
            per_sequence.as_decode_metadata(),
            schedule,
            Fa3PagedMetadataLayout::per_sequence(),
        )?;
        fa3_prepare_decode_metadata(decode_tail.as_decode_metadata(), schedule)?;

        assert_eq!(
            per_sequence.page_table.to_vec2::<i32>()?,
            decode_tail.page_table.to_vec2::<i32>()?
        );
        assert_eq!(
            per_sequence.seqused_k.to_vec1::<i32>()?,
            decode_tail.seqused_k.to_vec1::<i32>()?
        );

        let per_sequence_output = run_fa3(TestFa3Params {
            query: &query,
            key_cache: &key_cache,
            value_cache: &value_cache,
            metadata: &per_sequence,
            schedule,
            scales: TestScales::UNIT,
        })?
        .output;
        let decode_tail_output = run_fa3(TestFa3Params {
            query: &query,
            key_cache: &key_cache,
            value_cache: &value_cache,
            metadata: &decode_tail,
            schedule,
            scales: TestScales::UNIT,
        })?
        .output;
        let diff = max_diff(&per_sequence_output, &decode_tail_output)?;
        assert!(
            diff <= TEST_OUTPUT_TOLERANCE,
            "per-sequence and decode-tail output difference {diff} exceeds {TEST_OUTPUT_TOLERANCE}"
        );
        Ok(())
    }

    #[cfg(has_fa3_fp8_paged)]
    #[test]
    fn direct_fp8_paged_prefill_matches_scalar_causal_gqa() -> Result<()> {
        use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;

        if !crate::cuda::USE_FP8 {
            return Ok(());
        }
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let Device::Cuda(cuda_device) = &device else {
            unreachable!()
        };
        let stream = cuda_device.cuda_stream();
        let context = stream.context();
        let Ok(compute_major) =
            context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        else {
            return Ok(());
        };
        let Ok(num_sm) =
            context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        else {
            return Ok(());
        };
        if compute_major != 9 || num_sm <= 0 {
            return Ok(());
        }

        let total_q = TEST_BATCH_SIZE * TEST_QUERY_LEN;
        let schedule = Fa3DecodeSchedule {
            batch_size: TEST_BATCH_SIZE,
            query_len: TEST_QUERY_LEN,
            total_q,
            causal: true,
            q_heads: TEST_Q_HEADS,
            kv_heads: TEST_KV_HEADS,
            head_dim: TEST_HEAD_DIM,
            page_size: TEST_PAGE_SIZE,
            max_seqlen_k: TEST_MAX_SEQUENCE_LEN,
            num_splits: TEST_NUM_SPLITS,
            num_sm: num_sm as usize,
            device_id: context.ordinal(),
        };
        let scales = TEST_SCALES;
        assert!(scales.q != 1.0 && scales.k != 1.0 && scales.v != 1.0);

        let cache_tokens = TEST_NUM_PAGES * TEST_PAGE_SIZE;
        let key_values = (0..cache_tokens * TEST_KV_HEADS * TEST_HEAD_DIM)
            .map(|idx| (((idx * 13 + 7) % 47) as f32 - 23.0) / 32.0)
            .collect::<Vec<_>>();
        let value_values = (0..cache_tokens * TEST_KV_HEADS * TEST_HEAD_DIM)
            .map(|idx| (((idx * 11 + 5) % 37) as f32 - 18.0) / 32.0)
            .collect::<Vec<_>>();
        let query_values = (0..total_q * TEST_Q_HEADS * TEST_HEAD_DIM)
            .map(|idx| (((idx * 17 + 3) % 41) as f32 - 20.0) / 32.0)
            .collect::<Vec<_>>();
        let key = Tensor::from_vec(
            key_values.clone(),
            (cache_tokens, TEST_KV_HEADS, TEST_HEAD_DIM),
            &Device::Cpu,
        )?
        .to_dtype(DType::BF16)?
        .to_device(&device)?;
        let value = Tensor::from_vec(
            value_values.clone(),
            (cache_tokens, TEST_KV_HEADS, TEST_HEAD_DIM),
            &Device::Cpu,
        )?
        .to_dtype(DType::BF16)?
        .to_device(&device)?;
        let query = Tensor::from_vec(
            query_values.clone(),
            (total_q, TEST_Q_HEADS, TEST_HEAD_DIM),
            &Device::Cpu,
        )?
        .to_dtype(DType::BF16)?
        .to_device(&device)?;
        let key_cache = unsafe {
            Tensor::empty(
                (TEST_NUM_PAGES, TEST_KV_HEADS, TEST_PAGE_SIZE, TEST_HEAD_DIM),
                DType::F8E4M3,
                &device,
            )?
        };
        let value_cache = unsafe { Tensor::empty_like(&key_cache)? };
        let slot_mapping = Tensor::from_vec(
            (0..cache_tokens)
                .map(|slot| slot as i64)
                .collect::<Vec<_>>(),
            cache_tokens,
            &device,
        )?;
        crate::cuda::backend::reshape_and_cache_flashinfer(
            &key,
            &value,
            &key_cache,
            &value_cache,
            &slot_mapping,
            crate::KvCacheScales {
                k: scales.k,
                v: scales.v,
            },
        )?;

        let metadata = TestMetadata::new(
            vec![0, 2, 4],
            vec![1, 0, 3, 2],
            vec![3, 8],
            schedule,
            &device,
        )?;
        fa3_prepare_paged_metadata(
            metadata.as_decode_metadata(),
            schedule,
            Fa3PagedMetadataLayout::per_sequence(),
        )?;
        let reference_page_table = vec![vec![1, 0], vec![3, 2]];
        let sequence_lengths = vec![35usize, 40];
        assert_eq!(metadata.page_table.to_vec2::<i32>()?, reference_page_table);
        assert_eq!(metadata.seqused_k.to_vec1::<i32>()?, vec![35, 40]);

        let run = run_fa3(TestFa3Params {
            query: &query,
            key_cache: &key_cache,
            value_cache: &value_cache,
            metadata: &metadata,
            schedule,
            scales,
        })?;
        let expected = ScalarPagedAttention {
            query: &query_values,
            key: &key_values,
            value: &value_values,
            page_table: &reference_page_table,
            sequence_lengths: &sequence_lengths,
            schedule,
            scales,
        }
        .run();
        let actual = run
            .output
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let (max_error, mean_error) = error_stats(&actual, &expected);
        assert!(
            max_error <= TEST_REFERENCE_MAX_TOLERANCE
                && mean_error <= TEST_REFERENCE_MEAN_TOLERANCE,
            "FA3 scalar-reference errors max={max_error} mean={mean_error} exceed max={} mean={}",
            TEST_REFERENCE_MAX_TOLERANCE,
            TEST_REFERENCE_MEAN_TOLERANCE
        );

        let actual_query = run
            .quantized_query
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<float8::F8E4M3>()?
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let expected_query = query_values
            .iter()
            .map(|value| quantize_bf16_to_e4m3(*value, scales.q))
            .collect::<Vec<_>>();
        assert_eq!(error_stats(&actual_query, &expected_query).0, 0.0);

        for (cache, source, scale) in [
            (&key_cache, &key_values, scales.k),
            (&value_cache, &value_values, scales.v),
        ] {
            let actual_cache = cache
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<float8::F8E4M3>()?
                .into_iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            let expected_cache = source
                .iter()
                .map(|value| quantize_bf16_to_e4m3(*value, scale))
                .collect::<Vec<_>>();
            assert_eq!(error_stats(&actual_cache, &expected_cache).0, 0.0);
        }
        Ok(())
    }
}
