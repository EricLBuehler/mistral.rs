use super::*;

pub(super) const fn cuda_graph_exact_batch_buckets(
    component: CudaGraphComponent,
    q_len: usize,
) -> usize {
    if matches!(component, CudaGraphComponent::Target) && q_len == 1 {
        TARGET_SINGLE_TOKEN_EXACT_BATCH_BUCKETS
    } else {
        CUDA_GRAPH_COARSE_EXACT_BATCH_BUCKETS
    }
}

/// Graph batch bucket a decode batch pads up to, or None when it is too large to graph.
pub(crate) fn cuda_graph_batch_bucket(
    component: CudaGraphComponent,
    q_len: usize,
    batch: usize,
) -> Option<usize> {
    if batch == 0 {
        None
    } else if batch <= cuda_graph_exact_batch_buckets(component, q_len) {
        Some(batch)
    } else {
        let bucket = batch
            .div_ceil(CUDA_GRAPH_BATCH_BUCKET_GRANULARITY)
            .saturating_mul(CUDA_GRAPH_BATCH_BUCKET_GRANULARITY);
        (bucket <= CUDA_GRAPH_MAX_BATCH_BUCKET).then_some(bucket)
    }
}

pub(crate) fn cuda_graph_precapture_max_batch(
    component: CudaGraphComponent,
    q_len: usize,
    configured_max: usize,
) -> usize {
    if configured_max == 0 {
        0
    } else {
        cuda_graph_batch_bucket(component, q_len, configured_max)
            .unwrap_or(CUDA_GRAPH_MAX_BATCH_BUCKET)
    }
}

pub(crate) fn target_cuda_graph_cache_capacity(
    startup_shapes: usize,
    runtime_batch_shapes: usize,
) -> usize {
    let working_set = startup_shapes.saturating_add(runtime_batch_shapes);
    let rounded = working_set
        .div_ceil(TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY_QUANTUM)
        .saturating_mul(TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY_QUANTUM);
    rounded.clamp(
        TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY,
        TARGET_CUDA_DECODE_GRAPH_CACHE_MAX_CAPACITY,
    )
}

/// The batch buckets captured ahead of time at load.
pub(crate) fn cuda_graph_precapture_batches(
    component: CudaGraphComponent,
    q_len: usize,
) -> impl Iterator<Item = usize> {
    let exact = cuda_graph_exact_batch_buckets(component, q_len);
    (1..=exact).chain(
        (exact + CUDA_GRAPH_BATCH_BUCKET_GRANULARITY..=CUDA_GRAPH_PRECAPTURE_MAX_BATCH)
            .step_by(CUDA_GRAPH_BATCH_BUCKET_GRANULARITY),
    )
}

pub(crate) fn cuda_graph_startup_capture_allowed(q_len: usize) -> bool {
    q_len > 0
}

#[cfg(target_family = "unix")]
pub(crate) fn prepare_fa3_decode_schedules(
    metadata: &PagedAttentionInputMetadata,
) -> candle_core::Result<()> {
    let Some(flashinfer) = metadata.flashinfer.as_ref() else {
        return Ok(());
    };
    flashinfer.for_each_fa3_decode_schedule(|prepare| {
        mistralrs_paged_attn::fa3_prepare_decode_metadata(
            mistralrs_paged_attn::Fa3DecodeMetadata {
                paged_kv_indptr: prepare.paged_kv_indptr,
                paged_kv_indices: prepare.paged_kv_indices,
                paged_kv_last_page_len: prepare.paged_kv_last_page_len,
                page_table: &prepare.buffers.page_table,
                seqused_k: &prepare.buffers.seqused_k,
                cu_seqlens_q: &prepare.buffers.cu_seqlens_q,
                scheduler_metadata: &prepare.buffers.scheduler_metadata,
            },
            prepare.buffers.schedule(prepare.key)?,
        )
    })
}

#[cfg(not(target_family = "unix"))]
pub(crate) fn prepare_fa3_decode_schedules(
    _metadata: &PagedAttentionInputMetadata,
) -> candle_core::Result<()> {
    Ok(())
}
