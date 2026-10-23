use anyhow::Result;
#[cfg(all(feature = "cuda", target_family = "unix"))]
use candle_core::cuda_backend::cudarc::driver::{CudaEvent, CudaStream};
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::{Device, Tensor};
#[cfg(all(feature = "cuda", target_family = "unix"))]
use candle_core::{DeviceLocation, TensorId};
#[cfg(all(feature = "cuda", target_family = "unix"))]
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, Weak},
};

#[cfg(feature = "cuda")]
use super::Fa3DecodeState;
use super::{
    DeviceTensorMap, FlashInferMetadata, FlashInferPagedAttentionView,
    FlashInferPagedAttentionViews, FlashInferPagedKv, FlashInferTilePlan,
};
#[cfg(all(feature = "cuda", target_family = "unix"))]
use super::{Fa3DecodeBuffers, Fa3DecodeScheduleKey, Fa3DecodeView, Fa3PagedScheduleShape};
use crate::paged_attention::block_table_rows::BlockTableRows;
#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::paged_attention::AttentionBackendKind;
#[cfg(feature = "cuda")]
use crate::paged_attention::ModelConfigLike;

// Split-KV decode chunks each (sequence, kv head) context so the grid reaches about this many
// blocks per SM; grids that are already full keep one 2048-token chunk and skip the partial merge.
const DECODE_SPLIT_BLOCKS_PER_SM: usize = 2;
const DECODE_SPLIT_MIN_TOKENS: usize = 256;
const DECODE_SPLIT_MAX_TOKENS: usize = 2048;
// Used when the SM count can't be queried; errs toward more, smaller chunks.
const DECODE_SPLIT_FALLBACK_SM_COUNT: usize = 64;
#[cfg(all(feature = "cuda", target_family = "unix"))]
const CUDA_STREAM_PER_THREAD_HANDLE: usize = 2;

#[cfg(feature = "cuda")]
fn cuda_sm_count() -> usize {
    use candle_core::cuda::cudarc::driver::{result, sys};
    static SM_COUNT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SM_COUNT.get_or_init(|| {
        result::init()
            .ok()
            .and_then(|_| result::device::get(0).ok())
            .and_then(|dev| unsafe {
                result::device::get_attribute(
                    dev,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                )
                .ok()
            })
            .and_then(|count| usize::try_from(count).ok())
            .filter(|count| *count > 0)
            .unwrap_or(DECODE_SPLIT_FALLBACK_SM_COUNT)
    })
}

#[cfg(not(feature = "cuda"))]
fn cuda_sm_count() -> usize {
    DECODE_SPLIT_FALLBACK_SM_COUNT
}

/// Smallest chunk the planner can pick, in pages: tile plans are sized for it so their shape stays
/// fixed while the live chunk size varies with context.
pub(crate) fn decode_split_capacity_pages(block_size: usize) -> usize {
    DECODE_SPLIT_MIN_TOKENS.div_ceil(block_size).max(1)
}

/// Split-KV chunk size in pages for a decode batch: the largest chunk that still puts about
/// `DECODE_SPLIT_BLOCKS_PER_SM` blocks per SM in flight at the batch's longest context.
pub(crate) fn decode_split_pages(
    block_size: usize,
    batch_size: usize,
    num_kv_heads: usize,
    max_context_len: usize,
) -> usize {
    decode_split_tokens(batch_size, num_kv_heads, cuda_sm_count(), max_context_len)
        .div_ceil(block_size)
        .max(1)
}

fn decode_split_tokens(
    batch_size: usize,
    num_kv_heads: usize,
    sm_count: usize,
    max_context_len: usize,
) -> usize {
    let blocks_unsplit = (batch_size * num_kv_heads).max(1);
    let chunks_needed = (DECODE_SPLIT_BLOCKS_PER_SM * sm_count)
        .div_ceil(blocks_unsplit)
        .max(1);
    let tokens =
        (max_context_len / chunks_needed).clamp(DECODE_SPLIT_MIN_TOKENS, DECODE_SPLIT_MAX_TOKENS);
    1 << (usize::BITS - 1 - tokens.leading_zeros())
}

// Converts scheduler block tables into FlashInfer's paged-KV CSR tensors.
pub(crate) fn make_paged_kv_tensors<T: BlockTableRows + ?Sized>(
    tables: &T,
    context_lens: &[usize],
    block_size: usize,
    padded_indices_len: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let batch_size = tables.len();
    let mut paged_kv_indices = Vec::new();
    let mut paged_kv_indptr = Vec::with_capacity(batch_size + 1);
    let mut paged_kv_last_page_len = Vec::with_capacity(batch_size);
    paged_kv_indptr.push(0i32);
    let mut nnz_pages = 0i32;
    if batch_size != context_lens.len() {
        anyhow::bail!(
            "paged kv table/context length mismatch: tables={} context_lens={}",
            batch_size,
            context_lens.len()
        );
    }
    for (row, context_len) in context_lens.iter().enumerate() {
        let table = tables.row(row);
        let num_blocks = context_len.div_ceil(block_size);
        if num_blocks > table.len() {
            anyhow::bail!(
                "paged kv block table is too small: context_len={} block_size={} blocks={} table_len={}",
                context_len,
                block_size,
                num_blocks,
                table.len()
            );
        }
        nnz_pages = nnz_pages
            .checked_add(i32::try_from(num_blocks)?)
            .ok_or_else(|| anyhow::anyhow!("paged kv nnz pages overflow"))?;
        paged_kv_indptr.push(nnz_pages);
        for &block_idx in table.iter().take(num_blocks) {
            paged_kv_indices.push(i32::try_from(block_idx)?);
        }
        let last_page_len = if num_blocks == 0 {
            0usize
        } else {
            let consumed = (num_blocks - 1) * block_size;
            if *context_len < consumed {
                anyhow::bail!(
                    "paged kv context len underflow: context_len={} consumed={}",
                    context_len,
                    consumed
                );
            }
            *context_len - consumed
        };
        paged_kv_last_page_len.push(i32::try_from(last_page_len)?);
    }
    if paged_kv_indices.len() > padded_indices_len {
        anyhow::bail!(
            "paged kv indices exceed padded length: nnz={} padded={}",
            paged_kv_indices.len(),
            padded_indices_len
        );
    }
    paged_kv_indices.resize(padded_indices_len, 0);

    let paged_kv_indptr = Tensor::from_vec(paged_kv_indptr, (batch_size + 1,), &Device::Cpu)?;
    let paged_kv_indices = Tensor::from_vec(paged_kv_indices, (padded_indices_len,), &Device::Cpu)?;
    let paged_kv_last_page_len =
        Tensor::from_vec(paged_kv_last_page_len, (batch_size,), &Device::Cpu)?;
    Ok((paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len))
}

// Decode splits each request's KV pages into chunks and pads the tile queue for graphs.
pub(crate) fn make_paged_kv_decode_tensors<T: BlockTableRows + ?Sized>(
    tables: &T,
    context_lens: &[usize],
    block_size: usize,
    split_pages: Option<usize>,
    padded_tiles_len: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    if tables.len() != context_lens.len() {
        anyhow::bail!(
            "paged kv decode table/context length mismatch: tables={} context_lens={}",
            tables.len(),
            context_lens.len()
        );
    }
    let chunk_pages = split_pages.unwrap_or(usize::MAX).max(1);
    let mut request_indices = Vec::new();
    let mut kv_tile_indices = Vec::new();
    let mut o_indptr = Vec::with_capacity(tables.len() + 1);
    o_indptr.push(0i32);
    for (batch_idx, context_len) in context_lens.iter().enumerate() {
        let table = tables.row(batch_idx);
        let num_blocks = context_len.div_ceil(block_size);
        if num_blocks > table.len() {
            anyhow::bail!(
                "paged kv decode block table is too small: context_len={} block_size={} blocks={} table_len={}",
                context_len,
                block_size,
                num_blocks,
                table.len()
            );
        }
        let num_chunks = num_blocks.max(1).div_ceil(chunk_pages);
        for kv_tile_idx in 0..num_chunks {
            request_indices.push(i32::try_from(batch_idx)?);
            kv_tile_indices.push(i32::try_from(kv_tile_idx)?);
        }
        o_indptr.push(i32::try_from(request_indices.len())?);
    }
    if request_indices.len() > padded_tiles_len {
        anyhow::bail!(
            "paged kv decode tiles exceed padded length: tiles={} padded={}",
            request_indices.len(),
            padded_tiles_len
        );
    }
    let valid_tiles_len = request_indices.len();
    request_indices.resize(padded_tiles_len, 0);
    kv_tile_indices.resize(padded_tiles_len, 0);
    let mut block_valid_mask = vec![1u8; valid_tiles_len];
    block_valid_mask.resize(padded_tiles_len, 0);

    let request_indices = Tensor::from_vec(request_indices, (padded_tiles_len,), &Device::Cpu)?;
    let kv_tile_indices = Tensor::from_vec(kv_tile_indices, (padded_tiles_len,), &Device::Cpu)?;
    let o_indptr = Tensor::from_vec(o_indptr, (tables.len() + 1,), &Device::Cpu)?;
    let chunk_size = split_pages
        .unwrap_or(1)
        .checked_mul(block_size)
        .ok_or_else(|| anyhow::anyhow!("paged kv chunk size overflow"))?;
    let kv_chunk_size = Tensor::from_vec(vec![i32::try_from(chunk_size)?], (1,), &Device::Cpu)?;
    let block_valid_mask = Tensor::from_vec(block_valid_mask, (padded_tiles_len,), &Device::Cpu)?;
    Ok((
        request_indices,
        kv_tile_indices,
        o_indptr,
        kv_chunk_size,
        block_valid_mask,
    ))
}

pub(crate) fn make_paged_kv_decode_tensors_from_lens(
    context_lens: &[usize],
    block_size: usize,
    split_pages: Option<usize>,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let chunk_pages = split_pages.unwrap_or(usize::MAX).max(1);
    let tables = context_lens
        .iter()
        .map(|len| vec![0; len.div_ceil(block_size)])
        .collect::<Vec<_>>();
    let padded_tiles_len = context_lens
        .iter()
        .map(|len| len.div_ceil(block_size).max(1).div_ceil(chunk_pages))
        .sum::<usize>()
        .max(1);
    make_paged_kv_decode_tensors(
        &tables,
        context_lens,
        block_size,
        split_pages,
        padded_tiles_len,
    )
}

pub(crate) fn flashinfer_paged_kv(
    indptr: DeviceTensorMap,
    indices: DeviceTensorMap,
    last_page_len: DeviceTensorMap,
) -> FlashInferPagedKv {
    FlashInferPagedKv {
        indptr,
        indices,
        last_page_len,
    }
}

pub(crate) fn flashinfer_tile_plan(
    request_indices: DeviceTensorMap,
    kv_tile_indices: DeviceTensorMap,
    o_indptr: DeviceTensorMap,
    kv_chunk_size: DeviceTensorMap,
    block_valid_mask: DeviceTensorMap,
) -> FlashInferTilePlan {
    FlashInferTilePlan {
        request_indices,
        kv_tile_indices,
        o_indptr,
        kv_chunk_size,
        block_valid_mask,
    }
}

pub(crate) fn flashinfer_view(
    block_tables: Option<DeviceTensorMap>,
    context_lens: Option<DeviceTensorMap>,
    max_context_len: Option<usize>,
    paged_kv: FlashInferPagedKv,
    tile_plan: FlashInferTilePlan,
) -> FlashInferPagedAttentionView {
    FlashInferPagedAttentionView {
        block_tables,
        context_lens,
        max_context_len,
        paged_kv,
        tile_plan,
    }
}

pub(crate) fn flashinfer_metadata(
    logical: FlashInferPagedAttentionView,
    sliding: Option<FlashInferPagedAttentionView>,
) -> FlashInferMetadata {
    FlashInferMetadata {
        views: FlashInferPagedAttentionViews { logical, sliding },
        decode_tmp_v: None,
        decode_tmp_s: None,
        fa3_decode: None,
        #[cfg(feature = "cuda")]
        decode_tile_plan_used: None,
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn make_fa3_decode_state(
    metadata: &FlashInferMetadata,
    batch: usize,
    query_len: usize,
    kv_cache: &[(Tensor, Tensor)],
    model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
    activation_dtype: DType,
) -> candle_core::Result<Option<Fa3DecodeState>> {
    if !mistralrs_paged_attn::USE_FA3_FP8_PAGED
        || activation_dtype != DType::BF16
        || batch == 0
        || query_len == 0
    {
        return Ok(None);
    }
    let Some(model_metadata) = model_metadata else {
        return Ok(None);
    };

    let mut state = Fa3DecodeState::default();
    let layer_count = model_metadata.num_layers().min(kv_cache.len());
    for (layer_idx, (key_cache, value_cache)) in kv_cache.iter().enumerate().take(layer_count) {
        if !model_metadata.layer_has_paged_kv_cache(layer_idx)
            || model_metadata.attention_backend_kind_for_layer(layer_idx)
                != AttentionBackendKind::FlashInfer
            || key_cache.dtype() != DType::F8E4M3
            || value_cache.dtype() != DType::F8E4M3
        {
            continue;
        }
        let Some(num_sm) = fa3_device_num_sm(key_cache.device()) else {
            continue;
        };
        let (num_pages, kv_heads, page_size, head_dim) = key_cache.dims4()?;
        if num_pages == 0
            || value_cache.dims4()? != (num_pages, kv_heads, page_size, head_dim)
            || value_cache.device().location() != key_cache.device().location()
        {
            continue;
        }
        let q_heads = model_metadata.num_attn_heads_for_layer(layer_idx);
        let Some(key) = (Fa3PagedScheduleShape {
            device: key_cache.device().location(),
            view: Fa3DecodeView::Logical,
            batch,
            query_len,
            causal: query_len > 1,
            q_heads,
            kv_heads,
            head_dim,
            page_size,
        })
        .decode_schedule_key() else {
            continue;
        };
        if state.get(&key).is_some() {
            continue;
        }
        let Some(max_pages_per_sequence) = fa3_view_capacity(&metadata.views.logical, &key)? else {
            continue;
        };
        state.insert(
            key,
            allocate_fa3_decode_buffers(key_cache.device(), key, max_pages_per_sequence, num_sm)?,
        );
    }

    Ok((!state.is_empty()).then_some(state))
}

#[cfg(all(feature = "cuda", not(target_family = "unix")))]
pub(crate) fn make_fa3_decode_state(
    _metadata: &FlashInferMetadata,
    _batch: usize,
    _query_len: usize,
    _kv_cache: &[(Tensor, Tensor)],
    _model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
    _activation_dtype: DType,
) -> candle_core::Result<Option<Fa3DecodeState>> {
    Ok(None)
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn fa3_device_num_sm(device: &Device) -> Option<usize> {
    use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;

    let Device::Cuda(device) = device else {
        return None;
    };
    let stream = device.cuda_stream();
    let context = stream.context();
    let compute_major = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .ok()?;
    if compute_major != 9 {
        return None;
    }
    context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .ok()
        .and_then(|count| usize::try_from(count).ok())
        .filter(|count| *count > 0)
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn fa3_prefill_cache_num_sm(
    key_cache: &Tensor,
    value_cache: &Tensor,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    page_size: usize,
) -> candle_core::Result<Option<usize>> {
    let Ok((num_pages, cache_kv_heads, cache_page_size, cache_head_dim)) = key_cache.dims4() else {
        return Ok(None);
    };
    if !mistralrs_paged_attn::USE_FA3_FP8_PAGED
        || num_pages == 0
        || q_heads == 0
        || kv_heads == 0
        || !q_heads.is_multiple_of(kv_heads)
        || !matches!(q_heads / kv_heads, 1 | 2 | 3 | 4 | 6 | 8 | 16)
        || head_dim != 256
        || page_size == 0
        || !page_size.is_multiple_of(32)
        || (cache_kv_heads, cache_page_size, cache_head_dim) != (kv_heads, page_size, head_dim)
        || value_cache.dims4().ok()
            != Some((num_pages, cache_kv_heads, cache_page_size, cache_head_dim))
        || key_cache.dtype() != DType::F8E4M3
        || value_cache.dtype() != DType::F8E4M3
        || !key_cache.is_contiguous()
        || !value_cache.is_contiguous()
        || !key_cache.device().same_device(value_cache.device())
        || fa3_prefill_pool(key_cache)?.is_none()
    {
        return Ok(None);
    }
    Ok(fa3_device_num_sm(key_cache.device()))
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn fa3_view_capacity(
    view: &FlashInferPagedAttentionView,
    key: &Fa3DecodeScheduleKey,
) -> candle_core::Result<Option<usize>> {
    let Some(source_rows) = key.total_q() else {
        return Ok(None);
    };
    fa3_view_capacity_for_rows(view, key.device, source_rows)
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn fa3_view_capacity_for_rows(
    view: &FlashInferPagedAttentionView,
    device: candle_core::DeviceLocation,
    source_rows: usize,
) -> candle_core::Result<Option<usize>> {
    let Some(indptr) = view.paged_kv.indptr.get(&device) else {
        return Ok(None);
    };
    let Some(indices) = view.paged_kv.indices.get(&device) else {
        return Ok(None);
    };
    let Some(last_page_len) = view.paged_kv.last_page_len.get(&device) else {
        return Ok(None);
    };
    if indptr.dtype() != DType::I32
        || indices.dtype() != DType::I32
        || last_page_len.dtype() != DType::I32
        || indptr.elem_count() != source_rows + 1
        || last_page_len.elem_count() != source_rows
        || indices.elem_count() < source_rows
        || !indices.elem_count().is_multiple_of(source_rows)
        || indptr.device().location() != device
        || indices.device().location() != device
        || last_page_len.device().location() != device
    {
        return Ok(None);
    }
    Ok(Some(indices.elem_count() / source_rows))
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Default)]
struct Fa3PrefillWorkspace {
    query: Option<Tensor>,
    scheduler_metadata: Option<Tensor>,
    output_accum: Option<Tensor>,
    lse_accum: Option<Tensor>,
    output_lse: Option<Tensor>,
    page_table: Option<Tensor>,
    seqused_k: Option<Tensor>,
    cu_seqlens_q: Option<Tensor>,
    cu_seqlens_shape: Option<(usize, usize)>,
    owner_stream: Option<Arc<CudaStream>>,
    completion: Option<CudaEvent>,
    completion_pending: bool,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Fa3PrefillLaneKey {
    device: DeviceLocation,
    context: usize,
    stream: usize,
    thread: Option<std::thread::ThreadId>,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
#[derive(Default)]
struct Fa3PrefillWorkspacePool {
    lanes: Mutex<HashMap<Fa3PrefillLaneKey, Arc<Mutex<Fa3PrefillWorkspace>>>>,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn fa3_prefill_registry() -> &'static Mutex<HashMap<TensorId, Weak<Fa3PrefillWorkspacePool>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<TensorId, Weak<Fa3PrefillWorkspacePool>>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) struct Fa3PrefillWorkspaceRegistration {
    pool: Arc<Fa3PrefillWorkspacePool>,
    cache_ids: Vec<TensorId>,
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn register_fa3_prefill_caches(
    caches: &[(Tensor, Tensor)],
) -> candle_core::Result<Fa3PrefillWorkspaceRegistration> {
    let pool = Arc::new(Fa3PrefillWorkspacePool::default());
    let cache_ids = caches
        .iter()
        .map(|(key_cache, _)| key_cache)
        .filter(|key_cache| key_cache.device().is_cuda() && key_cache.elem_count() > 0)
        .map(Tensor::id)
        .collect::<Vec<_>>();
    let mut registry = fa3_prefill_registry()
        .lock()
        .map_err(|_| candle_core::Error::msg("FA3 prefill registry mutex was poisoned"))?;
    for &cache_id in &cache_ids {
        registry.insert(cache_id, Arc::downgrade(&pool));
    }
    drop(registry);
    Ok(Fa3PrefillWorkspaceRegistration { pool, cache_ids })
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl Drop for Fa3PrefillWorkspaceRegistration {
    fn drop(&mut self) {
        let Ok(mut registry) = fa3_prefill_registry().lock() else {
            return;
        };
        for cache_id in &self.cache_ids {
            let remove = registry
                .get(cache_id)
                .and_then(Weak::upgrade)
                .is_none_or(|registered| Arc::ptr_eq(&registered, &self.pool));
            if remove {
                registry.remove(cache_id);
            }
        }
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn fa3_prefill_pool(
    key_cache: &Tensor,
) -> candle_core::Result<Option<Arc<Fa3PrefillWorkspacePool>>> {
    let cache_id = key_cache.id();
    let mut registry = fa3_prefill_registry()
        .lock()
        .map_err(|_| candle_core::Error::msg("FA3 prefill registry mutex was poisoned"))?;
    let pool = registry.get(&cache_id).and_then(Weak::upgrade);
    if pool.is_none() {
        registry.remove(&cache_id);
    }
    Ok(pool)
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn checked_fa3_len(parts: &[usize], name: &str) -> candle_core::Result<usize> {
    parts.iter().try_fold(1usize, |len, part| {
        len.checked_mul(*part)
            .ok_or_else(|| candle_core::Error::msg(format!("FA3 {name} size overflow")))
    })
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn ensure_fa3_flat_buffer(
    tensor: &mut Option<Tensor>,
    len: usize,
    dtype: DType,
    device: &Device,
) -> candle_core::Result<()> {
    if tensor
        .as_ref()
        .is_some_and(|tensor| tensor.elem_count() >= len)
    {
        return Ok(());
    }
    *tensor = Some(unsafe { Tensor::empty((len,), dtype, device)? });
    Ok(())
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn fa3_flat_view(tensor: &Tensor, len: usize) -> candle_core::Result<Tensor> {
    tensor.narrow(0, 0, len)
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl Fa3PrefillWorkspace {
    fn ensure_completion_event(&mut self, stream: &Arc<CudaStream>) -> candle_core::Result<()> {
        if let Some(owner_stream) = &self.owner_stream {
            if owner_stream.cu_stream() != stream.cu_stream()
                || owner_stream.context().cu_ctx() != stream.context().cu_ctx()
            {
                candle_core::bail!("FA3 prefill workspace cannot change CUDA streams");
            }
        } else {
            self.owner_stream = Some(stream.clone());
        }
        if self.completion.is_none() {
            self.completion = Some(
                stream
                    .context()
                    .new_event(None)
                    .map_err(candle_core::Error::wrap)?,
            );
        }
        Ok(())
    }

    fn synchronize_completion(&mut self) -> candle_core::Result<()> {
        if self.completion_pending {
            self.completion
                .as_ref()
                .expect("pending FA3 workspace must have a completion event")
                .synchronize()
                .map_err(candle_core::Error::wrap)?;
            self.completion_pending = false;
        }
        Ok(())
    }

    fn record_completion(&mut self, stream: &Arc<CudaStream>) -> candle_core::Result<()> {
        let event = self
            .completion
            .as_ref()
            .expect("FA3 workspace completion event must be initialized");
        if let Err(err) = event.record(stream) {
            stream.synchronize().map_err(candle_core::Error::wrap)?;
            self.completion_pending = false;
            return Err(candle_core::Error::wrap(err));
        }
        self.completion_pending = true;
        Ok(())
    }

    fn buffers(
        &mut self,
        device: &Device,
        key: Fa3DecodeScheduleKey,
        max_pages_per_sequence: usize,
        num_sm: usize,
    ) -> candle_core::Result<Fa3DecodeBuffers> {
        let total_q = key
            .total_q()
            .ok_or_else(|| candle_core::Error::msg("FA3 query count overflow"))?;
        let query_len = checked_fa3_len(&[total_q, key.q_heads, key.head_dim], "query")?;
        let scheduler_len = fa3_scheduler_metadata_len(key.batch, key.causal);
        let output_accum_len = checked_fa3_len(
            &[key.num_splits, key.q_heads, total_q, key.head_dim],
            "output accumulator",
        )?;
        let lse_accum_len =
            checked_fa3_len(&[key.num_splits, key.q_heads, total_q], "LSE accumulator")?;
        let output_lse_len = checked_fa3_len(&[key.q_heads, total_q], "output LSE")?;
        let page_table_len = checked_fa3_len(&[key.batch, max_pages_per_sequence], "page table")?;
        let needs_rebuild = [
            (&self.query, query_len),
            (&self.scheduler_metadata, scheduler_len),
            (&self.output_accum, output_accum_len),
            (&self.lse_accum, lse_accum_len),
            (&self.output_lse, output_lse_len),
            (&self.page_table, page_table_len),
            (&self.seqused_k, key.batch),
        ]
        .into_iter()
        .any(|(tensor, len)| {
            tensor
                .as_ref()
                .is_none_or(|tensor| tensor.elem_count() < len)
        }) || self.cu_seqlens_shape != Some((key.batch, key.query_len));
        if needs_rebuild {
            self.synchronize_completion()?;
        }
        ensure_fa3_flat_buffer(&mut self.query, query_len, DType::F8E4M3, device)?;
        ensure_fa3_flat_buffer(
            &mut self.scheduler_metadata,
            scheduler_len,
            DType::I32,
            device,
        )?;
        ensure_fa3_flat_buffer(&mut self.output_accum, output_accum_len, DType::F32, device)?;
        ensure_fa3_flat_buffer(&mut self.lse_accum, lse_accum_len, DType::F32, device)?;
        ensure_fa3_flat_buffer(&mut self.output_lse, output_lse_len, DType::F32, device)?;
        ensure_fa3_flat_buffer(&mut self.page_table, page_table_len, DType::I32, device)?;
        ensure_fa3_flat_buffer(&mut self.seqused_k, key.batch, DType::I32, device)?;
        if self.cu_seqlens_shape != Some((key.batch, key.query_len)) {
            self.cu_seqlens_q = Some(Tensor::from_vec(
                (0..=key.batch)
                    .map(|row| {
                        row.checked_mul(key.query_len)
                            .and_then(|offset| i32::try_from(offset).ok())
                            .ok_or_else(|| candle_core::Error::msg("FA3 query offset overflow"))
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?,
                (key.batch + 1,),
                device,
            )?);
            self.cu_seqlens_shape = Some((key.batch, key.query_len));
        }

        Ok(Fa3DecodeBuffers {
            query: fa3_flat_view(self.query.as_ref().unwrap(), query_len)?.reshape((
                total_q,
                key.q_heads,
                key.head_dim,
            ))?,
            scheduler_metadata: fa3_flat_view(
                self.scheduler_metadata.as_ref().unwrap(),
                scheduler_len,
            )?,
            output_accum: fa3_flat_view(self.output_accum.as_ref().unwrap(), output_accum_len)?
                .reshape((key.num_splits, key.q_heads, total_q, key.head_dim))?,
            lse_accum: fa3_flat_view(self.lse_accum.as_ref().unwrap(), lse_accum_len)?
                .reshape((key.num_splits, key.q_heads, total_q))?,
            output_lse: fa3_flat_view(self.output_lse.as_ref().unwrap(), output_lse_len)?
                .reshape((key.q_heads, total_q))?,
            cu_seqlens_q: self.cu_seqlens_q.as_ref().unwrap().clone(),
            page_table: fa3_flat_view(self.page_table.as_ref().unwrap(), page_table_len)?
                .reshape((key.batch, max_pages_per_sequence))?,
            seqused_k: fa3_flat_view(self.seqused_k.as_ref().unwrap(), key.batch)?,
            max_pages_per_sequence,
            num_sm,
        })
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
impl Drop for Fa3PrefillWorkspace {
    fn drop(&mut self) {
        if self.completion_pending {
            if let Some(completion) = &self.completion {
                let _ = completion.synchronize();
            }
        }
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn fa3_prefill_lane_key(
    execution_device: &Device,
) -> candle_core::Result<(Fa3PrefillLaneKey, Arc<CudaStream>)> {
    let Device::Cuda(cuda) = execution_device else {
        candle_core::bail!("FA3 prefill execution device must be CUDA");
    };
    let stream = cuda.cuda_stream();
    let stream_handle = stream.cu_stream() as usize;
    Ok((
        Fa3PrefillLaneKey {
            device: execution_device.location(),
            context: stream.context().cu_ctx() as usize,
            stream: stream_handle,
            thread: fa3_prefill_lane_thread(stream_handle),
        },
        stream,
    ))
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn fa3_prefill_lane_thread(stream: usize) -> Option<std::thread::ThreadId> {
    (stream == CUDA_STREAM_PER_THREAD_HANDLE).then(|| std::thread::current().id())
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn with_fa3_prefill_workspace<R>(
    metadata: &FlashInferMetadata,
    key: Fa3DecodeScheduleKey,
    key_cache: &Tensor,
    execution_device: &Device,
    run: impl FnOnce(&Fa3DecodeBuffers) -> candle_core::Result<R>,
) -> candle_core::Result<R> {
    if !key.supported()
        || key.device != key_cache.device().location()
        || key.device != execution_device.location()
    {
        candle_core::bail!("FA3 prefill workspace request does not match its selected schedule");
    }
    let pool = fa3_prefill_pool(key_cache)?
        .ok_or_else(|| candle_core::Error::msg("FA3 prefill workspace is not registered"))?;
    let max_pages_per_sequence =
        fa3_view_capacity_for_rows(&metadata.views.logical, key.device, key.batch)?
            .ok_or_else(|| candle_core::Error::msg("FA3 prefill paged metadata is unavailable"))?;
    let num_sm = fa3_device_num_sm(execution_device).ok_or_else(|| {
        candle_core::Error::msg("FA3 prefill CUDA device metadata is unavailable")
    })?;
    let (lane_key, execution_stream) = fa3_prefill_lane_key(execution_device)?;
    let workspace = {
        let mut lanes = pool
            .lanes
            .lock()
            .map_err(|_| candle_core::Error::msg("FA3 prefill lane map mutex was poisoned"))?;
        lanes
            .entry(lane_key)
            .or_insert_with(|| Arc::new(Mutex::new(Fa3PrefillWorkspace::default())))
            .clone()
    };
    let view = &metadata.views.logical;
    let indptr = view
        .paged_kv
        .indptr
        .get(&key.device)
        .ok_or_else(|| candle_core::Error::msg("FA3 prefill indptr missing"))?;
    let indices = view
        .paged_kv
        .indices
        .get(&key.device)
        .ok_or_else(|| candle_core::Error::msg("FA3 prefill indices missing"))?;
    let last_page_len = view
        .paged_kv
        .last_page_len
        .get(&key.device)
        .ok_or_else(|| candle_core::Error::msg("FA3 prefill last-page lengths missing"))?;
    let mut workspace = workspace
        .lock()
        .map_err(|_| candle_core::Error::msg("FA3 prefill workspace mutex was poisoned"))?;
    workspace.ensure_completion_event(&execution_stream)?;
    let buffers = workspace.buffers(execution_device, key, max_pages_per_sequence, num_sm)?;
    let result = (|| {
        mistralrs_paged_attn::fa3_prepare_paged_metadata(
            mistralrs_paged_attn::Fa3DecodeMetadata {
                paged_kv_indptr: indptr,
                paged_kv_indices: indices,
                paged_kv_last_page_len: last_page_len,
                page_table: &buffers.page_table,
                seqused_k: &buffers.seqused_k,
                cu_seqlens_q: &buffers.cu_seqlens_q,
                scheduler_metadata: &buffers.scheduler_metadata,
            },
            buffers.schedule(key)?,
            mistralrs_paged_attn::Fa3PagedMetadataLayout::per_sequence(),
        )?;
        run(&buffers)
    })();
    let completion = workspace.record_completion(&execution_stream);
    match (result, completion) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err),
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn allocate_fa3_decode_buffers(
    device: &Device,
    key: Fa3DecodeScheduleKey,
    max_pages_per_sequence: usize,
    num_sm: usize,
) -> candle_core::Result<Fa3DecodeBuffers> {
    let total_q = key
        .total_q()
        .ok_or_else(|| candle_core::Error::msg("FA3 query count overflow"))?;
    let scheduler_len = fa3_scheduler_metadata_len(key.batch, key.causal);
    let cu_seqlens_q = Tensor::from_vec(
        (0..=key.batch)
            .map(|row| row.saturating_mul(key.query_len))
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?,
        (key.batch + 1,),
        device,
    )?;
    Ok(Fa3DecodeBuffers {
        query: unsafe {
            Tensor::empty((total_q, key.q_heads, key.head_dim), DType::F8E4M3, device)?
        },
        scheduler_metadata: unsafe { Tensor::empty((scheduler_len,), DType::I32, device)? },
        output_accum: unsafe {
            Tensor::empty(
                (key.num_splits, key.q_heads, total_q, key.head_dim),
                DType::F32,
                device,
            )?
        },
        lse_accum: unsafe {
            Tensor::empty((key.num_splits, key.q_heads, total_q), DType::F32, device)?
        },
        output_lse: unsafe { Tensor::empty((key.q_heads, total_q), DType::F32, device)? },
        cu_seqlens_q,
        page_table: unsafe {
            Tensor::empty((key.batch, max_pages_per_sequence), DType::I32, device)?
        },
        seqused_k: unsafe { Tensor::empty((key.batch,), DType::I32, device)? },
        max_pages_per_sequence,
        num_sm,
    })
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn fa3_scheduler_metadata_len(batch: usize, causal: bool) -> usize {
    (2 + usize::from(causal)) * batch.div_ceil(4) * 4 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_split_fills_small_grids_and_leaves_full_ones_alone() {
        // Qwen3.5-class: 4 kv heads, batch 1 on a 48-SM GPU wants 24 chunks per (seq, head)
        assert_eq!(decode_split_tokens(1, 4, 48, 8192), 256);
        assert_eq!(decode_split_tokens(1, 4, 48, 16384), 512);
        assert_eq!(decode_split_tokens(1, 4, 48, 65536), 2048);
        assert_eq!(decode_split_tokens(1, 4, 48, 100), 256);
        assert_eq!(decode_split_tokens(1, 8, 48, 8192), 512);
        // Already 64 blocks without splitting: keep one big chunk per (seq, head)
        assert_eq!(decode_split_tokens(8, 8, 48, 8192), 2048);
        assert_eq!(decode_split_tokens(64, 8, 48, 8192), 2048);
        assert_eq!(decode_split_capacity_pages(32), 8);
        assert!(decode_split_pages(32, 1, 4, 8192) >= decode_split_capacity_pages(32));
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_scheduler_metadata_covers_rounded_batch_rows_and_semaphore() {
        assert_eq!(fa3_scheduler_metadata_len(1, false), 9);
        assert_eq!(fa3_scheduler_metadata_len(4, false), 9);
        assert_eq!(fa3_scheduler_metadata_len(8, false), 17);
        assert_eq!(fa3_scheduler_metadata_len(16, false), 33);
        assert_eq!(fa3_scheduler_metadata_len(1, true), 13);
        assert_eq!(fa3_scheduler_metadata_len(8, true), 25);
        assert_eq!(fa3_scheduler_metadata_len(16, true), 49);
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_prefill_workspace_lengths_are_checked() {
        assert_eq!(
            checked_fa3_len(&[16, 24, 128, 256], "test").unwrap(),
            12_582_912
        );
        assert!(checked_fa3_len(&[usize::MAX, 2], "test").is_err());
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_prefill_workspace_registration_controls_lifetime() -> candle_core::Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let key_cache = Tensor::zeros((1,), DType::F32, &device)?;
        let value_cache = Tensor::zeros((1,), DType::F32, &device)?;
        let registration = register_fa3_prefill_caches(&[(key_cache.clone(), value_cache)])?;
        assert!(fa3_prefill_pool(&key_cache)?.is_some());
        drop(registration);
        assert!(fa3_prefill_pool(&key_cache)?.is_none());
        Ok(())
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_prefill_workspace_lanes_follow_execution_streams() -> candle_core::Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (first, stream) = fa3_prefill_lane_key(&device)?;
        let fork = stream.fork().map_err(candle_core::Error::wrap)?;
        let second = Fa3PrefillLaneKey {
            device: device.location(),
            context: fork.context().cu_ctx() as usize,
            stream: fork.cu_stream() as usize,
            thread: fa3_prefill_lane_thread(fork.cu_stream() as usize),
        };
        assert_eq!(first.context, second.context);
        assert_ne!(first.stream, second.stream);
        assert_ne!(first, second);
        Ok(())
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_prefill_workspace_keys_per_thread_default_streams() {
        let current = fa3_prefill_lane_thread(CUDA_STREAM_PER_THREAD_HANDLE).unwrap();
        let other =
            std::thread::spawn(|| fa3_prefill_lane_thread(CUDA_STREAM_PER_THREAD_HANDLE).unwrap())
                .join()
                .unwrap();
        assert_ne!(current, other);
        assert_eq!(fa3_prefill_lane_thread(0x1000), None);
    }
}
