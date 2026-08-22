use anyhow::Result;
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::{Device, Tensor};

#[cfg(feature = "cuda")]
use super::Fa3DecodeState;
use super::{
    DeviceTensorMap, FlashInferMetadata, FlashInferPagedAttentionView,
    FlashInferPagedAttentionViews, FlashInferPagedKv, FlashInferTilePlan,
};
#[cfg(all(feature = "cuda", target_family = "unix"))]
use super::{Fa3DecodeBuffers, Fa3DecodeScheduleKey, Fa3DecodeView, FA3_DECODE_NUM_SPLITS};
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
pub(crate) fn make_paged_kv_tensors(
    tables: &[Vec<usize>],
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
    for (table, context_len) in tables.iter().zip(context_lens.iter()) {
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
pub(crate) fn make_paged_kv_decode_tensors(
    tables: &[Vec<usize>],
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
    for (batch_idx, (table, context_len)) in tables.iter().zip(context_lens.iter()).enumerate() {
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
    kv_cache: &[(Tensor, Tensor)],
    model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
    activation_dtype: DType,
) -> candle_core::Result<Option<Fa3DecodeState>> {
    if !mistralrs_paged_attn::USE_FA3_FP8_PAGED || activation_dtype != DType::BF16 || batch == 0 {
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
        let key = Fa3DecodeScheduleKey {
            device: key_cache.device().location(),
            view: Fa3DecodeView::Logical,
            batch,
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            num_splits: FA3_DECODE_NUM_SPLITS,
        };
        if !key.supported() || state.get(&key).is_some() {
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
    _kv_cache: &[(Tensor, Tensor)],
    _model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
    _activation_dtype: DType,
) -> candle_core::Result<Option<Fa3DecodeState>> {
    Ok(None)
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn fa3_device_num_sm(device: &Device) -> Option<usize> {
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
fn fa3_view_capacity(
    view: &FlashInferPagedAttentionView,
    key: &Fa3DecodeScheduleKey,
) -> candle_core::Result<Option<usize>> {
    let Some(indptr) = view.paged_kv.indptr.get(&key.device) else {
        return Ok(None);
    };
    let Some(indices) = view.paged_kv.indices.get(&key.device) else {
        return Ok(None);
    };
    let Some(last_page_len) = view.paged_kv.last_page_len.get(&key.device) else {
        return Ok(None);
    };
    if indptr.dtype() != DType::I32
        || indices.dtype() != DType::I32
        || last_page_len.dtype() != DType::I32
        || indptr.elem_count() != key.batch + 1
        || last_page_len.elem_count() != key.batch
        || indices.elem_count() < key.batch
        || !indices.elem_count().is_multiple_of(key.batch)
        || indptr.device().location() != key.device
        || indices.device().location() != key.device
        || last_page_len.device().location() != key.device
    {
        return Ok(None);
    }
    Ok(Some(indices.elem_count() / key.batch))
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn allocate_fa3_decode_buffers(
    device: &Device,
    key: Fa3DecodeScheduleKey,
    max_pages_per_sequence: usize,
    num_sm: usize,
) -> candle_core::Result<Fa3DecodeBuffers> {
    let scheduler_len = fa3_scheduler_metadata_len(key.batch);
    let cu_seqlens_q = Tensor::from_vec(
        (0..=key.batch)
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?,
        (key.batch + 1,),
        device,
    )?;
    Ok(Fa3DecodeBuffers {
        query: unsafe {
            Tensor::empty(
                (key.batch, key.q_heads, key.head_dim),
                DType::F8E4M3,
                device,
            )?
        },
        scheduler_metadata: unsafe { Tensor::empty((scheduler_len,), DType::I32, device)? },
        output_accum: unsafe {
            Tensor::empty(
                (key.num_splits, key.q_heads, key.batch, key.head_dim),
                DType::F32,
                device,
            )?
        },
        lse_accum: unsafe {
            Tensor::empty((key.num_splits, key.q_heads, key.batch), DType::F32, device)?
        },
        output_lse: unsafe { Tensor::empty((key.q_heads, key.batch), DType::F32, device)? },
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
fn fa3_scheduler_metadata_len(batch: usize) -> usize {
    2 * batch.div_ceil(4) * 4 + 1
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
        assert_eq!(fa3_scheduler_metadata_len(1), 9);
        assert_eq!(fa3_scheduler_metadata_len(4), 9);
        assert_eq!(fa3_scheduler_metadata_len(8), 17);
        assert_eq!(fa3_scheduler_metadata_len(16), 33);
    }
}
