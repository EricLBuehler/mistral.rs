use anyhow::Result;
use candle_core::{Device, Tensor};

use super::{
    DeviceTensorMap, FlashInferMetadata, FlashInferPagedAttentionView,
    FlashInferPagedAttentionViews, FlashInferPagedKv, FlashInferTilePlan,
};

// Split-KV decode chunk target, not a context cap.
const DECODE_FIXED_SPLIT_TOKENS: usize = 2048;
pub(crate) fn decode_split_pages(block_size: usize) -> usize {
    DECODE_FIXED_SPLIT_TOKENS.div_ceil(block_size).max(1)
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
    }
}
