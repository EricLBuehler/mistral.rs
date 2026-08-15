//! Paged-attention metadata for proposer forwards where every row is an independent single-token
//! query with its own context length (one row per sequence for decode-style drafting, or several rows
//! per sequence when refreshing a drafter over accepted tokens).

use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};

use crate::{
    flashinfer::{
        FlashInferMetadata, FlashInferPagedAttentionView, FlashInferPagedAttentionViews,
        FlashInferPagedKv, FlashInferTilePlan,
    },
    get_mut_arcmutex,
    pipeline::text_models_inputs_processor::{PagedAttentionInputMetadata, PagedAttentionMeta},
};

/// `seq_ids[i]` may repeat; `context_lens[i]` is the row's position plus one, and the row's K/V
/// are written to that position's slot.
pub(crate) fn make_paged_rows_metadata(
    seq_ids: &[usize],
    context_lens: &[usize],
    paged_meta: &PagedAttentionMeta,
    device: &Device,
) -> Result<PagedAttentionInputMetadata> {
    if seq_ids.len() != context_lens.len() {
        candle_core::bail!(
            "paged rows metadata batch mismatch: seq_ids={}, context_lens={}",
            seq_ids.len(),
            context_lens.len()
        );
    }

    let kv_mgr = get_mut_arcmutex!(paged_meta.kv_cache_manager);
    let full_tables = seq_ids
        .iter()
        .map(|seq_id| {
            kv_mgr
                .get_block_ids(*seq_id)
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "sequence {seq_id} has no paged attention blocks"
                    ))
                })
                .map(|ids| ids.to_vec())
        })
        .collect::<Result<Vec<_>>>()?;
    drop(kv_mgr);

    let mut block_tables = Vec::with_capacity(seq_ids.len());
    let mut context_lens_windowed = Vec::with_capacity(seq_ids.len());
    let mut slot_mappings = Vec::with_capacity(seq_ids.len());
    for (full_table, context_len) in full_tables.iter().zip(context_lens.iter().copied()) {
        let (block_table, context_len_windowed) =
            if let Some(sliding_window) = paged_meta.sliding_window {
                // Keep paged MTP aligned with the normal-cache inclusive SWA mask.
                let window_start = context_len.saturating_sub(sliding_window + 1);
                let slide_idx = window_start / paged_meta.block_size;
                let block_aligned_start = slide_idx * paged_meta.block_size;
                (
                    full_table.get(slide_idx..).unwrap_or(&[]).to_vec(),
                    context_len.saturating_sub(block_aligned_start),
                )
            } else {
                (full_table.clone(), context_len)
            };
        block_tables.push(block_table);
        context_lens_windowed.push(context_len_windowed);

        let block_pos = context_len.saturating_sub(1);
        let slot = full_table
            .get(block_pos / paged_meta.block_size)
            .copied()
            .unwrap_or(0)
            * paged_meta.block_size
            + block_pos % paged_meta.block_size;
        slot_mappings.push(slot as i64);
    }

    let batch = seq_ids.len();
    let slot_mappings = Tensor::from_vec(slot_mappings, (batch,), device)?;

    let windowed_block_tables = block_tables;
    let block_tables = table_tensor(&windowed_block_tables, device)?;
    let full_block_tables = table_tensor(&full_tables, device)?;
    let context_lens_tensor = Tensor::from_vec(
        context_lens_windowed
            .iter()
            .map(|len| usize_to_u32(*len, "windowed context length"))
            .collect::<Result<Vec<_>>>()?,
        (batch,),
        device,
    )?;
    let full_context_lens_tensor = Tensor::from_vec(
        context_lens
            .iter()
            .map(|len| usize_to_u32(*len, "full context length"))
            .collect::<Result<Vec<_>>>()?,
        (batch,),
        device,
    )?;

    let (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len) = paged_kv_tensors(
        &windowed_block_tables,
        &context_lens_windowed,
        paged_meta.block_size,
        device,
    )?;
    let (full_paged_kv_indptr, full_paged_kv_indices, full_paged_kv_last_page_len) =
        paged_kv_tensors(&full_tables, context_lens, paged_meta.block_size, device)?;
    let batch_i32 = usize_to_i32(batch, "paged rows batch size")?;
    let request_indices = Tensor::from_vec((0..batch_i32).collect::<Vec<_>>(), (batch,), device)?;
    let kv_tile_indices = Tensor::from_vec(vec![0i32; batch], (batch,), device)?;
    let o_indptr = Tensor::from_vec((0..=batch_i32).collect::<Vec<_>>(), (batch + 1,), device)?;
    let kv_chunk_size = Tensor::from_vec(
        vec![usize_to_i32(paged_meta.block_size, "paged block size")?],
        (1,),
        device,
    )?;
    let block_valid_mask = Tensor::from_vec(vec![1u8; batch], (batch,), device)?;

    let location = device.location();
    let block_tables_map = HashMap::from([(location, block_tables)]);
    let context_lens_map = HashMap::from([(location, context_lens_tensor)]);
    let full_block_tables_map = HashMap::from([(location, full_block_tables)]);
    let full_context_lens_map = HashMap::from([(location, full_context_lens_tensor)]);
    let paged_kv = FlashInferPagedKv {
        indptr: HashMap::from([(location, paged_kv_indptr)]),
        indices: HashMap::from([(location, paged_kv_indices)]),
        last_page_len: HashMap::from([(location, paged_kv_last_page_len)]),
    };
    let full_paged_kv = FlashInferPagedKv {
        indptr: HashMap::from([(location, full_paged_kv_indptr)]),
        indices: HashMap::from([(location, full_paged_kv_indices)]),
        last_page_len: HashMap::from([(location, full_paged_kv_last_page_len)]),
    };
    let tile_plan = FlashInferTilePlan {
        request_indices: HashMap::from([(location, request_indices.clone())]),
        kv_tile_indices: HashMap::from([(location, kv_tile_indices.clone())]),
        o_indptr: HashMap::from([(location, o_indptr.clone())]),
        kv_chunk_size: HashMap::from([(location, kv_chunk_size.clone())]),
        block_valid_mask: HashMap::from([(location, block_valid_mask.clone())]),
    };
    let full_tile_plan = FlashInferTilePlan {
        request_indices: HashMap::from([(location, request_indices)]),
        kv_tile_indices: HashMap::from([(location, kv_tile_indices)]),
        o_indptr: HashMap::from([(location, o_indptr)]),
        kv_chunk_size: HashMap::from([(location, kv_chunk_size)]),
        block_valid_mask: HashMap::from([(location, block_valid_mask)]),
    };
    let flashinfer = Some(FlashInferMetadata {
        views: FlashInferPagedAttentionViews {
            logical: FlashInferPagedAttentionView {
                block_tables: Some(full_block_tables_map.clone()),
                context_lens: Some(full_context_lens_map.clone()),
                max_context_len: Some(context_lens.iter().copied().max().unwrap_or(0)),
                paged_kv: full_paged_kv,
                tile_plan: full_tile_plan,
            },
            sliding: Some(FlashInferPagedAttentionView {
                block_tables: Some(block_tables_map.clone()),
                context_lens: Some(context_lens_map.clone()),
                max_context_len: Some(context_lens_windowed.iter().copied().max().unwrap_or(0)),
                paged_kv,
                tile_plan,
            }),
        },
        decode_tmp_v: None,
        decode_tmp_s: None,
    });

    Ok(PagedAttentionInputMetadata {
        block_tables: Some(block_tables_map),
        context_lens: Some(context_lens_map),
        block_size: Some(paged_meta.block_size),
        paged_context_lens_cpu: Some(context_lens_windowed.to_vec()),
        full_paged_context_lens_cpu: Some(context_lens.to_vec()),
        slot_mappings: HashMap::from([(location, slot_mappings)]),
        max_context_len: Some(context_lens_windowed.iter().copied().max().unwrap_or(0)),
        full_block_tables: Some(full_block_tables_map),
        full_context_lens: Some(full_context_lens_map),
        full_max_context_len: Some(context_lens.iter().copied().max().unwrap_or(0)),
        is_first_prompt_chunk: false,
        is_final_prompt_chunk: true,
        prompt_chunk_attention_policy:
            crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
        has_noncausal_mm_context: false,
        mm_prefix_ranges: None,
        full_mm_prefix_ranges: None,
        prefill_attention_heads: 1,
        prefill_key_value_heads: 1,
        prefill_head_dim: 1,
        flashinfer,
        rope_positions: None,
        num_cached_tokens: None,
        query_lens: None,
        cu_seqlens_q: None,
        cu_seqlens_kv: None,
    })
}

fn table_tensor(rows: &[Vec<usize>], device: &Device) -> Result<Tensor> {
    let max_len = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let mut values = Vec::with_capacity(rows.len() * max_len);
    for row in rows {
        for value in row {
            values.push(usize_to_u32(*value, "block table entry")?);
        }
        values.extend(std::iter::repeat_n(0u32, max_len.saturating_sub(row.len())));
    }
    Tensor::from_vec(values, (rows.len(), max_len), device)
}

fn paged_kv_tensors(
    tables: &[Vec<usize>],
    context_lens: &[usize],
    block_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut indptr = Vec::with_capacity(tables.len() + 1);
    let mut indices = Vec::new();
    let mut last_page_len = Vec::with_capacity(tables.len());
    indptr.push(0i32);
    let mut nnz = 0i32;
    for (table, context_len) in tables.iter().zip(context_lens.iter().copied()) {
        nnz = nnz
            .checked_add(usize_to_i32(table.len(), "paged table length")?)
            .ok_or_else(|| candle_core::Error::Msg("paged table nnz overflowed".to_string()))?;
        indptr.push(nnz);
        for value in table {
            indices.push(usize_to_i32(*value, "paged block index")?);
        }
        let len = if table.is_empty() {
            0
        } else {
            usize_to_i32(
                context_len.saturating_sub((table.len() - 1) * block_size),
                "paged last page length",
            )?
        };
        last_page_len.push(len);
    }
    let indptr = Tensor::from_vec(indptr, (tables.len() + 1,), device)?;
    let indices_len = indices.len();
    let indices = Tensor::from_vec(indices, (indices_len,), device)?;
    let last_page_len = Tensor::from_vec(last_page_len, (tables.len(),), device)?;
    Ok((indptr, indices, last_page_len))
}

fn usize_to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg(format!("{name} exceeds u32::MAX: {value}")))
}

fn usize_to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| candle_core::Error::Msg(format!("{name} exceeds i32::MAX: {value}")))
}
