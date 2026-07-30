use std::ops::Range;

use candle_core::{Result, Tensor};

use crate::{
    gdn::{GatedDeltaNet, GdnLayerCache},
    pipeline::{ModelForwardContext, RecurrentBatchKind},
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackedGdnSegment {
    token_range: Range<usize>,
    state_index: usize,
}

fn packed_gdn_segments(
    physical_batch: usize,
    physical_tokens: usize,
    query_lens: &[usize],
) -> Result<Vec<PackedGdnSegment>> {
    if physical_batch != 1 {
        candle_core::bail!(
            "Qwen3.5 packed GDN requires physical batch size 1, got {physical_batch}"
        );
    }
    if query_lens.is_empty() {
        candle_core::bail!("Qwen3.5 packed GDN requires at least one logical sequence");
    }
    let mut offset = 0usize;
    let mut segments = Vec::with_capacity(query_lens.len());
    for (state_index, &query_len) in query_lens.iter().enumerate() {
        if query_len == 0 {
            candle_core::bail!("Qwen3.5 packed GDN logical sequence {state_index} has zero tokens");
        }
        let end = offset
            .checked_add(query_len)
            .ok_or_else(|| candle_core::Error::msg("Qwen3.5 packed GDN length overflow"))?;
        segments.push(PackedGdnSegment {
            token_range: offset..end,
            state_index,
        });
        offset = end;
    }
    if offset != physical_tokens {
        candle_core::bail!(
            "Qwen3.5 packed GDN has {offset} logical tokens but {physical_tokens} physical tokens"
        );
    }
    Ok(segments)
}

fn validate_packed_gdn_state_rows(
    logical_batch: usize,
    conv_state_batch: usize,
    recurrent_state_batch: usize,
) -> Result<()> {
    if conv_state_batch != logical_batch {
        candle_core::bail!(
            "Qwen3.5 packed GDN has {conv_state_batch} convolution state rows but {logical_batch} logical sequences"
        );
    }
    if recurrent_state_batch != logical_batch {
        candle_core::bail!(
            "Qwen3.5 packed GDN has {recurrent_state_batch} recurrent state rows but {logical_batch} logical sequences"
        );
    }
    Ok(())
}

pub(crate) fn packed_gdn_query_lens(
    x: &Tensor,
    ctx: &ModelForwardContext<'_>,
) -> Result<Option<Vec<usize>>> {
    if !ctx.flash_params().packed {
        return Ok(None);
    }
    if !ctx.is_first_prompt_chunk() {
        candle_core::bail!("Qwen3.5 packed GDN requires the first prompt chunk");
    }
    let query_lens = ctx
        .paged_input_metadata()
        .and_then(|metadata| metadata.query_lens.clone())
        .ok_or_else(|| {
            candle_core::Error::msg("Qwen3.5 packed GDN requires logical query lengths")
        })?;
    let recurrent_metadata = ctx
        .recurrent_metadata()
        .ok_or_else(|| candle_core::Error::msg("Qwen3.5 packed GDN requires recurrent metadata"))?;
    if recurrent_metadata.batch_kind() != RecurrentBatchKind::Prefill {
        candle_core::bail!("Qwen3.5 packed GDN cannot run a decode batch");
    }
    let (physical_batch, physical_tokens, _) = x.dims3()?;
    packed_gdn_segments(physical_batch, physical_tokens, &query_lens)?;
    let index_count = recurrent_metadata.state_indices().dims1()?;
    if index_count != query_lens.len() {
        candle_core::bail!(
            "Qwen3.5 packed GDN has {index_count} state indices but {} logical sequences",
            query_lens.len()
        );
    }
    if let Some(host_indices) = recurrent_metadata.state_indices_host() {
        if host_indices.len() != query_lens.len() {
            candle_core::bail!(
                "Qwen3.5 packed GDN has {} host state indices but {} logical sequences",
                host_indices.len(),
                query_lens.len()
            );
        }
    }
    Ok(Some(query_lens))
}

pub(crate) fn forward_packed_gdn(
    gdn: &GatedDeltaNet,
    x: &Tensor,
    cache: &mut GdnLayerCache,
    batch_kind: RecurrentBatchKind,
    query_lens: &[usize],
) -> Result<Tensor> {
    if batch_kind != RecurrentBatchKind::Prefill {
        candle_core::bail!("Qwen3.5 packed GDN cannot run a decode batch");
    }
    let (physical_batch, physical_tokens, _) = x.dims3()?;
    let (conv_state_batch, _, _) = cache.conv_state.dims3()?;
    let (recurrent_state_batch, _, _, _) = cache.recurrent_state.dims4()?;
    let segments = packed_gdn_segments(physical_batch, physical_tokens, query_lens)?;
    validate_packed_gdn_state_rows(segments.len(), conv_state_batch, recurrent_state_batch)?;
    if x.dtype() != cache.conv_state.dtype() {
        candle_core::bail!(
            "Qwen3.5 packed GDN dtype mismatch: tokens are {:?}, convolution state is {:?}",
            x.dtype(),
            cache.conv_state.dtype()
        );
    }
    if !x.device().same_device(cache.conv_state.device())
        || !x.device().same_device(cache.recurrent_state.device())
    {
        candle_core::bail!(
            "Qwen3.5 packed GDN tokens and recurrent states are on different devices"
        );
    }

    let mut outputs = Vec::with_capacity(segments.len());
    let mut next_conv_states = Vec::with_capacity(segments.len());
    let mut next_recurrent_states = Vec::with_capacity(segments.len());
    for segment in segments {
        let segment_x = x.narrow(1, segment.token_range.start, segment.token_range.len())?;
        let mut segment_cache = GdnLayerCache {
            conv_state: cache.conv_state.narrow(0, segment.state_index, 1)?,
            recurrent_state: cache.recurrent_state.narrow(0, segment.state_index, 1)?,
        };
        outputs.push(mistralrs_quant::with_lora_execution_row_range(
            segment.token_range.clone(),
            || gdn.forward(&segment_x, &mut segment_cache, RecurrentBatchKind::Prefill),
        )?);
        next_conv_states.push(segment_cache.conv_state);
        next_recurrent_states.push(segment_cache.recurrent_state);
    }
    cache.conv_state = Tensor::cat(&next_conv_states, 0)?;
    cache.recurrent_state = Tensor::cat(&next_recurrent_states, 0)?;
    Tensor::cat(&outputs, 1)
}

#[cfg(test)]
mod tests {
    use super::{packed_gdn_segments, validate_packed_gdn_state_rows, PackedGdnSegment};

    #[test]
    fn unequal_queries_map_to_matching_state_rows() {
        assert_eq!(
            packed_gdn_segments(1, 8, &[2, 5, 1]).unwrap(),
            vec![
                PackedGdnSegment {
                    token_range: 0..2,
                    state_index: 0,
                },
                PackedGdnSegment {
                    token_range: 2..7,
                    state_index: 1,
                },
                PackedGdnSegment {
                    token_range: 7..8,
                    state_index: 2,
                },
            ]
        );
    }

    #[test]
    fn rejects_query_and_state_cardinality_mismatches() {
        assert!(packed_gdn_segments(2, 8, &[2, 5, 1]).is_err());
        assert!(packed_gdn_segments(1, 0, &[]).is_err());
        assert!(packed_gdn_segments(1, 7, &[2, 5, 1]).is_err());
        assert!(packed_gdn_segments(1, 8, &[2, 0, 6]).is_err());
        assert!(packed_gdn_segments(1, usize::MAX, &[usize::MAX, 1]).is_err());
        assert!(validate_packed_gdn_state_rows(3, 2, 3).is_err());
        assert!(validate_packed_gdn_state_rows(3, 3, 2).is_err());
    }

    #[test]
    fn keeps_token_and_state_order_isolated() {
        let tokens = [10, 11, 20, 21, 22, 30, 31];
        let state_markers = [100, 200, 300];
        let observed = packed_gdn_segments(1, tokens.len(), &[2, 3, 2])
            .unwrap()
            .into_iter()
            .map(|segment| {
                (
                    state_markers[segment.state_index],
                    tokens[segment.token_range].to_vec(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            observed,
            vec![
                (100, vec![10, 11]),
                (200, vec![20, 21, 22]),
                (300, vec![30, 31]),
            ]
        );
    }
}
