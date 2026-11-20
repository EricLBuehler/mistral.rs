//! Model-agnostic DFlash proposing; a target only supplies its captures and embeddings.

use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::QuantMethod;
use rand::Rng;

use super::{
    CtxAppend, DFlashDraftModel, DFlashGraphProposalInputs, DFlashPreparedContext,
    DFlashProposalBatch, DFlashSamplingInputs, DEFAULT_MAX_DRAFTS,
};
#[cfg(cuda_flash_attn)]
use crate::pipeline::cuda_graph::{CudaGraphComponent, CudaGraphEvent, CudaGraphEventGuard};
use crate::speculative::hybrid_state::SpecCapture;
use crate::speculative::{
    SpeculativePrefillCtx, SpeculativeProposal, SpeculativeProposalBatch,
    SpeculativeProposeBatchCtx, SpeculativeProposePreparation, SpeculativeProposePrepareCtx,
    SpeculativeTapRouting,
};

pub(crate) trait DFlashHost {
    fn dflash_device(&self) -> &Device;
    fn dflash_embed_tokens(&self, tokens: &Tensor) -> Result<Tensor>;
    fn dflash_token_embedding(&self) -> &Arc<dyn QuantMethod>;
    fn dflash_spec_capture(&self) -> Option<SpecCapture>;
    fn dflash_full_capture(&self) -> Option<SpecCapture>;
}

struct DFlashPreparedRow {
    seq_id: usize,
    batch_idx: usize,
    start_pos: usize,
    offset: usize,
    rows: usize,
}

struct DFlashProposePreparation {
    context: DFlashPreparedContext,
    rows: Vec<DFlashPreparedRow>,
}

fn dflash_speculative_batch(batch: DFlashProposalBatch) -> Result<SpeculativeProposalBatch> {
    let proposals = match batch {
        DFlashProposalBatch::Tokens(tokens) => {
            tokens.into_iter().map(SpeculativeProposal::new).collect()
        }
        #[cfg(cuda_flash_attn)]
        DFlashProposalBatch::DeviceTokens(tokens) => {
            let batch = tokens.dim(0)?;
            (0..batch)
                .map(|row| SpeculativeProposal::from_device(tokens.get(row)?))
                .collect::<Result<Vec<_>>>()?
        }
        #[cfg(feature = "cuda")]
        DFlashProposalBatch::DeviceSparse {
            tokens,
            candidate_ids,
            candidate_probs,
        } => {
            let batch = tokens.dim(0)?;
            if candidate_ids.dim(0)? != batch || candidate_probs.dim(0)? != batch {
                candle_core::bail!("DFlash sparse proposal batch does not match token rows");
            }
            (0..batch)
                .map(|row| {
                    SpeculativeProposal::with_device_sparse_probs(
                        tokens.get(row)?,
                        candidate_ids.get(row)?,
                        candidate_probs.get(row)?,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        }
    };
    Ok(SpeculativeProposalBatch::new(proposals))
}

pub(crate) fn resolve_n_predict(
    requested: Option<usize>,
    block_size: usize,
    checkpoint_lanes: usize,
) -> Result<usize> {
    let max_drafts = block_size
        .checked_sub(1)
        .filter(|max_drafts| *max_drafts > 0)
        .ok_or_else(|| candle_core::Error::msg("DFlash block size must be at least 2"))?;
    let configured = requested.unwrap_or(max_drafts.min(DEFAULT_MAX_DRAFTS));
    if configured == 0 || configured > max_drafts {
        candle_core::bail!(
            "requested {configured} draft tokens but this DFlash drafter's block size is {block_size} (max {max_drafts} drafts)"
        );
    }
    if checkpoint_lanes == 1 {
        return Ok(configured);
    }
    let reserved = checkpoint_lanes.checked_sub(1).ok_or_else(|| {
        candle_core::Error::msg("recurrent checkpoint lane count must be nonzero")
    })?;
    if reserved > max_drafts {
        candle_core::bail!(
            "reserved {checkpoint_lanes} recurrent checkpoint lanes but this DFlash drafter's block size is {block_size} (max {} lanes)",
            max_drafts + 1
        );
    }
    if let Some(requested) = requested {
        if requested != reserved {
            candle_core::bail!(
                "requested {requested} draft tokens require {} recurrent checkpoint lanes, but {checkpoint_lanes} lanes were reserved",
                requested + 1
            );
        }
    }
    Ok(reserved)
}

/// `[batch, n + 1, hidden]` noise rows `[anchor, mask * n]`, embedded in one call.
fn noise_embedding(
    host: &dyn DFlashHost,
    drafter: &DFlashDraftModel,
    anchors: &[u32],
    n: usize,
) -> Result<Tensor> {
    let block = n + 1;
    let mut ids = Vec::with_capacity(anchors.len() * block);
    for anchor in anchors {
        ids.push(*anchor);
        ids.extend(std::iter::repeat_n(drafter.mask_token_id(), n));
    }
    let ids = Tensor::from_vec(ids, (anchors.len(), block), host.dflash_device())?;
    let mut emb = host.dflash_embed_tokens(&ids)?;
    let scale = drafter.input_embedding_scale();
    if (scale - 1.0).abs() > f64::EPSILON {
        emb = (emb * scale)?;
    }
    Ok(emb)
}

pub(crate) fn prepare_propose(
    host: &dyn DFlashHost,
    drafter: &DFlashDraftModel,
    ctx: SpeculativeProposePrepareCtx<'_>,
) -> Result<Option<Box<dyn SpeculativeProposePreparation>>> {
    if ctx.seq_ids.is_empty() {
        return Ok(None);
    }
    if ctx.seq_ids.len() != ctx.base_lens.len()
        || ctx.seq_ids.len() != ctx.target_rows.len()
        || drafter.has_dormant_seq(ctx.seq_ids)
    {
        return Ok(None);
    }
    let Some(capture) = host.dflash_spec_capture() else {
        return Ok(None);
    };
    if capture.taps.len() != drafter.target_layer_ids.len() {
        return Ok(None);
    }
    let (source_batch, source_rows, _) = capture.taps[0].dims3()?;
    let mut appends = Vec::with_capacity(ctx.seq_ids.len());
    let mut rows = Vec::with_capacity(ctx.seq_ids.len());
    let mut flat_row_indices = Vec::new();
    for ((seq_id, base_len), &(batch_idx, count)) in
        ctx.seq_ids.iter().zip(ctx.base_lens).zip(ctx.target_rows)
    {
        if count == 0
            || batch_idx >= source_batch
            || count > source_rows
            || drafter.ctx_next_pos(*seq_id) != Some(*base_len)
        {
            return Ok(None);
        }
        let flat_start = batch_idx
            .checked_mul(source_rows)
            .ok_or_else(|| candle_core::Error::msg("DFlash tap row index overflow"))?;
        let offset = flat_row_indices.len();
        for row in flat_start..flat_start + count {
            flat_row_indices.push(u32::try_from(row).map_err(candle_core::Error::wrap)?);
        }
        appends.push(CtxAppend {
            seq_id: *seq_id,
            rows: count,
            start_pos: *base_len,
        });
        rows.push(DFlashPreparedRow {
            seq_id: *seq_id,
            batch_idx,
            start_pos: *base_len,
            offset,
            rows: count,
        });
    }
    let context = drafter.prepare_ctx_batch(&capture.taps, flat_row_indices, &appends)?;
    Ok(Some(Box::new(DFlashProposePreparation { context, rows })))
}

/// Drafts up to `max_n` tokens per row, scoring them with `lm_head`.
pub(crate) fn propose(
    host: &dyn DFlashHost,
    drafter: &DFlashDraftModel,
    lm_head: &Arc<dyn QuantMethod>,
    max_n: usize,
    ctx: SpeculativeProposeBatchCtx<'_>,
) -> Result<Option<SpeculativeProposalBatch>> {
    let batch = ctx.seq_ids.len();
    if batch == 0 || max_n == 0 {
        return Ok(None);
    }
    let n_predict = ctx.proposal_len;
    if n_predict == 0 {
        return Ok(None);
    }
    if n_predict > max_n {
        candle_core::bail!("DFlash proposal length {n_predict} exceeds configured maximum {max_n}");
    }
    let Some(capture) = host.dflash_spec_capture() else {
        return Ok(None);
    };
    if capture.taps.len() != drafter.target_layer_ids.len() {
        return Ok(None);
    }
    if drafter.has_dormant_seq(ctx.seq_ids) {
        return Ok(None);
    }

    let prepared = ctx.preparation.and_then(|preparation| {
        preparation
            .as_any()
            .downcast_ref::<DFlashProposePreparation>()
    });
    let prepared_commit = prepared.and_then(|prepared| {
        let mut appends = Vec::with_capacity(batch);
        let mut row_indices = Vec::new();
        for (i, seq_id) in ctx.seq_ids.iter().enumerate() {
            let (batch_idx, count) = ctx.target_rows[i];
            let base_len = ctx.base_lens[i];
            let row = prepared
                .rows
                .iter()
                .find(|row| row.seq_id == *seq_id && row.batch_idx == batch_idx)?;
            if count == 0
                || count > row.rows
                || row.start_pos.checked_add(count) != Some(base_len)
                || drafter.ctx_next_pos(*seq_id) != Some(row.start_pos)
            {
                return None;
            }
            for index in row.offset..row.offset + count {
                row_indices.push(u32::try_from(index).ok()?);
            }
            appends.push(CtxAppend {
                seq_id: *seq_id,
                rows: count,
                start_pos: row.start_pos,
            });
        }
        Some((prepared, row_indices, appends))
    });
    if let Some((prepared, row_indices, appends)) = prepared_commit {
        drafter.commit_prepared_ctx_batch(&prepared.context, row_indices, &appends)?;
    } else {
        let source_rows = capture.taps[0].dim(1)?;
        let mut appends = Vec::with_capacity(batch);
        let mut flat_row_indices = Vec::new();
        for (i, seq_id) in ctx.seq_ids.iter().enumerate() {
            let (batch_idx, count) = ctx.target_rows[i];
            let base_len = ctx.base_lens[i];
            let ctx_next = drafter.ctx_next_pos(*seq_id);
            let needed = match ctx_next {
                Some(next) if next <= base_len => base_len - next,
                _ => count.min(base_len),
            };
            if needed > 0 {
                if needed > count || source_rows < count {
                    return Ok(None);
                }
                let start_row = count - needed;
                let flat_start = batch_idx
                    .checked_mul(source_rows)
                    .and_then(|row| row.checked_add(start_row))
                    .ok_or_else(|| candle_core::Error::msg("DFlash tap row index overflow"))?;
                let flat_end = flat_start
                    .checked_add(needed)
                    .ok_or_else(|| candle_core::Error::msg("DFlash tap row index overflow"))?;
                for row in flat_start..flat_end {
                    flat_row_indices.push(u32::try_from(row).map_err(candle_core::Error::wrap)?);
                }
                appends.push(CtxAppend {
                    seq_id: *seq_id,
                    rows: needed,
                    start_pos: base_len - needed,
                });
            }
        }
        drafter.append_ctx_batch(&capture.taps, flat_row_indices, &appends)?;
    }
    if !drafter.contexts_ready_for_draft(ctx.seq_ids) {
        return Ok(None);
    }
    let sampling_values = if drafter.has_selector()
        && drafter.draft_sampling_method()
            == crate::speculative::MtpDraftSamplingMethod::Probabilistic
    {
        let probabilistic_rows = ctx
            .sequences
            .iter()
            .map(|seq| {
                crate::speculative::verifier::stochastic_verification_allowed_for_sequence(seq)
            })
            .collect::<Vec<_>>();
        if probabilistic_rows.iter().any(|eligible| *eligible) {
            let mut inverse_temperatures = Vec::with_capacity(batch);
            for (seq, eligible) in ctx.sequences.iter().zip(&probabilistic_rows) {
                if !*eligible {
                    inverse_temperatures.push(0.0);
                    continue;
                }
                let Some(temperature) = seq.sampler().temperature() else {
                    return Ok(None);
                };
                #[allow(clippy::cast_possible_truncation)]
                let inverse_temperature = (1.0 / temperature) as f32;
                if !inverse_temperature.is_finite() || inverse_temperature <= 0.0 {
                    return Ok(None);
                }
                inverse_temperatures.push(inverse_temperature);
            }
            let mut uniforms = vec![0.0f32; batch * n_predict];
            for (row, inverse_temperature) in inverse_temperatures.iter().enumerate() {
                if *inverse_temperature > 0.0 {
                    let rng = ctx.sequences[row].sampling_rng(&ctx.rng);
                    let mut rng = rng.lock().expect("could not lock rng mutex");
                    for uniform in &mut uniforms[row * n_predict..(row + 1) * n_predict] {
                        *uniform = rng.random();
                    }
                }
            }
            Some((inverse_temperatures, uniforms))
        } else {
            None
        }
    } else {
        None
    };
    let sampling = sampling_values
        .as_ref()
        .map(|(inverse_temperatures, uniforms)| DFlashSamplingInputs {
            inverse_temperatures,
            uniforms,
        });
    let graph_proposals = drafter.proposals_cuda_graph(&DFlashGraphProposalInputs {
        seq_ids: ctx.seq_ids,
        anchors: ctx.sampled_tokens,
        start_positions: ctx.base_lens,
        n_predict,
        sampling,
        token_embedding: host.dflash_token_embedding(),
        lm_head,
    })?;
    if let Some(proposals) = graph_proposals {
        return dflash_speculative_batch(proposals).map(Some);
    }
    #[cfg(cuda_flash_attn)]
    let graph_event =
        CudaGraphEventGuard::new(CudaGraphComponent::DFlash, CudaGraphEvent::EagerFallback);
    let fallback_result = (|| {
        let noise = noise_embedding(host, drafter, ctx.sampled_tokens, n_predict)?;
        let hidden = drafter.draft_hidden_batch(ctx.seq_ids, &noise, ctx.base_lens)?;
        dflash_speculative_batch(drafter.finish_proposals(
            &hidden,
            ctx.sampled_tokens,
            sampling,
            lm_head,
        )?)
        .map(Some)
    })();
    #[cfg(cuda_flash_attn)]
    if fallback_result.is_ok() {
        graph_event.success();
    }
    fallback_result
}

pub(crate) fn prefill(
    host: &dyn DFlashHost,
    drafter: &DFlashDraftModel,
    ctx: SpeculativePrefillCtx<'_>,
) -> Result<()> {
    let Some(capture) = host.dflash_full_capture() else {
        return Ok(());
    };
    if capture.taps.len() != drafter.target_layer_ids.len() {
        return Ok(());
    }
    let (capture_batch, capture_rows, _) = capture.taps[0].dims3()?;
    let routing = SpeculativeTapRouting::new(
        ctx.capture_layout(),
        capture_batch,
        capture_rows,
        ctx.batch_indices,
        ctx.chunk_ranges,
    )?;
    drafter.activate_seqs(ctx.seq_ids);
    let mut appends = Vec::with_capacity(ctx.seq_ids.len());
    for ((seq_id, &(start, _)), span) in ctx
        .seq_ids
        .iter()
        .zip(ctx.chunk_ranges)
        .zip(routing.spans())
    {
        appends.push(CtxAppend {
            seq_id: *seq_id,
            rows: span.rows(),
            start_pos: start,
        });
    }
    let flat_row_indices = routing.flat_row_indices()?;
    drafter.append_ctx_batch(&capture.taps, flat_row_indices, &appends)
}

#[cfg(test)]
mod tests {
    use super::resolve_n_predict;

    #[test]
    fn auto_dflash_depth_follows_reserved_checkpoint_lanes() {
        assert_eq!(resolve_n_predict(None, 8, 8).unwrap(), 7);
        assert_eq!(resolve_n_predict(None, 8, 4).unwrap(), 3);
    }

    #[test]
    fn replay_sentinel_keeps_configured_or_default_dflash_depth() {
        assert_eq!(resolve_n_predict(None, 8, 1).unwrap(), 7);
        assert_eq!(resolve_n_predict(Some(3), 8, 1).unwrap(), 3);
    }

    #[test]
    fn explicit_dflash_depth_must_match_reserved_checkpoint_lanes() {
        assert_eq!(resolve_n_predict(Some(3), 8, 4).unwrap(), 3);
        let error = resolve_n_predict(Some(3), 8, 8)
            .expect_err("explicit depth must match the reserved lanes");
        assert!(error.to_string().contains(
            "requested 3 draft tokens require 4 recurrent checkpoint lanes, but 8 lanes were reserved"
        ));
    }

    #[test]
    fn reserved_dflash_depth_must_fit_the_drafter_block() {
        let error =
            resolve_n_predict(None, 8, 9).expect_err("reserved lanes must fit the drafter block");
        assert!(error
            .to_string()
            .contains("reserved 9 recurrent checkpoint lanes"));
    }
}
