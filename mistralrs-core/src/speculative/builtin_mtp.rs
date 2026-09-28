//! Built-in MTP head proposer shared by hybrid GDN targets (Qwen3.5 / Qwen3.8, Qwen4-Exp).
//!
//! Follows vLLM's EAGLE-style proposer: after every target step the drafter is refreshed over the
//! accepted rows (input token shifted by one, target hidden state, same position), which writes its
//! own paged KV for those positions and yields the first draft; further drafts are chained from the
//! drafter's own hidden state at consecutive positions.

#![allow(clippy::cast_possible_truncation)]

use std::{collections::HashMap, sync::Mutex};

use candle_core::{DType, Device, Result, Tensor};

use crate::{
    attention::AttentionMask,
    get_mut_arcmutex,
    layers::CausalMasker,
    layers_masker::CausalMaskConfig,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
    speculative::{
        hybrid_state::SpecCapture, paged_rows::make_paged_rows_metadata,
        proposer::sample_draft_rows, SpeculativeKvCache, SpeculativePrefillCtx,
        SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
        TargetAttentionInputs,
    },
};

const MROPE_DIMS: usize = 3;
// Stands in for the not-yet-sampled next token; the bootstrap refresh rewrites its KV
const PLACEHOLDER_TOKEN: u32 = 0;

pub struct MtpAttentionInputs<'a> {
    pub kv_cache: (Tensor, Tensor),
    pub metadata: &'a PagedAttentionInputMetadata,
    pub attention_mask: &'a AttentionMask,
    pub flash_params: &'a FlashParams,
    /// `[start, end)` of each batch row for a prompt-chunk prefill; `None` when every row is its own decode query.
    pub chunk_ranges: Option<&'a [(usize, usize)]>,
}

/// Output of one drafter forward, `[b, rows, *]`.
pub(crate) struct MtpDraftOutput {
    pub(crate) logits_hidden: Tensor,
    // Next chained target-hidden input when it differs from `logits_hidden`
    pub(crate) chain_hidden: Option<Tensor>,
}

impl MtpDraftOutput {
    fn chain(&self) -> &Tensor {
        self.chain_hidden.as_ref().unwrap_or(&self.logits_hidden)
    }
}

/// Target-side hooks the built-in MTP proposer drives.
pub(crate) trait BuiltinMtpHost {
    fn mtp_device(&self) -> &Device;
    fn mtp_dtype(&self) -> DType;
    /// Absolute paged-KV layer index the head writes to and reads from.
    fn mtp_kv_layer_idx(&self) -> usize;
    fn mtp_embed_tokens(&self, tokens: &Tensor) -> Result<Tensor>;
    /// `positions` are `[3, b, rows]` MRoPE ids; `target_hidden` is the capture (or chained) hidden.
    fn mtp_forward(
        &self,
        input_embeds: &Tensor,
        target_hidden: &Tensor,
        positions: &Tensor,
        attention: MtpAttentionInputs<'_>,
    ) -> Result<MtpDraftOutput>;
    /// `[1, rows, hidden]` -> `[rows, vocab]`
    fn mtp_draft_logits(&self, hidden: &Tensor) -> Result<Tensor>;
    fn mtp_spec_capture(&self) -> Option<SpecCapture>;
    fn mtp_full_capture(&self) -> Option<SpecCapture>;
}

/// The last prompt position of a sequence: its next token is only known once sampled, so the
/// drafter processes it during bootstrap instead of prefill.
pub(crate) struct PendingPromptTail {
    pub(crate) position: usize,
    pub(crate) hidden: Tensor,
    pub(crate) mrope: [u32; MROPE_DIMS],
}

/// One drafter query row: which sequence, at which target position, fed which (shifted) token.
struct DraftRow {
    seq_id: usize,
    position: usize,
    token: u32,
    mrope: [u32; MROPE_DIMS],
}

#[derive(Default)]
pub(crate) struct BuiltinMtpProposer {
    pending_prompt_tails: Mutex<HashMap<usize, PendingPromptTail>>,
}

impl BuiltinMtpProposer {
    /// Runs the drafter over `rows` (all sequences flattened, `[1, rows]`), writing drafter KV at each
    /// row's position.
    fn drafter_forward<H: BuiltinMtpHost + ?Sized>(
        host: &H,
        rows: &[DraftRow],
        target_hidden: &Tensor,
        kv_cache: &(Tensor, Tensor),
        paged_meta: &crate::pipeline::text_models_inputs_processor::PagedAttentionMeta,
    ) -> Result<MtpDraftOutput> {
        let device = host.mtp_device();
        let n = rows.len();
        let tokens = Tensor::from_vec(
            rows.iter().map(|row| row.token).collect::<Vec<_>>(),
            (1, n),
            device,
        )?;
        let mut mrope = Vec::with_capacity(MROPE_DIMS * n);
        for dim in 0..MROPE_DIMS {
            mrope.extend(rows.iter().map(|row| row.mrope[dim]));
        }
        let positions = Tensor::from_vec(mrope, (MROPE_DIMS, 1, n), device)?;
        let seq_ids = rows.iter().map(|row| row.seq_id).collect::<Vec<_>>();
        let context_lens = rows.iter().map(|row| row.position + 1).collect::<Vec<_>>();
        let metadata = make_paged_rows_metadata(&seq_ids, &context_lens, paged_meta, device)?;
        let embeds = host.mtp_embed_tokens(&tokens)?.to_dtype(host.mtp_dtype())?;
        let target_hidden = target_hidden
            .to_device(device)?
            .to_dtype(host.mtp_dtype())?;
        host.mtp_forward(
            &embeds,
            &target_hidden,
            &positions,
            MtpAttentionInputs {
                kv_cache: kv_cache.clone(),
                metadata: &metadata,
                attention_mask: &AttentionMask::None,
                flash_params: &FlashParams::empty(false),
                chunk_ranges: None,
            },
        )
    }

    /// Catch the drafter up over a whole prompt chunk with the target's own attention inputs: one causal
    /// prefill instead of one decode query per prompt row. Row p is fed token p+1; the last row of a
    /// final chunk has no next token yet, so it gets a placeholder whose KV the bootstrap refresh rewrites.
    fn drafter_prefill_chunk<H: BuiltinMtpHost + ?Sized>(
        host: &H,
        ctx: &SpeculativePrefillCtx<'_>,
        target: TargetAttentionInputs<'_>,
        capture: &SpecCapture,
        kv_cache: &(Tensor, Tensor),
    ) -> Result<()> {
        let device = host.mtp_device();
        let (batch, seq_len, _) = capture.hidden.dims3()?;
        if batch != ctx.chunk_ranges.len() {
            candle_core::bail!(
                "MTP prefill capture has {batch} rows for {} sequences",
                ctx.chunk_ranges.len()
            );
        }
        let mut shifted = Vec::with_capacity(batch * seq_len);
        let mut offsets = Vec::with_capacity(batch);
        for ((start, end), toks) in ctx.chunk_ranges.iter().zip(ctx.tokens.iter()) {
            offsets.push(*start);
            for row in 0..seq_len {
                let position = start + row;
                let token = (position + 1 < *end || !ctx.is_final_prompt_chunk)
                    .then(|| toks.get(position + 1).copied())
                    .flatten();
                shifted.push(token.unwrap_or(PLACEHOLDER_TOKEN));
            }
        }
        let tokens = Tensor::from_vec(shifted, (batch, seq_len), device)?;
        let embeds = host.mtp_embed_tokens(&tokens)?.to_dtype(host.mtp_dtype())?;
        let target_hidden = capture
            .hidden
            .to_device(device)?
            .to_dtype(host.mtp_dtype())?;
        let positions = capture.positions.to_device(device)?;
        // Same mask policy as the target's prompt forward: explicit causal mask on the first chunk only
        let attention_mask = if target.metadata.is_first_prompt_chunk {
            CausalMasker.make_causal_mask(
                &tokens,
                &offsets.as_slice(),
                host.mtp_dtype(),
                &CausalMaskConfig::default(),
            )?
        } else {
            AttentionMask::None
        };
        host.mtp_forward(
            &embeds,
            &target_hidden,
            &positions,
            MtpAttentionInputs {
                kv_cache: kv_cache.clone(),
                metadata: target.metadata,
                attention_mask: &attention_mask,
                flash_params: target.flash_params,
                chunk_ranges: Some(ctx.chunk_ranges),
            },
        )?;
        Ok(())
    }

    pub(crate) fn propose<H: BuiltinMtpHost + ?Sized>(
        &self,
        host: &H,
        ctx: SpeculativeProposeBatchCtx<'_>,
        max_n: usize,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        let n_predict = ctx.proposal_len;
        let batch = ctx.sequences.len();
        if batch == 0 || max_n == 0 {
            return Ok(None);
        }
        if n_predict == 0 || n_predict > max_n {
            candle_core::bail!(
                "MTP proposal length {n_predict} is outside the configured range 1..={max_n}"
            );
        }
        if ctx.target_rows.len() != batch || ctx.base_lens.len() != batch {
            candle_core::bail!(
                "MTP batch shape mismatch: sequences={batch}, target_rows={}, base_lens={}",
                ctx.target_rows.len(),
                ctx.base_lens.len()
            );
        }
        let SpeculativeKvCache::Paged {
            metadata: paged_meta,
            kv_cache,
        } = ctx.cache;
        let kv_cache = kv_cache
            .get(host.mtp_kv_layer_idx())
            .ok_or_else(|| candle_core::Error::msg("paged cache has no MTP layer"))?
            .clone();
        let Some(capture) = host.mtp_spec_capture() else {
            return Ok(None);
        };
        let CaptureView { hidden, mrope } = capture_view(&capture)?;

        // Reserve blocks for the drafter's chained positions and the next verify step up front.
        {
            let mut kv_mgr = get_mut_arcmutex!(paged_meta.kv_cache_manager);
            for (seq_id, base_len) in ctx.seq_ids.iter().zip(ctx.base_lens.iter()) {
                if kv_mgr
                    .allocate_slots(*seq_id, base_len + n_predict, &[])
                    .is_none()
                {
                    return Ok(None);
                }
            }
        }

        // Refresh over the anchor + accepted rows of every sequence, flattened into one forward.
        let mut rows = Vec::new();
        let mut hidden_rows = Vec::with_capacity(batch);
        let mut last_row_idx = Vec::with_capacity(batch);
        let mut pending_tails = self
            .pending_prompt_tails
            .lock()
            .expect("mtp tails poisoned");
        for (i, seq) in ctx.sequences.iter().enumerate() {
            let (batch_idx, count) = ctx.target_rows[i];
            let base_len = ctx.base_lens[i];
            let toks = seq.get_toks();
            if count == 0 || base_len < count || toks.len() <= base_len {
                candle_core::bail!(
                    "MTP refresh rows out of range: base_len={base_len}, count={count}, toks={}",
                    toks.len()
                );
            }
            if let Some(tail) = pending_tails.remove(seq.id()) {
                if tail.position + 1 < toks.len() {
                    rows.push(DraftRow {
                        seq_id: ctx.seq_ids[i],
                        position: tail.position,
                        token: toks[tail.position + 1],
                        mrope: tail.mrope,
                    });
                    hidden_rows.push(tail.hidden.to_device(hidden.device())?);
                }
            }
            for r in 0..count {
                let position = base_len - count + r;
                rows.push(DraftRow {
                    seq_id: ctx.seq_ids[i],
                    position,
                    token: toks[position + 1],
                    mrope: mrope_at(&mrope, batch_idx, r)?,
                });
            }
            hidden_rows.push(hidden.narrow(0, batch_idx, 1)?.narrow(1, 0, count)?);
            last_row_idx.push(rows.len() - 1);
        }
        pending_tails.retain(|seq_id, _| ctx.seq_ids.contains(seq_id));
        drop(pending_tails);
        let target_hidden = Tensor::cat(&hidden_rows, 1)?;
        let refreshed = Self::drafter_forward(host, &rows, &target_hidden, &kv_cache, paged_meta)?;
        let last_idx = Tensor::from_vec(
            last_row_idx.iter().map(|i| *i as u32).collect::<Vec<_>>(),
            (batch,),
            refreshed.logits_hidden.device(),
        )?;
        let mut hidden = MtpDraftOutput {
            logits_hidden: refreshed.logits_hidden.index_select(&last_idx, 1)?,
            chain_hidden: refreshed
                .chain_hidden
                .map(|chain| chain.index_select(&last_idx, 1))
                .transpose()?,
        };
        let mut cursor = last_row_idx
            .iter()
            .map(|i| (rows[*i].position, rows[*i].mrope))
            .collect::<Vec<_>>();

        let mut contexts = ctx
            .sequences
            .iter()
            .map(|seq| seq.get_toks().to_vec())
            .collect::<Vec<_>>();
        let mut tokens: Vec<Vec<u32>> = vec![Vec::with_capacity(n_predict); batch];
        let mut logits = Vec::with_capacity(n_predict);
        for step in 0..n_predict {
            let step_logits = host.mtp_draft_logits(&hidden.logits_hidden)?;
            let drafts = sample_draft_rows(&step_logits, ctx.sequences, &mut contexts, &ctx.rng)?;
            for (i, draft) in drafts.iter().enumerate() {
                tokens[i].push(*draft);
            }
            logits.push(step_logits);
            if step + 1 == n_predict {
                break;
            }
            // Chain: the draft becomes the next input at the next position, hidden state carried over.
            let chained = ctx
                .seq_ids
                .iter()
                .zip(cursor.iter_mut())
                .zip(drafts.iter())
                .map(|((seq_id, (position, mrope)), draft)| {
                    *position += 1;
                    for value in mrope.iter_mut() {
                        *value += 1;
                    }
                    DraftRow {
                        seq_id: *seq_id,
                        position: *position,
                        token: *draft,
                        mrope: *mrope,
                    }
                })
                .collect::<Vec<_>>();
            hidden = Self::drafter_forward(host, &chained, hidden.chain(), &kv_cache, paged_meta)?;
        }

        // [n_predict, batch, vocab] -> per sequence [n_predict, vocab]
        let logits = Tensor::stack(&logits, 1)?;
        let proposals = tokens
            .into_iter()
            .enumerate()
            .map(|(row, tokens)| Ok(SpeculativeProposal::with_logits(tokens, logits.get(row)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(SpeculativeProposalBatch::new(proposals)))
    }

    /// Catch the drafter up over a prompt chunk: every position gets (next token, target hidden),
    /// except the last prompt position whose next token is only known once sampled (bootstrap).
    pub(crate) fn prefill<H: BuiltinMtpHost + ?Sized>(
        &self,
        host: &H,
        ctx: SpeculativePrefillCtx<'_>,
    ) -> Result<()> {
        let SpeculativeKvCache::Paged {
            metadata: paged_meta,
            kv_cache,
        } = ctx.cache;
        let kv_cache = kv_cache
            .get(host.mtp_kv_layer_idx())
            .ok_or_else(|| candle_core::Error::msg("paged cache has no MTP layer"))?
            .clone();
        let Some(capture) = host.mtp_full_capture() else {
            return Ok(());
        };
        let CaptureView { hidden, mrope } = capture_view(&capture)?;

        let mut rows = Vec::new();
        let mut hidden_rows = Vec::new();
        let mut pending_tails = self
            .pending_prompt_tails
            .lock()
            .expect("mtp tails poisoned");
        for (i, seq_id) in ctx.seq_ids.iter().enumerate() {
            let batch_idx = ctx.batch_indices[i];
            let toks = ctx.tokens[i];
            let (start, end) = ctx.chunk_ranges[i];
            if end <= start || hidden.dim(1)? < end - start {
                candle_core::bail!(
                    "MTP prefill rows out of range: chunk=({start}, {end}), hidden rows={}",
                    hidden.dim(1)?
                );
            }
            let last = if ctx.is_final_prompt_chunk {
                let tail_row = end - 1 - start;
                pending_tails.insert(
                    *seq_id,
                    PendingPromptTail {
                        position: end - 1,
                        hidden: hidden.narrow(0, batch_idx, 1)?.narrow(1, tail_row, 1)?,
                        mrope: mrope_at(&mrope, batch_idx, tail_row)?,
                    },
                );
                end - 1
            } else {
                end
            };
            if ctx.target_attention.is_some() {
                continue;
            }
            let count = last - start;
            if count == 0 {
                continue;
            }
            if toks.len() <= last {
                candle_core::bail!(
                    "MTP prefill tokens out of range: chunk=({start}, {end}), toks={}",
                    toks.len()
                );
            }
            for r in 0..count {
                let position = start + r;
                rows.push(DraftRow {
                    seq_id: *seq_id,
                    position,
                    token: toks[position + 1],
                    mrope: mrope_at(&mrope, batch_idx, r)?,
                });
            }
            hidden_rows.push(hidden.narrow(0, batch_idx, 1)?.narrow(1, 0, count)?);
        }
        drop(pending_tails);
        if let Some(target) = ctx.target_attention {
            return Self::drafter_prefill_chunk(host, &ctx, target, &capture, &kv_cache);
        }
        if rows.is_empty() {
            return Ok(());
        }
        let target_hidden = Tensor::cat(&hidden_rows, 1)?;
        Self::drafter_forward(host, &rows, &target_hidden, &kv_cache, paged_meta)?;
        Ok(())
    }
}

pub(crate) struct CaptureView {
    pub(crate) hidden: Tensor,
    pub(crate) mrope: Vec<Vec<Vec<u32>>>,
}

pub(crate) fn capture_view(capture: &SpecCapture) -> Result<CaptureView> {
    let hidden = match capture.hidden.rank() {
        3 => capture.hidden.clone(),
        2 => capture.hidden.unsqueeze(1)?,
        rank => candle_core::bail!("unexpected MTP hidden rank {rank}"),
    };
    let positions = capture.positions.to_dtype(candle_core::DType::U32)?;
    let mrope = match positions.rank() {
        3 => positions.to_vec3::<u32>()?,
        2 => {
            let positions = positions.to_vec2::<u32>()?;
            vec![positions.clone(), positions.clone(), positions]
        }
        rank => candle_core::bail!("unexpected MTP position rank {rank}"),
    };
    Ok(CaptureView { hidden, mrope })
}

pub(crate) fn mrope_at(
    mrope: &[Vec<Vec<u32>>],
    batch_idx: usize,
    row: usize,
) -> Result<[u32; MROPE_DIMS]> {
    let mut out = [0u32; MROPE_DIMS];
    for (dim, slot) in out.iter_mut().enumerate() {
        *slot = *mrope
            .get(dim)
            .and_then(|b| b.get(batch_idx))
            .and_then(|r| r.get(row))
            .ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "MTP position ids missing for batch {batch_idx} row {row}"
                ))
            })?;
    }
    Ok(out)
}
