//! MTP speculative decoding for Qwen3.5 / Qwen3.8 using the checkpoint's built-in head.
//!
//! Follows vLLM's EAGLE-style proposer: after every target step the drafter is refreshed over the
//! accepted rows (input token shifted by one, target hidden state, same position), which writes its
//! own paged KV for those positions and yields the first draft; further drafts are chained from the
//! drafter's own hidden state at consecutive positions.

use std::sync::atomic::Ordering;

use candle_core::{IndexOp, Result, Tensor};

use crate::{
    get_mut_arcmutex,
    speculative::{
        paged_rows::make_paged_rows_metadata, proposer::sample_draft_rows, SpeculativeAttachInfo,
        SpeculativeCommitRow, SpeculativeConfig, SpeculativeKvCache, SpeculativePrefillCtx,
        SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
        SpeculativeTargetMixin,
    },
};

use super::{mtp::Qwen3_5MtpHead, text::SpecCapture, Qwen3_5Model};

/// vLLM's documented setting for these single-layer MTP heads.
pub const DEFAULT_MTP_N_PREDICT: usize = 2;
const MROPE_DIMS: usize = 3;

/// The last prompt position of a sequence: its next token is only known once sampled, so the
/// drafter processes it during bootstrap instead of prefill.
pub(super) struct PendingPromptTail {
    pub(super) position: usize,
    pub(super) hidden: Tensor,
    pub(super) mrope: [u32; MROPE_DIMS],
}

/// One drafter query row: which sequence, at which target position, fed which (shifted) token.
struct DraftRow {
    seq_id: usize,
    position: usize,
    token: u32,
    mrope: [u32; MROPE_DIMS],
}

impl Qwen3_5Model {
    fn mtp_n_predict(&self) -> usize {
        self.mtp_n_predict.load(Ordering::Relaxed)
    }

    fn mtp_head(&self) -> Result<&Qwen3_5MtpHead> {
        self.text
            .mtp
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("Qwen3.5 MTP head is not loaded"))
    }

    /// Runs the drafter over `rows` (all sequences flattened, `[1, rows]`) and returns the normed
    /// hidden state per row `[1, rows, hidden]`, writing drafter KV at each row's position.
    fn drafter_forward(
        &self,
        head: &Qwen3_5MtpHead,
        rows: &[DraftRow],
        target_hidden: &Tensor,
        kv_cache: &(Tensor, Tensor),
        paged_meta: &crate::pipeline::text_models_inputs_processor::PagedAttentionMeta,
    ) -> Result<Tensor> {
        let device = head.device();
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
        let embeds = self.text.embed_tokens(&tokens)?.to_dtype(head.dtype())?;
        let target_hidden = target_hidden.to_device(device)?.to_dtype(head.dtype())?;
        head.forward(
            &embeds,
            &target_hidden,
            &positions,
            kv_cache.clone(),
            &metadata,
        )
    }

    fn draft_logits(&self, normed_hidden: &Tensor) -> Result<Tensor> {
        // [1, rows, hidden] -> [rows, vocab]
        self.text.lm_head().forward(normed_hidden)?.squeeze(0)
    }

    fn mtp_propose(
        &self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        let head = self.mtp_head()?;
        let n_predict = self.mtp_n_predict();
        let batch = ctx.sequences.len();
        if batch == 0 || n_predict == 0 {
            return Ok(None);
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
            .get(head.kv_layer_idx())
            .ok_or_else(|| candle_core::Error::msg("paged cache has no MTP layer"))?
            .clone();
        let Some(capture) = self.text.last_spec_capture() else {
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
        let normed = self.drafter_forward(head, &rows, &target_hidden, &kv_cache, paged_meta)?;
        let last_idx = Tensor::from_vec(
            last_row_idx.iter().map(|i| *i as u32).collect::<Vec<_>>(),
            (batch,),
            normed.device(),
        )?;
        let mut hidden = normed.index_select(&last_idx, 1)?;
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
            let step_logits = self.draft_logits(&hidden)?;
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
            hidden = self.drafter_forward(head, &chained, &hidden, &kv_cache, paged_meta)?;
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
    fn mtp_prefill(&self, ctx: SpeculativePrefillCtx<'_>) -> Result<()> {
        let head = self.mtp_head()?;
        let SpeculativeKvCache::Paged {
            metadata: paged_meta,
            kv_cache,
        } = ctx.cache;
        let kv_cache = kv_cache
            .get(head.kv_layer_idx())
            .ok_or_else(|| candle_core::Error::msg("paged cache has no MTP layer"))?
            .clone();
        let Some(capture) = self.text.last_full_capture() else {
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
        if rows.is_empty() {
            return Ok(());
        }
        let target_hidden = Tensor::cat(&hidden_rows, 1)?;
        self.drafter_forward(head, &rows, &target_hidden, &kv_cache, paged_meta)?;
        Ok(())
    }
}

struct CaptureView {
    hidden: Tensor,
    mrope: Vec<Vec<Vec<u32>>>,
}

fn capture_view(capture: &SpecCapture) -> Result<CaptureView> {
    let hidden = match capture.hidden.rank() {
        3 => capture.hidden.clone(),
        2 => capture.hidden.unsqueeze(1)?,
        rank => candle_core::bail!("unexpected MTP hidden rank {rank}"),
    };
    let positions = capture.positions.to_dtype(candle_core::DType::U32)?;
    let mrope = match positions.rank() {
        3 => positions.to_vec3::<u32>()?,
        2 => positions.unsqueeze(2)?.to_vec3::<u32>()?,
        rank => candle_core::bail!("unexpected MTP position rank {rank}"),
    };
    Ok(CaptureView { hidden, mrope })
}

fn mrope_at(mrope: &[Vec<Vec<u32>>], batch_idx: usize, row: usize) -> Result<[u32; MROPE_DIMS]> {
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

impl SpeculativeTargetMixin for Qwen3_5Model {
    fn attach_speculative(
        &mut self,
        config: SpeculativeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        let SpeculativeConfig::Mtp(config) = config else {
            self.mtp_n_predict.store(0, Ordering::Relaxed);
            self.text.set_store_spec_hidden(false);
            return Ok(None);
        };
        if !config.is_builtin() {
            candle_core::bail!(
                "Qwen3.5 uses the MTP head built into its checkpoint; pass `--mtp` instead of `--mtp-model`."
            );
        }
        if self.text.mtp.is_none() {
            candle_core::bail!(
                "The built-in MTP head was not loaded; pass `--mtp` when loading the model."
            );
        }
        let n_predict = config.n_predict.unwrap_or(DEFAULT_MTP_N_PREDICT);
        if n_predict == 0 {
            candle_core::bail!("MTP n_predict must be at least 1.");
        }
        self.mtp_n_predict.store(n_predict, Ordering::Relaxed);
        self.text.set_store_spec_hidden(true);
        Ok(Some(SpeculativeAttachInfo::mtp(
            "built-in".to_string(),
            n_predict,
        )))
    }

    fn has_speculative_proposer(&self) -> bool {
        self.mtp_n_predict() > 0
    }

    fn speculative_proposal_len(&self) -> Option<usize> {
        let n = self.mtp_n_predict();
        (n > 0).then_some(n)
    }

    fn speculative_propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        self.mtp_propose(ctx)
    }

    fn speculative_target_hiddens(&self, rows: &[(usize, usize)]) -> Result<Option<Tensor>> {
        let Some(capture) = self.text.last_spec_capture() else {
            return Ok(None);
        };
        let hidden = capture_view(&capture)?.hidden;
        let gathered = rows
            .iter()
            .map(|(batch_idx, row)| hidden.i((*batch_idx, *row)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Tensor::stack(&gathered, 0)?))
    }

    fn speculative_prefill(&mut self, ctx: SpeculativePrefillCtx<'_>) -> Result<()> {
        if self.mtp_n_predict() == 0 {
            return Ok(());
        }
        self.mtp_prefill(ctx)
    }

    fn speculative_commit(&mut self, rows: &[SpeculativeCommitRow]) -> Result<()> {
        for row in rows.iter().filter(|row| !row.accepted_all) {
            self.text
                .replay_recurrent_prefix(row.batch_idx, row.keep_rows)?;
        }
        self.text.clear_gdn_replay_stash();
        Ok(())
    }
}
