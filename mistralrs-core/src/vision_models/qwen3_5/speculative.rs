//! MTP speculative decoding for Qwen3.5 / Qwen3.8 using the checkpoint's built-in head.
//!
//! Follows vLLM's EAGLE-style proposer: after every target step the drafter is refreshed over the
//! accepted rows (input token shifted by one, target hidden state, same position), which writes its
//! own paged KV for those positions and yields the first draft; further drafts are chained from the
//! drafter's own hidden state at consecutive positions.

use std::sync::atomic::Ordering;

use candle_core::{IndexOp, Result, Tensor};

use crate::{
    attention::AttentionMask,
    get_mut_arcmutex,
    layers::CausalMasker,
    layers_masker::CausalMaskConfig,
    pipeline::text_models_inputs_processor::FlashParams,
    speculative::{
        dflash::DFlashDraftModel, paged_rows::make_paged_rows_metadata,
        proposer::sample_draft_rows, SpeculativeAttachInfo, SpeculativeCommitRow,
        SpeculativeConfig, SpeculativeGraphState, SpeculativeKvCache, SpeculativePrefillCtx,
        SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
        SpeculativeTargetMixin, TargetAttentionInputs,
    },
};

use super::{
    mtp::{MtpAttentionInputs, Qwen3_5MtpHead},
    text::{SpecCapture, SpecGraphState},
    Qwen3_5Model,
};

/// vLLM's documented setting for these single-layer MTP heads.
pub const DEFAULT_MTP_N_PREDICT: usize = 2;
// On larger models the verify forward dominates the step, so deeper drafting pays; measured on
// Qwen3.8-27B: n=3 beats n=2 by ~5% while n=4 over-drafts (acceptance falls under 50%).
pub const DEFAULT_MTP_N_PREDICT_LARGE: usize = 3;
const MTP_LARGE_HIDDEN_SIZE: usize = 4096;
// Verify cost grows with block width and quantized targets accept shorter blocks anyway; deeper
// drafting stays available via --mtp-n-predict.
const DFLASH_DEFAULT_MAX_DRAFTS: usize = 7;
const MROPE_DIMS: usize = 3;
// Feeds prompt rows whose next token is not known yet; their KV is rewritten before it is ever attended to
const PLACEHOLDER_TOKEN: u32 = 0;

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
            MtpAttentionInputs {
                kv_cache: kv_cache.clone(),
                metadata: &metadata,
                attention_mask: &AttentionMask::None,
                flash_params: &FlashParams::empty(false),
            },
        )
    }

    /// Catch the drafter up over a whole prompt chunk with the target's own attention inputs: one causal
    /// prefill instead of one decode query per prompt row. Row p is fed token p+1; the last row of a
    /// final chunk has no next token yet, so it gets a placeholder whose KV the bootstrap refresh rewrites.
    fn drafter_prefill_chunk(
        &self,
        head: &Qwen3_5MtpHead,
        ctx: &SpeculativePrefillCtx<'_>,
        target: TargetAttentionInputs<'_>,
        capture: &SpecCapture,
        kv_cache: &(Tensor, Tensor),
    ) -> Result<()> {
        let device = head.device();
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
        let embeds = self.text.embed_tokens(&tokens)?.to_dtype(head.dtype())?;
        let target_hidden = capture.hidden.to_device(device)?.to_dtype(head.dtype())?;
        let positions = capture.positions.to_device(device)?;
        // Same mask policy as the target's prompt forward: explicit causal mask on the first chunk only
        let attention_mask = if target.metadata.is_first_prompt_chunk {
            CausalMasker.make_causal_mask(
                &tokens,
                &offsets.as_slice(),
                head.dtype(),
                &CausalMaskConfig::default(),
            )?
        } else {
            AttentionMask::None
        };
        head.forward(
            &embeds,
            &target_hidden,
            &positions,
            MtpAttentionInputs {
                kv_cache: kv_cache.clone(),
                metadata: target.metadata,
                attention_mask: &attention_mask,
                flash_params: target.flash_params,
            },
        )?;
        Ok(())
    }

    fn draft_logits(&self, normed_hidden: &Tensor) -> Result<Tensor> {
        let draft_head = self.draft_lm_head.lock().expect("draft lm_head poisoned");
        let head = draft_head.as_ref().unwrap_or_else(|| self.text.lm_head());
        // [1, rows, hidden] -> [rows, vocab]
        head.forward(normed_hidden)?.squeeze(0)
    }

    fn attach_dflash(
        &mut self,
        config: crate::speculative::MtpConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        let drafter = DFlashDraftModel::load(
            &config,
            self.text.layer_types.len(),
            self.text.cfg.hidden_size,
            &self.text.device,
            self.text.dtype,
            false,
        )?;
        let block = drafter.block_size();
        let n_predict = match config.n_predict {
            Some(n) if n + 1 > block => {
                candle_core::bail!(
                    "requested {n} draft tokens but this DFlash drafter's block size is {block} (max {} drafts)",
                    block - 1
                );
            }
            Some(0) => candle_core::bail!("MTP n_predict must be at least 1."),
            Some(n) => n,
            None => (block - 1).min(DFLASH_DEFAULT_MAX_DRAFTS),
        };
        self.text
            .set_dflash_tap_layers(drafter.target_layer_ids.clone());
        self.mtp_n_predict.store(n_predict, Ordering::Relaxed);
        self.text.set_store_spec_hidden(true);
        if let Some(ty) = config.draft_lm_head_isq {
            let head = self.text.lm_head().clone().apply_isq(
                Some(ty),
                self.text.device.clone(),
                &std::sync::atomic::AtomicUsize::new(0),
                None,
                mistralrs_quant::QuantizeOntoGuard::new(),
            )?;
            *self.draft_lm_head.lock().expect("draft lm_head poisoned") = Some(head);
        }
        // Explicit --mtp-n-predict pins the depth; the default adapts it to measured acceptance
        let adaptive = match std::env::var("MISTRALRS_DFLASH_ADAPTIVE").ok().as_deref() {
            Some("0" | "false") => false,
            Some(_) => true,
            None => config.n_predict.is_none(),
        };
        let adaptive = adaptive && drafter.enable_adaptive(n_predict);
        let kind = if drafter.has_selector() {
            "DFlash2"
        } else {
            "DFlash"
        };
        let depth = if adaptive {
            format!("adaptive depth <= {n_predict}")
        } else {
            format!("depth {n_predict}")
        };
        let name = format!(
            "{kind} `{}` (block {block}, {depth}, taps {:?})",
            config.model.as_deref().unwrap_or("dflash"),
            drafter.target_layer_ids
        );
        *self.dflash.lock().expect("dflash poisoned") = Some(std::sync::Arc::new(drafter));
        Ok(Some(SpeculativeAttachInfo::mtp(name, n_predict)))
    }

    /// `[batch, n + 1, hidden]` noise rows `[anchor, mask * n]`, embedded in one call.
    fn dflash_noise_embedding(
        &self,
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
        let ids = Tensor::from_vec(ids, (anchors.len(), block), &self.text.device)?;
        let mut emb = self.text.embed_tokens(&ids)?;
        let scale = drafter.input_embedding_scale();
        if (scale - 1.0).abs() > f64::EPSILON {
            emb = (emb * scale)?;
        }
        Ok(emb)
    }

    fn dflash_propose(
        &self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        let drafter = self
            .dflash
            .lock()
            .expect("dflash poisoned")
            .clone()
            .ok_or_else(|| candle_core::Error::msg("DFlash propose without a drafter"))?;
        let max_n = self.mtp_n_predict();
        let batch = ctx.seq_ids.len();
        if batch == 0 || max_n == 0 {
            return Ok(None);
        }
        // Must match what the driver just read via speculative_proposal_len; the adaptive update
        // below only takes effect next step.
        let n_predict = drafter.current_n(max_n);
        let Some(capture) = self.text.last_spec_capture() else {
            return Ok(None);
        };
        if capture.taps.len() != drafter.target_layer_ids.len() {
            return Ok(None);
        }
        drafter.retain_seqs(ctx.seq_ids);

        let draft_head = self.draft_lm_head.lock().expect("draft lm_head poisoned");
        let lm_head = draft_head.as_ref().unwrap_or_else(|| self.text.lm_head());

        let mut appends = Vec::with_capacity(batch);
        let mut accepted = Vec::with_capacity(batch);
        for (i, seq_id) in ctx.seq_ids.iter().enumerate() {
            let (batch_idx, count) = ctx.target_rows[i];
            let base_len = ctx.base_lens[i];
            let ctx_next = drafter.ctx_next_pos(*seq_id);
            let needed = match ctx_next {
                Some(next) if next <= base_len => {
                    // The committed rows since the last draft are the previous bonus token plus
                    // however many drafts survived verification
                    let stepped = base_len - next;
                    if stepped > 0 {
                        accepted.push(stepped - 1);
                    }
                    stepped
                }
                // Unknown or ahead (sequence restarted): resync from what this forward provides
                _ => count.min(base_len),
            };
            if needed > 0 {
                if needed > count || capture.taps[0].dim(1)? < count {
                    // Rows this forward can't cover (e.g. a prefix-cache hit skipped taps): skip
                    // proposing this round; the next verify refreshes everything.
                    return Ok(None);
                }
                let start_row = count - needed;
                let taps = capture
                    .taps
                    .iter()
                    .map(|t| t.narrow(0, batch_idx, 1)?.narrow(1, start_row, needed))
                    .collect::<Result<Vec<_>>>()?;
                appends.push(crate::speculative::dflash::CtxAppend {
                    seq_id: *seq_id,
                    taps: Tensor::cat(&taps, candle_core::D::Minus1)?,
                    start_pos: base_len - needed,
                });
            }
        }
        drafter.append_ctx_batch(&appends)?;
        let noise = self.dflash_noise_embedding(&drafter, ctx.sampled_tokens, n_predict)?;
        let hidden = drafter.draft_hidden_batch(ctx.seq_ids, &noise, ctx.base_lens)?;
        // The lm_head weights are read once for the whole batch; drafts are chosen per sequence
        let greedy = ctx.sequences.iter().all(|seq| seq.sampler().is_argmax());
        let finished = drafter.finish_drafts(&hidden, ctx.sampled_tokens, lm_head, greedy)?;
        drafter.adaptive_observe(&accepted, max_n, batch);
        let proposals = finished
            .into_iter()
            .map(|(tokens, logits)| SpeculativeProposal::with_logits(tokens, logits))
            .collect();
        Ok(Some(SpeculativeProposalBatch::new(proposals)))
    }

    fn dflash_prefill(&self, ctx: SpeculativePrefillCtx<'_>) -> Result<()> {
        let drafter = self
            .dflash
            .lock()
            .expect("dflash poisoned")
            .clone()
            .ok_or_else(|| candle_core::Error::msg("DFlash prefill without a drafter"))?;
        let Some(capture) = self.text.last_full_capture() else {
            return Ok(());
        };
        if capture.taps.len() != drafter.target_layer_ids.len() {
            return Ok(());
        }
        let mut appends = Vec::with_capacity(ctx.seq_ids.len());
        for (i, seq_id) in ctx.seq_ids.iter().enumerate() {
            let batch_idx = ctx.batch_indices[i];
            let (start, end) = ctx.chunk_ranges[i];
            if end <= start || capture.taps[0].dim(1)? < end - start {
                continue;
            }
            let rows = end - start;
            let taps = capture
                .taps
                .iter()
                .map(|t| t.narrow(0, batch_idx, 1)?.narrow(1, 0, rows))
                .collect::<Result<Vec<_>>>()?;
            appends.push(crate::speculative::dflash::CtxAppend {
                seq_id: *seq_id,
                taps: Tensor::cat(&taps, candle_core::D::Minus1)?,
                start_pos: start,
            });
        }
        drafter.append_ctx_batch(&appends)
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
            return self.drafter_prefill_chunk(head, &ctx, target, &capture, &kv_cache);
        }
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
            self.text.set_dflash_tap_layers(Vec::new());
            *self.dflash.lock().expect("dflash poisoned") = None;
            return Ok(None);
        };
        if !config.is_builtin() {
            return self.attach_dflash(config);
        }
        if self.text.mtp.is_none() {
            candle_core::bail!(
                "The built-in MTP head was not loaded; pass `--mtp` when loading the model."
            );
        }
        let default_n_predict = if self.text.cfg.hidden_size >= MTP_LARGE_HIDDEN_SIZE {
            DEFAULT_MTP_N_PREDICT_LARGE
        } else {
            DEFAULT_MTP_N_PREDICT
        };
        let n_predict = config.n_predict.unwrap_or(default_n_predict);
        if n_predict == 0 {
            candle_core::bail!("MTP n_predict must be at least 1.");
        }
        self.mtp_n_predict.store(n_predict, Ordering::Relaxed);
        self.text.set_store_spec_hidden(true);
        // The promoted (sensitive) lm_head is read once per draft; a base-type copy makes the
        // drafter cheaper without touching what the target verifies with
        if let Some(ty) = config.draft_lm_head_isq {
            let head = self.text.lm_head().clone().apply_isq(
                Some(ty),
                self.text.device.clone(),
                &std::sync::atomic::AtomicUsize::new(0),
                None,
                mistralrs_quant::QuantizeOntoGuard::new(),
            )?;
            *self.draft_lm_head.lock().expect("draft lm_head poisoned") = Some(head);
        } else {
            *self.draft_lm_head.lock().expect("draft lm_head poisoned") = None;
        }
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
        if n == 0 {
            return None;
        }
        if let Some(drafter) = self.dflash.lock().expect("dflash poisoned").as_ref() {
            return Some(drafter.current_n(n));
        }
        Some(n)
    }

    fn speculative_proposal_len_options(&self) -> Vec<usize> {
        let n = self.mtp_n_predict();
        if n == 0 {
            return Vec::new();
        }
        if let Some(drafter) = self.dflash.lock().expect("dflash poisoned").as_ref() {
            let tiers = drafter.adaptive_depths(n);
            if !tiers.is_empty() {
                return tiers;
            }
        }
        vec![n]
    }

    fn speculative_propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        if self.dflash.lock().expect("dflash poisoned").is_some() {
            return self.dflash_propose(ctx);
        }
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
        if self.dflash.lock().expect("dflash poisoned").is_some() {
            return self.dflash_prefill(ctx);
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

    fn take_speculative_graph_state(&self) -> Option<Box<dyn SpeculativeGraphState>> {
        self.text
            .take_spec_graph_state()
            .map(|state| Box::new(state) as Box<dyn SpeculativeGraphState>)
    }

    fn install_speculative_graph_state(&self, state: &dyn SpeculativeGraphState) -> Result<()> {
        let state = state
            .as_any()
            .downcast_ref::<SpecGraphState>()
            .ok_or_else(|| {
                candle_core::Error::msg("foreign speculative graph state for Qwen3.5")
            })?;
        self.text.install_spec_graph_state(state);
        Ok(())
    }
}
