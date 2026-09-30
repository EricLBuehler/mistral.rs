//! MTP speculative decoding for Qwen3.5 / Qwen3.8 using the checkpoint's built-in head.

use std::sync::{atomic::Ordering, Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use rand::Rng;

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
use crate::pipeline::cuda_graph::{CudaGraphComponent, CudaGraphEvent, CudaGraphEventGuard};
use crate::speculative::{
    dflash::{
        CtxAppend, DFlashDraftModel, DFlashGraphProposalInputs, DFlashLoadTarget,
        DFlashPreparedContext, DFlashProposalBatch, DFlashSamplingInputs,
    },
    MtpRuntimeConfig, SpeculativeAttachInfo, SpeculativeBatchPlan, SpeculativeCommitRow,
    SpeculativeConfig, SpeculativeGraphPlan, SpeculativeGraphState, SpeculativePrefillCtx,
    SpeculativePrefixReplay, SpeculativeProposal, SpeculativeProposalBatch,
    SpeculativeProposeBatchCtx, SpeculativeProposePreparation, SpeculativeProposePrepareCtx,
    SpeculativeTapRouting, SpeculativeTargetMixin,
};

use super::{mtp::Qwen3_5MtpHead, Qwen3_5Model};
use crate::speculative::{
    autotuner::{auto_depth_graph_plans, depths_up_to, AUTO_DEPTHS, AUTO_MAX_DEPTH},
    builtin_mtp::{
        capture_view, BuiltinMtpHost, MtpAttentionInputs, MtpDraftOutput, BUILTIN_MTP_PREFIX_REPLAY,
    },
    hybrid_state::{SpecCapture, SpecGraphState},
};

/// vLLM's documented setting for these single-layer MTP heads.
pub const DEFAULT_MTP_N_PREDICT: usize = 2;
// On larger models the verify forward dominates the step, so deeper drafting pays; measured on
// Qwen3.8-27B: n=3 beats n=2 by ~5% while n=4 over-drafts (acceptance falls under 50%).
pub const DEFAULT_MTP_N_PREDICT_LARGE: usize = 3;
const MTP_LARGE_HIDDEN_SIZE: usize = 4096;
// Verify cost grows with block width and quantized targets accept shorter blocks anyway; deeper
// drafting stays available via --mtp-n-predict.
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
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
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

fn resolve_dflash_n_predict(
    requested: Option<usize>,
    block_size: usize,
    checkpoint_lanes: usize,
) -> Result<usize> {
    let max_drafts = block_size
        .checked_sub(1)
        .filter(|max_drafts| *max_drafts > 0)
        .ok_or_else(|| candle_core::Error::msg("DFlash block size must be at least 2"))?;
    let configured =
        requested.unwrap_or(max_drafts.min(crate::speculative::dflash::DEFAULT_MAX_DRAFTS));
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

impl Qwen3_5Model {
    fn mtp_n_predict(&self) -> usize {
        self.mtp_n_predict.load(Ordering::Relaxed)
    }

    fn mtp_default_depth(&self) -> usize {
        if self.text.cfg.hidden_size >= MTP_LARGE_HIDDEN_SIZE {
            DEFAULT_MTP_N_PREDICT_LARGE
        } else {
            DEFAULT_MTP_N_PREDICT
        }
    }

    fn mtp_head(&self) -> Result<&Qwen3_5MtpHead> {
        self.text
            .mtp
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("Qwen3.5 MTP head is not loaded"))
    }

    fn attach_dflash(
        &mut self,
        config: crate::speculative::MtpConfig,
        runtime: MtpRuntimeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        let mut drafter = DFlashDraftModel::load(
            &config,
            DFlashLoadTarget {
                num_layers: self.text.layer_types.len(),
                hidden_size: self.text.cfg.hidden_size,
                yarn_rope_config: self.text.yarn_rope_config.as_ref(),
                device: &self.text.device,
                dtype: self.text.dtype,
            },
            false,
        )?;
        let block = drafter.block_size();
        let checkpoint_lanes = self.text.cache.hybrid().checkpoint_lanes();
        let n_predict = resolve_dflash_n_predict(config.n_predict, block, checkpoint_lanes)?;
        let sequence_capacity = self.text.cache.hybrid().recurrent_capacity();
        let windowed_kv =
            drafter.enable_windowed_kv(sequence_capacity, runtime.prefix_cache_capacity())?;
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
        let adaptive = config.n_predict.is_none()
            && windowed_kv
            && crate::speculative::dflash::dflash_adaptive_requested();
        let max_live_sequences =
            sequence_capacity.saturating_sub(crate::pipeline::RECURRENT_GRAPH_PAD_SLOTS);
        let adaptive = adaptive && drafter.enable_adaptive(n_predict, max_live_sequences);
        let autotuned = config.n_predict.is_none() && !adaptive;
        self.mtp_auto_depth.store(autotuned, Ordering::Relaxed);
        let kind = if drafter.has_selector() {
            "DFlash2"
        } else {
            "DFlash"
        };
        let draft_sampling = match drafter.draft_sampling_method() {
            crate::speculative::MtpDraftSamplingMethod::Auto => {
                unreachable!("DFlash draft sampling is resolved during loading")
            }
            crate::speculative::MtpDraftSamplingMethod::Greedy => "greedy draft",
            crate::speculative::MtpDraftSamplingMethod::Probabilistic => "probabilistic draft",
        };
        let depth = if adaptive {
            format!("batch-adaptive depth <= {n_predict}")
        } else if autotuned {
            format!("autotuned depth <= {n_predict}")
        } else {
            format!("depth {n_predict}")
        };
        let name = format!(
            "{kind} `{}` (block {block}, {depth}, {draft_sampling}, taps {:?})",
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

    fn prepare_dflash_propose(
        &self,
        ctx: SpeculativeProposePrepareCtx<'_>,
    ) -> Result<Option<Box<dyn SpeculativeProposePreparation>>> {
        let drafter = self
            .dflash
            .lock()
            .expect("dflash poisoned")
            .clone()
            .ok_or_else(|| candle_core::Error::msg("DFlash prepare without a drafter"))?;
        if ctx.seq_ids.is_empty() {
            return Ok(None);
        }
        if ctx.seq_ids.len() != ctx.base_lens.len()
            || ctx.seq_ids.len() != ctx.target_rows.len()
            || drafter.has_dormant_seq(ctx.seq_ids)
        {
            return Ok(None);
        }
        let Some(capture) = self.text.last_spec_capture() else {
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
        let n_predict = ctx.proposal_len;
        if n_predict == 0 {
            return Ok(None);
        }
        if n_predict > max_n {
            candle_core::bail!(
                "DFlash proposal length {n_predict} exceeds configured maximum {max_n}"
            );
        }
        let Some(capture) = self.text.last_spec_capture() else {
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
                        flat_row_indices
                            .push(u32::try_from(row).map_err(candle_core::Error::wrap)?);
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
        let draft_head = self.draft_lm_head.lock().expect("draft lm_head poisoned");
        let lm_head = draft_head.as_ref().unwrap_or_else(|| self.text.lm_head());
        let graph_proposals = drafter.proposals_cuda_graph(&DFlashGraphProposalInputs {
            seq_ids: ctx.seq_ids,
            anchors: ctx.sampled_tokens,
            start_positions: ctx.base_lens,
            n_predict,
            sampling,
            token_embedding: self.text.token_embedding(),
            lm_head,
        })?;
        if let Some(proposals) = graph_proposals {
            return dflash_speculative_batch(proposals).map(Some);
        }
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::DFlash, CudaGraphEvent::EagerFallback);
        let fallback_result = (|| {
            let noise = self.dflash_noise_embedding(&drafter, ctx.sampled_tokens, n_predict)?;
            let hidden = drafter.draft_hidden_batch(ctx.seq_ids, &noise, ctx.base_lens)?;
            dflash_speculative_batch(drafter.finish_proposals(
                &hidden,
                ctx.sampled_tokens,
                sampling,
                lm_head,
            )?)
            .map(Some)
        })();
        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
        if fallback_result.is_ok() {
            graph_event.success();
        }
        fallback_result
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
            appends.push(crate::speculative::dflash::CtxAppend {
                seq_id: *seq_id,
                rows: span.rows(),
                start_pos: start,
            });
        }
        let flat_row_indices = routing.flat_row_indices()?;
        drafter.append_ctx_batch(&capture.taps, flat_row_indices, &appends)
    }
}

impl BuiltinMtpHost for Qwen3_5Model {
    fn mtp_device(&self) -> &Device {
        self.mtp_head().expect("MTP head is loaded").device()
    }

    fn mtp_dtype(&self) -> DType {
        self.mtp_head().expect("MTP head is loaded").dtype()
    }

    fn mtp_kv_layer_idx(&self) -> usize {
        self.mtp_head().expect("MTP head is loaded").kv_layer_idx()
    }

    fn mtp_embed_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        self.text.embed_tokens(tokens)
    }

    fn mtp_forward(
        &self,
        input_embeds: &Tensor,
        target_hidden: &Tensor,
        positions: &Tensor,
        attention: MtpAttentionInputs<'_>,
    ) -> Result<MtpDraftOutput> {
        let logits_hidden =
            self.mtp_head()?
                .forward(input_embeds, target_hidden, positions, attention)?;
        Ok(MtpDraftOutput {
            logits_hidden,
            chain_hidden: None,
        })
    }

    fn mtp_draft_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let draft_head = self.draft_lm_head.lock().expect("draft lm_head poisoned");
        let head = draft_head.as_ref().unwrap_or_else(|| self.text.lm_head());
        head.forward(hidden)?.squeeze(0)
    }

    fn mtp_spec_capture(&self) -> Option<SpecCapture> {
        self.text.last_spec_capture()
    }

    fn mtp_full_capture(&self) -> Option<SpecCapture> {
        self.text.last_full_capture()
    }
}

impl SpeculativeTargetMixin for Qwen3_5Model {
    fn attach_speculative(
        &mut self,
        config: SpeculativeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        self.attach_speculative_with_runtime(config, MtpRuntimeConfig::default())
    }

    fn attach_speculative_with_runtime(
        &mut self,
        config: SpeculativeConfig,
        runtime: MtpRuntimeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        self.mtp_auto_depth.store(false, Ordering::Relaxed);
        let SpeculativeConfig::Mtp(config) = config else {
            self.mtp_n_predict.store(0, Ordering::Relaxed);
            self.text.set_store_spec_hidden(false);
            self.text.set_dflash_tap_layers(Vec::new());
            *self.dflash.lock().expect("dflash poisoned") = None;
            return Ok(None);
        };
        if !config.is_builtin() {
            return self.attach_dflash(config, runtime);
        }
        if self.text.mtp.is_none() {
            candle_core::bail!(
                "The built-in MTP head was not loaded; pass `--mtp` when loading the model."
            );
        }
        let n_predict = config.n_predict.unwrap_or(AUTO_MAX_DEPTH);
        if n_predict == 0 {
            candle_core::bail!("MTP n_predict must be at least 1.");
        }
        self.mtp_n_predict.store(n_predict, Ordering::Relaxed);
        self.mtp_auto_depth
            .store(config.n_predict.is_none(), Ordering::Relaxed);
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

    fn supports_recurrent_speculative_checkpoints(&self) -> bool {
        self.text.supports_recurrent_speculative_checkpoints()
    }

    fn supports_recurrent_speculative_transitions(&self) -> bool {
        self.text.supports_recurrent_speculative_transitions()
    }

    fn reserve_recurrent_speculative_transition_storage(&self) -> Result<bool> {
        self.text.reserve_recurrent_transition_storage()
    }

    fn reserve_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        if self.has_speculative_proposer() {
            Ok(false)
        } else {
            self.text.reserve_recurrent_decode_deferred_storage()
        }
    }

    fn disable_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        self.text.disable_recurrent_decode_deferred_storage()
    }

    fn apply_recurrent_speculative_transitions_for_current_batch(&self) -> Result<bool> {
        self.text.apply_current_recurrent_transitions()
    }

    fn flush_recurrent_state_for_current_batch(&self) -> Result<()> {
        self.text.flush_current_recurrent_state()
    }

    fn flush_recurrent_speculative_transitions(&self, seq_ids: &[usize]) -> Result<()> {
        self.text.flush_recurrent_transitions_for_sequences(seq_ids)
    }

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        self.dflash.lock().expect("dflash poisoned").is_some()
    }

    fn supports_speculative_packed_prefill(&self) -> bool {
        self.dflash.lock().expect("dflash poisoned").is_some()
    }

    fn speculative_prefix_replay(&self) -> SpeculativePrefixReplay {
        self.dflash
            .lock()
            .expect("dflash poisoned")
            .as_ref()
            .map_or_else(
                || {
                    if self.mtp_n_predict() > 0 {
                        BUILTIN_MTP_PREFIX_REPLAY
                    } else {
                        SpeculativePrefixReplay::NotRequired
                    }
                },
                |drafter| drafter.prefix_replay(),
            )
    }

    fn supports_paged_auxiliary_prefix_state(&self) -> bool {
        self.dflash
            .lock()
            .expect("dflash poisoned")
            .as_ref()
            .is_some_and(|drafter| drafter.supports_paged_auxiliary_prefix_state())
    }

    fn capture_paged_auxiliary_prefix_state(
        &mut self,
        sequence_id: usize,
        cached_tokens: usize,
    ) -> Result<Option<Arc<dyn crate::prefix_cacher::PagedAuxiliaryPrefixState>>> {
        let Some(drafter) = self
            .dflash
            .lock()
            .expect("dflash poisoned")
            .as_ref()
            .cloned()
        else {
            return Ok(None);
        };
        drafter.capture_paged_auxiliary_prefix_state(sequence_id, cached_tokens)
    }

    fn restore_paged_auxiliary_prefix_state(
        &mut self,
        sequence_id: usize,
        cached_tokens: usize,
        state: &dyn crate::prefix_cacher::PagedAuxiliaryPrefixState,
    ) -> Result<()> {
        let drafter = self
            .dflash
            .lock()
            .expect("dflash poisoned")
            .as_ref()
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("DFlash prefix restore without a drafter"))?;
        drafter.restore_paged_auxiliary_prefix_state(sequence_id, cached_tokens, state)
    }

    fn speculative_plan(&self, batch_size: usize) -> Option<SpeculativeBatchPlan> {
        let n = self.mtp_n_predict();
        if n == 0 {
            return None;
        }
        if let Some(drafter) = self.dflash.lock().expect("dflash poisoned").as_ref() {
            return Some(
                SpeculativeBatchPlan::new(drafter.plan_n(n, batch_size)).without_target_hiddens(),
            );
        }
        Some(SpeculativeBatchPlan::new(n))
    }

    fn speculative_depth_candidates(&self) -> Vec<usize> {
        if !self.mtp_auto_depth.load(Ordering::Relaxed) {
            return Vec::new();
        }
        if self.dflash.lock().expect("dflash poisoned").is_none() {
            return AUTO_DEPTHS.to_vec();
        }
        depths_up_to(
            &crate::speculative::dflash::AUTO_DEPTHS,
            self.mtp_n_predict(),
        )
    }

    fn speculative_graph_plans(&self) -> Vec<SpeculativeGraphPlan> {
        let n = self.mtp_n_predict();
        if n == 0 {
            return Vec::new();
        }
        if let Some(drafter) = self.dflash.lock().expect("dflash poisoned").as_ref() {
            if self.mtp_auto_depth.load(Ordering::Relaxed) {
                return auto_depth_graph_plans(&self.speculative_depth_candidates(), n);
            }
            return drafter.graph_plans(n);
        }
        if !self.mtp_auto_depth.load(Ordering::Relaxed) {
            return vec![SpeculativeGraphPlan::new(n, None)];
        }
        auto_depth_graph_plans(&AUTO_DEPTHS, self.mtp_default_depth())
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    fn precapture_speculative_cuda_graphs(&self) -> Result<()> {
        let Some(drafter) = self.dflash.lock().expect("dflash poisoned").clone() else {
            return Ok(());
        };
        let draft_head = self.draft_lm_head.lock().expect("draft lm_head poisoned");
        let lm_head = draft_head.as_ref().unwrap_or_else(|| self.text.lm_head());
        drafter.precapture_cuda_graphs(self.mtp_n_predict(), self.text.token_embedding(), lm_head)
    }

    fn evict_speculative_cuda_graphs(&self, max_entries: usize) -> usize {
        let Some(drafter) = self.dflash.lock().expect("dflash poisoned").clone() else {
            return 0;
        };
        drafter.evict_cuda_graphs_lru(max_entries)
    }

    fn speculative_bypass(&mut self, seq_ids: &[usize]) -> Result<()> {
        let flush_result = self.text.flush_recurrent_transitions_for_sequences(seq_ids);
        if let Some(drafter) = self.dflash.lock().expect("dflash poisoned").as_ref() {
            drafter.mark_seqs_dormant(seq_ids);
        }
        flush_result
    }

    fn release_speculative_sequences(&mut self, seq_ids: &[usize]) -> Result<()> {
        let flush_result = self.text.flush_recurrent_transitions_for_sequences(seq_ids);
        self.mtp_proposer.release_sequences(seq_ids);
        if let Some(drafter) = self.dflash.lock().expect("dflash poisoned").as_ref() {
            drafter.release_seqs(seq_ids);
        }
        flush_result
    }

    fn speculative_propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        if self.dflash.lock().expect("dflash poisoned").is_some() {
            return self.dflash_propose(ctx);
        }
        self.mtp_proposer.propose(self, ctx, self.mtp_n_predict())
    }

    fn speculative_prepare_propose(
        &mut self,
        ctx: SpeculativeProposePrepareCtx<'_>,
    ) -> Result<Option<Box<dyn SpeculativeProposePreparation>>> {
        if self.dflash.lock().expect("dflash poisoned").is_some() {
            return self.prepare_dflash_propose(ctx);
        }
        Ok(None)
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
        self.mtp_proposer.prefill(self, ctx)
    }

    fn speculative_commit(&mut self, rows: &[SpeculativeCommitRow]) -> Result<()> {
        let checkpoint_rows = rows
            .iter()
            .map(|row| (row.batch_idx, row.keep_rows))
            .collect::<Vec<_>>();
        if self.text.cache.hybrid().uses_recurrent_transition_log() {
            if !self.text.stage_recurrent_prefixes(rows)? {
                self.text.replay_recurrent_prefixes(&checkpoint_rows)?;
            }
            self.text.clear_gdn_replay_stash();
            return Ok(());
        }
        let checkpointed = {
            let mut cache = self.text.cache.hybrid();
            self.text
                .supports_recurrent_speculative_checkpoints_with_cache(&cache)
                && cache.commit_speculative_rows(&checkpoint_rows)?
        };
        if checkpointed {
            self.text.clear_gdn_replay_stash();
            return Ok(());
        }
        let rejected = rows
            .iter()
            .filter(|row| !row.accepted_all)
            .map(|row| (row.batch_idx, row.keep_rows))
            .collect::<Vec<_>>();
        self.text.replay_recurrent_prefixes(&rejected)?;
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
        self.text.install_spec_graph_state(state)
    }
}

#[cfg(test)]
mod tests {
    use super::{capture_view, resolve_dflash_n_predict, SpecCapture};
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn text_capture_positions_expand_to_equal_mrope_planes() {
        let device = Device::Cpu;
        let capture = SpecCapture {
            hidden: Tensor::zeros((2, 2, 4), DType::F32, &device).unwrap(),
            positions: Tensor::from_vec(vec![3u32, 4, 9, 10], (2, 2), &device).unwrap(),
            taps: Vec::new(),
        };
        let view = capture_view(&capture).unwrap();
        let expected = vec![vec![3u32, 4], vec![9, 10]];
        assert_eq!(
            view.mrope,
            vec![expected.clone(), expected.clone(), expected]
        );
    }

    #[test]
    fn auto_dflash_depth_follows_reserved_checkpoint_lanes() {
        assert_eq!(resolve_dflash_n_predict(None, 8, 8).unwrap(), 7);
        assert_eq!(resolve_dflash_n_predict(None, 8, 4).unwrap(), 3);
    }

    #[test]
    fn replay_sentinel_keeps_configured_or_default_dflash_depth() {
        assert_eq!(resolve_dflash_n_predict(None, 8, 1).unwrap(), 7);
        assert_eq!(resolve_dflash_n_predict(Some(3), 8, 1).unwrap(), 3);
    }

    #[test]
    fn explicit_dflash_depth_must_match_reserved_checkpoint_lanes() {
        assert_eq!(resolve_dflash_n_predict(Some(3), 8, 4).unwrap(), 3);
        let error = resolve_dflash_n_predict(Some(3), 8, 8)
            .expect_err("explicit depth must match the reserved lanes");
        assert!(error.to_string().contains(
            "requested 3 draft tokens require 4 recurrent checkpoint lanes, but 8 lanes were reserved"
        ));
    }

    #[test]
    fn reserved_dflash_depth_must_fit_the_drafter_block() {
        let error = resolve_dflash_n_predict(None, 8, 9)
            .expect_err("reserved lanes must fit the drafter block");
        assert!(error
            .to_string()
            .contains("reserved 9 recurrent checkpoint lanes"));
    }
}
