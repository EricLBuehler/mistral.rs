//! MTP speculative decoding for Qwen3.5 / Qwen3.8 using the checkpoint's built-in head.

use std::sync::{atomic::Ordering, Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor};

use crate::speculative::{
    dflash::{self, host::DFlashHost, DFlashDraftModel, DFlashLoadTarget},
    MtpRuntimeConfig, SpeculativeAttachInfo, SpeculativeBatchPlan, SpeculativeCommitRow,
    SpeculativeConfig, SpeculativeGraphPlan, SpeculativeGraphState, SpeculativePrefillCtx,
    SpeculativePrefixReplay, SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
    SpeculativeProposePreparation, SpeculativeProposePrepareCtx, SpeculativeTargetMixin,
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

    fn dflash(&self) -> Option<Arc<DFlashDraftModel>> {
        self.dflash.lock().expect("dflash poisoned").clone()
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
        let n_predict = dflash::host::resolve_n_predict(config.n_predict, block, checkpoint_lanes)?;
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
}

impl DFlashHost for Qwen3_5Model {
    fn dflash_device(&self) -> &Device {
        &self.text.device
    }

    fn dflash_embed_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        self.text.embed_tokens(tokens)
    }

    fn dflash_token_embedding(&self) -> &Arc<dyn mistralrs_quant::QuantMethod> {
        self.text.token_embedding()
    }

    fn dflash_spec_capture(&self) -> Option<SpecCapture> {
        self.text.last_spec_capture()
    }

    fn dflash_full_capture(&self) -> Option<SpecCapture> {
        self.text.last_full_capture()
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
                let depths = depths_up_to(&crate::speculative::dflash::AUTO_DEPTHS, n);
                return auto_depth_graph_plans(&depths, n);
            }
            return drafter.graph_plans(n);
        }
        if !self.mtp_auto_depth.load(Ordering::Relaxed) {
            return vec![SpeculativeGraphPlan::new(n, None)];
        }
        auto_depth_graph_plans(&AUTO_DEPTHS, self.mtp_default_depth())
    }

    #[cfg(cuda_flash_attn)]
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
        if let Some(drafter) = self.dflash() {
            let draft_head = self.draft_lm_head.lock().expect("draft lm_head poisoned");
            let lm_head = draft_head.as_ref().unwrap_or_else(|| self.text.lm_head());
            return dflash::host::propose(self, &drafter, lm_head, self.mtp_n_predict(), ctx);
        }
        self.mtp_proposer.propose(self, ctx, self.mtp_n_predict())
    }

    fn speculative_prepare_propose(
        &mut self,
        ctx: SpeculativeProposePrepareCtx<'_>,
    ) -> Result<Option<Box<dyn SpeculativeProposePreparation>>> {
        if let Some(drafter) = self.dflash() {
            return dflash::host::prepare_propose(self, &drafter, ctx);
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
        if let Some(drafter) = self.dflash() {
            return dflash::host::prefill(self, &drafter, ctx);
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
    use super::{capture_view, SpecCapture};
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
}
