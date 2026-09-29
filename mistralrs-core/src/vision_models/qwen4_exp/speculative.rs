//! MTP speculative decoding for Qwen4-Exp using the checkpoint's built-in head.

use std::sync::atomic::Ordering;

use candle_core::{DType, Device, IndexOp, Result, Tensor};

use super::{mtp::Qwen4ExpMtpHead, Qwen4ExpModel};
use crate::speculative::{
    autotuner::{AUTO_DEPTHS, AUTO_MAX_DEPTH},
    builtin_mtp::{capture_view, BuiltinMtpHost, MtpAttentionInputs, MtpDraftOutput},
    hybrid_state::{SpecCapture, SpecGraphState},
    MtpRuntimeConfig, SpeculativeAttachInfo, SpeculativeBatchPlan, SpeculativeCommitRow,
    SpeculativeConfig, SpeculativeGraphPlan, SpeculativeGraphState, SpeculativePrefillCtx,
    SpeculativeProposalBatch, SpeculativeProposeBatchCtx, SpeculativeTargetMixin,
};

impl Qwen4ExpModel {
    fn mtp_n_predict(&self) -> usize {
        self.mtp_n_predict.load(Ordering::Relaxed)
    }

    fn mtp_head(&self) -> Result<&Qwen4ExpMtpHead> {
        self.text
            .mtp
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("Qwen4-Exp MTP head is not loaded"))
    }
}

impl BuiltinMtpHost for Qwen4ExpModel {
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
        self.mtp_head()?
            .forward(input_embeds, target_hidden, positions, attention)
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

impl SpeculativeTargetMixin for Qwen4ExpModel {
    fn attach_speculative(
        &mut self,
        config: SpeculativeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        self.attach_speculative_with_runtime(config, MtpRuntimeConfig::default())
    }

    fn attach_speculative_with_runtime(
        &mut self,
        config: SpeculativeConfig,
        _runtime: MtpRuntimeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        let SpeculativeConfig::Mtp(config) = config else {
            self.mtp_n_predict.store(0, Ordering::Relaxed);
            self.text.set_store_spec_hidden(false);
            return Ok(None);
        };
        if !config.is_builtin() {
            candle_core::bail!("Qwen4-Exp supports only its built-in MTP head as a drafter.");
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
        // A base-type copy of the promoted lm_head makes drafting cheaper without touching verification
        let draft_head = config
            .draft_lm_head_isq
            .map(|ty| {
                self.text.lm_head().clone().apply_isq(
                    Some(ty),
                    self.text.device.clone(),
                    &std::sync::atomic::AtomicUsize::new(0),
                    None,
                    mistralrs_quant::QuantizeOntoGuard::new(),
                )
            })
            .transpose()?;
        *self.draft_lm_head.lock().expect("draft lm_head poisoned") = draft_head;
        Ok(Some(SpeculativeAttachInfo::mtp(
            "built-in".to_string(),
            n_predict,
        )))
    }

    fn has_speculative_proposer(&self) -> bool {
        self.mtp_n_predict() > 0
    }

    fn supports_recurrent_speculative_transitions(&self) -> bool {
        self.text.supports_recurrent_speculative_transitions()
    }

    fn speculative_verify_mutates_recurrent_state(&self) -> bool {
        true
    }

    fn reserve_recurrent_speculative_transition_storage(&self) -> Result<bool> {
        self.text.reserve_recurrent_transition_storage()
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

    fn speculative_plan(&self, _batch_size: usize) -> Option<SpeculativeBatchPlan> {
        let n = self.mtp_n_predict();
        (n > 0).then(|| SpeculativeBatchPlan::new(n))
    }

    fn speculative_depth_candidates(&self) -> Vec<usize> {
        if self.mtp_auto_depth.load(Ordering::Relaxed) {
            AUTO_DEPTHS.to_vec()
        } else {
            Vec::new()
        }
    }

    fn speculative_graph_plans(&self) -> Vec<SpeculativeGraphPlan> {
        let n = self.mtp_n_predict();
        if n == 0 {
            return Vec::new();
        }
        let depths = match self.speculative_depth_candidates() {
            depths if depths.is_empty() => vec![n],
            depths => depths,
        };
        depths
            .into_iter()
            .map(|depth| {
                // Wider verify batches take the grouped MoE path, whose per-expert padded workspaces every
                // captured graph would pin; on unified memory those do not fit beside a filled KV pool
                #[cfg(feature = "cuda")]
                let max_batch = Some((crate::moe::GROUPED_PREFILL_MIN_TOKENS - 1) / (1 + depth));
                #[cfg(not(feature = "cuda"))]
                let max_batch = None;
                SpeculativeGraphPlan::new(depth, max_batch)
            })
            .collect()
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
                candle_core::Error::msg("foreign speculative graph state for Qwen4-Exp")
            })?;
        self.text.install_spec_graph_state(state)
    }

    fn speculative_propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        self.mtp_proposer.propose(self, ctx, self.mtp_n_predict())
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
        self.mtp_proposer.prefill(self, ctx)
    }

    fn speculative_commit(&mut self, rows: &[SpeculativeCommitRow]) -> Result<()> {
        let rejected = rows
            .iter()
            .filter(|row| !row.accepted_all)
            .map(|row| (row.batch_idx, row.keep_rows))
            .collect::<Vec<_>>();
        let transition_log = self.text.cache.hybrid().uses_recurrent_transition_log();
        let result = match transition_log {
            true => self.text.stage_recurrent_prefixes(rows),
            false => Ok(false),
        }
        .and_then(|staged| match staged {
            true => self.text.rollback_ple_prefixes(&rejected),
            false => self.text.replay_recurrent_prefixes(&rejected),
        });
        self.text.clear_speculative_stash();
        result
    }
}
