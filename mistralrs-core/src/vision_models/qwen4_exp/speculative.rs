//! MTP speculative decoding for Qwen4-Exp using the checkpoint's built-in head.

use std::sync::atomic::Ordering;

use candle_core::{DType, Device, IndexOp, Result, Tensor};

use super::{mtp::Qwen4ExpMtpHead, Qwen4ExpModel};
use crate::speculative::{
    builtin_mtp::{capture_view, BuiltinMtpHost, MtpAttentionInputs, MtpDraftOutput},
    hybrid_state::{SpecCapture, SpecGraphState},
    MtpRuntimeConfig, SpeculativeAttachInfo, SpeculativeBatchPlan, SpeculativeCommitRow,
    SpeculativeConfig, SpeculativeGraphPlan, SpeculativeGraphState, SpeculativePrefillCtx,
    SpeculativeProposalBatch, SpeculativeProposeBatchCtx, SpeculativeTargetMixin,
};

pub const DEFAULT_MTP_N_PREDICT: usize = 3;

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
        let n_predict = config.n_predict.unwrap_or(DEFAULT_MTP_N_PREDICT);
        if n_predict == 0 {
            candle_core::bail!("MTP n_predict must be at least 1.");
        }
        self.mtp_n_predict.store(n_predict, Ordering::Relaxed);
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

    fn speculative_plan(&self, _batch_size: usize) -> Option<SpeculativeBatchPlan> {
        let n = self.mtp_n_predict();
        (n > 0).then(|| SpeculativeBatchPlan::new(n))
    }

    fn speculative_graph_plans(&self) -> Vec<SpeculativeGraphPlan> {
        let n = self.mtp_n_predict();
        if n == 0 {
            return Vec::new();
        }
        // Wider verify batches take the grouped MoE path, whose per-expert padded workspaces every
        // captured graph would pin; on unified memory those do not fit beside a filled KV pool
        #[cfg(feature = "cuda")]
        let max_batch = Some((crate::moe::GROUPED_PREFILL_MIN_TOKENS - 1) / (1 + n));
        #[cfg(not(feature = "cuda"))]
        let max_batch = None;
        vec![SpeculativeGraphPlan::new(n, max_batch)]
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
        let result = self.text.replay_recurrent_prefixes(&rejected);
        self.text.clear_speculative_stash();
        result
    }
}
