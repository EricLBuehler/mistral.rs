use candle_core::{Result, Tensor};

use super::{
    logging::log_attach, SpeculativeAttachInfo, SpeculativeBatchObservation, SpeculativeBatchPlan,
    SpeculativeCommitRow, SpeculativeConfig, SpeculativeGraphPlan, SpeculativePrefillCtx,
    SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
};

/// Everything a target forward leaves behind for the proposer/commit (captured hidden states, rollback
/// stashes). A CUDA graph replay never runs the forward, so the pipeline copies these into persistent
/// buffers at capture time and re-installs them after every replay.
pub trait SpeculativeGraphState: Send + Sync {
    /// Device tensors in a fixed order; `with_tensors` rebuilds the same structure around replacements.
    fn tensors(&self) -> Vec<Tensor>;
    fn with_tensors(&self, tensors: Vec<Tensor>) -> Result<Box<dyn SpeculativeGraphState>>;
    /// Build views for the live rows before launching a padded CUDA graph.
    fn for_real_batch(&self, real_batch: usize) -> Result<Box<dyn SpeculativeGraphState>>;
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait SpeculativeTargetMixin {
    fn attach_speculative(
        &mut self,
        config: SpeculativeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        match config {
            SpeculativeConfig::Off => Ok(None),
            _ => candle_core::bail!("This model does not support speculative decoding."),
        }
    }

    fn log_speculative_attach(&self, info: &SpeculativeAttachInfo) {
        log_attach(info);
    }

    fn has_speculative_proposer(&self) -> bool {
        false
    }

    fn supports_recurrent_speculative_checkpoints(&self) -> bool {
        false
    }

    fn speculative_plan(&self, _batch_size: usize) -> Option<SpeculativeBatchPlan> {
        None
    }

    fn speculative_graph_plans(&self) -> Vec<SpeculativeGraphPlan> {
        self.speculative_plan(1)
            .map(|plan| SpeculativeGraphPlan::new(plan.proposal_len, None))
            .into_iter()
            .collect()
    }

    fn speculative_observe(&self, _observation: SpeculativeBatchObservation) {}

    fn speculative_bypass(&mut self, _seq_ids: &[usize]) {}

    fn release_speculative_sequences(&mut self, _seq_ids: &[usize]) {}

    /// Returns `Ok(None)` when speculation is unsupported for the current step.
    /// Return `Err` only for real failures that should stop generation.
    fn speculative_propose(
        &mut self,
        _ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        Ok(None)
    }

    /// Returns `Ok(None)` when the active proposer does not need target hidden state.
    /// Return `Err` only when hidden state was expected but unavailable or invalid.
    fn speculative_target_hiddens(&self, _rows: &[(usize, usize)]) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Called after each prompt chunk so proposers with their own KV cache can process it.
    fn speculative_prefill(&mut self, _ctx: SpeculativePrefillCtx<'_>) -> Result<()> {
        Ok(())
    }

    /// Called once verification decided which rows of the last multi-token step survive, so models
    /// with state that is not a paged KV cache (recurrent layers) can roll rejected rows back.
    fn speculative_commit(&mut self, _rows: &[SpeculativeCommitRow]) -> Result<()> {
        Ok(())
    }

    /// Detach what the last forward left for the proposer. `None` means the model cannot be replayed
    /// through a CUDA graph while a proposer is attached.
    fn take_speculative_graph_state(&self) -> Option<Box<dyn SpeculativeGraphState>> {
        None
    }

    fn install_speculative_graph_state(&self, _state: &dyn SpeculativeGraphState) -> Result<()> {
        Ok(())
    }
}
