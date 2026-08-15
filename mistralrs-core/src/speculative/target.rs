use candle_core::{Result, Tensor};

use super::{
    logging::log_attach, SpeculativeAttachInfo, SpeculativeCommitRow, SpeculativeConfig,
    SpeculativePrefillCtx, SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
};

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

    fn speculative_proposal_len(&self) -> Option<usize> {
        None
    }

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
}
