use candle_core::{Result, Tensor};

use super::{SpeculativeConfig, SpeculativeProposalBatch, SpeculativeProposeBatchCtx};

pub trait SpeculativeTargetMixin {
    fn attach_speculative(&mut self, config: SpeculativeConfig) -> Result<()> {
        match config {
            SpeculativeConfig::Off => Ok(()),
            _ => candle_core::bail!("This model does not support speculative decoding."),
        }
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
}
