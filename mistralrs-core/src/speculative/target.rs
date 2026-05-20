use candle_core::{Result, Tensor};

use super::{SpeculativeConfig, SpeculativeProposal, SpeculativeProposeCtx};

pub trait SpeculativeTargetMixin {
    fn attach_speculative(&mut self, _config: SpeculativeConfig) -> Result<()> {
        candle_core::bail!("This model does not support speculative decoding.")
    }

    fn has_speculative_proposer(&self) -> bool {
        false
    }

    fn speculative_propose(
        &mut self,
        _ctx: SpeculativeProposeCtx<'_>,
    ) -> Result<Option<SpeculativeProposal>> {
        Ok(None)
    }

    fn speculative_target_hidden(&self, _row: usize) -> Result<Option<Tensor>> {
        Ok(None)
    }
}
