use std::sync::Arc;

use candle_core::{Result, Tensor};

use crate::prefix_cacher::PagedAuxiliaryPrefixState;

use super::{
    logging::log_attach, SpeculativeAttachInfo, SpeculativeBatchObservation, SpeculativeBatchPlan,
    SpeculativeCommitRow, SpeculativeConfig, SpeculativeGraphPlan, SpeculativePrefillCtx,
    SpeculativeProposalBatch, SpeculativeProposeBatchCtx, SpeculativeProposePreparation,
    SpeculativeProposePrepareCtx,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpeculativePrefixReplay {
    #[default]
    NotRequired,
    Suffix(usize),
    Full,
}

impl SpeculativePrefixReplay {
    pub fn replay_tokens(self, cached_tokens: usize) -> usize {
        match self {
            Self::NotRequired => 0,
            Self::Suffix(tokens) => tokens.min(cached_tokens),
            Self::Full => cached_tokens,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeculativePrefixCheckpointPolicy {
    fallback_replay: SpeculativePrefixReplay,
    text_auxiliary_state: bool,
}

impl SpeculativePrefixCheckpointPolicy {
    pub fn new(fallback_replay: SpeculativePrefixReplay, text_auxiliary_state: bool) -> Self {
        Self {
            fallback_replay,
            text_auxiliary_state,
        }
    }

    pub fn replay_for(self, modality_signature: u8) -> SpeculativePrefixReplay {
        if self.uses_auxiliary_state(modality_signature) {
            SpeculativePrefixReplay::NotRequired
        } else {
            self.fallback_replay
        }
    }

    pub fn fallback_replay(self) -> SpeculativePrefixReplay {
        self.fallback_replay
    }

    pub fn uses_auxiliary_state(self, modality_signature: u8) -> bool {
        self.text_auxiliary_state && modality_signature == 0
    }
}

pub(crate) fn clamp_speculative_prefix_cache_hit(
    cached_tokens: usize,
    block_size: usize,
    replay: SpeculativePrefixReplay,
) -> usize {
    let retained = match replay {
        SpeculativePrefixReplay::NotRequired => return cached_tokens,
        SpeculativePrefixReplay::Suffix(tokens) => cached_tokens.saturating_sub(tokens),
        SpeculativePrefixReplay::Full => 0,
    };
    retained - retained % block_size
}

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

    #[doc(hidden)]
    fn attach_speculative_with_runtime(
        &mut self,
        config: SpeculativeConfig,
        _runtime: super::MtpRuntimeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        self.attach_speculative(config)
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

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        false
    }

    fn speculative_prefix_replay(&self) -> SpeculativePrefixReplay {
        SpeculativePrefixReplay::NotRequired
    }

    fn supports_paged_auxiliary_prefix_state(&self) -> bool {
        false
    }

    fn capture_paged_auxiliary_prefix_state(
        &mut self,
        _sequence_id: usize,
        _cached_tokens: usize,
    ) -> Result<Option<Arc<dyn PagedAuxiliaryPrefixState>>> {
        Ok(None)
    }

    fn restore_paged_auxiliary_prefix_state(
        &mut self,
        _sequence_id: usize,
        _cached_tokens: usize,
        _state: &dyn PagedAuxiliaryPrefixState,
    ) -> Result<()> {
        candle_core::bail!("This model does not support auxiliary paged prefix state.")
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

    fn precapture_speculative_cuda_graphs(&self) -> Result<()> {
        Ok(())
    }

    fn evict_speculative_cuda_graphs(&self, _max_entries: usize) -> usize {
        0
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

    fn speculative_prepare_propose(
        &mut self,
        _ctx: SpeculativeProposePrepareCtx<'_>,
    ) -> Result<Option<Box<dyn SpeculativeProposePreparation>>> {
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

#[cfg(test)]
mod tests {
    use super::{
        clamp_speculative_prefix_cache_hit, SpeculativePrefixReplay, SpeculativeTargetMixin,
    };

    struct NoSpeculativeProposer;

    impl SpeculativeTargetMixin for NoSpeculativeProposer {}

    #[test]
    fn prefix_replay_clamp_preserves_block_alignment() {
        assert_eq!(
            clamp_speculative_prefix_cache_hit(4096, 32, SpeculativePrefixReplay::NotRequired),
            4096
        );
        assert_eq!(
            clamp_speculative_prefix_cache_hit(4096, 32, SpeculativePrefixReplay::Suffix(2048)),
            2048
        );
        assert_eq!(
            clamp_speculative_prefix_cache_hit(4096, 32, SpeculativePrefixReplay::Suffix(2049)),
            2016
        );
        assert_eq!(
            clamp_speculative_prefix_cache_hit(1024, 32, SpeculativePrefixReplay::Suffix(2048)),
            0
        );
        assert_eq!(
            clamp_speculative_prefix_cache_hit(4096, 32, SpeculativePrefixReplay::Full),
            0
        );
    }

    #[test]
    fn models_without_a_proposer_have_no_graphs_to_evict() {
        assert_eq!(
            NoSpeculativeProposer.evict_speculative_cuda_graphs(usize::MAX),
            0
        );
    }
}
