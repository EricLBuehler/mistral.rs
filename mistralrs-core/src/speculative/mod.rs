pub mod cache;
pub mod config;
pub mod dflash;
pub mod driver;
pub mod logging;
pub(crate) mod paged_rows;
pub mod policy;
pub mod proposer;
pub(crate) mod staging;
pub mod target;
pub mod verifier;

#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use crate::cuda::speculative_rejection::CudaSparseRejectionWorkspace;
pub use config::{
    reserve_external_mtp_memory, reserve_external_mtp_memory_with_runtime, MtpConfig,
    MtpDraftSamplingMethod, MtpRuntimeConfig, SpeculativeConfig,
};
pub use dflash::DFlashDraftModel;
pub use logging::{SpeculativeAttachInfo, SpeculativeAttachKind};
pub use policy::{SpeculativeBatchObservation, SpeculativeBatchPlan, SpeculativeGraphPlan};
pub use proposer::{
    SparseSpeculativeProbs, SpeculativeCommitRow, SpeculativeKvCache, SpeculativePrefillCtx,
    SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposalDistribution,
    SpeculativeProposeBatchCtx, SpeculativeProposePreparation, SpeculativeProposePrepareCtx,
    SpeculativeProposer, SpeculativeTokens, TargetAttentionInputs, TargetTokenEmbedder,
};
pub use target::{
    SpeculativeGraphState, SpeculativePrefixCheckpointPolicy, SpeculativePrefixReplay,
    SpeculativeTargetMixin,
};

#[cfg(test)]
mod tests {
    use super::{MtpConfig, MtpDraftSamplingMethod};

    #[test]
    fn mtp_config_supports_public_struct_literals() {
        let config = MtpConfig {
            model: Some("assistant".to_string()),
            n_predict: Some(3),
            draft_sampling_method: MtpDraftSamplingMethod::Probabilistic,
            draft_lm_head_isq: None,
        };

        assert_eq!(config.n_predict, Some(3));
    }
}
