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
    reserve_external_mtp_memory, MtpConfig, MtpDraftSamplingMethod, SpeculativeConfig,
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
pub use target::{SpeculativeGraphState, SpeculativePrefixReplay, SpeculativeTargetMixin};
