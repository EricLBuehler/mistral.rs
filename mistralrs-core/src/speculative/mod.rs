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

pub use config::{reserve_external_mtp_memory, MtpConfig, SpeculativeConfig};
pub use dflash::DFlashDraftModel;
pub use logging::{SpeculativeAttachInfo, SpeculativeAttachKind};
pub use policy::{SpeculativeBatchObservation, SpeculativeBatchPlan, SpeculativeGraphPlan};
pub use proposer::{
    SpeculativeCommitRow, SpeculativeKvCache, SpeculativePrefillCtx, SpeculativeProposal,
    SpeculativeProposalBatch, SpeculativeProposeBatchCtx, SpeculativeProposer,
    TargetAttentionInputs, TargetTokenEmbedder,
};
pub use target::{SpeculativeGraphState, SpeculativeTargetMixin};
