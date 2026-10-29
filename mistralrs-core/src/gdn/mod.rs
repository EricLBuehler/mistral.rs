#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Shared Gated Delta Net (GDN) implementation for hybrid models.

mod backend;
mod cache;
mod config;
mod deferred;
mod layer;
mod norm;
mod packed;
mod projection;
mod weights;

pub use cache::GdnLayerCache;
pub use config::{
    GdnConfig, GdnGateActivation, GdnStateDType, GdnVHeadLayout, GDN_V_HEAD_LAYOUT_CONFIG_KEY,
};
pub(crate) use deferred::reserve_deferred_decode_state;
pub(crate) use layer::GdnForwardContext;
pub use layer::{
    GatedDeltaNet, GdnForwardStash, GdnSpeculativeStash, GdnTransitionCommitConfig,
    GdnTransitionStash,
};
pub(crate) use packed::{try_forward_grouped_packed_gdn, PackedGdnLayout};
pub(crate) use weights::checkpoint_norm_storage;
pub use weights::GdnInputProjectionKind;
