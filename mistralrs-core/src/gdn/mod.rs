#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Shared Gated Delta Net (GDN) implementation for hybrid models.

mod backend;
mod cache;
mod config;
mod layer;
mod norm;
mod packed;
mod projection;
mod weights;

pub use cache::GdnLayerCache;
pub use config::{GdnConfig, GdnStateDType, GdnVHeadLayout, GDN_V_HEAD_LAYOUT_CONFIG_KEY};
pub use layer::{GatedDeltaNet, GdnForwardStash};
pub(crate) use packed::try_forward_uniform_packed_gdn;
pub use weights::GdnInputProjectionKind;
