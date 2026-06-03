#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Shared Gated Delta Net (GDN) implementation for hybrid models.

mod backend;
mod cache;
mod config;
mod layer;
mod norm;
mod projection;
mod weights;

pub use cache::GdnLayerCache;
pub use config::GdnConfig;
pub use layer::GatedDeltaNet;
pub use weights::GdnWeightMode;
