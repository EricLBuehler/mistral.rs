//! Multi-head Latent Attention (MLA) module for efficient decode kernels.
//!
//! This module provides shared infrastructure for MLA-based attention mechanisms
//! used in models like DeepSeek V2/V3 and GLM4 MoE Lite.

mod forward;
mod weights;

pub use forward::{
    mla_cache_forward, mla_decode_forward, should_use_mla_cache, should_use_mla_decode,
};
pub use weights::MlaWeights;
