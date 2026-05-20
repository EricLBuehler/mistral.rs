//! Runtime state attached to a pipeline to drive speculative decoding.
//!
//! The CLI/SDK loads the MTP head + builds the proposer, then attaches an
//! [`MtpRuntime`] to the pipeline via `Pipeline::attach_mtp`. From then on the
//! pipeline's `try_speculative_completion` hook (called from the engine's step
//! loop) runs the propose -> verify -> commit cycle.

use std::sync::{Arc, Mutex};

use crate::speculative::mtp::Gemma4MtpProposer;

pub struct MtpRuntime {
    pub proposer: Arc<Mutex<Gemma4MtpProposer>>,
    pub n_predict: usize,
    /// Index (within the target's layer_types) of the last `sliding_attention`
    /// layer that owns its own KV. The MTP attention layers tagged sliding
    /// attend over this layer's K, V.
    pub donor_sliding_layer: usize,
    /// Index of the last `full_attention` layer that owns its own KV.
    pub donor_full_layer: usize,
}

impl MtpRuntime {
    pub fn new(
        proposer: Gemma4MtpProposer,
        n_predict: usize,
        donor_sliding_layer: usize,
        donor_full_layer: usize,
    ) -> Self {
        Self {
            proposer: Arc::new(Mutex::new(proposer)),
            n_predict,
            donor_sliding_layer,
            donor_full_layer,
        }
    }
}
