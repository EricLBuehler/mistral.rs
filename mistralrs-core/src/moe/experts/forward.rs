use candle_core::{DType, Tensor};

use crate::layers::Activation;

#[derive(Clone, Copy)]
pub(super) enum MoEForwardPhase {
    Prefill,
    Decode,
}

impl MoEForwardPhase {
    pub(super) fn from_shape(seq_len: usize) -> Self {
        if seq_len > 1 {
            Self::Prefill
        } else {
            Self::Decode
        }
    }

    pub(super) fn is_prefill(self) -> bool {
        matches!(self, Self::Prefill)
    }
}

#[derive(Clone, Copy)]
pub(super) struct MoEForwardShape {
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub num_tokens: usize,
    pub phase: MoEForwardPhase,
}

impl MoEForwardShape {
    pub(super) fn new(batch_size: usize, seq_len: usize, hidden_dim: usize) -> Self {
        Self {
            batch_size,
            seq_len,
            hidden_dim,
            num_tokens: batch_size * seq_len,
            phase: MoEForwardPhase::from_shape(seq_len),
        }
    }

    pub(super) fn flat(self) -> (usize, usize) {
        (self.num_tokens, self.hidden_dim)
    }

    pub(super) fn output(self) -> (usize, usize, usize) {
        (self.batch_size, self.seq_len, self.hidden_dim)
    }
}

pub(super) struct MoEForward<'a> {
    pub xs: &'a Tensor,
    pub xs_flat: &'a Tensor,
    pub topk_weights: &'a Tensor,
    pub topk_ids: &'a Tensor,
    pub original_dtype: DType,
    pub shape: MoEForwardShape,
}

#[derive(Clone, Copy)]
pub(super) struct MoEForwardConfig {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub act: Activation,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(super) enum MoECudaFastPath {
    Decode,
    GroupedPrefill,
}
