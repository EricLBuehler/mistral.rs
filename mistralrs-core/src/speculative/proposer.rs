use candle_core::{Result, Tensor};

use crate::kv_cache::LayerCaches;
use crate::pipeline::text_models_inputs_processor::PagedAttentionMeta;
use crate::sequence::Sequence;

pub type TargetTokenEmbedder<'a> = dyn Fn(&Tensor) -> Result<Tensor> + 'a;

pub enum SpeculativeKvCache<'a> {
    Paged {
        metadata: &'a PagedAttentionMeta,
        kv_cache: &'a [(Tensor, Tensor)],
    },
    Normal {
        layers: &'a LayerCaches,
    },
}

pub struct SpeculativeProposeCtx<'a> {
    pub sampled_token: u32,
    pub seq_id: usize,
    pub base_len: usize,
    pub sequence: &'a Sequence,
    pub cache: SpeculativeKvCache<'a>,
    pub target_hidden: Option<Tensor>,
}

pub struct SpeculativeProposal {
    pub tokens: Vec<u32>,
}

impl SpeculativeProposal {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self { tokens }
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub trait SpeculativeProposer {
    fn max_proposal_len(&self) -> usize;

    fn propose(
        &mut self,
        ctx: SpeculativeProposeCtx<'_>,
        target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposal>;
}
