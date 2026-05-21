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
        layers: LayerCaches,
        cache_lens: Vec<Option<Vec<usize>>>,
    },
}

pub struct SpeculativeProposeBatchCtx<'a> {
    pub sampled_tokens: &'a [u32],
    pub seq_ids: &'a [usize],
    pub base_lens: &'a [usize],
    pub sequences: &'a [&'a Sequence],
    pub cache: SpeculativeKvCache<'a>,
    pub target_hiddens: Option<Tensor>,
}

#[derive(Clone, Debug)]
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

pub struct SpeculativeProposalBatch {
    pub proposals: Vec<SpeculativeProposal>,
}

impl SpeculativeProposalBatch {
    pub fn new(proposals: Vec<SpeculativeProposal>) -> Self {
        Self { proposals }
    }
}

pub trait SpeculativeProposer {
    fn proposal_len(&self) -> usize;

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch>;
}
