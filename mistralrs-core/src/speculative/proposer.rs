use std::sync::{Arc, Mutex};

use candle_core::{Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::pipeline::text_models_inputs_processor::{
    FlashParams, PagedAttentionInputMetadata, PagedAttentionMeta,
};
use crate::sequence::Sequence;

pub type TargetTokenEmbedder<'a> = dyn Fn(&Tensor) -> Result<Tensor> + 'a;

pub enum SpeculativeKvCache<'a> {
    Paged {
        metadata: &'a PagedAttentionMeta,
        kv_cache: &'a [(Tensor, Tensor)],
    },
}

pub struct SpeculativeProposeBatchCtx<'a> {
    pub proposal_len: usize,
    pub sampled_tokens: &'a [u32],
    pub sampled_tokens_emitted: bool,
    pub seq_ids: &'a [usize],
    pub base_lens: &'a [usize],
    pub sequences: &'a [&'a Sequence],
    pub cache: SpeculativeKvCache<'a>,
    pub target_hiddens: Option<Tensor>,
    /// Per active sequence: its row in the last target forward's batch and how many leading rows of
    /// that forward (the sampled anchor plus accepted drafts) the proposer must consume.
    pub target_rows: &'a [(usize, usize)],
    pub rng: Arc<Mutex<Isaac64Rng>>,
}

/// Outcome of verifying one sequence's staged drafts, in target-batch order.
#[derive(Clone, Copy, Debug)]
pub struct SpeculativeCommitRow {
    pub batch_idx: usize,
    /// Rows of the verify forward that stay committed: the anchor plus accepted drafts.
    pub keep_rows: usize,
    pub accepted_all: bool,
}

/// One prompt chunk the target just processed; proposers that keep their own KV cache use it to
/// catch up over the prompt (rows are `[start, end)` per sequence, in target-batch order).
pub struct SpeculativePrefillCtx<'a> {
    pub seq_ids: &'a [usize],
    pub batch_indices: &'a [usize],
    pub tokens: &'a [&'a [u32]],
    pub chunk_ranges: &'a [(usize, usize)],
    pub is_final_prompt_chunk: bool,
    pub cache: SpeculativeKvCache<'a>,
    /// The target's own attention inputs for this chunk; a drafter sharing the block table can run one prefill with them.
    pub target_attention: Option<TargetAttentionInputs<'a>>,
}

#[derive(Clone, Copy)]
pub struct TargetAttentionInputs<'a> {
    pub metadata: &'a PagedAttentionInputMetadata,
    pub flash_params: &'a FlashParams,
}

#[derive(Clone, Debug)]
pub struct SpeculativeProposal {
    pub tokens: Vec<u32>,
    pub distribution: Option<SpeculativeProposalDistribution>,
}

#[derive(Clone, Debug)]
pub enum SpeculativeProposalDistribution {
    Logits(Tensor),
    SparseProbs(SparseSpeculativeProbs),
}

#[derive(Clone, Debug)]
pub struct SparseSpeculativeProbs {
    token_ids: Tensor,
    probs: Tensor,
}

impl SparseSpeculativeProbs {
    pub fn new(token_ids: Tensor, probs: Tensor) -> Result<Self> {
        let [positions, candidates] = token_ids.dims() else {
            candle_core::bail!(
                "sparse speculative token ids must have shape [positions, candidates], got {:?}",
                token_ids.dims()
            );
        };
        if probs.dims() != [*positions, *candidates] {
            candle_core::bail!(
                "sparse speculative probability shape {:?} does not match token ids {:?}",
                probs.dims(),
                token_ids.dims()
            );
        }
        if *candidates == 0 {
            candle_core::bail!("sparse speculative probabilities must contain candidates");
        }
        Ok(Self {
            token_ids: token_ids.to_dtype(candle_core::DType::U32)?,
            probs: probs.to_dtype(candle_core::DType::F32)?,
        })
    }

    pub fn positions(&self) -> usize {
        self.token_ids
            .dim(0)
            .expect("validated sparse probability rank")
    }

    pub fn token_ids(&self) -> &Tensor {
        &self.token_ids
    }

    pub fn probs(&self) -> &Tensor {
        &self.probs
    }
}

impl SpeculativeProposal {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            distribution: None,
        }
    }

    pub fn with_logits(tokens: Vec<u32>, logits: Tensor) -> Self {
        Self {
            tokens,
            distribution: Some(SpeculativeProposalDistribution::Logits(logits)),
        }
    }

    pub fn with_sparse_probs(tokens: Vec<u32>, token_ids: Tensor, probs: Tensor) -> Result<Self> {
        let sparse = SparseSpeculativeProbs::new(token_ids, probs)?;
        if sparse.positions() != tokens.len() {
            candle_core::bail!(
                "sparse speculative probabilities have {} positions for {} tokens",
                sparse.positions(),
                tokens.len()
            );
        }
        Ok(Self {
            tokens,
            distribution: Some(SpeculativeProposalDistribution::SparseProbs(sparse)),
        })
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

/// Sample one draft token per row of `logits` (`[rows, vocab]`) with each row's own sampler.
pub fn sample_draft_rows(
    logits: &Tensor,
    sequences: &[&Sequence],
    contexts: &mut [Vec<u32>],
    rng: &Arc<Mutex<Isaac64Rng>>,
) -> Result<Vec<u32>> {
    let batch = sequences.len();
    if contexts.len() != batch || logits.dim(0)? != batch {
        candle_core::bail!(
            "draft sampling batch mismatch: logits={}, contexts={}, sequences={batch}",
            logits.dim(0)?,
            contexts.len()
        );
    }
    let mut tokens = Vec::with_capacity(batch);
    for (row, seq) in sequences.iter().enumerate() {
        let row_logits = logits.get(row)?.to_dtype(candle_core::DType::F32)?;
        let sampled = seq.sampler().sample(
            row_logits,
            &contexts[row],
            seq.prompt_tokens(),
            false,
            rng.clone(),
            false,
            batch > 1,
        )?;
        contexts[row].push(sampled.token);
        tokens.push(sampled.token);
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{SparseSpeculativeProbs, SpeculativeProposal};

    #[test]
    fn sparse_probabilities_validate_shape_and_proposal_length() {
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (2, 2), &Device::Cpu).unwrap();
        let probs = Tensor::from_vec(vec![0.25f32, 0.75, 0.6, 0.4], (2, 2), &Device::Cpu).unwrap();
        let proposal =
            SpeculativeProposal::with_sparse_probs(vec![2, 3], ids.clone(), probs.clone()).unwrap();
        assert!(proposal.distribution.is_some());

        assert!(
            SpeculativeProposal::with_sparse_probs(vec![2], ids.clone(), probs.clone()).is_err()
        );
        assert!(SparseSpeculativeProbs::new(
            ids,
            Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu).unwrap(),
        )
        .is_err());
        assert!(SparseSpeculativeProbs::new(
            Tensor::zeros((2, 0), candle_core::DType::U32, &Device::Cpu).unwrap(),
            Tensor::zeros((2, 0), candle_core::DType::F32, &Device::Cpu).unwrap(),
        )
        .is_err());
    }
}
