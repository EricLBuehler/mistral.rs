use std::sync::{Arc, Mutex};

use candle_core::{Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::pipeline::text_models_inputs_processor::PagedAttentionMeta;
use crate::sequence::Sequence;

pub type TargetTokenEmbedder<'a> = dyn Fn(&Tensor) -> Result<Tensor> + 'a;

pub enum SpeculativeKvCache<'a> {
    Paged {
        metadata: &'a PagedAttentionMeta,
        kv_cache: &'a [(Tensor, Tensor)],
    },
}

pub struct SpeculativeProposeBatchCtx<'a> {
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
}

#[derive(Clone, Debug)]
pub struct SpeculativeProposal {
    pub tokens: Vec<u32>,
    pub logits: Option<Tensor>,
}

impl SpeculativeProposal {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            logits: None,
        }
    }

    pub fn with_logits(tokens: Vec<u32>, logits: Tensor) -> Self {
        Self {
            tokens,
            logits: Some(logits),
        }
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
