use std::any::Any;
use std::sync::Arc;

use candle_core::{Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::pipeline::sampling::{finish_or_add_toks_to_seq, sample_sequence};
use crate::pipeline::text_models_inputs_processor::InputMetadata;
use crate::pipeline::Pipeline;
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::{Sequence, SequenceState};

use super::cache::{SpeculativeCacheAccess, SpeculativeCacheGuard, SpeculativeCacheOutcome};
use super::proposer::{SpeculativeProposalBatch, SpeculativeProposeBatchCtx};
use super::staging::{staged_batch_state, StagedBatchState};
use super::verifier::{finish_verified_step, VerificationOutcome};

pub trait SpeculativePipelineExt: Pipeline {
    fn has_speculative_proposer(&self) -> bool;

    fn speculative_proposal_len(&self) -> Option<usize>;

    fn speculative_target_hiddens(&self, rows: &[(usize, usize)]) -> Result<Option<Tensor>>;

    fn speculative_propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>>;

    fn build_speculative_verify_inputs(&self, input_meta: InputMetadata) -> Result<Box<dyn Any>>;
}

/// Drop staged speculative proposals when the next step cannot verify them.
///
/// Staged tokens are only valid for the immediately following speculative
/// verification forward pass. If batching, cache backend choice, or another
/// constraint makes specdec unavailable for that pass, keeping them would let a
/// later step verify tokens against the wrong target state.
pub(crate) fn clear_staged_speculative_tokens(seqs: &mut [&mut Sequence]) {
    for seq in seqs.iter_mut() {
        seq.clear_staged_speculative_tokens();
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn try_sample_speculative_causal_gen<P, C>(
    target: &mut P,
    seqs: &mut [&mut Sequence],
    logits: &[Tensor],
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    cache: &C,
) -> Result<bool>
where
    P: SpeculativePipelineExt,
    C: SpeculativeCacheAccess,
{
    if !target.has_speculative_proposer() || seqs.is_empty() || logits.len() != seqs.len() {
        clear_staged_speculative_tokens(seqs);
        return Ok(false);
    }

    let staged_state = staged_batch_state(seqs);
    match staged_state {
        StagedBatchState::Homogeneous(staged_len) => {
            verify_staged_batch(
                target,
                seqs,
                logits,
                staged_len,
                prefix_cacher,
                disable_eos_stop,
                rng,
                cache,
            )
            .await?;
            Ok(true)
        }
        StagedBatchState::Mixed => {
            trim_mixed_staged_allocations(seqs, cache)?;
            clear_staged_speculative_tokens(seqs);
            bootstrap_staged_batch(
                target,
                seqs,
                logits,
                prefix_cacher,
                disable_eos_stop,
                rng,
                cache,
            )
            .await?;
            Ok(true)
        }
        StagedBatchState::None => {
            bootstrap_staged_batch(
                target,
                seqs,
                logits,
                prefix_cacher,
                disable_eos_stop,
                rng,
                cache,
            )
            .await?;
            Ok(true)
        }
    }
}

fn trim_mixed_staged_allocations<C>(seqs: &mut [&mut Sequence], cache: &C) -> Result<()>
where
    C: SpeculativeCacheAccess,
{
    for seq in seqs.iter_mut() {
        let staged_len = seq.active_staged_speculative_len();
        if staged_len == 0 {
            continue;
        }
        let Some(base_len) = seq.get_toks().len().checked_sub(1) else {
            continue;
        };
        let mut guard = cache.guard_for_reserved(*seq.id(), base_len, staged_len + 1);
        guard.rollback_to(seq.get_toks().len())?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn bootstrap_staged_batch<P, C>(
    target: &mut P,
    seqs: &mut [&mut Sequence],
    logits: &[Tensor],
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    cache: &C,
) -> Result<()>
where
    P: SpeculativePipelineExt,
    C: SpeculativeCacheAccess,
{
    let general_metadata = target.get_metadata();
    let eos_tok = if disable_eos_stop {
        None
    } else {
        Some(&general_metadata.eos_tok[..])
    };
    let use_async_pool = seqs.len() > 1;

    let mut active_indices = Vec::new();
    let mut sampled_tokens = Vec::new();
    let mut base_lens = Vec::new();
    let mut hidden_rows = Vec::new();

    for (idx, (seq, logits)) in seqs.iter_mut().zip(logits.iter()).enumerate() {
        let base_len = seq.get_toks().len();
        let return_logprobs = seq.return_logprobs();
        let anchor = sample_sequence(
            logits.clone(),
            seq,
            return_logprobs,
            rng.clone(),
            use_async_pool,
            false,
            use_async_pool,
        )
        .await?;
        let sampled_token = anchor.token;
        finish_or_add_toks_to_seq(target, prefix_cacher, seq, anchor, eos_tok, true).await?;
        if !matches!(seq.getstate(), SequenceState::Done(_)) {
            active_indices.push(idx);
            sampled_tokens.push(sampled_token);
            base_lens.push(base_len);
            hidden_rows.push((idx, 0));
        }
    }

    propose_and_stage_batch(
        target,
        seqs,
        &active_indices,
        &sampled_tokens,
        &base_lens,
        &hidden_rows,
        rng,
        cache,
    )
}

#[allow(clippy::too_many_arguments)]
async fn verify_staged_batch<P, C>(
    target: &mut P,
    seqs: &mut [&mut Sequence],
    logits: &[Tensor],
    staged_len: usize,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    cache: &C,
) -> Result<()>
where
    P: SpeculativePipelineExt,
    C: SpeculativeCacheAccess,
{
    let mut outcomes: Vec<Option<VerificationOutcome>> = Vec::with_capacity(seqs.len());
    let mut cache_guards: Vec<Option<C::Guard>> = Vec::with_capacity(seqs.len());
    let mut cache_outcomes: Vec<Option<SpeculativeCacheOutcome>> = Vec::with_capacity(seqs.len());
    for (seq, logits) in seqs.iter_mut().zip(logits.iter()) {
        let Some(base_len) = seq.get_toks().len().checked_sub(1) else {
            cache_guards.push(None);
            cache_outcomes.push(None);
            outcomes.push(None);
            continue;
        };
        let proposal = seq.take_staged_speculative_tokens();
        let proposal_logits = seq.take_staged_speculative_logits();
        if proposal.len() != staged_len {
            seq.clear_staged_speculative_tokens();
            cache_guards.push(None);
            cache_outcomes.push(None);
            outcomes.push(None);
            continue;
        }

        let cache_guard = cache.guard_for_reserved(*seq.id(), base_len, staged_len + 1);
        let outcome = finish_verified_step(
            target,
            seq,
            logits.clone(),
            proposal,
            proposal_logits,
            base_len,
            prefix_cacher,
            disable_eos_stop,
            rng.clone(),
            None,
        )
        .await?;
        let accepted_all = outcome.accepted_drafts == outcome.proposed_drafts;
        cache_outcomes.push(Some(SpeculativeCacheOutcome {
            keep_len: outcome.keep_len,
            accepted_all,
        }));
        cache_guards.push(Some(cache_guard));
        outcomes.push(Some(outcome));
    }
    cache.finish_verification_batch(&mut cache_guards, seqs, &cache_outcomes)?;

    let mut active_indices = Vec::new();
    let mut sampled_tokens = Vec::new();
    let mut base_lens = Vec::new();
    let mut hidden_rows = Vec::new();
    for (idx, outcome) in outcomes.iter().enumerate() {
        let Some(outcome) = outcome else {
            continue;
        };
        let Some(continuation_token) = outcome.continuation_token else {
            continue;
        };
        active_indices.push(idx);
        sampled_tokens.push(continuation_token);
        base_lens.push(outcome.keep_len);
        hidden_rows.push((idx, outcome.accepted_drafts));
    }

    propose_and_stage_batch(
        target,
        seqs,
        &active_indices,
        &sampled_tokens,
        &base_lens,
        &hidden_rows,
        rng,
        cache,
    )
}

#[allow(clippy::too_many_arguments)]
fn propose_and_stage_batch<P, C>(
    target: &mut P,
    seqs: &mut [&mut Sequence],
    active_indices: &[usize],
    sampled_tokens: &[u32],
    base_lens: &[usize],
    hidden_rows: &[(usize, usize)],
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    cache: &C,
) -> Result<()>
where
    P: SpeculativePipelineExt,
    C: SpeculativeCacheAccess,
{
    // Staging in one concrete sequence:
    //
    //   1. Target verifies [A B C D E F G].
    //   2. Drafts B..G are accepted, then the verifier samples continuation H
    //      from the last verified row. The cache contains A..G, but not H.
    //   3. We still have the target hidden state for G, and we just sampled H,
    //      so the proposer can immediately draft [I J K L M N].
    //   4. Store [I J K L M N] on the sequence. The next target decode
    //      forward consumes [H I J K L M N], appending H and verifying the
    //      staged tokens in one pass.
    //
    // In a batch, every participating sequence follows the same shape with the
    // same proposal width. Acceptance can still differ per sequence; only the
    // target forward shape is fixed.
    if active_indices.is_empty() {
        return Ok(());
    }
    let Some(proposal_len) = target.speculative_proposal_len() else {
        clear_active_staged(seqs, active_indices);
        return Ok(());
    };
    if proposal_len == 0 {
        clear_active_staged(seqs, active_indices);
        return Ok(());
    }

    let can_stage = {
        let sequences = active_indices
            .iter()
            .map(|idx| &*seqs[*idx] as &Sequence)
            .collect::<Vec<_>>();
        cache.can_stage_proposal(&sequences, base_lens, proposal_len)
    };
    if !can_stage {
        clear_active_staged(seqs, active_indices);
        return Ok(());
    }

    let target_hiddens = match target.speculative_target_hiddens(hidden_rows)? {
        Some(hidden) => Some(hidden),
        None => {
            clear_active_staged(seqs, active_indices);
            return Ok(());
        }
    };

    let seq_ids = active_indices
        .iter()
        .map(|idx| *seqs[*idx].id())
        .collect::<Vec<_>>();
    let proposal_batch = {
        let sequences = active_indices
            .iter()
            .map(|idx| &*seqs[*idx] as &Sequence)
            .collect::<Vec<_>>();
        target.speculative_propose(SpeculativeProposeBatchCtx {
            sampled_tokens,
            sampled_tokens_emitted: true,
            seq_ids: &seq_ids,
            base_lens,
            sequences: &sequences,
            cache: cache.proposer_cache(&sequences)?,
            target_hiddens,
            rng: rng.clone(),
        })?
    };

    let Some(proposal_batch) = proposal_batch else {
        clear_active_staged(seqs, active_indices);
        return Ok(());
    };
    if proposal_batch.proposals.len() != active_indices.len() {
        candle_core::bail!(
            "speculative proposer returned {} proposals for {} active sequences",
            proposal_batch.proposals.len(),
            active_indices.len()
        );
    }

    for (idx, proposal) in active_indices.iter().zip(proposal_batch.proposals) {
        if proposal.tokens.len() == proposal_len {
            seqs[*idx].set_staged_speculative(proposal.tokens, proposal.logits);
        } else {
            seqs[*idx].clear_staged_speculative_tokens();
        }
    }

    Ok(())
}

fn clear_active_staged(seqs: &mut [&mut Sequence], active_indices: &[usize]) {
    for idx in active_indices {
        seqs[*idx].clear_staged_speculative_tokens();
    }
}
