use std::any::Any;
use std::sync::Arc;

use candle_core::{Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::pipeline::sampling::{finish_or_add_toks_to_seq, sample_sequence};
use crate::pipeline::text_models_inputs_processor::InputMetadata;
use crate::pipeline::Pipeline;
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::{Sequence, SequenceState};
use crate::IntervalLogger;

use super::cache::{SpeculativeCacheAccess, SpeculativeCacheGuard, SpeculativeCacheOutcome};
use super::proposer::{
    SpeculativeCommitRow, SpeculativeProposalBatch, SpeculativeProposalDistribution,
    SpeculativeProposeBatchCtx, SpeculativeProposePreparation, SpeculativeProposePrepareCtx,
    SpeculativeTokens,
};
use super::staging::{staged_batch_state, StagedBatchState};
#[cfg(feature = "cuda")]
use super::verifier::{
    complete_sparse_rejection_device_verify_batch, greedy_device_verify_batch,
    sparse_rejection_device_verify_batch, try_submit_sparse_rejection_device_verify_batch,
    GreedyDeviceVerifyInput, SparseRejectionVerifyInput,
};
use super::verifier::{
    finish_verified_step, DeviceVerification, VerificationInput, VerificationOutcome,
};
use super::{SpeculativeBatchObservation, SpeculativeBatchPlan};

struct PreparedVerification {
    base_len: usize,
    proposal: SpeculativeTokens,
    proposal_distribution: Option<SpeculativeProposalDistribution>,
}

#[cfg(any(feature = "cuda", test))]
fn materialize_prepared_proposals(prepared: &mut [Option<PreparedVerification>]) -> Result<()> {
    let device_rows = prepared
        .iter()
        .enumerate()
        .filter_map(|(idx, prepared)| {
            prepared
                .as_ref()?
                .proposal
                .as_device()
                .map(|tokens| (idx, tokens.clone()))
        })
        .collect::<Vec<_>>();
    if device_rows.is_empty() {
        return Ok(());
    }
    let device = device_rows[0].1.device();
    if device_rows
        .iter()
        .any(|(_, tokens)| !tokens.device().same_device(device))
    {
        candle_core::bail!("one speculative verification batch cannot span devices");
    }
    let tensors = device_rows
        .iter()
        .map(|(_, tokens)| tokens)
        .collect::<Vec<_>>();
    let rows = Tensor::stack(&tensors, 0)?.to_vec2::<u32>()?;
    for ((idx, _), tokens) in device_rows.into_iter().zip(rows) {
        prepared[idx]
            .as_mut()
            .expect("device proposal row disappeared")
            .proposal = SpeculativeTokens::Host(tokens);
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn complete_after_preparation<T, U>(
    preparation: Result<T>,
    complete: impl FnOnce() -> Result<U>,
) -> Result<(T, U)> {
    let completion = complete();
    match preparation {
        Ok(preparation) => Ok((preparation, completion?)),
        Err(error) => Err(error),
    }
}

pub trait SpeculativePipelineExt: Pipeline {
    fn has_speculative_proposer(&self) -> bool;

    fn speculative_plan(&self, batch_size: usize) -> Option<SpeculativeBatchPlan>;

    fn speculative_observe(&self, observation: SpeculativeBatchObservation);

    fn speculative_bypass(&mut self, seq_ids: &[usize]);

    fn speculative_target_hiddens(&self, rows: &[(usize, usize)]) -> Result<Option<Tensor>>;

    fn speculative_propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>>;

    fn speculative_prepare_propose(
        &mut self,
        ctx: SpeculativeProposePrepareCtx<'_>,
    ) -> Result<Option<Box<dyn SpeculativeProposePreparation>>>;

    fn speculative_commit(&mut self, rows: &[SpeculativeCommitRow]) -> Result<()>;

    fn build_speculative_verify_inputs(&self, input_meta: InputMetadata) -> Result<Box<dyn Any>>;

    #[cfg(feature = "cuda")]
    fn cuda_sparse_rejection_workspace(
        &self,
    ) -> &std::sync::Mutex<Option<crate::speculative::CudaSparseRejectionWorkspace>>;
}

/// Drop staged speculative proposals when the next step cannot verify them.
///
/// Staged tokens are only valid for the immediately following speculative
/// verification forward pass. If batching, cache backend choice, or another
/// constraint makes specdec unavailable for that pass, keeping them would let a
/// later step verify tokens against the wrong target state.
pub(crate) fn clear_staged_speculative_tokens(seqs: &mut [&mut Sequence]) {
    for seq in seqs.iter_mut() {
        if seq.active_staged_speculative_len() > 0 {
            metrics::counter!("mistralrs_speculative_staged_drops_total").increment(1);
        }
        seq.clear_staged_speculative_tokens();
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn try_sample_speculative_causal_gen<P, C>(
    target: &mut P,
    seqs: &mut [&mut Sequence],
    logits: &[Tensor],
    batched_logits: Option<&Tensor>,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    cache: &C,
    logger: &IntervalLogger,
) -> Result<bool>
where
    P: SpeculativePipelineExt,
    C: SpeculativeCacheAccess,
{
    #[cfg(not(feature = "cuda"))]
    let _ = batched_logits;

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
                batched_logits,
                staged_len,
                prefix_cacher,
                disable_eos_stop,
                rng,
                cache,
                logger,
            )
            .await?;
            Ok(true)
        }
        StagedBatchState::Mixed => {
            trim_mixed_staged_allocations(seqs, cache)?;
            clear_staged_speculative_tokens(seqs);
            let Some(plan) = target.speculative_plan(seqs.len()) else {
                return Ok(false);
            };
            if plan.proposal_len == 0 {
                mark_batch_bypassed(target, seqs);
                return Ok(false);
            }
            bootstrap_staged_batch(
                target,
                seqs,
                logits,
                prefix_cacher,
                disable_eos_stop,
                rng,
                cache,
                plan,
            )
            .await?;
            Ok(true)
        }
        StagedBatchState::None => {
            let Some(plan) = target.speculative_plan(seqs.len()) else {
                return Ok(false);
            };
            if plan.proposal_len == 0 {
                mark_batch_bypassed(target, seqs);
                return Ok(false);
            }
            bootstrap_staged_batch(
                target,
                seqs,
                logits,
                prefix_cacher,
                disable_eos_stop,
                rng,
                cache,
                plan,
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
    plan: SpeculativeBatchPlan,
) -> Result<()>
where
    P: SpeculativePipelineExt,
    C: SpeculativeCacheAccess,
{
    let general_metadata = target.get_metadata();
    let use_async_pool = seqs.len() > 1;

    let mut active_indices = Vec::new();
    let mut sampled_tokens = Vec::new();
    let mut base_lens = Vec::new();
    let mut hidden_rows = Vec::new();

    for (idx, (seq, logits)) in seqs.iter_mut().zip(logits.iter()).enumerate() {
        let base_len = seq.get_toks().len();
        let return_logprobs = seq.return_logprobs();
        let eos_tok = seq.effective_eos_tokens(&general_metadata.eos_tok, disable_eos_stop);
        let anchor = sample_sequence(
            logits.clone(),
            seq,
            return_logprobs,
            eos_tok,
            general_metadata.llg_factory.clone(),
            general_metadata.max_seq_len,
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
        None,
        Some(plan),
    )
}

#[allow(clippy::too_many_arguments)]
async fn verify_staged_batch<P, C>(
    target: &mut P,
    seqs: &mut [&mut Sequence],
    logits: &[Tensor],
    batched_logits: Option<&Tensor>,
    staged_len: usize,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    cache: &C,
    logger: &IntervalLogger,
) -> Result<()>
where
    P: SpeculativePipelineExt,
    C: SpeculativeCacheAccess,
{
    #[cfg(not(feature = "cuda"))]
    let _ = batched_logits;
    #[cfg(feature = "cuda")]
    let mut proposer_preparation: Option<Box<dyn SpeculativeProposePreparation>> = None;
    #[cfg(not(feature = "cuda"))]
    let proposer_preparation: Option<Box<dyn SpeculativeProposePreparation>> = None;

    let mut prepared = Vec::with_capacity(seqs.len());
    let mut cache_guards: Vec<Option<C::Guard>> = Vec::with_capacity(seqs.len());
    for seq in seqs.iter_mut() {
        let Some(base_len) = seq.get_toks().len().checked_sub(1) else {
            cache_guards.push(None);
            prepared.push(None);
            continue;
        };
        let proposal = seq.take_staged_speculative_tokens();
        let proposal_distribution = seq.take_staged_speculative_distribution();
        if proposal.len() != staged_len {
            if !proposal.is_empty() {
                metrics::counter!("mistralrs_speculative_staged_drops_total").increment(1);
            }
            seq.clear_staged_speculative_tokens();
            cache_guards.push(None);
            prepared.push(None);
            continue;
        }

        cache_guards.push(Some(cache.guard_for_reserved(
            *seq.id(),
            base_len,
            staged_len + 1,
        )));
        prepared.push(Some(PreparedVerification {
            base_len,
            proposal,
            proposal_distribution,
        }));
    }

    #[cfg(feature = "cuda")]
    let mut device_verifications = {
        let active_indices = prepared
            .iter()
            .enumerate()
            .filter_map(|(idx, prepared)| prepared.as_ref().map(|_| idx))
            .collect::<Vec<_>>();
        let inputs = active_indices
            .iter()
            .map(|&idx| GreedyDeviceVerifyInput {
                seq: seqs[idx],
                logits: &logits[idx],
            })
            .collect::<Vec<_>>();
        let verified = greedy_device_verify_batch(&inputs)?;
        let mut aligned = std::iter::repeat_with(|| None)
            .take(seqs.len())
            .collect::<Vec<_>>();
        for (idx, tokens) in active_indices.iter().copied().zip(verified) {
            aligned[idx] = tokens.map(DeviceVerification::TargetTokens);
        }
        let mut sparse_indices = Vec::new();
        let mut sparse_inputs = Vec::new();
        for &idx in &active_indices {
            if aligned[idx].is_some() {
                continue;
            }
            let Some(prepared) = prepared[idx].as_ref() else {
                continue;
            };
            sparse_indices.push(idx);
            sparse_inputs.push(SparseRejectionVerifyInput {
                seq: seqs[idx],
                logits: &logits[idx],
                proposal: &prepared.proposal,
                distribution: prepared.proposal_distribution.as_ref(),
            });
        }
        let batched_target_logits = batched_logits.filter(|_| {
            sparse_indices.len() == seqs.len() && sparse_indices.iter().copied().eq(0..seqs.len())
        });
        let pending_sparse = {
            let mut workspace = target
                .cuda_sparse_rejection_workspace()
                .lock()
                .expect("CUDA sparse rejection workspace mutex poisoned");
            try_submit_sparse_rejection_device_verify_batch(
                &sparse_inputs,
                batched_target_logits,
                &rng,
                &mut workspace,
            )?
        };
        let sparse_output = if let Some(pending_sparse) = pending_sparse {
            drop(sparse_inputs);
            let preparation = if sparse_indices.len() == active_indices.len()
                && sparse_indices.iter().eq(active_indices.iter())
                && active_indices.len() == seqs.len()
                && active_indices.iter().copied().eq(0..seqs.len())
            {
                let seq_ids = active_indices
                    .iter()
                    .map(|idx| *seqs[*idx].id())
                    .collect::<Vec<_>>();
                let base_lens = active_indices
                    .iter()
                    .map(|idx| {
                        prepared[*idx]
                            .as_ref()
                            .expect("active speculative row disappeared")
                            .base_len
                    })
                    .collect::<Vec<_>>();
                let target_rows = active_indices
                    .iter()
                    .map(|idx| (*idx, staged_len + 1))
                    .collect::<Vec<_>>();
                target.speculative_prepare_propose(SpeculativeProposePrepareCtx {
                    seq_ids: &seq_ids,
                    base_lens: &base_lens,
                    target_rows: &target_rows,
                })
            } else {
                Ok(None)
            };
            let (preparation, output) = complete_after_preparation(preparation, || {
                let mut workspace = target
                    .cuda_sparse_rejection_workspace()
                    .lock()
                    .expect("CUDA sparse rejection workspace mutex poisoned");
                complete_sparse_rejection_device_verify_batch(pending_sparse, &mut workspace)
            })?;
            proposer_preparation = preparation;
            output
        } else {
            let output = {
                let mut workspace = target
                    .cuda_sparse_rejection_workspace()
                    .lock()
                    .expect("CUDA sparse rejection workspace mutex poisoned");
                sparse_rejection_device_verify_batch(
                    &sparse_inputs,
                    batched_target_logits,
                    &rng,
                    &mut workspace,
                )?
            };
            drop(sparse_inputs);
            output
        };
        for ((idx, verification), proposal) in sparse_indices
            .into_iter()
            .zip(sparse_output.verifications)
            .zip(sparse_output.materialized_proposals)
        {
            aligned[idx] = verification;
            if let Some(proposal) = proposal {
                prepared[idx]
                    .as_mut()
                    .expect("sparse proposal disappeared during verification")
                    .proposal = SpeculativeTokens::Host(proposal);
            }
        }
        materialize_prepared_proposals(&mut prepared)?;
        aligned
    };
    #[cfg(not(feature = "cuda"))]
    let mut device_verifications = std::iter::repeat_with(|| None)
        .take(seqs.len())
        .collect::<Vec<Option<DeviceVerification>>>();

    let mut outcomes: Vec<Option<VerificationOutcome>> = Vec::with_capacity(seqs.len());
    let mut cache_outcomes: Vec<Option<SpeculativeCacheOutcome>> = Vec::with_capacity(seqs.len());
    for (idx, (seq, logits)) in seqs.iter_mut().zip(logits.iter()).enumerate() {
        let Some(prepared) = prepared[idx].take() else {
            cache_outcomes.push(None);
            outcomes.push(None);
            continue;
        };
        let outcome = finish_verified_step(
            target,
            seq,
            VerificationInput {
                verify_logits: logits.clone(),
                proposal: prepared.proposal.into_vec()?,
                proposal_distribution: prepared.proposal_distribution,
                base_len: prepared.base_len,
                anchor_to_emit: None,
                device_verification: device_verifications[idx].take(),
            },
            prefix_cacher,
            disable_eos_stop,
            rng.clone(),
        )
        .await?;
        let accepted_all = outcome.accepted_drafts == outcome.proposed_drafts;
        cache_outcomes.push(Some(SpeculativeCacheOutcome {
            keep_len: outcome.keep_len,
            accepted_all,
        }));
        outcomes.push(Some(outcome));
    }
    cache.finish_verification_batch(&mut cache_guards, seqs, &cache_outcomes)?;
    let commit_rows = outcomes
        .iter()
        .enumerate()
        .filter_map(|(batch_idx, outcome)| {
            outcome.as_ref().map(|outcome| SpeculativeCommitRow {
                batch_idx,
                keep_rows: outcome.accepted_drafts + 1,
                accepted_all: outcome.accepted_drafts == outcome.proposed_drafts,
            })
        })
        .collect::<Vec<_>>();
    target.speculative_commit(&commit_rows)?;
    for (seq, outcome) in seqs.iter_mut().zip(outcomes.iter()) {
        if let Some(outcome) = outcome {
            seq.set_num_computed_tokens(outcome.keep_len);
        }
    }

    let mut active_indices = Vec::new();
    let mut sampled_tokens = Vec::new();
    let mut base_lens = Vec::new();
    let mut hidden_rows = Vec::new();
    let mut num_drafts = 0usize;
    let mut num_draft_tokens = 0usize;
    let mut num_accepted_tokens = 0usize;
    let mut max_proposed = 0usize;
    for outcome in outcomes.iter().flatten() {
        num_drafts += 1;
        num_draft_tokens += outcome.proposed_drafts;
        num_accepted_tokens += outcome.accepted_drafts;
        max_proposed = max_proposed.max(outcome.proposed_drafts);
    }
    let mut accepted_per_pos = vec![0usize; max_proposed];
    for outcome in outcomes.iter().flatten() {
        for count in accepted_per_pos.iter_mut().take(outcome.accepted_drafts) {
            *count += 1;
        }
    }
    logger.add_speculative_stats(
        num_drafts,
        num_draft_tokens,
        num_accepted_tokens,
        &accepted_per_pos,
    );
    if num_drafts > 0 {
        target.speculative_observe(SpeculativeBatchObservation {
            batch_size: seqs.len(),
            proposal_len: staged_len,
            accepted_drafts: num_accepted_tokens,
            sequences: num_drafts,
            proposed_drafts: num_draft_tokens,
        });
    }
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
        proposer_preparation.as_deref(),
        None,
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
    preparation: Option<&dyn SpeculativeProposePreparation>,
    plan: Option<SpeculativeBatchPlan>,
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
    let Some(plan) = plan.or_else(|| target.speculative_plan(active_indices.len())) else {
        clear_active_staged(seqs, active_indices);
        return Ok(());
    };
    let proposal_len = plan.proposal_len;
    if proposal_len == 0 {
        let seq_ids = active_indices
            .iter()
            .map(|idx| *seqs[*idx].id())
            .collect::<Vec<_>>();
        target.speculative_bypass(&seq_ids);
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

    let target_hiddens = if plan.needs_target_hiddens {
        match target.speculative_target_hiddens(hidden_rows)? {
            Some(hidden) => Some(hidden),
            None => {
                clear_active_staged(seqs, active_indices);
                return Ok(());
            }
        }
    } else {
        None
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
        let target_rows = hidden_rows
            .iter()
            .map(|(batch_idx, accepted)| (*batch_idx, accepted + 1))
            .collect::<Vec<_>>();
        target.speculative_propose(SpeculativeProposeBatchCtx {
            proposal_len,
            sampled_tokens,
            sampled_tokens_emitted: true,
            seq_ids: &seq_ids,
            base_lens,
            sequences: &sequences,
            cache: cache.proposer_cache(&sequences)?,
            target_hiddens,
            target_rows: &target_rows,
            preparation,
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
            seqs[*idx].set_staged_speculative(proposal.tokens, proposal.distribution);
        } else {
            seqs[*idx].clear_staged_speculative_tokens();
        }
    }

    Ok(())
}

fn mark_batch_bypassed<P>(target: &mut P, seqs: &[&mut Sequence])
where
    P: SpeculativePipelineExt,
{
    let seq_ids = seqs.iter().map(|seq| *seq.id()).collect::<Vec<_>>();
    target.speculative_bypass(&seq_ids);
}

fn clear_active_staged(seqs: &mut [&mut Sequence], active_indices: &[usize]) {
    for idx in active_indices {
        seqs[*idx].clear_staged_speculative_tokens();
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use candle_core::{Device, Tensor};

    use super::{complete_after_preparation, materialize_prepared_proposals, PreparedVerification};
    use crate::speculative::SpeculativeTokens;

    #[test]
    fn materializes_device_proposals_as_one_batch() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let mut prepared = vec![
            Some(PreparedVerification {
                base_len: 0,
                proposal: SpeculativeTokens::from_device(Tensor::new(&[1u32, 2], &device)?)?,
                proposal_distribution: None,
            }),
            Some(PreparedVerification {
                base_len: 0,
                proposal: SpeculativeTokens::Host(vec![3, 4]),
                proposal_distribution: None,
            }),
            Some(PreparedVerification {
                base_len: 0,
                proposal: SpeculativeTokens::from_device(Tensor::new(&[5u32, 6], &device)?)?,
                proposal_distribution: None,
            }),
        ];
        materialize_prepared_proposals(&mut prepared)?;
        let rows = prepared
            .iter()
            .map(|prepared| {
                prepared
                    .as_ref()
                    .expect("prepared row")
                    .proposal
                    .as_host()
                    .expect("host proposal")
            })
            .collect::<Vec<_>>();
        assert_eq!(rows, [&[1, 2][..], &[3, 4][..], &[5, 6][..]]);
        Ok(())
    }

    #[test]
    fn completion_runs_after_preparation_error() {
        let completed = Cell::new(false);
        let preparation = Err::<(), _>(candle_core::Error::msg("preparation failed"));
        let error = complete_after_preparation(preparation, || {
            completed.set(true);
            Ok(())
        })
        .expect_err("preparation must remain the primary error");
        assert!(completed.get());
        assert!(error.to_string().contains("preparation failed"));
    }
}
