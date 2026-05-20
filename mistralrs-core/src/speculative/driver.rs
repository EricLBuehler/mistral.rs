use std::any::Any;
use std::sync::Arc;

use candle_core::{Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::pipeline::sampling::{finish_or_add_toks_to_seq, sample_sequence};
use crate::pipeline::text_models_inputs_processor::InputMetadata;
use crate::pipeline::{ForwardInputsResult, Pipeline};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;

use super::cache::SpeculativeCacheAccess;
use super::proposer::{SpeculativeProposal, SpeculativeProposeCtx};
use super::verifier::{finish_verified_step, VerificationOutcome};

pub trait SpeculativePipelineExt: Pipeline {
    fn has_speculative_proposer(&self) -> bool;

    fn speculative_target_hidden(&self, row: usize) -> Result<Option<Tensor>>;

    fn speculative_propose(
        &mut self,
        ctx: SpeculativeProposeCtx<'_>,
    ) -> Result<Option<SpeculativeProposal>>;

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
    if !target.has_speculative_proposer() {
        return Ok(false);
    }
    if seqs.len() != 1 || logits.len() != 1 {
        // Staged verification is single-sequence state. Once this call is
        // batched or otherwise shape-incompatible, any pending staged tokens
        // cannot be matched to the current target forward pass.
        clear_staged_speculative_tokens(seqs);
        return Ok(false);
    }

    let seq = &mut *seqs[0];
    let general_metadata = target.get_metadata();
    let staged_proposal = if cache.supports_staged_verification() {
        seq.take_staged_speculative_tokens()
    } else {
        // Some cache backends can verify non-staged proposals safely, but cannot
        // roll back a forward pass that already consumed staged tokens. Clear
        // the staged proposal and continue with an immediate proposal instead.
        seq.clear_staged_speculative_tokens();
        Vec::new()
    };
    if !staged_proposal.is_empty() {
        let Some(base_len) = seq.get_toks().len().checked_sub(1) else {
            return Ok(false);
        };
        let outcome = verify_proposal(
            target,
            seq,
            logits[0].clone(),
            staged_proposal,
            base_len,
            prefix_cacher,
            disable_eos_stop,
            rng,
            cache,
            None,
        )
        .await?;
        stage_next_proposal(target, seq, &outcome, cache)?;
        return Ok(true);
    }

    let base_len = seq.get_toks().len();
    let return_logprobs = seq.return_logprobs();
    let anchor = sample_sequence(
        logits[0].clone(),
        seq,
        return_logprobs,
        rng.clone(),
        false,
        false,
        false,
    )
    .await?;

    let eos_tok = if disable_eos_stop {
        None
    } else {
        Some(&general_metadata.eos_tok[..])
    };
    if seq
        .is_done(anchor.token, eos_tok, general_metadata.max_seq_len)
        .is_some()
    {
        finish_or_add_toks_to_seq(target, prefix_cacher, seq, anchor, eos_tok, true).await?;
        return Ok(true);
    }

    let target_hidden = match target.speculative_target_hidden(0) {
        Ok(hidden) => hidden,
        Err(err) => return Err(err),
    };
    let proposal = match target.speculative_propose(SpeculativeProposeCtx {
        sampled_token: anchor.token,
        seq_id: *seq.id(),
        base_len,
        sequence: seq,
        cache: cache.proposer_cache(),
        target_hidden,
    }) {
        Ok(Some(proposal)) => proposal,
        Ok(None) => {
            finish_or_add_toks_to_seq(target, prefix_cacher, seq, anchor, eos_tok, true).await?;
            return Ok(true);
        }
        Err(err) => return Err(err),
    };
    if proposal.is_empty() {
        finish_or_add_toks_to_seq(target, prefix_cacher, seq, anchor, eos_tok, true).await?;
        return Ok(true);
    }
    if target.device_mapper().is_none() {
        finish_or_add_toks_to_seq(target, prefix_cacher, seq, anchor, eos_tok, true).await?;
        return Ok(true);
    }

    let verify_len = proposal.tokens.len() + 1;
    let Some(mut cache_guard) = cache.begin(*seq.id(), base_len, verify_len)? else {
        finish_or_add_toks_to_seq(target, prefix_cacher, seq, anchor, eos_tok, true).await?;
        return Ok(true);
    };

    let mut verify_tokens = Vec::with_capacity(verify_len);
    verify_tokens.push(anchor.token);
    verify_tokens.extend(proposal.tokens.iter().copied());
    let input_meta = {
        let mapper = target.device_mapper().expect("checked above");
        cache.make_verify_input_metadata(
            &verify_tokens,
            *seq.id(),
            base_len,
            &target.device(),
            mapper,
        )?
    };
    let verify_inputs = target.build_speculative_verify_inputs(input_meta)?;
    let verify_logits = match target.forward_inputs(verify_inputs, false)? {
        ForwardInputsResult::CausalGeneration { logits } => logits,
        _ => candle_core::bail!("speculative verification expected causal generation logits."),
    };

    let outcome = finish_verified_step(
        target,
        seq,
        verify_logits,
        proposal.tokens,
        base_len,
        prefix_cacher,
        disable_eos_stop,
        rng,
        &mut cache_guard,
        Some(anchor),
    )
    .await?;
    stage_next_proposal(target, seq, &outcome, cache)?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
async fn verify_proposal<P, C>(
    target: &mut P,
    seq: &mut Sequence,
    verify_logits: Tensor,
    proposal: Vec<u32>,
    base_len: usize,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    cache: &C,
    anchor_to_emit: Option<crate::sampler::Logprobs>,
) -> Result<VerificationOutcome>
where
    P: SpeculativePipelineExt,
    C: SpeculativeCacheAccess,
{
    let mut cache_guard = cache.guard_for_reserved(*seq.id(), base_len, proposal.len() + 1);
    finish_verified_step(
        target,
        seq,
        verify_logits,
        proposal,
        base_len,
        prefix_cacher,
        disable_eos_stop,
        rng,
        &mut cache_guard,
        anchor_to_emit,
    )
    .await
}

// Staging in one concrete example:
//
//   1. Target verifies [A B C D E F G].
//   2. Drafts B..G are accepted, then the verifier samples continuation H from
//      the last verified row. The cache contains A..G, but not H.
//   3. We still have the target hidden state for G, and we just sampled H, so
//      MTP can immediately propose [I J K L M N].
//   4. Store [I J K L M N] on the sequence. The next target decode forward
//      consumes [H I J K L M N], appending H and verifying the staged tokens in
//      one pass.
//
// Without staging, the next step would first run the target on H, then run MTP
// to propose I..N, then run another target verification pass. Staging moves
// that MTP proposal off the next step's critical path.
fn stage_next_proposal<P, C>(
    target: &mut P,
    seq: &mut Sequence,
    outcome: &VerificationOutcome,
    cache: &C,
) -> Result<()>
where
    P: SpeculativePipelineExt,
    C: SpeculativeCacheAccess,
{
    if !cache.supports_staged_verification() {
        seq.clear_staged_speculative_tokens();
        return Ok(());
    }

    let Some(continuation_token) = outcome.continuation_token else {
        seq.clear_staged_speculative_tokens();
        return Ok(());
    };

    let target_hidden = match target.speculative_target_hidden(outcome.accepted_drafts) {
        Ok(Some(target_hidden)) => target_hidden,
        Ok(None) => {
            seq.clear_staged_speculative_tokens();
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let proposal = target.speculative_propose(SpeculativeProposeCtx {
        sampled_token: continuation_token,
        seq_id: *seq.id(),
        base_len: outcome.keep_len,
        sequence: seq,
        cache: cache.proposer_cache(),
        target_hidden: Some(target_hidden),
    });

    match proposal {
        Ok(Some(proposal)) if !proposal.is_empty() => {
            seq.set_staged_speculative_tokens(proposal.tokens);
        }
        Ok(_) => {
            seq.clear_staged_speculative_tokens();
        }
        Err(err) => return Err(err),
    }
    Ok(())
}
