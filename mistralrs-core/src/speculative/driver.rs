use std::any::Any;
use std::sync::Arc;

use candle_core::{Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::pipeline::sampling::{finish_or_add_toks_to_seq, sample_sequence};
use crate::pipeline::text_models_inputs_processor::{InputMetadata, PagedAttentionMeta};
use crate::pipeline::{ForwardInputsResult, Pipeline};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;

use super::cache::{PagedSpeculativeCacheTransaction, SpeculativeCacheTransaction};
use super::proposer::{SpeculativeKvCache, SpeculativeProposal, SpeculativeProposeCtx};
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

#[allow(clippy::too_many_arguments)]
pub async fn try_sample_speculative_causal_gen<P: SpeculativePipelineExt>(
    target: &mut P,
    seqs: &mut [&mut Sequence],
    logits: &[Tensor],
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    metadata: PagedAttentionMeta,
) -> Result<bool> {
    if !target.has_speculative_proposer() {
        return Ok(false);
    }
    if seqs.len() != 1 || logits.len() != 1 {
        for seq in seqs.iter_mut() {
            seq.clear_staged_speculative_tokens();
        }
        return Ok(false);
    }

    let seq = &mut *seqs[0];
    let general_metadata = target.get_metadata();
    let Some(cache_engine) = general_metadata.cache_engine.as_ref() else {
        seq.clear_staged_speculative_tokens();
        return Ok(false);
    };
    let staged_proposal = seq.take_staged_speculative_tokens();
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
            &metadata,
            None,
        )
        .await?;
        stage_next_proposal(target, seq, &outcome, &metadata, cache_engine)?;
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

    let kv_cache = cache_engine.get_kv_cache().clone();
    let target_hidden = match target.speculative_target_hidden(0) {
        Ok(hidden) => hidden,
        Err(_) => {
            finish_or_add_toks_to_seq(target, prefix_cacher, seq, anchor, eos_tok, true).await?;
            return Ok(true);
        }
    };
    let proposal = match target.speculative_propose(SpeculativeProposeCtx {
        sampled_token: anchor.token,
        seq_id: *seq.id(),
        base_len,
        sequence: seq,
        cache: SpeculativeKvCache::Paged {
            metadata: &metadata,
            kv_cache: &kv_cache,
        },
        target_hidden,
    }) {
        Ok(Some(proposal)) => proposal,
        Ok(None) | Err(_) => {
            finish_or_add_toks_to_seq(target, prefix_cacher, seq, anchor, eos_tok, true).await?;
            return Ok(true);
        }
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
    let transaction = PagedSpeculativeCacheTransaction::new(&metadata);
    let Some(mut cache_guard) = transaction.begin(*seq.id(), base_len, verify_len)? else {
        finish_or_add_toks_to_seq(target, prefix_cacher, seq, anchor, eos_tok, true).await?;
        return Ok(true);
    };

    let mut verify_tokens = Vec::with_capacity(verify_len);
    verify_tokens.push(anchor.token);
    verify_tokens.extend(proposal.tokens.iter().copied());
    let input_meta = {
        let mapper = target.device_mapper().expect("checked above");
        transaction.make_verify_input_metadata(
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
    stage_next_proposal(target, seq, &outcome, &metadata, cache_engine)?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
async fn verify_proposal<P: SpeculativePipelineExt>(
    target: &mut P,
    seq: &mut Sequence,
    verify_logits: Tensor,
    proposal: Vec<u32>,
    base_len: usize,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    metadata: &PagedAttentionMeta,
    anchor_to_emit: Option<crate::sampler::Logprobs>,
) -> Result<VerificationOutcome> {
    let transaction = PagedSpeculativeCacheTransaction::new(metadata);
    let mut cache_guard = transaction.guard_for_reserved(*seq.id(), base_len, proposal.len() + 1);
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

fn stage_next_proposal<P: SpeculativePipelineExt>(
    target: &mut P,
    seq: &mut Sequence,
    outcome: &VerificationOutcome,
    metadata: &PagedAttentionMeta,
    cache_engine: &crate::paged_attention::CacheEngine,
) -> Result<()> {
    let Some(continuation_token) = outcome.continuation_token else {
        seq.clear_staged_speculative_tokens();
        return Ok(());
    };

    let target_hidden = match target.speculative_target_hidden(outcome.accepted_drafts) {
        Ok(Some(target_hidden)) => target_hidden,
        Ok(None) | Err(_) => {
            seq.clear_staged_speculative_tokens();
            return Ok(());
        }
    };

    let kv_cache = cache_engine.get_kv_cache().clone();
    let proposal = target.speculative_propose(SpeculativeProposeCtx {
        sampled_token: continuation_token,
        seq_id: *seq.id(),
        base_len: outcome.keep_len,
        sequence: seq,
        cache: SpeculativeKvCache::Paged {
            metadata,
            kv_cache: &kv_cache,
        },
        target_hidden: Some(target_hidden),
    });

    match proposal {
        Ok(Some(proposal)) if !proposal.is_empty() => {
            seq.set_staged_speculative_tokens(proposal.tokens);
        }
        Ok(_) | Err(_) => {
            seq.clear_staged_speculative_tokens();
        }
    }
    Ok(())
}
