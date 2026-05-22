use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use rand::Rng;
use rand_isaac::Isaac64Rng;

use crate::pipeline::sampling::{finish_or_add_toks_to_seq, sample_sequence};
use crate::pipeline::Pipeline;
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sampler::Logprobs;
use crate::sequence::{Sequence, SequenceRecognizer, SequenceState};

pub struct VerificationOutcome {
    pub accepted_drafts: usize,
    pub proposed_drafts: usize,
    pub keep_len: usize,
    pub continuation_token: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
pub async fn finish_verified_step<P: Pipeline>(
    pipeline: &P,
    seq: &mut Sequence,
    verify_logits: Tensor,
    proposal: Vec<u32>,
    proposal_logits: Option<Tensor>,
    base_len: usize,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    anchor_to_emit: Option<Logprobs>,
) -> Result<VerificationOutcome> {
    let general_metadata = pipeline.get_metadata();
    let eos_tok = if disable_eos_stop {
        None
    } else {
        Some(&general_metadata.eos_tok[..])
    };
    let return_logprobs = seq.return_logprobs();

    if let Some(anchor) = anchor_to_emit {
        finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, anchor, eos_tok, true).await?;
        if matches!(seq.getstate(), SequenceState::Done(_)) {
            let keep_len = base_len + 1;
            seq.clear_staged_speculative_tokens();
            return Ok(VerificationOutcome {
                accepted_drafts: 0,
                proposed_drafts: proposal.len(),
                keep_len,
                continuation_token: None,
            });
        }
    }

    if let Some(proposal_logits) = proposal_logits {
        if !seq.sampler().is_argmax() && matches!(seq.recognizer, SequenceRecognizer::None) {
            return finish_verified_step_stochastic(
                pipeline,
                seq,
                verify_logits,
                proposal,
                proposal_logits,
                base_len,
                prefix_cacher,
                eos_tok,
                return_logprobs,
                rng,
            )
            .await;
        }
    }

    let mut accepted = 0usize;
    for (idx, draft) in proposal.iter().copied().enumerate() {
        let row = logit_row(&verify_logits, idx)?;
        let sampled = sample_sequence(
            row.clone(),
            seq,
            return_logprobs,
            rng.clone(),
            false,
            false,
            false,
        )
        .await?;
        let sampled_token = sampled.token;
        if sampled_token == draft {
            accepted += 1;
            finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, sampled, eos_tok, true).await?;
            if matches!(seq.getstate(), SequenceState::Done(_)) {
                let keep_len = base_len + 1 + accepted;
                seq.clear_staged_speculative_tokens();
                return Ok(VerificationOutcome {
                    accepted_drafts: accepted,
                    proposed_drafts: proposal.len(),
                    keep_len,
                    continuation_token: None,
                });
            }
        } else {
            let keep_len = base_len + 1 + accepted;
            finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, sampled, eos_tok, true).await?;
            if matches!(seq.getstate(), SequenceState::Done(_)) {
                seq.clear_staged_speculative_tokens();
                return Ok(VerificationOutcome {
                    accepted_drafts: accepted,
                    proposed_drafts: proposal.len(),
                    keep_len,
                    continuation_token: None,
                });
            }
            return Ok(VerificationOutcome {
                accepted_drafts: accepted,
                proposed_drafts: proposal.len(),
                keep_len,
                continuation_token: Some(sampled_token),
            });
        }
    }

    let row = logit_row(&verify_logits, accepted)?;
    let continuation = sample_sequence(
        row.clone(),
        seq,
        return_logprobs,
        rng.clone(),
        false,
        false,
        false,
    )
    .await?;
    let continuation_token = continuation.token;
    finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, continuation, eos_tok, true).await?;

    let keep_len = base_len + 1 + accepted;
    let continuation_token = if matches!(seq.getstate(), SequenceState::Done(_)) {
        seq.clear_staged_speculative_tokens();
        None
    } else {
        Some(continuation_token)
    };

    Ok(VerificationOutcome {
        accepted_drafts: accepted,
        proposed_drafts: proposal.len(),
        keep_len,
        continuation_token,
    })
}

#[allow(clippy::too_many_arguments)]
async fn finish_verified_step_stochastic<P: Pipeline>(
    pipeline: &P,
    seq: &mut Sequence,
    verify_logits: Tensor,
    proposal: Vec<u32>,
    proposal_logits: Tensor,
    base_len: usize,
    prefix_cacher: &mut PrefixCacheManagerV2,
    eos_tok: Option<&[u32]>,
    return_logprobs: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Result<VerificationOutcome> {
    let mut accepted = 0usize;
    for (idx, draft) in proposal.iter().copied().enumerate() {
        let target_row = logit_row(&verify_logits, idx)?;
        let candidate_row = logit_row(&proposal_logits, idx)?;
        let sampler = seq.sampler();
        let target_probs =
            sampler.speculative_target_probs(flat_logits(target_row.clone())?, seq.get_toks())?;
        let candidate_probs =
            sampler.speculative_candidate_probs(flat_logits(candidate_row)?, seq.get_toks())?;
        if target_probs.len() != candidate_probs.len() {
            candle_core::bail!(
                "speculative target/candidate vocab mismatch: target={}, candidate={}",
                target_probs.len(),
                candidate_probs.len()
            );
        }
        let draft_idx = draft as usize;
        let p_i = target_probs.get(draft_idx).copied().unwrap_or(0.0);
        let q_i = candidate_probs.get(draft_idx).copied().unwrap_or(0.0);
        let accept_prob = if q_i <= 0.0 {
            if p_i > 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            (p_i / q_i).min(1.0)
        };
        let draw = {
            let mut rng = rng.lock().expect("could not lock rng mutex");
            rng.random::<f32>()
        };

        if draw <= accept_prob {
            accepted += 1;
            let sampled = sampler.logprobs_from_probs(draft, &target_probs, return_logprobs)?;
            finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, sampled, eos_tok, true).await?;
            if matches!(seq.getstate(), SequenceState::Done(_)) {
                let keep_len = base_len + 1 + accepted;
                seq.clear_staged_speculative_tokens();
                return Ok(VerificationOutcome {
                    accepted_drafts: accepted,
                    proposed_drafts: proposal.len(),
                    keep_len,
                    continuation_token: None,
                });
            }
            continue;
        }

        let mut adjusted_probs = target_probs
            .iter()
            .zip(candidate_probs.iter())
            .map(|(p, q)| (p - q).max(0.0))
            .collect::<Vec<_>>();
        if normalize_probs(&mut adjusted_probs).is_err() {
            adjusted_probs = target_probs;
        }
        let sampled = sampler.sample_from_probs(&adjusted_probs, return_logprobs, rng.clone())?;
        let sampled_token = sampled.token;
        let keep_len = base_len + 1 + accepted;
        finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, sampled, eos_tok, true).await?;
        if matches!(seq.getstate(), SequenceState::Done(_)) {
            seq.clear_staged_speculative_tokens();
            return Ok(VerificationOutcome {
                accepted_drafts: accepted,
                proposed_drafts: proposal.len(),
                keep_len,
                continuation_token: None,
            });
        }
        return Ok(VerificationOutcome {
            accepted_drafts: accepted,
            proposed_drafts: proposal.len(),
            keep_len,
            continuation_token: Some(sampled_token),
        });
    }

    let row = logit_row(&verify_logits, accepted)?;
    let sampler = seq.sampler();
    let target_probs =
        sampler.speculative_target_probs(flat_logits(row.clone())?, seq.get_toks())?;
    let continuation = sampler.sample_from_probs(&target_probs, return_logprobs, rng)?;
    let continuation_token = continuation.token;
    finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, continuation, eos_tok, true).await?;

    let keep_len = base_len + 1 + accepted;
    let continuation_token = if matches!(seq.getstate(), SequenceState::Done(_)) {
        seq.clear_staged_speculative_tokens();
        None
    } else {
        Some(continuation_token)
    };

    Ok(VerificationOutcome {
        accepted_drafts: accepted,
        proposed_drafts: proposal.len(),
        keep_len,
        continuation_token,
    })
}

fn logit_row(logits: &Tensor, row: usize) -> Result<Tensor> {
    match logits.dims() {
        [_, rows, _] => {
            if row >= *rows {
                candle_core::bail!("speculative logit row {row} is out of range for {rows} rows");
            }
            logits.narrow(1, row, 1)
        }
        [rows, _] => {
            if row >= *rows {
                candle_core::bail!("speculative logit row {row} is out of range for {rows} rows");
            }
            logits.narrow(0, row, 1)
        }
        shape => candle_core::bail!("speculative logits have unsupported shape {shape:?}"),
    }
}

fn flat_logits(logits: Tensor) -> Result<Tensor> {
    match logits.dims() {
        [1, 1, _] => logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32),
        [1, _] => logits.squeeze(0)?.to_dtype(DType::F32),
        [_] => logits.to_dtype(DType::F32),
        dims => candle_core::bail!("speculative logit row must flatten to rank 1, got {dims:?}"),
    }
}

fn normalize_probs(probs: &mut [f32]) -> Result<()> {
    let sum: f32 = probs
        .iter()
        .copied()
        .filter(|prob| prob.is_finite() && *prob > 0.0)
        .sum();
    if sum <= 0.0 {
        candle_core::bail!("all probabilities are zero in speculative adjusted distribution");
    }
    for prob in probs.iter_mut() {
        if prob.is_finite() && *prob > 0.0 {
            *prob /= sum;
        } else {
            *prob = 0.0;
        }
    }
    Ok(())
}
