use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use rand::Rng;
use rand_isaac::Isaac64Rng;

use crate::pipeline::sampling::{finish_or_add_toks_to_seq, sample_sequence};
use crate::pipeline::Pipeline;
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sampler::Logprobs;
use crate::sequence::{Sequence, SequenceRecognizer, SequenceState};

pub(crate) fn can_greedy_device_verify(seq: &Sequence) -> bool {
    #[cfg(feature = "cuda")]
    {
        !seq.return_logprobs()
            && !seq.sampling_logprob_required()
            && matches!(seq.recognizer, SequenceRecognizer::None)
            && seq.tool_call_state.is_none()
            && seq
                .sampler()
                .cuda_batch_sampling_plan(false)
                .is_some_and(|plan| plan.kind.is_argmax())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = seq;
        false
    }
}

pub(crate) fn can_batch_greedy_device_verify(seqs: &[&mut Sequence]) -> bool {
    crate::speculative::staging::staged_batch_width(seqs).is_some()
        && seqs.iter().all(|seq| can_greedy_device_verify(seq))
}

#[cfg(feature = "cuda")]
pub(crate) struct GreedyDeviceVerifyInput<'a> {
    pub(crate) seq: &'a Sequence,
    pub(crate) logits: &'a Tensor,
}

#[cfg(feature = "cuda")]
pub(crate) fn greedy_device_verify_batch(
    inputs: &[GreedyDeviceVerifyInput<'_>],
) -> Result<Vec<Option<Vec<u32>>>> {
    let mut outputs = std::iter::repeat_with(|| None)
        .take(inputs.len())
        .collect::<Vec<_>>();
    let mut selected_indices = Vec::new();
    let mut selected_logits = Vec::new();
    let mut row_counts = Vec::new();
    let mut common_device: Option<candle_core::Device> = None;
    let mut common_dtype = None;
    let mut common_vocab = None;

    for (idx, input) in inputs.iter().enumerate() {
        if !input.logits.device().is_cuda() || !can_greedy_device_verify(input.seq) {
            continue;
        }
        let logits = match *input.logits.dims() {
            [1, rows, vocab] => input.logits.reshape((rows, vocab))?,
            [_, _] => input.logits.clone(),
            _ => continue,
        };
        let [rows, vocab] = *logits.dims() else {
            unreachable!("verify logits were normalized to rank two")
        };
        if rows == 0 || vocab == 0 {
            continue;
        }

        match (&common_device, common_dtype, common_vocab) {
            (Some(device), Some(dtype), Some(expected_vocab))
                if !device.same_device(logits.device())
                    || dtype != logits.dtype()
                    || expected_vocab != vocab =>
            {
                continue;
            }
            (Some(_), Some(_), Some(_)) => {}
            (None, None, None) => {
                common_device = Some(logits.device().clone());
                common_dtype = Some(logits.dtype());
                common_vocab = Some(vocab);
            }
            _ => unreachable!("CUDA verify batch properties are initialized together"),
        }
        selected_indices.push(idx);
        selected_logits.push(logits);
        row_counts.push(rows);
    }

    let Some(first_idx) = selected_indices.first().copied() else {
        return Ok(outputs);
    };
    let logits = if let [logits] = selected_logits.as_slice() {
        logits.clone()
    } else {
        Tensor::cat(&selected_logits.iter().collect::<Vec<_>>(), 0)?
    }
    .to_dtype(DType::F32)?
    .contiguous()?;
    let token_ids = inputs[first_idx]
        .seq
        .sampler()
        .submit_cuda_top1_batch_owned(&logits)?
        .complete()?
        .token_ids;
    let tokens = partition_device_tokens(token_ids, &row_counts)?;
    for (idx, tokens) in selected_indices.into_iter().zip(tokens) {
        outputs[idx] = Some(tokens);
    }
    Ok(outputs)
}

fn device_token_logprobs(token: u32) -> Logprobs {
    Logprobs {
        token,
        logprob: 0.0,
        top_logprobs: None,
        bytes: None,
    }
}

pub struct VerificationOutcome {
    pub accepted_drafts: usize,
    pub proposed_drafts: usize,
    pub keep_len: usize,
    pub continuation_token: Option<u32>,
}

pub(crate) struct VerificationInput {
    pub(crate) verify_logits: Tensor,
    pub(crate) proposal: Vec<u32>,
    pub(crate) proposal_logits: Option<Tensor>,
    pub(crate) base_len: usize,
    pub(crate) anchor_to_emit: Option<Logprobs>,
    pub(crate) device_tokens: Option<Vec<u32>>,
}

pub(crate) async fn finish_verified_step<P: Pipeline>(
    pipeline: &P,
    seq: &mut Sequence,
    input: VerificationInput,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Result<VerificationOutcome> {
    let VerificationInput {
        verify_logits,
        proposal,
        proposal_logits,
        base_len,
        anchor_to_emit,
        device_tokens,
    } = input;
    let general_metadata = pipeline.get_metadata();
    let eos_tok = seq.effective_eos_tokens(&general_metadata.eos_tok, disable_eos_stop);
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
        if stochastic_verification_allowed(
            seq.sampler().is_argmax(),
            !matches!(seq.recognizer, SequenceRecognizer::None),
            seq.tool_call_state.is_some(),
        ) {
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

    validate_device_token_count(device_tokens.as_deref(), proposal.len())?;

    let mut accepted = 0usize;
    for (idx, draft) in proposal.iter().copied().enumerate() {
        let sampled = match &device_tokens {
            Some(tokens) => device_token_logprobs(tokens[idx]),
            _ => {
                let row = logit_row(&verify_logits, idx)?;
                sample_sequence(
                    row.clone(),
                    seq,
                    return_logprobs,
                    eos_tok,
                    general_metadata.llg_factory.clone(),
                    general_metadata.max_seq_len,
                    rng.clone(),
                    false,
                    false,
                    false,
                )
                .await?
            }
        };
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

    let continuation = match &device_tokens {
        Some(tokens) => device_token_logprobs(tokens[accepted]),
        _ => {
            let row = logit_row(&verify_logits, accepted)?;
            sample_sequence(
                row.clone(),
                seq,
                return_logprobs,
                eos_tok,
                general_metadata.llg_factory.clone(),
                general_metadata.max_seq_len,
                rng.clone(),
                false,
                false,
                false,
            )
            .await?
        }
    };
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

fn validate_device_token_count(tokens: Option<&[u32]>, proposal_len: usize) -> Result<()> {
    if tokens.is_some_and(|tokens| tokens.len() < proposal_len + 1) {
        candle_core::bail!(
            "speculative CUDA verification returned fewer tokens than required: got {}, need {}",
            tokens.map_or(0, <[u32]>::len),
            proposal_len + 1
        );
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn partition_device_tokens(tokens: Vec<u32>, row_counts: &[usize]) -> Result<Vec<Vec<u32>>> {
    let expected = row_counts.iter().try_fold(0usize, |total, &rows| {
        total.checked_add(rows).ok_or_else(|| {
            candle_core::Error::Msg("speculative CUDA verify row count overflow".to_string())
        })
    })?;
    if tokens.len() != expected {
        candle_core::bail!(
            "speculative CUDA verification returned {} tokens for {expected} rows",
            tokens.len()
        );
    }
    let mut offset = 0;
    Ok(row_counts
        .iter()
        .map(|&rows| {
            let end = offset + rows;
            let row = tokens[offset..end].to_vec();
            offset = end;
            row
        })
        .collect())
}

fn stochastic_verification_allowed(
    is_argmax: bool,
    has_constraint: bool,
    has_tool_call_state: bool,
) -> bool {
    !is_argmax && !has_constraint && !has_tool_call_state
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
        let target_probs = sampler.speculative_target_probs(
            flat_logits(target_row.clone())?,
            seq.get_toks(),
            seq.prompt_tokens(),
        )?;
        let candidate_probs = sampler.speculative_candidate_probs(
            flat_logits(candidate_row)?,
            seq.get_toks(),
            seq.prompt_tokens(),
        )?;
        if target_probs.sampling.len() != candidate_probs.len() {
            candle_core::bail!(
                "speculative target/candidate vocab mismatch: target={}, candidate={}",
                target_probs.sampling.len(),
                candidate_probs.len()
            );
        }
        let draft_idx = draft as usize;
        let p_i = target_probs.sampling.get(draft_idx).copied().unwrap_or(0.0);
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

        if accepts_draft(draw, accept_prob) {
            accepted += 1;
            let sampled =
                sampler.logprobs_from_probs(draft, &target_probs.reporting, return_logprobs)?;
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
            .sampling
            .iter()
            .zip(candidate_probs.iter())
            .map(|(p, q)| (p - q).max(0.0))
            .collect::<Vec<_>>();
        if normalize_probs(&mut adjusted_probs).is_err() {
            adjusted_probs = target_probs.sampling;
        }
        let sampled = sampler.sample_from_probs(
            &adjusted_probs,
            &target_probs.reporting,
            return_logprobs,
            rng.clone(),
        )?;
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
    let target_probs = sampler.speculative_target_probs(
        flat_logits(row.clone())?,
        seq.get_toks(),
        seq.prompt_tokens(),
    )?;
    let continuation = sampler.sample_from_probs(
        &target_probs.sampling,
        &target_probs.reporting,
        return_logprobs,
        rng,
    )?;
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

fn accepts_draft(draw: f32, accept_prob: f32) -> bool {
    draw < accept_prob
}

#[cfg(test)]
mod tests {
    use super::{
        accepts_draft, partition_device_tokens, stochastic_verification_allowed,
        validate_device_token_count,
    };

    #[test]
    fn tool_sequences_use_per_row_target_sampling() {
        assert!(stochastic_verification_allowed(false, false, false));
        assert!(!stochastic_verification_allowed(false, false, true));
        assert!(!stochastic_verification_allowed(false, true, false));
        assert!(!stochastic_verification_allowed(true, false, false));
    }

    #[test]
    fn zero_probability_drafts_are_never_accepted() {
        assert!(!accepts_draft(0.0, 0.0));
        assert!(accepts_draft(0.0, 1.0));
        assert!(accepts_draft(0.999, 1.0));
    }

    #[test]
    fn partitions_batched_device_tokens_in_sequence_order() {
        let tokens = partition_device_tokens(vec![1, 2, 3, 4, 5, 6], &[2, 1, 3]).unwrap();
        assert_eq!(tokens, vec![vec![1, 2], vec![3], vec![4, 5, 6]]);
        assert!(partition_device_tokens(vec![1, 2], &[1, 2]).is_err());
    }

    #[test]
    fn device_verification_requires_a_continuation_token() {
        assert!(validate_device_token_count(None, 7).is_ok());
        assert!(validate_device_token_count(Some(&[0; 8]), 7).is_ok());
        assert!(validate_device_token_count(Some(&[0; 7]), 7).is_err());
    }
}
