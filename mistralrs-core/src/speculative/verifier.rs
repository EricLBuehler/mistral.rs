use std::sync::Arc;

use candle_core::{Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::pipeline::sampling::{finish_or_add_toks_to_seq, sample_sequence};
use crate::pipeline::Pipeline;
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sampler::Logprobs;
use crate::sequence::{Sequence, SequenceState};

pub struct VerificationOutcome {
    pub accepted_drafts: usize,
    pub proposed_drafts: usize,
    pub keep_len: usize,
    pub continuation_token: Option<u32>,
}

pub async fn finish_verified_step<P: Pipeline>(
    pipeline: &P,
    seq: &mut Sequence,
    verify_logits: Tensor,
    proposal: Vec<u32>,
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

    let mut accepted = 0usize;
    for (idx, draft) in proposal.iter().copied().enumerate() {
        let row = logit_row(&verify_logits, idx)?;
        let sampled =
            sample_sequence(row, seq, return_logprobs, rng.clone(), false, false, false).await?;
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
            finish_or_add_toks_to_seq(pipeline, prefix_cacher, seq, sampled, eos_tok, true).await?;
            let keep_len = base_len + 1 + accepted;
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
    let continuation =
        sample_sequence(row, seq, return_logprobs, rng.clone(), false, false, false).await?;
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
