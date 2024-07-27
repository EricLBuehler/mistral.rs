use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::{
    get_bias_if_not_allowed,
    prefix_cacher::PrefixCacheManager,
    sampler::Logprobs,
    sequence::{Sequence, SequenceRecognizer},
};

use super::Pipeline;

pub(crate) async fn finish_or_add_toks_to_seq(
    this: &dyn Pipeline,
    prefix_cacher: &mut PrefixCacheManager,
    seq: &mut Sequence,
    logprobs: Logprobs,
    eos_tok: Option<&[u32]>,
    use_prefix_cacher: bool,
) -> Result<()> {
    let is_done = seq.is_done(logprobs.token, eos_tok, this.get_metadata().max_seq_len);
    seq.add_token(
        logprobs.clone(),
        this.get_metadata().tok_trie.decode(&[logprobs.token]),
        &is_done,
    );
    // Handle streaming requests
    if seq.get_mut_group().is_streaming {
        const STREAMING_RATE_LIMIT: usize = 3;

        let token_index = seq.get_toks().len();
        let rate_limit_allowed = is_done.is_some() || token_index % STREAMING_RATE_LIMIT == 0;

        if rate_limit_allowed {
            if let Some(delta) = crate::handle_seq_error_ok!(seq.get_delta(), seq.responder()) {
                if seq.get_mut_group().is_chat {
                    seq.add_streaming_chunk_choice_to_group(crate::ChunkChoice {
                        delta: crate::Delta {
                            content: delta.clone(),
                            role: "assistant".to_string(),
                        },
                        index: seq.get_response_index(),
                        finish_reason: is_done.map(|x| x.to_string()),
                        logprobs: if seq.return_logprobs() {
                            Some(crate::ResponseLogprob {
                                token: delta,
                                bytes: logprobs.bytes.clone().into_bytes(),
                                logprob: logprobs.logprob,
                                top_logprobs: logprobs.top_logprobs.unwrap().clone(),
                            })
                        } else {
                            None
                        },
                    });
                } else {
                    seq.add_streaming_completion_chunk_choice_to_group(
                        crate::CompletionChunkChoice {
                            text: delta.clone(),
                            index: seq.get_response_index(),
                            finish_reason: is_done.map(|x| x.to_string()),
                            logprobs: if seq.return_logprobs() {
                                Some(crate::ResponseLogprob {
                                    token: delta,
                                    bytes: logprobs.bytes.clone().into_bytes(),
                                    logprob: logprobs.logprob,
                                    top_logprobs: logprobs.top_logprobs.unwrap().clone(),
                                })
                            } else {
                                None
                            },
                        },
                    );
                }

                if let Some(reason) = is_done {
                    if use_prefix_cacher {
                        prefix_cacher.add_sequence(seq);
                        prefix_cacher.evict_to_cpu()?;
                    }
                    seq.set_state(crate::sequence::SequenceState::Done(reason));
                    this.reset_non_granular_state();
                }

                if seq
                    .get_mut_group()
                    .maybe_send_streaming_response(seq, this.name().clone())
                    .await
                    .is_err()
                {
                    // If we can't send the response, cancel the sequence
                    seq.set_state(crate::sequence::SequenceState::Done(
                        crate::sequence::StopReason::Canceled,
                    ));
                    this.reset_non_granular_state();
                }
            }
        }
    } else if let Some(reason) = is_done {
        /*
        ***********************
        Finish the sequence now
        ***********************
        */
        {
            seq.set_state(crate::sequence::SequenceState::Done(reason));
            let (tokenizer, pipeline_name) = {
                let pipeline_name = this.name();
                let tokenizer = this.tokenizer();
                (tokenizer, pipeline_name)
            };

            let logprobs = if seq.return_logprobs() {
                let mut logprobs = Vec::new();
                for logprob in seq.logprobs() {
                    let resp_logprob = crate::ResponseLogprob {
                        token: crate::handle_seq_error_ok!(
                            tokenizer.decode(&[logprob.token], false),
                            seq.responder()
                        ),
                        bytes: logprob.bytes.clone().into_bytes(),
                        logprob: logprob.logprob,
                        top_logprobs: logprob.top_logprobs.clone().unwrap(),
                    };
                    logprobs.push(resp_logprob);
                }
                Some(logprobs)
            } else {
                None
            };

            let text = match reason {
                crate::sequence::StopReason::Length(_)
                | crate::sequence::StopReason::ModelLength(_)
                | crate::sequence::StopReason::Eos
                | crate::sequence::StopReason::StopTok(_)
                | crate::sequence::StopReason::Canceled => {
                    String::from_utf8_lossy(seq.completion_bytes())
                        .trim_start()
                        .to_string()
                }
                crate::sequence::StopReason::StopString {
                    completion_bytes_pos,
                    ..
                } => {
                    let txt = String::from_utf8_lossy(seq.completion_bytes());
                    txt[..completion_bytes_pos].trim_start().to_string()
                }
            };

            if seq.get_mut_group().is_chat {
                let choice = crate::Choice {
                    finish_reason: reason.to_string(),
                    index: seq.get_response_index(),
                    message: crate::ResponseMessage {
                        content: text,
                        role: "assistant".to_string(),
                    },
                    logprobs: logprobs.map(|l| crate::Logprobs { content: Some(l) }),
                };
                seq.add_choice_to_group(choice);
            } else {
                let choice = crate::CompletionChoice {
                    finish_reason: reason.to_string(),
                    index: seq.get_response_index(),
                    text,
                    logprobs: None,
                };
                seq.add_completion_choice_to_group(choice);
            }

            if use_prefix_cacher {
                prefix_cacher.add_sequence(seq);
                prefix_cacher.evict_to_cpu()?;
            }

            let group = seq.get_mut_group();
            if group.is_chat {
                group
                    .maybe_send_chat_done_response(
                        crate::ChatCompletionResponse {
                            id: seq.id().to_string(),
                            choices: group.get_choices().to_vec(),
                            created: seq.creation_time(),
                            model: pipeline_name,
                            system_fingerprint: crate::SYSTEM_FINGERPRINT.to_string(),
                            object: "chat.completion".to_string(),
                            usage: group.get_usage(),
                        },
                        seq.responder(),
                    )
                    .await
                    .map_err(candle_core::Error::msg)?;
            } else {
                group
                    .maybe_send_completion_done_response(
                        crate::CompletionResponse {
                            id: seq.id().to_string(),
                            choices: group.get_completion_choices().to_vec(),
                            created: seq.creation_time(),
                            model: pipeline_name,
                            system_fingerprint: crate::SYSTEM_FINGERPRINT.to_string(),
                            object: "text_completion".to_string(),
                            usage: group.get_usage(),
                        },
                        seq.responder(),
                    )
                    .await
                    .map_err(candle_core::Error::msg)?;
            }
        }
        this.reset_non_granular_state();
    }

    Ok(())
}

pub async fn sample_and_add_toks(
    this: &dyn Pipeline,
    seqs: &mut [&mut Sequence],
    logits: Tensor,
    prefix_cacher: &mut PrefixCacheManager,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Result<()> {
    let seqs_len = seqs.len();
    let logits_seq = logits.to_device(&Device::Cpu)?.chunk(seqs_len, 0)?;
    debug_assert_eq!(logits_seq.len(), seqs_len);

    let use_async_pool = seqs_len > 1;

    let sampling_futures: Vec<_> = std::iter::zip(logits_seq, seqs.iter_mut())
        .map(|(logits_per_seq, seq)| {
            let return_logprobs = seq.return_logprobs();
            sample_sequence(
                logits_per_seq,
                seq,
                return_logprobs,
                rng.clone(),
                use_async_pool,
                true, // Append result to trie
                false,
            )
        })
        .collect();
    let sampled_vec = futures::future::join_all(sampling_futures).await;

    for (sampled, seq) in std::iter::zip(sampled_vec, seqs.iter_mut()) {
        let next_token = crate::handle_seq_error_stateaware_ok!(sampled, seq);

        let metadata = this.get_metadata();
        let eos_tok = if disable_eos_stop {
            None
        } else {
            Some(&metadata.eos_tok[..])
        };

        finish_or_add_toks_to_seq(this, prefix_cacher, seq, next_token, eos_tok, true).await?;
    }

    Ok(())
}

/// Async sample optionally adding to trie.
#[allow(clippy::too_many_arguments)]
pub async fn sample_sequence(
    logits: Tensor,
    seq: &mut Sequence,
    return_logprobs: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    use_async_pool: bool,
    add_to_trie: bool,
    sample_speculative: bool,
) -> Result<Logprobs> {
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    let start_at = seq.get_toks().len().saturating_sub(seq.prompt_tokens());

    let sampler = seq.sampler();
    let ctx_clone = seq.get_toks()[start_at..].to_vec();
    let rng_clone = rng.clone();
    let logits_clone = logits.clone();
    let first_lobprobs_response = if use_async_pool {
        tokio_rayon::spawn(move || {
            sampler.sample(
                logits_clone,
                Some(&ctx_clone),
                return_logprobs,
                rng_clone,
                sample_speculative,
            )
        })
        .await?
    } else {
        sampler.sample(
            logits_clone,
            Some(&ctx_clone),
            return_logprobs,
            rng_clone,
            sample_speculative,
        )?
    };

    let bias_if_not_allowed = match &mut seq.recognizer {
        SequenceRecognizer::Regex(ref mut rx) => {
            get_bias_if_not_allowed!(seq.tok_trie, rx.as_mut(), first_lobprobs_response.token)
        }
        SequenceRecognizer::Cfg(ref mut cfg) => {
            get_bias_if_not_allowed!(seq.tok_trie, cfg.as_mut(), first_lobprobs_response.token)
        }
        SequenceRecognizer::None => None,
    };
    let second_logprobs_response = match bias_if_not_allowed {
        Some(token_set) => {
            let mut acc = vec![-f32::INFINITY; seq.tok_trie.vocab_size()];
            token_set.apply_to(&mut acc);
            let new_logits = (logits + Tensor::from_slice(&acc, acc.len(), &Device::Cpu)?)?;

            let ctx_clone = seq.get_toks()[start_at..].to_vec();
            let rng_clone = rng.clone();
            let sampler = seq.sampler();
            if use_async_pool {
                tokio_rayon::spawn(move || {
                    sampler.sample(
                        new_logits,
                        Some(&ctx_clone),
                        return_logprobs,
                        rng_clone,
                        sample_speculative,
                    )
                })
                .await?
            } else {
                sampler.sample(
                    new_logits,
                    Some(&ctx_clone),
                    return_logprobs,
                    rng_clone,
                    sample_speculative,
                )?
            }
        }
        None => first_lobprobs_response,
    };

    if add_to_trie {
        match seq.recognizer {
            SequenceRecognizer::Regex(ref mut rx) => {
                seq.tok_trie
                    .append_token(rx.as_mut(), second_logprobs_response.token)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            }
            SequenceRecognizer::Cfg(ref mut cfg) => {
                seq.tok_trie
                    .append_token(cfg.as_mut(), second_logprobs_response.token)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            }
            SequenceRecognizer::None => {}
        }
    }
    Ok(second_logprobs_response)
}

#[derive(Clone)]
pub struct SpeculativeSample {
    pub sample: Logprobs,
}

/// Async sample without modifying sequence.
pub async fn sample_target_sequence_speculative(
    logits: Tensor,
    seq: &mut Sequence,
    return_logprobs: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    n_toks: usize,
) -> Result<Vec<SpeculativeSample>> {
    let mut sampled = Vec::new();
    for chunk in logits.chunk(n_toks, 1)? {
        sampled.push(SpeculativeSample {
            sample: sample_sequence(
                chunk,
                seq,
                return_logprobs,
                rng.clone(),
                true,  // TODO(EricLBuehler): does this hurt perf?
                false, // Do not append to trie (yet)
                true,
            )
            .await?,
        });
    }
    Ok(sampled)
}
