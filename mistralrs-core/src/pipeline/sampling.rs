use candle_core::{IndexOp, Result, Tensor};

use crate::{
    get_bias_if_not_allowed,
    prefix_cacher::PrefixCacheManager,
    sampler::{Logprobs, NewSampler, SamplingMetadata},
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

fn prepare_sampling_metadata<'a>(
    seqs: impl Iterator<Item = &'a Sequence>,
) -> Result<SamplingMetadata<'a>> {
    let mut topp_per_seq = Vec::new();
    let mut topk_per_seq = Vec::new();
    let mut minp_per_seq = Vec::new();
    let mut completion_toks_per_seq = Vec::new();
    let mut pres_penalty_per_seq = Vec::new();
    let mut freq_penalty_per_seq = Vec::new();
    let mut temperatures = Vec::new();
    let mut tokenizers = Vec::new();
    for seq in seqs {
        let completion_toks = &seq.get_toks()[seq.prompt_tokens()..];
        topp_per_seq.push(&seq.sampling_metadata().topp);
        topk_per_seq.push(&seq.sampling_metadata().topk);
        minp_per_seq.push(&seq.sampling_metadata().minp);
        pres_penalty_per_seq.push(&seq.sampling_metadata().presence_penalty);
        freq_penalty_per_seq.push(&seq.sampling_metadata().freq_penalty);
        completion_toks_per_seq.push(Tensor::new(
            completion_toks,
            seq.sampling_metadata().topp.device(),
        )?);
        temperatures.push(&seq.sampling_metadata().temperature);
        tokenizers.push(&*seq.sampling_metadata().tokenizer);
    }
    let sampling_metadata = SamplingMetadata {
        output_tokens_tensor: Tensor::stack(&completion_toks_per_seq, 0)?,
        presence_penalties: Tensor::stack(&pres_penalty_per_seq, 0)?,
        freq_penalties: Tensor::stack(&freq_penalty_per_seq, 0)?,
        topk: Tensor::stack(&topk_per_seq, 0)?,
        topp: Tensor::stack(&topp_per_seq, 0)?,
        minp: Tensor::stack(&minp_per_seq, 0)?,
        temperature: Tensor::stack(&temperatures, 0)?,
        tokenizers,
    };
    Ok(sampling_metadata)
}

pub fn sample_sequences(
    seqs: &mut [&mut Sequence],
    logits: Tensor,
    add_to_trie: bool,
) -> Result<Vec<Logprobs>> {
    let seqs_len = seqs.len();
    #[allow(clippy::cast_possible_truncation)]
    let seqs_with_grammar = seqs
        .iter()
        .enumerate()
        .filter_map(|(i, seq)| {
            if matches!(seq.recognizer, SequenceRecognizer::None) {
                None
            } else {
                Some(i as u32)
            }
        })
        .collect::<Vec<_>>();
    #[allow(clippy::cast_possible_truncation)]
    let seqs_without_grammar = (0..seqs.len())
        .filter(|i| !seqs_with_grammar.contains(&(*i as u32)))
        .map(|x| x as u32)
        .collect::<Vec<_>>();

    let logits_without_grammar = if seqs_without_grammar.len() == seqs_len {
        let seqs_without_grammar_t = Tensor::from_slice(
            &seqs_without_grammar,
            (seqs_without_grammar.len(),),
            logits.device(),
        )?;
        logits.gather(
            &seqs_without_grammar_t
                .reshape(((), 1, 1))?
                .repeat((1, logits.dims()[1], logits.dims()[2]))?
                .contiguous()?,
            0,
        )?
    } else {
        logits.clone()
    };

    let mut all_sampling_results = vec![None; seqs_len];

    // Handle case with no grammar first
    {
        #[allow(clippy::cast_possible_truncation)]
        let metadata = prepare_sampling_metadata(
            seqs.iter()
                .enumerate()
                .filter(|(i, _)| seqs_without_grammar.contains(&(*i as u32)))
                .map(|(_, x)| &**x),
        )?;
        let sampled = NewSampler.sample_on_gpu(logits_without_grammar, metadata)?;
        for (id, sampled) in seqs_without_grammar.into_iter().zip(sampled) {
            all_sampling_results[id as usize] = Some(sampled);
        }
    }

    // Grammars must be processed sequentially
    {
        #[allow(clippy::cast_possible_truncation)]
        for (i, seq) in seqs
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| seqs_with_grammar.contains(&(*i as u32)))
        {
            let logits = logits.i(i)?.unsqueeze(0)?;
            let first_sampled_response = NewSampler.sample_on_gpu(
                logits.clone(),
                prepare_sampling_metadata(std::iter::once(&**seq))?,
            )?[0]
                .clone();
            let tok_trie = &seq.tok_trie;

            let bias_if_not_allowed = match &mut seq.recognizer {
                SequenceRecognizer::Regex(ref mut rx) => {
                    get_bias_if_not_allowed!(tok_trie, rx.as_mut(), first_sampled_response.token)
                }
                SequenceRecognizer::Cfg(ref mut cfg) => {
                    get_bias_if_not_allowed!(tok_trie, cfg.as_mut(), first_sampled_response.token)
                }
                SequenceRecognizer::None => None,
            };
            let second_logprobs_response = match bias_if_not_allowed {
                Some(token_set) => {
                    let mut acc = vec![-f32::INFINITY; tok_trie.vocab_size()];
                    token_set.apply_to(&mut acc);
                    let new_logits = (&logits + Tensor::new(acc, logits.device())?)?;

                    NewSampler.sample_on_gpu(
                        new_logits,
                        prepare_sampling_metadata(std::iter::once(&**seq))?,
                    )?[0]
                        .clone()
                }
                None => first_sampled_response,
            };

            if add_to_trie {
                match seq.recognizer {
                    SequenceRecognizer::Regex(ref mut rx) => {
                        tok_trie
                            .append_token(rx.as_mut(), second_logprobs_response.token)
                            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    }
                    SequenceRecognizer::Cfg(ref mut cfg) => {
                        tok_trie
                            .append_token(cfg.as_mut(), second_logprobs_response.token)
                            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    }
                    SequenceRecognizer::None => {}
                }
            }
        }
    }
    Ok(all_sampling_results
        .into_iter()
        .map(|x| x.expect("Somehow a sequence did not get a corresponding sampled output."))
        .collect::<Vec<_>>())
}

pub async fn sample_and_add_toks(
    this: &dyn Pipeline,
    seqs: &mut [&mut Sequence],
    logits: Tensor,
    prefix_cacher: &mut PrefixCacheManager,
    disable_eos_stop: bool,
) -> Result<()> {
    for (next_token, seq) in std::iter::zip(sample_sequences(seqs, logits, true)?, seqs.iter_mut())
    {
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

#[derive(Clone)]
pub struct SpeculativeSample {
    pub sample: Logprobs,
}

/// Sample without modifying sequence.
pub fn sample_target_sequence_speculative(
    logits: Tensor,
    seq: &mut Sequence,
    n_toks: usize,
) -> Result<Vec<SpeculativeSample>> {
    let mut sampled = Vec::new();
    for chunk in logits.chunk(n_toks, 1)? {
        sampled.push(SpeculativeSample {
            sample: sample_sequences(&mut [seq], chunk, false)?[0].clone(),
        });
    }
    Ok(sampled)
}
