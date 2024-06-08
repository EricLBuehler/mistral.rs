#[doc(hidden)]
#[macro_export]
macro_rules! finish_and_add_tokens_to_seq {
    ($this:expr, $prefix_cacher:expr, $seq:expr, $logprobs:expr, $eos_tok:expr, $use_prefix_cacher:expr) => {{
        let is_done = $seq.is_done($logprobs.token, $eos_tok, $this.metadata.max_seq_len);
        $seq.add_token(
            $logprobs.clone(),
            $this.get_metadata().tok_trie.decode(&[$logprobs.token]),
            &is_done,
        );
        // Handle streaming requests
        if $seq.get_mut_group().is_streaming && $seq.get_mut_group().is_chat {
            let token_index = $seq.get_toks().len();
            let rate_limit_allowed = is_done.is_some() || token_index % 3 == 0;

            if rate_limit_allowed {
                if let Some(delta) =
                    $crate::handle_seq_error_ok!($seq.get_delta(), $seq.responder())
                {
                    $seq.add_streaming_chunk_choice_to_group($crate::ChunkChoice {
                        delta: $crate::Delta {
                            content: delta.clone(),
                            role: "assistant".to_string(),
                        },
                        index: $seq.get_response_index(),
                        finish_reason: is_done.map(|x| x.to_string()),
                        logprobs: if $seq.return_logprobs() {
                            Some($crate::ResponseLogprob {
                                token: delta,
                                bytes: $logprobs.bytes.clone().into_bytes(),
                                logprob: $logprobs.logprob,
                                top_logprobs: $logprobs.top_logprobs.unwrap().clone(),
                            })
                        } else {
                            None
                        },
                    });

                    if let Some(reason) = is_done {
                        if $use_prefix_cacher {
                            $prefix_cacher.add_sequence($seq);
                            $prefix_cacher.evict_to_cpu()?;
                        }
                        $seq.set_state($crate::sequence::SequenceState::Done(reason));
                        $this.reset_non_granular_state();
                    }

                    if $seq
                        .get_mut_group()
                        .maybe_send_streaming_response($seq, $this.name().clone())
                        .await
                        .is_err()
                    {
                        // If we can't send the response, cancel the sequence
                        $seq.set_state($crate::sequence::SequenceState::Done(
                            $crate::sequence::StopReason::Canceled,
                        ));
                        $this.reset_non_granular_state();
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
                $seq.set_state($crate::sequence::SequenceState::Done(reason));
                let (tokenizer, pipeline_name) = {
                    let pipeline_name = $this.name();
                    let tokenizer = $this.tokenizer();
                    (tokenizer, pipeline_name)
                };

                let logprobs = if $seq.return_logprobs() {
                    let mut logprobs = Vec::new();
                    for logprob in $seq.logprobs() {
                        let resp_logprob = $crate::ResponseLogprob {
                            token: $crate::handle_seq_error_ok!(
                                tokenizer.decode(&[logprob.token], false),
                                $seq.responder()
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
                    $crate::sequence::StopReason::Length(_)
                    | $crate::sequence::StopReason::ModelLength(_)
                    | $crate::sequence::StopReason::Eos
                    | $crate::sequence::StopReason::StopTok(_)
                    | $crate::sequence::StopReason::Canceled => {
                        String::from_utf8_lossy($seq.completion_bytes())
                            .trim_start()
                            .to_string()
                    }
                    $crate::sequence::StopReason::StopString {
                        completion_bytes_pos,
                        ..
                    } => {
                        let txt = String::from_utf8_lossy($seq.completion_bytes());
                        txt[..completion_bytes_pos].trim_start().to_string()
                    }
                };

                if $seq.get_mut_group().is_chat {
                    let choice = $crate::Choice {
                        finish_reason: reason.to_string(),
                        index: $seq.get_response_index(),
                        message: $crate::ResponseMessage {
                            content: text,
                            role: "assistant".to_string(),
                        },
                        logprobs: logprobs.map(|l| $crate::Logprobs { content: Some(l) }),
                    };
                    $seq.add_choice_to_group(choice);
                } else {
                    let choice = $crate::CompletionChoice {
                        finish_reason: reason.to_string(),
                        index: $seq.get_response_index(),
                        text,
                        logprobs: None,
                    };
                    $seq.add_completion_choice_to_group(choice);
                }

                if $use_prefix_cacher {
                    $prefix_cacher.add_sequence($seq);
                    $prefix_cacher.evict_to_cpu()?;
                }

                let group = $seq.get_mut_group();
                if group.is_chat {
                    group
                        .maybe_send_done_response(
                            $crate::ChatCompletionResponse {
                                id: $seq.id().to_string(),
                                choices: group.get_choices().to_vec(),
                                created: $seq.creation_time(),
                                model: pipeline_name,
                                system_fingerprint: $crate::SYSTEM_FINGERPRINT.to_string(),
                                object: "chat.completion".to_string(),
                                usage: group.get_usage(),
                            },
                            $seq.responder(),
                        )
                        .await
                        .map_err(candle_core::Error::msg)?;
                } else {
                    group
                        .maybe_send_completion_done_response(
                            $crate::CompletionResponse {
                                id: $seq.id().to_string(),
                                choices: group.get_completion_choices().to_vec(),
                                created: $seq.creation_time(),
                                model: pipeline_name,
                                system_fingerprint: $crate::SYSTEM_FINGERPRINT.to_string(),
                                object: "text_completion".to_string(),
                                usage: group.get_usage(),
                            },
                            $seq.responder(),
                        )
                        .await
                        .map_err(candle_core::Error::msg)?;
                }
            }
            $this.reset_non_granular_state();
        }
    }};
}

/// Sample and add to the prefix cache.
#[doc(hidden)]
#[macro_export]
macro_rules! do_sample {
    ($this:expr, $seqs:expr, $logits:expr, $prefix_cacher:expr, $disable_eos_stop:expr, $rng:expr) => {{
        let seqs_len = $seqs.len();
        let logits_seq = $logits.to_device(&Device::Cpu)?.chunk(seqs_len, 0)?;
        debug_assert_eq!(logits_seq.len(), seqs_len);

        let use_async_pool = seqs_len > 1;

        let sampling_futures: Vec<_> = std::iter::zip(logits_seq, $seqs.iter_mut())
            .map(|(logits_per_seq, seq)| {
                let return_logprobs = seq.return_logprobs();
                $crate::pipeline::sampling::sample_sequence(
                    logits_per_seq,
                    seq,
                    return_logprobs,
                    $this.metadata.repeat_last_n,
                    $this.tok_trie.clone(),
                    $rng.clone(),
                    use_async_pool,
                    true, // Append result to trie
                    false,
                )
            })
            .collect();
        let sampled_vec = futures::future::join_all(sampling_futures).await;

        for (sampled, seq) in std::iter::zip(sampled_vec, $seqs.iter_mut()) {
            let next_token = $crate::handle_seq_error_stateaware_ok!(sampled, seq);

            let eos_tok = if $disable_eos_stop {
                None
            } else {
                Some(&$this.get_metadata().eos_tok[..])
            };

            $crate::finish_and_add_tokens_to_seq!(
                $this,
                $prefix_cacher,
                seq,
                next_token,
                eos_tok,
                true
            )
        }

        Ok(())
    }};
}
