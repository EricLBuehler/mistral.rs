use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::{
    prefix_cacher::PrefixCacheManagerV2,
    sampler::Logprobs,
    sequence::{Sequence, SequenceRecognizer, SequenceState, StopReason},
    tools::{parse_text_tools, ToolCallResponse, ToolCallType},
};
use mistralrs_mcp::CalledFunction;

use super::Pipeline;

macro_rules! fixup_sentencepiece {
    ($txt:expr) => {
        $txt.to_string().replace("▁", " ")
    };
    (Option $txt:expr) => {
        match &$txt {
            Some(txt) => Some(fixup_sentencepiece!(txt)),
            None => None,
        }
    };
}

pub(crate) async fn finish_or_add_toks_to_seq(
    this: &dyn Pipeline,
    prefix_cacher: &mut PrefixCacheManagerV2,
    seq: &mut Sequence,
    logprobs: Logprobs,
    eos_tok: Option<&[u32]>,
    use_prefix_cacher: bool,
) -> Result<()> {
    let mut is_done = seq.is_done(logprobs.token, eos_tok, this.get_metadata().max_seq_len);
    let metadata = this.get_metadata();
    let tok_env = metadata.tok_env().ok_or(candle_core::Error::Msg(
        "`finish_or_add_toks_to_seq` requires the pipeline to have a token trie".to_string(),
    ))?;
    // Include special tokens only when tool calling is active, so tool parsers
    // can see delimiters like <tool_call>, [TOOL_CALLS], <|python_tag|>.
    let include_special = seq.tools.is_some();
    let completion_bytes = tok_env
        .tok_trie()
        .decode_ext(&[logprobs.token], include_special);
    seq.add_token(logprobs.clone(), completion_bytes, &is_done);

    // If we can have a tool and we got a tool, stop the sequence early.
    // Doesn't conflict with the logic below because it does the same thing anyway.
    if let Some(ref t) = seq.tools {
        if let Ok(Some(ref d)) = seq.peek_delta() {
            let (_tool_use_still_possible, tool_use_is_done) =
                t.prefix_could_be_tool(this, d.as_str())?;

            if tool_use_is_done
                && matches!(
                    parse_text_tools(this, d, seq.tools.clone()),
                    Ok((None, _tools))
                )
            {
                seq.set_state(SequenceState::Done(StopReason::Eos));
                is_done = Some(StopReason::Eos);
            }
        }
    };

    // Handle streaming requests
    if seq.get_mut_group().is_streaming {
        let mut tool_use_still_possible = false;
        let mut tool_use_is_done = false;
        if let Some(ref t) = seq.tools {
            if let Ok(Some(ref d)) = seq.peek_delta() {
                (tool_use_still_possible, tool_use_is_done) =
                    t.prefix_could_be_tool(this, d.as_str())?;
            }
        };

        // let send = seq.get_toks().len() % 2 == 0 || is_done.is_some();
        let send = true;
        // Send chunks when:
        // 1. Tool call is not possible (!tool_use_still_possible) - normal streaming
        // 2. Tool call is complete (tool_use_is_done) - send the tool call
        // 3. Sequence is done (is_done.is_some()) - send buffered output as text since it wasn't a valid tool call
        if !tool_use_still_possible || tool_use_is_done || is_done.is_some() {
            if send {
                let delta_result = seq.get_delta();
                if let Some(delta) = crate::handle_seq_error_ok!(delta_result, seq.responder()) {
                    if seq.get_mut_group().is_chat {
                        // Check if we're in Harmony mode or think tag mode and use parsed content
                        let (content_delta, reasoning_delta) = if seq.is_harmony_mode() {
                            // In Harmony mode, use the parsed final content and reasoning
                            let final_delta = seq.get_harmony_final_delta();
                            let reasoning = seq.get_harmony_reasoning_delta();
                            (final_delta, reasoning)
                        } else if seq.is_think_tag_mode() {
                            // In think tag mode, use the parsed content and reasoning
                            let content = seq.get_think_tag_content_delta();
                            let reasoning = seq.get_think_tag_reasoning_delta();
                            (content, reasoning)
                        } else {
                            // Not in Harmony or think tag mode, use raw delta
                            let (text_new, _) =
                                parse_text_tools(this, delta.as_str(), seq.tools.clone())
                                    .map_err(candle_core::Error::msg)?;
                            (text_new.map(ToString::to_string), None)
                        };

                        // Detect tool calls
                        let tool_calls = if seq.is_harmony_mode() {
                            // In Harmony mode, only finalize tool calls when the sequence is done
                            // (EOS token or stop string), not when we first detect a tool call.
                            // This ensures tool call arguments are fully generated.
                            if is_done.is_some() && seq.has_harmony_tool_calls() {
                                // Sequence is done and has tool calls - finalize and send them
                                is_done = Some(StopReason::ToolCalls);
                                let harmony_tool_calls = seq.get_harmony_tool_calls();
                                harmony_tool_calls
                                    .into_iter()
                                    .enumerate()
                                    .map(|(i, tc)| ToolCallResponse {
                                        index: i,
                                        id: tc.id,
                                        tp: ToolCallType::Function,
                                        function: CalledFunction {
                                            name: tc.name,
                                            arguments: tc.arguments,
                                        },
                                    })
                                    .collect()
                            } else {
                                vec![]
                            }
                        } else {
                            // Not in Harmony mode - parse text for tool calls
                            let (_, tool_calls) =
                                parse_text_tools(this, delta.as_str(), seq.tools.clone())
                                    .map_err(candle_core::Error::msg)?;
                            if !tool_calls.is_empty() {
                                is_done = Some(StopReason::ToolCalls);
                            }
                            tool_calls
                        };

                        seq.add_streaming_chunk_choice_to_group(crate::ChunkChoice {
                            delta: crate::Delta {
                                content: fixup_sentencepiece!(Option content_delta),
                                role: "assistant".to_string(),
                                tool_calls: Some(tool_calls).filter(|v| !v.is_empty()),
                                reasoning_content: reasoning_delta,
                            },
                            index: seq.get_response_index(),
                            finish_reason: is_done.map(|x| x.to_string()),
                            logprobs: if seq.return_logprobs() {
                                Some(crate::ResponseLogprob {
                                    token: delta,
                                    bytes: logprobs.bytes.clone().map(|b| b.into_bytes()),
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
                                text: fixup_sentencepiece!(delta),
                                index: seq.get_response_index(),
                                finish_reason: is_done.map(|x| x.to_string()),
                                logprobs: if seq.return_logprobs() {
                                    Some(crate::ResponseLogprob {
                                        token: delta,
                                        bytes: logprobs.bytes.clone().map(|b| b.into_bytes()),
                                        logprob: logprobs.logprob,
                                        top_logprobs: logprobs.top_logprobs.unwrap().clone(),
                                    })
                                } else {
                                    None
                                },
                            },
                        );
                    }
                }
            }

            // Send usage on final chunk.
            let usage_opt = if is_done.is_some() {
                let usage = seq.get_mut_group().get_usage();
                seq.get_mut_group().total_prompt_toks = 0;
                seq.get_mut_group().total_toks = 0;
                Some(usage)
            } else {
                None
            };

            if seq
                .get_mut_group()
                .maybe_send_streaming_response(seq, this.name().clone(), usage_opt)
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

        // Handle Done state regardless of tool detection - must be outside the tool_use check
        // to ensure sequence completes even when tool detection thinks output might be a tool call
        if let Some(reason) = is_done {
            if use_prefix_cacher {
                prefix_cacher.add_sequence(seq);
                prefix_cacher.evict_caches()?;
            }
            seq.set_state(crate::sequence::SequenceState::Done(reason));
            this.reset_non_granular_state();
        }
    } else if let Some(mut reason) = is_done {
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
                        tokenizer
                        .as_ref()
                        .ok_or(candle_core::Error::Msg(
                            "`finish_or_add_toks_to_seq` requires the pipeline to have a tokenizer"
                                .to_string(),
                        ))?.decode(&[logprob.token], false),
                        seq.responder()
                    ),
                        bytes: logprob.bytes.clone().map(|b| b.into_bytes()),
                        logprob: logprob.logprob,
                        top_logprobs: logprob.top_logprobs.clone().unwrap(),
                    };
                    logprobs.push(resp_logprob);
                }
                Some(logprobs)
            } else {
                None
            };

            // Signal EOS to Harmony parser if in Harmony mode
            seq.harmony_process_eos();

            // Finalize think tag parser if in think tag mode
            seq.think_tag_finalize();

            let text = match reason {
                crate::sequence::StopReason::Length(_)
                | crate::sequence::StopReason::ModelLength(_)
                | crate::sequence::StopReason::Eos
                | crate::sequence::StopReason::StopTok(_)
                | crate::sequence::StopReason::Canceled
                | crate::sequence::StopReason::ToolCalls => {
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
                crate::sequence::StopReason::GeneratedImage
                | crate::sequence::StopReason::GeneratedSpeech => {
                    candle_core::bail!("Stop reason was `GeneratedImage`.")
                }
            };

            if seq.get_mut_group().is_chat {
                // In Harmony or think tag mode, use parsed content and tool calls
                let (text_new, tool_calls, reasoning_content) = if seq.is_harmony_mode() {
                    let final_content = seq.get_harmony_final_content();
                    let reasoning = seq.get_harmony_reasoning_content();

                    // Get Harmony tool calls
                    let harmony_tool_calls = seq.get_harmony_tool_calls();
                    let tool_calls: Vec<ToolCallResponse> = harmony_tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(i, tc)| ToolCallResponse {
                            index: i,
                            id: tc.id,
                            tp: ToolCallType::Function,
                            function: CalledFunction {
                                name: tc.name,
                                arguments: tc.arguments,
                            },
                        })
                        .collect();

                    (final_content, tool_calls, reasoning)
                } else if seq.is_think_tag_mode() {
                    // In think tag mode - finalize and get parsed content
                    seq.think_tag_finalize();
                    let final_content = seq.get_think_tag_content();
                    let reasoning = seq.get_think_tag_reasoning_content();

                    // Parse for tool calls in final content
                    let (text_new, tool_calls) = if let Some(ref content) = final_content {
                        parse_text_tools(this, content.as_str(), seq.tools.clone())
                            .map_err(candle_core::Error::msg)?
                    } else {
                        (None, vec![])
                    };
                    (
                        text_new.map(ToString::to_string).or(final_content),
                        tool_calls,
                        reasoning,
                    )
                } else {
                    // Not in Harmony or think tag mode - parse text for tool calls
                    let (text_new, tool_calls) =
                        parse_text_tools(this, text.as_str(), seq.tools.clone())
                            .map_err(candle_core::Error::msg)?;
                    (text_new.map(ToString::to_string), tool_calls, None)
                };

                if !tool_calls.is_empty() {
                    reason = StopReason::ToolCalls;
                }

                let choice = crate::Choice {
                    finish_reason: fixup_sentencepiece!(reason),
                    index: seq.get_response_index(),
                    message: crate::ResponseMessage {
                        content: text_new,
                        role: "assistant".to_string(),
                        tool_calls: Some(tool_calls).filter(|v| !v.is_empty()),
                        reasoning_content,
                    },
                    logprobs: logprobs.map(|l| crate::Logprobs { content: Some(l) }),
                };
                seq.add_choice_to_group(choice);
            } else {
                let choice = crate::CompletionChoice {
                    finish_reason: fixup_sentencepiece!(reason),
                    index: seq.get_response_index(),
                    text,
                    logprobs: logprobs.map(|l| crate::Logprobs { content: Some(l) }),
                };
                seq.add_completion_choice_to_group(choice);
            }

            if use_prefix_cacher {
                prefix_cacher.add_sequence(seq);
                prefix_cacher.evict_caches()?;
            }

            // Ensure timing info is synced to group before sending response
            seq.update_time_info();

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
    logits_seq: Vec<Tensor>,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Result<()> {
    let seqs_len = seqs.len();
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
                false,
                use_async_pool,
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
    sample_speculative: bool,
    multiple_sequences: bool,
) -> Result<Logprobs> {
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

    let sampler = seq.sampler();
    let ctx_clone = seq.get_toks().to_vec();
    let rng_clone = rng.clone();
    let logits_clone = logits.clone();
    let first_lobprobs_response = if use_async_pool {
        tokio_rayon::spawn(move || {
            sampler.sample(
                logits_clone,
                &ctx_clone,
                return_logprobs,
                rng_clone,
                sample_speculative,
                multiple_sequences,
            )
        })
        .await?
    } else {
        sampler.sample(
            logits_clone,
            &ctx_clone,
            return_logprobs,
            rng_clone,
            sample_speculative,
            multiple_sequences,
        )?
    };

    let bias_if_not_allowed = match &mut seq.recognizer {
        SequenceRecognizer::Llguidance(ref mut llg) => {
            if !llg.is_stopped()
                && llg
                    .validate_tokens(&[first_lobprobs_response.token])
                    .unwrap_or(0)
                    == 1
            {
                None
            } else {
                let mask = llg.compute_mask_or_eos().map_err(candle_core::Error::msg)?;
                if mask.is_allowed(first_lobprobs_response.token) {
                    // shouldn't really happen, except for EOS
                    None
                } else {
                    let mut acc = vec![-f32::INFINITY; logits.shape().dims1().unwrap()];
                    mask.iter_set_entries(|idx| {
                        if idx < acc.len() {
                            acc[idx] = 0.0;
                        }
                    });

                    Some(acc)
                }
            }
        }
        SequenceRecognizer::None => None,
    };
    let second_logprobs_response = match bias_if_not_allowed {
        Some(acc) => {
            let new_logits = (&logits + Tensor::from_slice(&acc, acc.len(), logits.device())?)?;

            let ctx_clone = seq.get_toks().to_vec();
            let rng_clone = rng.clone();
            let sampler = seq.sampler();
            if use_async_pool {
                tokio_rayon::spawn(move || {
                    sampler.sample(
                        new_logits,
                        &ctx_clone,
                        return_logprobs,
                        rng_clone,
                        sample_speculative,
                        multiple_sequences,
                    )
                })
                .await?
            } else {
                sampler.sample(
                    new_logits,
                    &ctx_clone,
                    return_logprobs,
                    rng_clone,
                    sample_speculative,
                    multiple_sequences,
                )?
            }
        }
        None => first_lobprobs_response,
    };

    match seq.recognizer {
        SequenceRecognizer::Llguidance(ref mut llg) => {
            if !llg.is_stopped() {
                llg.consume_token(second_logprobs_response.token)
                    .map_err(candle_core::Error::msg)?;
            }
        }
        SequenceRecognizer::None => {}
    }

    Ok(second_logprobs_response)
}

#[derive(Clone)]
pub struct SpeculativeSample {
    pub sample: Logprobs,
}

/// Async sample without modifying sequence (except for the constraint).
pub async fn sample_target_sequence_speculative(
    logits: Tensor,
    seq: &mut Sequence,
    return_logprobs: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    draft_samples: &[SpeculativeSample],
) -> Result<Vec<SpeculativeSample>> {
    let n_toks = draft_samples.len();

    // first, rollback the llg
    match &mut seq.recognizer {
        SequenceRecognizer::Llguidance(ref mut llg) => {
            llg.rollback(n_toks).map_err(candle_core::Error::msg)?;
        }
        SequenceRecognizer::None => {}
    }

    let mut sampled = Vec::new();
    for (chunk, draft) in logits
        .chunk(n_toks, 1)?
        .into_iter()
        .zip(draft_samples.iter())
    {
        let sample = sample_sequence(
            chunk,
            seq,
            return_logprobs,
            rng.clone(),
            true, // TODO(EricLBuehler): does this hurt perf?
            true,
            false,
        )
        .await?;
        let sampled_token = sample.token;
        sampled.push(SpeculativeSample { sample });
        if sampled_token != draft.sample.token {
            break;
        }
    }
    Ok(sampled)
}
