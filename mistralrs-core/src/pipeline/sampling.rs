use std::sync::Arc;

use candle_core::{DType, IndexOp, Result, Tensor};
#[cfg(feature = "cuda")]
use rand::distr::{Distribution, Uniform};
use rand_isaac::Isaac64Rng;

#[cfg(feature = "cuda")]
use crate::sampler::{CudaBatchSamplingKind, CudaTop1BatchSubmission};
use crate::{
    prefix_cacher::PrefixCacheManagerV2,
    sampler::Logprobs,
    sequence::{Sequence, SequenceRecognizer, SequenceState, StopReason, StreamingEmission},
    tools::ToolCallState,
};

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

#[cfg(test)]
fn parse_text_and_tool_calls(
    raw_text: &str,
    state: Option<&mut ToolCallState>,
) -> Result<(Option<String>, Vec<crate::tools::ToolCallResponse>)> {
    let Some(state) = state else {
        return Ok((Some(raw_text.to_string()), Vec::new()));
    };
    let parsed = state.finalize_for_response(raw_text, None, None, None)?;
    Ok((parsed.content, parsed.tool_calls))
}

#[cfg(test)]
fn parse_streaming_text_and_tool_calls(
    content_delta: Option<String>,
    raw_delta: &str,
    has_reasoning_parser: bool,
    state: Option<&mut ToolCallState>,
) -> Result<(Option<String>, Vec<crate::tools::ToolCallResponse>)> {
    let Some(state) = state else {
        return Ok((
            content_delta.or_else(|| Some(raw_delta.to_string())),
            Vec::new(),
        ));
    };
    let parsed =
        state.parse_streaming(content_delta, raw_delta, None, has_reasoning_parser, false)?;
    Ok((parsed.content, parsed.tool_calls))
}

fn activate_required_tool_call_grammar(
    seq: &mut Sequence,
    factory: Option<&Arc<llguidance::ParserFactory>>,
    max_model_len: usize,
    force_now: bool,
) {
    if !matches!(seq.recognizer, SequenceRecognizer::None) {
        return;
    }
    let generated = seq.generated_len();
    let max_generation_len = seq.max_generation_len(max_model_len);
    let (_, remaining, deadline_tokens) =
        ToolCallState::required_tool_call_deadline_status(generated, max_generation_len);
    let grm = seq.tool_call_state.as_mut().and_then(|state| {
        state.maybe_force_required_grammar(remaining, max_generation_len, force_now)
    });
    let Some(mut grm) = grm else {
        return;
    };
    let Some(factory) = factory else {
        tracing::warn!("Cannot force required tool call: llguidance is unavailable");
        return;
    };
    crate::tools::specialize_required_tool_call_grammar(&mut grm, factory.tok_env().tok_trie());
    match crate::pipeline::llg::constraint_from_llg_grammar(factory, grm) {
        Ok(matcher) => {
            seq.recognizer = SequenceRecognizer::Llguidance(Box::new(matcher));
            if let Some(state) = seq.tool_call_state.as_mut() {
                state.mark_grammar_active(true);
            }
            tracing::info!(
                generated_tokens = generated,
                remaining_tokens = remaining,
                deadline_tokens,
                "Forcing required tool call"
            );
        }
        Err(e) => {
            tracing::warn!("Failed to force required tool call grammar: {e}");
        }
    }
}

fn append_hidden_stop(mut text: Option<String>, hidden_stop: Option<&str>) -> Option<String> {
    if let (Some(text), Some(hidden_stop)) = (&mut text, hidden_stop) {
        text.push_str(hidden_stop);
    }
    text
}

fn response_stop_sequence(reason: Option<StopReason>, hidden_stop: Option<&str>) -> Option<String> {
    match reason {
        Some(StopReason::StopString { .. }) => hidden_stop.map(str::to_string),
        _ => None,
    }
}

// With a think-tag parser, only content outside the think block counts as tool text.
fn tool_detection_text(seq: &Sequence, hidden_stop: Option<&str>) -> Option<String> {
    let text = if seq.reasoning_mode().is_some() {
        seq.get_response_content()
    } else {
        seq.peek_delta().ok().flatten()
    };
    append_hidden_stop(text, hidden_stop)
}

fn streaming_response_logprob(emission: &StreamingEmission) -> crate::ResponseLogprob {
    crate::ResponseLogprob {
        token: emission
            .logprobs
            .bytes
            .clone()
            .unwrap_or_else(|| String::from_utf8_lossy(&emission.bytes).to_string()),
        bytes: Some(emission.bytes.clone()),
        logprob: emission.logprobs.logprob,
        top_logprobs: emission
            .logprobs
            .top_logprobs
            .clone()
            .expect("requested logprobs must include top logprobs"),
    }
}

fn cache_finished_sequence(
    this: &dyn Pipeline,
    prefix_cacher: &mut PrefixCacheManagerV2,
    seq: &mut Sequence,
) -> Result<()> {
    let recurrent_snapshots = if this.cache().is_hybrid() {
        let Some(idx) = seq.recurrent_state_idx() else {
            tracing::warn!(
                sequence_id = seq.id(),
                "Skipping hybrid prefix cache entry without recurrent state"
            );
            return Ok(());
        };
        match this
            .cache()
            .hybrid()
            .snapshot_recurrent_state(*seq.id(), idx)
        {
            Ok(snapshots) => Some(snapshots),
            Err(error) => {
                tracing::warn!(
                    sequence_id = seq.id(),
                    %error,
                    "Skipping hybrid prefix cache entry after recurrent snapshot failure"
                );
                return Ok(());
            }
        }
    } else {
        None
    };
    prefix_cacher.add_sequence(seq, recurrent_snapshots);
    prefix_cacher.evict_caches()?;
    Ok(())
}

pub(crate) async fn finish_or_add_toks_to_seq(
    this: &dyn Pipeline,
    prefix_cacher: &mut PrefixCacheManagerV2,
    seq: &mut Sequence,
    logprobs: Logprobs,
    eos_tok: Option<&[u32]>,
    use_prefix_cacher: bool,
) -> Result<()> {
    let is_done = seq.is_done(logprobs.token, eos_tok, this.get_metadata().max_seq_len);
    let metadata = this.get_metadata();
    let tok_env = metadata.tok_env().ok_or(candle_core::Error::Msg(
        "`finish_or_add_toks_to_seq` requires the pipeline to have a token trie".to_string(),
    ))?;
    // Include special tokens when tool calling is active (so tool parsers can see
    // delimiters like <tool_call>, [TOOL_CALLS], <|python_tag|>) or when think tag
    // mode is enabled (so <think>/<\/think> delimiters are visible in the output).
    let include_special = seq.tool_call_state.is_some() || seq.needs_special_tokens();
    let completion_bytes = tok_env
        .tok_trie()
        .decode_ext(&[logprobs.token], include_special);
    let mut is_done = seq.add_token(logprobs.clone(), completion_bytes, is_done);
    let hidden_stop = match is_done {
        Some(StopReason::StopString {
            stop_string_idx, ..
        }) => Some(seq.stop_strings()[stop_string_idx].clone()),
        _ => None,
    };

    // If we can have a tool and we got a tool, stop the sequence early.
    // Doesn't conflict with the logic below because it does the same thing anyway.
    if let Some(d) = tool_detection_text(seq, hidden_stop.as_deref()) {
        if let Some(ref mut state) = seq.tool_call_state {
            let (_tool_use_still_possible, tool_use_is_done) = state.prefix_status(d.as_str())?;

            if tool_use_is_done && state.stops_after_complete_tool_call() {
                if let Ok(tools) = state.complete_if_tool_call(d.as_str()) {
                    if !tools.is_empty() {
                        seq.set_state(SequenceState::Done(StopReason::Eos));
                        is_done = Some(StopReason::Eos);
                    }
                }
            }
        }
    };

    // Mid-stream grammar activation for tool calls.
    // When a tool call prefix is detected and no grammar is already active,
    // build a format-specific grammar and activate it so subsequent tokens
    // are constrained to valid tool call syntax.
    // Skip when the sequence is already done: peek_delta() still contains
    // the tool call prefix from earlier generation, which would spuriously
    // re-activate the grammar on a completed sequence.
    if matches!(seq.recognizer, SequenceRecognizer::None) && is_done.is_none() {
        let text = tool_detection_text(seq, None);
        let grm = seq
            .tool_call_state
            .as_mut()
            .and_then(|state| state.maybe_activate_continuation_grammar(text.as_deref()));

        if let Some(grm) = grm {
            if let Some(ref factory) = metadata.llg_factory {
                match crate::pipeline::llg::constraint_from_llg_grammar(factory, grm) {
                    Ok(matcher) => {
                        tracing::debug!("Activated tool call grammar");
                        seq.recognizer = SequenceRecognizer::Llguidance(Box::new(matcher));
                        if let Some(state) = seq.tool_call_state.as_mut() {
                            state.mark_grammar_active(false);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to build tool call grammar: {e}. \
                             Continuing without constraint."
                        );
                    }
                }
            }
        }
    }

    if is_done.is_some() {
        seq.flush_stop_pending_bytes();
    }

    // Handle streaming requests
    if seq.get_mut_group().is_streaming {
        let mut tool_use_still_possible = false;
        let mut tool_use_is_done = false;
        if let Some(d) = tool_detection_text(seq, hidden_stop.as_deref()) {
            if let Some(ref state) = seq.tool_call_state {
                (tool_use_still_possible, tool_use_is_done) = state.prefix_status(d.as_str())?;
            }
        };

        // Send chunks when:
        // 1. Tool call is not possible (!tool_use_still_possible) - normal streaming
        // 2. Tool call is complete (tool_use_is_done) - send the tool call
        // 3. Sequence is done (is_done.is_some()) - send buffered output as text since it wasn't a valid tool call
        if !tool_use_still_possible || tool_use_is_done || is_done.is_some() {
            if is_done.is_some() && seq.has_reasoning_state() {
                seq.finalize_reasoning();
            }
            let mut streaming_emissions = if seq.return_logprobs() {
                seq.take_ready_streaming_emissions(is_done.is_some())
            } else {
                Vec::new()
            };
            let delta_result = if seq.return_logprobs() {
                if streaming_emissions.is_empty() && is_done.is_none() {
                    Ok(None)
                } else {
                    Ok(Some(
                        streaming_emissions
                            .iter()
                            .map(|emission| emission.text.as_str())
                            .collect::<String>(),
                    ))
                }
            } else if is_done.is_some() {
                Ok(Some(seq.get_final_delta()))
            } else {
                seq.get_delta()
            };
            if let Some(mut delta) = crate::handle_seq_error_stateaware_ok!(delta_result, seq) {
                if seq.get_mut_group().is_chat {
                    let has_external_reasoning_parser = seq.reasoning_mode().is_some();
                    let has_reasoning_parser = seq.has_reasoning_state();
                    let reasoning_delta = if has_reasoning_parser {
                        seq.get_reasoning_content_delta()
                    } else {
                        None
                    };
                    let mut content_delta = if has_reasoning_parser {
                        seq.get_response_content_delta()
                    } else {
                        Some(delta.clone())
                    };

                    let tool_calls = if let Some(state) = seq.tool_call_state.as_mut() {
                        let parser_text = hidden_stop.as_deref().and_then(|hidden_stop| {
                            append_hidden_stop(
                                content_delta.clone().or_else(|| {
                                    (!has_external_reasoning_parser).then(|| delta.clone())
                                }),
                                Some(hidden_stop),
                            )
                        });
                        let parsed = state.parse_streaming(
                            content_delta.take(),
                            delta.as_str(),
                            parser_text.as_deref(),
                            has_external_reasoning_parser,
                            is_done.is_some(),
                        )?;
                        content_delta = parsed.content;
                        let parsed_tool_use_is_done = parsed.tool_use_is_done;
                        let _parsed_tool_use_still_possible = parsed.tool_use_still_possible;
                        if parsed_tool_use_is_done || !parsed.tool_calls.is_empty() {
                            is_done = Some(StopReason::ToolCalls);
                        }
                        parsed.tool_calls
                    } else {
                        Vec::new()
                    };

                    if is_done.is_some() && !seq.stop_strings().is_empty() {
                        seq.flush_stop_pending_bytes();
                        if seq.return_logprobs() {
                            let tail = seq.take_ready_streaming_emissions(true);
                            for emission in &tail {
                                delta.push_str(&emission.text);
                            }
                            streaming_emissions.extend(tail);
                        } else {
                            delta.push_str(&seq.get_final_delta());
                        }
                    }

                    let aligned_content = reasoning_delta.is_none()
                        && tool_calls.is_empty()
                        && content_delta.as_deref() == Some(delta.as_str());
                    if seq.return_logprobs() && aligned_content && !streaming_emissions.is_empty() {
                        let emission_count = streaming_emissions.len();
                        for (idx, emission) in streaming_emissions.iter().enumerate() {
                            seq.add_streaming_chunk_choice_to_group(crate::ChunkChoice {
                                delta: crate::Delta {
                                    content: Some(fixup_sentencepiece!(emission.text)),
                                    role: "assistant".to_string(),
                                    tool_calls: None,
                                    reasoning_content: None,
                                },
                                index: seq.get_response_index(),
                                finish_reason: if idx + 1 == emission_count {
                                    is_done.map(|reason| reason.to_string())
                                } else {
                                    None
                                },
                                stop_sequence: if idx + 1 == emission_count {
                                    response_stop_sequence(is_done, hidden_stop.as_deref())
                                } else {
                                    None
                                },
                                logprobs: Some(streaming_response_logprob(emission)),
                            });
                        }
                    } else {
                        if seq.return_logprobs() {
                            for emission in &streaming_emissions {
                                seq.add_streaming_chunk_choice_to_group(crate::ChunkChoice {
                                    delta: crate::Delta {
                                        content: None,
                                        role: "assistant".to_string(),
                                        tool_calls: None,
                                        reasoning_content: None,
                                    },
                                    index: seq.get_response_index(),
                                    finish_reason: None,
                                    stop_sequence: None,
                                    logprobs: Some(streaming_response_logprob(emission)),
                                });
                            }
                        }
                        seq.add_streaming_chunk_choice_to_group(crate::ChunkChoice {
                            delta: crate::Delta {
                                content: fixup_sentencepiece!(Option content_delta),
                                role: "assistant".to_string(),
                                tool_calls: Some(tool_calls).filter(|v| !v.is_empty()),
                                reasoning_content: reasoning_delta,
                            },
                            index: seq.get_response_index(),
                            finish_reason: is_done.map(|reason| reason.to_string()),
                            stop_sequence: response_stop_sequence(is_done, hidden_stop.as_deref()),
                            logprobs: None,
                        });
                    }
                } else {
                    if seq.return_logprobs() {
                        let emission_count = streaming_emissions.len();
                        for (idx, emission) in streaming_emissions.iter().enumerate() {
                            seq.add_streaming_completion_chunk_choice_to_group(
                                crate::CompletionChunkChoice {
                                    text: fixup_sentencepiece!(emission.text),
                                    index: seq.get_response_index(),
                                    finish_reason: if idx + 1 == emission_count {
                                        is_done.map(|reason| reason.to_string())
                                    } else {
                                        None
                                    },
                                    logprobs: Some(streaming_response_logprob(emission)),
                                },
                            );
                        }
                        if streaming_emissions.is_empty() && is_done.is_some() {
                            seq.add_streaming_completion_chunk_choice_to_group(
                                crate::CompletionChunkChoice {
                                    text: String::new(),
                                    index: seq.get_response_index(),
                                    finish_reason: is_done.map(|reason| reason.to_string()),
                                    logprobs: None,
                                },
                            );
                        }
                    } else {
                        seq.add_streaming_completion_chunk_choice_to_group(
                            crate::CompletionChunkChoice {
                                text: fixup_sentencepiece!(delta),
                                index: seq.get_response_index(),
                                finish_reason: is_done.map(|x| x.to_string()),
                                logprobs: None,
                            },
                        );
                    }
                }
            }

            // Send usage on final chunk.
            let usage_opt = if is_done.is_some() {
                seq.update_time_info();
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
                cache_finished_sequence(this, prefix_cacher, seq)?;
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
                let logprobs_for_response = seq.logprobs().to_vec();
                for logprob in logprobs_for_response {
                    let token = tokenizer
                        .as_ref()
                        .ok_or(candle_core::Error::Msg(
                            "`finish_or_add_toks_to_seq` requires the pipeline to have a tokenizer"
                                .to_string(),
                        ))?
                        .decode(&[logprob.token], false);
                    let token = crate::handle_seq_error_stateaware_ok!(token, seq);
                    let resp_logprob = crate::ResponseLogprob {
                        token,
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

            // Signal EOS to parsers before final response assembly.
            seq.finalize_reasoning();

            let text = match reason {
                crate::sequence::StopReason::Length(_)
                | crate::sequence::StopReason::ModelLength(_)
                | crate::sequence::StopReason::Eos
                | crate::sequence::StopReason::StopTok(_)
                | crate::sequence::StopReason::StopString { .. }
                | crate::sequence::StopReason::Canceled
                | crate::sequence::StopReason::ToolCalls => {
                    String::from_utf8_lossy(seq.completion_bytes())
                        .trim_start()
                        .to_string()
                }
                crate::sequence::StopReason::GeneratedImage
                | crate::sequence::StopReason::GeneratedSpeech => {
                    candle_core::bail!("Stop reason was `GeneratedImage`.")
                }
            };

            if seq.get_mut_group().is_chat {
                let has_reasoning_state = seq.has_reasoning_state();
                // An unclosed think block leaves no content; never fall back to the raw text
                let parsed_content = if has_reasoning_state {
                    seq.get_response_content().or_else(|| Some(String::new()))
                } else {
                    None
                };
                let reasoning_content = if has_reasoning_state {
                    seq.get_reasoning_content()
                } else {
                    None
                };
                let parsed = if let Some(state) = seq.tool_call_state.as_mut() {
                    let parser_text = hidden_stop.as_deref().map(|hidden_stop| {
                        let mut parser_text = parsed_content
                            .as_ref()
                            .cloned()
                            .unwrap_or_else(|| text.clone());
                        parser_text.push_str(hidden_stop);
                        parser_text
                    });
                    state.finalize_for_response(
                        text.as_str(),
                        parsed_content,
                        reasoning_content,
                        parser_text.as_deref(),
                    )?
                } else {
                    crate::tools::state::ToolCallParse {
                        content: parsed_content.or_else(|| Some(text.clone())),
                        reasoning_content,
                        tool_calls: Vec::new(),
                        tool_use_still_possible: false,
                        tool_use_is_done: false,
                    }
                };
                let text_new = parsed.content;
                let tool_calls = parsed.tool_calls;
                let reasoning_content = parsed.reasoning_content;

                if !tool_calls.is_empty() {
                    reason = StopReason::ToolCalls;
                }

                let choice = crate::Choice {
                    finish_reason: fixup_sentencepiece!(reason),
                    stop_sequence: response_stop_sequence(Some(reason), hidden_stop.as_deref()),
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
                cache_finished_sequence(this, prefix_cacher, seq)?;
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
                            adapter_generation: seq
                                .adapter_generation()
                                .map(|generation| generation.to_string()),
                            agentic_tool_calls: None,
                            files: None,
                            session_id: None,
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
                            adapter_generation: seq
                                .adapter_generation()
                                .map(|generation| generation.to_string()),
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

/// Append a block of pre-sampled tokens (e.g. a committed block-diffusion canvas) to each
/// sequence, running the standard per-token finalize path (EOS/length stop, tool parsing,
/// streaming, prefix caching) for every token. Stops consuming a block once its sequence
/// finishes.
pub(crate) async fn finalize_block_gen(
    this: &dyn Pipeline,
    seqs: &mut [&mut Sequence],
    token_blocks: Vec<Vec<u32>>,
    denoise_times: Vec<std::time::Duration>,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
) -> Result<()> {
    debug_assert_eq!(token_blocks.len(), seqs.len());

    for ((block, denoise_time), seq) in
        std::iter::zip(std::iter::zip(token_blocks, denoise_times), seqs.iter_mut())
    {
        seq.add_pending_denoise_time(denoise_time);
        let metadata = this.get_metadata();
        let eos_tok = seq.effective_eos_tokens(&metadata.eos_tok, disable_eos_stop);

        for token in block {
            if !seq.is_running() {
                break;
            }
            let logprobs = crate::sampler::Logprobs {
                token,
                logprob: 0.0,
                bytes: None,
                top_logprobs: None,
            };
            finish_or_add_toks_to_seq(this, prefix_cacher, seq, logprobs, eos_tok, true).await?;
        }
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
    sample_and_add_toks_inner(
        this,
        seqs,
        CausalLogitsBatch::PerSequence(logits_seq),
        prefix_cacher,
        disable_eos_stop,
        rng,
    )
    .await
}

pub async fn sample_and_add_toks_batched(
    this: &dyn Pipeline,
    seqs: &mut [&mut Sequence],
    logits: Tensor,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Result<()> {
    sample_and_add_toks_inner(
        this,
        seqs,
        CausalLogitsBatch::Batched(logits),
        prefix_cacher,
        disable_eos_stop,
        rng,
    )
    .await
}

enum CausalLogitsBatch {
    PerSequence(Vec<Tensor>),
    Batched(Tensor),
}

impl CausalLogitsBatch {
    fn len(&self) -> Result<usize> {
        match self {
            Self::PerSequence(logits) => Ok(logits.len()),
            Self::Batched(logits) => logits.dim(0),
        }
    }

    fn into_cpu_rows(self) -> Result<Vec<Tensor>> {
        match self {
            Self::PerSequence(logits) => coalesce_batch_logits_to_cpu(logits),
            Self::Batched(logits) => {
                let batch = logits.dim(0)?;
                let logits = logits.to_device(&candle_core::Device::Cpu)?;
                (0..batch).map(|idx| logits.i(idx)).collect()
            }
        }
    }
}

async fn sample_and_add_toks_inner(
    this: &dyn Pipeline,
    seqs: &mut [&mut Sequence],
    logits: CausalLogitsBatch,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Result<()> {
    let seqs_len = seqs.len();
    debug_assert_eq!(logits.len()?, seqs_len);

    let use_async_pool = seqs_len > 1;
    let metadata = this.get_metadata();
    let llg_factory = metadata.llg_factory.clone();
    let max_model_len = metadata.max_seq_len;
    let eos_toks = metadata.eos_tok.clone();

    let sampled_vec = match try_sample_batch_cuda(&logits, seqs, &rng)? {
        Some(sampled) => sampled,
        None => {
            let logits_seq = logits.into_cpu_rows()?;
            let sampling_futures: Vec<_> = std::iter::zip(logits_seq, seqs.iter_mut())
                .map(|(logits_per_seq, seq)| {
                    let return_logprobs = seq.return_logprobs();
                    let eos_tok = seq.effective_eos_tokens(&eos_toks, disable_eos_stop);
                    sample_sequence(
                        logits_per_seq,
                        seq,
                        return_logprobs,
                        eos_tok,
                        llg_factory.clone(),
                        max_model_len,
                        rng.clone(),
                        use_async_pool,
                        false,
                        use_async_pool,
                    )
                })
                .collect();
            futures::future::join_all(sampling_futures).await
        }
    };

    for (sampled, seq) in std::iter::zip(sampled_vec, seqs.iter_mut()) {
        let next_token = crate::handle_seq_error_stateaware_ok!(sampled, seq);

        let metadata = this.get_metadata();
        let eos_tok = seq.effective_eos_tokens(&metadata.eos_tok, disable_eos_stop);

        finish_or_add_toks_to_seq(this, prefix_cacher, seq, next_token, eos_tok, true).await?;
    }

    Ok(())
}

pub(crate) fn can_sample_batch_cuda(seqs: &[&mut Sequence]) -> bool {
    #[cfg(feature = "cuda")]
    {
        let mut categorical = None;
        for seq in seqs {
            if !matches!(&seq.recognizer, SequenceRecognizer::None) || seq.tool_call_state.is_some()
            {
                return false;
            }
            let Some(plan) = seq
                .sampler()
                .cuda_batch_sampling_plan(seq.return_logprobs())
            else {
                return false;
            };
            let plan_is_categorical = matches!(plan.kind, CudaBatchSamplingKind::Categorical);
            if categorical.is_some_and(|expected| expected != plan_is_categorical) {
                return false;
            }
            categorical = Some(plan_is_categorical);
        }
        categorical.is_some()
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = seqs;
        false
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaGreedyBatchSubmission {
    inner: CudaTop1BatchSubmission,
}

#[cfg(feature = "cuda")]
impl CudaGreedyBatchSubmission {
    pub(crate) fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    pub(crate) fn wait_on(
        &self,
        stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    ) -> Result<()> {
        self.inner.wait_on(stream)
    }

    pub(crate) fn release_after(
        &self,
        stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    ) -> Result<()> {
        self.inner.release_after(stream)
    }

    pub(crate) fn complete(self) -> Result<Vec<u32>> {
        Ok(self.inner.complete()?.token_ids)
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn try_submit_greedy_batch_cuda(
    logits: &Tensor,
    seqs: &[&mut Sequence],
    resident_input: &Tensor,
) -> Result<Option<CudaGreedyBatchSubmission>> {
    if !can_submit_greedy_batch_cuda_seqs(seqs) || !logits.device().is_cuda() {
        return Ok(None);
    }

    let logits = final_batched_logits(logits)?;
    if logits.dim(0)? != seqs.len() {
        return Ok(None);
    }
    let inner = seqs[0]
        .sampler()
        .submit_cuda_top1_batch_into(&logits, resident_input)?;
    Ok(Some(CudaGreedyBatchSubmission { inner }))
}

#[cfg(feature = "cuda")]
pub(crate) fn try_submit_greedy_batch_cuda_owned(
    logits: &Tensor,
    seqs: &[&mut Sequence],
) -> Result<Option<CudaGreedyBatchSubmission>> {
    if !can_submit_greedy_batch_cuda_seqs(seqs) || !logits.device().is_cuda() {
        return Ok(None);
    }
    let logits = final_batched_logits(logits)?;
    if logits.dim(0)? != seqs.len() {
        return Ok(None);
    }
    let inner = seqs[0].sampler().submit_cuda_top1_batch_owned(&logits)?;
    Ok(Some(CudaGreedyBatchSubmission { inner }))
}

pub(crate) fn can_submit_greedy_batch_cuda_seqs(seqs: &[&mut Sequence]) -> bool {
    #[cfg(feature = "cuda")]
    {
        !seqs.is_empty()
            && seqs.iter().all(|seq| {
                matches!(&seq.recognizer, SequenceRecognizer::None)
                    && seq.tool_call_state.is_none()
                    && !seq.sampling_logprob_required()
                    && seq.active_staged_speculative_len() == 0
                    && seq
                        .sampler()
                        .cuda_batch_sampling_plan(seq.return_logprobs())
                        .is_some_and(|plan| plan.kind.is_argmax())
            })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = seqs;
        false
    }
}

pub(crate) fn can_launch_one_token_lookahead(seqs: &[&mut Sequence], max_model_len: usize) -> bool {
    !seqs.is_empty()
        && seqs.iter().all(|seq| {
            !seq.is_finished_paged_attn()
                && one_token_stays_within_limits(
                    seq.generated_len(),
                    seq.max_generation_len(max_model_len),
                    seq.get_toks().len(),
                    max_model_len,
                )
        })
}

fn one_token_stays_within_limits(
    generated_len: usize,
    max_generation_len: usize,
    sequence_len: usize,
    max_model_len: usize,
) -> bool {
    generated_len.saturating_add(1) < max_generation_len && sequence_len < max_model_len
}

#[cfg(any(feature = "cuda", test))]
fn validate_greedy_batch_cardinality(
    sequence_count: usize,
    token_count: usize,
    commit_count: usize,
) -> Result<()> {
    if token_count != sequence_count || commit_count != sequence_count {
        candle_core::bail!(
            "greedy completion rows do not match the active batch: tokens={token_count}, commits={commit_count}, sequences={sequence_count}"
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn greedy_batch_will_finish(
    this: &dyn Pipeline,
    seqs: &[&mut Sequence],
    token_ids: &[u32],
    commit_rows: &[bool],
    disable_eos_stop: bool,
) -> Result<bool> {
    let metadata = this.get_metadata();
    greedy_batch_will_finish_with_metadata(
        seqs,
        token_ids,
        commit_rows,
        &metadata.eos_tok,
        metadata.max_seq_len,
        disable_eos_stop,
    )
}

#[cfg(any(feature = "cuda", test))]
fn greedy_batch_will_finish_with_metadata(
    seqs: &[&mut Sequence],
    token_ids: &[u32],
    commit_rows: &[bool],
    eos_tokens: &[u32],
    max_model_len: usize,
    disable_eos_stop: bool,
) -> Result<bool> {
    validate_greedy_batch_cardinality(seqs.len(), token_ids.len(), commit_rows.len())?;
    Ok(seqs
        .iter()
        .zip(token_ids)
        .zip(commit_rows)
        .any(|((seq, token), commit)| {
            if !*commit {
                return false;
            }
            let eos_tokens = seq.effective_eos_tokens(eos_tokens, disable_eos_stop);
            seq.is_done(*token, eos_tokens, max_model_len).is_some()
        }))
}

#[cfg(feature = "cuda")]
pub(crate) async fn finish_greedy_batch(
    this: &dyn Pipeline,
    seqs: &mut [&mut Sequence],
    token_ids: Vec<u32>,
    commit_rows: &[bool],
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
) -> Result<()> {
    validate_greedy_batch_cardinality(seqs.len(), token_ids.len(), commit_rows.len())?;
    let metadata = this.get_metadata();
    for ((token, commit), seq) in token_ids.into_iter().zip(commit_rows).zip(seqs.iter_mut()) {
        if !*commit {
            continue;
        }
        let next_token = Logprobs {
            token,
            logprob: 0.0,
            top_logprobs: None,
            bytes: None,
        };
        let eos_tok = seq.effective_eos_tokens(&metadata.eos_tok, disable_eos_stop);
        finish_or_add_toks_to_seq(this, prefix_cacher, seq, next_token, eos_tok, true).await?;
    }
    Ok(())
}

fn final_logits_row(logits: &Tensor) -> Result<Tensor> {
    logits.squeeze(0)?.squeeze(0)
}

fn stack_final_logits(logits: &[Tensor]) -> Result<Tensor> {
    if let [logits] = logits {
        return final_logits_row(logits)?.unsqueeze(0)?.to_dtype(DType::F32);
    }
    let rows = logits
        .iter()
        .map(final_logits_row)
        .collect::<Result<Vec<_>>>()?;
    Tensor::stack(&rows.iter().collect::<Vec<_>>(), 0)?
        .contiguous()?
        .to_dtype(DType::F32)
}

#[cfg(any(feature = "cuda", test))]
fn final_batched_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.dims();
    if dims.len() < 2 || dims[1..dims.len() - 1].iter().any(|&dim| dim != 1) {
        candle_core::bail!(
            "batched causal logits must have shape [batch, ..., vocab] with singleton middle dimensions, got {dims:?}"
        );
    }
    logits
        .contiguous()?
        .reshape((dims[0], dims[dims.len() - 1]))?
        .to_dtype(DType::F32)
}

fn coalesce_batch_logits_to_cpu(logits: Vec<Tensor>) -> Result<Vec<Tensor>> {
    if logits.len() <= 1 || logits.iter().all(|logits| logits.device().is_cpu()) {
        return Ok(logits);
    }
    let batch = stack_final_logits(&logits)?.to_device(&candle_core::Device::Cpu)?;
    (0..logits.len())
        .map(|idx| batch.i(idx)?.unsqueeze(0)?.unsqueeze(0))
        .collect()
}

#[cfg(feature = "cuda")]
fn try_sample_batch_cuda(
    logits: &CausalLogitsBatch,
    seqs: &[&mut Sequence],
    rng: &Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Result<Option<Vec<Result<Logprobs>>>> {
    let logits = match logits {
        CausalLogitsBatch::PerSequence(logits) => {
            if logits.is_empty()
                || logits.len() != seqs.len()
                || logits.iter().any(|logits| !logits.device().is_cuda())
            {
                return Ok(None);
            }
            stack_final_logits(logits)?
        }
        CausalLogitsBatch::Batched(logits) => {
            if logits.dim(0)? != seqs.len() || !logits.device().is_cuda() {
                return Ok(None);
            }
            final_batched_logits(logits)?
        }
    };

    let mut samplers_and_plans = Vec::with_capacity(seqs.len());
    let mut sampling_logprob_required = false;
    for seq in seqs {
        if !matches!(&seq.recognizer, SequenceRecognizer::None) || seq.tool_call_state.is_some() {
            return Ok(None);
        }
        let sampler = seq.sampler();
        let Some(plan) = sampler.cuda_batch_sampling_plan(seq.return_logprobs()) else {
            return Ok(None);
        };
        sampling_logprob_required |= seq.sampling_logprob_required();
        samplers_and_plans.push((sampler, plan));
    }

    let categorical = matches!(
        samplers_and_plans[0].1.kind,
        CudaBatchSamplingKind::Categorical
    );
    if samplers_and_plans
        .iter()
        .any(|(_, plan)| matches!(plan.kind, CudaBatchSamplingKind::Categorical) != categorical)
        || seqs.len() == 1 && !categorical
    {
        return Ok(None);
    }

    let all_argmax = samplers_and_plans
        .iter()
        .all(|(_, plan)| plan.kind.is_argmax());
    if all_argmax && !sampling_logprob_required {
        let packed = samplers_and_plans[0].0.sample_cuda_top1_batch(&logits)?;
        let sampled = std::iter::zip(packed, samplers_and_plans)
            .map(|(row, (sampler, _))| sampler.sample_cuda_top1_row(row.as_slice()))
            .collect();
        return Ok(Some(sampled));
    }

    let inverse_temperatures = samplers_and_plans
        .iter()
        .map(|(_, plan)| plan.inverse_temperature)
        .collect::<Vec<_>>();
    let inverse_temperatures = Tensor::from_vec(
        inverse_temperatures,
        samplers_and_plans.len(),
        logits.device(),
    )?;

    if categorical {
        let uniform = Uniform::new(0.0f32, 1.0f32).expect("valid unit uniform distribution");
        let uniforms = seqs
            .iter()
            .map(|seq| {
                let rng = seq.sampling_rng(rng);
                let mut rng = rng.lock().expect("could not lock rng mutex");
                uniform.sample(&mut *rng)
            })
            .collect::<Vec<_>>();
        let uniforms = Tensor::from_vec(uniforms, samplers_and_plans.len(), logits.device())?;
        let output = crate::ops::cuda_categorical_logits_f32_packed_batched(
            &logits,
            &inverse_temperatures,
            &uniforms,
        )?;
        let packed = output.packed.to_vec2::<f32>()?;
        let sampled = std::iter::zip(packed, samplers_and_plans)
            .map(|(row, (sampler, _))| sampler.sample_cuda_categorical_row(&row))
            .collect();
        return Ok(Some(sampled));
    }

    let common_k = samplers_and_plans
        .iter()
        .map(|(_, plan)| match plan.kind {
            CudaBatchSamplingKind::Greedy => 1,
            CudaBatchSamplingKind::TopK { k } => k,
            CudaBatchSamplingKind::Categorical => unreachable!(),
        })
        .max()
        .expect("batch is non-empty");
    let output =
        crate::ops::cuda_topk_logits_f32_packed_batched(&logits, common_k, &inverse_temperatures)?;
    let packed = output.packed.to_vec2::<f32>()?;
    Ok(Some(
        std::iter::zip(std::iter::zip(packed, samplers_and_plans), seqs.iter())
            .map(|((row, (sampler, plan)), seq)| {
                let rng = seq.sampling_rng(rng);
                let mut rng = rng.lock().expect("could not lock rng mutex");
                sampler.sample_cuda_topk_packed_row(&row, output.k, plan, &mut rng)
            })
            .collect(),
    ))
}

#[cfg(not(feature = "cuda"))]
fn try_sample_batch_cuda(
    _logits: &CausalLogitsBatch,
    _seqs: &[&mut Sequence],
    _rng: &Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Result<Option<Vec<Result<Logprobs>>>> {
    Ok(None)
}

/// Async sample optionally adding to trie.
#[allow(clippy::too_many_arguments)]
pub async fn sample_sequence(
    logits: Tensor,
    seq: &mut Sequence,
    return_logprobs: bool,
    eos_tok: Option<&[u32]>,
    llg_factory: Option<Arc<llguidance::ParserFactory>>,
    max_model_len: usize,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    use_async_pool: bool,
    sample_speculative: bool,
    multiple_sequences: bool,
) -> Result<Logprobs> {
    activate_required_tool_call_grammar(seq, llg_factory.as_ref(), max_model_len, false);
    let rng = seq.sampling_rng(&rng);

    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

    let sampler = seq.sampler();
    let ctx_clone = seq.get_toks().to_vec();
    let prompt_len = seq.prompt_tokens();
    let rng_clone = rng.clone();
    let logits_clone = logits.clone();
    let first_lobprobs_response = if use_async_pool {
        tokio_rayon::spawn(move || {
            sampler.sample(
                logits_clone,
                &ctx_clone,
                prompt_len,
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
            prompt_len,
            return_logprobs,
            rng_clone,
            sample_speculative,
            multiple_sequences,
        )?
    };

    let stop_token_requires_tool = seq.tool_call_state.as_ref().is_some_and(|state| {
        state.is_stop_token_blocked(first_lobprobs_response.token, eos_tok, seq.stop_tokens())
    });
    if stop_token_requires_tool {
        activate_required_tool_call_grammar(seq, llg_factory.as_ref(), max_model_len, true);
    }

    let bias_if_not_allowed = match &mut seq.recognizer {
        SequenceRecognizer::Llguidance(ref mut llg) => {
            // llguidance's EOS is <|endoftext|>-style; turn enders like <|im_end|> must pass once the grammar could stop
            let grammar_can_stop =
                llg.is_stopped() || llg.is_accepting().map_err(candle_core::Error::msg)?;
            let is_model_eos = |token: u32| eos_tok.is_some_and(|eos| eos.contains(&token));
            if !stop_token_requires_tool
                && (grammar_can_stop && is_model_eos(first_lobprobs_response.token)
                    || !llg.is_stopped()
                        && llg
                            .validate_tokens(&[first_lobprobs_response.token])
                            .unwrap_or(0)
                            == 1)
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
                    if grammar_can_stop && !stop_token_requires_tool {
                        for token in eos_tok.unwrap_or_default() {
                            if let Some(slot) = acc.get_mut(*token as usize) {
                                *slot = 0.0;
                            }
                        }
                    }

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
                        prompt_len,
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
                    prompt_len,
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
            let ends_turn = eos_tok
                .is_some_and(|eos| eos.contains(&second_logprobs_response.token))
                && llg.is_accepting().map_err(candle_core::Error::msg)?;
            if !llg.is_stopped() && !ends_turn {
                llg.consume_token(second_logprobs_response.token)
                    .map_err(candle_core::Error::msg)?;
            }
        }
        SequenceRecognizer::None => {}
    }

    if let SequenceRecognizer::Llguidance(ref llg) = seq.recognizer {
        if llg.is_stopped() {
            if let Some(state) = seq.tool_call_state.as_mut() {
                if state.clear_active_grammar() {
                    seq.recognizer = SequenceRecognizer::None;
                    tracing::debug!("Deactivated tool call grammar (body complete)");
                }
            }
        }
    }

    Ok(second_logprobs_response)
}

#[cfg(test)]
mod tests {
    use mistralrs_mcp::{Function, Tool, ToolType};
    use rand::SeedableRng;
    use std::{collections::HashMap, sync::Arc};
    use tokio::sync::{mpsc::channel, Mutex};

    use super::*;
    use crate::tools::{ToolCallState, ToolChoice};
    use crate::{
        sampler::Sampler,
        sequence::{SeqStepType, SequenceGroup},
    };

    fn terminal_test_sequence(
        stop_tokens: Vec<u32>,
        max_len: Option<usize>,
        ignore_eos: bool,
    ) -> Sequence {
        let (tx, _rx) = channel(1);
        let sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            32,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let group = Arc::new(Mutex::new(SequenceGroup::new(1, false, true, None)));
        Sequence::new_waiting(
            vec![1, 2, 3],
            "prompt".to_string(),
            0,
            0,
            0,
            tx,
            sampler,
            stop_tokens,
            vec![],
            max_len,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            ignore_eos,
            vec![],
            None,
        )
    }

    fn stochastic_test_sequence(seed: u64) -> Sequence {
        let (tx, _rx) = channel(1);
        let sampler = Sampler::new(
            Some(1.0),
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let group = Arc::new(Mutex::new(SequenceGroup::new(1, false, true, None)));
        Sequence::new_waiting(
            vec![1, 2, 3],
            "prompt".to_string(),
            0,
            0,
            0,
            tx,
            sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            false,
            vec![],
            Some(seed),
        )
    }

    async fn sample_test_token(
        seq: &mut Sequence,
        fallback: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> u32 {
        let logits = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.3, 0.4],
            (1, 1, 4),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        sample_sequence(
            logits, seq, false, None, None, 1024, fallback, false, false, false,
        )
        .await
        .unwrap()
        .token
    }

    #[tokio::test]
    async fn seeded_sampling_is_independent_of_sequence_order() {
        let fallback = || Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(999)));
        let (mut first_a, mut first_b) =
            (stochastic_test_sequence(42), stochastic_test_sequence(43));
        let a_then_b = (
            sample_test_token(&mut first_a, fallback()).await,
            sample_test_token(&mut first_b, fallback()).await,
        );

        let (mut second_a, mut second_b) =
            (stochastic_test_sequence(42), stochastic_test_sequence(43));
        let b = sample_test_token(&mut second_b, fallback()).await;
        let a = sample_test_token(&mut second_a, fallback()).await;

        assert_eq!(a_then_b, (a, b));
    }

    fn weather_tool() -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: Some("Get the current weather for a city.".to_string()),
                name: "get_weather".to_string(),
                parameters: None,
                strict: None,
            },
        }
    }

    #[test]
    fn gemma4_tool_call_suppresses_raw_content_without_suffix() {
        let tool = weather_tool();
        let mut state = ToolCallState::new(ToolChoice::Auto, Some(&[tool]), None).unwrap();
        let raw = r#"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}"#;

        let (content, tool_calls) = parse_text_and_tool_calls(raw, Some(&mut state)).unwrap();

        assert_eq!(content, None);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn canonical_tool_call_preserves_text_before_call() {
        let tool = weather_tool();
        let mut state = ToolCallState::new(ToolChoice::Auto, Some(&[tool]), None).unwrap();
        let raw = r#"I'll check that.<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#;

        let (content, tool_calls) = parse_text_and_tool_calls(raw, Some(&mut state)).unwrap();

        assert_eq!(content, Some("I'll check that.".to_string()));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn reasoning_stream_does_not_fallback_to_raw_delta() {
        let tool = weather_tool();
        let mut state = ToolCallState::new(ToolChoice::Auto, Some(&[tool]), None).unwrap();
        let raw = r#"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}"#;

        let (content, tool_calls) =
            parse_streaming_text_and_tool_calls(None, raw, true, Some(&mut state)).unwrap();

        assert_eq!(content, None);
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn non_reasoning_stream_uses_raw_delta() {
        let tool = weather_tool();
        let mut state = ToolCallState::new(ToolChoice::Auto, Some(&[tool]), None).unwrap();
        let raw = r#"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}"#;

        let (content, tool_calls) =
            parse_streaming_text_and_tool_calls(None, raw, false, Some(&mut state)).unwrap();

        assert_eq!(content, None);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn stack_final_logits_honors_view_offsets_and_order() {
        let backing = Tensor::from_vec(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            (3, 1, 1, 4),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let rows = vec![backing.i(2).unwrap(), backing.i(0).unwrap()];

        let packed = stack_final_logits(&rows).unwrap().to_vec2::<f32>().unwrap();

        assert_eq!(
            packed,
            vec![vec![8.0, 9.0, 10.0, 11.0], vec![0.0, 1.0, 2.0, 3.0]]
        );
    }

    #[test]
    fn final_batched_logits_flattens_singleton_axes() {
        let logits = Tensor::from_vec(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            (2, 1, 1, 4),
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let packed = final_batched_logits(&logits)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(
            packed,
            vec![vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]]
        );
    }

    #[test]
    fn lookahead_stops_before_known_length_boundaries() {
        assert!(one_token_stays_within_limits(4, 8, 12, 16));
        assert!(!one_token_stays_within_limits(7, 8, 12, 16));
        assert!(!one_token_stays_within_limits(4, 8, 16, 16));
    }

    #[test]
    fn greedy_terminal_prediction_matches_sequence_stop_rules() {
        let mut seq = terminal_test_sequence(vec![], None, false);
        let seqs = [&mut seq];
        assert!(
            greedy_batch_will_finish_with_metadata(&seqs, &[42], &[true], &[42], 1024, false,)
                .unwrap()
        );
        assert!(
            !greedy_batch_will_finish_with_metadata(&seqs, &[42], &[true], &[42], 1024, true,)
                .unwrap()
        );

        let mut seq = terminal_test_sequence(vec![9], None, false);
        let seqs = [&mut seq];
        assert!(
            greedy_batch_will_finish_with_metadata(&seqs, &[9], &[true], &[], 1024, false,)
                .unwrap()
        );
        assert!(
            !greedy_batch_will_finish_with_metadata(&seqs, &[9], &[false], &[], 1024, false,)
                .unwrap()
        );

        let mut seq = terminal_test_sequence(vec![], Some(1), false);
        let seqs = [&mut seq];
        assert!(
            greedy_batch_will_finish_with_metadata(&seqs, &[7], &[true], &[], 1024, false,)
                .unwrap()
        );
    }

    #[test]
    fn greedy_terminal_prediction_rejects_mismatched_rows() {
        let mut seq = terminal_test_sequence(vec![], None, false);
        let seqs = [&mut seq];
        assert!(
            greedy_batch_will_finish_with_metadata(&seqs, &[], &[true], &[], 1024, false,).is_err()
        );
        assert!(
            greedy_batch_will_finish_with_metadata(&seqs, &[7], &[], &[], 1024, false,).is_err()
        );
    }
}
