use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::{
    prefix_cacher::PrefixCacheManagerV2,
    sampler::Logprobs,
    sequence::{Sequence, SequenceRecognizer, SequenceState, StopReason},
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
    let parsed = state.finalize_for_response(raw_text, None, None)?;
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
    let parsed = state.parse_streaming(content_delta, raw_delta, has_reasoning_parser, false)?;
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
    let Some(grm) = grm else {
        return;
    };
    let Some(factory) = factory else {
        tracing::warn!("Cannot force required tool call: llguidance is unavailable");
        return;
    };
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
    // Include special tokens when tool calling is active (so tool parsers can see
    // delimiters like <tool_call>, [TOOL_CALLS], <|python_tag|>) or when think tag
    // mode is enabled (so <think>/<\/think> delimiters are visible in the output).
    let include_special = seq.tool_call_state.is_some() || seq.needs_special_tokens();
    let completion_bytes = tok_env
        .tok_trie()
        .decode_ext(&[logprobs.token], include_special);
    seq.add_token(logprobs.clone(), completion_bytes, &is_done);

    // If we can have a tool and we got a tool, stop the sequence early.
    // Doesn't conflict with the logic below because it does the same thing anyway.
    if let Ok(Some(d)) = seq.peek_delta() {
        if let Some(ref mut state) = seq.tool_call_state {
            let (_tool_use_still_possible, tool_use_is_done) = state.prefix_status(d.as_str())?;

            if tool_use_is_done {
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
        let text = seq.peek_delta().ok().flatten();
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

    // Handle streaming requests
    if seq.get_mut_group().is_streaming {
        let mut tool_use_still_possible = false;
        let mut tool_use_is_done = false;
        if let Ok(Some(d)) = seq.peek_delta() {
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
            let delta_result = seq.get_delta();
            if let Some(delta) = crate::handle_seq_error_stateaware_ok!(delta_result, seq) {
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
                        let parsed = state.parse_streaming(
                            content_delta.take(),
                            delta.as_str(),
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
                let recurrent_snapshots = if this.cache().is_hybrid() {
                    seq.recurrent_state_idx()
                        .and_then(|idx| this.cache().hybrid().snapshot_recurrent_state(idx).ok())
                } else {
                    None
                };
                prefix_cacher.add_sequence(seq, recurrent_snapshots);
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
                let has_reasoning_state = seq.has_reasoning_state();
                let parsed_content = if has_reasoning_state {
                    seq.get_response_content()
                } else {
                    None
                };
                let reasoning_content = if has_reasoning_state {
                    seq.get_reasoning_content()
                } else {
                    None
                };
                let parsed = if let Some(state) = seq.tool_call_state.as_mut() {
                    state.finalize_for_response(text.as_str(), parsed_content, reasoning_content)?
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
                let recurrent_snapshots = if this.cache().is_hybrid() {
                    seq.recurrent_state_idx()
                        .and_then(|idx| this.cache().hybrid().snapshot_recurrent_state(idx).ok())
                } else {
                    None
                };
                prefix_cacher.add_sequence(seq, recurrent_snapshots);
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
        let eos_tok = if disable_eos_stop {
            None
        } else {
            Some(&metadata.eos_tok[..])
        };

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
    let seqs_len = seqs.len();
    debug_assert_eq!(logits_seq.len(), seqs_len);

    let use_async_pool = seqs_len > 1;
    let metadata = this.get_metadata();
    let llg_factory = metadata.llg_factory.clone();
    let max_model_len = metadata.max_seq_len;
    let eos_toks = if disable_eos_stop {
        None
    } else {
        Some(metadata.eos_tok.clone())
    };

    let sampling_futures: Vec<_> = std::iter::zip(logits_seq, seqs.iter_mut())
        .map(|(logits_per_seq, seq)| {
            let return_logprobs = seq.return_logprobs();
            sample_sequence(
                logits_per_seq,
                seq,
                return_logprobs,
                eos_toks.as_deref(),
                llg_factory.clone(),
                max_model_len,
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
    eos_tok: Option<&[u32]>,
    llg_factory: Option<Arc<llguidance::ParserFactory>>,
    max_model_len: usize,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    use_async_pool: bool,
    sample_speculative: bool,
    multiple_sequences: bool,
) -> Result<Logprobs> {
    activate_required_tool_call_grammar(seq, llg_factory.as_ref(), max_model_len, false);

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

    let stop_token_requires_tool = seq.tool_call_state.as_ref().is_some_and(|state| {
        state.is_stop_token_blocked(first_lobprobs_response.token, eos_tok, seq.stop_tokens())
    });
    if stop_token_requires_tool {
        activate_required_tool_call_grammar(seq, llg_factory.as_ref(), max_model_len, true);
    }

    let bias_if_not_allowed = match &mut seq.recognizer {
        SequenceRecognizer::Llguidance(ref mut llg) => {
            if !llg.is_stopped()
                && llg
                    .validate_tokens(&[first_lobprobs_response.token])
                    .unwrap_or(0)
                    == 1
                && !stop_token_requires_tool
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

    use super::*;
    use crate::tools::{ToolCallState, ToolChoice};

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
}
