use std::{borrow::Cow, sync::Arc, time::Instant};

use bm25::{Embedder, EmbedderBuilder, Language, ScoredDocument, Scorer};
use either::Either;
use indexmap::IndexMap;
use tokenizers::InputSequence;
use tracing::{level_filters::LevelFilter, Dispatch};

use crate::{
    get_mut_arcmutex,
    request::SearchContextSize,
    search::{self, ExtractFunctionParameters, SearchFunctionParameters, SearchResult},
    MessageContent, NormalRequest, RequestMessage, Response, ToolCallResponse, ToolChoice,
    WebSearchOptions,
};

use super::Engine;

async fn do_search(
    this: Arc<Engine>,
    mut second_request: NormalRequest,
    tool_calls: &ToolCallResponse,
    web_search_options: &WebSearchOptions,
) -> NormalRequest {
    let messages = match &mut second_request.messages {
        RequestMessage::Chat { messages, .. } | RequestMessage::VisionChat { messages, .. } => {
            messages
        }
        _ => unreachable!(),
    };

    // Add assistant call message
    {
        let mut message: IndexMap<String, MessageContent> = IndexMap::new();
        message.insert("role".to_string(), Either::Left("assistant".to_string()));
        message.insert(
            "content".to_string(),
            Either::Left(format!(
                "{{\"name\":\"{}\",\"arguments\":\"{}\"}}",
                tool_calls.function.name, tool_calls.function.arguments
            )),
        );
        messages.push(message);
    }
    let tool_call_params: SearchFunctionParameters =
        serde_json::from_str(&tool_calls.function.arguments).unwrap();
    tracing::info!(
        "Called search tool with query `{}`.",
        tool_call_params.query
    );

    let start = Instant::now();
    // Add tool response
    {
        let tokenizer = get_mut_arcmutex!(this.pipeline)
            .tokenizer()
            .expect("A tokenizer is expected for non-diffusion models.");

        // Allow `info` and below; suppress `warn`
        let subscriber = tracing_subscriber::fmt::Subscriber::builder()
            .with_max_level(LevelFilter::INFO)
            .finish();
        let dispatch = Dispatch::new(subscriber);

        // Manage context size by # of tokens. Apply default here.
        let max_results_budget_toks =
            match web_search_options.search_context_size.unwrap_or_default() {
                SearchContextSize::High => 16384_usize,
                SearchContextSize::Medium => 8192_usize,
                SearchContextSize::Low => 4096_usize,
            };
        let (results, result_token_lens): (Vec<SearchResult>, Vec<usize>) =
            tokio::task::block_in_place(|| {
                tracing::dispatcher::with_default(&dispatch, || {
                    let base_results = if let Some(cb) = &this.search_callback {
                        cb(&tool_call_params).unwrap()
                    } else {
                        search::run_search_tool(&tool_call_params).unwrap()
                    };
                    base_results
                        .into_iter()
                        .map(|mut result| {
                            result = result
                                .cap_content_len(&tokenizer, max_results_budget_toks)
                                .unwrap();
                            let len = {
                                let inp = InputSequence::Raw(Cow::from(&result.content));
                                tokenizer
                                    .encode_fast(inp, false)
                                    .map(|x| x.len())
                                    .unwrap_or(usize::MAX)
                            };
                            (result, len)
                        })
                        .unzip()
                })
            });

        let mut combined: Vec<(SearchResult, usize)> = results
            .into_iter()
            .zip(result_token_lens.into_iter())
            .collect();
        combined.sort_by_key(|(_, len)| *len);
        let (results, result_token_lens): (Vec<SearchResult>, Vec<usize>) =
            combined.into_iter().unzip();

        let mut used_results = Vec::new();
        let mut used_len = 0;

        if let Some(search_pipeline) = &mut *get_mut_arcmutex!(this.search_pipeline) {
            let ranked_chunks = search::rag::rank_document_chunks(
                &tool_call_params.query,
                &results,
                search_pipeline,
            )
            .unwrap();

            if ranked_chunks.is_empty() {
                for (result, len) in results.iter().zip(result_token_lens.iter()) {
                    if used_len + len > max_results_budget_toks {
                        break;
                    }
                    used_len += len;
                    used_results.push(result.clone());
                }
            } else {
                for chunk in ranked_chunks {
                    if chunk.token_len == 0 {
                        continue;
                    }
                    if used_len + chunk.token_len > max_results_budget_toks {
                        break;
                    }
                    used_len += chunk.token_len;
                    let mut chunk_result = results[chunk.result_index].clone();
                    chunk_result.content = chunk.content;
                    used_results.push(chunk_result);
                }
            }
        } else {
            tracing::warn!(
                "No embedding model loaded; falling back to BM25 ranking for web search results."
            );

            let docs: Vec<String> = results.iter().map(|res| res.content.clone()).collect();
            let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();

            let embedder: Embedder =
                EmbedderBuilder::with_fit_to_corpus(Language::English, &doc_refs).build();

            let mut scorer = Scorer::<usize>::new();
            for (i, doc_text) in docs.iter().enumerate() {
                let doc_embedding = embedder.embed(doc_text);
                scorer.upsert(&i, doc_embedding);
            }

            let query_embedding = embedder.embed(&tool_call_params.query);

            let mut scored_docs: Vec<ScoredDocument<usize>> = docs
                .iter()
                .enumerate()
                .filter_map(|(i, _)| {
                    scorer
                        .score(&i, &query_embedding)
                        .map(|score| ScoredDocument { id: i, score })
                })
                .collect();

            scored_docs.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for doc in scored_docs {
                let idx = doc.id;
                let len = result_token_lens[idx];
                if used_len + len > max_results_budget_toks {
                    break;
                }
                used_len += len;
                used_results.push(results[idx].clone());
            }
        }

        let tool_result = serde_json::to_string(&serde_json::json!({
            "output": used_results
        }))
        .unwrap();
        let end = Instant::now();
        tracing::info!(
            "Web search executed in {:.2}s, using {used_len} tokens of {} search results.",
            (end - start).as_secs_f32(),
            used_results.len()
        );

        let mut message: IndexMap<String, MessageContent> = IndexMap::new();
        message.insert("role".to_string(), Either::Left("tool".to_string()));
        message.insert(
            "content".to_string(),
            Either::Left(
                // Format the tool output JSON and append the search tool description for context
                format!(
                    "{}\n\n{}\n\n{}",
                    tool_result,
                    search::SEARCH_DESCRIPTION,
                    search::EXTRACT_DESCRIPTION,
                ),
            ),
        );
        messages.push(message);
    }

    // Allow the assistant to invoke tools again on the next turn
    second_request.tool_choice = Some(ToolChoice::Auto);
    // Recursion is enabled here!
    second_request.web_search_options = Some(web_search_options.clone());

    second_request
}

async fn do_extraction(
    this: Arc<Engine>,
    mut second_request: NormalRequest,
    tool_calls: &ToolCallResponse,
    web_search_options: &WebSearchOptions,
) -> NormalRequest {
    let messages = match &mut second_request.messages {
        RequestMessage::Chat { messages, .. } | RequestMessage::VisionChat { messages, .. } => {
            messages
        }
        _ => unreachable!(),
    };

    // Add assistant call message
    {
        let mut message: IndexMap<String, MessageContent> = IndexMap::new();
        message.insert("role".to_string(), Either::Left("assistant".to_string()));
        message.insert(
            "content".to_string(),
            Either::Left(format!(
                "{{\"name\":\"{}\",\"arguments\":\"{}\"}}",
                tool_calls.function.name, tool_calls.function.arguments
            )),
        );
        messages.push(message);
    }
    let tool_call_params: ExtractFunctionParameters =
        serde_json::from_str(&tool_calls.function.arguments).unwrap();
    tracing::info!(
        "Called extrcation tool with query `{}`.",
        tool_call_params.url
    );

    let start = Instant::now();
    // Add tool response
    {
        let tokenizer = get_mut_arcmutex!(this.pipeline)
            .tokenizer()
            .expect("A tokenizer is expected for non-diffusion models.");

        // Allow `info` and below; suppress `warn`
        let subscriber = tracing_subscriber::fmt::Subscriber::builder()
            .with_max_level(LevelFilter::INFO)
            .finish();
        let dispatch = Dispatch::new(subscriber);

        // Manage context size by # of tokens. Apply default here.
        let max_results_budget_toks =
            match web_search_options.search_context_size.unwrap_or_default() {
                SearchContextSize::High => 16384_usize,
                SearchContextSize::Medium => 8192_usize,
                SearchContextSize::Low => 4096_usize,
            };

        let res = {
            let extract_result = tokio::task::block_in_place(|| {
                tracing::dispatcher::with_default(&dispatch, || {
                    search::run_extract_tool(&tool_call_params).unwrap()
                })
            });
            extract_result
                .cap_content_len(&tokenizer, max_results_budget_toks)
                .unwrap()
        };

        let tool_result = serde_json::to_string(&res)
            .unwrap()
            .replace("\\n", "\n")
            .replace("\\\"", "\"")
            .replace("\\\\", "\\");
        let end = Instant::now();
        let used_len = {
            let inp = InputSequence::Raw(Cow::from(&tool_result));
            tokenizer
                .encode_fast(inp, false)
                .map(|x| x.len())
                .unwrap_or(usize::MAX)
        };
        tracing::info!(
            "Extraction executed in {:.2}s, using {used_len} tokens.",
            (end - start).as_secs_f32(),
        );

        let mut message: IndexMap<String, MessageContent> = IndexMap::new();
        message.insert("role".to_string(), Either::Left("tool".to_string()));
        message.insert(
            "content".to_string(),
            Either::Left(
                // Format the tool output JSON and append the search tool description for context
                format!(
                    "{{\"output\": \"{}\"}}\n\n{}\n\n{}",
                    tool_result,
                    search::SEARCH_DESCRIPTION,
                    search::EXTRACT_DESCRIPTION,
                ),
            ),
        );
        messages.push(message);
    }

    // Allow the assistant to invoke tools again on the next turn
    second_request.tool_choice = Some(ToolChoice::Auto);
    // Recursion is enabled here!
    second_request.web_search_options = Some(web_search_options.clone());

    second_request
}

async fn do_custom_tool(
    this: Arc<Engine>,
    mut second_request: NormalRequest,
    tool_calls: &ToolCallResponse,
) -> NormalRequest {
    let messages = match &mut second_request.messages {
        RequestMessage::Chat { messages, .. } | RequestMessage::VisionChat { messages, .. } => {
            messages
        }
        _ => unreachable!(),
    };

    {
        let mut message: IndexMap<String, MessageContent> = IndexMap::new();
        message.insert("role".to_string(), Either::Left("assistant".to_string()));
        message.insert(
            "content".to_string(),
            Either::Left(format!(
                "{{\"name\":\"{}\",\"arguments\":\"{}\"}}",
                tool_calls.function.name, tool_calls.function.arguments
            )),
        );
        messages.push(message);
    }

    let result = if let Some(cb) = this.tool_callbacks.get(&tool_calls.function.name) {
        cb(&tool_calls.function).unwrap_or_else(|e| {
            tracing::error!(
                "Error when calling tool `{}`: {e}",
                tool_calls.function.name
            );
            format!("ERROR: {e}")
        })
    } else if let Some(callback_with_tool) = this
        .tool_callbacks_with_tools
        .get(&tool_calls.function.name)
    {
        (callback_with_tool.callback)(&tool_calls.function).unwrap_or_else(|e| {
            tracing::error!(
                "Error when calling tool `{}`: {e}",
                tool_calls.function.name
            );
            format!("ERROR: {e}")
        })
    } else {
        tracing::error!(
            "Attempted to call tool `{}`, but it doesn't exist.",
            tool_calls.function.name
        );
        format!("ERROR: no tool callback for {}", tool_calls.function.name)
    };

    {
        let mut message: IndexMap<String, MessageContent> = IndexMap::new();
        message.insert("role".to_string(), Either::Left("tool".to_string()));
        message.insert("content".to_string(), Either::Left(result));
        messages.push(message);
    }

    second_request.tool_choice = Some(ToolChoice::Auto);
    second_request
}

/// Drive one or more web-search / extraction rounds without recursion.
///
/// Strategy:
/// 1. Send a "probe" request that may call the search/extract tools.
/// 2. If such a tool is called, run it (`do_search` / `do_extraction`) to
///    mutate the conversational context and build the next request.
/// 3. Repeat until no further tool call is made.
/// 4. Forward every user-visible reply **except** the first, which is just the
///    probe that discovers whether a tool call is needed.
pub(super) async fn search_request(this: Arc<Engine>, request: NormalRequest) {
    let web_search_options = request.web_search_options.clone();

    // The sender that ultimately delivers data back to the caller.
    let user_sender = request.response.clone();
    let is_streaming = request.is_streaming;

    // ---------------------------------------------------------------------
    // Build the *first* request (the “probe”).
    // ---------------------------------------------------------------------
    let mut probe = request.clone();
    if let Some(ref opts) = web_search_options {
        probe
            .tools
            .get_or_insert_with(Vec::new)
            .extend(search::get_search_tools(opts).unwrap());
    }

    // Add Tool definitions from tool callbacks with tools if they're not already present
    if !this.tool_callbacks_with_tools.is_empty() {
        let tools = probe.tools.get_or_insert_with(Vec::new);
        let existing_tool_names: Vec<String> =
            tools.iter().map(|t| t.function.name.clone()).collect();

        for (name, callback_with_tool) in &this.tool_callbacks_with_tools {
            if !existing_tool_names.contains(name) {
                tools.push(callback_with_tool.tool.clone());
            }
        }
    }

    probe.tool_choice = Some(ToolChoice::Auto);
    // Prevent accidental infinite recursion on the probe itself.
    probe.web_search_options = None;

    // The conversation context that the user *will* see.
    let mut visible_req = probe.clone();
    visible_req.response = user_sender.clone();

    // We'll drive everything inside a single spawned task.
    let this_clone = this.clone();
    let handle = tokio::spawn(async move {
        // `current` is what we actually dispatch each loop.
        // The very first time that is the hidden probe.
        let mut current = probe;

        loop {
            // Each dispatch gets its own one-shot channel so we can peek at
            // the response before (optionally) forwarding it.
            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
            current.response = sender;

            // Kick the request into the engine via the channel.
            // This avoids lock contention with the main engine loop.
            let _ = this_clone
                .tx
                .send(crate::request::Request::Normal(Box::new(current)))
                .await;

            // ----------------------- NON-STREAMING ------------------------
            if !is_streaming {
                let resp = receiver.recv().await.unwrap();
                // Handle the response, forwarding errors and non-Done responses to user
                let done = match resp {
                    Response::Done(done) => done,
                    // Forward error responses to user and return
                    Response::InternalError(e) => {
                        let _ = user_sender.send(Response::InternalError(e)).await;
                        return;
                    }
                    Response::ValidationError(e) => {
                        let _ = user_sender.send(Response::ValidationError(e)).await;
                        return;
                    }
                    Response::ModelError(msg, resp) => {
                        let _ = user_sender.send(Response::ModelError(msg, resp)).await;
                        return;
                    }
                    // Forward other response types to user and return
                    Response::Chunk(res) => {
                        let _ = user_sender.send(Response::Chunk(res)).await;
                        return;
                    }
                    Response::CompletionChunk(res) => {
                        let _ = user_sender.send(Response::CompletionChunk(res)).await;
                        return;
                    }
                    Response::CompletionModelError(msg, resp) => {
                        let _ = user_sender
                            .send(Response::CompletionModelError(msg, resp))
                            .await;
                        return;
                    }
                    Response::CompletionDone(res) => {
                        let _ = user_sender.send(Response::CompletionDone(res)).await;
                        return;
                    }
                    Response::ImageGeneration(res) => {
                        let _ = user_sender.send(Response::ImageGeneration(res)).await;
                        return;
                    }
                    Response::Raw {
                        logits_chunks,
                        tokens,
                    } => {
                        let _ = user_sender
                            .send(Response::Raw {
                                logits_chunks,
                                tokens,
                            })
                            .await;
                        return;
                    }
                    Response::Embeddings {
                        embeddings,
                        prompt_tokens,
                        total_tokens,
                    } => {
                        let _ = user_sender
                            .send(Response::Embeddings {
                                embeddings,
                                prompt_tokens,
                                total_tokens,
                            })
                            .await;
                        return;
                    }
                    Response::Speech {
                        pcm,
                        rate,
                        channels,
                    } => {
                        let _ = user_sender
                            .send(Response::Speech {
                                pcm,
                                rate,
                                channels,
                            })
                            .await;
                        return;
                    }
                };

                // Did the assistant ask to run a tool?
                let tc_opt = match &done.choices[0].message.tool_calls {
                    Some(calls) if calls.len() == 1 => Some(&calls[0]),
                    _ => None,
                };

                // No tool call? We are finished.
                if tc_opt.is_none() {
                    user_sender
                        .send(Response::Done(done.clone()))
                        .await
                        .unwrap();
                    return;
                }

                // Tool requested -> build the next turn.
                let tc = tc_opt.unwrap();
                let next_visible = if search::search_tool_called(&tc.function.name) {
                    let web_search_options = web_search_options.as_ref().unwrap();
                    if tc.function.name == search::SEARCH_TOOL_NAME {
                        do_search(this_clone.clone(), visible_req, tc, web_search_options).await
                    } else {
                        do_extraction(this_clone.clone(), visible_req, tc, web_search_options).await
                    }
                } else {
                    do_custom_tool(this_clone.clone(), visible_req, tc).await
                };

                // The fresh request becomes both the user-visible context and
                // the next `current` we will dispatch.
                visible_req = next_visible.clone();
                visible_req.response = user_sender.clone();
                current = visible_req.clone();
            }
            // ------------------------- STREAMING -------------------------
            else {
                // We need the *last* chunk to see whether a tool was called.
                let mut last_choice = None;

                while let Some(resp) = receiver.recv().await {
                    match resp {
                        Response::Chunk(chunk) => {
                            // Forward every content‑bearing chunk immediately, but
                            // *suppress* the ones that initiate a tool call. This ensures
                            // the user sees the assistant's streamed text from the very
                            // first probe turn while still hiding the internal
                            // search/extract trigger.
                            let first_choice = &chunk.choices[0];
                            if first_choice.delta.tool_calls.is_none() {
                                let _ = user_sender.send(Response::Chunk(chunk.clone())).await;
                            }
                            last_choice = Some(first_choice.clone());

                            // Stop once the model marks completion.
                            if last_choice
                                .as_ref()
                                .and_then(|c| c.finish_reason.as_ref())
                                .is_some()
                            {
                                break;
                            }
                        }
                        // Forward error responses to user and return
                        Response::InternalError(e) => {
                            let _ = user_sender.send(Response::InternalError(e)).await;
                            return;
                        }
                        Response::ValidationError(e) => {
                            let _ = user_sender.send(Response::ValidationError(e)).await;
                            return;
                        }
                        Response::ModelError(msg, resp) => {
                            let _ = user_sender.send(Response::ModelError(msg, resp)).await;
                            return;
                        }
                        // Forward other response types to user and return
                        Response::CompletionChunk(res) => {
                            let _ = user_sender.send(Response::CompletionChunk(res)).await;
                            return;
                        }
                        Response::CompletionModelError(msg, resp) => {
                            let _ = user_sender
                                .send(Response::CompletionModelError(msg, resp))
                                .await;
                            return;
                        }
                        Response::Done(res) => {
                            let _ = user_sender.send(Response::Done(res)).await;
                            return;
                        }
                        Response::CompletionDone(res) => {
                            let _ = user_sender.send(Response::CompletionDone(res)).await;
                            return;
                        }
                        Response::ImageGeneration(res) => {
                            let _ = user_sender.send(Response::ImageGeneration(res)).await;
                            return;
                        }
                        Response::Raw {
                            logits_chunks,
                            tokens,
                        } => {
                            let _ = user_sender
                                .send(Response::Raw {
                                    logits_chunks,
                                    tokens,
                                })
                                .await;
                            return;
                        }
                        Response::Embeddings {
                            embeddings,
                            prompt_tokens,
                            total_tokens,
                        } => {
                            let _ = user_sender
                                .send(Response::Embeddings {
                                    embeddings,
                                    prompt_tokens,
                                    total_tokens,
                                })
                                .await;
                            return;
                        }
                        Response::Speech {
                            pcm,
                            rate,
                            channels,
                        } => {
                            let _ = user_sender
                                .send(Response::Speech {
                                    pcm,
                                    rate,
                                    channels,
                                })
                                .await;
                            return;
                        }
                    }
                }

                let Some(choice) = last_choice else { break };

                let tc_opt = match &choice.delta.tool_calls {
                    Some(calls) if calls.len() == 1 => Some(&calls[0]),
                    _ => None,
                };

                if tc_opt.is_none() {
                    break; // No more tool calls -> done.
                }

                let tc = tc_opt.unwrap();
                let next_visible = if search::search_tool_called(&tc.function.name) {
                    let web_search_options = web_search_options.as_ref().unwrap();
                    if tc.function.name == search::SEARCH_TOOL_NAME {
                        do_search(this_clone.clone(), visible_req, tc, web_search_options).await
                    } else {
                        do_extraction(this_clone.clone(), visible_req, tc, web_search_options).await
                    }
                } else {
                    do_custom_tool(this_clone.clone(), visible_req, tc).await
                };

                visible_req = next_visible.clone();
                visible_req.response = user_sender.clone();
                current = visible_req.clone();
            }
        }
    });

    get_mut_arcmutex!(this.handles).push(handle);
}
