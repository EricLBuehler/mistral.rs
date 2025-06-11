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
    MessageContent, NormalRequest, RequestMessage, Response, ResponseOk, ToolCallResponse,
    ToolChoice, WebSearchOptions,
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
    println!();
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
        let mut results = tokio::task::block_in_place(|| {
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
                    .collect::<Vec<_>>()
            })
        });

        // Sort increasing by tokenized length, if it fails, put it at the end.
        results.sort_by_key(|(_, len)| *len);

        {
            // Determine ranking: use embedding model if available, otherwise fallback to BM25
            let decreasing_indexes: Vec<usize> = if let Some(bert_pipeline) =
                &mut *get_mut_arcmutex!(this.bert_pipeline)
            {
                // Semantic reranking with embeddings
                let device = get_mut_arcmutex!(this.pipeline).device();
                search::rag::compute_most_similar(
                    &device,
                    &tool_call_params.query,
                    results.iter().map(|(res, _)| res).collect::<Vec<_>>(),
                    bert_pipeline,
                )
                .unwrap()
            } else {
                tracing::warn!("No embedding model loaded; falling back to BM25 ranking for web search results.");

                // Build an Embedder over the corpus, fitting to the entire set of documents.
                //    - Language::English is chosen here
                //    - This computes an in‑memory sparse embedding for each document.

                let docs: Vec<String> =
                    results.iter().map(|(res, _)| res.content.clone()).collect();
                let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();

                let embedder: Embedder =
                    EmbedderBuilder::with_fit_to_corpus(Language::English, &doc_refs).build();

                // Initialize a Scorer keyed by usize (document index type).
                let mut scorer = Scorer::<usize>::new();

                // For each document, compute its embedding and upsert into the scorer.
                for (i, doc_text) in docs.iter().enumerate() {
                    let doc_embedding = embedder.embed(doc_text);
                    scorer.upsert(&i, doc_embedding);
                }

                // Embed the query string into the same sparse embedding space.
                let query_embedding = embedder.embed(&tool_call_params.query);

                // Score all documents individually
                let mut scored_docs: Vec<ScoredDocument<usize>> = docs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, _)| {
                        scorer
                            .score(&i, &query_embedding)
                            .map(|score| ScoredDocument { id: i, score })
                    })
                    .collect();

                // Sort the scored documents by descending `score` (f32).
                scored_docs.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Extract only the document indices (usize) in ranked order.
                let decreasing_indexes: Vec<usize> =
                    scored_docs.into_iter().map(|d| d.id).collect();

                decreasing_indexes
            };
            // Reorder results according to ranking
            let mut old = Vec::new();
            std::mem::swap(&mut old, &mut results);
            for &idx in &decreasing_indexes {
                let mut item: (SearchResult, usize) = Default::default();
                std::mem::swap(&mut item, &mut old[idx]);
                results.push(item);
            }
        }

        let mut used_results = Vec::new();
        let mut used_len = 0;
        for (item, len) in results {
            if used_len + len > max_results_budget_toks {
                break;
            }
            // So the info! below gets the correct value
            used_len += len;
            used_results.push(item);
        }

        let tool_result = serde_json::to_string(&used_results)
            .unwrap()
            .replace("\\n", "\n")
            .replace("\\\"", "\"")
            .replace("\\\\", "\\");
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
    println!();
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
        tracing::info!("Called tool `{}`.", tool_calls.function.name);
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
        tracing::info!("Called tool `{}`.", tool_calls.function.name);
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
/// 1. Send a “probe” request that may call the search/extract tools.  
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

            // Kick the request into the engine.
            this_clone.add_request(current).await;

            // ----------------------- NON-STREAMING ------------------------
            if !is_streaming {
                let done = match receiver.recv().await.unwrap().as_result().unwrap() {
                    ResponseOk::Done(done) => done,
                    other => {
                        match other {
                            ResponseOk::Chunk(res) => {
                                user_sender.send(Response::Chunk(res)).await.unwrap()
                            }
                            ResponseOk::CompletionChunk(res) => user_sender
                                .send(Response::CompletionChunk(res))
                                .await
                                .unwrap(),
                            ResponseOk::Done(_) => unreachable!(),
                            ResponseOk::CompletionDone(res) => user_sender
                                .send(Response::CompletionDone(res))
                                .await
                                .unwrap(),
                            ResponseOk::ImageGeneration(res) => user_sender
                                .send(Response::ImageGeneration(res))
                                .await
                                .unwrap(),
                            ResponseOk::Raw {
                                logits_chunks,
                                tokens,
                            } => user_sender
                                .send(Response::Raw {
                                    logits_chunks,
                                    tokens,
                                })
                                .await
                                .unwrap(),
                            ResponseOk::Speech {
                                pcm,
                                rate,
                                channels,
                            } => user_sender
                                .send(Response::Speech {
                                    pcm,
                                    rate,
                                    channels,
                                })
                                .await
                                .unwrap(),
                        };
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

                // Tool requested → build the next turn.
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
                    match resp.as_result().unwrap() {
                        ResponseOk::Chunk(chunk) => {
                            // Forward every content‑bearing chunk immediately, but
                            // *suppress* the ones that initiate a tool call. This ensures
                            // the user sees the assistant’s streamed text from the very
                            // first probe turn while still hiding the internal
                            // search/extract trigger.
                            let first_choice = &chunk.choices[0];
                            if first_choice.delta.tool_calls.is_none() {
                                user_sender
                                    .send(Response::Chunk(chunk.clone()))
                                    .await
                                    .unwrap();
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
                        other => {
                            match other {
                                ResponseOk::Chunk(_) => unreachable!(),
                                ResponseOk::CompletionChunk(res) => user_sender
                                    .send(Response::CompletionChunk(res))
                                    .await
                                    .unwrap(),
                                ResponseOk::Done(res) => {
                                    user_sender.send(Response::Done(res)).await.unwrap()
                                }
                                ResponseOk::CompletionDone(res) => user_sender
                                    .send(Response::CompletionDone(res))
                                    .await
                                    .unwrap(),
                                ResponseOk::ImageGeneration(res) => user_sender
                                    .send(Response::ImageGeneration(res))
                                    .await
                                    .unwrap(),
                                ResponseOk::Raw {
                                    logits_chunks,
                                    tokens,
                                } => user_sender
                                    .send(Response::Raw {
                                        logits_chunks,
                                        tokens,
                                    })
                                    .await
                                    .unwrap(),
                                ResponseOk::Speech {
                                    pcm,
                                    rate,
                                    channels,
                                } => user_sender
                                    .send(Response::Speech {
                                        pcm,
                                        rate,
                                        channels,
                                    })
                                    .await
                                    .unwrap(),
                            };
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
                    break; // No more tool calls → done.
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
