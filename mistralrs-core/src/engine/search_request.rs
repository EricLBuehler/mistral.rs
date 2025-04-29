use std::{borrow::Cow, sync::Arc, time::Instant};

use either::Either;
use indexmap::IndexMap;
use tokenizers::InputSequence;
use tracing::{level_filters::LevelFilter, Dispatch};

use crate::{
    get_mut_arcmutex,
    request::SearchContextSize,
    search::{self, SearchFunctionParameters, SearchResult},
    MessageContent, NormalRequest, RequestMessage, Response, ResponseOk, ToolCallResponse,
    WebSearchOptions,
};

use super::Engine;

async fn do_search(
    this: Arc<Engine>,
    mut second_request: NormalRequest,
    tool_calls: &ToolCallResponse,
    web_search_options: &WebSearchOptions,
) {
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
                SearchContextSize::High => 10000_usize,
                SearchContextSize::Medium => 7500_usize,
                SearchContextSize::Low => 3000_usize,
            };
        let mut results = tracing::dispatcher::with_default(&dispatch, || {
            search::run_search_tool(&tool_call_params)
                .unwrap()
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
        });

        // Sort increasing by tokenized length, if it fails, put it at the end.
        results.sort_by_key(|(_, len)| *len);

        {
            let device = get_mut_arcmutex!(this.pipeline).device();

            let Some(bert_pipeline) = &mut *get_mut_arcmutex!(this.bert_pipeline) else {
                unreachable!()
            };

            let decreasing_indexes = search::rag::compute_most_similar(
                &device,
                &tool_call_params.query,
                results.iter().map(|(res, _)| res).collect::<Vec<_>>(),
                bert_pipeline,
            )
            .unwrap();

            // Rerank the results
            let mut results_old = Vec::new();
            std::mem::swap(&mut results_old, &mut results);
            for &index in &decreasing_indexes {
                let mut current_result: (SearchResult, usize) = Default::default();
                std::mem::swap(&mut current_result, &mut results_old[index]);

                results.push(current_result);
            }
        }

        let mut used_results = Vec::new();
        let mut used_len = 0;
        for (item, len) in results {
            if used_len + len >= max_results_budget_toks {
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
            Either::Left(format!("{{\"output\": \"{tool_result}\"}}")),
        );
        messages.push(message);
    }

    // Recursion is enabled here!
    second_request.web_search_options = Some(web_search_options.clone());

    this.add_request(second_request).await;
}

/// The strategy is:
/// - Send the first request to allow a tool call
///     - If no, tool call, early return
/// - Proceed to `do_search`
///     1) Execute search
///     2) Rank by relevance
/// - Send final tool call, which is allowed to have web search for repeated queries.
pub(super) async fn search_request(this: Arc<Engine>, request: NormalRequest) {
    let Some(web_search_options) = request.web_search_options.clone() else {
        unreachable!()
    };
    let mut first_request = request.clone();
    // Actually add the search tool here
    first_request
        .tools
        .get_or_insert_with(Vec::new)
        .push(search::get_search_tool(&web_search_options).unwrap());

    let mut second_request = first_request.clone();
    first_request.web_search_options = None;
    second_request.web_search_options = None;

    let this_clone = this.clone();

    if !request.is_streaming {
        let handle = tokio::spawn(async move {
            let (new_sender, mut first_receiver) = tokio::sync::mpsc::channel(1);
            second_request.response = new_sender;
            std::mem::swap(&mut first_request.response, &mut second_request.response);

            this_clone.add_request(first_request).await;
            let ResponseOk::Done(done) = first_receiver.recv().await.unwrap().as_result().unwrap()
            else {
                unreachable!()
            };

            let tool_calls = match &done.choices[0].message.tool_calls {
                Some(tool_calls)
                    if tool_calls.len() == 1
                        && tool_calls[0].function.name == search::SEARCH_TOOL_NAME =>
                {
                    &tool_calls[0]
                }
                None => {
                    second_request
                        .response
                        .send(Response::Done(done))
                        .await
                        .unwrap();
                    return;
                }
                Some(_) => {
                    second_request
                        .response
                        .send(Response::Done(done))
                        .await
                        .unwrap();
                    return;
                }
            };

            do_search(this_clone, second_request, tool_calls, &web_search_options).await;
        });
        get_mut_arcmutex!(this.handles).push(handle);
    } else {
        let handle = tokio::spawn(async move {
            let (new_sender, mut first_receiver) = tokio::sync::mpsc::channel(1);
            second_request.response = new_sender;
            std::mem::swap(&mut first_request.response, &mut second_request.response);

            this_clone.add_request(first_request).await;
            let ResponseOk::Chunk(done) = first_receiver.recv().await.unwrap().as_result().unwrap()
            else {
                unreachable!()
            };
            second_request
                .response
                .send(Response::Chunk(done.clone()))
                .await
                .unwrap();

            let mut choice = done.choices[0].clone();

            while choice.finish_reason.is_none() {
                let ResponseOk::Chunk(done) =
                    first_receiver.recv().await.unwrap().as_result().unwrap()
                else {
                    unreachable!()
                };
                second_request
                    .response
                    .send(Response::Chunk(done.clone()))
                    .await
                    .unwrap();

                choice = done.choices[0].clone();
            }

            let tool_calls = match &choice.delta.tool_calls {
                Some(tool_calls)
                    if tool_calls.len() == 1
                        && tool_calls[0].function.name == search::SEARCH_TOOL_NAME =>
                {
                    &tool_calls[0]
                }
                None => {
                    return;
                }
                Some(_) => {
                    return;
                }
            };

            do_search(this_clone, second_request, tool_calls, &web_search_options).await;
        });
        get_mut_arcmutex!(this.handles).push(handle);
    }
}
