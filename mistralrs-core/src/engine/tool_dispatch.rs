//! Tool execution dispatch.
//!
//! Centralises the logic for executing a tool call (web search, content
//! extraction, or user-registered callback) and returning the result as a
//! string.  The orchestration loop in `search_request.rs` is responsible
//! for message construction and request mutation; this module only runs
//! the tool and returns its output.

use std::{borrow::Cow, sync::Arc, time::Instant};

use bm25::{Embedder, EmbedderBuilder, Language, ScoredDocument, Scorer};
use tokenizers::InputSequence;
use tracing::{level_filters::LevelFilter, Dispatch};

use crate::{
    get_mut_arcmutex,
    request::SearchContextSize,
    search::{self, ExtractFunctionParameters, SearchFunctionParameters, SearchResult},
    ToolCallResponse, WebSearchOptions,
};

use super::Engine;

/// The result of executing a tool call.
pub(super) struct ToolResult {
    pub content: String,
}

/// Resolve the token budget from [`SearchContextSize`].
fn token_budget(opts: &WebSearchOptions) -> usize {
    match opts.search_context_size.unwrap_or_default() {
        SearchContextSize::High => 16384,
        SearchContextSize::Medium => 8192,
        SearchContextSize::Low => 4096,
    }
}

/// Create a tracing dispatcher that allows `info` and below but suppresses
/// `warn` (the HTTP scraping crates are chatty).
fn quiet_dispatch() -> Dispatch {
    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(LevelFilter::INFO)
        .finish();
    Dispatch::new(subscriber)
}

// ── Search ─────────────────────────────────────────────────────────────────

pub(super) async fn execute_search(
    engine: &Arc<Engine>,
    tc: &ToolCallResponse,
    opts: &WebSearchOptions,
) -> ToolResult {
    let params: SearchFunctionParameters =
        serde_json::from_str(&tc.function.arguments).unwrap();
    tracing::info!("Called search tool with query `{}`.", params.query);

    let start = Instant::now();
    let tokenizer = get_mut_arcmutex!(engine.pipeline)
        .tokenizer()
        .expect("A tokenizer is expected for non-diffusion models.");
    let dispatch = quiet_dispatch();
    let max_toks = token_budget(opts);

    // Fetch and cap results.
    let (results, result_token_lens): (Vec<SearchResult>, Vec<usize>) =
        tokio::task::block_in_place(|| {
            tracing::dispatcher::with_default(&dispatch, || {
                let base = if let Some(cb) = &engine.search_callback {
                    cb(&params).unwrap()
                } else {
                    search::run_search_tool(&params).unwrap()
                };
                base.into_iter()
                    .map(|mut r| {
                        r = r.cap_content_len(&tokenizer, max_toks).unwrap();
                        let len = {
                            let inp = InputSequence::Raw(Cow::from(&r.content));
                            tokenizer
                                .encode_fast(inp, false)
                                .map(|x| x.len())
                                .unwrap_or(usize::MAX)
                        };
                        (r, len)
                    })
                    .unzip()
            })
        });

    // Sort by token length (shortest first).
    let mut combined: Vec<(SearchResult, usize)> = results
        .into_iter()
        .zip(result_token_lens.into_iter())
        .collect();
    combined.sort_by_key(|(_, len)| *len);
    let (results, result_token_lens): (Vec<SearchResult>, Vec<usize>) =
        combined.into_iter().unzip();

    // Rank and select results within the token budget.
    let mut used_results = Vec::new();
    let mut used_len = 0;

    if let Some(search_pipeline) = &mut *get_mut_arcmutex!(engine.search_pipeline) {
        let ranked_chunks =
            search::rag::rank_document_chunks(&params.query, &results, search_pipeline).unwrap();

        if ranked_chunks.is_empty() {
            for (result, len) in results.iter().zip(result_token_lens.iter()) {
                if used_len + len > max_toks {
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
                if used_len + chunk.token_len > max_toks {
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
            scorer.upsert(&i, embedder.embed(doc_text));
        }

        let query_embedding = embedder.embed(&params.query);
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
            let len = result_token_lens[doc.id];
            if used_len + len > max_toks {
                break;
            }
            used_len += len;
            used_results.push(results[doc.id].clone());
        }
    }

    let content = serde_json::to_string(&serde_json::json!({ "output": used_results })).unwrap();
    tracing::info!(
        "Web search executed in {:.2}s, using {used_len} tokens of {} search results.",
        (Instant::now() - start).as_secs_f32(),
        used_results.len()
    );

    ToolResult { content }
}

// ── Extraction ─────────────────────────────────────────────────────────────

pub(super) async fn execute_extraction(
    engine: &Arc<Engine>,
    tc: &ToolCallResponse,
    opts: &WebSearchOptions,
) -> ToolResult {
    let params: ExtractFunctionParameters =
        serde_json::from_str(&tc.function.arguments).unwrap();
    tracing::info!("Called extraction tool with url `{}`.", params.url);

    let start = Instant::now();
    let tokenizer = get_mut_arcmutex!(engine.pipeline)
        .tokenizer()
        .expect("A tokenizer is expected for non-diffusion models.");
    let dispatch = quiet_dispatch();
    let max_toks = token_budget(opts);

    let res = {
        let raw = tokio::task::block_in_place(|| {
            tracing::dispatcher::with_default(&dispatch, || {
                search::run_extract_tool(&params).unwrap()
            })
        });
        raw.cap_content_len(&tokenizer, max_toks).unwrap()
    };

    let content = serde_json::to_string(&res)
        .unwrap()
        .replace("\\n", "\n")
        .replace("\\\"", "\"")
        .replace("\\\\", "\\");
    let used_len = {
        let inp = InputSequence::Raw(Cow::from(&content));
        tokenizer
            .encode_fast(inp, false)
            .map(|x| x.len())
            .unwrap_or(usize::MAX)
    };
    tracing::info!(
        "Extraction executed in {:.2}s, using {used_len} tokens.",
        (Instant::now() - start).as_secs_f32(),
    );

    ToolResult {
        content: format!("{{\"output\": \"{content}\"}}"),
    }
}

// ── Custom tool callbacks ──────────────────────────────────────────────────

pub(super) fn execute_custom_tool(engine: &Engine, tc: &ToolCallResponse) -> ToolResult {
    let name = &tc.function.name;

    let content = if let Some(cb) = engine.tool_callbacks.get(name) {
        cb(&tc.function).unwrap_or_else(|e| {
            tracing::error!("Error when calling tool `{name}`: {e}");
            format!("ERROR: {e}")
        })
    } else if let Some(cb_with_tool) = engine.tool_callbacks_with_tools.get(name) {
        (cb_with_tool.callback)(&tc.function).unwrap_or_else(|e| {
            tracing::error!("Error when calling tool `{name}`: {e}");
            format!("ERROR: {e}")
        })
    } else {
        tracing::error!("Attempted to call tool `{name}`, but it doesn't exist.");
        format!("ERROR: no tool callback for {name}")
    };

    ToolResult { content }
}
