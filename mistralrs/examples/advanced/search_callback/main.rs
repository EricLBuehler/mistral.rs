//! Custom search callback to override the default web search function.
//!
//! Run with: `cargo run --release --example search_callback -p mistralrs`

use anyhow::Result;
use mistralrs::{
    IsqBits, ModelBuilder, RequestBuilder, SearchResult, TextMessageRole, TextMessages,
    WebSearchOptions,
};

use std::fs;
use std::sync::Arc;
use walkdir::WalkDir;

/// Very small helper that searches the current repository for files that match the given query.
/// The file name is matched against the query (case-sensitive contains check) and the whole file
/// content is returned so the model can decide which parts are useful.
fn local_search(query: &str) -> Result<Vec<SearchResult>> {
    let mut results = Vec::new();

    for entry in WalkDir::new(".") {
        let entry = entry?;
        if entry.file_type().is_file() {
            let name = entry.file_name().to_string_lossy();
            if name.contains(query) {
                let path = entry.path().display().to_string();
                let content = fs::read_to_string(entry.path()).unwrap_or_default();

                results.push(SearchResult {
                    title: name.into_owned(),
                    description: path.clone(),
                    url: path,
                    content,
                });
            }
        }
    }

    // Simple ordering – longest titles (usually most specific) first.
    results.sort_by_key(|r| r.title.clone());
    results.reverse();
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Build the model enabling web-search support.  We supply a custom search callback that is
    // used **instead** of the default remote web search when we set `web_search_options` later
    // in the request.

    // The EmbeddingGemma reranker is **not** required even when using a custom callback – it is
    // used inside the retrieval pipeline to cluster / rank the results that our callback returns.
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_search_callback(Arc::new(|params: &mistralrs::SearchFunctionParameters| {
            // In a real application there could be network or database calls here – but for the
            // sake of demonstration we simply perform a local filesystem search.
            local_search(&params.query)
        }))
        .build()
        .await?;

    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "Where is Cargo.toml in this repo?");

    // Enable searching for this request.  Because we provided a custom search callback above, the
    // model will call **our** function instead of performing an online web search.
    let messages =
        RequestBuilder::from(messages).with_web_search_options(WebSearchOptions::default());

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
