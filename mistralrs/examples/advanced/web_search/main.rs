//! Web-search-augmented generation using OpenAI-compatible web_search_options.
//!
//! Run with: `cargo run --release --example web_search -p mistralrs`

use anyhow::Result;
use mistralrs::{
    IsqBits, ModelBuilder, RequestBuilder, SearchEmbeddingModel, TextMessageRole, TextMessages,
    WebSearchOptions,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-3-4b-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_search(SearchEmbeddingModel::default())
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "What is the weather forecast for Boston?",
    );
    let messages =
        RequestBuilder::from(messages).with_web_search_options(WebSearchOptions::default());

    let response = model.send_chat_request(messages).await?;

    println!("What is the weather forecast for Boston?\n\n");
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
