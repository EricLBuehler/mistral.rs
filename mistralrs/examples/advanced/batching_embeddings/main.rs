//! Batch multiple embedding requests for efficient parallel encoding.
//!
//! Run with: `cargo run --release --example batching_embeddings -p mistralrs`

use anyhow::Result;
use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest};

#[tokio::main]
async fn main() -> Result<()> {
    let model = EmbeddingModelBuilder::new("google/embeddinggemma-300m")
        .with_logging()
        .build()
        .await?;

    let a = model
        .generate_embeddings(
            EmbeddingRequest::builder()
                .add_prompt("task: search result | query: What is graphene?"),
        )
        .await?;
    let b =
        model
            .generate_embeddings(EmbeddingRequest::builder().add_prompt(
                "task: search result | query: What is an apple's significance to gravity?",
            ))
            .await?;

    let batched = model
        .generate_embeddings(EmbeddingRequest::builder().add_prompts((0..100).map(|i| {
            if i % 2 == 0 {
                "task: search result | query: What is graphene?"
            } else {
                "task: search result | query: What is an apple's significance to gravity?"
            }
        })))
        .await?;

    for (i, embedding) in batched.into_iter().enumerate() {
        if i % 2 == 0 {
            assert_eq!(embedding, a[0]);
        } else {
            assert_eq!(embedding, b[0]);
        }
    }

    Ok(())
}
