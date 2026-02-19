//! Generate text embeddings using an embedding model.
//!
//! Run with: `cargo run --release --example embedding -p mistralrs`

use anyhow::Result;
use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest};

#[tokio::main]
async fn main() -> Result<()> {
    let model = EmbeddingModelBuilder::new("google/embeddinggemma-300m")
        .with_logging()
        .build()
        .await?;

    let embeddings = model
        .generate_embeddings(
            EmbeddingRequest::builder()
                .add_prompt("task: search result | query: What is graphene?"),
        )
        .await?;
    println!("{:?}", embeddings.first());

    Ok(())
}
