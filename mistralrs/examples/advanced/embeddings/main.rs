//! Compute and compare text embeddings with cosine similarity.
//!
//! Run with: `cargo run --release --example embeddings -p mistralrs`

use anyhow::Result;
use mistralrs::{Device, EmbeddingModelBuilder, EmbeddingRequest, Tensor};

#[tokio::main]
async fn main() -> Result<()> {
    let model = EmbeddingModelBuilder::new("Qwen/Qwen3-Embedding-0.6B")
        .with_logging()
        .build()
        .await?;

    let embeddings = model
        .generate_embeddings(EmbeddingRequest::builder().add_prompt("What is graphene"))
        .await?;

    let y = Tensor::new(embeddings[0].clone(), &Device::Cpu)?;
    y.write_npy("test.npy")?;

    Ok(())
}
