use std::time::Instant;

use anyhow::Result;
use mistralrs::{
    DiffusionGenerationParams, DiffusionLoaderType, EmbeddingModelBuilder,
    ImageGenerationResponseFormat, Tensor,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = EmbeddingModelBuilder::new("google/embeddinggemma-300m")
        .with_logging()
        .build()
        .await?;

    let response = model
        .generate_embeddings("What is graphene?".to_string())
        .await?;
    println!("{response}");

    let xs = Tensor::read_npy("embeddings.npy")?;
    println!("{xs}");

    Ok(())
}
