use anyhow::Result;
use mistralrs::EmbeddingModelBuilder;

#[tokio::main]
async fn main() -> Result<()> {
    let model = EmbeddingModelBuilder::new("google/embeddinggemma-300m")
        .with_logging()
        .build()
        .await?;

    let response = model
        .generate_embeddings("task: search result | query: What is graphene?".to_string())
        .await?;
    println!("{response:?}");

    Ok(())
}
