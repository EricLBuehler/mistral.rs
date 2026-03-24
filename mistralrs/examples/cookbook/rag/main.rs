/// Simple RAG (Retrieval-Augmented Generation) example.
///
/// Demonstrates:
/// 1. Loading an embedding model to vectorize documents
/// 2. Computing cosine similarity to find the most relevant document
/// 3. Sending the retrieved context + query to a text model
///
/// Run with: `cargo run --release --example cookbook_rag -p mistralrs`
use anyhow::Result;
use mistralrs::{
    EmbeddingModelBuilder, EmbeddingRequest, IsqBits, ModelBuilder, TextMessageRole, TextMessages,
};

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[tokio::main]
async fn main() -> Result<()> {
    // ---- Step 1: Load an embedding model ----
    let embed_model = EmbeddingModelBuilder::new("google/embeddinggemma-300m")
        .with_logging()
        .build()
        .await?;

    // ---- Step 2: Embed a small document corpus ----
    let documents = [
        "Rust is a systems programming language focused on safety, speed, and concurrency.",
        "Python is widely used for data science, machine learning, and scripting.",
        "mistral.rs is a blazing-fast LLM inference engine written in Rust.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    ];

    let mut doc_embeddings = Vec::new();
    for doc in &documents {
        let emb = embed_model
            .generate_embeddings(EmbeddingRequest::builder().add_prompt(format!("passage: {doc}")))
            .await?;
        doc_embeddings.push(emb.first().unwrap().clone());
    }

    // ---- Step 3: Embed the query and find the best match ----
    let query = "What is mistral.rs?";
    let query_emb = embed_model
        .generate_embeddings(EmbeddingRequest::builder().add_prompt(format!("query: {query}")))
        .await?;
    let query_vec = query_emb.first().unwrap();

    let (best_idx, best_score) = doc_embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(query_vec, emb)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("Query: {query}");
    println!(
        "Best match (score={best_score:.3}): \"{}\"",
        documents[best_idx]
    );

    // ---- Step 4: Send context + query to a text model ----
    let text_model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .build()
        .await?;

    let prompt = format!(
        "Context: {}\n\nQuestion: {}\n\nAnswer based on the context above.",
        documents[best_idx], query
    );

    let messages = TextMessages::new().add_message(TextMessageRole::User, prompt);

    let response = text_model.send_chat_request(messages).await?;
    println!(
        "\nAnswer: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    Ok(())
}
