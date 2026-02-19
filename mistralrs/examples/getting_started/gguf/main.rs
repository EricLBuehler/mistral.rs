//! Load and run a GGUF-quantized model from Hugging Face.
//!
//! Run with: `cargo run --release --example gguf -p mistralrs`

use anyhow::Result;
use mistralrs::{GgufModelBuilder, TextMessageRole, TextMessages};

#[tokio::main]
async fn main() -> Result<()> {
    let model = GgufModelBuilder::new(
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        vec!["Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"],
    )
    .with_tok_model_id("meta-llama/Meta-Llama-3.1-8B-Instruct")
    .with_logging()
    .build()
    .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
