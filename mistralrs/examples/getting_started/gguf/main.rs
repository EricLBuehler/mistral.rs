//! Load a GGUF model from Hugging Face.
//!
//! Configuration and tokenizer assets are discovered automatically. Use `with_tok_model_id`
//! only to override that choice or when the source cannot be identified.
//!
//! Run with: `cargo run --release --example gguf -p mistralrs`

use anyhow::Result;
use mistralrs::{GgufModelBuilder, TextMessageRole, TextMessages};

#[tokio::main]
async fn main() -> Result<()> {
    let model = GgufModelBuilder::new("unsloth/Qwen3-0.6B-GGUF", vec!["Qwen3-0.6B-Q4_K_M.gguf"])
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
