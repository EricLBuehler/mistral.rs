//! In-situ quantization (ISQ) with explicit and automatic type selection.
//!
//! Run with: `cargo run --release --example isq -p mistralrs`

use anyhow::Result;
use mistralrs::{
    IsqBits, IsqType, ModelBuilder, PagedAttentionMetaBuilder, TextMessageRole, TextMessages,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Eight)
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
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

    // Next example: re-ISQ the model at runtime
    model.re_isq_model(IsqType::HQQ4).await?;

    let messages = TextMessages::new().add_message(TextMessageRole::User, "Why is the sky blue?");

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
