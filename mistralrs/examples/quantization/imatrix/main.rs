//! ISQ quantization using a calibration file for importance-matrix collection.
//!
//! Run with: `cargo run --release --example imatrix -p mistralrs`

use anyhow::Result;
use mistralrs::{IsqBits, ModelBuilder, PagedAttentionMetaBuilder, TextMessageRole, TextMessages};

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-3-4b-it")
        .with_auto_isq(IsqBits::Four)
        .with_calibration_file("calibration_data/calibration_datav3_small.txt".into())
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

    Ok(())
}
