//! MoQE: quantize only MoE expert layers at a different precision.
//!
//! Run with: `cargo run --release --example mixture_of_quant_experts -p mistralrs`

use anyhow::Result;
use mistralrs::{
    IsqBits, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-MoE-instruct")
        .with_auto_isq(IsqBits::Four)
        .with_mixture_qexperts_isq()
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
