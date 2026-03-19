//! Load and run a model with a LoRA adapter.
//!
//! Run with: `cargo run --release --example lora -p mistralrs`

use anyhow::Result;
use mistralrs::{LoraModelBuilder, TextMessageRole, TextMessages, TextModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = LoraModelBuilder::from_text_model_builder(
        TextModelBuilder::new("meta-llama/Llama-3.2-1B-Instruct").with_logging(),
        vec!["danielhanchen/llama-3.2-lora".to_string()],
    )
    .build()
    .await?;

    let messages = TextMessages::new().add_message(
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
