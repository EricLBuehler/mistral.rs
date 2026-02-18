//! X-LoRA: load a model with X-LoRA adapter mixing.
//!
//! Run with: `cargo run --release --example xlora -p mistralrs`

use std::fs::File;

use anyhow::Result;
use mistralrs::{TextMessageRole, TextMessages, TextModelBuilder, XLoraModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        XLoraModelBuilder::from_text_model_builder(
            TextModelBuilder::new("HuggingFaceH4/zephyr-7b-beta").with_logging(),
            "lamm-mit/x-lora",
            serde_json::from_reader(File::open("my-ordering-file.json").unwrap_or_else(|_| {
                panic!("Could not load ordering file at my-ordering-file.json")
            }))?,
        )
        .build()
        .await?;

    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "Hello! What is graphene.");

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
