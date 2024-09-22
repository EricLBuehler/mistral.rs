use std::fs::File;

use anyhow::Result;
use mistralrs::{LoraModelBuilder, TextMessageRole, TextMessages, TextModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        LoraModelBuilder::from_text_model_builder(
            TextModelBuilder::new("HuggingFaceH4/zephyr-7b-beta").with_logging(),
            "lamm-mit/x-lora",
            serde_json::from_reader(File::open("my-ordering-file.json").unwrap_or_else(|_| {
                panic!("Could not load ordering file at my-ordering-file.json")
            }))?,
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
