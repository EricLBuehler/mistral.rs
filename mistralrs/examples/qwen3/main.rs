use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, TextMessages, TextModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("Qwen/Qwen3-30B-A3B")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .build()
        .await?;

    let mut messages = TextMessages::new();

    // ------------------------------------------------------------------
    // First question, thinking mode is enabled by default
    // ------------------------------------------------------------------
    messages = messages.add_message(TextMessageRole::User, "Hello! How many rs in strawberry?");
    let response = model.send_chat_request(messages.clone()).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    messages = messages.add_message(
        TextMessageRole::Assistant,
        response.choices[0].message.content.as_ref().unwrap(),
    );

    // ------------------------------------------------------------------
    // Second question, disable thinking mode with RequestBuilder or /no_think
    // ------------------------------------------------------------------
    messages = messages.add_message(TextMessageRole::User, "How many rs in blueberry? /no_think");
    let response = model.send_chat_request(messages.clone()).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    messages = messages.add_message(
        TextMessageRole::Assistant,
        response.choices[0].message.content.as_ref().unwrap(),
    );

    // ------------------------------------------------------------------
    // Third question, reenable thinking mode with RequestBuilder or /think
    // ------------------------------------------------------------------
    messages = messages.add_message(TextMessageRole::User, "Are you sure? /think");
    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
