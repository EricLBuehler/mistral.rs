use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, TextMessages, TextModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("MiniMaxAI/MiniMax-M2")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .build()
        .await?;

    let mut messages = TextMessages::new();

    messages = messages.add_message(TextMessageRole::User, "Hello! How many rs in strawberry?");
    let response = model.send_chat_request(messages.clone()).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
