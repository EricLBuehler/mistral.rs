use anyhow::Result;
use mistralrs::{
    IsqType, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole, TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("google/gemma-2-9b-it")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    let request = RequestBuilder::new().add_message(
        TextMessageRole::User,
        "Please write a mathematical equation where a few numbers are added.",
    );

    let response = model.send_chat_request(request).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
