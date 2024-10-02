use anyhow::Result;
use mistralrs::{
    IsqType, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole, TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    // Bullet list regex
    let request = RequestBuilder::new()
        .set_constraint(mistralrs::Constraint::Regex(
            "(- [^\n]*\n)+(- [^\n]*)(\n\n)?".to_string(),
        ))
        .add_message(TextMessageRole::User, "Please write a few jokes.");

    let response = model.send_chat_request(request).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
