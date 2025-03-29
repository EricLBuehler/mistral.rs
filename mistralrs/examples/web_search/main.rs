use anyhow::Result;
use mistralrs::{
    BertEmbeddingModel, IsqType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder,
    WebSearchOptions,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("NousResearch/Hermes-3-Llama-3.1-8B")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_search(BertEmbeddingModel::default())
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "What is the weather forecast for Boston?",
    );
    let messages =
        RequestBuilder::from(messages).with_web_search_options(WebSearchOptions::default());

    let response = model.send_chat_request(messages).await?;

    println!("What is the weather forecast for Boston?\n\n");
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
