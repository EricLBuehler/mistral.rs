use anyhow::Result;
use mistralrs::{
    IsqType, RequestBuilder, SpeculativeConfig, TextMessageRole, TextMessages, TextModelBuilder,
    TextSpeculativeBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let target = TextModelBuilder::new("../hf_models/llama3.1_8b").with_logging();
    let draft = TextModelBuilder::new("../hf_models/llama3.2_3b")
        .with_logging()
        .with_isq(IsqType::Q8_0);
    let spec_cfg = SpeculativeConfig { gamma: 16 };
    let model = TextSpeculativeBuilder::new(target, draft, spec_cfg)?
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

    // Next example: Return some logprobs with the `RequestBuilder`, which enables higher configurability.
    let request = RequestBuilder::new().return_logprobs(true).add_message(
        TextMessageRole::User,
        "Please write a mathematical equation where a few numbers are added.",
    );

    let response = model.send_chat_request(request).await?;

    println!(
        "Logprobs: {:?}",
        &response.choices[0]
            .logprobs
            .as_ref()
            .unwrap()
            .content
            .as_ref()
            .unwrap()[0..3]
    );

    Ok(())
}
