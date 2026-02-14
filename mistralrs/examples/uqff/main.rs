use anyhow::Result;
use mistralrs::{PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole, UqffTextModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = UqffTextModelBuilder::new(
        "EricB/Phi-3.5-mini-instruct-UQFF",
        vec!["phi3.5-mini-instruct-q8_0.uqff".into()],
    )
    .into_inner()
    .with_logging()
    .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
    .build()
    .await?;

    let request = RequestBuilder::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        )
        .set_sampler_max_len(256);

    let response = model.send_chat_request(request).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    // Next example: Return some logprobs with the `RequestBuilder`, which enables higher configurability.
    let request = RequestBuilder::new()
        .return_logprobs(true)
        .add_message(
            TextMessageRole::User,
            "Please write a mathematical equation where a few numbers are added.",
        )
        .set_sampler_max_len(256);

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
