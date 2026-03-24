//! Load and run a GGUF model from a local file path.
//!
//! Run with: `cargo run --release --example gguf_locally -p mistralrs`

use anyhow::Result;
use mistralrs::{
    GgufModelBuilder, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole, TextMessages,
};

#[tokio::main]
async fn main() -> Result<()> {
    // We do not use any files from remote servers here, and instead load the
    // chat template from the specified file, and the tokenizer and model from a
    // local GGUF file at the path specified.
    let model = GgufModelBuilder::new(
        "gguf_models/mistral_v0.1/",
        vec!["mistral-7b-instruct-v0.1.Q4_K_M.gguf"],
    )
    .with_chat_template("chat_templates/mistral.json")
    .with_logging()
    .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
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
