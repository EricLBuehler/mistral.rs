//! Python code execution example.
//!
//! The model is given `execute_python` and `reset_python_session` tools and
//! can write and run Python code to answer questions.
//!
//! Run with: `cargo run --release --features code-execution --example code_execution -p mistralrs`

use anyhow::Result;
use mistralrs::{
    CodeExecutionConfig, IsqBits, ModelBuilder, RequestBuilder, TextMessageRole, TextMessages,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-4-E4B-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_code_execution(CodeExecutionConfig::default())
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "Use Python to calculate the first 20 prime numbers and their sum.",
    );
    let request = RequestBuilder::from(messages);

    let response = model.send_chat_request(request).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
