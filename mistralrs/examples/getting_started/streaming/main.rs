//! Streaming text generation with token-by-token output.
//!
//! Run with: `cargo run --release --example streaming -p mistralrs`

use anyhow::Result;
use mistralrs::{
    ChatCompletionChunkResponse, ChunkChoice, Delta, IsqBits, ModelBuilder,
    PagedAttentionMetaBuilder, RequestBuilder, Response, TextMessageRole, TextMessages,
};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-3-4b-it")
        .with_auto_isq(IsqBits::Eight)
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
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

    let mut stream = model.stream_chat_request(request).await?;

    let stdout = std::io::stdout();
    let lock = stdout.lock();
    let mut buf = std::io::BufWriter::new(lock);
    while let Some(chunk) = stream.next().await {
        if let Response::Chunk(ChatCompletionChunkResponse { choices, .. }) = chunk {
            if let Some(ChunkChoice {
                delta:
                    Delta {
                        content: Some(content),
                        ..
                    },
                ..
            }) = choices.first()
            {
                buf.write_all(content.as_bytes())?;
            };
        } else {
            // Handle errors
        }
    }

    Ok(())
}
