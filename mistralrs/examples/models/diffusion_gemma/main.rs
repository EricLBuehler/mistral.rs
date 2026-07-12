//! DiffusionGemma: block-diffusion text generation.
//!
//! The model denoises 256-token blocks in parallel instead of sampling tokens one at a
//! time, so streamed output arrives block by block. Sampling parameters are ignored in
//! favor of the checkpoint's denoising schedule.
//!
//! Run with: `cargo run --release --example diffusion_gemma -p mistralrs`

use std::io::Write;

use anyhow::Result;
use mistralrs::{
    ChatCompletionChunkResponse, ChunkChoice, Delta, MultimodalModelBuilder, Response,
    TextMessageRole, TextMessages,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = MultimodalModelBuilder::new("google/diffusiongemma-26B-A4B-it")
        .with_logging()
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "Explain how block diffusion differs from autoregressive text generation.",
    );

    let mut stream = model.stream_chat_request(messages).await?;
    while let Some(chunk) = stream.next().await {
        if let Response::Chunk(ChatCompletionChunkResponse { choices, .. }) = chunk {
            if let Some(ChunkChoice {
                delta: Delta {
                    content: Some(content),
                    ..
                },
                ..
            }) = choices.first()
            {
                print!("{content}");
                std::io::stdout().flush()?;
            }
        }
    }
    println!();

    Ok(())
}
