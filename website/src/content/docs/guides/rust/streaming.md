---
title: Stream chat responses from Rust
description: Use stream_chat_request with a futures Stream. Handle chunks, errors, and the final completion event.
sidebar:
  order: 1
---

The Rust SDK exposes streaming through `stream_chat_request`, which returns a `futures::Stream` of `Response` values. This is the idiomatic Rust shape; anything that knows how to consume a futures stream works with it.

## The minimal example

```rust
use anyhow::Result;
use futures::StreamExt;
use mistralrs::{
    ChatCompletionChunkResponse, ChunkChoice, Delta, IsqBits, ModelBuilder,
    Response, TextMessageRole, TextMessages,
};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, "Explain borrowing in Rust.");

    let mut stream = model.stream_chat_request(messages).await?;
    let mut out = std::io::BufWriter::new(std::io::stdout());

    while let Some(item) = stream.next().await {
        match item {
            Response::Chunk(ChatCompletionChunkResponse { choices, .. }) => {
                if let Some(ChunkChoice {
                    delta: Delta { content: Some(text), .. },
                    ..
                }) = choices.first()
                {
                    out.write_all(text.as_bytes())?;
                    out.flush()?;
                }
            }
            Response::Done(_) => break,
            Response::InternalError(e) | Response::ModelError(e, _) => {
                eprintln!("stream error: {e}");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
```

The `Response` enum has several variants:

- `Response::Chunk` is the common case. Carries an incremental piece of generated text in `choices[0].delta.content`.
- `Response::Done` signals the end of the stream. You can break the loop or collect final usage stats from this variant.
- `Response::InternalError` is an engine-level failure. The stream will not produce more values after this.
- `Response::ModelError` is a model-level failure. Usually accompanied by partial state that can be inspected.
- Other variants exist for tool calls, logprobs, and multimodal responses; the [Rust API reference](/mistral.rs/reference/rust-api/) documents all of them.

Exhaustive pattern matching is a good habit here. The match in the example above uses `_ => {}` for brevity; production code should consider each variant deliberately.

## Streaming with tool calls

When tool calling is enabled and the model invokes a tool mid-stream, the engine emits the tool-call round as a side effect and continues streaming. The chunks you receive from your client code represent only the final user-facing response, not the tool round-trips.

If you want to observe tool rounds as they happen, use the `Response::AgenticToolCallProgress` variant:

```rust
Response::AgenticToolCallProgress(progress) => {
    match progress.phase.as_str() {
        "calling" => println!("[calling {}]", progress.tool_name),
        "complete" => println!("[completed {}]", progress.tool_name),
        _ => {}
    }
}
```

These events interleave with the content chunks in the stream order.

## Backpressure and cancellation

The stream is `Send` and `'static`, which means you can move it across await points, spawn it in a task, or send it down a channel. If the consumer stops polling, the engine pauses generation automatically.

To cancel a stream early, drop it. The engine will free the associated sequence and stop generating. Any in-flight tokens that have already been produced but not yet read are discarded.

## Collecting the full response

If you only want streaming for early feedback and also want the final assembled response at the end, collect chunks into a string as you go:

```rust
let mut full_response = String::new();

while let Some(item) = stream.next().await {
    if let Response::Chunk(chunk) = item {
        if let Some(choice) = chunk.choices.first() {
            if let Some(text) = &choice.delta.content {
                full_response.push_str(text);
                out.write_all(text.as_bytes())?;
                out.flush()?;
            }
        }
    }
}

// `full_response` now holds the complete assistant output.
```

This is the pattern for streaming to a terminal or web client while simultaneously logging the final transcript to a database.
