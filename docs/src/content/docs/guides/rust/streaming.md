---
title: Stream chat responses from Rust
description: Use stream_chat_request with a futures Stream. Handle chunks, errors, and the final completion event.
sidebar:
  order: 1
---

`stream_chat_request` returns a `futures::Stream` of `Response` values. Anything that consumes a futures stream works with it.

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
            Response::InternalError(e) => {
                eprintln!("stream error: {e}");
                break;
            }
            Response::ModelError(msg, _) => {
                eprintln!("stream error: {msg}");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
```

`Response` variants:

- `Response::Chunk`: the common case. Carries incremental text in `choices[0].delta.content`.
- `Response::Done`: end of stream. Use to break the loop or collect final usage stats.
- `Response::InternalError`: engine-level failure. The stream produces no further values.
- `Response::ModelError`: model-level failure. Often accompanied by inspectable partial state.
- Other variants for tool calls, logprobs, and multimodal responses; see [docs.rs/mistralrs](https://docs.rs/mistralrs).

The example uses `_ => {}` for brevity. Production code should match the other variants explicitly.

## Streaming with tool calls

When tool calling is enabled and the model invokes a tool mid-stream, the engine emits the tool round as a side effect and continues. Client code receives only the final user-facing chunks, not the tool round-trips.

To observe tool rounds, use `Response::AgenticToolCallProgress`:

```rust
use mistralrs::core::AgenticToolCallPhase;

Response::AgenticToolCallProgress {
    tool_name, phase, ..
} => {
    match phase {
        AgenticToolCallPhase::Calling(_) => println!("[calling {tool_name}]"),
        AgenticToolCallPhase::Complete(_) => println!("[completed {tool_name}]"),
    }
}
```

These events interleave with content chunks in stream order.

## Backpressure and cancellation

The stream borrows the `Model` for its lifetime. It can move across await points within the same scope, but to send it down a channel or spawn it in a detached task, clone the `Model` first and consume the stream inside the task. When the consumer stops polling, the engine pauses generation automatically.

To cancel early, drop the stream. The engine frees the sequence and stops generating. In-flight generated-but-unread tokens are discarded.

## Collecting the full response

To stream for early feedback while also assembling the final response:

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

This pattern streams to a terminal or web client while logging the final transcript to a database.
