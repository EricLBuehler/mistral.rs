---
title: Stream chat responses from Rust
description: Handle Response variants, errors, tool progress events, cancellation, and task spawning with stream_chat_request.
---

`stream_chat_request` returns a value implementing `futures::Stream<Item = Response>`. The minimal loop is in [getting started](/mistral.rs/guides/rust/getting-started/#streaming); this guide covers the variant taxonomy and production patterns.

## Handling every variant

```rust
use futures::StreamExt;
use mistralrs::{ChatCompletionChunkResponse, ChunkChoice, Delta, Response};
use std::io::Write;

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
```

`Response` variants seen on a chat stream:

- `Response::Chunk`: the common case. Carries incremental text in `choices[0].delta.content`.
- `Response::Done`: end of stream, with the final `ChatCompletionResponse` and usage stats.
- `Response::InternalError`: engine-level failure. The stream produces no further values.
- `Response::ModelError`: model-level failure, accompanied by the partial response built so far.
- `Response::AgenticToolCallProgress`, `Response::AgenticToolApprovalRequired`, `Response::File`: emitted when server-side tools run mid-stream (next section).

The example uses `_ => {}` for brevity; production code should match the agentic variants explicitly. Full example: [streaming](/mistral.rs/examples/rust/getting-started/streaming/), [error-handling](/mistral.rs/examples/rust/advanced/error-handling/).

## Streaming with tool calls

When the [agentic loop](/mistral.rs/guides/agents/build-an-agent/) executes a tool mid-stream (web search, code execution, [MCP (Model Context Protocol)](/mistral.rs/guides/agents/connect-mcp-server/) tools), the stream interleaves progress events with content chunks, in stream order:

```rust
use mistralrs::core::AgenticToolCallPhase;

Response::AgenticToolCallProgress { round, tool_name, phase } => {
    match phase {
        AgenticToolCallPhase::Calling(_) => println!("[round {round}: calling {tool_name}]"),
        AgenticToolCallPhase::Complete(_) => println!("[round {round}: completed {tool_name}]"),
    }
}
```

Note that the non-streaming `send_chat_request` skips these events internally and returns only the final response.

## Spawning, backpressure, and cancellation

The stream borrows the `Model` for its lifetime, so it can move across await points within the same scope but cannot be sent into a detached task on its own. `Model` does not implement `Clone`. To stream inside a spawned task, share the model via `Arc` and create the stream inside the task:

```rust
use std::sync::Arc;

let model = Arc::new(model);

let handle = tokio::spawn({
    let model = Arc::clone(&model);
    async move {
        let mut stream = model.stream_chat_request(messages).await?;
        while let Some(item) = stream.next().await {
            // forward chunks to a channel, websocket, etc.
        }
        anyhow::Ok(())
    }
});
```

The response channel behind the stream is bounded, so a consumer that stops polling applies backpressure to the engine. To cancel early, drop the stream: the channel closes and the engine stops generating for that request.

## Collecting the full response

To stream for early feedback while also assembling the final text:

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

`full_response` holds the complete assistant output once the stream ends; use it to log or persist the final text.
