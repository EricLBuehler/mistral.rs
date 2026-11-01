---
title: Stream chat responses from Rust
description: Handle Response variants, errors, tool progress events, cancellation, and task spawning with stream_chat_request.
---

`stream_chat_request` returns a `futures::Stream` of `Response` values that you read with `.next().await`. The minimal loop is in [getting started](/guides/rust/getting-started/#streaming); this page covers errors, tool events, cancellation, and collecting the full reply.

## Handling every variant

```rust
use mistralrs::Response;
use std::io::Write;

let mut stream = model.stream_chat_request(messages).await?;
let mut out = std::io::BufWriter::new(std::io::stdout());

while let Some(item) = stream.next().await {
    match item {
        Response::Chunk(chunk) => {
            if let Some(text) = chunk.choices.first().and_then(|c| c.delta.content.as_ref()) {
                out.write_all(text.as_bytes())?;
                out.flush()?;
            }
        }
        Response::InternalError(e) | Response::ValidationError(e) => {
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

These are the `Response` variants a chat stream can send:

| Variant | Meaning |
|---|---|
| `Response::Chunk` | New output. Text is in `choices[0].delta.content`. |
| `Response::InternalError`, `Response::ValidationError` | The engine or the request failed. Nothing else follows, so stop reading. |
| `Response::ModelError` | The model failed. It carries the error message and the partial response built so far. |
| `Response::AgenticToolCallProgress`, `Response::AgenticToolApprovalRequired`, `Response::File` | A server-side tool is starting or finished, needs your approval, or produced a file. See the next section. |
| `Response::BlockDenoisingProgress` | Progress from a diffusion language model. |

Each choice's last chunk carries its `finish_reason`, and the stream ends after that. The last chunk also carries token counts in `usage`.

For [chat speech output](/reference/rust-sdk/#chat-speech-output), select `ChatAudioFormat::Pcm16`. Audio then arrives in `Response::Chunk` as base64 PCM in `choices[].delta.audio.data`: decode each chunk and append the bytes in the order they arrive, including the final one.

The loop above prints only the first choice and skips everything else; match the agentic variants too if your app uses tools. Full examples: [streaming](/examples/rust/getting-started/streaming/) and [error-handling](/examples/rust/advanced/error-handling/).

## Streaming with tool calls

When the [agentic loop](/guides/agents/build-an-agent/) executes a tool mid-stream (web search, code execution, shell, [MCP (Model Context Protocol)](/guides/agents/connect-mcp-server/) tools), the stream interleaves progress events with content chunks, in stream order:

```rust
use mistralrs::core::AgenticToolCallPhase;

Response::AgenticToolCallProgress { round, tool_name, phase } => {
    match phase {
        AgenticToolCallPhase::Calling(_) => println!("[round {round}: calling {tool_name}]"),
        AgenticToolCallPhase::Complete(_) => println!("[round {round}: completed {tool_name}]"),
    }
}
```

The non-streaming `send_chat_request` doesn't surface these events; it returns only the final response.

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

To show text as it arrives and also keep the whole reply, append each piece to a `String`:

```rust
use mistralrs::Response;
use std::io::Write;

let mut stream = model.stream_chat_request(messages).await?;
let mut reply = String::new();

while let Some(item) = stream.next().await {
    match item {
        Response::Chunk(chunk) => {
            if let Some(text) = chunk.choices.first().and_then(|c| c.delta.content.as_deref()) {
                print!("{text}");
                std::io::stdout().flush()?;
                reply.push_str(text);
            }
        }
        Response::InternalError(e) | Response::ValidationError(e) => {
            eprintln!("stream error: {e}");
            break;
        }
        Response::ModelError(msg, _) => {
            eprintln!("model error: {msg}");
            break;
        }
        _ => {}
    }
}
```

When the loop ends, `reply` holds the whole reply, or everything generated before an error. With `n > 1`, chunks for different choices arrive interleaved: keep one buffer per choice and pick it with `choice.index`.
