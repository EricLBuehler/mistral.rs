---
title: Call a model from Rust
description: Add mistralrs to a Cargo project, load a model in-process, and stream a response. About fifteen minutes.
sidebar:
  order: 4
---

The Rust SDK exposes the same engine that powers the `mistralrs` binary. It is the right choice when you want to embed model inference directly inside a Rust service rather than running the HTTP server as a separate process. You get full control over the runtime, you avoid the serialization round-trip, and you can plug custom logic (tool callbacks, request routing, logging) into the engine without going through a network boundary.

This tutorial loads Gemma 4, sends one chat request, and then does the same thing again with streaming. It assumes you have a Rust toolchain installed and understand how to run `cargo new`. If you do not, [rustup.rs](https://rustup.rs) is the canonical starting point.

## Creating the project

```bash
cargo new --bin hello-mistralrs
cd hello-mistralrs
```

Open `Cargo.toml` and add the dependencies:

```toml
[dependencies]
anyhow = "1"
mistralrs = "0.8"
tokio = { version = "1", features = ["full"] }
```

The `mistralrs` crate pulls the engine in with default features, which is fine for CPU. If you want GPU acceleration, enable the feature that matches your hardware:

```toml
# NVIDIA GPU (CUDA)
mistralrs = { version = "0.8", features = ["cuda", "flash-attn", "cudnn"] }

# Apple Silicon (Metal)
mistralrs = { version = "0.8", features = ["metal"] }

# Intel CPU with MKL
mistralrs = { version = "0.8", features = ["mkl"] }
```

The feature names match the ones used when you build the CLI from source, so if you already know your preferred combination from there, use the same here. The [cargo features reference](/mistral.rs/reference/cargo-features/) lists every option.

## A minimal request

Replace `src/main.rs` with the following:

```rust
use anyhow::Result;
use mistralrs::{IsqBits, ModelBuilder, TextMessageRole, TextMessages};

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-4-E4B-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "In one sentence, what is Rust known for?",
    );

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
```

Run it with `cargo run --release`. The release profile matters here: the debug build of the engine is much slower than the release build, to the point where you will think something is wrong on the first token.

The first execution downloads Gemma 4 into your Hugging Face cache, so budget a few minutes. If you have not accepted the Gemma license yet, see the first section of [Tutorial 2](/mistral.rs/tutorials/02-serve-an-api/#accepting-the-gemma-license) before running.

A few things are worth pointing out about this program.

`ModelBuilder` is a fluent configuration object. Every method on it returns `self`, so you can chain any number of options before the final `.build().await?`. The only required input is the Hugging Face repository id, which goes into `ModelBuilder::new`. Everything else has a default.

`with_auto_isq(IsqBits::Four)` does the same thing as `--isq 4` on the CLI. The engine picks a 4-bit format that is optimal for your platform, which on Metal means AFQ4 and on CUDA or CPU means Q4K. If you want an exact format instead of letting the engine choose, use `with_isq(IsqType::Q4K)` or similar; that reference lives in the [quantization reference](/mistral.rs/reference/quantization-types/).

`TextMessages` is the simple way to assemble a chat conversation. For anything more advanced (per-message sampling, tool schemas, logprobs) you would use `RequestBuilder` instead. For a first example, `TextMessages` is enough.

## Streaming

`stream_chat_request` returns a futures `Stream` of response chunks:

```rust
use anyhow::Result;
use mistralrs::{
    ChatCompletionChunkResponse, ChunkChoice, Delta, IsqBits, ModelBuilder, Response,
    TextMessageRole, TextMessages,
};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-4-E4B-it")
        .with_auto_isq(IsqBits::Four)
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "Write me a haiku about ownership.",
    );

    let mut stream = model.stream_chat_request(messages).await?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    while let Some(item) = stream.next().await {
        if let Response::Chunk(ChatCompletionChunkResponse { choices, .. }) = item {
            if let Some(ChunkChoice {
                delta: Delta { content: Some(text), .. },
                ..
            }) = choices.first()
            {
                out.write_all(text.as_bytes())?;
                out.flush()?;
            }
        }
    }

    Ok(())
}
```

The stream yields `Response` values. Most of them will be `Response::Chunk` carrying a piece of assistant output in `choices[0].delta.content`. Other variants exist for errors and for the final completion event; this minimal example ignores them, but production code should pattern-match exhaustively. The [Rust API reference](/mistral.rs/reference/rust-api/) walks through every variant.

## Before you leave

Creating a `ModelBuilder::build()` future is expensive because it does all the model loading. Do it once, at startup, and share the resulting `Model` across your application. `Model` is cheap to clone (it is internally reference-counted) and thread-safe, so you can hand copies to every request handler in a server without worrying.

Keep in mind that sending requests through the Rust SDK bypasses the HTTP layer entirely. That means there is no `/v1/chat/completions` endpoint in play and no OpenAI compatibility shim; you are talking straight to the engine. If you want both (direct access for internal code and an HTTP endpoint for external clients), the [embed-in-axum guide](/mistral.rs/guides/rust/embed-in-axum/) shows how to expose a `Model` instance over HTTP using `mistralrs-server-core`.

## What to try next

- [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/) takes the engine you just wired up and turns on tool calling and code execution.
- [Tutorial 6](/mistral.rs/tutorials/06-quantize-a-model/) goes deeper on quantization, including how to choose between the different ISQ bit widths.
- The [Rust SDK guides](/mistral.rs/guides/rust/streaming/) cover async streaming patterns, multimodal input, and embedding mistral.rs inside existing web applications.
