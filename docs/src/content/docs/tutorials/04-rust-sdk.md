---
title: Call a model from Rust
description: Add mistralrs to a Cargo project, load a model in-process, and stream a response. About fifteen minutes.
sidebar:
  order: 4
---

The Rust SDK embeds the engine directly into a Rust program. A Rust toolchain is required; see [rustup.rs](https://rustup.rs).

## Creating the project

```bash
cargo new --bin hello-mistralrs
cd hello-mistralrs
```

Add the dependencies to `Cargo.toml`:

```toml
[dependencies]
anyhow = "1"
mistralrs = "0.8"
tokio = { version = "1", features = ["full"] }
```

The default features build for CPU. For GPU acceleration, enable the matching feature:

```toml
# NVIDIA GPU (CUDA)
mistralrs = { version = "0.8", features = ["cuda", "flash-attn", "cudnn"] }

# Apple Silicon (Metal)
mistralrs = { version = "0.8", features = ["metal"] }

# Intel CPU with MKL
mistralrs = { version = "0.8", features = ["mkl"] }
```

Feature names match the CLI build features. The [cargo features reference](/mistral.rs/reference/cargo-features/) lists every option.

## A minimal request

Replace `src/main.rs`:

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

Run with `cargo run --release`. 

The first run downloads Gemma 4 into the Hugging Face cache. The Gemma license must be accepted first; see [Tutorial 2](/mistral.rs/tutorials/02-serve-an-api/#accepting-the-gemma-license).

`ModelBuilder` is a fluent configuration object. Each method returns `self`. The only required input is the Hugging Face repository id passed to `ModelBuilder::new`. Everything else has a default.

`with_auto_isq(IsqBits::Four)` matches `--isq 4` on the CLI. The engine selects an optimal 4-bit format per platform: AFQ4 on Metal, Q4K on CUDA or CPU. To pin a specific format, use `with_isq(IsqType::Q4K)`, see the [quantization reference](/mistral.rs/reference/quantization-types/).

`TextMessages` assembles a basic chat conversation. For per-message sampling, tool schemas, or logprobs, use `RequestBuilder`.

## Streaming

`stream_chat_request` returns a futures `Stream` of response chunks:

```rust
use anyhow::Result;
use futures::StreamExt;
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

The stream yields `Response` values. Most are `Response::Chunk` carrying assistant output in `choices[0].delta.content`. Other variants cover errors and the final completion event; production code should pattern-match exhaustively. See [docs.rs/mistralrs](https://docs.rs/mistralrs).

## Notes

`ModelBuilder::build()` performs all model loading and is expensive. Call it once at startup and share the resulting `Model`. `Model` is reference-counted, cheap to clone, and thread-safe.

Requests through the Rust SDK bypass the HTTP layer; there is no `/v1/chat/completions` endpoint and no OpenAI compatibility shim. To expose a `Model` over HTTP alongside direct in-process access, see the [embed-in-axum guide](/mistral.rs/guides/rust/embed-in-axum/).

## Next steps

- [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/): add tool calling and code execution.
- [Tutorial 6](/mistral.rs/tutorials/06-quantize-a-model/): choose between ISQ bit widths.
- The [Rust SDK guides](/mistral.rs/guides/rust/streaming/) cover async streaming, multimodal input, and embedding mistral.rs in existing web applications.
