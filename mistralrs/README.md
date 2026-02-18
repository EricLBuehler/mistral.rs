# mistralrs — Blazing-Fast LLM Inference in Rust

The Rust SDK for [mistral.rs](https://github.com/EricLBuehler/mistral.rs), a high-performance
LLM inference engine supporting text, vision, speech, image generation, and embedding models.

[API Docs](https://docs.rs/mistralrs) | [GitHub](https://github.com/EricLBuehler/mistral.rs) | [Examples](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples) | [Discord](https://discord.gg/SZrecqK8qw)

## Quick Start

```rust
use mistralrs::{IsqBits, ModelBuilder, TextMessages, TextMessageRole};

#[tokio::main]
async fn main() -> mistralrs::error::Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .build()
        .await?;

    let response = model.chat("What is Rust's ownership model?").await?;
    println!("{response}");
    Ok(())
}
```

## Capabilities

| Capability | Builder | Example |
|---|---|---|
| Any model (auto-detect) | `ModelBuilder` | `examples/getting_started/text_generation/` |
| Text generation | `TextModelBuilder` | `examples/getting_started/text_generation/` |
| Vision (image+text) | `VisionModelBuilder` | `examples/getting_started/vision/` |
| GGUF quantized models | `GgufModelBuilder` | `examples/getting_started/gguf/` |
| Image generation | `DiffusionModelBuilder` | `examples/models/diffusion/` |
| Speech synthesis | `SpeechModelBuilder` | `examples/models/speech/` |
| Embeddings | `EmbeddingModelBuilder` | `examples/getting_started/embedding/` |
| Structured output | `Model::generate_structured` | `examples/advanced/json_schema/` |
| Tool calling | `Tool`, `ToolChoice` | `examples/advanced/tools/` |
| Agents | `AgentBuilder` | `examples/advanced/agent/` |
| LoRA / X-LoRA | `LoraModelBuilder`, `XLoraModelBuilder` | `examples/advanced/lora/` |
| AnyMoE | `AnyMoeModelBuilder` | `examples/advanced/anymoe/` |
| MCP client | `McpClientConfig` | `examples/advanced/mcp_client/` |

## Choosing a Request Type

| Type | Use When | Sampling |
|---|---|---|
| `TextMessages` | Simple text-only chat | Deterministic |
| `VisionMessages` | Prompt includes images or audio | Deterministic |
| `RequestBuilder` | Tools, logprobs, custom sampling, constraints, adapters, or web search | Configurable |

`TextMessages` and `VisionMessages` convert into `RequestBuilder` via `Into<RequestBuilder>`.

## Feature Flags

| Flag | Effect |
|---|---|
| `cuda` | CUDA GPU support |
| `flash-attn` | Flash Attention 2 kernels (requires `cuda`) |
| `cudnn` | cuDNN acceleration (requires `cuda`) |
| `nccl` | Multi-GPU via NCCL (requires `cuda`) |
| `metal` | Apple Metal GPU support |
| `accelerate` | Apple Accelerate framework |
| `mkl` | Intel MKL acceleration |

The default feature set (no flags) builds with pure Rust — no C compiler or system libraries required.

## License

MIT
