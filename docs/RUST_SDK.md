# mistralrs Rust SDK

The `mistralrs` crate provides a high-level Rust API for running LLM inference with mistral.rs.

> **Full API reference:** [docs.rs/mistralrs](https://docs.rs/mistralrs)

**Table of contents**
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Builders](#model-builders)
- [Request Types](#request-types)
- [Streaming](#streaming)
- [Structured Output](#structured-output)
- [Tool Calling](#tool-calling)
- [Agents](#agents)
- [Blocking API](#blocking-api)
- [Feature Flags](#feature-flags)
- [Examples](#examples)

## Installation

```bash
cargo add mistralrs
```

Or in your `Cargo.toml`:
```toml
[dependencies]
mistralrs = "0.7"
```

For GPU acceleration, enable the appropriate feature:
```toml
mistralrs = { version = "0.7", features = ["metal"] }     # macOS
mistralrs = { version = "0.7", features = ["cuda"] }       # NVIDIA
```

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

## Model Builders

All models are created through builder structs. Use `ModelBuilder` for auto-detection, or a specific builder for more control.

| Builder | Use Case |
|---|---|
| `ModelBuilder` | Auto-detects model type (text, vision, embedding) |
| `TextModelBuilder` | Text generation models |
| `VisionModelBuilder` | Vision + text models (image/audio input) |
| `GgufModelBuilder` | GGUF quantized model files |
| `EmbeddingModelBuilder` | Text embedding models |
| `DiffusionModelBuilder` | Image generation (e.g., FLUX) |
| `SpeechModelBuilder` | Speech synthesis (e.g., Dia) |
| `LoraModelBuilder` | Text model with LoRA adapters |
| `XLoraModelBuilder` | Text model with X-LoRA adapters |
| `AnyMoeModelBuilder` | AnyMoE Mixture of Experts |
| `TextSpeculativeBuilder` | Speculative decoding (target + draft) |

All builders share common configuration methods:

```rust
let model = TextModelBuilder::new("Qwen/Qwen3-4B")
    .with_auto_isq(IsqBits::Four)      // Platform-optimal quantization
    .with_logging()                      // Enable logging
    .with_paged_attn(                    // PagedAttention for memory efficiency
        PagedAttentionMetaBuilder::default().build()?
    )
    .build()
    .await?;
```

Key builder methods include `with_isq()`, `with_auto_isq()`, `with_dtype()`, `with_force_cpu()`, `with_device_mapping()`, `with_chat_template()`, `with_paged_attn()`, `with_max_num_seqs()`, `with_mcp_client()`, and more. See the [API docs](https://docs.rs/mistralrs) for the full list.

## Request Types

| Type | Use When | Sampling |
|---|---|---|
| `TextMessages` | Simple text-only chat | Deterministic |
| `VisionMessages` | Prompt includes images or audio | Deterministic |
| `RequestBuilder` | Tools, logprobs, custom sampling, constraints, or web search | Configurable |

`TextMessages` and `VisionMessages` convert into `RequestBuilder` via `Into<RequestBuilder>` if you start simple and later need more control.

```rust
// Simple
let messages = TextMessages::new()
    .add_message(TextMessageRole::User, "Hello!");
let response = model.send_chat_request(messages).await?;

// Advanced
let request = RequestBuilder::new()
    .add_message(TextMessageRole::System, "You are helpful.")
    .add_message(TextMessageRole::User, "Hello!")
    .set_tools(tools)
    .with_sampling(SamplingParams { temperature: Some(0.7), ..Default::default() });
let response = model.send_chat_request(request).await?;
```

## Streaming

`Model::stream_chat_request` returns a `Stream` that implements `futures::Stream`:

```rust
use futures::StreamExt;
use mistralrs::*;

let mut stream = model.stream_chat_request(messages).await?;
while let Some(chunk) = stream.next().await {
    if let Response::Chunk(c) = chunk {
        if let Some(text) = c.choices.first().and_then(|ch| ch.delta.content.as_ref()) {
            print!("{text}");
        }
    }
}
```

## Structured Output

Derive `schemars::JsonSchema` on your type and the model will be constrained to produce valid JSON:

```rust
use mistralrs::*;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct City {
    name: String,
    country: String,
    population: u64,
}

let messages = TextMessages::new()
    .add_message(TextMessageRole::User, "Give me info about Paris.");

let city: City = model.generate_structured::<City>(messages).await?;
println!("{}: pop. {}", city.name, city.population);
```

## Tool Calling

### Manual tool definition

```rust
let tools = vec![Tool {
    tp: ToolType::Function,
    function: Function {
        description: Some("Get the weather for a location".to_string()),
        name: "get_weather".to_string(),
        parameters: Some(parameters_json),
    },
}];

let request = RequestBuilder::new()
    .add_message(TextMessageRole::User, "What's the weather in NYC?")
    .set_tools(tools);

let response = model.send_chat_request(request).await?;
```

### Using the `#[tool]` macro

```rust
use mistralrs::tool;

#[tool(description = "Get the current weather for a location")]
fn get_weather(
    #[description = "The city name"] city: String,
) -> Result<String> {
    Ok(format!("Sunny, 72F in {city}"))
}
```

See [Tool Calling](TOOL_CALLING.md) for full details, or the [`examples/advanced/tools/`](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/tools) example.

## Agents

`AgentBuilder` wraps the tool-calling loop, automatically dispatching tool calls and feeding results back:

```rust
use mistralrs::*;

let agent = AgentBuilder::new(model)
    .with_system_prompt("You are a helpful assistant with tools.")
    .with_sync_tool(get_weather_tool, get_weather_callback)
    .with_max_iterations(10)
    .build();

let response = agent.run("What's the weather in NYC and London?").await?;
println!("{}", response.final_text);
```

See the [`examples/advanced/agent/`](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/agent) example for streaming agents and the `#[tool]` macro.

## Blocking API

For non-async applications, use `BlockingModel`:

```rust
use mistralrs::blocking::BlockingModel;
use mistralrs::{IsqBits, ModelBuilder};

fn main() -> mistralrs::error::Result<()> {
    let model = BlockingModel::from_builder(
        ModelBuilder::new("Qwen/Qwen3-4B")
            .with_auto_isq(IsqBits::Four),
    )?;
    let answer = model.chat("What is 2+2?")?;
    println!("{answer}");
    Ok(())
}
```

> **Note:** `BlockingModel` creates its own tokio runtime. Do not call it from within an existing tokio runtime.

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

## Examples

The crate includes 48 runnable examples organized by topic:

| Category | Examples |
|---|---|
| **Getting Started** | `text_generation`, `streaming`, `vision`, `gguf`, `gguf_locally`, `embedding` |
| **Models** | `text_models`, `vision_models`, `audio`, `diffusion`, `speech`, `multimodal` |
| **Quantization** | `isq`, `imatrix`, `uqff`, `topology`, `mixture_of_quant_experts` |
| **Advanced** | `tools`, `agent`, `grammar`, `json_schema`, `web_search`, `mcp_client`, `batching`, `paged_attn`, `speculative`, `lora`, `error_handling`, and more |
| **Cookbook** | `cookbook_rag`, `cookbook_structured`, `cookbook_multiturn`, `cookbook_agent` |

Run any example with:
```bash
cargo run --release --features <features> --example <name>
```

Browse all examples: [`mistralrs/examples/`](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples)
