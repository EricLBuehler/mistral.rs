<a name="top"></a>
<!--
<h1 align="center">
  mistral.rs
</h1>
-->

<div align="center">
  <img src="https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/res/banner.png" alt="mistral.rs" width="100%" style="max-width: 800px;">
</div>


<h3 align="center">
Fast, flexible LLM inference.
</h3>
<div align = "center">
  
[![Stars](https://img.shields.io/github/stars/ericlbuehler/mistral.rs?label=Stars&color=af68ff&logo=github&logoColor=white&labelColor=464646&style=for-the-badge)](https://github.com/ericlbuehler/mistral.rs/stargazers)
[![Forks](https://img.shields.io/github/forks/ericlbuehler/mistral.rs?label=Forks&color=ff7b72&logo=github&logoColor=white&labelColor=464646&style=for-the-badge)](https://github.com/ericlbuehler/mistral.rs/network)
[![Watchers](https://img.shields.io/github/watchers/ericlbuehler/mistral.rs?label=Watchers&color=2ea043&logo=github&logoColor=white&labelColor=464646&style=for-the-badge)](https://github.com/ericlbuehler/mistral.rs/watchers)
[![License](https://img.shields.io/github/license/ericlbuehler/mistral.rs?label=License&color=8957e5&logo=github&logoColor=white&labelColor=464646&style=for-the-badge)](https://github.com/ericlbuehler/mistral.rs/blob/main/LICENSE)
[![Downloads](https://img.shields.io/github/downloads/ericlbuehler/mistral.rs/total?label=Downloads&color=f0883e&logo=github&logoColor=white&labelColor=464646&style=for-the-badge)](https://github.com/ericlbuehler/mistral.rs/releases)
[![Release](https://img.shields.io/github/v/release/ericlbuehler/mistral.rs?label=Release&color=fd8c73&logo=github&logoColor=white&labelColor=464646&style=for-the-badge)](https://github.com/ericlbuehler/mistral.rs/releases)
[![Made with Rust](https://img.shields.io/badge/Made_with-Rust-f74c00?style=for-the-badge&logo=rust&logoColor=white&labelColor=464646)](https://www.rust-lang.org/)

</div>
<p align="center">
  | <a href="https://ericlbuehler.github.io/mistral.rs/"><b>Documentation</b></a> | <a href="https://crates.io/crates/mistralrs"><b>Rust SDK</b></a> | <a href="https://ericlbuehler.github.io/mistral.rs/PYTHON_SDK.html"><b>Python SDK</b></a> | <a href="https://discord.gg/SZrecqK8qw"><b>Discord</b></a> |
</p>

<p align="center">
  <a href="https://github.com/EricLBuehler/mistral.rs/stargazers">
    <img src="https://img.shields.io/github/stars/EricLBuehler/mistral.rs?style=social&label=Star" alt="GitHub stars">
  </a>
</p>

## Latest

- **Gemma 4**: Full multimodal: text, image, video, and audio input. [Guide](https://ericlbuehler.github.io/mistral.rs/GEMMA4.html) | [Video setup](https://ericlbuehler.github.io/mistral.rs/VIDEO.html)
- **MXFP4 ISQ quantization**: MXFP4 with optimized decode kernels for faster, smaller models. [Quantization docs](https://ericlbuehler.github.io/mistral.rs/QUANTS.html)
- **Qwen 3.5 model family**: Support for the Qwen 3.5 series including vision. [Guide](https://ericlbuehler.github.io/mistral.rs/QWEN3_5.html)

## Why mistral.rs?

- **Any Hugging Face model, zero config**: Just `mistralrs run -m user/model`.
- **True multimodality**: Text, vision, video, and audio, speech generation, image generation, and embeddings in one engine.
- **Full quantization control**: Choose the precise quantization you want to use, or make your own UQFF with `mistralrs quantize`.
- **Built-in web UI**: `mistralrs serve --ui` gives you a web interface instantly.
- **Hardware-aware**: `mistralrs tune` benchmarks your system and picks optimal quantization + device mapping.
- **Flexible SDKs**: Python package and Rust crate to build your projects.
- **[Agentic features](https://ericlbuehler.github.io/mistral.rs/AGENTS.html)** — server-side tool loop, web search, MCP client, and HTTP tool dispatch

## Quick Start

### Install

**Linux/macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex
```

[Manual installation & other platforms](https://ericlbuehler.github.io/mistral.rs/INSTALLATION.html)

### Run Your First Model

```bash
# Interactive chat
mistralrs run -m Qwen/Qwen3-4B

# One-shot prompt (no interactive session)
mistralrs run -m Qwen/Qwen3-4B -i "What is the capital of France?"

# One-shot with an image
mistralrs run -m google/gemma-4-E4B-it --image photo.jpg -i "Describe this image"

# Or start a server with web UI
mistralrs serve --ui -m google/gemma-4-E4B-it
```

Then visit `http://localhost:1234/ui` for the web chat interface.

### The `mistralrs` CLI

The CLI is designed to be **zero-config**: just point it at a model and go.

- **Auto-detection**: Automatically detects model architecture, quantization format, and chat template
- **All-in-one**: Single binary for chat, server, benchmarks, and web UI (`run`, `serve`, `bench`)
- **Hardware tuning**: Run `mistralrs tune` to automatically benchmark and configure optimal settings for your hardware
- **Format-agnostic**: Works with Hugging Face models, GGUF files, and [UQFF quantizations](https://ericlbuehler.github.io/mistral.rs/UQFF.html) seamlessly

```bash
# Auto-tune for your hardware and emit a config file
mistralrs tune -m Qwen/Qwen3-4B --emit-config config.toml

# Run using the generated config
mistralrs from-config -f config.toml

# Diagnose system issues (CUDA, Metal, HuggingFace connectivity)
mistralrs doctor
```

[Full CLI documentation](https://ericlbuehler.github.io/mistral.rs/CLI.html)

<details open>
  <summary><b>Web Chat Demo</b></summary>
  <br>
  <img src="https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/res/chat.gif" alt="Web Chat UI Demo" />
</details>

## What Makes It Fast

**Performance**
- Continuous batching support by default on all devices.
- CUDA with [FlashAttention](https://ericlbuehler.github.io/mistral.rs/FLASH_ATTENTION.html) V2/V3, Metal, [multi-GPU tensor parallelism](https://ericlbuehler.github.io/mistral.rs/DISTRIBUTED/DISTRIBUTED.html)
- [PagedAttention](https://ericlbuehler.github.io/mistral.rs/PAGED_ATTENTION.html) for high throughput continuous batching on CUDA or Apple Silicon, prefix caching (including multimodal)

**Quantization** ([full docs](https://ericlbuehler.github.io/mistral.rs/QUANTS.html))
- [In-situ quantization (ISQ)](https://ericlbuehler.github.io/mistral.rs/ISQ.html) of any Hugging Face model
- GGUF (2-8 bit), GPTQ, AWQ, HQQ, FP8, BNB support
- ⭐ [Per-layer topology](https://ericlbuehler.github.io/mistral.rs/TOPOLOGY.html): Fine-tune quantization per layer for optimal quality/speed
- ⭐ Auto-select fastest quant method for your hardware

**Flexibility**
- [LoRA & X-LoRA](https://ericlbuehler.github.io/mistral.rs/ADAPTER_MODELS.html) with weight merging
- [AnyMoE](https://ericlbuehler.github.io/mistral.rs/ANYMOE.html): Create mixture-of-experts on any base model
- [Multiple models](https://ericlbuehler.github.io/mistral.rs/multi_model/overview.html): Load/unload at runtime

**Agentic Features**
- Integrated [tool calling](https://ericlbuehler.github.io/mistral.rs/TOOL_CALLING.html) with grammar enforcement and strict schema mode
- ⭐ Server-side [agentic loop](https://ericlbuehler.github.io/mistral.rs/TOOL_CALLING.html#agentic-loop): auto-execute tools and feed results back
- ⭐ [Tool dispatch URL](https://ericlbuehler.github.io/mistral.rs/TOOL_CALLING.html#tool-dispatch-url): POST tool calls to your own endpoint
- ⭐ [Web search integration](https://ericlbuehler.github.io/mistral.rs/WEB_SEARCH.html) with embedding-based ranking
- ⭐ [MCP client](https://ericlbuehler.github.io/mistral.rs/MCP/client.html): Connect to external tools via Process, HTTP, or WebSocket
- Python/Rust [tool callbacks](https://ericlbuehler.github.io/mistral.rs/TOOL_CALLING.html#tool-callbacks) for custom execution

[Full feature documentation](https://ericlbuehler.github.io/mistral.rs/)

## Supported Models

<details>
<summary><b>Text Models</b></summary>

- Granite 4.0
- SmolLM 3
- DeepSeek V3
- GPT-OSS
- DeepSeek V2
- Qwen 3 Next
- Qwen 3 MoE
- Phi 3.5 MoE
- Qwen 3
- GLM 4
- GLM-4.7-Flash
- GLM-4.7 (MoE)
- Gemma 2
- Qwen 2
- Starcoder 2
- Phi 3
- Mixtral
- Phi 2
- Gemma
- Llama
- Mistral
</details>

<details>
<summary><b>Multimodal Models</b></summary>

- Qwen 3.5
- Qwen 3.5 MoE
- Qwen 3-VL
- Qwen 3-VL MoE
- Gemma 3n
- Llama 4
- Gemma 3
- Mistral 3
- Phi 4 multimodal
- Qwen 2.5-VL
- MiniCPM-O
- Llama 3.2 Vision
- Qwen 2-VL
- Idefics 3
- Idefics 2
- LLaVA Next
- LLaVA
- Phi 3V
</details>

<details>
<summary><b>Speech Models</b></summary>

- Voxtral (ASR/speech-to-text)
- Dia
</details>

<details>
<summary><b>Image Generation Models</b></summary>

- FLUX
</details>

<details>
<summary><b>Embedding Models</b></summary>

- Embedding Gemma
- Qwen 3 Embedding
</details>

[Request a new model](https://github.com/EricLBuehler/mistral.rs/issues/156) | [Full compatibility tables](https://ericlbuehler.github.io/mistral.rs/SUPPORTED_MODELS.html)

## Python SDK

```bash
pip install mistralrs  # or mistralrs-cuda, mistralrs-metal, mistralrs-mkl, mistralrs-accelerate
```

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
```

[Python SDK](https://ericlbuehler.github.io/mistral.rs/PYTHON_SDK.html) | [Installation](https://ericlbuehler.github.io/mistral.rs/PYTHON_INSTALLATION.html) | [Examples](examples/python) | [Cookbook](examples/python/cookbook.ipynb)

## Rust SDK

```bash
cargo add mistralrs
```

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, TextMessages, MultimodalModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = MultimodalModelBuilder::new("google/gemma-4-E4B-it")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "Hello!",
    );

    let response = model.send_chat_request(messages).await?;

    println!("{:?}", response.choices[0].message.content);

    Ok(())
}
```

[API Docs](https://docs.rs/mistralrs) | [Crate](https://crates.io/crates/mistralrs) | [Examples](mistralrs/examples)

## Docker

For quick containerized deployment:

```bash
docker pull ghcr.io/ericlbuehler/mistral.rs:latest
docker run --gpus all -p 1234:1234 ghcr.io/ericlbuehler/mistral.rs:latest \
  serve -m Qwen/Qwen3-4B
```

[Docker images](https://github.com/EricLBuehler/mistral.rs/pkgs/container/mistral.rs)

> For production use, we recommend installing the CLI directly for maximum flexibility.

## Documentation

For complete documentation, see the **[Documentation](https://ericlbuehler.github.io/mistral.rs/)**.

**Quick Links:**
- [CLI Reference](https://ericlbuehler.github.io/mistral.rs/CLI.html) - All commands and options
- [HTTP API](https://ericlbuehler.github.io/mistral.rs/HTTP.html) - OpenAI-compatible endpoints
- [Quantization](https://ericlbuehler.github.io/mistral.rs/QUANTS.html) - ISQ, GGUF, GPTQ, and more
- [Device Mapping](https://ericlbuehler.github.io/mistral.rs/DEVICE_MAPPING.html) - Multi-GPU and CPU offloading
- [MCP Integration](https://ericlbuehler.github.io/mistral.rs/MCP/client.html) - MCP integration documentation
- [Troubleshooting](https://ericlbuehler.github.io/mistral.rs/TROUBLESHOOTING.html) - Common issues and solutions
- [Configuration](https://ericlbuehler.github.io/mistral.rs/CONFIGURATION.html) - Environment variables for configuration

## Contributing

Contributions welcome! Please [open an issue](https://github.com/EricLBuehler/mistral.rs/issues) to discuss new features or report bugs. If you want to add a new model, please contact us via an issue and we can coordinate.

## Credits

This project would not be possible without the excellent work at [Candle](https://github.com/huggingface/candle). Thank you to all [contributors](https://github.com/EricLBuehler/mistral.rs/graphs/contributors)!

mistral.rs is not affiliated with Mistral AI.

<p align="right">
  <a href="#top">Back to Top</a>
</p>
