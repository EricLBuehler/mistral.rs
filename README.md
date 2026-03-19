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

<p align="center">
  | <a href="https://ericlbuehler.github.io/mistral.rs/"><b>Documentation</b></a> | <a href="https://crates.io/crates/mistralrs"><b>Rust SDK</b></a> | <a href="https://ericlbuehler.github.io/mistral.rs/PYTHON_SDK.html"><b>Python SDK</b></a> | <a href="https://discord.gg/SZrecqK8qw"><b>Discord</b></a> |
</p>

<p align="center">
  <a href="https://github.com/EricLBuehler/mistral.rs/stargazers">
    <img src="https://img.shields.io/github/stars/EricLBuehler/mistral.rs?style=social&label=Star" alt="GitHub stars">
  </a>
</p>

## Why mistral.rs?

- **Any HuggingFace model, zero config**: Just `mistralrs run -m user/model`. Auto-detects architecture, quantization, chat template.
- **True multimodality**: Vision, audio, speech generation, image generation, embeddings.
- **Not another model registry**: Use HuggingFace models directly. No converting, no uploading to a separate service.
- **Full quantization control**: Choose the precise quantization you want to use, or make your own UQFF with `mistralrs quantize`.
- **Built-in web UI**: `mistralrs serve --ui` gives you a web interface instantly.
- **Hardware-aware**: `mistralrs tune` benchmarks your system and picks optimal quantization + device mapping.
- **Flexible SDKs**: Python package and Rust crate to build your projects.

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

[Manual installation & other platforms](docs/INSTALLATION.md)

### Run Your First Model

```bash
# Interactive chat
mistralrs run -m Qwen/Qwen3-4B

# Or start a server with web UI
mistralrs serve --ui -m google/gemma-3-4b-it
```

Then visit `http://localhost:1234/ui` for the web chat interface.

### The `mistralrs` CLI

The CLI is designed to be **zero-config**: just point it at a model and go.

- **Auto-detection**: Automatically detects model architecture, quantization format, and chat template
- **All-in-one**: Single binary for chat, server, benchmarks, and web UI (`run`, `serve`, `bench`)
- **Hardware tuning**: Run `mistralrs tune` to automatically benchmark and configure optimal settings for your hardware
- **Format-agnostic**: Works with Hugging Face models, GGUF files, and [UQFF quantizations](docs/UQFF.md) seamlessly

```bash
# Auto-tune for your hardware and emit a config file
mistralrs tune -m Qwen/Qwen3-4B --emit-config config.toml

# Run using the generated config
mistralrs from-config -f config.toml

# Diagnose system issues (CUDA, Metal, HuggingFace connectivity)
mistralrs doctor
```

[Full CLI documentation](docs/CLI.md)

<details open>
  <summary><b>Web Chat Demo</b></summary>
  <br>
  <img src="https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/res/chat.gif" alt="Web Chat UI Demo" />
</details>

## What Makes It Fast

**Performance**
- Continuous batching support by default on all devices.
- CUDA with [FlashAttention](docs/FLASH_ATTENTION.md) V2/V3, Metal, [multi-GPU tensor parallelism](docs/DISTRIBUTED/DISTRIBUTED.md)
- [PagedAttention](docs/PAGED_ATTENTION.md) for high throughput continuous batching on CUDA or Apple Silicon, prefix caching (including multimodal)

**Quantization** ([full docs](docs/QUANTS.md))
- [In-situ quantization (ISQ)](docs/ISQ.md) of any Hugging Face model
- GGUF (2-8 bit), GPTQ, AWQ, HQQ, FP8, BNB support
- ⭐ [Per-layer topology](docs/TOPOLOGY.md): Fine-tune quantization per layer for optimal quality/speed
- ⭐ Auto-select fastest quant method for your hardware

**Flexibility**
- [LoRA & X-LoRA](docs/ADAPTER_MODELS.md) with weight merging
- [AnyMoE](docs/ANYMOE.md): Create mixture-of-experts on any base model
- [Multiple models](docs/multi_model/README.md): Load/unload at runtime

**Agentic Features**
- Integrated [tool calling](docs/TOOL_CALLING.md) with Python/Rust callbacks
- ⭐ [Web search integration](docs/WEB_SEARCH.md)
- ⭐ [MCP client](docs/MCP/client.md): Connect to external tools automatically

[Full feature documentation](docs/README.md)

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
<summary><b>Vision Models</b></summary>

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

[Request a new model](https://github.com/EricLBuehler/mistral.rs/issues/156) | [Full compatibility tables](docs/SUPPORTED_MODELS.md)

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

- ⚡ **Smart Per-Layer Optimization** - Fine-tune quantization and device placement per layer: [documentation](docs/TOPOLOGY.md)
  <details>
    <summary>Show examples</summary>

    **Optimize memory usage with mixed quantization (fits large models in limited VRAM):**
    ```bash
    # Use aggressive quantization on less important layers, preserve quality on critical ones
    ./mistralrs-server -i --topology topologies/isq.yml run -m meta-llama/Llama-3.2-8B-Instruct
    ```

    **Example topology file (`topologies/isq.yml`):**
    ```yaml
    # Early layers: lower quantization for embeddings
    0-8:
      isq: Q3K
    # Middle layers: balanced quantization
    8-24:
      isq: Q4K
    # Final layers: higher quality for output
    24-32:
      isq: Q6K
    ```

    **Advanced: Target specific components with regex patterns:**
    ```yaml
    # Quantize attention layers differently from FFN layers
    '/attn\.q_proj$/':
      isq: Q4K
    '/ffn_.*\.weight$/':
      isq: Q3K
    ```

    **Multi-device deployment (split across GPUs/CPU):**
    ```yaml
    0-16:
      isq: Q4K
      device: cuda[0]
    16-32:
      isq: Q4K
      device: cuda[1]
    # Or offload some layers to CPU for very large models
    ```

    **Python example:**
    ```python
    runner = mistralrs.Runner(
        which=mistralrs.Which.Plain(
            model_id="meta-llama/Llama-3.2-8B-Instruct",
            topology="topologies/isq.yml",
        ),
    )
    ```
  </details>

## Description

[mistral.rs](https://github.com/EricLBuehler/mistral.rs) is a blazing-fast, cross-platform LLM inference engine with support for text, vision, image generation, and speech.

**Key Benefits:**

1. **Ease of Use**
   - [OpenAI-compatible HTTP server](docs/HTTP.md)
   - [Rust API](https://ericlbuehler.github.io/mistral.rs/mistralrs/) & [Python API](mistralrs-pyo3/API.md)
   - [Automatic device mapping](docs/DEVICE_MAPPING.md) (multi-GPU, CPU)
   - [Chat templates](docs/CHAT_TOK.md) & tokenizer auto-detection
   - [MCP server](docs/MCP/server.md) for structured, realtime tool calls
   - ⭐ [MCP client](examples/MCP_QUICK_START.md) to connect to external tools and services automatically

2. **Performance**
   - CPU acceleration (MKL, AVX, NEON, Accelerate)
   - GPU acceleration (CUDA with [FlashAttention](docs/FLASH_ATTENTION.md) & cuDNN, Metal)
   - Automatic [tensor parallelism](docs/DISTRIBUTED/DISTRIBUTED.md) for splitting models across multiple devices
     - CUDA-specialized [NCCL](docs/DISTRIBUTED/NCCL.md)
     - Heterogeneous, flexible [Ring backend](docs/DISTRIBUTED/RING.md)

3. **Quantization & Optimization**
   - ⭐ [**Per-layer topology**](docs/TOPOLOGY.md): Fine-tune quantization per layer for optimal quality/speed balance
   - [In-place quantization (ISQ)](docs/ISQ.md) of Hugging Face models
   - [GGML & GGUF support](docs/QUANTS.md): 2–8 bit
   - [GPTQ](docs/QUANTS.md), [AWQ](scripts/convert_awq_marlin.py), [AFQ](docs/QUANTS.md), [HQQ](docs/QUANTS.md), [FP8](docs/QUANTS.md), [BNB](https://github.com/TimDettmers/bitsandbytes) (int8/fp4/nf4)
   - ⭐ Auto-select the fastest quant method
   - [KV cache quantization](docs/PAGED_ATTENTION.md#kv-cache-quantization)

4. **Flexibility**
   - [LoRA](docs/ADAPTER_MODELS.md) & [X-LoRA](docs/ADAPTER_MODELS.md) adapters with weight merging
   - [AnyMoE](docs/ANYMOE.md): create MoE models on any base model
   - [Sampling & penalty options](docs/SAMPLING.md)
   - Prompt chunking for large inputs
   - Integrated [tool calling](docs/TOOL_CALLING.md) with customizable Python/Rust native tool and search callbacks

5. **Advanced Features**
   - High-throughput with [PagedAttention](docs/PAGED_ATTENTION.md) & FlashAttention V2/V3
   - Prefix caching (including multimodal)
   - [UQFF format](docs/UQFF.md) for custom quantization
   - Speculative decoding across models
   - ⭐ Agentic [web search integration](docs/WEB_SEARCH.md)
   - ⭐ **Parking Lot Scheduler**: Configurable thread pool and resource management for optimal throughput

### Parking Lot Scheduler Configuration

When using the `parking-lot-scheduler` feature, you can configure the thread pool and resource limits via YAML configuration file.

**Configuration File Locations (priority order):**
1. `--scheduler-config <path>` CLI flag
2. `MISTRALRS_SCHEDULER_CONFIG` environment variable
3. `~/.mistralrs-server/scheduler.yaml` (default location)
4. Built-in defaults if no config found

**CLI Override Flags:**
- `--worker-threads <n>`: Number of worker threads
- `--thread-stack-size <bytes>`: Thread stack size
- `--scheduler-max-units <n>`: Maximum KV cache blocks
- `--scheduler-max-queue <n>`: Maximum queue depth
- `--scheduler-timeout <secs>`: Request timeout

CLI flags take precedence over YAML configuration.

**Example Usage:**

```bash
# Use default config location
cargo run --release --features parking-lot-scheduler -- plain ...

# Use custom config file
cargo run --release --features parking-lot-scheduler -- \
  --scheduler-config ./my-scheduler.yaml plain ...

# Override specific settings via CLI
cargo run --release --features parking-lot-scheduler -- \
  --worker-threads 8 \
  --scheduler-max-units 8192 \
  plain ...

# Use environment variable
export MISTRALRS_SCHEDULER_CONFIG=~/configs/scheduler.yaml
cargo run --release --features parking-lot-scheduler -- plain ...
```

See [`examples/scheduler-config.yaml`](examples/scheduler-config.yaml) for full configuration documentation and hardware-specific profiles.

## APIs and Integrations

### Rust Crate

Rust multithreaded/async API for easy integration into any application.

- [Docs](https://ericlbuehler.github.io/mistral.rs/mistralrs/)
- [Examples](mistralrs/examples/) including [MCP client integration](mistralrs/examples/mcp_client)
- To use: add `mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs.git" }` to your Cargo.toml
- **MCP Client**: Connect to external tools automatically - [Quick Start](examples/MCP_QUICK_START.md)

### Python API

Python API for mistral.rs.

- [Installation including PyPI](mistralrs-pyo3/_README.md)
- [Docs](mistralrs-pyo3/API.md)
- [Examples](examples/python) including [MCP client usage](examples/python/mcp_client.py)
- [Cookbook](examples/python/cookbook.ipynb)
- **MCP Client**: Full MCP integration - [Quick Start](examples/MCP_QUICK_START.md)

### HTTP Server

OpenAI API compatible API server

- [API Docs](docs/HTTP.md) - includes chat completions, completions, and **Responses API** for stateful conversations
- [Launching the server or use the CLI](README.md#using-the-cli)
- [Example](examples/server/chat.py)
- [Responses API examples](examples/server/responses.py) - maintain conversation context without resending history
- [Use or extend the server in other axum projects](https://ericlbuehler.github.io/mistral.rs/mistralrs_server_core/)
- **MCP Client**: Configure via `--mcp-config` flag for automatic tool integration - [Quick Start](examples/MCP_QUICK_START.md)

### MCP Protocol

Serve the same models over the open [MCP](docs/MCP/server.md) (Model Context Protocol) in parallel to the HTTP API:

```bash
./mistralrs-server --mcp-port 4321 plain -m Qwen/Qwen3-4B
```

[Python SDK](https://ericlbuehler.github.io/mistral.rs/PYTHON_SDK.html) | [Installation](https://ericlbuehler.github.io/mistral.rs/PYTHON_INSTALLATION.html) | [Examples](examples/python) | [Cookbook](examples/python/cookbook.ipynb)

## Rust SDK

```bash
cargo add mistralrs
```

```rust
use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, TextMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new("google/gemma-3-4b-it")
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

## Build from Source

1) Download the code:

   ```bash
   git clone https://github.com/EricLBuehler/mistral.rs.git
   cd mistral.rs
   ```

2) <b>*Optional:*</b> Configure environment variables (see [`env.example`](env.example) for all available options)

   ```bash
   cp env.example .env
   # Edit .env to set your HF_TOKEN, MISTRALRS_SCHEDULER_CONFIG, etc.
   ```

3) Build or install `mistralrs-server`:

   - Build the `mistralrs-server` binary, which can be found at `target/release/mistralrs-server`.

     ```bash
     cargo build --release --features <specify feature(s) here>
     ```

   - Install with `cargo install` for easy command line usage.

     Pass the same values to `--features` as you would for `cargo build`.

     ```bash
     cargo install --path mistralrs-server --features <specify feature(s) here>
     ```

4) If you used `cargo build`, the build process will output a binary `mistralrs-server` at `./target/release/mistralrs-server`. You can switch to that directory so that the binary can be accessed as `./mistralrs-server`:

   ```bash
   cd target/release
   ```

5) Use the available APIs and integrations:

   [APIs and integrations list](#apis-and-integrations)

## Getting models
<details>
<summary>Show: How to get models (Hub, local, GGUF, adapters, etc.)</summary>

### Getting models from Hugging Face Hub
- **Default:** Downloads from Hugging Face Hub.
- For gated models, you can optionally set token source:
    - CLI: `./mistralrs-server --token-source env:HF_TOKEN ...`
    - Python: See [examples/python/token_source.py](examples/python/token_source.py)
    - If no token is found, tries `~/.cache/huggingface/token` or runs with no token.

### Loading models from local files
- Pass a path to a downloaded model from Hugging Face hub:
    - Example:  
      ```
      ./mistralrs-server -i run -m path/to/model
      ```

### Running GGUF models
- Minimal example:
  ```
  ./mistralrs-server gguf -m author/model-repo -f model-quant.gguf
  ```
- Specify tokenizer (if needed):
  ```
  ./mistralrs-server gguf -m author/model-repo -f file.gguf -t author/official-tokenizer
  ```
  (Or use the built-in GGUF tokenizer.)

### Adapters, X-LoRA, LoRA, Chat Templates
- Use the correct subcommand (`x-lora-*`, `lora-*`), pass model, adapter, or quant file as needed.
- See [docs/ADAPTER_MODELS.md](docs/ADAPTER_MODELS.md) for details.
- For chat templates: usually auto-detected, override with `--chat-template <file>`.  
  See [docs/CHAT_TOK.md](docs/CHAT_TOK.md).

### More model CLI examples
- See [Run with the CLI](#run-with-the-cli) below or [full documentation](docs/README.md).

</details>

## Using the CLI

Mistral.rs uses subcommands to control the model type. Please run `./mistralrs-server --help` to see the subcommands which categorize the models by kind.

> **🚨 Important:** The `run` subcommand (alias for `plain`/`vision-plain`) only auto-detects and runs **text** and **vision** models. It does **not** support **diffusion** or **speech** models. 
> To run a diffusion model (e.g. FLUX series), use the `diffusion` subcommand:
> ```bash
> mistralrs-server -i diffusion -m <model-id> [options]
> ```
> To run a speech model (e.g. Dia), use the `speech` subcommand:
> ```bash
> mistralrs-server -i speech -m <model-id> [options]
> ```
> If you attempt to use `run` with diffusion or speech models, model loading will fail.

### Interactive mode

**Llama 3.2 3B running on an M3 Max with 8-bit ISQ:**

<img src="./res/demo.gif" alt="Interactive demo" />

You can launch interactive mode, a simple chat application running in the terminal, by passing `-i`:

```bash
./mistralrs-server -i run -m meta-llama/Llama-3.2-3B-Instruct
```

## Documentation

For complete documentation, see the **[Documentation](https://ericlbuehler.github.io/mistral.rs/)**.

**Quick Links:**
- [CLI Reference](docs/CLI.md) - All commands and options
- [HTTP API](docs/HTTP.md) - OpenAI-compatible endpoints
- [Quantization](docs/QUANTS.md) - ISQ, GGUF, GPTQ, and more
- [Device Mapping](docs/DEVICE_MAPPING.md) - Multi-GPU and CPU offloading
- [MCP Integration](docs/MCP/README.md) - MCP integration documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Configuration](docs/CONFIGURATION.md) - Environment variables for configuration

## Contributing

Contributions welcome! Please [open an issue](https://github.com/EricLBuehler/mistral.rs/issues) to discuss new features or report bugs. If you want to add a new model, please contact us via an issue and we can coordinate.

## Credits

This project would not be possible without the excellent work at [Candle](https://github.com/huggingface/candle). Thank you to all [contributors](https://github.com/EricLBuehler/mistral.rs/graphs/contributors)!

mistral.rs is not affiliated with Mistral AI.

<p align="right">
  <a href="#top">Back to Top</a>
</p>
