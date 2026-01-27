<a name="top"></a>
<h1 align="center">
  mistral.rs
</h1>

<h3 align="center">
Blazingly fast LLM inference.
</h3>

<p align="center">
| <a href="https://ericlbuehler.github.io/mistral.rs/mistralrs/"><b>Rust Documentation</b></a> | <a href="https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/API.md"><b>Python Documentation</b></a> | <a href="https://discord.gg/SZrecqK8qw"><b>Discord</b></a> | <a href="https://matrix.to/#/#mistral.rs:matrix.org"><b>Matrix</b></a> |
</p>

<p align="center">
  <a href="https://github.com/EricLBuehler/mistral.rs/stargazers">
    <img src="https://img.shields.io/github/stars/EricLBuehler/mistral.rs?style=social&label=Star" alt="GitHub stars">
  </a>
</p>

**Mistral.rs is a cross-platform, highly-multimodal inference engine that brings you:**
- All-in-one multimodal workflow: text↔text, text+vision↔text, text+vision+audio↔text, text→speech, text→image, text→embeddings
- APIs: Rust, Python, OpenAI HTTP server (with Chat Completions, [Responses API compatible with OpenResponses](docs/OPENRESPONSES.md)), MCP server
- 🔗 **MCP Client**: Connect to external tools and services automatically (file systems, web search, databases, APIs)
- Performance: ISQ, PagedAttention, FlashAttention, [MLA](docs/MLA.md), **per-layer topology optimization**
- Support for embedding, speech generation, and image generation models

Please submit requests for new models [here](https://github.com/EricLBuehler/mistral.rs/issues/156).

## Supported Models

<details>
<summary><b>Text Models</b></summary>

- Granite 4.0
- SmolLM 3
- DeepSeek V3
- GPT-OSS
- DeepSeek V2
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

- Qwen 3-VL**
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

## Get started fast 🚀

1) [Install](#installation-and-build)

2) [Get models](#getting-models)

3) Deploy with our easy to use APIs
    - [Python](examples/python)
    - [Rust](mistralrs/examples)
    - [OpenAI-compatible HTTP server](README.md#openai-http-server)
    - [Interactive mode](README.md#interactive-mode)
    - 🔗 [**MCP Client**](examples/MCP_QUICK_START.md) - Connect to external tools automatically

4) Try the **web chat app** for local in-browser conversation (text, vision, and speech support):
    ```bash
    mistralrs serve --ui -m Qwen/Qwen3-4B
    ```
    Then visit [http://localhost:8080/ui](http://localhost:8080/ui) in your browser.

<br>

<!-- Web Chat App -->
<details open>
  <summary>🖥️ <strong>Web Chat App</strong></summary>
  <br>
  <img src="./res/chat.gif" alt="Web Chat UI Demo" />
  <br>
  Try our modern in-browser chat with text, vision, and speech support (TTS generation).
</details>

<!-- Interactive Mode -->
<details>
  <summary>💻 <strong>Terminal Interactive Mode</strong></summary>
  <br>
  <img src="./res/demo.gif" alt="Terminal Interactive Mode" />
  <br>
  Prefer the terminal? Use interactive mode for a classic CLI experience.
</details>

<br>

## Quick examples

*After following installation instructions*

- 🔎 Generate embeddings with **EmbeddingGemma** or **Qwen3 Embedding** across APIs: [EmbeddingGemma guide](docs/EMBEDDINGGEMMA.md) | [Qwen3 guide](docs/QWEN3_EMBEDDING.md) | [overview](docs/EMBEDDINGS.md)  
  <details>
    <summary>Show commands</summary>

    ```bash
    # HTTP API (OpenAI-compatible)
    mistralrs serve -p 1234 -m google/embeddinggemma-300m

    # Python API example
    python examples/python/embedding_gemma.py

    # Rust API example
    cargo run --package mistralrs --example embedding_gemma

    # Qwen3 Embedding server
    mistralrs serve -p 1234 -m Qwen/Qwen3-Embedding-0.6B

    # Qwen3 Embedding Python example
    python examples/python/qwen3_embedding.py

    # Qwen3 Embedding Rust example
    cargo run --package mistralrs --example qwen3_embedding
    ```
  </details>

- 💎🪆💎🪆💎 Run the **Gemma 3n** family (E2B, E4B) with **vision**, **audio**, and **MatFormer** support: [documentation](docs/GEMMA3N.md)  
  <details>
    <summary>Show commands</summary>

    **Normal use, run the full model (E4B or E2B):**
    ```bash
    mistralrs run --isq 8 -m google/gemma-3n-E4B-it
    ```

    **Use [MatFormer](docs/GEMMA3N.md#using-matformer-with-gemma-3n) to get a balanced smaller model:**
    ```bash
    mistralrs run --isq 8 -m google/gemma-3n-E4B-it \
      --matformer-config-path matformer_configs/gemma3n.csv \
      --matformer-slice-name "Config for E2.49B (block-level)"
    ```
  </details>

- 🧠+📷 Run the **Qwen 3 VL** reasoning vision models with full tool-calling support: [documentation](docs/QWEN3VL.md)  
  <details>
    <summary>Show command</summary>

    ```bash
    mistralrs run --isq 8 -m Qwen/Qwen3-VL-4B-Thinking
    ```
  </details>
  

- 🤗🤗🤗 Run the **SmolLM 3** long-context hybrid-reasoning model with full tool-calling support: [documentation](docs/SMOLLM3.md)  
  <details>
    <summary>Show command</summary>

    **Default, easiest:**
    ```bash
    mistralrs run --isq 8 -m HuggingFaceTB/SmolLM3-3B
    ```

    **UQFF prequantized:**
    ```bash
    mistralrs run -m EricB/SmolLM3-3B-UQFF --from-uqff smollm33b-q4k-0.uqff
    ```
  </details>

- 🔊 Run the **Dia 1.6b** model for highly-realistic dialogue generation: [documentation](docs/DIA.md)  
  <details>
    <summary>Show command</summary>

    ```bash
    mistralrs run -m nari-labs/Dia-1.6B
    ```
  </details>

- 🦙 Run the **Llama 3.\* and Llama 4** models with long context & vision support: [docs (llama 3.2)](docs/VLLAMA.md), [docs (llama 4)](docs/LLAMA4.md)  
  <details>
    <summary>Show commands</summary>

    **Llama 4:**

    ```bash
    mistralrs run --isq 4 -m meta-llama/Llama-4-Scout-17B-16E-Instruct
    ```

    **Llama 3.1/3.2/3.3:**

    ```bash
    mistralrs run --isq 8 -m meta-llama/Llama-3.2-3B-Instruct
    ```

    **Llama 3.2 vision:**

    ```bash
    mistralrs run --isq 8 -m meta-llama/Llama-3.2-11B-Vision-Instruct
    ```

  </details>

- 💎💎💎 Run the **Gemma 3** family (1b, 4b, 12b, 27b) with 128k context & vision support: [documentation](docs/GEMMA3.md)  
  <details>
    <summary>Show command</summary>

    ```bash
    mistralrs run --isq 8 -m google/gemma-3-4b-it
    ```
  </details>

- 🌲📷 Run the **FLUX.1** diffusion model: [documentation](docs/FLUX.md)  
  <details>
    <summary>Show command</summary>

    ```bash
    mistralrs run -m black-forest-labs/FLUX.1-schnell
    ```
  </details>

- 🧠 Run the **Qwen 3** hybrid-reasoning model with full tool-calling support: [documentation](docs/QWEN3.md)  
  <details>
    <summary>Show command</summary>

    ```bash
    mistralrs run --isq 8 -m Qwen/Qwen3-8B
    ```
  </details>

- 🔗 **MCP Client** - Connect to external tools and services automatically: [**Quick Start Guide**](examples/MCP_QUICK_START.md)
  <details>
    <summary>Show examples</summary>

    **1. Create config file (`mcp-config.json`):**
    ```json
    {
      "servers": [{
        "name": "Filesystem Tools",
        "source": {
          "type": "Process",
          "command": "npx",
          "args": ["@modelcontextprotocol/server-filesystem", "/tmp", "-y"]
        }
      }],
      "auto_register_tools": true
    }
    ```

    **2. Start server with tools:**
    ```bash
    mistralrs serve --mcp-config mcp-config.json -p 1234 -m Qwen/Qwen3-4B
    ```

    **3. Tools work automatically:**
    ```bash
    curl -X POST http://localhost:1234/v1/chat/completions \
      -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"List files in /tmp and create hello.txt"}]}'
    ```

    **Python API:**
    ```python
    mcp_config = mistralrs.McpClientConfigPy(
        servers=[mistralrs.McpServerConfigPy(
            name="Filesystem",
            source=mistralrs.McpServerSourcePy.Process(
                command="npx",
                args=["@modelcontextprotocol/server-filesystem", "/tmp", "-y"]
            )
        )],
        auto_register_tools=True
    )

    runner = mistralrs.Runner(
        which=mistralrs.Which.Plain(model_id="Qwen/Qwen3-4B"),
        mcp_client_config=mcp_config
    )
    # Tools automatically available!
    ```

    **Rust API:**
    ```rust
    let model = TextModelBuilder::new("Qwen/Qwen3-4B")
        .with_mcp_client(mcp_config) // Tools automatically available!
        .build().await?;
    ```
  </details>

- ⚡ **Smart Per-Layer Optimization** - Fine-tune quantization and device placement per layer: [documentation](docs/TOPOLOGY.md)
  <details>
    <summary>Show examples</summary>

    **Optimize memory usage with mixed quantization (fits large models in limited VRAM):**
    ```bash
    # Use aggressive quantization on less important layers, preserve quality on critical ones
    mistralrs run --topology topologies/isq.yml -m meta-llama/Llama-3.2-8B-Instruct
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
   - Load [multiple models](docs/multi_model/README.md) and unload/reload at runtime.

5. **Advanced Features**
   - High-throughput with [PagedAttention](docs/PAGED_ATTENTION.md) & FlashAttention V2/V3
   - Prefix caching (including multimodal)
   - [UQFF format](docs/UQFF.md) for custom quantization
   - Speculative decoding across models
   - ⭐ Agentic [web search integration](docs/WEB_SEARCH.md)

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

- [API Docs](docs/HTTP.md) - includes chat completions, completions, and [**OpenResponses API**](docs/OPENRESPONSES.md) for stateful conversations
- [Launching the server or use the CLI](README.md#using-the-cli)
- [Example](examples/server/chat.py)
- [Responses API examples](examples/server/responses.py) - maintain conversation context without resending history
- [Use or extend the server in other axum projects](https://ericlbuehler.github.io/mistral.rs/mistralrs_server_core/)
- **MCP Client**: Configure via `--mcp-config` flag for automatic tool integration - [Quick Start](examples/MCP_QUICK_START.md)

### MCP Protocol

Serve the same models over the open [MCP](docs/MCP/server.md) (Model Context Protocol) in parallel to the HTTP API:

```bash
mistralrs serve --mcp-port 4321 -m Qwen/Qwen3-4B
```

See the [docs](docs/MCP/server.md) for feature flags, examples and limitations.


### Llama Index integration

- Docs: https://docs.llamaindex.ai/en/stable/examples/llm/mistral_rs/

---

## Supported accelerators

| Accelerator              | Feature Flag  | Additional Flags       |
|--------------------------|---------------|------------------------|
| NVIDIA GPUs (CUDA)       | `cuda`        | `flash-attn`, `flash-attn-v3`, `cudnn`  |
| Apple Silicon GPU (Metal)| `metal`       |                        |
| CPU (Intel)              | `mkl`         |                        |
| CPU (Apple Accelerate)   | `accelerate`  |                        |
| Generic CPU (ARM/AVX)    | _none_        | ARM NEON / AVX enabled by default |

To enable one or more features, pass them to Cargo. For example:

```bash
cargo build --release --features "cuda flash-attn cudnn"
```

> **Note for Linux users:** The `metal` feature is macOS-only and should not be used on Linux. Use `--features "cuda flash-attn cudnn"` for NVIDIA GPUs or `--features mkl` for Intel CPUs instead of `--all-features`.

## Installation and Build

### Quick Install (Recommended)

The install script automatically detects your hardware (CUDA, Metal, MKL) and builds with optimal features.

**Linux/macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex
```

### Manual Installation

<details>
<summary>Click to expand manual installation steps</summary>

#### Prerequisites

1. Install required packages:
   - OpenSSL: `sudo apt install libssl-dev` (Ubuntu)
   - pkg-config (Linux only): `sudo apt install pkg-config`

2. Install Rust from https://rustup.rs/
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

3. (Optional) Set up HuggingFace authentication:
   ```bash
   mistralrs login
   ```
   Or use `huggingface-cli login` as documented [here](https://huggingface.co/docs/huggingface_hub/en/installation).

#### Feature Detection

Determine which features to enable based on your hardware:

| Hardware | Features |
|----------|----------|
| NVIDIA GPU (Ampere+, compute >=80) | `cuda cudnn flash-attn` |
| NVIDIA GPU (Hopper, compute 90) | `cuda cudnn flash-attn flash-attn-v3` |
| NVIDIA GPU (older) | `cuda cudnn` |
| Apple Silicon (macOS) | `metal accelerate` |
| Intel CPU with MKL | `mkl` |
| CPU only | (no features needed) |

#### Install from crates.io

```bash
cargo install mistralrs-cli --features "<your-features>"
```

#### Build from Source

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
cargo install --path mistralrs-cli --features "<your-features>"
```

</details>

### Docker

> Note: You can use our [Docker containers here](https://github.com/EricLBuehler/mistral.rs/pkgs/container/mistral.rs).
> Learn more about running Docker containers: https://docs.docker.com/engine/reference/run/

### Python Package

- Install the [Python package here](mistralrs-pyo3/_README.md).
- The Python package has [wheels on PyPi](mistralrs-pyo3/_README.md#installation-from-pypi)!

### After Installation

Use our APIs and integrations: [APIs and integrations list](#apis-and-integrations)

## Getting models
<details>
<summary>Show: How to get models (Hub, local, GGUF, adapters, etc.)</summary>

### Getting models from Hugging Face Hub
- **Default:** Downloads from Hugging Face Hub.
- For gated models, you can optionally set token source:
    - CLI: `mistralrs run --token-source env:HF_TOKEN -m <model>`
    - Python: See [examples/python/token_source.py](examples/python/token_source.py)
    - If no token is found, tries `~/.cache/huggingface/token` or runs with no token.

### Loading models from local files
- Pass a path to a downloaded model from Hugging Face hub:
    - Example:
      ```bash
      mistralrs run -m path/to/model
      ```

### Running GGUF models
- Minimal example:
  ```bash
  mistralrs run --format gguf -m author/model-repo -f model-quant.gguf
  ```
- Specify tokenizer (if needed):
  ```bash
  mistralrs run --format gguf -m author/model-repo -f file.gguf -t author/official-tokenizer
  ```
  (Or use the built-in GGUF tokenizer.)

### Adapters, X-LoRA, LoRA, Chat Templates
- Use the `--lora` or `--xlora` flags to load adapters.
- See [docs/ADAPTER_MODELS.md](docs/ADAPTER_MODELS.md) for details.
- For chat templates: usually auto-detected, override with `--chat-template <file>`.
  See [docs/CHAT_TOK.md](docs/CHAT_TOK.md).

### More model CLI examples
- See [Using the CLI](#using-the-cli) below or [full CLI documentation](docs/CLI.md).

</details>

## Using the CLI

The `mistralrs` CLI provides commands for interactive mode, HTTP server, benchmarking, and more. Run `mistralrs --help` to see all available commands.

> **ℹ️ Auto-detection:** The CLI auto-detects model types (**text**, **vision**, **embedding**, **diffusion**, **speech**) so you don't need to specify them explicitly.

For complete CLI documentation, see [docs/CLI.md](docs/CLI.md).

### Commands Overview

| Command | Description |
|---------|-------------|
| `mistralrs run` | Interactive chat mode |
| `mistralrs serve` | HTTP server with OpenAI-compatible API |
| `mistralrs bench` | Performance benchmarking |
| `mistralrs tune` | Get quantization recommendations |
| `mistralrs quantize` | Generate UQFF quantized files |
| `mistralrs doctor` | System diagnostics |
| `mistralrs login` | HuggingFace authentication |
| `mistralrs cache` | Manage HF model cache |
| `mistralrs from-config` | Run from TOML config file |

### Common Options

| Option | Description |
|--------|-------------|
| `-m, --model-id <MODEL>` | HuggingFace model ID or local path |
| `--isq <TYPE>` | Apply ISQ quantization (e.g., `q4k`, `q8_0`) |
| `-c, --chat-template <FILE>` | Override chat template with JINJA file |
| `--token-source <SOURCE>` | HF token source: `cache`, `literal:<TOKEN>`, `env:<VAR>`, `path:<FILE>`, `none` |
| `-n, --device-layers <SPEC>` | Device layer mapping (e.g., `0:16;1:16` for multi-GPU) |
| `--enable-search` | Enable web search integration |
| `--mcp-config <FILE>` | Path to MCP client configuration JSON file |
| `--enable-thinking` | Enable thinking mode for supported models (Qwen3, SmolLM3) |
| `--seed <SEED>` | Set random seed for reproducibility |
| `-l, --log <FILE>` | Log requests/responses to file |

### PagedAttention Options

| Option | Description |
|--------|-------------|
| `--paged-attn <MODE>` | PagedAttention mode: `auto`, `on`, or `off` |
| `--pa-context-len <LENGTH>` | Allocate KV cache for this context length |
| `--pa-memory-mb <MB>` | GPU memory for KV cache in megabytes |
| `--pa-memory-fraction <RATIO>` | Fraction of GPU memory (0.0-1.0) |
| `--pa-block-size <SIZE>` | Block size for PagedAttention |
| `--pa-cache-type <TYPE>` | KV cache type: `auto` or `f8e4m3` |

### Interactive Mode

**Llama 3.2 3B running on an M3 Max with 8-bit ISQ:**

<img src="./res/demo.gif" alt="Interactive demo" />

Launch interactive mode with `mistralrs run`:

```bash
mistralrs run -m meta-llama/Llama-3.2-3B-Instruct
```

Vision models work seamlessly (auto-detected):

```bash
mistralrs run -m lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k
```

Diffusion models (auto-detected):

```bash
mistralrs run -m black-forest-labs/FLUX.1-schnell
```

Speech generation in your terminal (auto-detected):

```bash
mistralrs run -m nari-labs/Dia-1.6B
```

### OpenAI HTTP Server

Launch an HTTP server with `mistralrs serve`:

```bash
mistralrs serve -p 1234 -m microsoft/Phi-3.5-MoE-instruct
```

Or with the built-in web UI:

```bash
mistralrs serve --ui -m microsoft/Phi-3.5-MoE-instruct
```

You can find documentation about the server itself [here](docs/HTTP.md).

### Multi-model Support

Serve multiple models simultaneously from a single server instance using a TOML config file. This is useful for comparing models, A/B testing, or serving different models for different use cases.

```bash
mistralrs from-config --file multi-model-config.toml
```

Select models in your requests using the `model` parameter:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-3B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Model unloading**: Free memory by unloading models while preserving configuration for automatic reload on next request.

📖 **[Complete multi-model documentation →](docs/multi_model/README.md)**

### Structured Selection with a TOML File

Use a TOML configuration file for complex setups. See [docs/CLI_CONFIG.md](docs/CLI_CONFIG.md) for the full format.

Example:
```bash
mistralrs from-config --file config.toml
```

### Architecture for plain models

> Note: for plain models, you can specify the data type to load and run in. This must be one of `f32`, `f16`, `bf16` or `auto` to choose based on the device. This is specified in the `--dype`/`-d` parameter after the model architecture (`plain`). For quantized models (gguf/ggml), you may specify data type of `f32` or `bf16` (`f16` is not recommended due to its lower precision in quantized inference).

If you do not specify the architecture, an attempt will be made to use the model's config. If this fails, please raise an issue.

<details>
  <summary>Show plain architectures</summary>

- `mistral`
- `gemma`
- `mixtral`
- `llama`
- `phi2`
- `phi3`
- `phi3.5moe`
- `qwen2`
- `gemma2`
- `glm4`
- `glm4moelite`
- `glm4moe`
- `starcoder2`
- `deepseekv2`
- `deepseekv3`
- `qwen3`
- `qwen3moe`
- `smollm3`
- `granitemoehybrid`
- `gpt_oss`

</details>

### Architecture for vision models

> Note: for vision models, you can specify the data type to load and run in. This must be one of `f32`, `f16`, `bf16` or `auto` to choose based on the device. This is specified in the `--dype`/`-d` parameter after the model architecture (`vision-plain`).

<details>
  <summary>Show vision architectures</summary>

- `phi3v`
- `idefics2`
- `llava_next`
- `llava`
- `vllama`
- `qwen2vl`
- `idefics3`
- `minicpmo`
- `phi4mm`
- `qwen2_5vl`
- `gemma3`
- `mistral3`
- `llama4`
- `gemma3n`
- `qwen3vl`

</details>

### Architecture for embedding models

> Note: for embedding models, you can specify the data type to load and run in. This must be one of `f32`, `f16`, `bf16` or `auto` to choose based on the device. This is specified in the `--dype`/`-d` parameter after the model architecture (`vision-plain`).

<details>
  <summary>Show embedding architectures</summary>

- `embeddinggemma`
- `qwen3embedding`

</details>

### Supported GGUF architectures

<details>
  <summary>Show supported GGUF architectures</summary>

**Plain:**
- llama
- phi2
- phi3
- starcoder2
- qwen2
- qwen3

**With adapters:**
- llama
- phi3

</details>

---

Please submit more benchmarks via raising an issue!

## Supported models

<details>
<summary>Show quantization support</summary>

**Quantization support**
|Model|GGUF|GGML|ISQ|
|--|--|--|--|
|Mistral|✅| |✅|
|Gemma| | |✅|
|Llama|✅|✅|✅|
|Mixtral|✅| |✅|
|Phi 2|✅| |✅|
|Phi 3|✅| |✅|
|Phi 3.5 MoE| | |✅|
|Qwen 2.5| | |✅|
|Phi 3 Vision| | |✅|
|Idefics 2| | |✅|
|Gemma 2| | |✅|
|GLM4| | |✅|
|GLM-4.7-Flash (MoE)| | |✅|
|GLM-4.7 (MoE)| | |✅|
|Starcoder 2| |✅|✅|
|LLaVa Next| | |✅|
|LLaVa| | |✅|
|Llama 3.2 Vision| | |✅|
|Qwen2-VL| | |✅|
|Idefics 3| | |✅|
|Deepseek V2| | |✅|
|Deepseek V3| | |✅|
|MiniCPM-O 2.6| | |✅|
|Qwen2.5-VL| | |✅|
|Gemma 3| | |✅|
|Mistral 3| | |✅|
|Llama 4| | |✅|
|Qwen 3|✅| |✅|
|SmolLM3| | |✅|
|Dia 1.6b| | |✅|
|Gemma 3n| | |✅|
|Qwen 3 VL | |✅|
|Granite 4.0| | |✅|
|GPT-OSS| | |✅|
</details>

<details>
<summary>Show device mapping support</summary>

**Device mapping support**
|Model category|Supported|
|--|--|
|Plain|✅|
|GGUF|✅|
|GGML| |
|Vision Plain|✅|
</details>

<details>
<summary>Show X-LoRA and LoRA support</summary>

**X-LoRA and LoRA support**
|Model|X-LoRA|X-LoRA+GGUF|X-LoRA+GGML|
|--|--|--|--|
|Mistral|✅|✅| |
|Gemma|✅| | |
|Llama|✅|✅|✅|
|Mixtral|✅|✅| |
|Phi 2|✅| | |
|Phi 3|✅|✅| |
|Phi 3.5 MoE| | | |
|Qwen 2.5| | | |
|Phi 3 Vision| | | |
|Idefics 2| | | |
|Gemma 2|✅| | |
|GLM4|✅| | |
|GLM-4.7-Flash (MoE)| | | |
|GLM-4.7 (MoE)| | | |
|Starcoder 2|✅| | |
|LLaVa Next| | | |
|LLaVa| | | |
|Qwen2-VL| | | |
|Idefics 3| | | |
|Deepseek V2| | | |
|Deepseek V3| | | |
|MiniCPM-O 2.6| | | |
|Qwen2.5-VL| | | |
|Gemma 3| | | |
|Mistral 3| | | |
|Llama 4| | | |
|Qwen 3| | | |
|SmolLM3|✅| | |
|Gemma 3n| | | |
|Qwen 3 VL | | |
|Granite 4.0| | | |
|GPT-OSS| | | |
</details>

<details>
<summary>Show AnyMoE support</summary>

**AnyMoE support**
|Model|AnyMoE|
|--|--|
|Mistral 7B|✅|
|Gemma|✅|
|Llama|✅|
|Mixtral| |
|Phi 2|✅|
|Phi 3|✅|
|Phi 3.5 MoE| |
|Qwen 2.5|✅|
|Phi 3 Vision| |
|Idefics 2| |
|Gemma 2|✅|
|GLM-4.7-Flash (MoE)| |
|GLM-4.7 (MoE)| |
|Starcoder 2|✅|
|LLaVa Next|✅|
|LLaVa|✅|
|Llama 3.2 Vision| |
|Qwen2-VL| |
|Idefics 3|✅|
|Deepseek V2| |
|Deepseek V3| |
|MiniCPM-O 2.6| |
|Qwen2.5-VL| |
|Gemma 3|✅|
|Mistral 3|✅|
|Llama 4| |
|Qwen 3| |
|SmolLM3|✅|
|Gemma 3n| |
|Qwen 3 VL | |
|Granite 4.0| |
|GPT-OSS| |
</details>

### Using Derivative and Adapter Models

Model type is auto-detected. Use flags for quantized models and adapters:

- **See all options:** Run `mistralrs run --help` or `mistralrs serve --help`
- **Docs:** [Adapter models](docs/ADAPTER_MODELS.md), [Chat templates](docs/CHAT_TOK.md)

<details>
<summary>Arguments by model type</summary>

| Model Type          | Required Arguments                                                     |
|---------------------|-----------------------------------------------------------------------|
| Plain               | `-m <model-id>`                                                       |
| GGUF Quantized      | `-m <model-id> --format gguf -f <file>`                               |
| ISQ Quantized       | `-m <model-id> --isq <level>`                                         |
| UQFF Quantized      | `-m <model-id> --from-uqff <file>`                                    |
| LoRA                | `-m <model-id> --lora <adapter>`                                      |
| X-LoRA              | `-m <model-id> --xlora <adapter> --xlora-order <file>`                |

</details>

<details>
<summary>Example: Zephyr GGUF model</summary>

```bash
mistralrs serve -p 1234 --log output.txt --format gguf -t HuggingFaceH4/zephyr-7b-beta -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q5_0.gguf
```
</details>

Chat template and tokenizer are usually auto-detected.
If you need to override, see the [chat templates doc](docs/CHAT_TOK.md).

Please find docs for adapter models [here](docs/ADAPTER_MODELS.md). Examples may be found [here](docs/LORA_XLORA.md).

### Chat Templates and Tokenizer
Mistral.rs will attempt to automatically load a chat template and tokenizer. This enables high flexibility across models and ensures accurate and flexible chat templating. However, this behavior can be customized. Please find detailed documentation [here](docs/CHAT_TOK.md).

## Contributing

Thank you for contributing! If you have any problems or want to contribute something, please raise an issue or pull request.
If you want to add a new model, please contact us via an issue and we can coordinate how to do this.

## FAQ
- Debugging with the environment variable `MISTRALRS_DEBUG=1` causes the following things
    - If loading a GGUF or GGML model, this will output a file containing the names, shapes, and types of each tensor.
        - `mistralrs_gguf_tensors.txt` or `mistralrs_ggml_tensors.txt`
    - More logging.
- Setting the CUDA compiler path:
    - Set the `NVCC_CCBIN` environment variable during build.
- Error: `recompile with -fPIE`:
    - Some Linux distributions require compiling with `-fPIE`.
    - Set the `CUDA_NVCC_FLAGS` environment variable to `-fPIE` during build: `CUDA_NVCC_FLAGS=-fPIE`
- Error `CUDA_ERROR_NOT_FOUND` or symbol not found when using a normal or vison model:
    - For non-quantized models, you can specify the data type to load and run in. This must be one of `f32`, `f16`, `bf16` or `auto` to choose based on the device.
- What is the minimum supported CUDA compute cap?
    - The minimum CUDA compute cap is **5.3**.
- Metal not found (error: unable to find utility "metal", not a developer tool or in PATH)
    1) Install Xcode: `xcode-select --install`
    2) Set the active developer directory: `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`
- Disabling Metal kernel precompilation:
    - By default, Metal kernels are precompiled during build time for better performance
    - To skip Metal kernel precompilation (useful for CI or when Metal is not needed), set `MISTRALRS_METAL_PRECOMPILE=0` or `MISTRALRS_METAL_PRECOMPILE=false`
    - Example: `MISTRALRS_METAL_PRECOMPILE=0 cargo build --release --features metal`
- Disabling mmap loading
  - Set `MISTRALRS_NO_MMAP=1` to disable mmap during loading.

## Environment Variables

### Runtime Environment Variables

| Variable | Description |
|----------|-------------|
| `MISTRALRS_DEBUG=1` | Enable debug mode: outputs tensor info files for GGUF/GGML models, increases logging verbosity |
| `MISTRALRS_NO_MMAP=1` | Disable memory-mapped file loading, forcing all tensor data into memory |
| `MISTRALRS_NO_MLA=1` | Disable [MLA](docs/MLA.md) (Multi-head Latent Attention) optimization for DeepSeek V2/V3 and GLM-4.7-Flash |
| `MISTRALRS_ISQ_SINGLETHREAD=1` | Force ISQ (In-Situ Quantization) to run single-threaded |
| `MCP_CONFIG_PATH` | Fallback path for MCP client configuration (used if `--mcp-config` not provided) |
| `KEEP_ALIVE_INTERVAL` | SSE keep-alive interval in milliseconds (default: 10000) |
| `HF_HUB_CACHE` | Override Hugging Face Hub cache directory |

### Build-Time Environment Variables

| Variable | Description |
|----------|-------------|
| `MISTRALRS_METAL_PRECOMPILE=0` | Skip Metal kernel precompilation (useful for CI) |
| `NVCC_CCBIN` | Set CUDA compiler path |
| `CUDA_NVCC_FLAGS=-fPIE` | Required on some Linux distributions |
| `CUDA_COMPUTE_CAP` | Override CUDA compute capability (e.g., "80" for RTX 3090) |

### Multi-Node Distributed Training

For multi-node setups, configure the head node and workers:

**Head Node:**
| Variable | Description |
|----------|-------------|
| `MISTRALRS_MN_GLOBAL_WORLD_SIZE` | Total number of devices across all nodes |
| `MISTRALRS_MN_HEAD_NUM_WORKERS` | Number of worker nodes |
| `MISTRALRS_MN_HEAD_PORT` | Port for head node communication |

**Worker Nodes:**
| Variable | Description |
|----------|-------------|
| `MISTRALRS_MN_WORKER_SERVER_ADDR` | Address of head server to connect to |
| `MISTRALRS_MN_WORKER_ID` | This worker's ID |
| `MISTRALRS_MN_LOCAL_WORLD_SIZE` | Number of GPUs on this node |
| `MISTRALRS_NO_NCCL=1` | Disable NCCL (use alternative backend) |

## Server Defaults

When running the HTTP server, these defaults apply:

| Setting | Default Value |
|---------|---------------|
| Server IP | `0.0.0.0` (all interfaces) |
| Max request body | 50 MB |
| Max running sequences | 16 |
| Prefix cache count | 16 |
| SSE keep-alive | 10 seconds |
| PagedAttention (CUDA) | Enabled |
| PagedAttention (Metal) | Disabled |
| PA GPU memory usage | 90% of free memory |
| PA block size | 32 tokens |

## Engine Behaviors

### Warmup Run

When a text or vision model is loaded in a multi-threaded runtime, mistral.rs automatically performs a warmup ("dummy") run:

- Sends a short completion request ("hello" with max 1 token) to initialize CUDA kernels and caches
- Logs "Beginning dummy run." when starting and "Dummy run completed in Xs." when finished
- Helps ensure more consistent performance for the first real user request
- Only runs for text and vision models (not diffusion/speech)

### Automatic Engine Recovery

If the inference engine thread dies unexpectedly (e.g., due to a panic), mistral.rs can automatically recover:

- Detects dead engine threads when sending requests
- Automatically reboots the engine using saved configuration
- Logs "Engine {model_id} is dead, rebooting" followed by "Successfully rebooted engine {model_id}"
- Preserves all original configuration including KV cache settings, prefix cache, and tool callbacks

This ensures high availability without manual intervention.

## Credits
This project would not be possible without the excellent work at [`candle`](https://github.com/huggingface/candle). Additionally, thank you to all contributors! Contributing can range from raising an issue or suggesting a feature to adding some new functionality.

<p align="right">
  <a href="#top">⬆️ Back to Top</a>
</p>
