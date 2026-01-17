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
- Performance: ISQ, PagedAttention, FlashAttention, **per-layer topology optimization**
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
    - Quickstart [here](mistralrs-web-chat/README.md)
    - Run the server and visit [http://localhost:8080](http://localhost:8080) by default.

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
    ./mistralrs-server --port 1234 run -m google/embeddinggemma-300m

    # Python API example
    python examples/python/embedding_gemma.py

    # Rust API example
    cargo run --package mistralrs --example embedding_gemma

    # Qwen3 Embedding server
    ./mistralrs-server --port 1234 run -m Qwen/Qwen3-Embedding-0.6B

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
    ./mistralrs-server -i --isq 8 run -m google/gemma-3n-E4B-it
    ```

    **Use [MatFormer](docs/GEMMA3N.md#using-matformer-with-gemma-3n) to get a balanced smaller model:**
    ```bash
    ./mistralrs-server -i --isq 8 run -m google/gemma-3n-E4B-it \
      --matformer-config-path matformer_configs/gemma3n.csv \
      --matformer-slice-name "Config for E2.49B (block-level)"
    ```
  </details>

- 🧠+📷 Run the **Qwen 3 VL** reasoning vision models with full tool-calling support: [documentation](docs/QWEN3VL.md)  
  <details>
    <summary>Show command</summary>

    ```bash
    ./mistralrs-server -i --isq 8 run -m Qwen/Qwen3-VL-4B-Thinking
    ```
  </details>
  

- 🤗🤗🤗 Run the **SmolLM 3** long-context hybrid-reasoning model with full tool-calling support: [documentation](docs/SMOLLM3.md)  
  <details>
    <summary>Show command</summary>

    **Default, easiest:**
    ```bash
    ./mistralrs-server -i --isq 8 run -m HuggingFaceTB/SmolLM3-3B
    ```

    **UQFF prequantized:**
    ```bash
    ./mistralrs-server -i run -m EricB/SmolLM3-3B-UQFF -f smollm33b-q4k-0.uqff
    ```
  </details>

- 🔊 Run the **Dia 1.6b** model for highly-realistic dialogue generation: [documentation](docs/DIA.md)  
  <details>
    <summary>Show command</summary>

    ```bash
    ./mistralrs-server -i speech -m nari-labs/Dia-1.6B -a dia
    ```
  </details>

- 🦙 Run the **Llama 3.\* and Llama 4** models with long context & vision support: [docs (llama 3.2)](docs/VLLAMA.md), [docs (llama 4)](docs/LLAMA4.md)  
  <details>
    <summary>Show commands</summary>

    **Llama 4:**

    ```bash
    ./mistralrs-server -i --isq 4 run -m meta-llama/Llama-4-Scout-17B-16E-Instruct
    ```

    **Llama 3.1/3.2/3.3:**

    ```
    ./mistralrs-server -i --isq 8 run -m meta-llama/Llama-3.2-3B-Instruct
    ```

    **Llama 3.2 vision:**

    ```
    ./mistralrs-server -i --isq 8 run -m meta-llama/Llama-3.2-11B-Vision-Instruct
    ```

  </details>

- 💎💎💎 Run the **Gemma 3** family (1b, 4b, 12b, 27b) with 128k context & vision support: [documentation](docs/GEMMA3.md)  
  <details>
    <summary>Show command</summary>

    ```bash
    ./mistralrs-server -i --isq 8 run -m google/gemma-3-4b-it
    ```
  </details>

- 🌲📷 Run the **FLUX.1** diffusion model: [documentation](docs/FLUX.md)  
  <details>
    <summary>Show command</summary>

    ```bash
    ./mistralrs-server -i diffusion -m black-forest-labs/FLUX.1-schnell -a flux
    ```
  </details>

- 🧠 Run the **Qwen 3** hybrid-reasoning model with full tool-calling support: [documentation](docs/QWEN3.md)  
  <details>
    <summary>Show command</summary>

    ```bash
    ./mistralrs-server -i --isq 8 run -m Qwen/Qwen3-8B
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
    ./mistralrs-server --mcp-config mcp-config.json --port 1234 run -m Qwen/Qwen3-4B
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
./mistralrs-server --mcp-port 4321 plain -m Qwen/Qwen3-4B
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

> Note: You can use our [Docker containers here](https://github.com/EricLBuehler/mistral.rs/pkgs/container/mistral.rs).
> Learn more about running Docker containers: https://docs.docker.com/engine/reference/run/

- Install the [Python package here](mistralrs-pyo3/_README.md).
- The Python package has [wheels on PyPi](mistralrs-pyo3/_README.md#installation-from-pypi)!

1) Install required packages:
    - `OpenSSL` (*Example on Ubuntu:* `sudo apt install libssl-dev`)
    - <b>*Linux only:*</b> `pkg-config` (*Example on Ubuntu:* `sudo apt install pkg-config`)

2) Install Rust: https://rustup.rs/

    *Example on Ubuntu:*
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    ```

3) <b>*Optional:*</b> Set HF token correctly (skip if already set or your model is not gated, or if you want to use the `token_source` parameters in Python or the command line.)
    - Note: you can install `huggingface-cli` as documented [here](https://huggingface.co/docs/huggingface_hub/en/installation). 
    ```bash
    huggingface-cli login
    ```

4) Download the code:
    ```bash
    git clone https://github.com/EricLBuehler/mistral.rs.git
    cd mistral.rs
    ```

5) Build or install `mistralrs-server`:
    - Build the `mistralrs-server` binary, which can be found at `target/release/mistralrs-server`.
        ```bash
        cargo build --release --features <specify feature(s) here>
        ```
    - Install with `cargo install` for easy command line usage

        Pass the same values to `--features` as you would for `cargo build`
        ```bash
        cargo install --path mistralrs-server --features <specify feature(s) here>
        ```

6) (*If you used `cargo build`*) The build process will output a binary `mistralrs-server` at `./target/release/mistralrs-server`. We can switch to that directory so that the binary can be accessed as `./mistralrs-server` with the following command:

    *Example on Ubuntu:*
    ```
    cd target/release
    ```

7) Use our APIs and integrations: 
    
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

### Global CLI options

These options apply across all subcommands:

| Option | Description |
|--------|-------------|
| `-i, --interactive-mode` | Run in interactive terminal mode |
| `--port <PORT>` | Run HTTP server on specified port |
| `-c, --chat-template <FILE>` | Override chat template with JINJA file |
| `-j, --jinja-explicit <FILE>` | Explicit JINJA template (overrides all auto-detection) |
| `--token-source <SOURCE>` | HF token source: `cache`, `literal:<TOKEN>`, `env:<VAR>`, `path:<FILE>`, `none` |
| `--isq <TYPE>` | Apply ISQ quantization (e.g., `Q4K`, `Q8_0`) |
| `--num-device-layers <SPEC>` | Device layer mapping (e.g., `0:16;1:16` for multi-GPU) |
| `--enable-search` | Enable web search integration |
| `--search-embedding-model <MODEL>` | Built-in search embedding model (e.g., `embedding_gemma`) |
| `--mcp-config <FILE>` | Path to MCP client configuration JSON file |
| `--enable-thinking` | Enable thinking mode for supported models (Qwen3, SmolLM3) |
| `-s, --seed <SEED>` | Set random seed for reproducibility |
| `-l, --log <FILE>` | Log requests/responses to file |

### PagedAttention options

| Option | Description |
|--------|-------------|
| `--pa-gpu-mem <MB>` | GPU memory for KV cache in megabytes |
| `--pa-gpu-mem-usage <RATIO>` | Fraction of free GPU memory (0.0-1.0) |
| `--pa-ctxt-len <LENGTH>` | Total context length for KV cache |
| `--pa-blk-size <SIZE>` | Block size for PagedAttention |
| `--pa-cache-type <TYPE>` | KV cache type: `auto` or `f8e4m3` |
| `--paged-attn` | Enable PagedAttention (for Metal) |
| `--no-paged-attn` | Disable PagedAttention (for CUDA) |

### Interactive mode

**Llama 3.2 3B running on an M3 Max with 8-bit ISQ:**

<img src="./res/demo.gif" alt="Interactive demo" />

You can launch interactive mode, a simple chat application running in the terminal, by passing `-i`:

```bash
./mistralrs-server -i plain -m meta-llama/Llama-3.2-3B-Instruct
```

Vision models work seamlessly:

```bash
./mistralrs-server -i vision-plain -m lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k
```

Diffusion models can be run too (quantization and adapters are not yet supported):

```bash
./mistralrs-server -i diffusion -m black-forest-labs/FLUX.1-schnell -a flux
```

And you can run speech generation in your terminal!

```bash
./mistralrs-server -i speech -m nari-labs/Dia-1.6B -a dia
```

### OpenAI HTTP server

You can launch an HTTP server by replacing `-i` with `--port <port>`. For instance:

```bash
./mistralrs-server --port 1234 run -m microsoft/Phi-3.5-MoE-instruct
```

You can find documentation about the server itself [here](docs/HTTP.md).

### Multi-model support

Serve multiple models simultaneously from a single server instance. This is useful for comparing models, A/B testing, or serving different models for different use cases.

```bash
./mistralrs-server --port 1234 multi-model --config example-multi-model-config.json --default-model-id meta-llama/Llama-3.2-3B-Instruct
```

Select models in your requests using the `model` parameter:
```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-3B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```

📖 **[Complete multi-model documentation →](docs/multi_model/README.md)**

### Structured selection with a `.toml` file

We provide a method to select models with a `.toml` file. The keys are the same as the command line, with `no_kv_cache` and `tokenizer_json` being "global" keys.

Example:
```bash
./mistralrs-server --port 1234 toml -f toml-selectors/gguf.toml
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

### Using derivative and adapter models

To use a derivative or adapter model (e.g., quantized, LoRA, X-LoRA, vision, etc.), select the correct architecture subcommand and pass the required arguments—typically model id, and for quantized/adapters, also the quantization filename, tokenizer, or adapter ordering if needed.

- **See all options:** Run `./mistralrs-server <subcommand> --help`  
- **Docs:** [Adapter models](docs/ADAPTER_MODELS.md), [Chat templates](docs/CHAT_TOK.md)

<details>
<summary>Arguments by model type</summary>

| Model Type          | Required Arguments                                                     |
|---------------------|-----------------------------------------------------------------------|
| Plain               | model id                                                              |
| Quantized           | model id, quantized filename, tokenizer id                            |
| X-LoRA              | model id, X-LoRA ordering (if not default)                            |
| X-LoRA quantized    | model id, quantized filename, tokenizer id, X-LoRA ordering           |
| LoRA                | model id, LoRA ordering (if not default)                              |
| LoRA quantized      | model id, quantized filename, tokenizer id, LoRA ordering             |
| Vision Plain        | model id                                                              |

</details>

<details>
<summary>Example: Zephyr GGUF model</summary>

```bash
./mistralrs-server --port 1234 --log output.txt gguf -t HuggingFaceH4/zephyr-7b-beta -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q5_0.gguf
```
</details>

Chat template and tokenizer are usually auto-detected.  
If you need to override, see the [chat templates doc](docs/CHAT_TOK.md).

An adapter model is a model with X-LoRA or LoRA. X-LoRA support is provided by selecting the `x-lora-*` architecture, and LoRA support by selecting the `lora-*` architecture. Please find docs for adapter models [here](docs/ADAPTER_MODELS.md). Examples may be found [here](docs/LORA_XLORA.md).

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
