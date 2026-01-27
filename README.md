<a name="top"></a>
<h1 align="center">
  mistral.rs
</h1>

<h3 align="center">
Blazingly fast LLM inference.
</h3>

<p align="center">
| <a href="https://ericlbuehler.github.io/mistral.rs/mistralrs/"><b>Rust Docs</b></a> | <a href="https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/API.md"><b>Python Docs</b></a> | <a href="https://discord.gg/SZrecqK8qw"><b>Discord</b></a> | <a href="https://matrix.to/#/#mistral.rs:matrix.org"><b>Matrix</b></a> |
</p>

<p align="center">
  <a href="https://github.com/EricLBuehler/mistral.rs/stargazers">
    <img src="https://img.shields.io/github/stars/EricLBuehler/mistral.rs?style=social&label=Star" alt="GitHub stars">
  </a>
</p>

**Mistral.rs is a cross-platform, highly-multimodal inference engine that brings you:**
- All-in-one multimodal workflow: text↔text, text+vision↔text, text+vision+audio↔text, text→speech, text→image, text→embeddings
- APIs: Rust, Python, OpenAI HTTP server (with Chat Completions, [Responses API compatible with OpenResponses](docs/OPENRESPONSES.md)), MCP server
- MCP Client: Connect to external tools and services automatically (file systems, web search, databases, APIs)
- Performance: ISQ, PagedAttention, FlashAttention.
- Support for embedding, speech generation, and image generation models

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

Then visit [http://localhost:1234/ui](http://localhost:1234/ui) for the web chat interface.

<details open>
  <summary><b>Web Chat Demo</b></summary>
  <br>
  <img src="./res/chat.gif" alt="Web Chat UI Demo" />
</details>

## Capabilities

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

[Request a new model](https://github.com/EricLBuehler/mistral.rs/issues/156) | [Full compatibility tables](docs/SUPPORTED_MODELS.md)

## Python

```bash
pip install mistralrs-cuda  # or mistralrs-metal, mistralrs-mkl, mistralrs
```

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="Q4K",
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

[Full Python documentation](mistralrs-pyo3/API.md) | [Examples](examples/python) | [Cookbook](examples/python/cookbook.ipynb)

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

For complete documentation, see the **[Documentation Index](docs/README.md)**.

**Quick Links:**
- [CLI Reference](docs/CLI.md) - All commands and options
- [HTTP API](docs/HTTP.md) - OpenAI-compatible endpoints
- [Quantization](docs/QUANTS.md) - ISQ, GGUF, GPTQ, and more
- [Device Mapping](docs/DEVICE_MAPPING.md) - Multi-GPU and CPU offloading
- [MCP Integration](docs/MCP/README.md) - External tool connections
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Contributing

Contributions welcome! Please [open an issue](https://github.com/EricLBuehler/mistral.rs/issues) to discuss new features or report bugs. If you want to add a new model, please contact us via an issue and we can coordinate.

## Credits

This project would not be possible without the excellent work at [Candle](https://github.com/huggingface/candle). Thank you to all [contributors](https://github.com/EricLBuehler/mistral.rs/graphs/contributors)!

<p align="right">
  <a href="#top">Back to Top</a>
</p>
