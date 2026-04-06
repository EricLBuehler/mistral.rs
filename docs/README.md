![mistral.rs](banner.png)

**Fast, flexible LLM inference.** Run 40+ model families (text, vision, video, audio, speech, image generation, and embeddings) with automatic hardware optimization, OpenAI-compatible APIs, and built-in agentic features.

## Quick Start

```bash
# Install (Linux/macOS, auto-detects CUDA, Metal, MKL)
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh

# Install (Windows PowerShell)
irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex

# Run a model interactively
mistralrs run --isq 4 -m Qwen/Qwen3-4B

# Or serve it as an OpenAI-compatible API with a web UI
mistralrs serve --ui --isq 4 -m Qwen/Qwen3-4B
```

> **New to mistral.rs?** Follow the [Getting Started tutorial](GETTING_STARTED.md) for a guided walkthrough.

## Choose Your Path

| I want to... | Start here |
|---|---|
| Follow a step-by-step tutorial | [Getting Started](GETTING_STARTED.md) |
| Run a model from the command line | [CLI Reference](CLI.md) |
| Serve models over HTTP | [HTTP Server](HTTP.md) |
| Build a Python application | [Python SDK](PYTHON_SDK.md) |
| Build a Rust application | [Rust SDK](RUST_SDK.md) |
| Optimize performance and memory | [Performance Guide](PERFORMANCE.md) |
| Choose a quantization method | [Quantization Overview](QUANTS.md) |
| Use agentic features (tools, search, MCP) | [Agentic Features Guide](AGENTS.md) |
| Find a specific model | [Supported Models](SUPPORTED_MODELS.md) |
| Troubleshoot an issue | [Troubleshooting](TROUBLESHOOTING.md) |

## Highlights

- **Zero config**: `mistralrs run -m <model>` auto-detects architecture, chat template, and hardware
- **Quantization control**: ISQ, GGUF, GPTQ, AWQ, HQQ, AFQ, FP8, MXFP4, or just `--isq 4`
- **Auto-tuning**: `mistralrs tune` benchmarks your hardware and recommends optimal settings
- **Agentic**: Server-side tool calling loop, web search, MCP client, HTTP tool dispatch
- **Multi-GPU**: Automatic tensor parallelism via NCCL, or cross-machine via Ring backend
- **Web UI**: `--ui` flag for instant browser-based chat

## Community

- [Discord](https://discord.gg/SZrecqK8qw)
- [GitHub Issues](https://github.com/EricLBuehler/mistral.rs/issues)
- [Contributing](https://github.com/EricLBuehler/mistral.rs/blob/master/README.md#contributing)
