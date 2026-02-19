![mistral.rs](banner.png)

## Quick Links

| I want to... | Go to... |
|--------------|----------|
| Install mistral.rs | [Installation Guide](INSTALLATION.md) |
| Understand cargo features | [Cargo Features](CARGO_FEATURES.md) |
| Run a model | [CLI Reference](CLI.md) |
| Use the HTTP API | [HTTP Server](HTTP.md) |
| Fix an error | [Troubleshooting](TROUBLESHOOTING.md) |
| Configure environment | [Configuration](CONFIGURATION.md) |
| Check model support | [Supported Models](SUPPORTED_MODELS.md) |

## Getting Started

- [Installation Guide](INSTALLATION.md) - Install mistral.rs on your system
- [Cargo Features](CARGO_FEATURES.md) - Complete cargo features reference
- [CLI Reference](CLI.md) - Complete CLI command reference
- [CLI TOML Configuration](CLI_CONFIG.md) - Configure via TOML files
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions

## SDKs & APIs

- [Python SDK](PYTHON_SDK.md) - Python package documentation
- [Python Installation](PYTHON_INSTALLATION.md) - Python SDK installation guide
- [Rust SDK](https://docs.rs/mistralrs/) - Rust crate documentation
- [HTTP Server](HTTP.md) - OpenAI-compatible HTTP API
- [OpenResponses API](OPENRESPONSES.md) - Stateful conversation API

## Models

### By Category
- [Supported Models](SUPPORTED_MODELS.md) - Complete model list and compatibility
- [Vision Models](VISION_MODELS.md) - Vision model overview
- [Image Generation](IMAGEGEN_MODELS.md) - Diffusion models
- [Embeddings](EMBEDDINGS.md) - Embedding model overview

### Model-Specific Guides

<details>
<summary>Click to expand model guides</summary>

**Text Models:**
- [DeepSeek V2](DEEPSEEKV2.md) | [DeepSeek V3](DEEPSEEKV3.md)
- [Gemma 2](GEMMA2.md) | [Gemma 3](GEMMA3.md) | [Gemma 3n](GEMMA3N.md)
- [GLM4](GLM4.md) | [GLM-4.7-Flash](GLM4_MOE_LITE.md) | [GLM-4.7](GLM4_MOE.md)
- [Qwen 3](QWEN3.md) | [SmolLM3](SMOLLM3.md) | [GPT-OSS](GPT_OSS.md)

**Vision Models:**
- [Idefics 2](IDEFICS2.md) | [Idefics 3](IDEFICS3.md)
- [LLaVA](LLaVA.md) | [Llama 3.2 Vision](VLLAMA.md) | [Llama 4](LLAMA4.md)
- [MiniCPM-O 2.6](MINICPMO_2_6.md) | [Mistral 3](MISTRAL3.md)
- [Phi 3.5 MoE](PHI3.5MOE.md) | [Phi 3.5 Vision](PHI3V.md) | [Phi 4 Multimodal](PHI4MM.md)
- [Qwen 2-VL](QWEN2VL.md) | [Qwen 3 VL](QWEN3VL.md)

**Other Models:**
- [FLUX (Diffusion)](FLUX.md) | [Dia (Speech)](DIA.md)
- [EmbeddingGemma](EMBEDDINGGEMMA.md) | [Qwen3 Embedding](QWEN3_EMBEDDING.md)

</details>

## Quantization & Optimization

- [Quantization Overview](QUANTS.md) - All supported quantization methods
- [ISQ (In-Situ Quantization)](ISQ.md) - Quantize models at load time
- [UQFF Format](UQFF.md) - Pre-quantized model format | [Layout](UQFF/LAYOUT.md)
- [Topology](TOPOLOGY.md) - Per-layer quantization and device mapping
- [Importance Matrix](IMATRIX.md) - Improve ISQ accuracy

## Adapters & Model Customization

- [Adapter Models](ADAPTER_MODELS.md) - LoRA and X-LoRA support
- [LoRA/X-LoRA Examples](LORA_XLORA.md)
- [Non-Granular Scalings](NON_GRANULAR.md) - X-LoRA optimization
- [AnyMoE](ANYMOE.md) - Create MoE models from dense models
- [MatFormer](MATFORMER.md) - Dynamic model sizing

## Performance & Hardware

- [Device Mapping](DEVICE_MAPPING.md) - Multi-GPU and CPU offloading
- [PagedAttention](PAGED_ATTENTION.md) - Efficient KV cache management
- [Speculative Decoding](SPECULATIVE_DECODING.md) - Accelerate generation with draft models
- [Flash Attention](FLASH_ATTENTION.md) - Accelerated attention
- [MLA](MLA.md) - Multi-head Latent Attention
- [Distributed Inference](DISTRIBUTED/DISTRIBUTED.md)
  - [NCCL Backend](DISTRIBUTED/NCCL.md)
  - [Ring Backend](DISTRIBUTED/RING.md)

## Features

- [Tool Calling](TOOL_CALLING.md) - Function calling support
- [Web Search](WEB_SEARCH.md) - Integrated web search
- [Chat Templates](CHAT_TOK.md) - Template customization
- [Sampling Options](SAMPLING.md) - Generation parameters
- [TOML Selector](TOML_SELECTOR.md) - Model selection syntax
- [Multi-Model Support](multi_model/overview.md) - Load multiple models

## MCP (Model Context Protocol)

- [MCP Client](MCP/client.md) - Connect to external tools
- [MCP Server](MCP/server.md) - Serve models over MCP
- [MCP Configuration](MCP/configuration.md)
- [MCP Transports](MCP/transports.md)
- [MCP Advanced Usage](MCP/advanced.md)

## Reference

- [Configuration](CONFIGURATION.md) - Environment variables and server defaults
- [Engine Internals](ENGINE.md) - Engine behaviors and recovery
- [Supported Models](SUPPORTED_MODELS.md) - Complete compatibility tables

---

## Contributing

See the main [README](https://github.com/EricLBuehler/mistral.rs/blob/master/README.md#contributing) for contribution guidelines.
