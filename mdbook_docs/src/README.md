# Documentation

This mdBook gathers all of the platform and feature documentation for `mistral.rs`. Model-specific guides (DeepSeek, Gemma, Qwen, etc.) intentionally stay in the legacy `docs/` directory so they can evolve independently of the shared reference material.

## Getting Started

- [HTTP Server](HTTP.md) — CLI flags, OpenAI-compatible endpoints, and usage tips.
- [Web Search](WEB_SEARCH.md) — augment completions with retrieval plug-ins.
- [Sampling](SAMPLING.md) — nucleus, temperature, beam search, grammar, and more.

## Core Inference Building Blocks

- [Chat templates & tokenizers](CHAT_TOK.md)
- [Device mapping automation](DEVICE_MAPPING.md)
- [Topology files for per-layer ISQ/device settings](TOPOLOGY.md)
- [Multi-model serving](multi_model/README.md)
- [Distributed inference (NCCL/Ring)](DISTRIBUTED/DISTRIBUTED.md)

## Modalities & APIs

- [Vision models](VISION_MODELS.md)
- [Image generation models](IMAGEGEN_MODELS.md)
- [Speech/DIA pipeline](DIA.md)
- [Tool calling](TOOL_CALLING.md)
- [TOML selector](TOML_SELECTOR.md)
- [MCP client/server guides](MCP/README.md)

## Adapters, Quantization & Performance

- [Adapter models](ADAPTER_MODELS.md), [LoRA/X-LoRA](LORA_XLORA.md), and [non-granular layouts](NON_GRANULAR.md)
- [AnyMoE](ANYMOE.md) mixture of experts support
- [Quantization overview](QUANTS.md), [ISQ](ISQ.md), [IMatrix](IMATRIX.md), and [UQFF](UQFF.md)
- [Paged Attention](PAGED_ATTENTION.md) and [Flash Attention](FLASH_ATTENTION.md)
- [Matformer runtime slicing](MATFORMER.md)

## Embeddings

- [Embeddings overview](EMBEDDINGS.md) — inference tips plus Python/Rust samples.

> **Tip:** Each section of the book mirrors the structure in `SUMMARY.md`, so you can navigate via the sidebar or the links above. Model-specific feature notes continue to live in `docs/` (and can be linked to directly from this book when needed).
