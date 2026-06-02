---
title: Guides
description: Task-oriented recipes for things you want to do.
---

Guides answer "how do I..." questions. They assume mistral.rs is installed. Otherwise, start with the [Tutorials](/mistral.rs/tutorials/).

## Choose by task

| If you need to... | Start here |
|---|---|
| Install for a specific platform or deployment target | [Install and deploy](/mistral.rs/guides/install/) |
| Run an HTTP server, compatibility API, or web UI | [Serve models](/mistral.rs/guides/serve/) |
| Reduce memory use or improve throughput | [Performance](/mistral.rs/guides/perf/) |
| Add tools, search, code execution, or MCP | [Build agents](/mistral.rs/guides/agents/) |
| Use the Python package | [Python SDK](/mistral.rs/guides/python/) |
| Use the Rust crate | [Rust SDK](/mistral.rs/guides/rust/) |
| Work with vision, speech, image generation, or embeddings | [Model types](/mistral.rs/guides/models/) |
| Change model behavior or load adapters | [Customize](/mistral.rs/guides/customize/) |

## Guide sections

- [Install and deploy](/mistral.rs/guides/install/): platform-specific install steps, Docker images, and pre-production checks.
- [Serve models](/mistral.rs/guides/serve/): HTTP server configuration, multi-model serving, the web UI, OpenAI-compatible APIs, and the Anthropic Messages API.
- [Performance](/mistral.rs/guides/perf/): quantization selection, the `tune` command, Flash and Paged attention, and multi-GPU or multi-machine splits.
- [Build agents](/mistral.rs/guides/agents/): tool calling, code execution, web search, MCP, and persistent sessions.
- [Python SDK](/mistral.rs/guides/python/): streaming completions, image and video input, and the multi-turn session API.
- [Rust SDK](/mistral.rs/guides/rust/): streaming and embedding mistral.rs in an Axum application.
- [Model types](/mistral.rs/guides/models/): vision input, image generation, speech, and embedding models.
- [Customize](/mistral.rs/guides/customize/): LoRA adapters, AnyMoE, MatFormer, sampling parameters, and TOML config.
