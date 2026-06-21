---
title: Reference
description: Exhaustive lookup pages for flags, endpoints, schemas, and APIs.
---

Lookup-oriented pages. For task-oriented documentation, start at the [Quickstart](/mistral.rs/quickstart/); for internals, the [Developer Guide](/mistral.rs/developer/).

- [CLI](/mistral.rs/reference/cli/): every subcommand and flag, generated from the clap definitions.
- [TOML configuration](/mistral.rs/reference/cli-toml-config/): the `mistralrs from-config` schema, with CLI flag mapping.
- [HTTP API](/mistral.rs/reference/http-api/): endpoints, request/response schemas, SSE events.
- [OpenAI compatibility](/mistral.rs/reference/openai-compatibility/): what is implemented, ignored, and extended relative to OpenAI's surface.
- [Python API](/mistral.rs/reference/python/): generated from the package's type stub.
- [Rust SDK reference](/mistral.rs/reference/rust-sdk/): the `Model` API surface; full rustdoc at [docs.rs/mistralrs](https://docs.rs/mistralrs).
- [MCP configuration schema](/mistral.rs/reference/mcp-config-schema/): the MCP client config file format.
- [Sandbox](/mistral.rs/reference/sandbox/): isolation layers and threat model for code execution.
- [Hardware support](/mistral.rs/reference/hardware-support/): supported GPUs, compute capabilities, and accelerators; which prebuilt binaries are published.
- [Supported models](/mistral.rs/reference/supported-models/): architectures, modalities, quantization compatibility, per-family notes.
- [Quantization types](/mistral.rs/reference/quantization-types/): bit widths, hardware constraints, quality.
- [UQFF format](/mistral.rs/reference/uqff-format/): on-disk layout of the UQFF quantization format.
- [Cargo features](/mistral.rs/reference/cargo-features/): build-from-source feature flags.
- [Environment variables](/mistral.rs/reference/environment-variables/): every env var the binary or build scripts read.
- [Troubleshooting](/mistral.rs/reference/troubleshooting/): symptom-to-cause index.
