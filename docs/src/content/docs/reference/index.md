---
title: Reference
description: Exhaustive lookup pages for flags, endpoints, schemas, and APIs.
---

Lookup-oriented pages. For task-oriented documentation, start at the [Quickstart](/quickstart/); for internals, the [Developer Guide](/developer/).

- [CLI](/reference/cli/): every subcommand and flag, generated from the clap definitions.
- [TOML configuration](/reference/cli-toml-config/): the `mistralrs from-config` schema, with CLI flag mapping.
- [HTTP API](/reference/http-api/): endpoints, request/response schemas, SSE events.
- [OpenAI compatibility](/reference/openai-compatibility/): what is implemented, ignored, and extended relative to OpenAI's surface.
- [Python API](/reference/python/): generated from the package's type stub.
- [Rust SDK reference](/reference/rust-sdk/): the `Model` API surface; full rustdoc at [docs.rs/mistralrs](https://docs.rs/mistralrs).
- [MCP configuration schema](/reference/mcp-config-schema/): the MCP client config file format.
- [Sandbox](/reference/sandbox/): isolation layers and threat model for code execution.
- [Hardware support](/reference/hardware-support/): supported GPUs, compute capabilities, and accelerators; which prebuilt binaries are published.
- [Supported models](/reference/supported-models/): architectures, modalities, quantization compatibility, per-family notes.
- [Quantization types](/reference/quantization-types/): bit widths, hardware constraints, quality.
- [UQFF format](/reference/uqff-format/): on-disk layout of the UQFF quantization format.
- [Cargo features](/reference/cargo-features/): build-from-source feature flags.
- [Environment variables](/reference/environment-variables/): every env var the binary or build scripts read.
- [Troubleshooting](/reference/troubleshooting/): symptom-to-cause index.
