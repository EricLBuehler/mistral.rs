---
title: Reference
description: Exhaustive lookup pages for flags, endpoints, schemas, and APIs.
---

Reference pages are short, complete, and not particularly fun to read. They exist so you can find the one flag or method or field you need and get out. There is no narrative, no motivation, and no handholding. If you want any of those things, the [Guides](/mistral.rs/guides/) or [Explanation](/mistral.rs/explanation/) sections are where they live.

## What is here

**CLI.** Every subcommand and flag of the `mistralrs` binary: `run`, `serve`, `bench`, `tune`, `login`, `from-config`, and the rest.

**TOML configuration.** The schema for the config file that `mistralrs from-config` reads.

**HTTP API.** Endpoint-by-endpoint documentation for the server, including request and response schemas.

**OpenAI compatibility.** A table of which parts of OpenAI's Chat Completions and Responses surface we implement, and which we intentionally do not.

**Python API.** The public surface of the `mistralrs` Python package: `Runner`, `Which`, request and response types.

**Rust API.** The stable surface of the `mistralrs` crate, with pointers to docs.rs for the auto-generated rustdoc.

**MCP configuration schema.** The JSON schema for MCP client configuration files.

**Supported models.** A table of every architecture we support, the modalities each one accepts, and the quantization methods that work with it.

**Model notes.** A short FAQ for the handful of models with genuinely surprising behavior. Everything not listed here behaves like its peers.

**Cargo features.** The feature flags you need when building from source.

**UQFF format.** The on-disk binary layout for the UQFF quantization format, for people writing tools that consume it.

**Quantization types.** A table of bit counts, hardware requirements, and relative quality for each supported method.

**Troubleshooting.** A symptom-to-cause index for common errors and what to do about them.
