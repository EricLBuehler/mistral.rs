---
title: Cargo features
description: Feature flags for the mistralrs workspace crates. What each one enables and when to turn it on.
sidebar:
  order: 11
---

mistral.rs uses Cargo features to gate platform-specific and optional functionality. This page lists every workspace feature, its purpose, and the crates it applies to.

## Accelerator features

| Feature | Crate(s) | Purpose |
|---|---|---|
| `cuda` | `mistralrs-cli`, `mistralrs`, `mistralrs-core`, `mistralrs-quant` | NVIDIA GPU support via CUDA. |
| `flash-attn` | as above | Flash attention v2 on NVIDIA (Ampere+). |
| `flash-attn-v3` | as above | Flash attention v3 on NVIDIA Hopper. |
| `cudnn` | as above | cuDNN-accelerated kernels. |
| `metal` | as above | Apple Silicon GPU support via Metal. |
| `accelerate` | as above | Apple Accelerate framework for CPU math. |
| `mkl` | as above | Intel MKL for CPU math. |

Combinable. Typical combinations:

- NVIDIA Hopper: `cuda flash-attn flash-attn-v3 cudnn`
- NVIDIA Ampere or Ada: `cuda flash-attn cudnn`
- NVIDIA older: `cuda cudnn`
- Apple Silicon: `metal accelerate`
- Intel CPU with MKL: `mkl`
- Generic CPU: no features (SIMD enabled by default)

## Functional features

| Feature | Crate(s) | Purpose |
|---|---|---|
| `code-execution` | `mistralrs-cli`, `mistralrs` | Enables the `--enable-code-execution` flag and the corresponding Rust API. Enabled by default. |
| `ring` | `mistralrs-cli`, `mistralrs` | Enables the multi-machine ring backend for distributed inference. Linux only. |
| `swagger-ui` | `mistralrs-server-core` | Mounts a Swagger UI at `/docs` with the OpenAPI spec. |

## Development-only features

| Feature | Crate | Purpose |
|---|---|---|
| `utoipa` | several | Generates OpenAPI schemas for types. Used when building the docs; not needed at runtime. |
| `bench` | `mistralrs-bench` | Legacy benchmark crate. Prefer `mistralrs bench` subcommand. |

## How to enable features

From `cargo install`:

```bash
cargo install mistralrs-cli --features "cuda flash-attn cudnn"
```

From a source checkout:

```bash
cargo install --path mistralrs-cli --features "cuda flash-attn cudnn"
```

In a consumer crate depending on `mistralrs`:

```toml
[dependencies]
mistralrs = { version = "0.8", features = ["cuda", "flash-attn", "cudnn"] }
```

## Default features

`mistralrs-cli` has `code-execution` in its default features — on unless `--no-default-features` is passed.

Other crates enable no accelerator features by default. Wrong accelerator selection produces hard-to-diagnose build errors. Opt in to the accelerator matching your hardware.

## Feature verification

`mistralrs doctor` reports compiled-in features after build:

```
Features compiled in: cuda, cudnn, flash-attn, code-execution
```

A missing expected feature means the binary was built without it. Rebuild with the feature added.
