---
title: Cargo features
description: Feature flags for the mistralrs workspace crates. What each one enables and when to turn it on.
sidebar:
  order: 11
---

mistral.rs uses Cargo features to gate platform-specific and optional functionality. This page lists every feature in the workspace, what it does, and which crates it applies to.

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

You can combine these. Typical combinations:

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

In a consumer crate that depends on `mistralrs`:

```toml
[dependencies]
mistralrs = { version = "0.8", features = ["cuda", "flash-attn", "cudnn"] }
```

## Default features

The top-level `mistralrs-cli` crate has `code-execution` in its default features. That means it is on unless you pass `--no-default-features`.

Other crates do not enable any accelerator features by default because picking the wrong one produces build errors that are hard to diagnose. You have to opt into the accelerator matching your hardware.

## Feature verification

After building, `mistralrs doctor` reports which features are compiled in:

```
Features compiled in: cuda, cudnn, flash-attn, code-execution
```

If a feature you expected is missing, the binary was built without it. Rebuild with the missing feature added.
