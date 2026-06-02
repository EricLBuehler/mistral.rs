---
title: Cargo features
description: Feature flags for the mistralrs workspace crates.
sidebar:
  order: 11
---

mistral.rs uses Cargo features to gate platform-specific and optional functionality.

## Accelerator features

| Feature | Crates | Purpose |
|---|---|---|
| `cuda` | `mistralrs-cli`, `mistralrs`, `mistralrs-core`, `mistralrs-server-core` | NVIDIA GPU support via CUDA, including CUDA PagedAttention and FlashInfer paged kernels. |
| `cudnn` | as above | cuDNN-accelerated kernels. |
| `flash-attn` | as above | Flash attention v2 (Ampere+, requires `cuda`). |
| `flash-attn-v3` | `mistralrs-cli`, `mistralrs-core`, `mistralrs-server-core` | Flash attention v3 (Hopper, requires `cuda`). Not exposed by the top-level `mistralrs` crate. |
| `metal` | as above | Apple Silicon GPU support via Metal. |
| `accelerate` | as above | Apple Accelerate framework for CPU math. |
| `mkl` | as above | Intel MKL for CPU math. |
| `nccl` | `mistralrs-cli`, `mistralrs`, `mistralrs-core`, `mistralrs-server-core` | NCCL single-machine CUDA multi-GPU support. Requires the NCCL runtime library at build and runtime. |

Typical combinations:

- NVIDIA Hopper: `cuda flash-attn flash-attn-v3 cudnn`
- NVIDIA Ampere or Ada: `cuda flash-attn cudnn`
- NVIDIA older: `cuda cudnn`
- Apple Silicon: `metal accelerate`
- Intel CPU with MKL: `mkl`

For Linux CUDA multi-GPU, add `nccl` when NCCL is installed. The Linux installer and CUDA wheel builder add it automatically when they detect `libnccl`.

## Functional features

| Feature | Crates | Purpose |
|---|---|---|
| `code-execution` | `mistralrs-cli`, `mistralrs`, `mistralrs-core`, `mistralrs-server-core` | Python code execution tool. In `mistralrs-cli` defaults. |
| `ring` | as above | Multi-machine ring distributed inference. |
| `swagger-ui` | `mistralrs-server-core` | Mounts Swagger UI on the HTTP server. |

## Enabling features

From `cargo install`:

```bash
cargo install mistralrs-cli --features "cuda nccl flash-attn cudnn"
```

From a source checkout:

```bash
cargo install --path mistralrs-cli --features "cuda nccl flash-attn cudnn"
```

In a consumer crate depending on `mistralrs`:

```toml
[dependencies]
mistralrs = { version = "0.8", features = ["cuda", "nccl", "flash-attn", "cudnn"] }
```

## Default features

`mistralrs-cli`'s default feature is `code-execution`. To exclude it, use `--no-default-features`.

Other crates enable no accelerator features by default. Opt in to the accelerator matching your hardware.

## Feature verification

`mistralrs doctor` prints a `Build features:` line listing compiled-in features.
