---
title: Cargo features
description: Feature flags for the mistralrs workspace crates.
---

mistral.rs uses Cargo features to gate platform-specific and optional functionality.

## Accelerator features

| Feature | Crates | Purpose |
|---|---|---|
| `cuda` | `mistralrs-cli`, `mistralrs`, `mistralrs-core`, `mistralrs-server-core` | NVIDIA CUDA acceleration, including [paged attention](/mistral.rs/guides/perf/paged-attention/). |
| `cudnn` | as above | cuDNN-accelerated kernels. |
| `flash-attn` | as above | Flash attention v2 (Ampere+, requires `cuda`). |
| `flash-attn-v3` | `mistralrs-cli`, `mistralrs-core`, `mistralrs-server-core` | Flash attention v3 (Hopper, requires `cuda`). Not exposed by the top-level `mistralrs` crate. |
| `cutile` | `mistralrs-cli`, `mistralrs-core` | Optional cuTile acceleration for MoE and routed LoRA. Requires CUDA >= 13.2 on Ampere/Ada and Blackwell+, CUDA >= 13.3 on Hopper, and a compatible `tileiras` installation. See [cuTile setup](/mistral.rs/developer/moe-backends/). Not exposed by the top-level `mistralrs` crate. |
| `metal` | as above | Apple Silicon GPU support via Metal. |
| `accelerate` | as above | Apple Accelerate framework for CPU math. |
| `mkl` | as above | Intel MKL for CPU math. |
| `nccl` | `mistralrs-cli`, `mistralrs`, `mistralrs-core`, `mistralrs-server-core` | NCCL single-machine CUDA multi-GPU support. Requires the NCCL runtime library at build and runtime. |

Typical combinations:

- NVIDIA Hopper: `cuda flash-attn flash-attn-v3 cudnn` (add `cutile` with CUDA >= 13.3)
- NVIDIA Ampere or Ada: `cuda flash-attn cudnn` (add `cutile` with CUDA >= 13.2)
- NVIDIA Blackwell with CUDA >= 13.2 and a compatible `tileiras`: `cuda flash-attn cudnn cutile`
- NVIDIA older: `cuda cudnn`
- Apple Silicon: `metal`
- Intel CPU with MKL: `mkl`

For Linux CUDA multi-GPU, add `nccl` when NCCL is installed. The Linux installer and CUDA wheel builder add it automatically when they detect `libnccl`.

## Functional features

| Feature | Crates | Purpose |
|---|---|---|
| `code-execution` | `mistralrs-cli`, `mistralrs`, `mistralrs-core`, `mistralrs-server-core` | Python code execution tool. In `mistralrs-cli` defaults. |
| `ring` | as above | Multi-machine ring distributed inference. |
| `swagger-ui` | `mistralrs-server-core` | Mounts Swagger UI on the HTTP server. On by default in `mistralrs-server-core`. |

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

`mistralrs-cli`'s default feature is `code-execution`. `mistralrs-server-core`'s default feature is `swagger-ui`. To exclude defaults, use `--no-default-features`.

No crate enables an accelerator feature by default. Opt in to the accelerator matching your hardware.

## Feature verification

`mistralrs doctor` prints a `Build features:` line listing the compiled-in accelerator features (`cuda`, `metal`, `cudnn`, `flash-attn`, `flash-attn-v3`, `cutile`, `accelerate`, `mkl`). Other features such as `nccl`, `ring`, `code-execution`, and `swagger-ui` are not shown on that line.
