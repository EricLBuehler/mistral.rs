---
title: Cargo features
description: Feature flags for the mistralrs workspace crates.
---

mistral.rs uses Cargo features to gate platform-specific and optional functionality.

## Accelerator features

| Feature | Crates | Purpose |
|---|---|---|
| `cuda` | `mistralrs-cli`, `mistralrs`, `mistralrs-core`, `mistralrs-server-core` | NVIDIA GPU support via CUDA, including CUDA [paged attention](/mistral.rs/guides/perf/paged-attention/) and FlashInfer (paged-attention kernel library) paged kernels. |
| `cudnn` | as above | cuDNN-accelerated kernels. |
| `flash-attn` | as above | Flash attention v2 (Ampere+, requires `cuda`). |
| `flash-attn-v3` | `mistralrs-cli`, `mistralrs-core`, `mistralrs-server-core` | Flash attention v3 (Hopper, requires `cuda`). Not exposed by the top-level `mistralrs` crate. |
| `cutile` | `mistralrs-cli`, `mistralrs-core` | cuTile JIT MoE (Mixture of Experts) kernels. Requires CUDA >= 13.2 on Ampere/Ada, CUDA >= 13.3 on Hopper, CUDA >= 13.1 on Blackwell+, and the `tileiras` assembler at runtime. Without it, MoE models fall back to the built-in CUTLASS (NVIDIA GEMM template library) kernels. See [MoE expert backends](/mistral.rs/developer/moe-backends/). Not exposed by the top-level `mistralrs` crate. |
| `metal` | as above | Apple Silicon GPU support via Metal. |
| `accelerate` | as above | Apple Accelerate framework for CPU math. |
| `mkl` | as above | Intel MKL for CPU math. |
| `nccl` | `mistralrs-cli`, `mistralrs`, `mistralrs-core`, `mistralrs-server-core` | NCCL single-machine CUDA multi-GPU support. Requires the NCCL runtime library at build and runtime. |

Typical combinations:

- NVIDIA Hopper: `cuda flash-attn flash-attn-v3 cudnn` (add `cutile` with CUDA >= 13.3)
- NVIDIA Ampere or Ada: `cuda flash-attn cudnn` (add `cutile` with CUDA >= 13.2)
- NVIDIA Blackwell with CUDA >= 13.1: `cuda flash-attn cudnn cutile`
- NVIDIA older: `cuda cudnn`
- Apple Silicon: `metal`
- Intel CPU with MKL: `mkl`

For Linux CUDA multi-GPU, add `nccl` when NCCL is installed. The Linux installer and CUDA wheel builder add it automatically when they detect `libnccl`.

## Functional features

| Feature | Crates | Purpose |
|---|---|---|
| `audio` | `mistralrs`, `mistralrs-core` | Audio input decoding and speech-model support. Enabled by default for compatibility. |
| `mcp` | `mistralrs`, `mistralrs-core` | Model Context Protocol client integration. Enabled by default for compatibility. |
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

### Minimal in-process builds

The `mistralrs-core` and top-level `mistralrs` crates expose default-on `audio` and `mcp`
features. The top-level crate forwards both features to `mistralrs-core`. Disable defaults
when an in-process application does not need either subsystem:

```toml
[dependencies]
mistralrs = { version = "0.9", default-features = false, features = ["metal"] }
```

The same pattern works for a direct core dependency:

```toml
[dependencies]
mistralrs-core = { version = "0.9", default-features = false, features = ["metal"] }
```

Enable one subsystem without the other when needed:

```toml
[dependencies]
mistralrs = { version = "0.9", default-features = false, features = ["metal", "mcp"] }
```

With `audio` disabled, the runtime dependency graph excludes `mistralrs-audio`, `symphonia`,
and `hound`. Phi-4 Multimodal and Gemma 3n remain available for text and image inputs;
attempting an audio path reports that the `audio` feature is required. Some DSP crates used
by the models' non-audio internals remain unconditional dependencies.

With `mcp` disabled, the graph excludes `mistralrs-mcp` and `rust-mcp-schema`. Generic tool
calling and code-execution types remain available through a small shared types crate; only MCP
client configuration and connectivity APIs are removed from the public surface.

To inspect a minimal dependency graph:

```bash
cargo tree -p mistralrs-core --no-default-features -e normal
cargo check -p mistralrs-core --no-default-features
```

## Default features

`mistralrs-core` and `mistralrs` enable `audio` and `mcp` by default. `mistralrs-cli`'s default
feature is `code-execution`. `mistralrs-server-core`'s default feature is `swagger-ui`. To exclude
a crate's defaults, use `--no-default-features`.

No crate enables an accelerator feature by default. Opt in to the accelerator matching your hardware.

## Feature verification

`mistralrs doctor` prints a `Build features:` line listing the compiled-in accelerator features (`cuda`, `metal`, `cudnn`, `flash-attn`, `flash-attn-v3`, `accelerate`, `mkl`). Other features such as `cutile`, `nccl`, `ring`, `code-execution`, and `swagger-ui` are not shown on that line.
