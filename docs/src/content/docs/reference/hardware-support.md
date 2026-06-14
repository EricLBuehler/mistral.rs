---
title: Hardware support
description: GPUs, compute capabilities, and accelerators mistral.rs supports, and which prebuilt binaries are published.
---

The authoritative list of supported accelerators. Other pages link here rather than restating it.

## Accelerators

| Platform | Acceleration | Prebuilt binary |
|---|---|---|
| Linux x86_64 + NVIDIA GPU | CUDA (Ampere and newer) | yes, per compute capability |
| Linux aarch64 + NVIDIA GPU | CUDA (Grace: GH200/GB200/GB10) | yes, sm90/100/121 |
| Apple Silicon (macOS arm64) | Metal | yes |
| Linux x86_64 / aarch64, no GPU | CPU | yes |
| Windows x86_64 | CPU | yes |
| Intel Mac, unlisted GPU | source build | no |

## NVIDIA compute capabilities

The minimum supported NVIDIA GPU is **Ampere (compute capability 8.0)**. Turing (sm75: RTX 20-series, GTX 16-series, Tesla T4) and older are not supported: candle's pre-Ampere CUDA path no longer builds against current toolkits. Such GPUs can still attempt a source build with an older CUDA toolkit, but this is untested and unsupported.

| Compute capability | Architecture | Representative GPUs |
|---|---|---|
| 8.0 | Ampere (datacenter) | A100, A30 |
| 8.6 | Ampere (consumer) | RTX 3090/3080/3070/3060, A40, A10 |
| 8.9 | Ada | RTX 4090/4080, L40, L4 |
| 9.0 | Hopper | H100, H200 |
| 10.0 | Blackwell (datacenter) | B200, GB200 |
| 12.0 | Blackwell (consumer) | RTX 5090/5080 |
| 12.1 | GB10 | DGX Spark |

## Prebuilt CUDA artifacts

Prebuilt CUDA artifacts are published per compute capability:

| Architecture | Compute capabilities |
|---|---|
| x86_64 | 80, 86, 89, 90, 100, 120 |
| aarch64 (NVIDIA Grace: GH200, GB200, GB10/DGX Spark) | 90, 100, 121 |

The [install script](/mistral.rs/quickstart/) downloads the binary matching your GPU and architecture; a GPU outside this set builds from source. The same binaries back the [Docker images](/mistral.rs/guides/deploy/docker/), and the same compute capabilities are published as Python wheels (install with `--find-links` - see [Python getting started](/mistral.rs/guides/python/getting-started/#installing)). Each is self-contained: bundled CUDA runtime libraries, no toolkit needed at runtime.

## Feature availability by architecture

| Feature | Requirement |
|---|---|
| `flash-attn` (v2) | compute capability 8.0+ |
| `flash-attn-v3` | Hopper (9.0) |
| FP8 matmul | compute capability 8.9+ |
| cuTile MoE backend | Ampere/Ada (8.x) or Blackwell+ (10.x/12.x), not Hopper; CUDA >= 13.1 |
| CUTLASS MoE backend | compute capability 8.0+ |

See [cargo features](/mistral.rs/reference/cargo-features/) for the feature flags and [MoE expert backends](/mistral.rs/developer/moe-backends/) for backend selection.
