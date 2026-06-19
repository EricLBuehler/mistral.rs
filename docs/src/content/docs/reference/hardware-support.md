---
title: Hardware support
description: GPUs, compute capabilities, and accelerators mistral.rs supports, and which prebuilt binaries are published.
---

Use this page to check whether a prebuilt binary exists for your accelerator. If you are installing on a normal workstation or server, start with the [quickstart](/mistral.rs/quickstart/) and let the installer choose.

## Accelerators

| Platform | Acceleration | Prebuilt binary |
|---|---|---|
| Linux x86_64 + NVIDIA GPU | CUDA (Ampere and newer) | yes, per compute capability and driver CUDA support |
| Linux aarch64 + NVIDIA GPU | CUDA (Grace: GH200/GB200/GB10) | yes, sm90/100/121, per driver CUDA support |
| Apple Silicon (macOS arm64) | Metal | yes |
| Linux x86_64 / aarch64, no GPU | CPU | yes |
| Windows x86_64 | CPU | yes |
| Intel Mac, unlisted GPU | source build | no |

## NVIDIA compute capabilities

The minimum supported NVIDIA GPU is **Ampere (compute capability 8.0)**. Turing (`sm75`: RTX 20-series, GTX 16-series, Tesla T4) and older are not supported by current prebuilts.

| Compute capability | Architecture | Representative GPUs |
|---|---|---|
| 8.0 | Ampere (datacenter) | A100, A30 |
| 8.6 | Ampere (consumer) | RTX 3090/3080/3070/3060, A40, A10 |
| 8.9 | Ada | RTX 4090/4080, L40, L4 |
| 9.0 | Hopper | H100, H200 |
| 10.0 | Blackwell (datacenter) | B200, GB200 |
| 12.0 | Blackwell (consumer) | RTX 5090/5080 |
| 12.1 | GB10 | DGX Spark |

## CUDA artifacts

CUDA artifact names encode both the toolkit lane and compute capability:

```text
mistralrs-cuda128-sm90-aarch64-unknown-linux-gnu.tar.gz
```

The installer chooses the newest published lane that the installed NVIDIA driver can load.

| Driver reports | Artifact lane | Notes |
|---|---|---|
| CUDA 13.1+ | `cuda131` | latest lane; includes cuTile where supported |
| CUDA 12.9 or 13.0 on GB10 / `sm121` | `cuda129` | needed because CUDA 12.8 does not target `sm121` |
| CUDA 12.8 to 13.0 on other GPUs | `cuda128` | baseline lane for Ampere and newer |

| Architecture | Compute capabilities |
|---|---|
| x86_64 | 80, 86, 89, 90, 100, 120 |
| aarch64 (NVIDIA Grace: GH200, GB200, GB10/DGX Spark) | 90, 100, 121 |

| Asset token | Built with | Minimum `nvidia-smi` CUDA version | Published compute capabilities | cuTile |
|---|---|---|---|---|
| `cuda128` | CUDA 12.8.1 | 12.8 | x86_64: 80, 86, 89, 90, 100, 120; aarch64: 90, 100 | no |
| `cuda129` | CUDA 12.9.1 | 12.9 | aarch64: 121 | no |
| `cuda131` | CUDA 13.1.2 | 13.1 | x86_64: 80, 86, 89, 90, 100, 120; aarch64: 90, 100, 121 | yes, where the GPU architecture supports it |

Each artifact bundles the CUDA runtime libraries it needs, so no CUDA toolkit is required at runtime. The installed NVIDIA driver still has to be new enough for that artifact's toolkit lane.

The same compatibility lanes are used by the [Docker images](/mistral.rs/guides/deploy/docker/) and CUDA Python wheels. See [Python getting started](/mistral.rs/guides/python/getting-started/#installing) for wheel install commands.

## Feature availability by architecture

| Feature | Requirement |
|---|---|
| `flash-attn` (v2) | compute capability 8.0+ |
| `flash-attn-v3` | Hopper (9.0) |
| FP8 matmul | compute capability 8.9+ |
| cuTile MoE backend | Ampere/Ada (8.x) or Blackwell+ (10.x/12.x), not Hopper; CUDA >= 13.1 |
| CUTLASS MoE backend | compute capability 8.0+ |

See [cargo features](/mistral.rs/reference/cargo-features/) for the feature flags and [MoE expert backends](/mistral.rs/developer/moe-backends/) for backend selection.
