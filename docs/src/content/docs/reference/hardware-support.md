---
title: Hardware support
description: GPUs, compute capabilities, and accelerators mistral.rs supports, and which prebuilt binaries are published.
---

Use this page to check whether a prebuilt binary exists for your accelerator. If you are installing on a normal workstation or server, start with the [quickstart](/quickstart/) and let the installer choose.

## Accelerators

| Platform | Acceleration | Prebuilt binary |
|---|---|---|
| Linux x86_64 + NVIDIA GPU | CUDA (Ampere and newer) | yes, see [CUDA artifacts](#cuda-artifacts) |
| Linux aarch64 + NVIDIA GPU | CUDA (Grace: GH200/GB200/GB10) | yes, see [CUDA artifacts](#cuda-artifacts) |
| Apple Silicon (macOS arm64) | Metal | yes |
| Linux x86_64 / aarch64, no GPU | CPU | yes |
| Windows x86_64, CPU selected or no NVIDIA GPU | CPU | yes |
| Windows with an NVIDIA GPU | CUDA | no; the installer selects a source build |
| Intel Mac, unlisted GPU | source build | no |

The [guided installer](/reference/installer/) shows the selected backend and installation method before starting. Windows CUDA source builds require the development tools listed in [platform prerequisites](/developer/from-source/#platform-prerequisites); select CPU for the Windows x86_64 prebuilt.

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

Each CUDA binary is built with one CUDA toolkit version and for one GPU compute capability, and its name says both:

```text
mistralrs-cuda128-sm90-aarch64-unknown-linux-gnu.tar.gz
```

Here `cuda128` means it was built with CUDA 12.8, and `sm90` means it targets compute capability 9.0. Your NVIDIA driver must support at least that CUDA version: check the `CUDA Version` that `nvidia-smi` reports. On Linux, the installer picks the newest build your driver supports that's published for your GPU.

| CUDA version in name | Built with | x86_64 compute capabilities | aarch64 (Grace) compute capabilities | cuTile |
|---|---|---|---|---|
| `cuda133` | CUDA 13.3.0 | 90 | 90 | yes |
| `cuda132` | CUDA 13.2.0 | 80, 86, 89, 100, 120 | 100, 121 | yes |
| `cuda131` | CUDA 13.1.2 | 80, 86, 89, 90, 100, 120 | 90, 100, 121 | no |
| `cuda130` | CUDA 13.0.0 | 80, 86, 89, 90, 100, 120 | 90, 100, 121 | no |
| `cuda129` | CUDA 12.9.1 | none | 121 | no |
| `cuda128` | CUDA 12.8.1 | 80, 86, 89, 90, 100, 120 | 90, 100 | no |

`cuda129` exists for GB10 because CUDA 12.8 can't target compute capability 12.1.

Each binary bundles the CUDA runtime libraries it needs, so you don't need a CUDA toolkit installed to run it. cuTile builds don't include NVIDIA's `tileiras` tool; install it separately by following [cuTile setup](/developer/moe-backends/).

The [Docker images](/guides/deploy/docker/) and CUDA Python wheels use the same CUDA versions. See [Python getting started](/guides/python/getting-started/#installing) for wheel install commands.

## Feature availability by architecture

| Feature | Requirement |
|---|---|
| `flash-attn` (v2) | compute capability 8.0+ |
| `flash-attn-v3` | Hopper (9.0) |
| FP8 matmul | compute capability 8.9+ |
| cuTile acceleration | Ampere/Ada (8.x) and Blackwell+ (10.x/12.x) with CUDA >= 13.2; Hopper (9.0) with CUDA >= 13.3 |

See [cargo features](/reference/cargo-features/) for the feature flags and [cuTile setup](/developer/moe-backends/) for installation.
