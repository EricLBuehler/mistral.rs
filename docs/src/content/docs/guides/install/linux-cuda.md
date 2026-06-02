---
title: Install on Linux with CUDA
description: Get mistral.rs running on an NVIDIA GPU under Linux, including which features to enable for your card.
sidebar:
  order: 1
---

The install script handles most Linux-with-CUDA cases:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

The rest of this guide covers non-default cases: unusual CUDA toolchain layouts, manual feature selection, and verifying the install.

## Prerequisites

1. An NVIDIA driver compatible with the target CUDA version. `nvidia-smi` should report the GPU and driver version.
2. The CUDA toolkit on `PATH`. `nvcc --version` should print a version.
3. `libssl-dev` and `pkg-config`. On Ubuntu/Debian: `sudo apt install libssl-dev pkg-config`. On Fedora/RHEL: `sudo dnf install openssl-devel pkgconfig`.
4. Rust 1.88 or newer via [rustup](https://rustup.rs).
5. NCCL for CUDA multi-GPU tensor parallelism. This is optional for single-GPU use. The installer enables `nccl` when it can find `libnccl`.

For video input, install FFmpeg (`sudo apt install ffmpeg` or equivalent). It is optional; without it, video features error at request time. The full checklist is in [Set up video input](/mistral.rs/guides/models/video-setup/).

## Feature selection

The install script picks features from the first CUDA-capable GPU detected. Manual mapping:

| GPU | Features |
|---|---|
| H100, H200 | `cuda cudnn flash-attn flash-attn-v3` |
| A100, A40, Ampere consumer (30-series, L4, L40) | `cuda cudnn flash-attn` |
| Turing (RTX 20-series, T4), Volta (V100) | `cuda cudnn` |
| Pascal and older | `cuda` |

Flash attention requires compute capability 8.0+. Flash attention v3 requires 9.0 (Hopper). cuDNN is optional but faster when available. Add `nccl` on Linux when NCCL is installed and you want CUDA multi-GPU tensor parallelism.

Install with a specific feature set from crates.io:

```bash
cargo install mistralrs-cli --features "cuda nccl flash-attn cudnn"
```

From a source checkout:

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
cargo install --path mistralrs-cli --features "cuda nccl flash-attn cudnn"
```

## Unusual CUDA layouts

For a non-standard toolkit location, set `CUDA_ROOT` before building:

```bash
export CUDA_ROOT=/opt/cuda-12.4
cargo install mistralrs-cli --features "cuda nccl flash-attn cudnn"
```

`CUDA_ROOT` overrides the `nvcc` discovered via `PATH`. The build uses the `nvcc` matching the `CUDA_ROOT` value.

For runtime CUDA or NCCL libraries in non-standard directories (common on HPC clusters with module loaders), add the lib directory to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="/opt/cuda-12.4/lib64:$LD_LIBRARY_PATH"
```

## Verifying the install

`mistralrs doctor` reports GPU detection and compiled features:

```bash
mistralrs doctor
```

The output includes lines like:

```
[INFO] CUDA: nvcc 12.4, driver 12.4
[INFO] CUDA[0]: 40.0 GB total, 35.2 GB free - Compute 8.0 (FA v2: ✅, v3: ❌)
[INFO] Build features: cuda, nccl, cudnn, flash-attn
```

If `cuda` is missing from build features, rebuild with the feature. If `nccl` is missing on a multi-GPU Linux host, install NCCL and rebuild, or run without NCCL using layer mapping. If no CUDA device is reported, verify `nvidia-smi` and `nvcc --version`, then rebuild with the correct `CUDA_ROOT`.

## Troubleshooting

Most build-time failures are one of: missing CUDA toolkit, `pkg-config` cannot find OpenSSL, or outdated Rust. Specific fixes are in the [troubleshooting reference](/mistral.rs/reference/troubleshooting/).
