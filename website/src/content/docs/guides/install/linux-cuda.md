---
title: Install on Linux with CUDA
description: Get mistral.rs running on an NVIDIA GPU under Linux, including which features to enable for your card.
sidebar:
  order: 1
---

The install script handles most Linux-with-CUDA cases on its own. Run it, and if everything works, you can stop reading:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

The rest of this guide is for when the script's defaults are not what you want, when your CUDA toolchain is set up in an unusual way, or when you need to build with a specific combination of features.

## Prerequisites

Before building anything, make sure you have:

1. An NVIDIA driver recent enough for the CUDA version you want to use. `nvidia-smi` should report the GPU and show a driver version.
2. The CUDA toolkit installed and on your `PATH`. The `nvcc --version` command should print a version.
3. `libssl-dev` and `pkg-config` from your distribution's package manager. On Ubuntu or Debian, `sudo apt install libssl-dev pkg-config` is enough. On Fedora or RHEL, `sudo dnf install openssl-devel pkgconfig`.
4. A Rust toolchain at 1.88 or newer, installed through [rustup](https://rustup.rs).

If you want video input support, you also need FFmpeg (`sudo apt install ffmpeg` or equivalent). It is optional; mistral.rs builds and runs without it, but any video-related features will error at request time if it is not available.

## Feature selection

The install script picks features based on the first CUDA-capable GPU it sees, but there is a non-trivial amount of choice hiding inside "CUDA." The table below maps GPU generations to the features that benefit each one:

| Your GPU | Features to enable |
|---|---|
| H100, H200 | `cuda cudnn flash-attn flash-attn-v3` |
| A100, A40, Ampere consumer (30-series, L4, L40) | `cuda cudnn flash-attn` |
| Turing (RTX 20-series, T4), Volta (V100) | `cuda cudnn` |
| Pascal and older | `cuda` |

Flash attention needs compute capability 8.0 or newer. Flash attention v3 needs compute capability 9.0 (Hopper). cuDNN is optional everywhere but noticeably faster when available.

To install with a specific feature set from crates.io:

```bash
cargo install mistralrs-cli --features "cuda flash-attn cudnn"
```

Or from a source checkout:

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
cargo install --path mistralrs-cli --features "cuda flash-attn cudnn"
```

## Unusual CUDA layouts

If your CUDA toolkit is in a non-standard location, set `CUDA_ROOT` before building:

```bash
export CUDA_ROOT=/opt/cuda-12.4
cargo install mistralrs-cli --features "cuda flash-attn cudnn"
```

If you have multiple CUDA versions installed and need to pick one, `CUDA_ROOT` overrides whatever `which nvcc` would find. The build will use the `nvcc` that matches the `CUDA_ROOT` you set.

For machines where the CUDA libraries are in an unusual directory at runtime (common on HPC clusters with module loaders), `LD_LIBRARY_PATH` needs to include the lib directory:

```bash
export LD_LIBRARY_PATH="/opt/cuda-12.4/lib64:$LD_LIBRARY_PATH"
```

## Verifying the install

Once the binary is installed, `mistralrs doctor` will tell you whether the engine found the GPU correctly and which features were compiled in:

```bash
mistralrs doctor
```

The relevant lines look something like:

```
CUDA runtime: 12.4
GPU 0: NVIDIA A100-SXM4-40GB (compute capability 8.0)
Features compiled in: cuda, cudnn, flash-attn
```

If `cuda` is not listed, the binary was built without the CUDA feature. If the GPU is not listed, the driver or toolkit is not reaching the engine; double-check `nvidia-smi` and `nvcc --version` and rebuild with the right `CUDA_ROOT`.

## Troubleshooting

Most failures at build time are one of three things: the CUDA toolkit is not installed, `pkg-config` cannot find OpenSSL, or the Rust version is too old. Each of those produces a different error, and the [troubleshooting reference](/mistral.rs/reference/troubleshooting/) has specific fixes for each.
