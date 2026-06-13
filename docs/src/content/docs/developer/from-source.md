---
title: Build from source
description: Compile mistral.rs from a repository checkout with exactly the feature flags you want.
---

:::tip[Most users do not need this]
The [quickstart install script](/mistral.rs/quickstart/) is the recommended way to install. It downloads a prebuilt binary for your platform; `MISTRALRS_INSTALL_TAG=<tag>` installs a specific release, and `MISTRALRS_INSTALL_FROM_SOURCE=1` makes it build the latest `master` from source for you. Build from a checkout manually only to pin an arbitrary commit, apply a local patch, or use a feature combination the published binaries do not include.
:::

## Platform prerequisites

A source build needs Rust 1.88+ ([rustup](https://rustup.rs)) plus, per platform:

- **Linux CUDA:** an NVIDIA driver matching the target CUDA version (`nvidia-smi`), the CUDA toolkit on `PATH` (`nvcc --version`), and `libssl-dev` + `pkg-config` (`sudo apt install libssl-dev pkg-config`, or `sudo dnf install openssl-devel pkgconfig`). For a non-standard toolkit location set `CUDA_ROOT` (e.g. `export CUDA_ROOT=/opt/cuda-12.4`); for runtime libraries in non-standard directories (common on HPC module systems) add them to `LD_LIBRARY_PATH`.
- **macOS:** the Xcode Command Line Tools (`xcode-select --install`); a full Xcode install is not required. Apple Silicon uses `--features metal`; Intel Macs use `--features accelerate` (or `mkl` if Intel MKL is installed).
- **Windows:** Visual Studio 2022 Build Tools (the `rustup-init.exe` installer can add these) and, for CUDA, the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads). If `cargo build` fails on long paths, enable long-path support: `Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1 -Type DWord` (the registry setting, not `git config core.longpaths`). Native Windows lacks the ring-backend and some experimental features; for full parity use **WSL2** (`wsl --install -d Ubuntu`, verify `nvidia-smi` inside it, then follow the Linux steps).

## Clone and build

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
```

The CLI binary is in the `mistralrs-cli` crate:

```bash
# Release build in-place
cargo build --release --features "cuda nccl flash-attn cudnn" -p mistralrs-cli

# Or install globally from the checkout
cargo install --path mistralrs-cli --features "cuda nccl flash-attn cudnn"
```

The flags above target CUDA. On macOS use `--features metal`; on CPU omit `--features` entirely. See the [cargo features reference](/mistral.rs/reference/cargo-features/) for the full list.

The in-place build leaves the binary at `target/release/mistralrs`. The install variant copies it to `~/.cargo/bin/mistralrs`, which is on `PATH` after a rustup install.

## Feature flag combinations

Common per-platform flag strings:

| Platform | `--features` |
|---|---|
| CPU | (none) |
| macOS / Metal | `metal` |
| CUDA | `cuda flash-attn cudnn` |
| CUDA multi-GPU | `cuda flash-attn cudnn nccl` |

The full flag list, per-hardware recommendations, and per-flag effects live in the [cargo features reference](/mistral.rs/reference/cargo-features/). Add `nccl` on Linux when NCCL is installed and you want CUDA multi-GPU tensor parallelism.

## Developing against a local checkout

Use `cargo build --release -p mistralrs-cli` for incremental development.

Some tests are gated behind feature flags. Core test suite:

```bash
cargo test -p mistralrs-core -p mistralrs-quant -p mistralrs-vision
```

In the quantization crate, some tests run only with a specific backend feature enabled.

## Python wheels

Building the Python SDK from source requires `maturin`:

```bash
pip install maturin[patchelf]
cd mistralrs-pyo3
maturin develop --release --features "cuda nccl flash-attn cudnn"
```

This installs the package into the current Python environment. `maturin build` produces a redistributable wheel under `target/wheels/`.

## Version pinning in a consumer crate

To depend on the workspace directly (e.g., to use an unreleased change), add a git dependency:

```toml
[dependencies]
mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs", branch = "master" }
```

For production, pin to a release tag or a specific commit SHA for reproducible builds.
