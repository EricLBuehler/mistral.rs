---
title: Build from source
description: Compile mistral.rs from a repository checkout with exactly the feature flags you want.
---

This guide is for working from a local checkout, such as making changes to mistral.rs or choosing custom build features. For installation, start with the [quickstart](/quickstart/).

## Platform prerequisites

A source build needs Rust 1.95+ ([rustup](https://rustup.rs)) plus, per platform:

- **Linux CUDA:**
  - An NVIDIA driver that supports the target CUDA version (check with `nvidia-smi`).
  - The CUDA toolkit on `PATH` (check with `nvcc --version`). For a toolkit in a non-standard location, set `CUDA_ROOT`, for example `export CUDA_ROOT=/opt/cuda-12.4`.
  - `libssl-dev` and `pkg-config`: `sudo apt install libssl-dev pkg-config`, or `sudo dnf install openssl-devel pkgconfig`.
  - If CUDA runtime libraries live in non-standard directories, as is common with HPC module systems, add them to `LD_LIBRARY_PATH`.
- **macOS:** the Xcode Command Line Tools (`xcode-select --install`). Full Xcode isn't needed. Apple Silicon builds use `--features metal`. Intel Macs use `--features accelerate`, or `mkl` if Intel MKL is installed.
- **Windows:**
  - Visual Studio 2022 Build Tools. The `rustup-init.exe` installer can add them.
  - For CUDA, the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).
  - If `cargo build` fails on long paths, turn on long-path support in the registry (`git config core.longpaths` doesn't help): `Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1 -Type DWord`.
  - Native Windows builds lack the ring backend and some experimental features. For full parity, use **WSL2**: run `wsl --install -d Ubuntu`, check `nvidia-smi` inside it, then follow the Linux steps.

## Clone and build

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
```

The CLI binary is in the `mistralrs-cli` crate:

```bash
# Release build in-place
cargo build --release --locked --features "cuda nccl flash-attn cudnn" -p mistralrs-cli

# Or install globally from the checkout
cargo install --path mistralrs-cli --locked --features "cuda nccl flash-attn cudnn"
```

The flags above are for CUDA. Use `--features metal` on macOS, and leave out `--features` for a CPU build. See the [cargo features reference](/reference/cargo-features/) for the full list.

CUDA 13.4 is supported. When several toolkits are installed, point the CUDA headers, libraries, `nvcc`, and cuTile at the same one before building:

```bash
export CUDA_HOME=/usr/local/cuda-13.4
export CUDA_ROOT="$CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
export CUDA_TOOLKIT_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The driver must support the selected toolkit's PTX version. If you use [cuTile](/developer/moe-backends/), keep these exports in the environment that runs mistral.rs too.

By default, Metal builds precompile the Metal shader libraries (metallibs) for macOS, iOS, and tvOS, including the GDN and Mamba/SSM kernels. Two build-time variables change this:

- `MISTRALRS_METAL_PLATFORMS=macos` builds only the macOS libraries, which is faster for local development.
- `MISTRALRS_METAL_PRECOMPILE=0` skips precompilation. The shaders are then compiled at runtime, when the library loads.

Separately, when a model loads on Metal, mistral.rs compiles the paged-attention and quantized-linear kernels for that model's shapes and dtype before it accepts requests (`mistralrs-core/src/pipeline/metal_warmup.rs`). This doesn't run a prompt or touch the KV cache. A few request-specific kernels still compile on first use.

The in-place build leaves the binary at `target/release/mistralrs`. The install variant copies it to `~/.cargo/bin/mistralrs`, which is on `PATH` after a rustup install.

## Feature flag combinations

Common per-platform flag strings:

| Platform | `--features` |
|---|---|
| CPU | (none) |
| macOS / Metal | `metal` |
| CUDA | `cuda flash-attn cudnn` |
| CUDA multi-GPU | `cuda flash-attn cudnn nccl` |

The full flag list, per-hardware recommendations, and per-flag effects live in the [cargo features reference](/reference/cargo-features/). Add `nccl` on Linux when NCCL is installed and you want CUDA multi-GPU tensor parallelism.

## Developing against a local checkout

Use `cargo build --release -p mistralrs-cli` for incremental development.

Run the core test suite with:

```bash
cargo test -p mistralrs-core -p mistralrs-quant -p mistralrs-vision
```

Some `mistralrs-quant` tests compile only with a backend feature, and are skipped otherwise:

| Tests | Feature |
|---|---|
| `grouped_mmq_*_cuda_tests.rs`, `indexed_moe_lora_cuda_tests.rs`, `mmq_sum_range_cuda_tests.rs`, `routed_lora_cuda_tests.rs` in `mistralrs-quant/tests/`, plus some CUDA unit tests | `cuda` |
| `cutlass_moe_tests.rs` | `cuda`, built for compute capability 8.0 or newer, not with MSVC |
| `cutile_moe_tests.rs`, `cutile_routed_lora_tests.rs`, `nvfp4_cuda_tests.rs` | `cuda` and `cutile` |
| Metal unit tests | `metal` |

For example, `cargo test -p mistralrs-quant --features cuda,cutile` runs the cuTile tests.

## Python wheels

Building the Python SDK from source requires `maturin`:

```bash
pip install maturin[patchelf]
cd mistralrs-pyo3
maturin develop --release --features "cuda nccl flash-attn cudnn pyo3/extension-module"
```

This installs the package into the current Python environment. `maturin build` produces a redistributable wheel under `target/wheels/`.

## Version pinning in a consumer crate

To depend on the workspace directly (e.g., to use an unreleased change), add a git dependency:

```toml
[dependencies]
mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs", branch = "master" }
```

For production, pin to a release tag or a specific commit SHA for reproducible builds.
