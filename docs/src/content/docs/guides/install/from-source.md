---
title: Build from source
description: Compile mistral.rs from a repository checkout with exactly the feature flags you want.
sidebar:
  order: 4
---

Build from a source checkout to pin a specific commit, apply a local patch, or use a feature combination not in the published binaries. For most installs, the install script or `cargo install mistralrs-cli` is sufficient.

## Clone and build

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
```

The CLI binary is in the `mistralrs-cli` crate:

```bash
# Release build in-place
cargo build --release --features "cuda flash-attn cudnn" -p mistralrs-cli

# Or install globally from the checkout
cargo install --path mistralrs-cli --features "cuda flash-attn cudnn"
```

The in-place build leaves the binary at `target/release/mistralrs`. The install variant copies it to `~/.cargo/bin/mistralrs`, which is on `PATH` after a rustup install.

## Feature flag combinations

Per-hardware recommendations:

- NVIDIA Hopper (H100): `cuda flash-attn flash-attn-v3 cudnn`
- NVIDIA Ampere or Ada (A100, L40, 30/40-series): `cuda flash-attn cudnn`
- NVIDIA older: `cuda cudnn`
- Apple Silicon: `metal accelerate`
- Intel CPU with MKL: `mkl`
- Generic CPU: no features (SIMD on by default)

Full list and per-flag effects: [cargo features reference](/mistral.rs/reference/cargo-features/).

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
maturin develop --release --features "cuda flash-attn cudnn"
```

This installs the package into the current Python environment. `maturin build` produces a redistributable wheel under `target/wheels/`.

## Version pinning in a consumer crate

To depend on the workspace directly (e.g., to use an unreleased change), add a git dependency:

```toml
[dependencies]
mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs", branch = "master" }
```

For production, pin to a release tag or a specific commit SHA for reproducible builds.
