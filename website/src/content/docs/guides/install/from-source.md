---
title: Build from source
description: Compile mistral.rs from a repository checkout with exactly the feature flags you want.
sidebar:
  order: 4
---

The install script and `cargo install mistralrs-cli` are the right tools for most cases. Building from a source checkout makes sense when you want to pin a specific commit, apply a local patch, or use a feature combination that is not one of the published binaries.

## Clone and build

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
```

The workspace has several crates and binaries. The one you want is `mistralrs-cli`:

```bash
# Release build in-place
cargo build --release --features "cuda flash-attn cudnn" -p mistralrs-cli

# Or install globally from the checkout
cargo install --path mistralrs-cli --features "cuda flash-attn cudnn"
```

The in-place build leaves the binary at `target/release/mistralrs`. The install variant copies it into `~/.cargo/bin/mistralrs`, which is already on your `PATH` if you installed Rust through rustup.

## Feature flag combinations

The right features depend on your hardware. The short version:

- NVIDIA, Hopper (H100): `cuda flash-attn flash-attn-v3 cudnn`
- NVIDIA, Ampere or Ada (A100, L40, 30-series, 40-series): `cuda flash-attn cudnn`
- NVIDIA, older: `cuda cudnn`
- Apple Silicon: `metal accelerate`
- Intel CPU with MKL: `mkl`
- Generic CPU: no features, SIMD is enabled by default

The exhaustive list, including what each flag changes, is in the [cargo features reference](/mistral.rs/reference/cargo-features/).

## Developing against a local checkout

If you are hacking on mistral.rs itself, the `cargo build --release -p mistralrs-cli` form is what you want, because it keeps the workspace's incremental compilation cache intact between rebuilds. Full rebuilds of the core crate take a couple of minutes on a modern laptop; incremental rebuilds after a small edit typically finish in under ten seconds.

Some tests are gated behind feature flags. Running the full test suite looks like:

```bash
cargo test -p mistralrs-core -p mistralrs-quant -p mistralrs-vision
```

For the quantization crate in particular, some tests only run when you build with a specific backend feature, so the numbers you see without features enabled will not be the full coverage.

## Python wheels

Building the Python SDK from source needs `maturin`:

```bash
pip install maturin[patchelf]
cd mistralrs-pyo3
maturin develop --release --features "cuda flash-attn cudnn"
```

This installs the package into your current Python environment. If you want a wheel file you can redistribute, `maturin build` produces one under `target/wheels/`.

## Version pinning in a consumer crate

If you are using the Rust SDK in your own project and want to pull from the workspace directly (to use an unreleased change, say), point your Cargo.toml at a git dependency:

```toml
[dependencies]
mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs", branch = "master" }
```

This is useful while you are iterating on something that depends on a pending engine change, but for production use you should pin either to a specific release or to a specific commit SHA so your builds are reproducible.
