---
title: Install on macOS with Metal
description: Get mistral.rs running on Apple Silicon. The install script handles the common case; this guide covers the rest.
sidebar:
  order: 2
---

On Apple Silicon (M1, M2, M3, M4), the install script detects the chip and builds with Metal support:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

For manual builds, follow the steps below.

## Prerequisites

- macOS 13 (Ventura) or newer. Earlier Metal Performance Shaders versions lack required operations.
- Xcode Command Line Tools. Install with `xcode-select --install`.
- Rust 1.88 or newer via [rustup](https://rustup.rs).
- Homebrew with FFmpeg for video input: `brew install ffmpeg`. See [Set up video input](/mistral.rs/guides/models/video-setup/) for the full checklist.

A full Xcode install is not required. The command-line tools include the Metal Shading Language compiler and required headers.

## Feature selection

The standard combination on Apple Silicon is `metal accelerate`. Metal is the GPU backend; Accelerate provides BLAS/LAPACK on the CPU.

From crates.io:

```bash
cargo install mistralrs-cli --features "metal accelerate"
```

From source:

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
cargo install --path mistralrs-cli --features "metal accelerate"
```

For Intel Macs, omit Metal and use `accelerate`, or use `mkl` if Intel's MKL is installed.

## Memory and unified memory notes

Apple Silicon uses unified memory; the GPU and CPU share physical RAM. Implications:

- No separate VRAM budget. A model that fits in RAM fits on the GPU.
- `mistralrs doctor` reports total system memory rather than separate GPU memory.
- Default paged attention block sizes are tuned for dedicated VRAM. See the [paged attention guide](/mistral.rs/guides/perf/use-paged-attention/) for tuning on unified memory.

Total RAM caps model size. A 32 GB machine fits models up to ~20B parameters at 4-bit with moderate context. A 64 GB machine fits 70B-class models at 4-bit.

## AFQ quantization

On Metal, `--isq` defaults to AFQ (Adaptive Float Quantization) formats. AFQ is tuned for Apple's GPU. `--isq 4` and similar flags select AFQ automatically.

Existing GGUF files load and run directly. AFQ performance benefits require re-quantization with ISQ.

## Verifying the install

```bash
mistralrs doctor
```

The output includes a `[INFO] Build features: ...` line listing `metal` (and `accelerate` if compiled in). If `metal` is missing, rebuild with `--features "metal accelerate"`.
