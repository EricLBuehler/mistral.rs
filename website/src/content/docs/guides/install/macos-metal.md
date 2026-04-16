---
title: Install on macOS with Metal
description: Get mistral.rs running on Apple Silicon. The install script handles the common case; this guide covers the rest.
sidebar:
  order: 2
---

For Apple Silicon (M1, M2, M3, M4), the install script will detect your chip and build mistral.rs with Metal support automatically:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

That is almost always what you want. The rest of this guide covers the cases where you need something specific.

## What you need installed

- macOS 13 (Ventura) or newer. Metal Performance Shaders on older versions lack some of the operations we use.
- The Xcode Command Line Tools. If you do not have them yet, `xcode-select --install` prompts you to install.
- Rust 1.88 or newer from [rustup](https://rustup.rs).
- Homebrew if you want FFmpeg for video input: `brew install ffmpeg`.

You do not need a full Xcode install. The command-line tools include the Metal Shading Language compiler and the headers the build needs.

## Feature selection

On Apple Silicon, the combination you usually want is `metal accelerate`. Metal is the GPU backend; Accelerate is Apple's BLAS/LAPACK library, which we use for the parts of the pipeline that run on the CPU.

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

If you have an Intel Mac (which we support but do not optimize for), skip the Metal feature and use `accelerate` on its own, or use `mkl` if you have Intel's MKL library installed.

## Memory and unified memory notes

Apple Silicon uses unified memory, so the GPU sees the same physical RAM the CPU does. That matters for a few reasons:

- There is no separate VRAM budget to worry about. A model that fits in your RAM will fit on the GPU.
- `mistralrs doctor` reports total system memory rather than separate GPU memory, because that number is what the engine actually has to work with.
- Default paged attention block sizes are tuned for GPU configurations with dedicated VRAM. On unified memory you can often use smaller block sizes without hurting throughput. The [paged attention guide](/mistral.rs/guides/perf/use-paged-attention/) covers this.

For larger models, RAM is the real constraint. A 32 GB machine can run most models up to around 20 billion parameters at 4-bit quantization with a reasonable context length. A 64 GB machine comfortably handles 70B-class models at 4-bit. Anything bigger than that usually needs the larger M-series chips (M3 Ultra, M4 Ultra) or a different platform.

## AFQ quantization

On Metal, the `--isq` flag picks AFQ (Adaptive Float Quantization) formats by default. These are designed specifically for Apple's GPU architecture and are meaningfully faster than the generic Q*K formats you would use on CUDA. You do not need to opt into this; `--isq 4` and friends just do the right thing.

If you want to use a GGUF file you already have, that works too. The engine will load it and run the quantized weights directly, you just will not get the AFQ performance benefit unless you re-quantize with ISQ.

## Verifying the install

```bash
mistralrs doctor
```

The output should include something like:

```
Metal device: Apple M2 Pro
Features compiled in: metal, accelerate
```

If Metal is not listed, the binary was built without the feature. Rebuild with `--features "metal accelerate"`.
