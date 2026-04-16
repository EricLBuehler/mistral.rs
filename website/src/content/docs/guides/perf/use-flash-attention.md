---
title: Use flash attention
description: Enable flash attention and flash-attn v3 for faster attention kernels on NVIDIA GPUs.
sidebar:
  order: 3
---

Flash attention is a rewrite of the attention operation that fuses several memory-intensive steps into one kernel. The result is lower memory traffic and faster execution, particularly at long sequence lengths. On modern NVIDIA GPUs it is almost always a win.

mistral.rs supports two versions:

- **flash-attn** (v2) works on compute capability 8.0 and newer (Ampere and up).
- **flash-attn-v3** works on compute capability 9.0 (Hopper) only, and is faster still.

## Enabling it at build time

Flash attention is a Cargo feature, which means it has to be compiled into the binary. The install script does this automatically when it detects a supported GPU. If you are building from source:

```bash
# Ampere, Ada, older Hopper
cargo install --path mistralrs-cli --features "cuda flash-attn cudnn"

# Hopper (H100), for v3
cargo install --path mistralrs-cli --features "cuda flash-attn flash-attn-v3 cudnn"
```

If you built the binary without the feature, nothing at runtime will enable it. `mistralrs doctor` will show which features are compiled in:

```
Features compiled in: cuda, cudnn, flash-attn
```

## What you will see

With flash attention on, you get:

- Generation speed roughly 1.5x to 2x higher on Ampere, higher still on Hopper with v3.
- Noticeably lower VRAM usage during attention, because intermediate tensors no longer materialize. This lets you use longer contexts on the same card.
- No change to output. Flash attention is numerically equivalent to standard attention within floating-point rounding.

## When it is not the answer

A few situations where flash attention is not the lever to pull:

- **On CPU or Metal.** Flash attention is CUDA-only. Metal has its own efficient attention path and does not need it. CPU inference is attention-bound in a different way that flash does not address.
- **At very short sequence lengths.** The benefit of flash attention scales with sequence length. For short-prompt workloads where attention is a small fraction of total time, the difference is negligible.
- **On older NVIDIA hardware.** Compute capability 7.x and below do not support flash attention. The fallback is the standard attention path, which works correctly but is slower.

## Flash attention and paged attention

Flash attention and paged attention solve different problems and compose cleanly. You can use both together without doing anything special; the engine will route paged-attention blocks through flash kernels when both are enabled.

The [paged attention guide](/mistral.rs/guides/perf/use-paged-attention/) covers when to turn on paged attention specifically.

## Troubleshooting

If you built with flash attention and inference runs slower than expected, a few things to check:

- Run `mistralrs doctor`. Features and hardware capability both show up there.
- Check for a CUDA version mismatch. Flash attention uses PTX that targets specific CUDA capabilities; a very old driver might refuse to load the kernels.
- Look at the logs for a warning about falling back to the standard path. The engine logs when it cannot use the flash kernels for a given request.

If all else fails, the [troubleshooting reference](/mistral.rs/reference/troubleshooting/) has specific fixes for common flash attention issues.
