---
title: Use flash attention
description: Enable flash attention and flash-attn v3 for faster attention kernels on NVIDIA GPUs.
sidebar:
  order: 3
---

Flash attention is a fused attention kernel that reduces memory traffic and runs faster, particularly at long sequence lengths. On modern NVIDIA GPUs it is almost always a win.

mistral.rs supports two versions:

- **flash-attn** (v2) — compute capability 8.0+ (Ampere and newer).
- **flash-attn-v3** — compute capability 9.0 (Hopper) only. Faster than v2.

## Enabling it at build time

Flash attention is a Cargo feature and must be compiled in. The install script enables it when a supported GPU is detected. From source:

```bash
# Ampere, Ada, older Hopper
cargo install --path mistralrs-cli --features "cuda flash-attn cudnn"

# Hopper (H100), for v3
cargo install --path mistralrs-cli --features "cuda flash-attn flash-attn-v3 cudnn"
```

A binary built without the feature cannot enable it at runtime. `mistralrs doctor` lists compiled features:

```
Features compiled in: cuda, cudnn, flash-attn
```

## Effects

With flash attention enabled:

- Generation speed roughly 1.5×–2× higher on Ampere; higher on Hopper with v3.
- Lower VRAM usage during attention because intermediates do not materialize. Enables longer contexts on the same card.
- Numerically equivalent to standard attention within floating-point rounding.

## When it does not help

- **CPU or Metal.** Flash attention is CUDA-only. Metal has its own efficient attention path. CPU is bound differently.
- **Very short sequences.** Benefit scales with sequence length. Negligible for short prompts.
- **Older NVIDIA hardware.** Compute capability 7.x and below fall back to standard attention.

## Composition with paged attention

Flash and paged attention compose. Both can be on simultaneously; the engine routes paged-attention blocks through flash kernels.

See the [paged attention guide](/mistral.rs/guides/perf/use-paged-attention/).

## Troubleshooting

If flash attention is built in but performance is below expectations:

- Run `mistralrs doctor` to verify features and hardware capability.
- Check for CUDA version mismatch. Old drivers may refuse newer PTX.
- Inspect logs for fallback warnings — the engine logs when it cannot use flash kernels for a request.

For specific issues, see the [troubleshooting reference](/mistral.rs/reference/troubleshooting/).
