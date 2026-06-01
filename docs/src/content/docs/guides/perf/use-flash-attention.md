---
title: Use flash attention
description: Enable flash attention kernels on NVIDIA GPUs.
sidebar:
  order: 3
---

Flash attention is a fused attention kernel that reduces memory traffic. mistral.rs supports two versions:

- **flash-attn** (v2): compute capability 8.0+ (Ampere and newer).
- **flash-attn-v3**: compute capability 9.0 (Hopper) only.

## Enabling at build time

Flash attention is a Cargo feature. The install script enables it when a supported GPU is detected. From source:

```bash
# Ampere, Ada, older Hopper
cargo install --path mistralrs-cli --features "cuda flash-attn cudnn"

# Hopper (H100), for v3
cargo install --path mistralrs-cli --features "cuda flash-attn flash-attn-v3 cudnn"
```

`mistralrs doctor` lists compiled features.

## Composition with paged attention

Flash and paged attention compose. Both can be on simultaneously, but they are not the same backend:

- `flash-attn` and `flash-attn-v3` are Cargo features for the standard attention path and fallback varlen paths.
- FlashInfer paged decode and prefill kernels are built with the `cuda` feature as part of PagedAttention.

On CUDA with PagedAttention enabled, mistral.rs uses the FlashInfer paged layout and decode kernel for compatible KV caches by default. Set `MISTRALRS_FLASHINFER_DECODE=0` only when debugging or comparing against the generic paged path.

See the [paged attention guide](/mistral.rs/guides/perf/use-paged-attention/).
