---
title: Use paged attention
description: Turn on block-based KV caching for predictable memory use under concurrent load.
sidebar:
  order: 4
---

Standard attention allocates one contiguous KV cache per sequence, sized for the maximum context length. This is fast for one request but wastes memory when most contexts are short, and caps concurrency at roughly `(VRAM budget) / (max-context cache size)`.

Paged attention splits the KV cache into fixed-size blocks, allocates them on demand, and reuses freed blocks across requests. Effects:

- Many more concurrent requests at moderate context lengths on the same GPU.
- Predictable VRAM usage; the reserved budget is set up front.
- No cache penalty proportional to maximum context length for short prompts.

## When to turn it on

Use paged attention when:

- Serving more than a handful of concurrent requests.
- Predictable VRAM usage is required (resource accounting, GPU sharing).
- Running long-context models (32k+) where standard caches would be enormous.

Skip when:

- Single-request local generation. Overhead is small but so is the benefit.
- Metal with a small model. Unified memory makes the savings less important.

## Configuration

Paged attention is on by default when the engine detects it should help. Explicit control:

```bash
# Force on, with a specific block memory budget
mistralrs serve --paged-attn --paged-attn-gpu-mem 8192 -m <model>

# Force off
mistralrs serve --no-paged-attn -m <model>
```

`--paged-attn-gpu-mem` is the reserved KV cache memory in MB. Default is a fraction of available VRAM.

`--paged-attn-block-size` tunes block size. Smaller blocks: finer allocation, more bookkeeping. Defaults are 16 (CUDA) and 32 (Metal).

## Interactions

Composes cleanly with flash attention. Both on simultaneously; the engine routes block-sliced attention through flash kernels.

Has small effects on speculative decoding and exotic attention variants (notably MLA). If performance looks off in those cases, try `--no-paged-attn` as a sanity check.

## Memory sizing

`--paged-attn-gpu-mem` depends on:

- Total VRAM.
- Model weight size after quantization.
- Other GPU processes.
- Expected concurrency.

Rough formula:

```
--paged-attn-gpu-mem = (total VRAM) - (model weights) - (2 GB for activations and overhead)
```

For a 24 GB card running a 7B model at 4-bit (~4 GB), about 18 GB remains for KV cache.

## Further reading

The [explanation page](/mistral.rs/explanation/paged-attention/) covers the design and the block-based cache internals.
