---
title: Use paged attention
description: Block-based KV caching for higher concurrency.
sidebar:
  order: 4
---

Standard attention allocates one contiguous KV cache per sequence, sized for the maximum context length. Paged attention splits the cache into fixed-size blocks and allocates them on demand from a central pool.

## Enabling paged attention

Use paged attention when:

- Serving more than a handful of concurrent requests.
- Predictable VRAM usage is required.
- Running long-context models (32k+) where standard caches would be enormous.

## Configuration

Paged attention is on by default when the engine detects it should help. Explicit control:

```bash
# Force on, with a specific KV cache memory budget in MB
mistralrs serve --paged-attn on --pa-memory-mb 8192 -m <model>

# Force off
mistralrs serve --paged-attn off -m <model>
```

`--paged-attn` accepts `auto` (default), `on`, or `off`.

Memory budget options (mutually exclusive with `--pa-context-len`):

- `--pa-memory-mb <mb>`: KV cache budget in MB.
- `--pa-memory-fraction <f>`: KV cache budget as a fraction of VRAM.
- `--pa-context-len <n>`: allocate KV cache sized for this context length.

`--pa-block-size <n>` tunes block size (default 32 on CUDA). `--pa-cache-type` controls KV cache quantization.

## Composition

Paged attention composes with flash attention. Both can be on simultaneously.

## Further reading

The [explanation page](/mistral.rs/explanation/paged-attention/) covers the design.
