---
title: PagedAttention
description: Block-based KV caching in mistralrs.
sidebar:
  order: 5
---

mistral.rs's paged attention splits the KV cache into fixed-size blocks drawn from a central pool, instead of allocating one contiguous slab per sequence.

## Block size

The default is 32 tokens per block. Supported block sizes on CUDA are 8, 16, and 32, values outside that set fail to load because the attention kernel dispatches on block size explicitly. Override with `--pa-block-size`.

## Memory budget

The pool's size is set via one of three mutually exclusive flags:

- `--pa-memory-mb <n>`: budget in MB.
- `--pa-memory-fraction <f>`: fraction of available VRAM.
- `--pa-context-len <n>`: budget sized to hold the configured maximum concurrent sequences at that context length.

When none are set, the engine defaults to 90% of available VRAM on CUDA.

## Block lifecycle

Each sequence holds a list of block pointers. On each decoding step, the scheduler checks whether the sequence has a free slot in its tail block; if not, it allocates a new block from the pool. When a sequence finishes, its blocks return to the pool.

Shared-prefix optimization: sequences that begin with identical tokens share the blocks holding those tokens. A shared block is reference-counted rather than duplicated. This is the mechanism behind prefix caching when paged attention is on.

## Cache types

`--pa-cache-type` sets the KV cache's numeric representation:

- `auto` (default): match the model's compute dtype.
- Explicit quantization types reduce the per-block memory cost at some quality cost.

This is separate from `--isq`, which quantizes model weights. Weight and cache quantization are chosen independently.

## Composition with flash attention

The paged path gathers the sequence's blocks before dispatching to the attention kernel. On CUDA with flash-attn compiled in, the flash kernel is used through this gather path automatically.

## Default behavior

The engine enables paged attention by default for server workloads and disables it for single-request CLI use. Paged attention has a gather overhead versus a contiguous cache. Explicit control is via `--paged-attn auto|on|off`.

MLA's latent cache is supported through a dedicated kernel path.

## See also

- Guide: [use paged attention](/mistral.rs/guides/perf/use-paged-attention/).
- Reference: [CLI `--pa-*` flags](/mistral.rs/reference/cli/).
