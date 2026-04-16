---
title: Use paged attention
description: Turn on block-based KV caching for predictable memory use under concurrent load.
sidebar:
  order: 4
---

Standard attention allocates one contiguous KV cache per sequence, sized for the maximum context length. That is simple and fast for a single request, but it wastes memory when most requests have short contexts, and it caps concurrency at roughly `(VRAM budget) / (max-context cache size)`.

Paged attention fixes both problems. It splits the KV cache into fixed-size blocks, allocates them on demand as each sequence grows, and reuses freed blocks between requests. In practice this means:

- The same GPU can serve many more concurrent requests at moderate context lengths.
- VRAM use becomes predictable; you tell the engine how much to reserve up front.
- Short-prompt workloads no longer pay a cache penalty proportional to the longest possible sequence.

## When to turn it on

Use paged attention when:

- You are running a server with more than a handful of concurrent requests.
- You want predictable VRAM usage (for resource accounting or to share the GPU with something else).
- You are using a model with a long context window (32k+) that would otherwise allocate enormous caches.

Skip it when:

- You are doing single-request generation locally. The overhead is small but so is the benefit.
- You are on Metal with a small model. Unified memory makes the memory savings less important, and there is no contention with other processes.

## Configuration

Paged attention is on by default when the engine detects it is likely to help. To control it explicitly:

```bash
# Force it on, with a specific memory budget for blocks
mistralrs serve --paged-attn --paged-attn-gpu-mem 8192 -m <model>

# Force it off
mistralrs serve --no-paged-attn -m <model>
```

`--paged-attn-gpu-mem` sets the memory (in MB) reserved for KV cache blocks. The default is a fraction of available VRAM, which is usually fine.

Block size can be tuned with `--paged-attn-block-size` if you know what you are doing. Smaller blocks mean finer-grained allocation but more bookkeeping overhead. The default (16 on CUDA, 32 on Metal) is appropriate for most workloads.

## Interactions

Paged attention composes cleanly with flash attention; both can be on at once and the engine will use flash kernels for the block-sliced attention computation. No special flags are needed for the combination.

Paged attention does have a small effect on speculative decoding and on some of the more exotic attention variants (MLA in particular). If you are running one of those and performance looks off, try `--no-paged-attn` as a sanity check.

## Memory sizing

The right value for `--paged-attn-gpu-mem` depends on:

- How much VRAM your GPU has.
- How much the model weights take (after quantization).
- What other processes are on the GPU.
- Your expected concurrency.

A rough formula that usually works:

```
--paged-attn-gpu-mem = (total VRAM) - (model weights) - (2 GB for activations and overhead)
```

For a 24 GB card running a 7B model at 4-bit (about 4 GB), that leaves around 18 GB for KV cache, which is enough for a lot of concurrent long-context requests.

## Further reading

The [explanation page](/mistral.rs/explanation/paged-attention/) goes into the design rationale and what the block-based cache looks like internally. If you are trying to squeeze maximum concurrency out of a server, it is worth a read.
