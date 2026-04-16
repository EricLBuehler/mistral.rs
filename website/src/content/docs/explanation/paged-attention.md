---
title: PagedAttention
description: Why block-based KV caching helps concurrent serving, how it works internally, and when it does not.
sidebar:
  order: 5
---

PagedAttention is a specific implementation of the KV cache used by transformer language models. Instead of allocating one contiguous cache per sequence, it allocates fixed-size blocks on demand as each sequence grows. This turns out to be a large win for concurrent serving, and a small but non-trivial complication for everything else.

This page is about the design: why the block-based layout helps, what it costs, and when paged attention is or is not the right thing to turn on.

## The problem

The standard approach is to allocate, for each active sequence, a contiguous slab of memory large enough to hold the sequence's entire KV cache. The slab is sized for the maximum context length the model supports.

For a 128k-context model, that slab is huge. A 7B model with 4096-dimensional attention, 32 layers, fp16: roughly 64 MB per sequence at max context. Serving 32 concurrent sequences needs 2 GB just for KV cache, even if most of those sequences are only using a few thousand tokens worth of context.

That slab is also permanent. Even when a sequence finishes, the memory is reserved until the allocation is explicitly freed. For a server with high churn (many short requests), fragmentation becomes a problem: you have lots of freed slabs at various sizes, and the next request either fits somewhere or has to wait.

The conjunction of these two costs (oversized per-sequence allocation, fragmentation across requests) means the maximum concurrency on a given GPU is a fraction of what the hardware could handle.

## The paged approach

PagedAttention splits the cache into fixed-size blocks (16 tokens per block on CUDA, 32 on Metal). Each sequence has a list of block pointers tracking where its KV cache lives in GPU memory. As a sequence grows, new blocks are allocated from a central pool. When a sequence finishes, its blocks go back to the pool and are available for the next request.

Blocks are the unit of allocation, not sequences. The total memory reserved is whatever you configure, and it is not apportioned per sequence. This means:

- Short requests use only as many blocks as they need.
- Long requests use lots of blocks, but still only while they are active.
- Freed blocks are immediately available to other requests, regardless of whether they fit in the "space" the freed sequence used.

The result: a server with the same hardware can handle many more concurrent requests, especially in workloads with mixed context lengths.

## The tradeoff

Blocks have overhead. Every attention step has to look up the blocks for the active sequence, which is an extra level of indirection compared to contiguous cache. For a single sequence running on an empty server, paged attention is marginally slower than the non-paged path.

This is why mistralrs turns paged attention on automatically for serving workloads (where concurrency is the common case) and leaves it off for single-request CLI use (where the overhead costs without benefit).

Block size is the other knob. Smaller blocks mean less internal fragmentation (fewer unused slots in the last block of each sequence) but more bookkeeping overhead per token. The defaults (16 on CUDA, 32 on Metal) are chosen empirically and are fine for almost every workload.

## Where it shines

PagedAttention is at its best when:

- Many concurrent requests.
- Variable context lengths (mix of short and long).
- Long context model with high turnover.

For a chat server handling many users with 4k-context conversations, paged attention roughly doubles the concurrency you can get out of a given GPU.

For a batch-processing workload where every request has the same long context, the benefit is smaller because there is less fragmentation to avoid. You still get the central pool, which helps with turnover, but the per-sequence savings are closer to zero.

## Where it is neutral or slightly bad

- Single-request interactive use. There is nothing to schedule around, so the paged overhead is pure cost.
- Very short contexts. At 512 tokens the allocation math is so small that the contiguous path is fine.
- Models with non-standard attention. MLA and a few others have custom cache layouts that do not map cleanly to blocks. They work under paged attention but are sometimes slower.

For these cases, `--no-paged-attn` turns it off.

## The interaction with flash attention

Flash attention is an attention kernel optimization. PagedAttention is a KV cache layout decision. They are orthogonal.

In practice, the engine can route paged attention through flash kernels, giving you both speedups together. On CUDA with both features enabled, this is the combined fast path.

Paged attention without flash attention still works; it just leaves the standard attention kernel in place. Flash attention without paged attention also works, which is what you want for single-request workloads.

## Memory sizing

The relevant question for paged attention configuration is how much GPU memory to give the block pool. Too little, and you cannot concurrent as many sequences as you could. Too much, and the block pool steals from other things (like a large KV cache per sequence being allocated if the scheduler overfills it).

A rule of thumb:

```
(total VRAM) - (model weights after quantization) - (2 GB overhead) = block pool budget
```

For a 24 GB card running a 7B model at 4-bit (about 4 GB): roughly 18 GB for blocks, which is enough for lots of concurrent long-context requests.

## The paper and the implementation

The vLLM team published [the original PagedAttention paper](https://arxiv.org/abs/2309.06180) describing this design. Our implementation is not literally their code, but the structure follows the same ideas: fixed-size blocks, a central pool, per-sequence block lists, paged attention kernels that take an extra level of indirection into account.

The mistralrs implementation lives in `mistralrs-paged-attn`. It is separate from the main engine crate because it has substantial CUDA and Metal kernel code that we develop and test independently.

## When you would want to understand this

Most users never need to. The defaults work, and configuration is limited to a couple of flags documented in the [use-paged-attention guide](/mistral.rs/guides/perf/use-paged-attention/).

The cases where understanding the internals matters:

- You are trying to diagnose why concurrent throughput is lower than expected.
- You are writing a tool that interacts with the KV cache directly (benchmarks, custom schedulers).
- You are porting a new model architecture and need to decide how its cache layout interacts with paging.

For those cases, the paper and the source code are the authoritative references.
