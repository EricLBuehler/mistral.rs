---
title: PagedAttention
description: Why block-based KV caching helps concurrent serving, how it works internally, and when it does not.
sidebar:
  order: 5
---

PagedAttention is a KV cache implementation that allocates fixed-size blocks on demand instead of one contiguous cache per sequence. It is a large win for concurrent serving and a small complication for everything else.

## The problem

The standard approach allocates, per active sequence, a contiguous slab sized for the maximum context length the model supports.

For a 128k-context model, this slab is large. A 7B model with 4096-dim attention, 32 layers, fp16: ~64 MB per sequence at max context. Serving 32 concurrent sequences requires 2 GB of KV cache — even when most sequences only use a few thousand tokens.

The slab is permanent until explicitly freed. High-churn servers fragment: many freed slabs at various sizes, with the next request either fitting somewhere or waiting.

Combined, oversized per-sequence allocation and fragmentation cap concurrency well below hardware capability.

## The paged approach

PagedAttention splits the cache into fixed-size blocks (16 tokens on CUDA, 32 on Metal). Each sequence holds a list of block pointers tracking its KV cache locations in GPU memory. New blocks come from a central pool as sequences grow. Finished sequences return blocks to the pool for reuse.

Allocation unit is the block, not the sequence. Total reserved memory is configurable and not per-sequence-apportioned. Effects:

- Short requests use only as many blocks as needed.
- Long requests use many blocks but only while active.
- Freed blocks are immediately available regardless of where the freed sequence lived.

The result: many more concurrent requests on the same hardware, especially with mixed context lengths.

## The tradeoff

Blocks have overhead. Each attention step looks up the active sequence's blocks — extra indirection compared to contiguous cache. For a single sequence on an empty server, paged attention is marginally slower than the non-paged path.

This is why mistral.rs enables paged attention automatically for serving (concurrency-dominant) and disables it for single-request CLI use (overhead without benefit).

Block size is the other knob. Smaller blocks mean less internal fragmentation (fewer unused slots in the last block) but more bookkeeping. The defaults (16 CUDA, 32 Metal) work for almost every workload.

## Where it shines

PagedAttention is best with:

- Many concurrent requests.
- Variable context lengths (mixed short and long).
- Long-context models with high turnover.

For a chat server with many users at 4k-context, paged attention roughly doubles concurrency on the same GPU.

For batch processing where every request has the same long context, the benefit is smaller — less fragmentation to avoid. The central pool still helps with turnover, but per-sequence savings approach zero.

## Where it is neutral or slightly bad

- Single-request interactive use. Nothing to schedule around; paged overhead is pure cost.
- Very short contexts. At 512 tokens, allocation math is small enough that the contiguous path is fine.
- Models with non-standard attention. MLA and a few others have custom cache layouts that do not map cleanly to blocks. They work under paged attention but can be slower.

For these, `--no-paged-attn` disables it.

## Interaction with flash attention

Flash attention is an attention kernel optimization. PagedAttention is a KV cache layout decision. Orthogonal.

The engine routes paged attention through flash kernels for combined speedups. On CUDA with both enabled, this is the combined fast path.

Paged attention without flash attention uses the standard attention kernel. Flash attention without paged attention is correct for single-request workloads.

## Memory sizing

The configuration question: how much GPU memory for the block pool. Too little caps concurrency. Too much steals from other uses.

Rule of thumb:

```
(total VRAM) - (model weights after quantization) - (2 GB overhead) = block pool budget
```

For a 24 GB card running a 7B model at 4-bit (~4 GB): about 18 GB for blocks, sufficient for many concurrent long-context requests.

## The paper and the implementation

The vLLM team published [the PagedAttention paper](https://arxiv.org/abs/2309.06180) describing the design. The mistral.rs implementation is not literal vLLM code but follows the same structure: fixed-size blocks, a central pool, per-sequence block lists, paged attention kernels handling the extra indirection.

The implementation lives in `mistralrs-paged-attn`, separate from the main engine crate due to substantial CUDA and Metal kernel code with independent development and testing.

## When understanding the internals matters

Most users do not need to. The defaults work, configuration is a couple of flags in the [use-paged-attention guide](/mistral.rs/guides/perf/use-paged-attention/).

Cases where internals matter:

- Diagnosing concurrent throughput below expectations.
- Tools interacting directly with the KV cache (benchmarks, custom schedulers).
- Porting a new model architecture and deciding how its cache layout interacts with paging.

For these, the paper and source code are authoritative.
